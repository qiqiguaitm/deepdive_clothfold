import dataclasses
import glob
import os

import cv2
import einops
import numpy as np
from openpi_client import image_tools

from openpi import transforms
from openpi.models import model as _model


def make_libero_example() -> dict:
    """Creates a random input example for the Libero policy."""
    return {
        "observation/state": np.random.rand(8),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class HintLookupTransform(transforms.DataTransformFn):
    """pi05 × LMWM (A1/A2): 按 (dataset_id, episode_index, frame_index) 查预算 hint 注入 data["lmwm_hint"].

    hint.npz (export_pi05_hint.py 产) 含平行数组 suite/episode_index/frame_index/hint[N,(K,)D].
    suite → dataset_id 由 datasets_yaml 的 domain_ids 顺序决定 (与本 transform 的 suite_order 一致).
    须在 repack **之前**运行 (读原始样本的 episode_index/frame_index/dataset_id); repack structure
    需含 "lmwm_hint":"lmwm_hint" 以保留. 缺失帧回退零向量 (训练不崩; 应极少).

    hint_path: hint.npz 路径; suite_order: list[str] 按 dataset_id 索引 (suite_order[did]=suite 名).
    """

    hint_path: str
    suite_order: tuple

    def __post_init__(self):
        z = np.load(self.hint_path, allow_pickle=True)
        suites = z["suite"].astype(str)
        eps = z["episode_index"].astype(np.int64)
        fis = z["frame_index"].astype(np.int64)
        hint = z["hint"]  # [N, D] 或 [N, K, D]
        # 建 (did, ep) -> {frame: row} 的紧凑索引: per-(did,ep) 存一个 frame->row 映射.
        name_to_did = {s: i for i, s in enumerate(self.suite_order)}
        index: dict = {}
        for row in range(len(hint)):
            did = name_to_did.get(str(suites[row]))
            if did is None:
                continue
            index.setdefault((did, int(eps[row])), {})[int(fis[row])] = row
        object.__setattr__(self, "_hint", np.asarray(hint))
        object.__setattr__(self, "_index", index)
        object.__setattr__(self, "_dim", hint.shape[1:])  # (D,) 或 (K,D)

    def __call__(self, data: dict) -> dict:
        # 单-repo 路径(如 robotwin)不注入 dataset_id → 默认 0(配 suite_order=("robotwin",) 单套件).
        did = int(np.asarray(data.get("dataset_id", 0)).reshape(-1)[0])
        ep = int(np.asarray(data["episode_index"]).reshape(-1)[0])
        fi = int(np.asarray(data["frame_index"]).reshape(-1)[0])
        row = self._index.get((did, ep), {}).get(fi)
        # 模型消费形状 = [hint_len, D]: 单发 D→[1,D]; best-of-K [K,D]→[K,D].
        if row is None:
            shape = (1, self._dim[-1]) if len(self._dim) == 1 else self._dim
            data["lmwm_hint"] = np.zeros(shape, dtype=np.float32)
        else:
            h = self._hint[row].astype(np.float32)          # [D] 或 [K, D]
            data["lmwm_hint"] = h[None] if h.ndim == 1 else h
        return data


@dataclasses.dataclass(frozen=True)
class RobotwinTargetImageLookupTransform(transforms.DataTransformFn):
    """A3 live-target LMWM: attach the mined representative target frame image.

    Reads a pairs.npz containing cur_ep/cur_fi/tgt_fi and a RoboTwin JPEG frame
    cache. For each training sample, the target frame is decoded and stored as
    data["lmwm_target_image"] in HWC uint8 224x224. The model re-encodes it with
    the *current* pi05 visual encoder and stop-gradient, avoiding stale offline
    feature spaces. Missing pairs get mask=0 and reuse the current cam_high image
    as a harmless placeholder.
    """

    pairs_path: str
    frame_cache_root: str
    camera: str = "observation.images.cam_high"
    height: int = 224
    width: int = 224

    def __post_init__(self):
        z = np.load(self.pairs_path)
        horizon_frames = int(z["horizon_frames"]) if "horizon_frames" in z.files else None
        if horizon_frames is None:
            cur_ep = z["cur_ep"].astype(np.int64)
            cur_fi = z["cur_fi"].astype(np.int64)
            tgt_fi = z["tgt_fi"].astype(np.int64)
            index = {(int(e), int(f)): int(t) for e, f, t in zip(cur_ep, cur_fi, tgt_fi, strict=False)}
        else:
            if horizon_frames <= 0:
                raise ValueError(f"fixed target horizon must be positive, got {horizon_frames}")
            index = {}
        ep_paths = {}
        pattern = os.path.join(self.frame_cache_root, "chunk-*", self.camera, "episode_*.npz")
        for path in glob.glob(pattern):
            ep = int(os.path.basename(path).split("_")[1].split(".")[0])
            ep_paths[ep] = path
        object.__setattr__(self, "_index", index)
        object.__setattr__(self, "_horizon_frames", horizon_frames)
        object.__setattr__(self, "_ep_paths", ep_paths)
        object.__setattr__(self, "_cache", {})

    def _decode(self, ep: int, fi: int) -> np.ndarray | None:
        path = self._ep_paths.get(ep)
        if path is None:
            return None
        cache = self._cache
        data = cache.get(ep)
        if data is None:
            data = np.load(path)
            if len(cache) > 16:
                cache.pop(next(iter(cache)))
            cache[ep] = data
        key = str(fi)
        if key not in data:
            return None
        img = cv2.imdecode(data[key], cv2.IMREAD_COLOR)
        if img is None:
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return image_tools.resize_with_pad(img, self.height, self.width).astype(np.uint8)

    def __call__(self, data: dict) -> dict:
        ep = int(np.asarray(data["episode_index"]).reshape(-1)[0])
        fi = int(np.asarray(data["frame_index"]).reshape(-1)[0])
        tgt = (
            fi + self._horizon_frames
            if self._horizon_frames is not None
            else self._index.get((ep, fi))
        )
        img = None if tgt is None else self._decode(ep, tgt)
        if img is None:
            cur = data.get("observation", {}).get("images", {}).get("cam_high")
            if cur is not None:
                cur = np.asarray(cur)
                if cur.shape[0] == 3:
                    cur = einops.rearrange(cur, "c h w -> h w c")
                img = image_tools.resize_with_pad(cur, self.height, self.width).astype(np.uint8)
            else:
                img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            data["lmwm_target_mask"] = np.asarray(False)
        else:
            data["lmwm_target_mask"] = np.asarray(True)
        data["lmwm_target_image"] = img
        return data


@dataclasses.dataclass(frozen=True)
class RobotwinTransitionLookupTransform(transforms.DataTransformFn):
    """Attach automatically mined task/current/next milestone-stage IDs.

    The frozen recurrence artifact assigns a task-local segment ordinal to each
    covered frame. The next stage is the following ordinal, including a
    terminal ordinal after the last segment. Frames outside the frozen artifact
    receive explicit null sentinels so the raw training dataset remains matched
    to A0 rather than being silently filtered.
    """

    pairs_path: str
    num_tasks: int = 6
    num_stages: int = 10

    def __post_init__(self):
        z = np.load(self.pairs_path)
        required = {"cur_ep", "cur_fi", "cur_ms", "pair_task"}
        missing = required.difference(z.files)
        if missing:
            raise ValueError(f"transition pairs missing keys: {sorted(missing)}")
        cur_ep = z["cur_ep"].astype(np.int64)
        cur_fi = z["cur_fi"].astype(np.int64)
        cur_ms = z["cur_ms"].astype(np.int64)
        pair_task = z["pair_task"].astype(np.int64)
        if np.any(pair_task < 0) or np.any(pair_task >= self.num_tasks):
            raise ValueError("transition task IDs exceed configured task vocabulary")
        if np.any(cur_ms < 0) or np.any(cur_ms + 1 >= self.num_stages):
            raise ValueError("transition stage IDs exceed configured stage vocabulary")
        index = {
            (int(ep), int(fi)): (int(task), int(stage), int(stage + 1))
            for ep, fi, task, stage in zip(cur_ep, cur_fi, pair_task, cur_ms, strict=False)
        }
        object.__setattr__(self, "_index", index)
        episode_ranges = []
        for task in np.unique(pair_task).tolist():
            task_episodes = cur_ep[pair_task == task]
            episode_ranges.append((int(task_episodes.min()), int(task_episodes.max()), task))
        ordered = sorted(episode_ranges)
        if any(left[1] >= right[0] for left, right in zip(ordered, ordered[1:], strict=False)):
            raise ValueError("transition task episode ranges overlap")
        object.__setattr__(self, "_episode_ranges", tuple(episode_ranges))

    def __call__(self, data: dict) -> dict:
        ep = int(np.asarray(data["episode_index"]).reshape(-1)[0])
        fi = int(np.asarray(data["frame_index"]).reshape(-1)[0])
        value = self._index.get((ep, fi))
        if value is None:
            task = next(
                (task for lower, upper, task in self._episode_ranges if lower <= ep <= upper),
                self.num_tasks,
            )
            current, nxt = self.num_stages, self.num_stages
            available = False
        else:
            task, current, nxt = value
            available = True
        data["lmwm_transition_task"] = np.asarray(task, dtype=np.int32)
        data["lmwm_transition_current"] = np.asarray(current, dtype=np.int32)
        data["lmwm_transition_next"] = np.asarray(nxt, dtype=np.int32)
        data["lmwm_transition_mask"] = np.asarray(available)
        return data


@dataclasses.dataclass(frozen=True)
class RobotwinCraveTargetLookupTransform(transforms.DataTransformFn):
    """Attach frozen R0 recurrence targets without changing dataset sampling."""

    targets_path: str

    def __post_init__(self):
        z = np.load(self.targets_path)
        required = {
            "cur_ep",
            "cur_fi",
            "progress_change",
            "target_recurrence_density",
            "phase_boundary_crossing",
        }
        missing = required.difference(z.files)
        if missing:
            raise ValueError(f"CRAVE targets missing keys: {sorted(missing)}")
        keys = list(zip(z["cur_ep"].astype(np.int64), z["cur_fi"].astype(np.int64), strict=True))
        if len(set(keys)) != len(keys):
            raise ValueError("CRAVE targets contain duplicate (episode, frame) rows")
        progress = z["progress_change"].astype(np.float32)
        density = z["target_recurrence_density"].astype(np.float32)
        boundary = z["phase_boundary_crossing"].astype(np.bool_)
        if not np.all(np.isfinite(progress)) or not np.all(np.isfinite(density)):
            raise ValueError("CRAVE continuous targets must be finite")
        if np.any(progress < -1.0) or np.any(progress > 1.0):
            raise ValueError("CRAVE progress-change targets must lie in [-1, 1]")
        if np.any(density < 0.0) or np.any(density > 1.0):
            raise ValueError("CRAVE recurrence-density targets must lie in [0, 1]")
        index = {
            (int(ep), int(frame)): (float(delta), float(rho), bool(crossing))
            for (ep, frame), delta, rho, crossing in zip(
                keys, progress, density, boundary, strict=True
            )
        }
        object.__setattr__(self, "_index", index)

    def __call__(self, data: dict) -> dict:
        ep = int(np.asarray(data["episode_index"]).reshape(-1)[0])
        frame = int(np.asarray(data["frame_index"]).reshape(-1)[0])
        value = self._index.get((ep, frame))
        if value is None:
            delta, density, boundary, available = 0.0, 0.0, False, False
        else:
            delta, density, boundary = value
            available = True
        data["crave_progress_change"] = np.asarray(delta, dtype=np.float32)
        data["crave_target_density"] = np.asarray(density, dtype=np.float32)
        data["crave_boundary_crossing"] = np.asarray(boundary, dtype=np.bool_)
        data["crave_target_mask"] = np.asarray(available, dtype=np.bool_)
        return data


@dataclasses.dataclass(frozen=True)
class LiberoInputs(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format. It is used for both training and inference.

    For your own dataset, you can copy this class and modify the keys based on the comments below to pipe
    the correct elements of your dataset into the model.
    """

    # Determines which model will be used.
    # Do not change this for your own dataset.
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference.
        # Keep this for your own dataset, but if your dataset stores the images
        # in a different key than "observation/image" or "observation/wrist_image",
        # you should change it below.
        # Pi0 models support three image inputs at the moment: one third-person view,
        # and two wrist views (left and right). If your dataset does not have a particular type
        # of image, e.g. wrist images, you can comment it out here and replace it with zeros like we do for the
        # right wrist image below.
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        # Create inputs dict. Do not change the keys in the dict below.
        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                # Pad any non-existent images with zero-arrays of the appropriate shape.
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # We only mask padding images for pi0 model, not pi0-FAST. Do not change this for your own dataset.
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        # Pad actions to the model action dimension. Keep this for your own dataset.
        # Actions are only available during training.
        if "actions" in data:
            inputs["actions"] = data["actions"]

        # Pass the prompt (aka language instruction) to the model.
        # Keep this for your own dataset (but modify the key if the instruction is not
        # stored in "prompt"; the output dict always needs to have the key "prompt").
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        # LMWM hint (pi05 × LMWM A1/A2): offline-precomputed subgoal vector, injected by
        # HintLookupTransform upstream (per episode/frame). Absent for A0 → pass-through no-op.
        if "lmwm_hint" in data:
            inputs["lmwm_hint"] = data["lmwm_hint"]

        return inputs


@dataclasses.dataclass(frozen=True)
class LiberoOutputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back the the dataset specific format. It is
    used for inference only.

    For your own dataset, you can copy this class and modify the action dimension based on the comments below.
    """

    def __call__(self, data: dict) -> dict:
        # Only return the first N actions -- since we padded actions above to fit the model action
        # dimension, we need to now parse out the correct number of actions in the return dict.
        # For Libero, we only return the first 7 actions (since the rest is padding).
        # For your own dataset, replace `7` with the action dimension of your dataset.
        return {"actions": np.asarray(data["actions"][:, :7])}
