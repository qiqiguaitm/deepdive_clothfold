import dataclasses
import logging
import re
from typing import Protocol, runtime_checkable

import flax.traverse_util
import numpy as np

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.download as download

logger = logging.getLogger(__name__)


@runtime_checkable
class WeightLoader(Protocol):
    def load(self, params: at.Params) -> at.Params:
        """Loads the model weights.

        Args:
            params: Parameters of the model. This is a nested structure of array-like objects that
                represent the model's parameters.

        Returns:
            Loaded parameters. The structure must be identical to `params`. If returning a subset of
            the parameters the loader must merge the loaded parameters with `params`.
        """


@dataclasses.dataclass(frozen=True)
class NoOpWeightLoader(WeightLoader):
    def load(self, params: at.Params) -> at.Params:
        return params


@dataclasses.dataclass(frozen=True)
class CheckpointWeightLoader(WeightLoader):
    """Loads an entire set of weights from a checkpoint.

    Compatible with:
      trained checkpoints:
        example: "./checkpoints/<config>/<exp>/<step>/params"
      released checkpoints:
        example: "gs://openpi-assets/checkpoints/<model>/params"
    """

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        # We are loading np.ndarray and relying on the training code to properly convert and shard the params.
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        # Whitelist for keys present in the *model* but absent in the *ckpt*: keep the
        # model's init for these (instead of erroring). Currently:
        #   .*lora.*                  — LoRA adapter ranks
        #   .*soft_prompt_hub.*       — X-VLA soft prompt hub (Track B, new keys when
        #                                warming up from a pi05 base ckpt that predates it)
        #   .*action_head_cond_hub.*  — Track C Action Head Cond Token (方案 A), same
        #                                pattern as soft_prompt_hub
        #   .*lmwm_hint_proj.*        — pi05×LMWM hint 投影层 (A1/A2, warm-start pi05_base 时新增,
        #   .*lmwm_live_pred_.*       — A3 live-target LMWM predictor, also new vs pi05_base
        #   .*lmwm_spatial_adapter.*  — spatial future-condition adapter, also new vs pi05_base
        #   .*lmwm_transition_adapter.* — action-expert milestone-transition route
        #   .*lmwm_transition_tracker.* — MT3 learned stage tracker
        #                                pi05_base 无此键 → 保留随机初始化, 其余载 pi05_base)
        return _merge_params(
            loaded_params,
            params,
            missing_regex=".*(lora|soft_prompt_hub|action_head_cond_hub|lmwm_hint_proj|lmwm_live_pred_|lmwm_spatial_adapter|lmwm_transition_adapter|lmwm_transition_tracker|lmwm_local_adapter|predictive_action_adapter|recurrence_action_adapter).*",
        )


def convert_mt3_torch_tracker_state(
    state: dict[str, np.ndarray], candidate: str
) -> dict[str, np.ndarray]:
    """Convert a frozen tracker-only PyTorch state to NNX parameter layout."""
    if candidate == "current_frame":
        names = {
            "hidden1/kernel": "backbone.0.weight",
            "hidden1/bias": "backbone.0.bias",
            "hidden2/kernel": "backbone.2.weight",
            "hidden2/bias": "backbone.2.bias",
            "current_head/kernel": "current_head.weight",
            "current_head/bias": "current_head.bias",
            "next_head/kernel": "next_head.weight",
            "next_head/bias": "next_head.bias",
        }
    elif candidate == "history_proprio":
        names = {
            "input_proj/kernel": "temporal.weight_ih_l0",
            "input_proj/bias": "temporal.bias_ih_l0",
            "recurrent_proj/kernel": "temporal.weight_hh_l0",
            "recurrent_proj/bias": "temporal.bias_hh_l0",
            "current_head/kernel": "current_head.weight",
            "current_head/bias": "current_head.bias",
            "next_head/kernel": "next_head.weight",
            "next_head/bias": "next_head.bias",
        }
    else:
        raise ValueError(f"unknown MT3 tracker candidate: {candidate!r}")

    missing = sorted(set(names.values()) - set(state))
    if missing:
        raise ValueError(f"MT3 tracker checkpoint is missing tensors: {missing}")
    converted = {}
    for target, source in names.items():
        value = np.asarray(state[source])
        converted[target] = value.T if target.endswith("/kernel") else value
    return converted


@dataclasses.dataclass(frozen=True)
class CheckpointWithMT3TrackerWeightLoader(WeightLoader):
    """Load pi0.5 initialization, then overlay a selected tracker-only checkpoint."""

    params_path: str
    tracker_path: str
    candidate: str

    def load(self, params: at.Params) -> at.Params:
        import torch

        merged = CheckpointWeightLoader(self.params_path).load(params)
        # Parameter-free NNX modules still occupy an empty node in the model
        # PyTree, which flatten/unflatten-based checkpoint merging drops.
        if "lmwm_transition_dropout" in params:
            merged["lmwm_transition_dropout"] = params["lmwm_transition_dropout"]
        checkpoint = torch.load(
            download.maybe_download(self.tracker_path), map_location="cpu", weights_only=False
        )
        checkpoint_candidate = str(checkpoint.get("candidate", ""))
        if checkpoint_candidate != self.candidate:
            raise ValueError(
                f"tracker candidate mismatch: checkpoint={checkpoint_candidate!r}, expected={self.candidate!r}"
            )
        torch_state = {
            name: tensor.detach().cpu().numpy()
            for name, tensor in checkpoint["model"].items()
        }
        tracker = convert_mt3_torch_tracker_state(torch_state, self.candidate)
        flat = flax.traverse_util.flatten_dict(merged, sep="/")
        for relative_name, value in tracker.items():
            name = f"lmwm_transition_tracker/{relative_name}"
            if name not in flat:
                raise ValueError(f"model parameter tree is missing selected tracker tensor: {name}")
            if value.shape != flat[name].shape:
                raise ValueError(
                    f"tracker tensor shape mismatch for {name}: checkpoint={value.shape}, model={flat[name].shape}"
                )
            flat[name] = value.astype(flat[name].dtype, copy=False)
        return flax.traverse_util.unflatten_dict(flat, sep="/")


@dataclasses.dataclass(frozen=True)
class CheckpointWithPredictiveAdapterWeightLoader(WeightLoader):
    """Load official pi0.5 parameters and overlay only a gated P0 adapter."""

    params_path: str
    adapter_params_path: str

    def load(self, params: at.Params) -> at.Params:
        merged = CheckpointWeightLoader(self.params_path).load(params)
        adapter_checkpoint = _model.restore_params(
            download.maybe_download(self.adapter_params_path), restore_type=np.ndarray
        )
        flat = flax.traverse_util.flatten_dict(merged, sep="/")
        flat_adapter_checkpoint = flax.traverse_util.flatten_dict(
            adapter_checkpoint, sep="/"
        )
        adapter_names = sorted(
            name
            for name in flat_adapter_checkpoint
            if "predictive_action_adapter" in name
        )
        if not adapter_names:
            raise ValueError("P0 checkpoint contains no predictive adapter parameters")
        expected_names = sorted(
            name for name in flat if "predictive_action_adapter" in name
        )
        if adapter_names != expected_names:
            missing = sorted(set(expected_names) - set(adapter_names))
            unexpected = sorted(set(adapter_names) - set(expected_names))
            raise ValueError(
                "P0 predictive adapter tree mismatch: "
                f"missing={missing}, unexpected={unexpected}"
            )
        for name in adapter_names:
            value = np.asarray(flat_adapter_checkpoint[name])
            if value.shape != flat[name].shape:
                raise ValueError(
                    f"P0 adapter shape mismatch for {name}: "
                    f"checkpoint={value.shape}, model={flat[name].shape}"
                )
            flat[name] = value.astype(flat[name].dtype, copy=False)
        return flax.traverse_util.unflatten_dict(flat, sep="/")


@dataclasses.dataclass(frozen=True)
class PaliGemmaWeightLoader(WeightLoader):
    """Loads weights from the official PaliGemma checkpoint.

    This will overwrite existing weights with similar names while keeping all extra weights intact.
    This allows us to support the action expert which is used by the Pi0 model.
    """

    def load(self, params: at.Params) -> at.Params:
        path = download.maybe_download(
            "gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz", gs={"token": "anon"}
        )
        with path.open("rb") as f:
            flat_params = dict(np.load(f, allow_pickle=False))
        loaded_params = {"PaliGemma": flax.traverse_util.unflatten_dict(flat_params, sep="/")["params"]}
        # Add all missing weights.
        return _merge_params(loaded_params, params, missing_regex=".*")


@dataclasses.dataclass(frozen=True)
class PaliGemmaLocalWeightLoader(WeightLoader):
    """Like PaliGemmaWeightLoader but loads the big_vision PaliGemma .npz from a LOCAL path (offline
    clusters where the official GCS bucket is unreachable / anon-revoked). Tolerates a community-mirror
    export that (a) lacks the top-level 'params/' wrapper (keys are 'img/...'/'llm/...') and/or (b) is
    f16 — the merge casts every loaded array to the model param's dtype, so precision is unified at load.
    Path resolved from `npz_path` field, else env `PALIGEMMA_NPZ`."""

    npz_path: str = ""

    def load(self, params: at.Params) -> at.Params:
        import os
        path = self.npz_path or os.environ.get("PALIGEMMA_NPZ", "")
        if not path or not os.path.isfile(path):
            raise FileNotFoundError(f"PaliGemma npz not found: {path!r} (set field npz_path or env PALIGEMMA_NPZ)")
        with open(path, "rb") as f:
            flat = dict(np.load(f, allow_pickle=False))
        tree = flax.traverse_util.unflatten_dict(flat, sep="/")
        sub = tree["params"] if "params" in tree else tree  # community npz often lacks the 'params/' wrapper
        loaded_params = {"PaliGemma": sub}
        return _merge_params(loaded_params, params, missing_regex=".*")


def _merge_params(loaded_params: at.Params, params: at.Params, *, missing_regex: str) -> at.Params:
    """Merges the loaded parameters with the reference parameters.

    Args:
        loaded_params: The parameters to merge.
        params: The reference parameters.
        missing_regex: A regex pattern for all missing keys that should be merged from the reference parameters.

    Returns:
        A new dictionary with the merged parameters.
    """
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")

    # First, take all weights that are a subset of the reference weights.
    result = {}
    for k, v in flat_loaded.items():
        if k in flat_ref:
            result[k] = v.astype(flat_ref[k].dtype) if v.dtype != flat_ref[k].dtype else v

    flat_loaded.clear()

    # Then, merge any missing weights as defined by the missing regex.
    pattern = re.compile(missing_regex)
    for k in {k for k in flat_ref if pattern.fullmatch(k)}:
        if k not in result:
            result[k] = flat_ref[k]

    return flax.traverse_util.unflatten_dict(result, sep="/")
