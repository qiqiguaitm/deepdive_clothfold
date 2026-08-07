from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch


_VALID_MODES = {"normal", "null", "persistence", "shuffled"}
_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9_.-]+$")


def future_off_condition(predicted: torch.Tensor, *, enabled: bool) -> torch.Tensor:
    return torch.zeros_like(predicted) if enabled else predicted


def future_off_zero_tether(predicted: torch.Tensor, *, enabled: bool) -> torch.Tensor:
    if not enabled:
        return predicted.new_zeros(())
    # Keep the same trainable and optimizer trees while supplying no learning
    # signal from the disabled future route.
    return predicted.float().sum() * 0.0


def _safe_component(value: Any, *, field: str) -> str:
    text = str(value)
    if not text or not _SAFE_COMPONENT.fullmatch(text):
        raise ValueError(f"Invalid temporal-grounding {field}: {value!r}")
    return text


def _normalized_context(context: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(context, dict):
        raise ValueError("Temporal-grounding capture/shuffle requires an inference context dict.")
    required = ("task", "eval_seed", "scene_seed", "episode_id", "query_index")
    missing = [key for key in required if key not in context]
    if missing:
        raise ValueError(f"Temporal-grounding context is missing fields: {missing}")
    return {
        "task": _safe_component(context["task"], field="task"),
        "eval_seed": int(context["eval_seed"]),
        "scene_seed": int(context["scene_seed"]),
        "episode_id": int(context["episode_id"]),
        "query_index": int(context["query_index"]),
    }


def _feature_path(root: Path, context: dict[str, Any]) -> Path:
    return (
        root
        / context["task"]
        / f"eval_seed_{context['eval_seed']}"
        / f"scene_seed_{context['scene_seed']}"
        / f"query_{context['query_index']:06d}.npy"
    )


def _atomic_save(path: Path, value: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp.{os.getpid()}")
    with temporary.open("wb") as stream:
        np.save(stream, value.detach().float().cpu().numpy(), allow_pickle=False)
    os.replace(temporary, path)


def _rms_match(source: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    dims = tuple(range(1, source.ndim))
    source_rms = source.float().square().mean(dim=dims, keepdim=True).sqrt()
    reference_rms = reference.float().square().mean(dim=dims, keepdim=True).sqrt()
    scale = reference_rms / source_rms.clamp_min(1e-8)
    return source * scale.to(dtype=source.dtype)


class FutureConditionIntervention:
    """Frozen inference-only interventions for the LaWAM future condition.

    The hook is inert unless ``LAWAM_FUTURE_INTERVENTION`` or
    ``LAWAM_FUTURE_CAPTURE_ROOT`` is set. Normal-mode capture records the
    checkpoint's own endpoint prediction by paired scene and query. Shuffled
    mode consumes only those frozen records through a prespecified no-self
    scene permutation.
    """

    def __init__(
        self,
        *,
        mode: str = "normal",
        capture_root: str | os.PathLike[str] | None = None,
        shuffle_manifest: str | os.PathLike[str] | None = None,
    ) -> None:
        self.mode = str(mode).strip().lower()
        if self.mode not in _VALID_MODES:
            raise ValueError(
                f"Unsupported LAWAM_FUTURE_INTERVENTION={mode!r}; expected {sorted(_VALID_MODES)}."
            )
        self.capture_root = Path(capture_root).expanduser().resolve() if capture_root else None
        self.shuffle_manifest_path = (
            Path(shuffle_manifest).expanduser().resolve() if shuffle_manifest else None
        )
        self.shuffle_mapping: dict[str, Any] | None = None
        if self.mode == "shuffled":
            if self.capture_root is None or self.shuffle_manifest_path is None:
                raise ValueError(
                    "Shuffled future intervention requires LAWAM_FUTURE_CAPTURE_ROOT and "
                    "LAWAM_FUTURE_SHUFFLE_MANIFEST."
                )
            payload = json.loads(self.shuffle_manifest_path.read_text(encoding="utf-8"))
            mapping = payload.get("mapping")
            if not isinstance(mapping, dict):
                raise ValueError("Future shuffle manifest must contain an object-valued `mapping`.")
            self.shuffle_mapping = mapping

    @classmethod
    def from_environment(cls) -> "FutureConditionIntervention":
        return cls(
            mode=os.getenv("LAWAM_FUTURE_INTERVENTION", "normal"),
            capture_root=os.getenv("LAWAM_FUTURE_CAPTURE_ROOT"),
            shuffle_manifest=os.getenv("LAWAM_FUTURE_SHUFFLE_MANIFEST"),
        )

    @property
    def enabled(self) -> bool:
        return self.mode != "normal" or self.capture_root is not None

    def _source_scene_seed(self, context: dict[str, Any]) -> int:
        assert self.shuffle_mapping is not None
        try:
            source = self.shuffle_mapping[context["task"]][str(context["eval_seed"])][
                str(context["scene_seed"])
            ]
        except (KeyError, TypeError) as exc:
            raise KeyError(
                "No shuffled source for "
                f"task={context['task']} eval_seed={context['eval_seed']} "
                f"scene_seed={context['scene_seed']}"
            ) from exc
        source = int(source)
        if source == context["scene_seed"]:
            raise ValueError("Shuffled future manifest contains a forbidden self-match.")
        return source

    def _load_shuffled(self, context: dict[str, Any], reference: torch.Tensor) -> torch.Tensor:
        assert self.capture_root is not None
        source_context = dict(context)
        source_context["scene_seed"] = self._source_scene_seed(context)
        scene_dir = _feature_path(self.capture_root, source_context).parent
        candidates = sorted(scene_dir.glob("query_*.npy"))
        if not candidates:
            raise FileNotFoundError(f"No captured future features under {scene_dir}")
        # Query counts can differ across paired episodes. The modulo rule is
        # frozen before intervention outcomes and never inspects success.
        selected = candidates[context["query_index"] % len(candidates)]
        value = np.load(selected, allow_pickle=False)
        tensor = torch.from_numpy(value).to(device=reference.device, dtype=reference.dtype)
        if tuple(tensor.shape) != tuple(reference.shape):
            raise ValueError(
                f"Shuffled feature shape mismatch: {selected} has {tuple(tensor.shape)}, "
                f"expected {tuple(reference.shape)}."
            )
        return tensor

    def apply(
        self,
        *,
        predicted: torch.Tensor,
        current: torch.Tensor,
        contexts: Sequence[dict[str, Any] | None] | None,
    ) -> torch.Tensor:
        if not self.enabled:
            return predicted
        if predicted.shape != current.shape:
            raise ValueError(
                "Future intervention requires current and predicted features to have identical "
                f"shape, got {tuple(current.shape)} vs {tuple(predicted.shape)}."
            )
        if contexts is None or len(contexts) != int(predicted.shape[0]):
            raise ValueError(
                "Temporal-grounding context count must match inference batch size: "
                f"contexts={None if contexts is None else len(contexts)} batch={predicted.shape[0]}."
            )
        normalized = [_normalized_context(context) for context in contexts]

        if self.mode == "normal":
            output = predicted
        elif self.mode == "null":
            output = torch.zeros_like(predicted)
        elif self.mode == "persistence":
            output = _rms_match(current, predicted)
        else:
            output = torch.stack(
                [self._load_shuffled(context, predicted[index]) for index, context in enumerate(normalized)],
                dim=0,
            )

        if self.capture_root is not None and self.mode == "normal":
            for index, context in enumerate(normalized):
                _atomic_save(_feature_path(self.capture_root, context), predicted[index])
        return output
