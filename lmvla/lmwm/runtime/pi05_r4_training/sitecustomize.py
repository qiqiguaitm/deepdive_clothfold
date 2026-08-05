"""Install the opt-in LeRobot runtime contract for R4 direct action chunks."""

from __future__ import annotations

import copy
import os
from pathlib import Path


def _install() -> None:
    import torch

    from lerobot.policies import factory as policy_factory
    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    from lerobot.processor import converters
    from lerobot.utils import sample_weighting
    from lerobot.utils.constants import ACTION

    try:
        from lerobot.datasets import factory as dataset_factory
    except ImportError:  # pragma: no cover - compatibility with older LeRobot layouts.
        from lerobot.common.datasets import factory as dataset_factory

    if getattr(sample_weighting, "_pi05_r4_runtime_installed", False):
        return

    for field in ("sample_weight",):
        if field not in converters._COMPLEMENTARY_KEYS:
            converters._COMPLEMENTARY_KEYS = (*converters._COMPLEMENTARY_KEYS, field)

    original_resolve_delta_timestamps = dataset_factory.resolve_delta_timestamps
    original_make_policy = policy_factory.make_policy
    original_make_processors = policy_factory.make_pre_post_processors
    original_make_weighter = sample_weighting.make_sample_weighter

    def resolve_delta_timestamps(config, dataset_metadata):
        resolved = original_resolve_delta_timestamps(config, dataset_metadata)
        if not isinstance(config, PI05Config):
            return resolved
        if resolved is None:
            return None
        resolved = dict(resolved)
        resolved.pop(ACTION, None)
        return resolved or None

    def make_policy(*args, **kwargs):
        config = kwargs.get("cfg")
        if config is None and args:
            config = args[0]
        original_output_features = None
        if isinstance(config, PI05Config):
            original_output_features = copy.deepcopy(config.output_features)
        policy = original_make_policy(*args, **kwargs)
        if original_output_features is not None:
            policy.config.output_features = original_output_features
        return policy

    def make_pre_post_processors(*args, **kwargs):
        # Fine-tuning starts from the exact public policy, so its saved mean/std
        # normalization remains authoritative. The R4 dataset stats describe
        # direct chunks and must not silently replace those processors.
        policy_config = kwargs.get("policy_cfg")
        if policy_config is None and args:
            policy_config = args[0]
        for key in ("preprocessor_overrides", "postprocessor_overrides"):
            overrides = kwargs.get(key)
            if overrides is None:
                continue
            overrides = copy.deepcopy(overrides)
            for value in overrides.values():
                if isinstance(value, dict):
                    value.pop("stats", None)
            kwargs[key] = overrides
        if isinstance(policy_config, PI05Config):
            tokenizer_path = Path(
                os.environ.get(
                    "PI05_R4_TOKENIZER_PATH",
                    "/vePFS/tim/hf_models/paligemma_tokenizer",
                )
            )
            for required in ("tokenizer_config.json", "tokenizer.model"):
                if not (tokenizer_path / required).is_file():
                    raise FileNotFoundError(tokenizer_path / required)
            overrides = copy.deepcopy(kwargs.get("preprocessor_overrides") or {})
            overrides.setdefault("tokenizer_processor", {})["tokenizer_name"] = str(
                tokenizer_path
            )
            kwargs["preprocessor_overrides"] = overrides
        kwargs["dataset_stats"] = None
        return original_make_processors(*args, **kwargs)

    class BatchFieldWeighter(sample_weighting.SampleWeighter):
        def __init__(self, *, device: torch.device, field: str):
            self.device = device
            self.field = field
            self._batches = 0
            self._samples = 0

        def compute_batch_weights(self, batch: dict) -> tuple[torch.Tensor, dict]:
            if self.field not in batch:
                raise KeyError(f"R4 weight field is absent after preprocessing: {self.field}")
            weights = torch.as_tensor(batch[self.field], device=self.device, dtype=torch.float32)
            if weights.ndim == 2 and weights.shape[1] == 1:
                weights = weights[:, 0]
            if weights.ndim != 1:
                raise ValueError(
                    f"R4 weight field must have one scalar per sample, got {tuple(weights.shape)}"
                )
            if not torch.isfinite(weights).all() or torch.any(weights <= 0):
                raise ValueError("R4 sample weights must be finite and strictly positive")
            self._batches += 1
            self._samples += int(weights.numel())
            return weights, {
                "mean_weight": float(weights.mean().detach().cpu()),
                "min_weight": float(weights.min().detach().cpu()),
                "max_weight": float(weights.max().detach().cpu()),
                "type": "batch_field",
            }

        def get_stats(self) -> dict:
            return {
                "type": "batch_field",
                "field": self.field,
                "batches": self._batches,
                "samples": self._samples,
            }

    class SidecarIndexWeighter(sample_weighting.SampleWeighter):
        def __init__(self, *, device: torch.device, path: Path, field: str):
            import numpy as np

            if not path.is_file():
                raise FileNotFoundError(path)
            with np.load(path, allow_pickle=False) as payload:
                if field not in payload.files:
                    raise KeyError(f"R4 sidecar field is absent: {field}")
                values = np.asarray(payload[field], dtype=np.float32)
            if values.ndim != 1 or not np.isfinite(values).all() or np.any(values <= 0):
                raise ValueError("R4 sidecar weights must be finite positive scalars")
            self.device = device
            self.path = path
            self.field = field
            self.values = torch.from_numpy(values)
            self.count = len(values)
            self._batches = 0
            self._samples = 0

        def compute_batch_weights(self, batch: dict) -> tuple[torch.Tensor, dict]:
            if "index" not in batch:
                raise KeyError("R4 sidecar weighting requires the preserved dataset index")
            indices = torch.as_tensor(batch["index"], dtype=torch.long).reshape(-1).cpu()
            if torch.any(indices < 0) or torch.any(indices >= len(self.values)):
                raise IndexError("R4 sidecar index is outside the frozen weight array")
            weights = self.values[indices].to(self.device)
            self._batches += 1
            self._samples += int(weights.numel())
            return weights, {
                "mean_weight": float(weights.mean().detach().cpu()),
                "min_weight": float(weights.min().detach().cpu()),
                "max_weight": float(weights.max().detach().cpu()),
                "type": "sidecar_index",
            }

        def get_stats(self) -> dict:
            return {
                "type": "sidecar_index",
                "path": str(self.path),
                "field": self.field,
                "count": self.count,
                "batches": self._batches,
                "samples": self._samples,
            }

    def make_sample_weighter(config, policy, device, dataset_root=None, dataset_repo_id=None):
        if config is not None and config.type == "batch_field":
            field = str(config.extra_params.get("field", "sample_weight"))
            if field != "sample_weight":
                raise ValueError(f"R4 only authorizes sample_weight, got {field!r}")
            return BatchFieldWeighter(device=device, field=field)
        if config is not None and config.type == "sidecar_index":
            path = Path(str(config.extra_params.get("path", ""))).resolve()
            field = str(config.extra_params.get("field", "weight"))
            if field != "weight":
                raise ValueError(f"R4 only authorizes sidecar field 'weight', got {field!r}")
            return SidecarIndexWeighter(device=device, path=path, field=field)
        return original_make_weighter(
            config,
            policy,
            device,
            dataset_root=dataset_root,
            dataset_repo_id=dataset_repo_id,
        )

    resolve_delta_timestamps._pi05_r4_runtime = True
    make_policy._pi05_r4_runtime = True
    make_pre_post_processors._pi05_r4_runtime = True
    make_sample_weighter._pi05_r4_runtime = True
    dataset_factory.resolve_delta_timestamps = resolve_delta_timestamps
    policy_factory.make_policy = make_policy
    policy_factory.make_pre_post_processors = make_pre_post_processors
    sample_weighting.make_sample_weighter = make_sample_weighter
    sample_weighting.BatchFieldWeighter = BatchFieldWeighter
    sample_weighting.SidecarIndexWeighter = SidecarIndexWeighter
    sample_weighting._pi05_r4_runtime_installed = True


if os.environ.get("PI05_R4_TRAINING_RUNTIME") == "1":
    _install()
