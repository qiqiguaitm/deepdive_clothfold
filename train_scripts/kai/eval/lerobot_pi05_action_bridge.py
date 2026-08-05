"""Validate and remove PI0.5's padded action dimensions before unnormalization."""

from __future__ import annotations

from typing import Any

import torch


def action_feature_dim(config: Any) -> int:
    output_features = getattr(config, "output_features", None)
    if not output_features or "action" not in output_features:
        raise ValueError("Policy config does not define output_features.action")
    action_feature = output_features["action"]
    shape = (
        action_feature.get("shape")
        if isinstance(action_feature, dict)
        else getattr(action_feature, "shape", None)
    )
    if not shape:
        raise ValueError("Policy config output_features.action has no shape")
    action_dim = int(shape[-1])
    if action_dim <= 0:
        raise ValueError(f"Invalid policy action dimension: {action_dim}")
    return action_dim


def trim_action_for_postprocessor(action: torch.Tensor, action_dim: int) -> torch.Tensor:
    if action.ndim < 1:
        raise ValueError(f"Expected an action tensor with at least one dimension, got {action.shape}")
    if action.shape[-1] < action_dim:
        raise ValueError(
            f"Model action dimension {action.shape[-1]} is smaller than configured "
            f"output action dimension {action_dim}"
        )
    return action[..., :action_dim]
