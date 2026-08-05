"""KAI0-owned X-VLA action-space extensions.

Importing this module registers the extensions in LeRobot's X-VLA action
registry. Keep these local behaviors outside the vendored ``xvla/X-VLA``
submodule so the parent repository remains self-contained.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from lerobot.policies.xvla import action_hub


class _ContinuousEE6DBase(action_hub.BaseActionSpace):
    dim_action = 20
    gripper_idx = (9, 19)
    GRIPPER_SCALE = 100.0
    XYZ_SCALE = 500.0
    ROT_SCALE = 10.0
    POS_IDX_1 = (0, 1, 2)
    POS_IDX_2 = (10, 11, 12)
    ROT_IDX_1 = (3, 4, 5, 6, 7, 8)
    ROT_IDX_2 = (13, 14, 15, 16, 17, 18)

    def __init__(self) -> None:
        super().__init__()
        self.mse = nn.MSELoss()

    def _shared_losses(self, pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        if pred.shape != target.shape:
            raise ValueError(f"pred/target shape mismatch: {pred.shape} != {target.shape}")
        if pred.shape[-1] <= max(self.gripper_idx):
            raise IndexError(f"20D EE6D action required, got last dimension {pred.shape[-1]}")
        pos_loss = (
            self.mse(pred[..., self.POS_IDX_1], target[..., self.POS_IDX_1])
            + self.mse(pred[..., self.POS_IDX_2], target[..., self.POS_IDX_2])
        ) * self.XYZ_SCALE
        rot_loss = (
            self.mse(pred[..., self.ROT_IDX_1], target[..., self.ROT_IDX_1])
            + self.mse(pred[..., self.ROT_IDX_2], target[..., self.ROT_IDX_2])
        ) * self.ROT_SCALE
        return {"position_loss": pos_loss, "rotate6D_loss": rot_loss}

    def preprocess(self, proprio, action, mode="train"):
        return proprio, action


class EE6DContinuousActionSpace(_ContinuousEE6DBase):
    """MSE on sigmoid(gripper logit), retained for v2 checkpoint compatibility."""

    def compute_loss(self, pred, target):
        losses = self._shared_losses(pred, target)
        losses["gripper_loss"] = sum(
            self.mse(torch.sigmoid(pred[..., i]), target[..., i]) for i in self.gripper_idx
        ) / len(self.gripper_idx) * self.GRIPPER_SCALE
        return losses

    def postprocess(self, action: torch.Tensor) -> torch.Tensor:
        if action.shape[-1] > max(self.gripper_idx):
            action[..., self.gripper_idx] = torch.sigmoid(action[..., self.gripper_idx])
        return action


class EE6DAlphaActionSpace(_ContinuousEE6DBase):
    """Pure-MSE continuous gripper alpha in [0,1], without sigmoid-space mismatch."""

    def compute_loss(self, pred, target):
        losses = self._shared_losses(pred, target)
        losses["gripper_loss"] = self.mse(
            pred[..., self.gripper_idx], target[..., self.gripper_idx]
        ) * self.GRIPPER_SCALE
        return losses

    def postprocess(self, action: torch.Tensor) -> torch.Tensor:
        if action.shape[-1] > max(self.gripper_idx):
            action[..., self.gripper_idx] = action[..., self.gripper_idx].clamp(0.0, 1.0)
        return action


def ensure_registered() -> None:
    """Idempotently add KAI0 action spaces to the active LeRobot registry."""
    for name, cls in {
        "ee6d_continuous": EE6DContinuousActionSpace,
        "ee6d_alpha": EE6DAlphaActionSpace,
    }.items():
        existing = action_hub.ACTION_REGISTRY.get(name)
        if existing is None:
            action_hub.register_action(name)(cls)
        elif existing is not cls:
            action_hub.ACTION_REGISTRY[name] = cls
            cls.name = name


ensure_registered()
