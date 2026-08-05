from __future__ import annotations

"""Isolated pi0.5 client for the preregistered R2 adaptive-execution screen."""

import json
import os
from pathlib import Path
import sys
from typing import Any, Optional, Sequence

import numpy as np

from examples.Robotwin.eval_files.model2robotwin_openpi import (
    OpenpiRobotwinModelClient,
)


REPO = Path(os.environ.get("RT_REPO", "/vePFS/tim/workspace/deepdive_kai0"))
R2_SCRIPTS = REPO / "lmvla/lmwm/scripts"
CRAVE_SRC = REPO / "lmvla/crave/src"
if str(R2_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(R2_SCRIPTS))

from pi05_r2_adaptive_execution import (  # noqa: E402
    CausalExecutionController,
    CausalRecurrenceReadout,
)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_task_readout(path: Path, task_name: str) -> CausalRecurrenceReadout:
    manifest_path = path.with_name("readout_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest["acceptance"]["accepted"]:
        raise RuntimeError(f"R2 readout was not accepted: {manifest_path}")
    if task_name not in manifest["tasks"]:
        raise KeyError(f"R2 readout has no task {task_name!r}")
    task_id = int(manifest["tasks"][task_name]["task_id"])
    payload = np.load(path)
    prefix = f"task{task_id}_"
    return CausalRecurrenceReadout(
        reference_features=payload[prefix + "features"],
        episode_offsets=payload[prefix + "episode_offsets"],
        reference_progress=payload[prefix + "progress"],
        reference_density=payload[prefix + "density"],
        sigma=float(payload[prefix + "sigma"]),
        density_calibration=payload[prefix + "density_calibration"],
        boundary_progress=payload[prefix + "boundary_progress"],
    )


class R2OpenpiRobotwinModelClient(OpenpiRobotwinModelClient):
    CONDITIONS = {"fixed4", "adaptive"}

    def __init__(
        self,
        *args: Any,
        condition: str,
        task_name: str,
        eval_seed: int,
        readout_path: str | None = None,
        diagnostics_root: str | None = None,
        **kwargs: Any,
    ) -> None:
        condition = str(condition).strip().lower()
        if condition not in self.CONDITIONS:
            raise ValueError(f"unknown R2 condition {condition!r}; expected {sorted(self.CONDITIONS)}")
        kwargs["replan_steps"] = 4 if condition == "fixed4" else None
        super().__init__(*args, **kwargs)
        self.condition = condition
        self.task_name = str(task_name)
        self.eval_seed = int(eval_seed)
        self.diagnostics_root = Path(diagnostics_root) if diagnostics_root else None
        self._controllers: dict[int, CausalExecutionController] = {}
        self._desired_horizons: dict[int, int] = {}
        self._readout = None
        self._encoder = None
        if self.condition == "adaptive":
            if not readout_path:
                raise ValueError("adaptive R2 condition requires ROBOTWIN_R2_READOUT")
            self._readout = load_task_readout(Path(readout_path), self.task_name)
            if str(CRAVE_SRC) not in sys.path:
                sys.path.insert(0, str(CRAVE_SRC))
            from crave.encoders import load_encoder

            self._encoder = load_encoder("dinov3-base", dtype="bf16")

    def _controller(self, slot_id: int) -> CausalExecutionController:
        slot_id = int(slot_id)
        if slot_id not in self._controllers:
            self._controllers[slot_id] = CausalExecutionController()
        return self._controllers[slot_id]

    def reset(self, task_description: str = "", slot_id: int = 0) -> None:
        super().reset(task_description=task_description, slot_id=slot_id)
        self._controllers[int(slot_id)] = CausalExecutionController()
        self._desired_horizons.pop(int(slot_id), None)

    def reset_slots(self, slot_ids=None) -> None:
        super().reset_slots(slot_ids)
        if slot_ids is None:
            self._controllers.clear()
            self._desired_horizons.clear()
            return
        for slot_id in slot_ids:
            self._controllers.pop(int(slot_id), None)
            self._desired_horizons.pop(int(slot_id), None)

    def _encode_head(self, image: np.ndarray) -> np.ndarray:
        if self._encoder is None:
            raise RuntimeError("R2 encoder is unavailable")
        import torch

        with torch.inference_mode():
            grid = self._encoder.encode_grid([np.asarray(image)])
            if hasattr(grid, "detach"):
                pooled = grid.detach().float().mean(dim=(2, 3)).cpu().numpy()
            else:
                pooled = np.asarray(grid).mean(axis=(2, 3))
        if pooled.shape != (1, 768) or not np.isfinite(pooled).all():
            raise ValueError(f"invalid online DINO feature: {pooled.shape}")
        return np.asarray(pooled[0], dtype=np.float32)

    def observe(self, example: dict[str, Any], *, slot_id: int, step: int) -> None:
        if self.condition != "adaptive":
            return
        if self._readout is None:
            raise RuntimeError("adaptive R2 readout is unavailable")
        head = np.asarray(example["image"][0])
        fields = self._readout.query(self._encode_head(head))
        controller = self._controller(slot_id)
        controller.observe(
            step=int(step),
            progress=fields["progress"],
            density=fields["density"],
            confidence=fields["confidence"],
            boundary_proximity=fields["boundary_proximity"],
        )
        state = self._get_slot_state(slot_id)
        remaining = (
            0
            if state.raw_actions is None
            else max(0, int(state.raw_actions.shape[0]) - int(state.action_cursor))
        )
        decision = controller.decide(cache_remaining=remaining)
        if decision.force_replan:
            if state.raw_actions is not None:
                state.action_cursor = int(state.raw_actions.shape[0])
            self._desired_horizons[int(slot_id)] = int(decision.horizon)

    def step_batch(
        self,
        examples: Sequence[dict[str, Any]],
        *,
        slot_ids: Optional[Sequence[int]] = None,
    ) -> list[np.ndarray]:
        resolved_slots = list(range(len(examples))) if slot_ids is None else [int(value) for value in slot_ids]
        actions = super().step_batch(examples, slot_ids=resolved_slots)
        if self.condition == "adaptive":
            for slot_id in resolved_slots:
                state = self._get_slot_state(slot_id)
                horizon = self._desired_horizons.pop(slot_id, 4)
                if state.raw_actions is None:
                    raise RuntimeError(f"slot {slot_id} has no inferred action chunk")
                state.raw_actions = state.raw_actions[: min(horizon, len(state.raw_actions))]
        return actions

    def diagnostics(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "condition": self.condition,
            "task_name": self.task_name,
            "eval_seed": self.eval_seed,
            "causal": True,
            "future_observation_used": False,
            "slots": {
                str(slot_id): controller.diagnostics()
                for slot_id, controller in sorted(self._controllers.items())
            },
        }

    def close(self) -> None:
        if self.diagnostics_root is not None:
            atomic_json(
                self.diagnostics_root
                / self.condition
                / f"seed{self.eval_seed}"
                / f"{self.task_name}.json",
                self.diagnostics(),
            )
        super().close()


def get_model(usr_args):
    return R2OpenpiRobotwinModelClient(
        policy_ckpt_path=usr_args.get("policy_ckpt_path"),
        host=usr_args.get("host", "127.0.0.1"),
        port=usr_args.get("port", 8000),
        action_ensemble=False,
        condition=os.environ.get("ROBOTWIN_R2_CONDITION", ""),
        task_name=str(usr_args["task_name"]),
        eval_seed=int(usr_args.get("seed", 0)),
        readout_path=os.environ.get("ROBOTWIN_R2_READOUT"),
        diagnostics_root=os.environ.get("ROBOTWIN_R2_DIAGNOSTICS_ROOT"),
    )


def reset_model(model):
    model.reset_slots()
