import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).with_name("benchmark_pi05_policy_latency.py")
SPEC = importlib.util.spec_from_file_location("benchmark_pi05_policy_latency", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_mt3_observation_has_frozen_transition_and_history_shapes() -> None:
    observation = MODULE.make_observation(
        0, transition_task_id=2, history_steps=3
    )
    assert observation["lmwm_transition_task"].shape == ()
    assert observation["lmwm_transition_task"].item() == 2
    assert observation["lmwm_transition_mask"].item() is True
    assert observation["lmwm_transition_history_images"].shape == (
        3,
        3,
        480,
        640,
    )
    assert observation["lmwm_transition_history_state"].shape == (3, 14)
