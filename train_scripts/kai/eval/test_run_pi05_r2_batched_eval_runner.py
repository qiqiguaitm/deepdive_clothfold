from __future__ import annotations

from pathlib import Path

from run_pi05_r2_batched_eval_runner import render_r2_bridge


REPO = Path(__file__).resolve().parents[3]
BRIDGE = REPO / "lmvla/lawam/examples/Robotwin/eval_files/robotwin_batch_bridge.py"


def test_rendered_bridge_has_isolated_r2_hooks() -> None:
    original = BRIDGE.read_text(encoding="utf-8")
    rendered = render_r2_bridge(original)
    assert "model2robotwin_openpi_r2" in rendered
    assert 'observe(example, slot_id=slot_id, step=int(message["step"]))' in rendered
    assert 'SCRIPT_DIR = Path(os.environ["ROBOTWIN_R2_BASE_EVAL_DIR"]).resolve()' in rendered
    assert "model2robotwin_openpi_r2" not in original


def test_r2_bridge_patch_is_rejected_after_first_application() -> None:
    rendered = render_r2_bridge(BRIDGE.read_text(encoding="utf-8"))
    try:
        render_r2_bridge(rendered)
    except ValueError as error:
        assert "expected one match" in str(error)
    else:
        raise AssertionError("R2 bridge patch unexpectedly applied twice")
