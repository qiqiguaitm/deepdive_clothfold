from __future__ import annotations

import json
from pathlib import Path

import pytest

from analyze_pi05_r4_formal import analyze
from test_analyze_pi05_r4_formal import _write_report
from finalize_pi05_r4_seed1000_evidence import finalize, markdown


def test_finalizer_reports_every_task_and_claim_boundary(tmp_path: Path) -> None:
    ordinary = tmp_path / "ordinary.json"
    crave = tmp_path / "crave.json"
    terminal = tmp_path / "terminal.json"
    gate = tmp_path / "gate.json"
    _write_report(ordinary, 30)
    _write_report(crave, 32)
    _write_report(terminal, 35)
    gate.write_text(
        json.dumps(
            analyze(
                ordinary,
                terminal,
                crave,
                bootstrap_samples=20,
                bootstrap_seed=7,
            )
        )
    )

    payload = finalize(
        {
            "ordinary": ordinary,
            "outcome_free_crave": crave,
            "terminal_outcome": terminal,
        },
        gate,
    )

    assert payload["accepted"] is True
    assert len(payload["tasks"]) == 6
    assert payload["tasks"]["task_0"]["terminal_minus_ordinary"] == pytest.approx(0.1)
    assert "does not estimate Q-values" in payload["claim_boundary"]
    rendered = markdown(payload)
    assert "| task_0 | 60.0 | 64.0 | 70.0 | +10.0 | +6.0 |" in rendered
    assert "Decision: **accepted**" in rendered
