from __future__ import annotations

from analyze_pi05_r2_adaptive_screen import analyze


def rows(*, adaptive: bool) -> dict:
    output = {}
    for task in ("a", "b"):
        for index in range(20):
            fixed_success = index < 10
            success = index < 14 if adaptive else fixed_success
            output[(task, 0, index)] = {
                "success": success,
                "steps": 100,
                "cell_query_per_episode": 24.0 if adaptive else 25.0,
                "cell_elapsed_per_episode": 2.0 if adaptive else 1.0,
            }
    return output


def test_adaptive_gate_uses_success_and_query_fairness() -> None:
    result = analyze(rows(adaptive=False), rows(adaptive=True))
    assert result["adaptive_minus_fixed4"] == 0.2
    assert result["efficiency"]["adaptive_query_ratio"] < 1.0
    assert result["gate"]["accepted"]


def test_query_overuse_rejects_otherwise_positive_arm() -> None:
    fixed = rows(adaptive=False)
    adaptive = rows(adaptive=True)
    for value in adaptive.values():
        value["cell_query_per_episode"] = 30.0
    result = analyze(fixed, adaptive)
    assert not result["gate"]["accepted"]
    assert not result["gate"]["adaptive_query_ratio_at_most_1_05"]
