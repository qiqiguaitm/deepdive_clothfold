from analyze_mt5_complementarity import complementarity_gate


def contrast(point: float, low: float, high: float = 0.2):
    return {
        "available": True,
        "point_estimate_macro_delta": point,
        "ci95": [low, high],
    }


def test_complementarity_requires_combined_to_beat_both_single_factors():
    accepted = complementarity_gate(
        {
            "combined_minus_local": contrast(0.05, 0.01),
            "combined_minus_transition": contrast(0.04, 0.005),
        }
    )
    assert accepted["accepted"]

    rejected = complementarity_gate(
        {
            "combined_minus_local": contrast(0.05, -0.01),
            "combined_minus_transition": contrast(0.04, 0.005),
        }
    )
    assert not rejected["accepted"]
