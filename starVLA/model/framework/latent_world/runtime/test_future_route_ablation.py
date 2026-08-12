import pytest

from starVLA.model.framework.latent_world.runtime.future_route_ablation import (
    FutureRouteAblation,
)


@pytest.mark.parametrize(
    ("environment", "auxiliary_enabled", "conditioning_enabled"),
    [
        ({}, True, True),
        ({"LAWAM_AUXILIARY_OFF": "1"}, False, True),
        ({"LAWAM_CONDITIONING_OFF": "1"}, True, False),
        ({"LAWAM_FUTURE_OFF": "1"}, False, False),
    ],
)
def test_route_truth_table(environment, auxiliary_enabled, conditioning_enabled) -> None:
    routes = FutureRouteAblation.from_environment(
        environment,
        future_prediction=True,
        dual_route=False,
    )
    assert routes.auxiliary_enabled is auxiliary_enabled
    assert routes.conditioning_enabled is conditioning_enabled


@pytest.mark.parametrize(
    "environment",
    [
        {"LAWAM_AUXILIARY_OFF": "1"},
        {"LAWAM_CONDITIONING_OFF": "1"},
        {"LAWAM_FUTURE_OFF": "1"},
    ],
)
def test_route_ablation_requires_future_prediction(environment) -> None:
    with pytest.raises(ValueError, match="future_prediction=true"):
        FutureRouteAblation.from_environment(
            environment,
            future_prediction=False,
            dual_route=False,
        )


def test_future_off_rejects_single_route_flags() -> None:
    with pytest.raises(ValueError, match="cannot be combined"):
        FutureRouteAblation.from_environment(
            {"LAWAM_FUTURE_OFF": "1", "LAWAM_AUXILIARY_OFF": "1"},
            future_prediction=True,
            dual_route=False,
        )
