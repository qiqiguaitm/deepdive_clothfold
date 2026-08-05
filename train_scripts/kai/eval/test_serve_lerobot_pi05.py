from types import SimpleNamespace

import pytest
import torch

from lerobot_pi05_action_bridge import action_feature_dim
from lerobot_pi05_action_bridge import trim_action_for_postprocessor


def test_action_feature_dim_supports_policy_feature_objects() -> None:
    config = SimpleNamespace(
        output_features={"action": SimpleNamespace(shape=(50, 14))}
    )

    assert action_feature_dim(config) == 14


def test_action_feature_dim_supports_serialized_features() -> None:
    config = SimpleNamespace(output_features={"action": {"shape": [50, 14]}})

    assert action_feature_dim(config) == 14


def test_trim_action_removes_only_native_padding() -> None:
    action = torch.arange(32, dtype=torch.float32).reshape(1, 32)

    trimmed = trim_action_for_postprocessor(action, 14)

    assert trimmed.shape == (1, 14)
    torch.testing.assert_close(trimmed, action[:, :14])


def test_trim_action_rejects_output_smaller_than_feature() -> None:
    with pytest.raises(ValueError, match="smaller than configured"):
        trim_action_for_postprocessor(torch.zeros(1, 13), 14)


@pytest.mark.parametrize(
    "config",
    [
        SimpleNamespace(output_features={}),
        SimpleNamespace(output_features={"action": {"shape": []}}),
        SimpleNamespace(output_features={"action": {"shape": [50, 0]}}),
    ],
)
def test_action_feature_dim_rejects_invalid_config(config: SimpleNamespace) -> None:
    with pytest.raises(ValueError):
        action_feature_dim(config)
