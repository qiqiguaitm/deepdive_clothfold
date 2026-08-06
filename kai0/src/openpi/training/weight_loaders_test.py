import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np
import torch

from openpi.models.pi0 import HistoryProprioStageTracker
from openpi.training import weight_loaders


def test_merge_params_keeps_spatial_adapter_initialization() -> None:
    reference = {
        "policy": {"kernel": jax.ShapeDtypeStruct((2, 2), np.float32)},
        "lmwm_spatial_adapter": {"gate": jax.ShapeDtypeStruct((2, 1), np.float32)},
    }
    loaded = {"policy": {"kernel": np.ones((2, 2), dtype=np.float32)}}

    merged = weight_loaders._merge_params(
        loaded,
        reference,
        missing_regex=".*lmwm_spatial_adapter.*",
    )

    np.testing.assert_array_equal(merged["policy"]["kernel"], loaded["policy"]["kernel"])
    assert merged["lmwm_spatial_adapter"]["gate"] is reference["lmwm_spatial_adapter"]["gate"]


def test_checkpoint_loader_whitelist_covers_transition_adapter() -> None:
    reference = {
        "base": {"kernel": np.zeros((2, 2), dtype=np.float32)},
        "lmwm_transition_adapter": {"encoder": np.ones((2, 2), dtype=np.float32)},
    }
    loaded = {"base": {"kernel": np.full((2, 2), 3, dtype=np.float32)}}
    merged = weight_loaders._merge_params(
        loaded,
        reference,
        missing_regex=".*(lmwm_transition_adapter).*",
    )
    np.testing.assert_array_equal(merged["base"]["kernel"], loaded["base"]["kernel"])
    assert merged["lmwm_transition_adapter"]["encoder"] is reference["lmwm_transition_adapter"]["encoder"]


def test_convert_mt3_current_tracker_transposes_linear_weights() -> None:
    state = {
        "backbone.0.weight": np.arange(12).reshape(3, 4),
        "backbone.0.bias": np.arange(3),
        "backbone.2.weight": np.arange(9).reshape(3, 3),
        "backbone.2.bias": np.arange(3),
        "current_head.weight": np.arange(6).reshape(2, 3),
        "current_head.bias": np.arange(2),
        "next_head.weight": np.arange(6).reshape(2, 3),
        "next_head.bias": np.arange(2),
    }
    converted = weight_loaders.convert_mt3_torch_tracker_state(state, "current_frame")
    np.testing.assert_array_equal(converted["hidden1/kernel"], state["backbone.0.weight"].T)
    np.testing.assert_array_equal(converted["current_head/bias"], state["current_head.bias"])


def test_convert_mt3_history_tracker_preserves_separate_gru_biases() -> None:
    state = {
        "temporal.weight_ih_l0": np.arange(18).reshape(6, 3),
        "temporal.bias_ih_l0": np.arange(6),
        "temporal.weight_hh_l0": np.arange(12).reshape(6, 2),
        "temporal.bias_hh_l0": np.arange(6) + 10,
        "current_head.weight": np.arange(4).reshape(2, 2),
        "current_head.bias": np.arange(2),
        "next_head.weight": np.arange(4).reshape(2, 2),
        "next_head.bias": np.arange(2),
    }
    converted = weight_loaders.convert_mt3_torch_tracker_state(state, "history_proprio")
    np.testing.assert_array_equal(converted["input_proj/bias"], state["temporal.bias_ih_l0"])
    np.testing.assert_array_equal(converted["recurrent_proj/bias"], state["temporal.bias_hh_l0"])


def test_history_tracker_matches_pytorch_gru_after_conversion() -> None:
    torch.manual_seed(7)
    temporal = torch.nn.GRU(5, 3, batch_first=True)
    current_head = torch.nn.Linear(3, 2)
    next_head = torch.nn.Linear(3, 2)
    torch_state = {
        **{f"temporal.{name}": value.detach().numpy() for name, value in temporal.state_dict().items()},
        **{f"current_head.{name}": value.detach().numpy() for name, value in current_head.state_dict().items()},
        **{f"next_head.{name}": value.detach().numpy() for name, value in next_head.state_dict().items()},
    }
    converted = weight_loaders.convert_mt3_torch_tracker_state(torch_state, "history_proprio")
    tracker = HistoryProprioStageTracker(3, 2, 3, 2, rngs=nnx.Rngs(0))
    for name, value in converted.items():
        module_name, parameter_name = name.split("/")
        getattr(getattr(tracker, module_name), parameter_name).value = jnp.asarray(value)

    rng = np.random.default_rng(11)
    visual = rng.normal(size=(2, 3, 3)).astype(np.float32)
    proprio = rng.normal(size=(2, 3, 2)).astype(np.float32)
    sequence = np.concatenate([visual, proprio], axis=-1)
    with torch.no_grad():
        _, hidden = temporal(torch.from_numpy(sequence))
        expected = (
            current_head(hidden[-1]).numpy(),
            next_head(hidden[-1]).numpy(),
        )
    actual = tracker(jnp.asarray(visual), jnp.asarray(proprio))
    np.testing.assert_allclose(actual[0], expected[0], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(actual[1], expected[1], rtol=1e-5, atol=1e-6)


def test_mt3_loader_contract_preserves_parameter_free_dropout_node() -> None:
    params = {"lmwm_transition_dropout": {}}
    merged = {}
    if "lmwm_transition_dropout" in params:
        merged["lmwm_transition_dropout"] = params["lmwm_transition_dropout"]
    assert merged == params


def test_merge_params_does_not_silently_fill_unapproved_missing_weights() -> None:
    reference = {
        "policy": {"kernel": jax.ShapeDtypeStruct((2, 2), np.float32)},
        "unapproved": {"kernel": jax.ShapeDtypeStruct((1,), np.float32)},
    }
    loaded = {"policy": {"kernel": np.ones((2, 2), dtype=np.float32)}}

    merged = weight_loaders._merge_params(
        loaded,
        reference,
        missing_regex=".*lmwm_spatial_adapter.*",
    )

    assert "unapproved" not in merged


def test_predictive_adapter_loader_overlays_only_adapter(monkeypatch) -> None:
    reference = {
        "policy": {"kernel": np.zeros((2, 2), dtype=np.float32)},
        "predictive_action_adapter": {
            "route": np.zeros((2, 1), dtype=np.float32),
        },
    }
    official = {
        "policy": {"kernel": np.full((2, 2), 3, dtype=np.float32)},
    }
    p0 = {
        "policy": {"kernel": np.full((2, 2), 99, dtype=np.float32)},
        "predictive_action_adapter": {
            "route": np.full((2, 1), 7, dtype=np.float32),
        },
    }

    def restore(path, *, restore_type):
        del restore_type
        return official if str(path) == "official" else p0

    monkeypatch.setattr(weight_loaders._model, "restore_params", restore)
    monkeypatch.setattr(weight_loaders.download, "maybe_download", lambda path: path)
    loader = weight_loaders.CheckpointWithPredictiveAdapterWeightLoader(
        "official", "p0"
    )
    merged = loader.load(reference)

    np.testing.assert_array_equal(merged["policy"]["kernel"], official["policy"]["kernel"])
    np.testing.assert_array_equal(
        merged["predictive_action_adapter"]["route"],
        p0["predictive_action_adapter"]["route"],
    )
