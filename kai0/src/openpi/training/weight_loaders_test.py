import jax
import numpy as np

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
