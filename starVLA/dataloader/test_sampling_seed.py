from types import SimpleNamespace

import pytest

from starVLA.dataloader import _dataset_sampling_seed


def test_sampling_seed_uses_training_seed() -> None:
    assert _dataset_sampling_seed(SimpleNamespace(seed=1002)) == 1002


def test_sampling_seed_requires_explicit_training_seed() -> None:
    with pytest.raises(ValueError, match="must define `seed`"):
        _dataset_sampling_seed(SimpleNamespace())
