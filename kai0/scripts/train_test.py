import dataclasses
import os
import pathlib
from types import SimpleNamespace

import pytest

os.environ["JAX_PLATFORMS"] = "cpu"

from openpi.training import config as _config

from . import train


@pytest.mark.parametrize("config_name", ["debug"])
def test_train(tmp_path: pathlib.Path, config_name: str):
    if config_name not in _config._CONFIGS_DICT:  # noqa: SLF001
        pytest.skip(f"training config {config_name!r} is not registered")
    config = dataclasses.replace(
        _config._CONFIGS_DICT[config_name],  # noqa: SLF001
        batch_size=2,
        checkpoint_base_dir=str(tmp_path / "checkpoint"),
        exp_name="test",
        overwrite=False,
        resume=False,
        num_train_steps=2,
        log_interval=1,
    )
    train.main(config)

    # test resuming
    config = dataclasses.replace(config, resume=True, num_train_steps=4)
    train.main(config)


@pytest.mark.parametrize(
    ("step", "save_final_checkpoint", "expected"),
    [(9, True, True), (9, False, False), (5, False, True), (4, True, False)],
)
def test_should_save_checkpoint(step: int, save_final_checkpoint: bool, expected: bool):
    config = SimpleNamespace(
        save_interval=5,
        save_final_checkpoint=save_final_checkpoint,
        num_train_steps=10,
    )
    assert train.should_save_checkpoint(step, start_step=0, config=config) is expected
