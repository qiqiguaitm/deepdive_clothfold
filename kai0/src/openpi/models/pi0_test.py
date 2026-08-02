import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

import openpi.models.pi0_config as _pi0_config
from openpi.models import model as _model
from openpi.models.pi0 import SpatialConditionAdapter
from openpi.models.pi0 import _spatial_pool_tokens


def _get_frozen_state(config: _pi0_config.Pi0Config) -> nnx.State:
    abstract_model = nnx.eval_shape(config.create, jax.random.key(0))

    freeze_filter = config.get_freeze_filter()
    return nnx.state(abstract_model, nnx.All(nnx.Param, freeze_filter)).flat_state()


def test_pi0_full_finetune():
    config = _pi0_config.Pi0Config()
    state = _get_frozen_state(config)
    assert len(state) == 0


def test_pi0_gemma_lora():
    config = _pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora")
    state = _get_frozen_state(config)
    assert len(state) == 9
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)
    assert all("_1" not in p for p in state)


def test_pi0_action_expert_lora():
    config = _pi0_config.Pi0Config(action_expert_variant="gemma_300m_lora")
    state = _get_frozen_state(config)
    # excluding embedder, rest of the params should be same as gemma_lora.
    assert len(state) == 8
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)
    # all frozen params should have _1 in their path since it's the action expert.
    assert all(any("_1" in p for p in path) for path in state)


def test_pi0_all_lora():
    config = _pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora")
    state = _get_frozen_state(config)
    # sum of gemma_lora and action_expert_lora's frozen params.
    assert len(state) == 17
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)


def test_spatial_pool_preserves_grid_order():
    tokens = jnp.arange(16, dtype=jnp.float32).reshape(1, 16, 1)
    pooled = _spatial_pool_tokens(tokens, 2)
    np.testing.assert_allclose(pooled[..., 0], [[2.5, 4.5, 10.5, 12.5]])


def test_spatial_condition_detaches_source_but_trains_adapter():
    adapter = SpatialConditionAdapter(4, 6, 2, 3, rngs=nnx.Rngs(0))
    source = jnp.arange(64, dtype=jnp.float32).reshape(1, 16, 4)

    source_grad = jax.grad(lambda value: adapter(value, batch_size=1)[0].astype(jnp.float32).sum())(source)
    np.testing.assert_array_equal(source_grad, jnp.zeros_like(source_grad))

    module_grads = nnx.grad(
        lambda module: module(source, batch_size=1)[0].astype(jnp.float32).sum()
    )(adapter)
    assert np.linalg.norm(np.asarray(module_grads.adapter_in.kernel.value)) > 0
    assert np.linalg.norm(np.asarray(module_grads.adapter_out.kernel.value)) > 0
    assert np.linalg.norm(np.asarray(module_grads.gate.kernel.value)) > 0


def test_spatial_condition_missing_target_is_deterministic_and_parameter_matched():
    adapters = [SpatialConditionAdapter(4, 6, 2, 3, rngs=nnx.Rngs(0)) for _ in range(3)]
    param_sizes = [
        sum(np.asarray(variable.value).size for variable in nnx.state(module, nnx.Param).flat_state().values())
        for module in adapters
    ]
    assert param_sizes[0] == param_sizes[1] == param_sizes[2]

    first, first_stats = adapters[0](None, batch_size=2)
    second, second_stats = adapters[0](None, batch_size=2)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first_stats["availability"], [False, False])
    np.testing.assert_array_equal(first_stats["gate"], second_stats["gate"])


def test_spatial_target_stays_outside_observation_cameras():
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    observation = _model.Observation.from_dict(
        {
            "image": {"base_0_rgb": image.copy()},
            "image_mask": {"base_0_rgb": np.asarray(True)},
            "state": np.zeros((4,), dtype=np.float32),
            "lmwm_target_image": np.full_like(image, 255),
            "lmwm_target_mask": np.asarray(True),
        }
    )
    assert set(observation.images) == {"base_0_rgb"}
    assert observation.lmwm_target_image is not None
    assert not np.array_equal(observation.images["base_0_rgb"], observation.lmwm_target_image)


def test_spatial_target_content_changes_condition_tokens():
    adapter = SpatialConditionAdapter(4, 6, 2, 3, rngs=nnx.Rngs(0))
    target = jnp.arange(64, dtype=jnp.float32).reshape(1, 16, 4)
    shuffled = jnp.roll(target, shift=4, axis=1)
    correct_tokens, _ = adapter(target, batch_size=1)
    shuffled_tokens, _ = adapter(shuffled, batch_size=1)
    assert not np.array_equal(correct_tokens, shuffled_tokens)
