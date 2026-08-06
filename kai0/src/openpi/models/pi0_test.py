import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

import openpi.models.pi0_config as _pi0_config
from openpi.models import model as _model
from openpi.models.pi0 import SpatialConditionAdapter
from openpi.models.pi0 import TransitionConditionAdapter
from openpi.models.pi0 import CurrentFrameStageTracker
from openpi.models.pi0 import HistoryProprioStageTracker
from openpi.models.pi0 import LocalDynamicsAdapter
from openpi.models.pi0 import PredictiveActionAdapter
from openpi.models.pi0 import RecurrenceActionAdapter
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


def test_transition_adapter_is_parameter_matched_and_content_sensitive():
    oracle = TransitionConditionAdapter(6, 10, 8, 16, 24, rngs=nnx.Rngs(0))
    null = TransitionConditionAdapter(6, 10, 8, 16, 24, rngs=nnx.Rngs(0))
    sizes = [
        sum(np.asarray(v.value).size for v in nnx.state(m, nnx.Param).flat_state().values())
        for m in (oracle, null)
    ]
    assert sizes[0] == sizes[1]

    task = jnp.asarray([2], dtype=jnp.int32)
    current = jnp.asarray([3], dtype=jnp.int32)
    nxt = jnp.asarray([4], dtype=jnp.int32)
    correct, correct_stats = oracle(task, current, nxt, batch_size=1)
    shuffled, _ = oracle(task, current, nxt, batch_size=1, intervention="shuffle")
    null_value, null_stats = oracle(task, current, nxt, batch_size=1, intervention="null")
    assert not np.array_equal(correct, shuffled)
    assert not np.array_equal(correct, null_value)
    np.testing.assert_array_equal(correct_stats["availability"], [True])
    np.testing.assert_array_equal(null_stats["availability"], [False])


def test_transition_adapter_routes_gradients_to_value_and_gate():
    adapter = TransitionConditionAdapter(6, 10, 8, 16, 24, rngs=nnx.Rngs(0))
    grads = nnx.grad(
        lambda module: module(
            jnp.asarray([1]),
            jnp.asarray([2]),
            jnp.asarray([3]),
            batch_size=1,
        )[0].astype(jnp.float32).sum()
    )(adapter)
    assert np.linalg.norm(np.asarray(grads.encoder_in.kernel.value)) > 0
    assert np.linalg.norm(np.asarray(grads.encoder_out.kernel.value)) > 0
    assert np.linalg.norm(np.asarray(grads.gate_in.kernel.value)) > 0
    assert np.linalg.norm(np.asarray(grads.gate_out.kernel.value)) > 0


def test_local_dynamics_adapter_is_action_conditioned_and_routes_gradients():
    adapter = LocalDynamicsAdapter(32, 6, 16, 24, rngs=nnx.Rngs(0))
    current = jnp.ones((2, 32), dtype=jnp.float32)
    actions = jnp.zeros((2, 5, 6), dtype=jnp.float32)
    shifted = actions.at[:, -1, :].set(1.0)
    prediction, route = adapter(current, actions)
    shifted_prediction, shifted_route = adapter(current, shifted)
    assert prediction.shape == shifted_prediction.shape == (2, 32)
    assert route.shape == shifted_route.shape == (2, 24)
    assert not np.array_equal(prediction, shifted_prediction)
    assert not np.array_equal(route, shifted_route)
    grads = nnx.grad(lambda module: module(current, shifted)[1].astype(jnp.float32).sum())(adapter)
    assert np.linalg.norm(np.asarray(grads.predictor_in.kernel.value)) > 0
    assert np.linalg.norm(np.asarray(grads.route_out.kernel.value)) > 0


def test_predictive_action_adapter_has_exact_zero_policy_route():
    adapter = PredictiveActionAdapter(32, 6, 5, 16, 24, 2, rngs=nnx.Rngs(0))
    current = jnp.arange(512, dtype=jnp.float32).reshape(1, 16, 32)
    actions = jnp.arange(30, dtype=jnp.float32).reshape(1, 5, 6) / 30

    prediction, route = adapter(current, actions)

    assert prediction.shape == (1, 4, 32)
    assert route.shape == (1, 5, 24)
    np.testing.assert_array_equal(route, jnp.zeros_like(route))


def test_predictive_action_adapter_uses_actions_and_supports_controls():
    adapter = PredictiveActionAdapter(32, 6, 5, 16, 24, 2, rngs=nnx.Rngs(1))
    current = jnp.arange(1024, dtype=jnp.float32).reshape(2, 16, 32) / 1024
    actions = jnp.arange(60, dtype=jnp.float32).reshape(2, 5, 6) / 60

    normal, _ = adapter(current, actions, intervention="normal")
    shuffled, _ = adapter(current, actions, intervention="shuffled")
    masked, _ = adapter(current, actions, intervention="masked")

    assert not np.array_equal(normal, shuffled)
    assert not np.array_equal(normal, masked)


def test_predictive_action_shuffle_is_effective_for_single_request():
    adapter = PredictiveActionAdapter(32, 6, 5, 16, 24, 2, rngs=nnx.Rngs(3))
    current = jnp.arange(512, dtype=jnp.float32).reshape(1, 16, 32) / 512
    actions = jnp.arange(30, dtype=jnp.float32).reshape(1, 5, 6) / 30

    normal, normal_route = adapter(current, actions, intervention="normal")
    shuffled, shuffled_route = adapter(current, actions, intervention="shuffled")

    assert not np.array_equal(normal, shuffled)
    np.testing.assert_array_equal(normal_route, shuffled_route)


def test_predictive_loss_detaches_visual_tokens_but_updates_adapter():
    adapter = PredictiveActionAdapter(32, 6, 5, 16, 24, 2, rngs=nnx.Rngs(2))
    current = jnp.arange(512, dtype=jnp.float32).reshape(1, 16, 32) / 512
    actions = jnp.arange(30, dtype=jnp.float32).reshape(1, 5, 6) / 30
    target = jnp.flip(_spatial_pool_tokens(current, 2), axis=1)

    def loss_for_current(value):
        prediction, _ = adapter(value, actions)
        return jnp.mean(jnp.square(prediction - target))

    current_grad = jax.grad(loss_for_current)(current)
    np.testing.assert_array_equal(current_grad, jnp.zeros_like(current_grad))

    module_grads = nnx.grad(
        lambda module: jnp.mean(jnp.square(module(current, actions)[0] - target))
    )(adapter)
    assert np.linalg.norm(np.asarray(module_grads.action_in.kernel.value)) > 0
    assert np.linalg.norm(np.asarray(module_grads.predictor_out.kernel.value)) > 0


def test_predictive_adapter_only_freeze_filter():
    config = _pi0_config.Pi0Config(
        pi05=True,
        predictive_adapter_mode="offline",
        freeze_mode="only_predictive_adapter",
    )
    abstract_model = nnx.eval_shape(config.create, jax.random.key(0))
    trainable = nnx.state(
        abstract_model,
        nnx.All(nnx.Param, nnx.Not(config.get_freeze_filter())),
    ).flat_state()
    assert trainable
    assert all("predictive_action_adapter" in path for path in trainable)


def test_predictive_only_model_does_not_instantiate_recurrence_route():
    config = _pi0_config.Pi0Config(
        pi05=True,
        predictive_adapter_mode="offline",
        freeze_mode="only_predictive_adapter",
    )
    abstract_model = nnx.eval_shape(config.create, jax.random.key(0))
    parameter_paths = [
        "/".join(map(str, path))
        for path in nnx.state(abstract_model, nnx.Param).flat_state()
    ]
    assert any("predictive_action_adapter" in path for path in parameter_paths)
    assert not any("recurrence_action_adapter" in path for path in parameter_paths)


def test_recurrence_adapter_has_zero_action_indexed_route_and_distribution_heads():
    adapter = RecurrenceActionAdapter(32, 6, 5, 16, 24, 11, rngs=nnx.Rngs(7))
    current = jnp.arange(512, dtype=jnp.float32).reshape(1, 16, 32) / 512
    actions = jnp.arange(30, dtype=jnp.float32).reshape(1, 5, 6) / 30

    predictions, route = adapter(current, actions)

    assert predictions["progress_logits"].shape == (1, 11)
    assert predictions["density_logits"].shape == (1, 11)
    assert predictions["boundary_logits"].shape == (1, 2)
    assert route.shape == (1, 5, 24)
    np.testing.assert_array_equal(route, jnp.zeros_like(route))


def test_recurrence_adapter_controls_work_for_single_policy_request():
    adapter = RecurrenceActionAdapter(32, 6, 5, 16, 24, 11, rngs=nnx.Rngs(8))
    current = jnp.arange(512, dtype=jnp.float32).reshape(1, 16, 32) / 512
    actions = jnp.arange(30, dtype=jnp.float32).reshape(1, 5, 6) / 30

    normal, _ = adapter(current, actions, intervention="normal")
    shuffled, _ = adapter(current, actions, intervention="shuffled")
    masked, _ = adapter(current, actions, intervention="masked")

    assert not np.array_equal(normal["progress_logits"], shuffled["progress_logits"])
    assert not np.array_equal(normal["density_logits"], masked["density_logits"])


def test_recurrence_heads_detach_visual_tokens_but_train_adapter():
    adapter = RecurrenceActionAdapter(32, 6, 5, 16, 24, 11, rngs=nnx.Rngs(9))
    current = jnp.arange(512, dtype=jnp.float32).reshape(1, 16, 32) / 512
    actions = jnp.arange(30, dtype=jnp.float32).reshape(1, 5, 6) / 30

    def loss_for_current(value):
        predictions, _ = adapter(value, actions)
        return sum(jnp.mean(jnp.square(logits)) for logits in predictions.values())

    current_grad = jax.grad(loss_for_current)(current)
    np.testing.assert_array_equal(current_grad, jnp.zeros_like(current_grad))

    module_grads = nnx.grad(
        lambda module: sum(
            jnp.mean(jnp.square(logits))
            for logits in module(current, actions)[0].values()
        )
    )(adapter)
    assert np.linalg.norm(np.asarray(module_grads.action_in.kernel.value)) > 0
    assert np.linalg.norm(np.asarray(module_grads.progress_head.kernel.value)) > 0


def test_mt5_models_instantiate_local_and_optional_transition_routes():
    local = nnx.eval_shape(
        _pi0_config.Pi0Config(pi05=True, lmwm_local_dynamics=True).create,
        jax.random.key(0),
    )
    combined = nnx.eval_shape(
        _pi0_config.Pi0Config(
            pi05=True,
            lmwm_local_dynamics=True,
            lmwm_transition_condition="learned",
        ).create,
        jax.random.key(1),
    )
    local_paths = ["/".join(map(str, path)) for path in nnx.state(local, nnx.Param).flat_state()]
    combined_paths = ["/".join(map(str, path)) for path in nnx.state(combined, nnx.Param).flat_state()]
    assert any("lmwm_local_adapter" in path for path in local_paths)
    assert not any("lmwm_transition_adapter" in path for path in local_paths)
    assert any("lmwm_local_adapter" in path for path in combined_paths)
    assert any("lmwm_transition_adapter" in path for path in combined_paths)


def test_transition_adapter_soft_embeddings_are_content_sensitive_and_differentiable():
    adapter = TransitionConditionAdapter(6, 10, 8, 16, 24, rngs=nnx.Rngs(0))
    current_logits = jnp.zeros((2, 10), dtype=jnp.float32)
    next_logits = jnp.zeros((2, 10), dtype=jnp.float32).at[:, 3].set(4.0)

    def route(logits):
        return adapter(
            jnp.asarray([1, 2]),
            None,
            None,
            batch_size=2,
            current_probs=jax.nn.softmax(current_logits),
            next_probs=jax.nn.softmax(logits),
        )[0].astype(jnp.float32)

    correct = route(next_logits)
    shifted, _ = adapter(
        jnp.asarray([1, 2]),
        None,
        None,
        batch_size=2,
        intervention="shuffle",
        current_probs=jax.nn.softmax(current_logits),
        next_probs=jax.nn.softmax(next_logits),
    )
    assert not np.array_equal(correct, shifted)
    assert np.linalg.norm(np.asarray(jax.grad(lambda value: route(value).sum())(next_logits))) > 0


def test_mt3_trackers_match_frozen_output_contract():
    current = CurrentFrameStageTracker(32, 16, 10, rngs=nnx.Rngs(0))
    history = HistoryProprioStageTracker(32, 14, 16, 10, rngs=nnx.Rngs(1))
    view_features = jnp.ones((2, 3, 32), dtype=jnp.bfloat16)
    proprio = jnp.ones((2, 3, 14), dtype=jnp.float32)

    current_logits = current(view_features)
    history_logits = history(view_features, proprio)
    assert current_logits[0].shape == current_logits[1].shape == (2, 10)
    assert history_logits[0].shape == history_logits[1].shape == (2, 10)
    assert np.all(np.isfinite(np.asarray(current_logits[0])))
    assert np.all(np.isfinite(np.asarray(history_logits[0])))


def test_pi0_learned_transition_model_instantiates_both_frozen_trackers():
    for tracker in ("current_frame", "history_proprio"):
        config = _pi0_config.Pi0Config(
            pi05=True,
            lmwm_transition_condition="learned",
            lmwm_transition_tracker=tracker,
        )
        abstract_model = nnx.eval_shape(config.create, jax.random.key(0))
        params = nnx.state(abstract_model, nnx.Param).flat_state()
        paths = ["/".join(map(str, path)) for path in params]
        assert any("lmwm_transition_tracker" in path for path in paths)
        assert any("lmwm_transition_adapter" in path for path in paths)


def test_transition_history_observation_roundtrip_and_preprocess():
    history = np.zeros((3, 32, 48, 3), dtype=np.uint8)
    observation = _model.Observation.from_dict(
        {
            "image": {
                name: np.zeros((32, 48, 3), dtype=np.uint8)
                for name in _model.IMAGE_KEYS
            },
            "image_mask": {name: np.asarray(True) for name in _model.IMAGE_KEYS},
            "state": np.zeros((14,), dtype=np.float32),
            "lmwm_transition_history_images": history,
            "lmwm_transition_history_state": np.zeros((3, 14), dtype=np.float32),
        }
    )
    batched = jax.tree.map(lambda value: value[None] if hasattr(value, "shape") else value, observation)
    processed = _model.preprocess_observation(None, batched, train=False)
    assert processed.lmwm_transition_history_images.shape == (1, 3, 224, 224, 3)
    assert processed.lmwm_transition_history_state.shape == (1, 3, 14)
