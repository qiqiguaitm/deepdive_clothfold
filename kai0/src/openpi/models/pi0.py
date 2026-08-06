import logging
import os

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
import numpy as np
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")

_LMWM_LIVE_OTHER_TASK_FEATURE = None
if os.environ.get("LMWM_LIVE_HINT_INTERVENTION", "").strip().lower() == "other-task":
    feature_path = os.environ.get("LMWM_LIVE_OTHER_TASK_FEATURE_PATH")
    if not feature_path:
        raise ValueError(
            "LMWM_LIVE_OTHER_TASK_FEATURE_PATH is required for the other-task intervention"
        )
    _LMWM_LIVE_OTHER_TASK_FEATURE = np.asarray(np.load(feature_path), dtype=np.float32).reshape(-1)

# X-VLA-style init for the soft prompt hub. Cached as a module-level constant so that
# eval_shape(init) and the actual jit(init) trace see the SAME function object —
# otherwise nnx puts the closure in static_fields and the prefix tree (out_shardings)
# vs full tree pytrees mismatch on closure identity.
_SOFT_PROMPT_INIT = nnx.initializers.normal(stddev=0.02)


def _spatial_pool_tokens(tokens: jax.Array, grid_size: int) -> jax.Array:
    """Pool square patch tokens to a fixed grid while preserving 2D order."""
    token_count = tokens.shape[1]
    side = int(np.sqrt(token_count))
    if side * side != token_count:
        raise ValueError(f"spatial condition requires square patch tokens, got {token_count}")
    if side % grid_size != 0:
        raise ValueError(f"patch grid side {side} is not divisible by output grid {grid_size}")
    block = side // grid_size
    batch_size, width = tokens.shape[0], tokens.shape[-1]
    tokens = tokens.reshape(batch_size, grid_size, block, grid_size, block, width)
    return tokens.mean(axis=(2, 4)).reshape(batch_size, grid_size * grid_size, width)


class SpatialConditionAdapter(nnx.Module):
    """Parameter-matched spatial condition used by all privileged-gate arms."""

    def __init__(self, input_dim: int, output_dim: int, grid_size: int, bottleneck_dim: int, *, rngs: nnx.Rngs):
        self.grid_size = grid_size
        self.no_goal = nnx.Param(
            jax.random.normal(rngs.params(), (grid_size * grid_size, input_dim), dtype=jnp.float32) * 0.02
        )
        self.adapter_in = nnx.Linear(input_dim, bottleneck_dim, rngs=rngs)
        self.adapter_out = nnx.Linear(bottleneck_dim, output_dim, rngs=rngs)
        self.gate = nnx.Linear(output_dim, 1, rngs=rngs)

    def __call__(
        self,
        source_tokens: jax.Array | None,
        *,
        batch_size: int,
        available: jax.Array | None = None,
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        no_goal = jnp.broadcast_to(self.no_goal[None, ...], (batch_size, *self.no_goal.shape))
        if source_tokens is None:
            source = no_goal
            availability = jnp.zeros((batch_size,), dtype=jnp.bool_)
        else:
            pooled = jax.lax.stop_gradient(_spatial_pool_tokens(source_tokens, self.grid_size))
            availability = (
                jnp.ones((batch_size,), dtype=jnp.bool_)
                if available is None
                else available.astype(jnp.bool_)
            )
            source = jnp.where(availability[:, None, None], pooled, no_goal)

        adapted = self.adapter_out(nnx.swish(self.adapter_in(source.astype(jnp.float32))))
        gate = jax.nn.sigmoid(self.gate(adapted))
        condition = (adapted * gate).astype(jnp.bfloat16)
        stats = {
            "availability": availability,
            "gate": gate[..., 0],
            "token_norm": jnp.linalg.norm(condition.astype(jnp.float32), axis=-1),
        }
        return condition, stats


class LocalDynamicsAdapter(nnx.Module):
    """Action-conditioned fixed-horizon predictor routed to the action expert."""

    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
        hidden_dim: int,
        output_dim: int,
        *,
        rngs: nnx.Rngs,
    ):
        self.predictor_in = nnx.Linear(feature_dim + 3 * action_dim, hidden_dim, rngs=rngs)
        self.predictor_out = nnx.Linear(hidden_dim, feature_dim, rngs=rngs)
        self.route_in = nnx.Linear(feature_dim, hidden_dim, rngs=rngs)
        self.route_out = nnx.Linear(hidden_dim, output_dim, rngs=rngs)
        self.route_gate = nnx.Linear(output_dim, 1, rngs=rngs)

    def __call__(
        self, current_feature: jax.Array, candidate_actions: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        current = jax.lax.stop_gradient(current_feature.astype(jnp.float32))
        actions = candidate_actions.astype(jnp.float32)
        mean = jnp.mean(actions, axis=-2)
        displacement = actions[..., -1, :] - actions[..., 0, :]
        std = jnp.sqrt(jnp.mean(jnp.square(actions - mean[..., None, :]), axis=-2) + 1e-6)
        predictor_input = jnp.concatenate([current, mean, displacement, std], axis=-1)
        residual = self.predictor_out(nnx.swish(self.predictor_in(predictor_input)))
        prediction = current + residual
        route = self.route_out(nnx.swish(self.route_in(residual)))
        gate = jax.nn.sigmoid(self.route_gate(route))
        return prediction, (route * gate).astype(jnp.bfloat16)


class PredictiveActionAdapter(nnx.Module):
    """Patch-level future predictor with an action-indexed, zero-output route."""

    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
        action_horizon: int,
        hidden_dim: int,
        output_dim: int,
        grid_size: int,
        *,
        rngs: nnx.Rngs,
    ):
        self.grid_size = grid_size
        self.action_horizon = action_horizon
        self.current_in = nnx.Linear(feature_dim, hidden_dim, rngs=rngs)
        self.action_in = nnx.Linear(action_dim, hidden_dim, rngs=rngs)
        self.action_position = nnx.Param(
            jax.random.normal(
                rngs.params(), (action_horizon, hidden_dim), dtype=jnp.float32
            )
            * 0.02
        )
        self.action_summary = nnx.Linear(action_horizon * hidden_dim, hidden_dim, rngs=rngs)
        self.predictor_hidden = nnx.Linear(2 * hidden_dim, hidden_dim, rngs=rngs)
        self.predictor_out = nnx.Linear(hidden_dim, feature_dim, rngs=rngs)
        self.route_hidden = nnx.Linear(2 * hidden_dim, hidden_dim, rngs=rngs)
        self.route_out = nnx.Linear(
            hidden_dim,
            output_dim,
            kernel_init=nnx.initializers.zeros,
            bias_init=nnx.initializers.zeros,
            rngs=rngs,
        )

    def _intervene(self, actions: jax.Array, intervention: str) -> jax.Array:
        if intervention in {"normal", "zero_gate"}:
            return actions
        if intervention == "shuffled":
            # RoboTwin serves one policy request at a time, so a batch-axis
            # permutation would be an identity. Reverse the temporal proposal
            # instead to provide a deterministic, batch-size-independent wrong
            # action control.
            return jnp.flip(actions, axis=-2)
        if intervention == "masked":
            return jnp.zeros_like(actions)
        raise ValueError(f"unsupported predictive adapter intervention={intervention!r}")

    def __call__(
        self,
        current_tokens: jax.Array,
        candidate_actions: jax.Array,
        *,
        intervention: str = "normal",
    ) -> tuple[jax.Array, jax.Array]:
        if candidate_actions.shape[-2] != self.action_horizon:
            raise ValueError(
                "predictive adapter action horizon mismatch: "
                f"expected {self.action_horizon}, got {candidate_actions.shape[-2]}"
            )
        current = jax.lax.stop_gradient(
            _spatial_pool_tokens(current_tokens.astype(jnp.float32), self.grid_size)
        )
        actions = self._intervene(candidate_actions.astype(jnp.float32), intervention)
        action_tokens = nnx.swish(self.action_in(actions) + self.action_position[None])
        action_context = nnx.swish(
            self.action_summary(action_tokens.reshape(action_tokens.shape[0], -1))
        )

        current_hidden = nnx.swish(self.current_in(current))
        context = jnp.broadcast_to(
            action_context[:, None, :], (*current_hidden.shape[:-1], action_context.shape[-1])
        )
        prediction_delta = self.predictor_out(
            nnx.swish(self.predictor_hidden(jnp.concatenate([current_hidden, context], axis=-1)))
        )
        prediction = current + prediction_delta

        predicted_summary = jnp.mean(
            nnx.swish(self.current_in(prediction_delta)), axis=1
        )
        route_context = jnp.broadcast_to(
            predicted_summary[:, None, :], action_tokens.shape
        )
        route = self.route_out(
            nnx.swish(
                self.route_hidden(
                    jnp.concatenate([action_tokens, route_context], axis=-1)
                )
            )
        )
        if intervention == "zero_gate":
            route = jnp.zeros_like(route)
        return prediction, route.astype(jnp.bfloat16)


class RecurrenceActionAdapter(nnx.Module):
    """Distributional CRAVE teacher heads with an action-indexed policy route."""

    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
        action_horizon: int,
        hidden_dim: int,
        output_dim: int,
        bins: int,
        *,
        rngs: nnx.Rngs,
    ):
        if bins < 3:
            raise ValueError("recurrence distribution heads require at least three bins")
        self.action_horizon = action_horizon
        self.bins = bins
        self.current_in = nnx.Linear(feature_dim, hidden_dim, rngs=rngs)
        self.action_in = nnx.Linear(action_dim, hidden_dim, rngs=rngs)
        self.action_position = nnx.Param(
            jax.random.normal(
                rngs.params(), (action_horizon, hidden_dim), dtype=jnp.float32
            )
            * 0.02
        )
        self.context_hidden = nnx.Linear(2 * hidden_dim, hidden_dim, rngs=rngs)
        self.progress_head = nnx.Linear(hidden_dim, bins, rngs=rngs)
        self.density_head = nnx.Linear(hidden_dim, bins, rngs=rngs)
        self.boundary_head = nnx.Linear(hidden_dim, 2, rngs=rngs)
        self.route_hidden = nnx.Linear(2 * hidden_dim, hidden_dim, rngs=rngs)
        self.route_out = nnx.Linear(
            hidden_dim,
            output_dim,
            kernel_init=nnx.initializers.zeros,
            bias_init=nnx.initializers.zeros,
            rngs=rngs,
        )

    @staticmethod
    def _intervene(actions: jax.Array, intervention: str) -> jax.Array:
        if intervention in {"normal", "zero_gate"}:
            return actions
        if intervention == "shuffled":
            return jnp.flip(actions, axis=-2)
        if intervention == "masked":
            return jnp.zeros_like(actions)
        raise ValueError(f"unsupported recurrence adapter intervention={intervention!r}")

    def __call__(
        self,
        current_tokens: jax.Array,
        candidate_actions: jax.Array,
        *,
        intervention: str = "normal",
    ) -> tuple[dict[str, jax.Array], jax.Array]:
        if candidate_actions.shape[-2] != self.action_horizon:
            raise ValueError(
                "recurrence adapter action horizon mismatch: "
                f"expected {self.action_horizon}, got {candidate_actions.shape[-2]}"
            )
        current = jax.lax.stop_gradient(
            jnp.mean(current_tokens.astype(jnp.float32), axis=1)
        )
        actions = self._intervene(candidate_actions.astype(jnp.float32), intervention)
        action_tokens = nnx.swish(self.action_in(actions) + self.action_position[None])
        current_hidden = nnx.swish(self.current_in(current))
        action_summary = jnp.mean(action_tokens, axis=1)
        context = nnx.swish(
            self.context_hidden(jnp.concatenate([current_hidden, action_summary], axis=-1))
        )
        predictions = {
            "progress_logits": self.progress_head(context),
            "density_logits": self.density_head(context),
            "boundary_logits": self.boundary_head(context),
        }
        route_context = jnp.broadcast_to(context[:, None, :], action_tokens.shape)
        route = self.route_out(
            nnx.swish(
                self.route_hidden(
                    jnp.concatenate([action_tokens, route_context], axis=-1)
                )
            )
        )
        if intervention == "zero_gate":
            route = jnp.zeros_like(route)
        return predictions, route.astype(jnp.bfloat16)


class TransitionConditionAdapter(nnx.Module):
    """Gated transition route applied directly to action-expert tokens."""

    def __init__(
        self,
        num_tasks: int,
        num_stages: int,
        embed_dim: int,
        hidden_dim: int,
        output_dim: int,
        *,
        rngs: nnx.Rngs,
    ):
        self.num_tasks = num_tasks
        self.num_stages = num_stages
        # The final entry in each vocabulary is the explicit null sentinel.
        self.task_embed = nnx.Embed(num_tasks + 1, embed_dim, rngs=rngs)
        self.current_embed = nnx.Embed(num_stages + 1, embed_dim, rngs=rngs)
        self.next_embed = nnx.Embed(num_stages + 1, embed_dim, rngs=rngs)
        self.encoder_in = nnx.Linear(3 * embed_dim, hidden_dim, rngs=rngs)
        self.encoder_out = nnx.Linear(hidden_dim, output_dim, rngs=rngs)
        self.gate_in = nnx.Linear(3 * embed_dim, hidden_dim, rngs=rngs)
        self.gate_out = nnx.Linear(hidden_dim, 1, rngs=rngs)

    def __call__(
        self,
        task: jax.Array | None,
        current: jax.Array | None,
        nxt: jax.Array | None,
        *,
        batch_size: int,
        available: jax.Array | None = None,
        intervention: str = "correct",
        task_probs: jax.Array | None = None,
        current_probs: jax.Array | None = None,
        next_probs: jax.Array | None = None,
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        soft = current_probs is not None or next_probs is not None or task_probs is not None
        if soft and (current_probs is None or next_probs is None):
            raise ValueError("soft transition routing requires both current_probs and next_probs")
        if soft:
            if task is None:
                task = jnp.full((batch_size,), self.num_tasks, dtype=jnp.int32)
                availability = jnp.zeros((batch_size,), dtype=jnp.bool_)
            else:
                task = task.astype(jnp.int32)
                availability = (
                    jnp.ones((batch_size,), dtype=jnp.bool_)
                    if available is None
                    else available.astype(jnp.bool_)
                )
            current = jnp.zeros((batch_size,), dtype=jnp.int32)
            nxt = jnp.zeros((batch_size,), dtype=jnp.int32)
        elif task is None or current is None or nxt is None:
            task = jnp.full((batch_size,), self.num_tasks, dtype=jnp.int32)
            current = jnp.full((batch_size,), self.num_stages, dtype=jnp.int32)
            nxt = jnp.full((batch_size,), self.num_stages, dtype=jnp.int32)
            availability = jnp.zeros((batch_size,), dtype=jnp.bool_)
        else:
            task = task.astype(jnp.int32)
            current = current.astype(jnp.int32)
            nxt = nxt.astype(jnp.int32)
            availability = (
                jnp.ones((batch_size,), dtype=jnp.bool_)
                if available is None
                else available.astype(jnp.bool_)
            )

        if intervention == "null":
            availability = jnp.zeros_like(availability)
        elif intervention == "shuffle":
            if soft:
                next_probs = jnp.roll(next_probs, shift=1, axis=-1)
            else:
                nxt = jnp.mod(nxt + 1, self.num_stages)
        elif intervention == "cross-task":
            if task_probs is not None:
                task_probs = jnp.roll(task_probs, shift=1, axis=-1)
            else:
                task = jnp.mod(task + 1, self.num_tasks)
        elif intervention != "correct":
            raise ValueError(f"unsupported transition intervention={intervention!r}")

        if soft:
            current_probs = current_probs.astype(jnp.float32)
            next_probs = next_probs.astype(jnp.float32)
            if current_probs.shape != (batch_size, self.num_stages):
                raise ValueError(
                    f"current_probs must have shape {(batch_size, self.num_stages)}, got {current_probs.shape}"
                )
            if next_probs.shape != (batch_size, self.num_stages):
                raise ValueError(
                    f"next_probs must have shape {(batch_size, self.num_stages)}, got {next_probs.shape}"
                )
            if task_probs is None:
                task_probs = jax.nn.one_hot(task, self.num_tasks, dtype=jnp.float32)
            else:
                task_probs = task_probs.astype(jnp.float32)
                if task_probs.shape != (batch_size, self.num_tasks):
                    raise ValueError(
                        f"task_probs must have shape {(batch_size, self.num_tasks)}, got {task_probs.shape}"
                    )

            task_value = task_probs @ self.task_embed.embedding.value[: self.num_tasks]
            current_value = current_probs @ self.current_embed.embedding.value[: self.num_stages]
            next_value = next_probs @ self.next_embed.embedding.value[: self.num_stages]
            task_null = self.task_embed(jnp.full((batch_size,), self.num_tasks, dtype=jnp.int32))
            current_null = self.current_embed(jnp.full((batch_size,), self.num_stages, dtype=jnp.int32))
            next_null = self.next_embed(jnp.full((batch_size,), self.num_stages, dtype=jnp.int32))
            task_value = jnp.where(availability[:, None], task_value, task_null)
            current_value = jnp.where(availability[:, None], current_value, current_null)
            next_value = jnp.where(availability[:, None], next_value, next_null)
            encoded = jnp.concatenate([task_value, current_value, next_value], axis=-1).astype(jnp.float32)
        else:
            task = jnp.where(availability, task, self.num_tasks)
            current = jnp.where(availability, current, self.num_stages)
            nxt = jnp.where(availability, nxt, self.num_stages)
            encoded = jnp.concatenate(
                [self.task_embed(task), self.current_embed(current), self.next_embed(nxt)], axis=-1
            ).astype(jnp.float32)
        value = self.encoder_out(nnx.swish(self.encoder_in(encoded)))
        gate = jax.nn.sigmoid(self.gate_out(nnx.swish(self.gate_in(encoded))))
        condition = (value * gate).astype(jnp.bfloat16)
        return condition, {
            "availability": availability,
            "gate": gate[..., 0],
            "condition_norm": jnp.linalg.norm(condition.astype(jnp.float32), axis=-1),
        }


class CurrentFrameStageTracker(nnx.Module):
    """Frozen MT3 current-frame tracker architecture."""

    def __init__(self, feature_dim: int, hidden_dim: int, num_stages: int, *, rngs: nnx.Rngs):
        self.hidden1 = nnx.Linear(3 * feature_dim, hidden_dim, rngs=rngs)
        self.hidden2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.current_head = nnx.Linear(hidden_dim, num_stages, rngs=rngs)
        self.next_head = nnx.Linear(hidden_dim, num_stages, rngs=rngs)

    def __call__(self, view_features: jax.Array) -> tuple[jax.Array, jax.Array]:
        if view_features.ndim != 3 or view_features.shape[1] != 3:
            raise ValueError(f"view_features must have shape [batch, 3, feature], got {view_features.shape}")
        hidden = nnx.gelu(self.hidden1(view_features.astype(jnp.float32).reshape(view_features.shape[0], -1)))
        hidden = nnx.gelu(self.hidden2(hidden))
        return self.current_head(hidden), self.next_head(hidden)


class HistoryProprioStageTracker(nnx.Module):
    """Frozen MT3 three-frame base-camera plus proprioception tracker."""

    def __init__(
        self,
        feature_dim: int,
        proprio_dim: int,
        hidden_dim: int,
        num_stages: int,
        *,
        rngs: nnx.Rngs,
    ):
        # Keep separate input/recurrent biases so PyTorch nn.GRU tracker-only
        # checkpoints transfer exactly, including the reset-gated candidate bias.
        self.input_proj = nnx.Linear(feature_dim + proprio_dim, 3 * hidden_dim, rngs=rngs)
        self.recurrent_proj = nnx.Linear(hidden_dim, 3 * hidden_dim, rngs=rngs)
        self.hidden_dim = hidden_dim
        self.current_head = nnx.Linear(hidden_dim, num_stages, rngs=rngs)
        self.next_head = nnx.Linear(hidden_dim, num_stages, rngs=rngs)

    def __call__(self, history_features: jax.Array, proprio: jax.Array) -> tuple[jax.Array, jax.Array]:
        if history_features.ndim != 3 or history_features.shape[1] != 3:
            raise ValueError(
                f"history_features must have shape [batch, 3, feature], got {history_features.shape}"
            )
        if proprio.shape[:2] != history_features.shape[:2]:
            raise ValueError(
                f"proprio must share [batch, time] with history_features, got {proprio.shape}"
            )
        sequence = jnp.concatenate(
            [history_features.astype(jnp.float32), proprio.astype(jnp.float32)], axis=-1
        )
        carry = jnp.zeros((sequence.shape[0], self.hidden_dim), dtype=jnp.float32)
        for index in range(sequence.shape[1]):
            input_r, input_z, input_n = jnp.split(self.input_proj(sequence[:, index]), 3, axis=-1)
            hidden_r, hidden_z, hidden_n = jnp.split(self.recurrent_proj(carry), 3, axis=-1)
            reset = jax.nn.sigmoid(input_r + hidden_r)
            update = jax.nn.sigmoid(input_z + hidden_z)
            candidate = jnp.tanh(input_n + reset * hidden_n)
            carry = (1.0 - update) * candidate + update * carry
        return self.current_head(carry), self.next_head(carry)


def _dct2_last_time_axis(x: jnp.ndarray) -> jnp.ndarray:
    """Orthonormal DCT-II along axis=-2 (time). x shape [..., T, D]. Returns same shape.

    Uses direct matmul implementation: O(T^2) per-element but fine for T~50.
    Formula: y_k = alpha_k * sum_n x_n * cos(pi * (n+0.5) * k / T)
    with alpha_0 = sqrt(1/T), alpha_{k>0} = sqrt(2/T).
    """
    T = x.shape[-2]
    n = jnp.arange(T, dtype=x.dtype)
    k = jnp.arange(T, dtype=x.dtype)
    kernel = jnp.cos(jnp.pi * (n[:, None] + 0.5) * k[None, :] / T)  # [T, T]
    scale = jnp.full((T,), jnp.sqrt(2.0 / T), dtype=x.dtype).at[0].set(jnp.sqrt(1.0 / T))
    kernel = kernel * scale[None, :]
    # Contract over time axis: x [..., T, D] @ kernel [T, T] -> [..., T, D]
    return jnp.einsum("...td,tk->...kd", x, kernel)


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        self.augment_level = getattr(config, "augment_level", "mild")
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        vision_mlp_lora_config = None
        if getattr(config, "vision_mlp_lora_rank", None):
            import openpi.models.lora as _lora
            vision_mlp_lora_config = _lora.LoRAConfig(
                rank=config.vision_mlp_lora_rank,
                alpha=config.vision_mlp_lora_alpha,
            )
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
                mlp_lora_config=vision_mlp_lora_config,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # X-VLA style soft prompt hub: per-domain learnable tokens prepended to LLM input.
        # Only created when enabled in config; otherwise this attribute is absent and embed_prefix is a no-op.
        self.soft_prompt_num_domains = int(getattr(config, "soft_prompt_num_domains", 0) or 0)
        self.soft_prompt_len = int(getattr(config, "soft_prompt_len", 0) or 0)
        if self.soft_prompt_num_domains > 0 and self.soft_prompt_len > 0:
            # X-VLA paper uses nn.init.normal_(std=0.02) for soft prompts (matches
            # GPT-style word-embedding initialization). nnx default variance_scaling
            # gives std ≈ 0.011 here (close but not identical) — use explicit normal
            # to match the X-VLA prototype.
            self.soft_prompt_hub = nnx.Embed(
                num_embeddings=self.soft_prompt_num_domains,
                features=self.soft_prompt_len * paligemma_config.width,
                embedding_init=_SOFT_PROMPT_INIT,
                rngs=rngs,
            )

        # Track C action head conditioning hub (方案 A: Concat domain token at action
        # expert input). 1 learnable token per domain, prepended in embed_suffix.
        # PaliGemma is unaware of domain — only the action expert sees the conditioning.
        # See docs/deployment/strategy/cross_embodiment_strategy.md §5.3 for design.
        # Backward compatible: when num_domains=0 (default), no module created;
        # old ckpts load unchanged.
        self.action_head_cond_num_domains = int(getattr(config, "action_head_cond_num_domains", 0) or 0)
        # TAC (Training-time Action Conditioning) — see Pi0Config docstring.
        self.tac_enabled = bool(getattr(config, "tac_enabled", False))
        self.tac_max_delay = int(getattr(config, "tac_max_delay", 6) or 6)
        if self.action_head_cond_num_domains > 0:
            self.action_head_cond_hub = nnx.Embed(
                num_embeddings=self.action_head_cond_num_domains,
                features=action_expert_config.width,
                embedding_init=_SOFT_PROMPT_INIT,  # reuse N(0, 0.02) init
                rngs=rngs,
            )

        # LMWM hint injection (PLAN_pi05_lmwm_sameencoder §2.2). One learnable Linear
        # projects the offline hint (D→width) which is then injected as extra token(s).
        # Backward compatible: dim=0 (default) → no module created, embed_* is a no-op,
        # forward is bit-identical to upstream pi05; old ckpts load unchanged.
        self.lmwm_hint_dim = int(getattr(config, "lmwm_hint_dim", 0) or 0)
        self.lmwm_hint_len = int(getattr(config, "lmwm_hint_len", 1) or 1)
        self.lmwm_hint_target = str(getattr(config, "lmwm_hint_target", "prefix") or "prefix")
        self.lmwm_live_hint = bool(getattr(config, "lmwm_live_hint", False))
        self.lmwm_live_residual = bool(getattr(config, "lmwm_live_residual", True))
        self.lmwm_live_loss_weight = float(getattr(config, "lmwm_live_loss_weight", 0.0) or 0.0)
        self.lmwm_live_intervention = os.environ.get("LMWM_LIVE_HINT_INTERVENTION", "correct").strip().lower()
        if self.lmwm_live_intervention not in {"correct", "current", "zero", "shuffle", "other-task"}:
            raise ValueError(f"Unsupported LMWM_LIVE_HINT_INTERVENTION={self.lmwm_live_intervention!r}")
        self.fuse_vision_batch = bool(getattr(config, "fuse_vision_batch", False))
        self.lmwm_spatial_condition = str(getattr(config, "lmwm_spatial_condition", "none") or "none")
        self.lmwm_spatial_grid_size = int(getattr(config, "lmwm_spatial_grid_size", 4) or 4)
        if self.lmwm_spatial_condition not in {"none", "no_goal", "current", "privileged"}:
            raise ValueError(f"Unsupported lmwm_spatial_condition={self.lmwm_spatial_condition!r}")
        if self.lmwm_spatial_condition != "none":
            self.lmwm_spatial_adapter = SpatialConditionAdapter(
                paligemma_config.width,
                paligemma_config.width,
                self.lmwm_spatial_grid_size,
                int(getattr(config, "lmwm_spatial_bottleneck_dim", 256) or 256),
                rngs=rngs,
            )
        self.lmwm_transition_condition = str(
            getattr(config, "lmwm_transition_condition", "none") or "none"
        )
        if self.lmwm_transition_condition not in {"none", "oracle", "null", "learned"}:
            raise ValueError(
                f"Unsupported lmwm_transition_condition={self.lmwm_transition_condition!r}"
            )
        default_transition_intervention = (
            "predicted" if self.lmwm_transition_condition == "learned" else "correct"
        )
        self.lmwm_transition_intervention = os.environ.get(
            "LMWM_TRANSITION_INTERVENTION", default_transition_intervention
        ).strip().lower()
        if self.lmwm_transition_condition == "null":
            self.lmwm_transition_intervention = "null"
        valid_interventions = {"correct", "shuffle", "cross-task", "null"}
        if self.lmwm_transition_condition == "learned":
            valid_interventions |= {"predicted", "oracle", "within_task"}
        if self.lmwm_transition_intervention not in valid_interventions:
            raise ValueError(
                f"Unsupported LMWM_TRANSITION_INTERVENTION={self.lmwm_transition_intervention!r}"
            )
        if self.lmwm_transition_condition != "none":
            self.lmwm_transition_adapter = TransitionConditionAdapter(
                int(getattr(config, "lmwm_transition_num_tasks", 6)),
                int(getattr(config, "lmwm_transition_num_stages", 10)),
                int(getattr(config, "lmwm_transition_embed_dim", 128)),
                int(getattr(config, "lmwm_transition_hidden_dim", 512)),
                action_expert_config.width,
                rngs=rngs,
            )
        if self.lmwm_transition_condition == "learned":
            self.lmwm_transition_tracker_type = str(
                getattr(config, "lmwm_transition_tracker", "current_frame") or "current_frame"
            )
            feature_dim = int(getattr(config, "lmwm_transition_tracker_feature_dim", 2048))
            num_stages = int(getattr(config, "lmwm_transition_num_stages", 10))
            if self.lmwm_transition_tracker_type == "current_frame":
                self.lmwm_transition_tracker = CurrentFrameStageTracker(
                    feature_dim,
                    int(getattr(config, "lmwm_transition_tracker_current_hidden_dim", 512)),
                    num_stages,
                    rngs=rngs,
                )
            elif self.lmwm_transition_tracker_type == "history_proprio":
                self.lmwm_transition_tracker = HistoryProprioStageTracker(
                    feature_dim,
                    int(getattr(config, "lmwm_transition_tracker_proprio_dim", 14)),
                    int(getattr(config, "lmwm_transition_tracker_history_hidden_dim", 256)),
                    num_stages,
                    rngs=rngs,
                )
            else:
                raise ValueError(
                    f"Unsupported lmwm_transition_tracker={self.lmwm_transition_tracker_type!r}"
                )
            self.lmwm_transition_auxiliary_loss_weight = float(
                getattr(config, "lmwm_transition_auxiliary_loss_weight", 0.1)
            )
            self.lmwm_transition_dropout = nnx.Dropout(
                rate=float(getattr(config, "lmwm_transition_dropout_probability", 0.2)),
                broadcast_dims=(1,),
                rngs=rngs,
            )
        self.lmwm_local_dynamics = bool(getattr(config, "lmwm_local_dynamics", False))
        self.lmwm_local_loss_weight = float(getattr(config, "lmwm_local_loss_weight", 0.05))
        if self.lmwm_local_dynamics:
            self.lmwm_local_adapter = LocalDynamicsAdapter(
                paligemma_config.width,
                self.action_dim,
                int(getattr(config, "lmwm_local_hidden_dim", 512)),
                action_expert_config.width,
                rngs=rngs,
            )
        self.predictive_adapter_mode = str(
            getattr(config, "predictive_adapter_mode", "none") or "none"
        )
        self.predictive_adapter_loss_weight = float(
            getattr(config, "predictive_adapter_loss_weight", 1.0)
        )
        self.predictive_adapter_intervention = os.environ.get(
            "PREDICTIVE_ACTION_INTERVENTION",
            str(getattr(config, "predictive_adapter_intervention", "normal")),
        ).strip().lower()
        if self.predictive_adapter_intervention not in {
            "normal",
            "shuffled",
            "masked",
            "zero_gate",
        }:
            raise ValueError(
                "unsupported predictive adapter intervention="
                f"{self.predictive_adapter_intervention!r}"
            )
        if self.predictive_adapter_mode != "none":
            self.predictive_action_adapter = PredictiveActionAdapter(
                paligemma_config.width,
                self.action_dim,
                self.action_horizon,
                int(getattr(config, "predictive_adapter_hidden_dim", 512)),
                action_expert_config.width,
                int(getattr(config, "predictive_adapter_grid_size", 4)),
                rngs=rngs,
            )
        self.recurrence_adapter_mode = str(
            getattr(config, "recurrence_adapter_mode", "none") or "none"
        )
        self.recurrence_adapter_loss_weight = float(
            getattr(config, "recurrence_adapter_loss_weight", 0.1)
        )
        self.recurrence_adapter_intervention = os.environ.get(
            "RECURRENCE_ACTION_INTERVENTION",
            str(getattr(config, "recurrence_adapter_intervention", "normal")),
        ).strip().lower()
        if self.recurrence_adapter_intervention not in {
            "normal",
            "shuffled",
            "masked",
            "zero_gate",
        }:
            raise ValueError(
                "unsupported recurrence adapter intervention="
                f"{self.recurrence_adapter_intervention!r}"
            )
        if self.recurrence_adapter_mode != "none":
            self.recurrence_action_adapter = RecurrenceActionAdapter(
                paligemma_config.width,
                self.action_dim,
                self.action_horizon,
                int(getattr(config, "recurrence_adapter_hidden_dim", 256)),
                action_expert_config.width,
                int(getattr(config, "recurrence_adapter_bins", 21)),
                rngs=rngs,
            )
        if self.lmwm_hint_dim > 0:
            if self.lmwm_hint_target not in ("prefix", "suffix"):
                raise ValueError(f"lmwm_hint_target must be 'prefix' or 'suffix', got {self.lmwm_hint_target!r}")
            # prefix → paligemma (VLM) width; suffix → action_expert width.
            hint_out_width = paligemma_config.width if self.lmwm_hint_target == "prefix" else action_expert_config.width
            if self.lmwm_live_hint:
                if self.lmwm_hint_target != "prefix":
                    raise ValueError("lmwm_live_hint currently supports prefix target only")
                if self.lmwm_live_intervention == "other-task" and (
                    _LMWM_LIVE_OTHER_TASK_FEATURE is None
                    or _LMWM_LIVE_OTHER_TASK_FEATURE.shape != (hint_out_width,)
                    or not np.all(np.isfinite(_LMWM_LIVE_OTHER_TASK_FEATURE))
                ):
                    shape = None if _LMWM_LIVE_OTHER_TASK_FEATURE is None else _LMWM_LIVE_OTHER_TASK_FEATURE.shape
                    raise ValueError(
                        "Invalid A3 other-task feature: "
                        f"shape={shape}, expected=({hint_out_width},), "
                        f"finite={_LMWM_LIVE_OTHER_TASK_FEATURE is not None and np.all(np.isfinite(_LMWM_LIVE_OTHER_TASK_FEATURE))}"
                    )
                self.lmwm_live_pred_in = nnx.Linear(hint_out_width, hint_out_width, rngs=rngs)
                self.lmwm_live_pred_out = nnx.Linear(hint_out_width, hint_out_width, rngs=rngs)
            else:
                self.lmwm_hint_proj = nnx.Linear(self.lmwm_hint_dim, hint_out_width, rngs=rngs)

        # Store augment_level so compute_loss can read it (Pi0 doesn't keep full config).
        self.augment_level = getattr(config, "augment_level", "mild")
        self.awbc_loss_weight = getattr(config, "awbc_loss_weight", False)
        self.use_dct_loss = getattr(config, "use_dct_loss", False)
        if self.use_dct_loss:
            self._dct_loss_weight = config.dct_loss_weight
            self._dct_low_freq_weight = config.dct_low_freq_weight
            self._dct_high_freq_weight = config.dct_high_freq_weight

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        tokens, input_mask, ar_mask, _ = self._embed_prefix_impl(obs, want_lmwm_aux=False)
        return tokens, input_mask, ar_mask

    def encode_base_image(self, image: at.Array) -> at.Array:
        """Return pi0.5 SigLIP tokens for an already preprocessed base image."""
        return self.PaliGemma.img(image, train=False)[0]

    def _predictive_tokens(
        self, obs: _model.Observation
    ) -> tuple[jax.Array, jax.Array]:
        if obs.lmwm_target_image is None:
            raise ValueError("predictive adapter requires lmwm_target_image")
        if "base_0_rgb" not in obs.images:
            raise ValueError("predictive adapter requires base_0_rgb")

        current_image = obs.images["base_0_rgb"]
        target_image = obs.lmwm_target_image
        if current_image.shape != target_image.shape:
            raise ValueError(
                "offline predictive current/target image shape mismatch: "
                f"{current_image.shape} != {target_image.shape}"
            )

        batch_size = current_image.shape[0]
        fused_tokens, _ = self.PaliGemma.img(
            jnp.concatenate([current_image, target_image], axis=0), train=False
        )
        current_tokens = jax.lax.stop_gradient(fused_tokens[:batch_size])
        target_tokens = jax.lax.stop_gradient(
            _spatial_pool_tokens(
                fused_tokens[batch_size:].astype(jnp.float32),
                self.predictive_action_adapter.grid_size,
            )
        )
        return current_tokens, target_tokens

    def _predictive_patch_cosine_from_tokens(
        self,
        current_tokens: jax.Array,
        target_tokens: jax.Array,
        actions: _model.Actions,
        *,
        intervention: str,
    ) -> jax.Array:
        if intervention == "zero_gate":
            intervention = "normal"
        prediction, _ = self.predictive_action_adapter(
            current_tokens, actions, intervention=intervention
        )
        pred_unit = prediction / jnp.maximum(
            jnp.linalg.norm(prediction, axis=-1, keepdims=True), 1e-6
        )
        target_unit = target_tokens / jnp.maximum(
            jnp.linalg.norm(target_tokens, axis=-1, keepdims=True), 1e-6
        )
        return jnp.sum(pred_unit * target_unit, axis=-1)

    def _predictive_patch_cosine(
        self,
        obs: _model.Observation,
        actions: _model.Actions,
        *,
        intervention: str,
    ) -> jax.Array:
        """Return per-sample, per-patch cosine for a preprocessed observation."""
        current_tokens, target_tokens = self._predictive_tokens(obs)
        return self._predictive_patch_cosine_from_tokens(
            current_tokens,
            target_tokens,
            actions,
            intervention=intervention,
        )

    def compute_predictive_cosine(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        intervention: str,
    ) -> jax.Array:
        """Return held-out per-sample cosine under a fixed action intervention."""
        observation = _model.preprocess_observation(
            rng, observation, train=False, augment_level=self.augment_level
        )
        patch_cosine = self._predictive_patch_cosine(
            observation, actions, intervention=intervention
        )
        return jnp.mean(patch_cosine, axis=-1)

    def compute_predictive_control_cosines(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
    ) -> dict[str, jax.Array]:
        """Evaluate all preregistered P0 controls with one vision encoding."""
        observation = _model.preprocess_observation(
            rng, observation, train=False, augment_level=self.augment_level
        )
        current_tokens, target_tokens = self._predictive_tokens(observation)
        return {
            intervention: jnp.mean(
                self._predictive_patch_cosine_from_tokens(
                    current_tokens,
                    target_tokens,
                    actions,
                    intervention=intervention,
                ),
                axis=-1,
            )
            for intervention in ("normal", "shuffled", "masked")
        }

    def _compute_offline_predictive_loss(
        self, obs: _model.Observation, actions: _model.Actions
    ) -> dict[str, jax.Array]:
        """Compute the adapter-only objective without tracing the policy stack."""
        patch_cosine = self._predictive_patch_cosine(
            obs, actions, intervention=self.predictive_adapter_intervention
        )
        per_patch = 1.0 - patch_cosine
        batch_size = actions.shape[0]
        mask = (
            jnp.ones((batch_size,), dtype=per_patch.dtype)
            if obs.lmwm_target_mask is None
            else obs.lmwm_target_mask.astype(per_patch.dtype)
        )
        predictive_loss = jnp.sum(per_patch * mask[:, None]) / jnp.maximum(
            jnp.sum(mask) * per_patch.shape[-1], 1.0
        )
        main_loss = jnp.zeros(actions.shape[:-1], dtype=predictive_loss.dtype)
        return {
            "main_loss": main_loss,
            "main_weight": jnp.asarray(0.0, dtype=predictive_loss.dtype),
            "predictive_loss": predictive_loss,
            "predictive_weight": jnp.asarray(
                self.predictive_adapter_loss_weight, dtype=predictive_loss.dtype
            ),
            "predictive_cosine": 1.0 - predictive_loss,
        }

    def _embed_prefix_impl(
        self, obs: _model.Observation, *, want_lmwm_aux: bool
    ) -> tuple[at.Array, at.Array, at.Array, dict[str, at.Array]]:
        input_mask = []
        ar_mask = []
        tokens = []
        aux = {}
        base_image_tokens = None
        target_image_tokens = None
        # X-VLA style soft prompt: prepend per-domain learnable tokens.
        # Bidirectional (non-AR) — images/language can attend to them and vice versa.
        if (
            self.soft_prompt_num_domains > 0
            and self.soft_prompt_len > 0
            and obs.dataset_id is not None
        ):
            B = obs.dataset_id.shape[0]
            soft = self.soft_prompt_hub(obs.dataset_id)
            llm_width = self.soft_prompt_hub.features // self.soft_prompt_len
            # Cast to bfloat16 to match image/language token dtype downstream.
            soft = soft.astype(jnp.bfloat16).reshape(B, self.soft_prompt_len, llm_width)
            tokens.append(soft)
            input_mask.append(jnp.ones((B, self.soft_prompt_len), dtype=jnp.bool_))
            ar_mask += [False] * self.soft_prompt_len
        image_names = list(obs.images)
        image_values = [obs.images[name] for name in image_names]
        include_target = (
            (want_lmwm_aux or self.lmwm_spatial_condition == "privileged")
            and obs.lmwm_target_image is not None
        )
        vision_inputs = [*image_values, *([obs.lmwm_target_image] if include_target else [])]
        can_fuse = (
            self.fuse_vision_batch
            and len(vision_inputs) > 1
            and all(image.shape[1:] == vision_inputs[0].shape[1:] for image in vision_inputs)
        )
        if can_fuse:
            batch_sizes = [image.shape[0] for image in vision_inputs]
            fused_tokens, _ = self.PaliGemma.img(jnp.concatenate(vision_inputs, axis=0), train=False)
            offsets = [0]
            for batch_size in batch_sizes:
                offsets.append(offsets[-1] + batch_size)
            encoded_images = [
                fused_tokens[offsets[idx] : offsets[idx + 1]]
                for idx in range(len(image_values))
            ]
            if include_target:
                target_image_tokens = fused_tokens[offsets[-2] : offsets[-1]]
        else:
            encoded_images = [self.PaliGemma.img(image, train=False)[0] for image in image_values]
            if include_target:
                target_image_tokens, _ = self.PaliGemma.img(obs.lmwm_target_image, train=False)

        if self.lmwm_transition_condition == "learned":
            if self.lmwm_transition_tracker_type == "current_frame":
                encoded_by_name = dict(zip(image_names, encoded_images, strict=True))
                required_views = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                pooled_views = jnp.stack(
                    [
                        jnp.mean(encoded_by_name[name].astype(jnp.float32), axis=1)
                        for name in required_views
                    ],
                    axis=1,
                )
                # Track the encoder's live feature space without allowing the
                # tracker objective or routing path to steer that space.
                pooled_views = jax.lax.stop_gradient(pooled_views)
                current_logits, next_logits = self.lmwm_transition_tracker(pooled_views)
            else:
                if (
                    obs.lmwm_transition_history_images is None
                    or obs.lmwm_transition_history_state is None
                ):
                    raise ValueError(
                        "history_proprio tracker requires lmwm_transition_history_images and "
                        "lmwm_transition_history_state"
                    )
                history = obs.lmwm_transition_history_images
                batch_size, history_length = history.shape[:2]
                if history_length != 3:
                    raise ValueError(f"history_proprio tracker requires three frames, got {history_length}")
                history_tokens, _ = self.PaliGemma.img(
                    history.reshape(batch_size * history_length, *history.shape[2:]), train=False
                )
                history_features = jnp.mean(history_tokens.astype(jnp.float32), axis=1).reshape(
                    batch_size, history_length, -1
                )
                history_features = jax.lax.stop_gradient(history_features)
                current_logits, next_logits = self.lmwm_transition_tracker(
                    history_features, obs.lmwm_transition_history_state
                )
            aux["lmwm_transition_current_probs"] = jax.nn.softmax(current_logits, axis=-1)
            aux["lmwm_transition_next_probs"] = jax.nn.softmax(next_logits, axis=-1)
            if want_lmwm_aux and obs.lmwm_transition_current is not None and obs.lmwm_transition_next is not None:
                current_loss = -jnp.take_along_axis(
                    jax.nn.log_softmax(current_logits, axis=-1),
                    obs.lmwm_transition_current.astype(jnp.int32)[:, None],
                    axis=-1,
                )[:, 0]
                next_loss = -jnp.take_along_axis(
                    jax.nn.log_softmax(next_logits, axis=-1),
                    obs.lmwm_transition_next.astype(jnp.int32)[:, None],
                    axis=-1,
                )[:, 0]
                mask = (
                    jnp.ones_like(current_loss)
                    if obs.lmwm_transition_mask is None
                    else obs.lmwm_transition_mask.astype(current_loss.dtype)
                )
                aux["lmwm_transition_loss"] = jnp.sum((current_loss + next_loss) * mask) / jnp.maximum(
                    2.0 * jnp.sum(mask), 1.0
                )

        # embed images
        for name, image_tokens in zip(image_names, encoded_images, strict=True):
            if name == "base_0_rgb":
                base_image_tokens = image_tokens

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        if self.lmwm_local_dynamics:
            if base_image_tokens is None:
                raise ValueError("local dynamics requires base_0_rgb image tokens")
            aux["lmwm_local_current_feature"] = jax.lax.stop_gradient(
                jnp.mean(base_image_tokens.astype(jnp.float32), axis=1)
            )
            if target_image_tokens is not None:
                aux["lmwm_local_target_feature"] = jax.lax.stop_gradient(
                    jnp.mean(target_image_tokens.astype(jnp.float32), axis=1)
                )
                aux["lmwm_local_target_mask"] = obs.lmwm_target_mask

        if self.predictive_adapter_mode != "none":
            if base_image_tokens is None:
                raise ValueError("predictive adapter requires base_0_rgb image tokens")
            aux["predictive_current_tokens"] = jax.lax.stop_gradient(base_image_tokens)
            if target_image_tokens is not None:
                aux["predictive_target_tokens"] = jax.lax.stop_gradient(
                    _spatial_pool_tokens(
                        target_image_tokens.astype(jnp.float32),
                        self.predictive_action_adapter.grid_size,
                    )
                )
                aux["predictive_target_mask"] = obs.lmwm_target_mask

        if self.recurrence_adapter_mode != "none":
            if base_image_tokens is None:
                raise ValueError("recurrence adapter requires base_0_rgb image tokens")
            aux["recurrence_current_tokens"] = jax.lax.stop_gradient(base_image_tokens)

        if self.lmwm_spatial_condition != "none":
            if self.lmwm_spatial_condition == "current":
                spatial_source = base_image_tokens
                spatial_available = None
            elif self.lmwm_spatial_condition == "privileged":
                spatial_source = target_image_tokens
                spatial_available = obs.lmwm_target_mask
            else:
                spatial_source = None
                spatial_available = None
            spatial_tokens, spatial_stats = self.lmwm_spatial_adapter(
                spatial_source,
                batch_size=obs.state.shape[0],
                available=spatial_available,
            )
            tokens.append(spatial_tokens)
            input_mask.append(jnp.ones(spatial_tokens.shape[:2], dtype=jnp.bool_))
            ar_mask += [False] * spatial_tokens.shape[1]
            aux["lmwm_spatial_availability"] = spatial_stats["availability"]
            aux["lmwm_spatial_gate"] = spatial_stats["gate"]
            aux["lmwm_spatial_token_norm"] = spatial_stats["token_norm"]

        if self.lmwm_hint_dim > 0 and self.lmwm_hint_target == "prefix":
            if self.lmwm_live_hint and base_image_tokens is not None:
                cur = jnp.mean(base_image_tokens.astype(jnp.float32), axis=1)
                delta = self.lmwm_live_pred_out(nnx.swish(self.lmwm_live_pred_in(cur)))
                pred = cur + delta if self.lmwm_live_residual else delta
                if self.lmwm_live_intervention == "current":
                    pred = cur
                elif self.lmwm_live_intervention == "zero":
                    pred = jnp.zeros_like(pred)
                elif self.lmwm_live_intervention == "shuffle":
                    pred = jnp.roll(pred, shift=pred.shape[-1] // 3, axis=-1)
                elif self.lmwm_live_intervention == "other-task":
                    pred = jnp.broadcast_to(
                        jnp.asarray(_LMWM_LIVE_OTHER_TASK_FEATURE, dtype=pred.dtype), pred.shape
                    )
                hint = pred[:, None, :].astype(jnp.bfloat16)
                tokens.append(hint)
                input_mask.append(jnp.ones(hint.shape[:2], dtype=jnp.bool_))
                ar_mask += [False] * hint.shape[1]

                if target_image_tokens is not None:
                    tgt = jax.lax.stop_gradient(jnp.mean(target_image_tokens.astype(jnp.float32), axis=1))
                    per = jnp.mean(jnp.square(pred.astype(jnp.float32) - tgt), axis=-1)
                    if obs.lmwm_target_mask is not None:
                        mask = obs.lmwm_target_mask.astype(per.dtype)
                        aux["lmwm_loss"] = jnp.sum(per * mask) / jnp.maximum(jnp.sum(mask), 1.0)
                    else:
                        aux["lmwm_loss"] = jnp.mean(per)
            elif obs.lmwm_hint is not None:
                # Offline A1/A2 hint path.
                hint = self.lmwm_hint_proj(obs.lmwm_hint)  # (B, hint_len, llm_width)
                hint = hint.astype(jnp.bfloat16)
                tokens.append(hint)
                input_mask.append(jnp.ones(hint.shape[:2], dtype=jnp.bool_))
                ar_mask += [False] * hint.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, aux

    @at.typecheck
    def embed_suffix(
        self,
        obs: _model.Observation,
        noisy_actions: _model.Actions,
        timestep,
        transition_probs: tuple[jax.Array, jax.Array] | None = None,
        local_condition: jax.Array | None = None,
        predictive_condition: jax.Array | None = None,
        recurrence_condition: jax.Array | None = None,
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        # TAC 路径下是 per-token time_for_emb (b, s, emb); 非 TAC 是 (b, emb) 或 None.
        at.Float[at.Array, "b emb"] | at.Float[at.Array, "b s emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []

        # Track C action head conditioning (方案 A): prepend 1 domain token to suffix.
        # Only the action expert (not paligemma) sees this token — paligemma is
        # unaware of domain. The token forms its own ar group (ar_mask=True) so it
        # breaks from the prefix; subsequent state/action tokens follow their
        # original ar pattern (each starts a new group).
        if self.action_head_cond_num_domains > 0 and obs.dataset_id is not None:
            B = obs.dataset_id.shape[0]
            domain_token = self.action_head_cond_hub(obs.dataset_id)
            domain_token = domain_token.astype(jnp.bfloat16)[:, None, :]  # (B, 1, action_expert_width)
            tokens.append(domain_token)
            input_mask.append(jnp.ones((B, 1), dtype=jnp.bool_))
            # standalone group; action_expert tokens follow with their own ar pattern
            ar_mask += [True]

        # LMWM hint (suffix target): project offline hint → prepend token(s) to the
        # suffix. Only the action expert sees this (paligemma unaware) — conditioning
        # only modulates action denoising, not VLM language alignment. Mirrors the
        # action_head_cond domain token (each hint token forms its own ar group).
        if (
            self.lmwm_hint_dim > 0
            and self.lmwm_hint_target == "suffix"
            and obs.lmwm_hint is not None
        ):
            hint = self.lmwm_hint_proj(obs.lmwm_hint)  # (B, hint_len, action_expert_width)
            hint = hint.astype(jnp.bfloat16)
            tokens.append(hint)
            input_mask.append(jnp.ones(hint.shape[:2], dtype=jnp.bool_))
            ar_mask += [True] * hint.shape[1]

        if not self.pi05:
            # add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        if local_condition is not None:
            action_tokens = action_tokens + local_condition[:, None, :].astype(action_tokens.dtype)
        if predictive_condition is not None:
            if predictive_condition.shape != action_tokens.shape:
                raise ValueError(
                    "predictive action route must match action tokens: "
                    f"{predictive_condition.shape} != {action_tokens.shape}"
                )
            action_tokens = action_tokens + predictive_condition.astype(action_tokens.dtype)
        if recurrence_condition is not None:
            if recurrence_condition.shape != action_tokens.shape:
                raise ValueError(
                    "recurrence action route must match action tokens: "
                    f"{recurrence_condition.shape} != {action_tokens.shape}"
                )
            action_tokens = action_tokens + recurrence_condition.astype(action_tokens.dtype)
        if self.lmwm_transition_condition != "none":
            use_learned = self.lmwm_transition_condition == "learned" and self.lmwm_transition_intervention != "oracle"
            adapter_intervention = self.lmwm_transition_intervention
            if adapter_intervention == "oracle":
                adapter_intervention = "correct"
            if use_learned:
                if transition_probs is None:
                    raise ValueError("learned transition routing requires tracker probabilities")
                adapter_intervention = {
                    "predicted": "correct",
                    "within_task": "shuffle",
                }.get(adapter_intervention, adapter_intervention)
            transition, _ = self.lmwm_transition_adapter(
                obs.lmwm_transition_task,
                obs.lmwm_transition_current,
                obs.lmwm_transition_next,
                batch_size=obs.state.shape[0],
                available=(
                    obs.lmwm_transition_mask
                    if not use_learned
                    else (
                        None
                        if obs.lmwm_transition_task is None
                        else obs.lmwm_transition_task
                        < self.lmwm_transition_adapter.num_tasks
                    )
                ),
                intervention=adapter_intervention,
                current_probs=None if not use_learned else transition_probs[0],
                next_probs=None if not use_learned else transition_probs[1],
            )
            if use_learned:
                transition = self.lmwm_transition_dropout(
                    transition, deterministic=self.deterministic
                )
            action_tokens = action_tokens + transition[:, None, :].astype(action_tokens.dtype)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        # TAC support: timestep may be (b,) (standard, per-sample) or (b, ah) (per-token).
        # posemb_sincos requires 1D, so flatten and reshape back.
        time_in_ndim = timestep.ndim
        time_flat = timestep.reshape(-1) if time_in_ndim > 1 else timestep
        time_emb = posemb_sincos(time_flat, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if time_in_ndim > 1:
            time_emb = time_emb.reshape(*timestep.shape, -1)  # (b, ah, emb)
        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ):
        """Flow-matching loss. Returns per-sample-per-horizon tensor OR a dict when aux losses are enabled.

        Return formats:
          - No aux flags set: Float[*b, ah]  (unchanged historical behavior)
          - use_dct_loss=True: dict containing
              'main_loss' : Float[*b, ah]   flow-matching MSE
              'dct_loss'  : scalar          DCT-MSE
              'dct_weight': scalar          weight for summing
        """
        preprocess_rng, noise_rng, time_rng, delay_rng = jax.random.split(rng, 4)
        observation = _model.preprocess_observation(
            preprocess_rng, observation, train=train,
            augment_level=self.augment_level,
        )

        if self.predictive_adapter_mode == "offline":
            return self._compute_offline_predictive_loss(observation, actions)

        batch_shape = actions.shape[:-2]
        ah = actions.shape[-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001

        # TAC (Training-time Action Conditioning, paper 2512.05964 Algorithm 1).
        # Default tac_enabled=False → original per-sample flow matching (back-compat).
        if self.tac_enabled:
            # Sample delay per sample ∈ [0, tac_max_delay+1) inclusive of 0.
            delay = jax.random.randint(delay_rng, batch_shape, 0, self.tac_max_delay + 1)
            # Per-token mask: True where idx < delay (prefix region, gets clean GT action).
            token_idx = jnp.arange(ah)
            for _ in range(len(batch_shape)):
                token_idx = token_idx[None, ...]
            prefix_mask_tac = token_idx < delay[..., None]  # (*batch_shape, ah)
            postfix_mask_tac = jnp.logical_not(prefix_mask_tac)
            # Per-token time: prefix=0.0 (clean GT), postfix=sampled time (broadcast).
            # openpi convention: t=1=noise, t=0=clean data. x_t = t*noise + (1-t)*actions,
            # so prefix t=0 → x_t = actions, the "previously committed action" context that
            # the TAC paper (Algorithm 1) requires.
            # Earlier value 1.0 was a convention flip bug — fed pure noise as prefix, making
            # TAC training equivalent to "loss-mask prefix tokens" with no prefix-conditioning
            # signal (verified by diagnostic: TAC v7 P1=0.067 ≈ no-TAC baseline 0.026 worse,
            # 2026-05-27 chunk/noise diagnostic).
            time_per_token = jnp.where(prefix_mask_tac, 0.0, time[..., None])  # (*b, ah)
            time_expanded = time_per_token[..., None]  # (*b, ah, 1)
            time_for_emb = time_per_token  # passed to embed_suffix as (b, ah)
        else:
            time_expanded = time[..., None, None]
            time_for_emb = time

        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # one big forward pass of prefix + suffix at once
        prefix_tokens, prefix_mask, prefix_ar_mask, lmwm_aux = self._embed_prefix_impl(
            observation,
            want_lmwm_aux=(
                self.lmwm_live_hint and self.lmwm_live_loss_weight > 0
            )
            or self.lmwm_transition_condition == "learned"
            or self.lmwm_local_dynamics
            or self.predictive_adapter_mode != "none"
            or self.recurrence_adapter_mode != "none",
        )
        transition_probs = None
        if self.lmwm_transition_condition == "learned":
            transition_probs = (
                lmwm_aux["lmwm_transition_current_probs"],
                lmwm_aux["lmwm_transition_next_probs"],
            )
        local_condition = None
        if self.lmwm_local_dynamics:
            current_feature = lmwm_aux["lmwm_local_current_feature"]
            local_prediction, _ = self.lmwm_local_adapter(current_feature, actions)
            _, local_condition = self.lmwm_local_adapter(current_feature, x_t)
            target_feature = lmwm_aux.get("lmwm_local_target_feature")
            if target_feature is not None:
                per = jnp.mean(jnp.square(local_prediction - target_feature), axis=-1)
                mask = lmwm_aux.get("lmwm_local_target_mask")
                if mask is None:
                    lmwm_aux["lmwm_local_loss"] = jnp.mean(per)
                else:
                    mask = mask.astype(per.dtype)
                    lmwm_aux["lmwm_local_loss"] = jnp.sum(per * mask) / jnp.maximum(jnp.sum(mask), 1.0)
        predictive_condition = None
        if self.predictive_adapter_mode != "none":
            current_tokens = lmwm_aux["predictive_current_tokens"]
            prediction_intervention = self.predictive_adapter_intervention
            if prediction_intervention == "zero_gate":
                prediction_intervention = "normal"
            prediction, _ = self.predictive_action_adapter(
                current_tokens,
                actions,
                intervention=prediction_intervention,
            )
            _, predictive_condition = self.predictive_action_adapter(
                current_tokens,
                x_t,
                intervention=self.predictive_adapter_intervention,
            )
            target_tokens = lmwm_aux.get("predictive_target_tokens")
            if target_tokens is not None:
                pred_unit = prediction / jnp.maximum(
                    jnp.linalg.norm(prediction, axis=-1, keepdims=True), 1e-6
                )
                target_unit = target_tokens / jnp.maximum(
                    jnp.linalg.norm(target_tokens, axis=-1, keepdims=True), 1e-6
                )
                per = 1.0 - jnp.sum(pred_unit * target_unit, axis=-1)
                mask = lmwm_aux.get("predictive_target_mask")
                if mask is None:
                    lmwm_aux["predictive_loss"] = jnp.mean(per)
                else:
                    mask = mask.astype(per.dtype)
                    lmwm_aux["predictive_loss"] = jnp.sum(per * mask[:, None]) / jnp.maximum(
                        jnp.sum(mask) * per.shape[-1], 1.0
                    )
                lmwm_aux["predictive_cosine"] = 1.0 - lmwm_aux["predictive_loss"]
        recurrence_condition = None
        if self.recurrence_adapter_mode != "none":
            current_tokens = lmwm_aux["recurrence_current_tokens"]
            label_intervention = self.recurrence_adapter_intervention
            if label_intervention == "zero_gate":
                label_intervention = "normal"
            recurrence_predictions, _ = self.recurrence_action_adapter(
                current_tokens,
                actions,
                intervention=label_intervention,
            )
            _, recurrence_condition = self.recurrence_action_adapter(
                current_tokens,
                x_t,
                intervention=self.recurrence_adapter_intervention,
            )
            if (
                observation.crave_progress_change is not None
                and observation.crave_target_density is not None
                and observation.crave_boundary_crossing is not None
            ):
                bins = self.recurrence_action_adapter.bins

                def binned_loss(logits, values, lower, upper):
                    scaled = (jnp.clip(values, lower, upper) - lower) / (upper - lower)
                    indices = jnp.minimum(
                        jnp.floor(scaled * bins).astype(jnp.int32), bins - 1
                    )
                    return -jnp.take_along_axis(
                        jax.nn.log_softmax(logits, axis=-1), indices[:, None], axis=-1
                    )[:, 0]

                progress_loss = binned_loss(
                    recurrence_predictions["progress_logits"],
                    observation.crave_progress_change.astype(jnp.float32),
                    -1.0,
                    1.0,
                )
                density_loss = binned_loss(
                    recurrence_predictions["density_logits"],
                    observation.crave_target_density.astype(jnp.float32),
                    0.0,
                    1.0,
                )
                boundary_index = observation.crave_boundary_crossing.astype(jnp.int32)
                boundary_loss = -jnp.take_along_axis(
                    jax.nn.log_softmax(
                        recurrence_predictions["boundary_logits"], axis=-1
                    ),
                    boundary_index[:, None],
                    axis=-1,
                )[:, 0]
                mask = (
                    jnp.ones_like(progress_loss)
                    if observation.crave_target_mask is None
                    else observation.crave_target_mask.astype(progress_loss.dtype)
                )
                per_sample = (progress_loss + density_loss + boundary_loss) / 3.0
                lmwm_aux["recurrence_loss"] = jnp.sum(per_sample * mask) / jnp.maximum(
                    jnp.sum(mask), 1.0
                )
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
            observation,
            x_t,
            time_for_emb,
            transition_probs,
            local_condition,
            predictive_condition,
            recurrence_condition,
        )
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        per_token_loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)  # (*b, ah)

        # AWBC ② loss-weighting: scale each frame's loss by its class-derived weight.
        # w=0 (class2 preintv) → zero gradient; w=2 (class1 intv/grasp) → double. base/robot w=1.
        if self.awbc_loss_weight and getattr(observation, "sample_weight", None) is not None:
            w = observation.sample_weight.astype(per_token_loss.dtype)  # (*b,)
            per_token_loss = per_token_loss * w[..., None]

        if self.tac_enabled:
            # Mask: only postfix tokens contribute. Per-sample mean over kept tokens.
            mask_f = postfix_mask_tac.astype(per_token_loss.dtype)
            denom = jnp.maximum(jnp.sum(mask_f, axis=-1, keepdims=True), 1.0)
            main_loss = (per_token_loss * mask_f).sum(axis=-1, keepdims=True) / denom
            # Return same shape (*b, ah) by broadcasting (with zeros for prefix tokens).
            main_loss = jnp.broadcast_to(main_loss, per_token_loss.shape) * mask_f
        else:
            main_loss = per_token_loss

        if not self.use_dct_loss and not lmwm_aux:
            return main_loss

        out = {"main_loss": main_loss}
        if self.predictive_adapter_mode == "offline":
            out["main_weight"] = jnp.asarray(0.0, dtype=main_loss.dtype)
        if "lmwm_loss" in lmwm_aux:
            out["lmwm_loss"] = lmwm_aux["lmwm_loss"]
            out["lmwm_weight"] = jnp.asarray(self.lmwm_live_loss_weight, dtype=main_loss.dtype)
        if "lmwm_transition_loss" in lmwm_aux:
            out["lmwm_loss"] = lmwm_aux["lmwm_transition_loss"]
            out["lmwm_weight"] = jnp.asarray(
                self.lmwm_transition_auxiliary_loss_weight, dtype=main_loss.dtype
            )
        if "lmwm_local_loss" in lmwm_aux:
            out["lmwm_local_loss"] = lmwm_aux["lmwm_local_loss"]
            out["lmwm_local_weight"] = jnp.asarray(
                self.lmwm_local_loss_weight, dtype=main_loss.dtype
            )
        if "predictive_loss" in lmwm_aux:
            out["predictive_loss"] = lmwm_aux["predictive_loss"]
            out["predictive_weight"] = jnp.asarray(
                self.predictive_adapter_loss_weight, dtype=main_loss.dtype
            )
            out["predictive_cosine"] = lmwm_aux["predictive_cosine"]
        if "recurrence_loss" in lmwm_aux:
            out["recurrence_loss"] = lmwm_aux["recurrence_loss"]
            out["recurrence_weight"] = jnp.asarray(
                self.recurrence_adapter_loss_weight, dtype=main_loss.dtype
            )

        if self.use_dct_loss:
            # DCT-II on time axis; weight low/high frequencies differently to
            # penalise high-frequency jitter more than low-frequency structure.
            freqs_v = _dct2_last_time_axis(v_t)
            freqs_u = _dct2_last_time_axis(u_t)
            T = v_t.shape[-2]
            denom = max(T - 1, 1)
            freq_idx = jnp.arange(T, dtype=v_t.dtype) / denom
            weights = (
                self._dct_low_freq_weight * (1.0 - freq_idx)
                + self._dct_high_freq_weight * freq_idx
            )
            dct = jnp.mean(weights[None, :, None] * jnp.square(freqs_v - freqs_u))
            out["dct_loss"] = dct
            out["dct_weight"] = jnp.asarray(self._dct_loss_weight, dtype=dct.dtype)

        return out

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
        tac_prefix: at.Float[at.Array, "b ah ad"] | None = None,
        tac_delay: int | at.Int[at.Array, ""] = 0,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # TAC faithful conditioning (eval only): clamp prefix [0:tac_delay] to clean
        # normalized GT at per-token time=0, matching compute_loss's training setup,
        # and only denoise the postfix. Off by default → non-TAC path unchanged.
        tac_pos_mask = None
        if tac_prefix is not None:
            tac_pos_mask = (jnp.arange(self.action_horizon) < tac_delay)[None, :, None]  # (1, ah, 1)
            noise = jnp.where(tac_pos_mask, tac_prefix, noise)

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask, lmwm_aux = self._embed_prefix_impl(
            observation, want_lmwm_aux=False
        )
        transition_probs = None
        if self.lmwm_transition_condition == "learned":
            transition_probs = (
                lmwm_aux["lmwm_transition_current_probs"],
                lmwm_aux["lmwm_transition_next_probs"],
            )
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
            if tac_prefix is not None:
                x_t = jnp.where(tac_pos_mask, tac_prefix, x_t)
                time_emb_arg = jnp.where(tac_pos_mask[..., 0], 0.0, time)  # (1, ah) per-token
            else:
                time_emb_arg = jnp.broadcast_to(time, batch_size)
            local_condition = None
            if self.lmwm_local_dynamics:
                _, local_condition = self.lmwm_local_adapter(
                    lmwm_aux["lmwm_local_current_feature"], x_t
                )
            predictive_condition = None
            if self.predictive_adapter_mode != "none":
                _, predictive_condition = self.predictive_action_adapter(
                    lmwm_aux["predictive_current_tokens"],
                    x_t,
                    intervention=self.predictive_adapter_intervention,
                )
            recurrence_condition = None
            if self.recurrence_adapter_mode != "none":
                _, recurrence_condition = self.recurrence_action_adapter(
                    lmwm_aux["recurrence_current_tokens"],
                    x_t,
                    intervention=self.recurrence_adapter_intervention,
                )
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation,
                x_t,
                time_emb_arg,
                transition_probs,
                local_condition,
                predictive_condition,
                recurrence_condition,
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0
