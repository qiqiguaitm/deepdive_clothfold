import flax.nnx as nnx
import jax.numpy as jnp

from openpi.shared import nnx_utils


class Scale(nnx.Module):
    def __init__(self):
        self.weight = nnx.Param(jnp.asarray(2.0))

    def apply(self, value):
        return self.weight[...] * value


def test_module_jit_exposes_lowering_for_frozen_state_cost_analysis() -> None:
    wrapped = nnx_utils.module_jit(Scale().apply)
    assert float(wrapped(jnp.asarray(3.0))) == 6.0
    cost = (
        wrapped.lower(jnp.ones((16,), dtype=jnp.float32))
        .compile()
        .cost_analysis()
    )
    assert float(cost["flops"]) > 0.0
