from types import SimpleNamespace

import torch

from starVLA.model.framework.latent_world.batch_builder import (
    LatentWorldPolicyInferBatchBuilder,
)
from starVLA.model.framework.latent_world.runtime.runner import LatentWorldPolicyRunner


class _BatchBuilder:
    def __init__(self) -> None:
        self.examples = None

    def build_infer_batch(self, examples):
        self.examples = examples
        return {"action_hz": torch.tensor([30.0] * len(examples))}


class _Backend:
    def __init__(self) -> None:
        self.flow = SimpleNamespace(config=SimpleNamespace(horizon_sec=1.2))
        self.contexts = None

    def predict_action(self, *, batch, temporal_grounding_contexts, **kwargs):
        del kwargs
        self.contexts = temporal_grounding_contexts
        return torch.zeros((len(batch["action_hz"]), 36, 14))


def test_temporal_grounding_context_is_allowed_and_forwarded() -> None:
    context = {
        "task": "stack_blocks_two",
        "eval_seed": 0,
        "scene_seed": 123,
        "episode_id": 4,
        "query_index": 2,
    }
    example = {"temporal_grounding_context": context}
    builder = _BatchBuilder()
    backend = _Backend()

    assert "temporal_grounding_context" in LatentWorldPolicyInferBatchBuilder._ALLOWED_INFER_KEYS

    output = LatentWorldPolicyRunner(
        policy_backend=backend,
        infer_batch_builder=builder,
    ).infer_step([example])

    assert builder.examples == [example]
    assert backend.contexts == [context]
    assert output["normalized_actions"].shape == (1, 36, 14)
