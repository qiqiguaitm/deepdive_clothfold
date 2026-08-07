from examples.Robotwin.eval_files.robotwin_batch_bridge import (
    _should_issue_fresh_seed,
    reset_model_for_episode,
)


def test_fixed_seed_queue_does_not_over_issue_at_final_inflight_episode() -> None:
    assert not _should_issue_fresh_seed(
        fixed_seed_mode=True,
        remaining_fixed_seeds=0,
        completed_episodes=49,
        outstanding_episodes=0,
        test_num=50,
    )


def test_fixed_seed_queue_issues_each_remaining_seed() -> None:
    assert _should_issue_fresh_seed(
        fixed_seed_mode=True,
        remaining_fixed_seeds=1,
        completed_episodes=48,
        outstanding_episodes=1,
        test_num=50,
    )


def test_unfixed_seed_mode_uses_episode_accounting() -> None:
    assert _should_issue_fresh_seed(
        fixed_seed_mode=False,
        remaining_fixed_seeds=0,
        completed_episodes=48,
        outstanding_episodes=1,
        test_num=50,
    )
    assert not _should_issue_fresh_seed(
        fixed_seed_mode=False,
        remaining_fixed_seeds=0,
        completed_episodes=49,
        outstanding_episodes=1,
        test_num=50,
    )


def test_reset_model_for_episode_preserves_legacy_backend_signature() -> None:
    class Legacy:
        def __init__(self):
            self.slot_id = None

        def reset(self, slot_id=0):
            self.slot_id = slot_id

    model = Legacy()
    reset_model_for_episode(model, slot_id=2, episode_id=3, scene_seed=100007, eval_seed=1)
    assert model.slot_id == 2


def test_reset_model_for_episode_supplies_temporal_context_when_supported() -> None:
    class Temporal:
        def reset(self, slot_id=0, *, episode_id=None, scene_seed=None, eval_seed=None):
            self.values = (slot_id, episode_id, scene_seed, eval_seed)

    model = Temporal()
    reset_model_for_episode(model, slot_id=2, episode_id=3, scene_seed=100007, eval_seed=1)
    assert model.values == (2, 3, 100007, 1)
