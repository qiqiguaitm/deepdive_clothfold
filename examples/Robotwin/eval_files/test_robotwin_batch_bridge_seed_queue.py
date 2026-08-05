from examples.Robotwin.eval_files.robotwin_batch_bridge import _should_issue_fresh_seed


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
