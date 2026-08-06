import numpy as np
import pytest

from openpi.policies.libero_policy import RobotwinTargetImageLookupTransform
from openpi.policies.libero_policy import RobotwinCraveTargetLookupTransform
from openpi.policies.libero_policy import RobotwinTransitionLookupTransform


def test_fixed_horizon_target_lookup_avoids_materializing_pair_index(tmp_path):
    pairs = tmp_path / "pairs.npz"
    np.savez(
        pairs,
        cur_ep=np.arange(1000, dtype=np.int32),
        cur_fi=np.arange(1000, dtype=np.int32),
        tgt_fi=np.arange(1000, dtype=np.int32) + 50,
        horizon_frames=np.asarray(50, dtype=np.int32),
    )
    cache = tmp_path / "cache"
    cache.mkdir()
    transform = RobotwinTargetImageLookupTransform(
        pairs_path=str(pairs), frame_cache_root=str(cache)
    )
    assert transform._horizon_frames == 50
    assert transform._index == {}


def test_transition_lookup_uses_mined_ids_and_explicit_null(tmp_path):
    pairs = tmp_path / "pairs.npz"
    np.savez(
        pairs,
        cur_ep=np.asarray([550, 550]),
        cur_fi=np.asarray([0, 1]),
        cur_ms=np.asarray([0, 1]),
        pair_task=np.asarray([2, 2]),
    )
    transform = RobotwinTransitionLookupTransform(str(pairs), num_tasks=6, num_stages=10)

    covered = transform({"episode_index": np.asarray(550), "frame_index": np.asarray(1)})
    assert int(covered["lmwm_transition_task"]) == 2
    assert int(covered["lmwm_transition_current"]) == 1
    assert int(covered["lmwm_transition_next"]) == 2
    assert bool(covered["lmwm_transition_mask"])

    missing = transform({"episode_index": np.asarray(551), "frame_index": np.asarray(0)})
    assert int(missing["lmwm_transition_task"]) == 6
    assert int(missing["lmwm_transition_current"]) == 10
    assert int(missing["lmwm_transition_next"]) == 10
    assert not bool(missing["lmwm_transition_mask"])


def test_transition_lookup_recovers_task_for_unlabelled_frames(tmp_path):
    pairs = tmp_path / "pairs.npz"
    np.savez(
        pairs,
        cur_ep=np.asarray([550, 1099]),
        cur_fi=np.asarray([0, 0]),
        cur_ms=np.asarray([0, 0]),
        pair_task=np.asarray([0, 0]),
    )
    transform = RobotwinTransitionLookupTransform(
        str(pairs),
        num_tasks=1,
        num_stages=10,
    )
    missing_label = transform(
        {
            "episode_index": np.asarray(700),
            "frame_index": np.asarray(7),
        }
    )
    assert int(missing_label["lmwm_transition_task"]) == 0
    assert not bool(missing_label["lmwm_transition_mask"])


def test_transition_lookup_rejects_vocab_overflow(tmp_path):
    pairs = tmp_path / "pairs.npz"
    np.savez(
        pairs,
        cur_ep=np.asarray([1]),
        cur_fi=np.asarray([0]),
        cur_ms=np.asarray([10]),
        pair_task=np.asarray([0]),
    )
    with pytest.raises(ValueError, match="stage IDs"):
        RobotwinTransitionLookupTransform(str(pairs), num_tasks=6, num_stages=10)


def test_crave_target_lookup_preserves_missing_rows_with_mask(tmp_path):
    targets = tmp_path / "targets.npz"
    np.savez(
        targets,
        cur_ep=np.asarray([12, 12]),
        cur_fi=np.asarray([3, 4]),
        progress_change=np.asarray([0.25, -0.1], dtype=np.float32),
        target_recurrence_density=np.asarray([0.8, 0.3], dtype=np.float32),
        phase_boundary_crossing=np.asarray([True, False]),
    )
    transform = RobotwinCraveTargetLookupTransform(str(targets))

    covered = transform({"episode_index": np.asarray(12), "frame_index": np.asarray(3)})
    assert float(covered["crave_progress_change"]) == pytest.approx(0.25)
    assert float(covered["crave_target_density"]) == pytest.approx(0.8)
    assert bool(covered["crave_boundary_crossing"])
    assert bool(covered["crave_target_mask"])

    missing = transform({"episode_index": np.asarray(12), "frame_index": np.asarray(9)})
    assert float(missing["crave_progress_change"]) == 0.0
    assert float(missing["crave_target_density"]) == 0.0
    assert not bool(missing["crave_boundary_crossing"])
    assert not bool(missing["crave_target_mask"])


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"cur_ep": np.asarray([1, 1]), "cur_fi": np.asarray([2, 2])}, "duplicate"),
        ({"progress_change": np.asarray([1.1, 0.0])}, "progress-change"),
        ({"target_recurrence_density": np.asarray([-0.1, 0.5])}, "recurrence-density"),
    ],
)
def test_crave_target_lookup_rejects_invalid_artifacts(tmp_path, updates, message):
    values = {
        "cur_ep": np.asarray([1, 1]),
        "cur_fi": np.asarray([2, 3]),
        "progress_change": np.asarray([0.1, 0.0]),
        "target_recurrence_density": np.asarray([0.2, 0.5]),
        "phase_boundary_crossing": np.asarray([False, True]),
    }
    values.update(updates)
    targets = tmp_path / "targets.npz"
    np.savez(targets, **values)
    with pytest.raises(ValueError, match=message):
        RobotwinCraveTargetLookupTransform(str(targets))
