from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).with_name("extract_mt3_tracker_features.py")
SPEC = importlib.util.spec_from_file_location("extract_mt3_tracker_features", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_row_shards_are_disjoint_and_preserve_frozen_split_membership():
    episode = np.asarray([10, 10, 11, 11, 12, 99])
    split = {"train_episodes": [10, 12], "val_episodes": [11]}
    rows0, split0 = MODULE.select_rows(episode, split, shard_index=0, num_shards=2)
    rows1, split1 = MODULE.select_rows(episode, split, shard_index=1, num_shards=2)
    assert not set(rows0).intersection(rows1)
    assert sorted(np.concatenate([rows0, rows1]).tolist()) == [0, 1, 2, 3, 4]
    labels = dict(zip(np.concatenate([rows0, rows1]), np.concatenate([split0, split1]), strict=True))
    assert labels == {0: 0, 1: 0, 2: 1, 3: 1, 4: 0}
    shard_by_episode = {}
    for shard_index, rows in enumerate((rows0, rows1)):
        for episode_id in episode[rows]:
            assert shard_by_episode.setdefault(int(episode_id), shard_index) == shard_index


def test_episode_sharding_balances_row_count_deterministically():
    episode = np.repeat(np.arange(8), [10, 9, 8, 7, 6, 5, 4, 3])
    split = {"train_episodes": list(range(6)), "val_episodes": [6, 7]}
    shards = [
        MODULE.select_rows(episode, split, shard_index=index, num_shards=3)[0]
        for index in range(3)
    ]
    assert sorted(np.concatenate(shards).tolist()) == list(range(len(episode)))
    assert max(map(len, shards)) - min(map(len, shards)) <= 10


def test_history_ids_are_sample_major_and_clamped_at_episode_start():
    assert MODULE.history_sample_ids([(7, 10), (8, 20)]) == [
        (7, 0), (7, 3), (7, 10), (8, 5), (8, 13), (8, 20)
    ]


def test_pooled_features_unpack_current_view_blocks_and_sample_major_history():
    pooled = np.arange(12 * 2, dtype=np.float32).reshape(12, 2)
    current, history = MODULE.unpack_pooled(pooled, batch_size=2)
    assert current.shape == (2, 3, 2)
    assert history.shape == (2, 3, 2)
    np.testing.assert_array_equal(current[0], pooled[[0, 2, 4]])
    np.testing.assert_array_equal(current[1], pooled[[1, 3, 5]])
    np.testing.assert_array_equal(history[0], pooled[[6, 7, 8]])
    np.testing.assert_array_equal(history[1], pooled[[9, 10, 11]])


def test_history_tensor_cache_reuses_values_and_evicts_oldest():
    class Dataset:
        def __init__(self):
            self.calls = []

        def __getitem__(self, index):
            self.calls.append(index)
            return {
                "image": {"base_0_rgb": np.full((2, 2, 3), index, dtype=np.uint8)},
                "state": np.arange(32, dtype=np.float32) + index,
            }

    dataset = Dataset()
    cache = MODULE.HistoryTensorCache(dataset, {(1, 0): 10, (1, 1): 11, (1, 2): 12}, 2)
    image, state = cache.get((1, 0))
    assert image[0, 0, 0] == 10 and state.shape == (14,)
    cache.get((1, 0))
    cache.get((1, 1))
    cache.get((1, 2))
    cache.get((1, 0))
    assert dataset.calls == [10, 11, 12, 10]
    assert cache.hits == 1
    assert cache.misses == 4
