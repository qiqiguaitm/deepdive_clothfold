import json
import math
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from app.eef_kinematics import (
    ACTION_EEF_LEFT,
    ACTION_EEF_RIGHT,
    STATE_DIM,
    append_absolute_eef,
    apply_relative_eef_actions,
    matrix_to_rotation_6d,
    piper_fk_matrix,
    rotation_6d_to_matrix,
    write_modality_json,
)
from app.dataset_writer import EpisodeWriter, features_block


class EefKinematicsTest(unittest.TestCase):
    def setUp(self):
        self._old_record_eef = os.environ.get("KAI0_RECORD_EEF")
        os.environ["KAI0_RECORD_EEF"] = "1"

    def tearDown(self):
        if self._old_record_eef is None:
            os.environ.pop("KAI0_RECORD_EEF", None)
        else:
            os.environ["KAI0_RECORD_EEF"] = self._old_record_eef

    def test_zero_joint_fk_matches_piper_sdk_reference(self):
        transform = piper_fk_matrix([0.0] * 6)
        np.testing.assert_allclose(
            transform[:3, 3],
            np.array([0.056128, 0.0, 0.213266]),
            atol=1e-6,
        )

    def test_rotation_6d_round_trip(self):
        angle = math.pi / 3
        rotation = np.array(
            [
                [math.cos(angle), -math.sin(angle), 0.0],
                [math.sin(angle), math.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        decoded = rotation_6d_to_matrix(matrix_to_rotation_6d(rotation))
        np.testing.assert_allclose(decoded, rotation, atol=1e-7)

    def test_each_arm_uses_its_own_base_frame(self):
        state = append_absolute_eef([0.0] * 14)
        self.assertEqual(len(state), STATE_DIM)
        np.testing.assert_allclose(state[14:23], state[23:32], atol=1e-7)

    def test_relative_actions_reconstruct_next_kept_pose(self):
        first = append_absolute_eef([0.0] * 14)
        moved = [0.0] * 14
        moved[0], moved[7] = 0.1, -0.1
        second = append_absolute_eef(moved)
        actions = apply_relative_eef_actions(
            [first, second], [[0.0] * 14, [0.0] * 14]
        )

        for eef_slice in (ACTION_EEF_LEFT, ACTION_EEF_RIGHT):
            delta = np.asarray(actions[0][eef_slice])
            current = np.asarray(first[eef_slice])
            following = np.asarray(second[eef_slice])
            np.testing.assert_allclose(
                current[:3] + delta[:3], following[:3], atol=1e-7
            )
            reconstructed = (
                rotation_6d_to_matrix(delta[3:])
                @ rotation_6d_to_matrix(current[3:])
            )
            np.testing.assert_allclose(
                reconstructed,
                rotation_6d_to_matrix(following[3:]),
                atol=1e-7,
            )

        np.testing.assert_allclose(actions[-1][ACTION_EEF_LEFT][:3], 0.0)
        np.testing.assert_allclose(
            rotation_6d_to_matrix(actions[-1][ACTION_EEF_LEFT][3:]),
            np.eye(3),
            atol=1e-7,
        )

    def test_modality_json_has_gr00t_slices(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_modality_json(Path(tmp))
            modality = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(
            modality["state"]["left_eef_9d"], {"start": 14, "end": 23}
        )
        self.assertEqual(
            modality["state"]["right_eef_9d"], {"start": 23, "end": 32}
        )
        self.assertEqual(modality["action"], modality["state"])
        self.assertEqual(
            modality["video"]["top_head"]["original_key"],
            "observation.images.top_head",
        )

    def test_lerobot_features_advertise_32_dimensions(self):
        features = features_block()
        self.assertEqual(features["observation.state"]["shape"], [32])
        self.assertEqual(features["action"]["shape"], [32])

    def test_parquet_writes_absolute_state_and_relative_action(self):
        first = append_absolute_eef([0.0] * 14)
        moved = [0.0] * 14
        moved[0] = 0.1
        second = append_absolute_eef(moved)
        with tempfile.TemporaryDirectory() as tmp:
            writer = EpisodeWriter.__new__(EpisodeWriter)
            writer.ep = 3
            writer.pq_path = Path(tmp) / "episode_000003.parquet"
            writer._rows_state = [first, second]
            writer._rows_action = [[0.0] * 14, [0.0] * 14]
            writer._rows_intervention = [-1, -1]
            writer._frame_class = False
            writer._write_parquet()
            table = pq.read_table(writer.pq_path)

        self.assertEqual(len(table["observation.state"][0].as_py()), 32)
        self.assertEqual(len(table["action"][0].as_py()), 32)
        action = np.asarray(table["action"][0].as_py())
        np.testing.assert_allclose(
            first[14:17] + action[14:17],
            second[14:17],
            atol=1e-7,
        )


if __name__ == "__main__":
    unittest.main()
