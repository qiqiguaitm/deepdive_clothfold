import numpy as np

from examples.Robotwin.eval_files.model2robotwin_openpi import OpenpiRobotwinModelClient, build_robotwin_example


def test_transition_metadata_reaches_openpi_observation():
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    observation = {
        "observation": {
            "head_camera": {"rgb": image},
            "left_camera": {"rgb": image},
            "right_camera": {"rgb": image},
        },
        "joint_action": {"vector": np.zeros((14,), dtype=np.float32)},
        "lmwm_transition_task": np.asarray(2, dtype=np.int32),
        "lmwm_transition_current": np.asarray(3, dtype=np.int32),
        "lmwm_transition_next": np.asarray(4, dtype=np.int32),
        "lmwm_transition_mask": np.asarray(True),
    }

    example = build_robotwin_example("stack blocks", observation)
    client = object.__new__(OpenpiRobotwinModelClient)
    client.hint_computer = None
    client.hint_encoder = None
    openpi_observation = client._build_openpi_obs(example)

    assert int(openpi_observation["lmwm_transition_task"]) == 2
    assert int(openpi_observation["lmwm_transition_current"]) == 3
    assert int(openpi_observation["lmwm_transition_next"]) == 4
    assert bool(openpi_observation["lmwm_transition_mask"])


def test_transition_history_reaches_openpi_as_chw_images():
    image = np.zeros((8, 10, 3), dtype=np.uint8)
    observation = {
        "observation": {
            "head_camera": {"rgb": image},
            "left_camera": {"rgb": image},
            "right_camera": {"rgb": image},
        },
        "joint_action": {"vector": np.zeros((14,), dtype=np.float32)},
        "lmwm_transition_history_images": np.stack([image, image, image]),
        "lmwm_transition_history_state": np.zeros((3, 14), dtype=np.float32),
    }

    example = build_robotwin_example("stack blocks", observation)
    client = object.__new__(OpenpiRobotwinModelClient)
    client.hint_computer = None
    client.hint_encoder = None
    openpi_observation = client._build_openpi_obs(example)

    assert openpi_observation["lmwm_transition_history_images"].shape == (3, 3, 8, 10)
    assert openpi_observation["lmwm_transition_history_state"].shape == (3, 14)
