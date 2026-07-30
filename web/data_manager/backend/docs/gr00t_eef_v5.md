# KAI0 v5 GR00T EEF data contract

## Scope

The interactive data-manager recorder writes v5 episodes with bimanual
end-effector state and action fields compatible with the GR00T N1.7
LeRobot-v2 convention. Existing DAgger and autonomy writers remain 14-D unless
they explicitly set `KAI0_RECORD_EEF=1`.

## Coordinate frames and tool point

- `left_eef_9d` is expressed in the left Piper arm's own `base_link`.
- `right_eef_9d` is expressed in the right Piper arm's own `base_link`.
- The two poses are therefore not in a shared world frame.
- The tool point is the Piper DH `link6` origin. In the deployed URDF,
  `joint6_to_gripper_base` is fixed with zero translation and rotation, so this
  is also the `gripper_base` origin.
- Translation uses metres.

## Parquet layout

`observation.state` is an absolute 32-D vector:

| Slice | GR00T modality | Meaning |
| --- | --- | --- |
| `[0:6]` | `left_joint_position` | Left arm joints, radians |
| `[6:7]` | `left_gripper_position` | Left gripper opening, metres |
| `[7:13]` | `right_joint_position` | Right arm joints, radians |
| `[13:14]` | `right_gripper_position` | Right gripper opening, metres |
| `[14:23]` | `left_eef_9d` | Left absolute XYZ + rotation 6D |
| `[23:32]` | `right_eef_9d` | Right absolute XYZ + rotation 6D |

`action` has the same slicing. Its first 14 dimensions retain the existing
KAI0 action convention. Each EEF action is relative to the next kept frame:

```text
delta_xyz = xyz[t+1] - xyz[t]
delta_R   = R[t+1] @ transpose(R[t])
R[t+1]    = delta_R @ R[t]
```

Relative actions are calculated after front/tail trimming. The last kept
frame uses zero translation and identity rotation.

Rotation 6D is the first two columns of a 3x3 rotation matrix, flattened as:

```text
[r00, r01, r10, r11, r20, r21]
```

`meta/modality.json` maps these slices and the four video streams for the
GR00T loader.

## Kinematics

EEF state is derived from the measured slave-arm joint state with the Piper
SDK DH parameters using the 2-degree joint-2/joint-3 offset mode
(`dh_is_offset=0x01`). It does not use the leader-arm pose or a camera/world
calibration.

## Versioning and compatibility

Importing `app.recorder` enables `KAI0_RECORD_EEF=1` and selects:

```text
KAI0_DATASET_VERSION=v5
KAI0_DATE_SUFFIX=-v5
```

Thus new interactive captures land below `.../<subset>/v5/<date>-v5/` and
cannot be mixed accidentally with v4 14-D episodes. Setting
`KAI0_RECORD_EEF=0` before starting the backend restores the legacy schema.

## Verification evidence

Run:

```bash
cd web/data_manager/backend
.venv/bin/python -m unittest discover -s tests -p 'test_eef_kinematics.py' -v
.venv/bin/python -m compileall -q app tests
```

The tests cover the Piper zero-joint reference pose, rotation-6D round trip,
independent per-arm base frames, relative-action reconstruction, terminal
action identity, GR00T modality slices, 32-D feature metadata, and an actual
Parquet write/read round trip.
