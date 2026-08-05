# FastWAM real-robot deployment

This directory is the canonical home for FastWAM-specific deployment code.
The similarly named files under `start_scripts/kai/` are compatibility entry
points only; shared ROS launch and policy-control code remains under `ros2_ws/`.

## Isolated deployment

- `start_autonomy_isolated.sh`: observe-only-by-default stack launcher.
- `start_a1.sh`: complete A1 artifact/configuration preset; accepts the same runtime flags.
- `run_a1_execute.sh`: direct-execute A1 entry point; automatically starts or reuses the server.
- `run_a1_full_chunk_execute.sh`: diagnostic open-loop mode; executes all 48
  actions at 30 Hz before requesting the next chunk, with temporal post-processing disabled.
- `isolated_control.py`: acknowledged execution on/off control.
- `preflight.py`: read-only camera, gripper, and start-pose checks.
- `refs/`: task-specific start-pose reference files.
- `archive/`: historical deployment-script backups; never invoked at runtime.

Run `start_autonomy_isolated.sh --check-only` before starting the stack. Never
enable execution until preflight passes and an operator is at the emergency stop.

For repeated trials, add `--keep-server` to the first launch. After stopping the
ROS stack, use `--reuse-server` on later launches; the launcher verifies that
the running process uses the exact requested weights, stats, and T5 cache. A new
controller connection resets gripper bootstrap state without reloading the model.

The first A1 validation stage uses `publish_rate=30 Hz` and
`speed_factor=0.5`, preserving a conservative 2x slowdown while matching the
training action clock. Later stages must be selected explicitly after reviewing
the preceding trial.
