# dagger_manager — Web UI for DAgger sessions

Unified model-deployment and DAgger web app for the dual-arm Piper rig. Runs
alongside `web/data_manager/` on separate ports
(backend 8788 / frontend 5174 vs 8787 / 5173).

## Optional master-arm modes

The active host device profile is inspected per side. Both slave arms remain
required, but master arms are optional: no master gives rollout-only operation;
one master teleoperates only its paired slave while the other slave holds; two
masters provide normal dual-arm DAgger. With two masters, the policy pauses when
the first side enters teach mode and teleoperation becomes ready only after all
available masters enter teach mode. All available masters leaving teach mode
hands control back to the policy.

The pedal only controls capture: first press starts a correction segment and
the second saves it. Leaving teach mode while recording also saves before
handback. Episode metadata records `teleop_scope`, available sides, the HOLD
policy, and a 14-dimensional `human_supervision_mask` so a single-arm correction
is not accidentally treated as a dual-arm expert target.

## What it does

- **Drive the dagger ROS2 stack**: pick a checkpoint, start
  `start_dagger_collect.sh` from the web, stop with one click. Each stack
  invocation logs to `logs/stack_<timestamp>.log` for triage.
- **Deployment modes**: `observe` (inference only), `deploy` (real-arm model
  test) and `dagger` (model execution + human intervention recording).
- **Control policy**: typed RTC, chunk-blend, timing and publish-time EMA
  configuration with presets and HOT / SAFE_IDLE / RESTART update planning.
- **Safety gate**: a policy always loads with `execute_mode=false`; enabling
  execution requires a policy ROS node, recorder/ROS health, all required
  cameras and `/policy/execute` readback.
- **Dataset provenance**: the active policy publishes an owner-checked
  deployment manifest. Every autonomy/inference/DAgger episode embeds its
  checkpoint path and fingerprint, packed `train_config.json`, prompt,
  RTC/blend/EMA/timing values, gripper/device settings and `run_id`; a copy is
  indexed under `meta/deployments/<run_id>.json`.
- **Soft controls** (driver fallback when operator can't reach a switch):
  takeover (publish `/dagger/takeover True`), handback (False), record
  toggle (`/dagger/pedal_toggled`), policy execute (`/policy/execute`).
- **Live state**: 5 Hz WebSocket snapshot of state machine, both freedrive
  buttons, pedal age, policy execute, episode counts on disk.
- **Checkpoint picker**: enumerates `/data1/DATA_IMP/checkpoints/*/` and
  flags entries missing `train_config.json` or norm_stats.

## Quick start

```bash
# from anywhere:
cd web/dagger_manager
./run.sh start            # boots backend + frontend, opens ports
./run.sh logs backend     # tail
./run.sh restart
./run.sh stop
```

For normal robot use, start the bundled infra + web lifecycle with
`./start_scripts/start_dagger_collect.sh`, then open `http://<ipc-ip>:5174/`.
Select a checkpoint and control preset, load the policy in observe mode, run
preflight, then explicitly select deploy/dagger before enabling execution.

## Architecture

```
backend (FastAPI :8788)
├── main.py        endpoints + WS /ws/dagger
├── ros_bridge.py  rclpy node — subs /dagger/state, /master_button_*,
│                  /policy/execute, /dagger/pedal_toggled
│                  pubs /dagger/takeover, /dagger/pedal_toggled,
│                  /policy/execute
├── stack.py       forks start_dagger_collect.sh in own session,
│                  kills via SIGINT→SIGTERM→SIGKILL escalation;
│                  also lists ckpts + counts episodes on disk
├── status_hub.py  5 Hz aggregator + WebSocket broadcast
├── models.py      pydantic API schemas
└── deployment/
    ├── control_policy.py typed domain config, presets, validation/update plan
    ├── ros_gateway.py    sole domain-config → ROS parameter boundary
    └── controller.py     operation mode, preflight and execute safety gate

frontend (Vite + React, :5174)
├── App.tsx        WS auto-reconnect, 4-card layout
├── components/
│   ├── StateCard       state badge + LEDs (button L/R + pedal age)
│   ├── ControlsCard    takeover/handback/record/execute buttons,
│   │                   disabled by current state
│   ├── CkptPicker      list + select + start/stop stack
│   └── EpisodesCard    inference + dagger counts on disk
└── api.ts         REST + WS client (vite proxies /api + /ws → :8788)
```

State machine values (from `dagger_recorder_node.py`):
`POLICY_RUN → ALIGNING → PRE_RECORD ↔ HUMAN_RECORD → RETURNING → POLICY_RUN`.

## Why a separate app vs. extending data_manager?

DAgger sessions have a fundamentally different lifecycle than passive
teleop recording — they bring up a policy server, run a state machine, and
own both `inference/` and `dagger/` datasets simultaneously. Folding those
into data_manager's "open / save / discard episode" model would obscure the
state machine and force every change to consider both modes. Independent
apps cost code duplication (camera streaming would have to be re-copied;
intentionally omitted from this MVP) but keep each UI focused.

## Endpoints (REST)

| Method | Path | Notes |
|---|---|---|
| GET  | `/api/health`               | liveness |
| GET  | `/api/dagger/status`        | full snapshot (same shape as WS) |
| GET  | `/api/dagger/ckpts`         | ckpt list under `/data1/DATA_IMP/checkpoints/` |
| POST | `/api/dagger/stack/start`   | `{ckpt, task?, subset?, prompt?}` |
| POST | `/api/dagger/stack/stop`    | SIGINT process group |
| POST | `/api/dagger/takeover`      | `{enable: bool}` → `/dagger/takeover` |
| POST | `/api/dagger/record/toggle` | publishes `/dagger/pedal_toggled` (Empty) |
| POST | `/api/dagger/execute`       | safety-gated `{enable: bool}` + readback |
| GET  | `/api/deployment/control/presets` | RTC/EMA/control presets |
| POST | `/api/deployment/control/plan` | classify desired changes |
| PATCH| `/api/deployment/control` | apply HOT-only acknowledged changes |
| POST | `/api/deployment/mode` | `{mode: observe\|deploy\|dagger}` |
| POST | `/api/deployment/preflight` | run current execution preflight |
| WS   | `/ws/dagger`                | 5 Hz snapshot push (no client→server) |
