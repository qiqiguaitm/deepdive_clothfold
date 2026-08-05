# RoboTwin Rollout Artifact Audit

- Root: `/vePFS/tim/workspace/deepdive_kai0/lmvla/lawam_local/results/eval_runs/robotwin`
- Valid summaries: 1245/1245
- Episodes: 61343 (48245 success, 13098 failure; 78.65%)
- Mean steps, success/failure: 332.34302000207276 / 943.0447396549091
- Trajectory-like files: 0
- Episode fields: `episode_id, seed, slot_id, steps, success`

## Verdict

Historical artifacts support outcome and episode-duration analysis only. They do not support post-hoc CRAVE progress, density, stall-lead-time, or regression analyses.

## Required Follow-up

- Saved summaries contain episode outcomes and lengths but no frame observations.
- CRAVE progress, recurrence density, stall lead time, and regression detection cannot be reconstructed.
- A new outcome-labeled rollout collection with saved visual observations is required.
