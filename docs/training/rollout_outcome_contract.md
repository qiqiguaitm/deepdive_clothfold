# Rollout outcome contract

Newly recorded episodes store a versioned `rollout_outcome` object in
`meta/episodes.jsonl`. Frame parquet files remain unchanged.

```json
{
  "success": false,
  "rollout_outcome": {
    "schema_version": 1,
    "label": "partial_success",
    "rollout_mode": "intervention",
    "stage_outcomes": [
      {"stage": "flatten", "success": true, "progress": 1.0},
      {"stage": "left_fold", "success": false, "failure_mode": "missed_grasp"}
    ],
    "failure_modes": ["missed_grasp"],
    "intervention_count": 1,
    "recovery_success": false,
    "unsafe_event": false,
    "time_limit_reached": false
  }
}
```

`label` is one of `success`, `partial_success`, `failure`, or `aborted`.
`rollout_mode` is one of `demonstration`, `autonomous`, `intervention`, or
`recovery`. Existing manifests without this object remain valid and are read as
demonstrations, with the label derived from their legacy `success` field.

Training and data-build code should use
`openpi.training.episode_outcomes.normalize_rollout_outcome` instead of parsing
these fields independently.

## Compatibility guarantees

- Existing save clients may continue to send only `success`, `note`, and
  `scene_tags`; all new request fields have defaults.
- Existing episode manifests require no rewrite. A missing `rollout_outcome`
  is interpreted from the legacy `success` flag.
- No parquet columns, feature declarations, `task_index`, normalization stats,
  or current training transforms are changed.
- The new object is namespaced, so existing scripts that copy or ignore unknown
  episode metadata keys continue to behave as before.
