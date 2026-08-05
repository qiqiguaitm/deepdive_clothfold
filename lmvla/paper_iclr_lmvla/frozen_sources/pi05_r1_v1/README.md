# pi0.5 P1/R1 frozen source overrides

These files restore the exact source identities recorded by
`pi05_predictive_adapter_p1_baseline_audit.json` and
`pi05_r1_protocol_v1.json` without changing the shared working tree.

They were recovered from the recorded pre-drift tool output and reversible
post-freeze patches. Each recovered file was accepted only after its SHA-256
matched the corresponding frozen manifest:

| Path | SHA-256 |
|---|---|
| `kai0/src/openpi/training/config.py` | `eac7a10bdfd1ecc2a5d6679e78f1f45965bc833cc471594abba8374790e133f1` |
| `train_scripts/kai/run_pi05_predictive_adapter_p1_train.sh` | `f52fa26d554c890746818d849059ceaa11de7262299f16970a630a6e111e2534` |
| `train_scripts/kai/run_pi05_r1_train.sh` | `75a30aeb26dc0a8a02e26515451d6a7f15f09cbcd69f65c701eaa6430cad40ac` |
| `train_scripts/kai/eval/run_pi05_r1_formal.sh` | `eda2f3b7568a16c1e482c95b84c4da80b49d4b9272b486a9d85674441bf0ccdd` |

The shared versions contain unrelated Task-N configuration and checkpoint
resume support. Do not copy these overrides back over shared files. A launch
may use them only through a separately audited source overlay that also keeps
checkpoints, datasets, normalization assets, and result paths on their
canonical mounts.
