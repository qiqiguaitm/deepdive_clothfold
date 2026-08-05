# pi0.5 Semantic-Subtask Interface Audit

- Verdict: `no_native_semantic_subtask_channel_in_audited_implementation`
- Public input feature types: `STATE, VISUAL, VISUAL, VISUAL`
- Public output feature types: `ACTION`
- Public output is action-only: `True`
- Semantic API identifiers in policy/config AST: `none`
- Native semantic-subtask channel available: `False`

## Interpretation

The audited policy exposes image/language/state conditioning and action-flow output only. Generic stage or milestone embeddings are experimental conditioning interfaces, not a native pi0.5 semantic-subtask prediction head. Run the privileged semantic upper bound first; any learned semantic predictor then requires a separately specified and preflighted interface.

This result is an implementation-contract audit, not evidence that semantic subtasks lack control utility.

## Source Identity

- model: `/vePFS/tim/workspace/lerobot-main/src/lerobot/policies/pi05/modeling_pi05.py` (`8faec89af9d31f5c2f0e12b4ea369d2af6bf0c25050a13adcaefbd9b8aa03b69`)
- configuration: `/vePFS/tim/workspace/lerobot-main/src/lerobot/policies/pi05/configuration_pi05.py` (`a8b80f54ffe98993529d8013b1f8ddde4289a71b5c6a5e41d7346ac7622da6a6`)
- public_checkpoint_config: `/vePFS/tim/hf_models/SidneyXie_pi05_robotwin/config.json` (`e6ce5bd5aef6640db61f2dbe41178a1aa669821c403b0b1f3cc34fe48bbc83a6`)
