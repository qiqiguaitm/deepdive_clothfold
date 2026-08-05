import json
from pathlib import Path

from audit_pi05_semantic_subtask_interface import audit


LEROBOT_ROOT = Path("/vePFS/tim/workspace/lerobot-main")
PUBLIC_CONFIG = Path("/vePFS/tim/hf_models/SidneyXie_pi05_robotwin/config.json")


def test_public_pi05_contract_is_action_only() -> None:
    result = audit(
        LEROBOT_ROOT / "src/lerobot/policies/pi05/modeling_pi05.py",
        LEROBOT_ROOT / "src/lerobot/policies/pi05/configuration_pi05.py",
        PUBLIC_CONFIG,
    )
    assert result["public_checkpoint_contract"]["output_feature_types"] == ["ACTION"]
    assert result["source_contract"]["semantic_api_identifiers"] == []
    assert result["supports_native_semantic_subtask"] is False


def test_semantic_identifier_alone_does_not_imply_output_channel(tmp_path: Path) -> None:
    model = tmp_path / "model.py"
    config = tmp_path / "config.py"
    checkpoint = tmp_path / "config.json"
    model.write_text("class PI05Policy:\n    def predict_subtask(self): pass\n", encoding="utf-8")
    config.write_text("class PI05Config: pass\n", encoding="utf-8")
    checkpoint.write_text(json.dumps({"output_features": {"action": {"type": "ACTION"}}}), encoding="utf-8")
    result = audit(model, config, checkpoint)
    assert result["source_contract"]["has_semantic_api"] is True
    assert result["source_contract"]["has_semantic_output"] is False
    assert result["supports_native_semantic_subtask"] is False
