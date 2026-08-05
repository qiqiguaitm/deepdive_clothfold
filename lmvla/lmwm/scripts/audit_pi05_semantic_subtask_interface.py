#!/usr/bin/env python3
"""Audit whether a pi0.5 implementation exposes a semantic-subtask policy channel."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import Any


SEMANTIC_TERMS = {"subtask", "semantic_subtask", "high_level_subtask", "subtask_logits", "subtask_tokens"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _function_names(tree: ast.AST, class_name: str) -> list[str]:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return sorted(child.name for child in node.body if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)))
    return []


def _declared_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id.lower())
        elif isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name.lower())
        elif isinstance(node, ast.Attribute):
            names.add(node.attr.lower())
        elif isinstance(node, ast.arg):
            names.add(node.arg.lower())
    return names


def _feature_types(config: dict[str, Any], key: str) -> list[str]:
    features = config.get(key, {})
    if not isinstance(features, dict):
        return []
    return sorted(
        str(value.get("type", "<missing>"))
        for value in features.values()
        if isinstance(value, dict)
    )


def audit(model_source: Path, config_source: Path, checkpoint_config: Path) -> dict[str, Any]:
    model_tree = ast.parse(model_source.read_text(encoding="utf-8"), filename=str(model_source))
    config_tree = ast.parse(config_source.read_text(encoding="utf-8"), filename=str(config_source))
    public_config = json.loads(checkpoint_config.read_text(encoding="utf-8"))

    declared = _declared_names(model_tree) | _declared_names(config_tree)
    semantic_names = sorted(
        name for name in declared if any(term in name for term in SEMANTIC_TERMS)
    )
    policy_methods = _function_names(model_tree, "PI05Policy")
    core_methods = _function_names(model_tree, "PI05Pytorch")
    output_types = _feature_types(public_config, "output_features")
    input_types = _feature_types(public_config, "input_features")

    has_semantic_api = bool(semantic_names)
    has_semantic_output = any("SUBTASK" in value.upper() or "SEMANTIC" in value.upper() for value in output_types)
    action_only_public_output = bool(output_types) and set(output_types) == {"ACTION"}
    supports_native_semantic_subtask = has_semantic_api and has_semantic_output

    return {
        "sources": {
            "model": {"path": str(model_source.resolve()), "sha256": sha256(model_source)},
            "configuration": {"path": str(config_source.resolve()), "sha256": sha256(config_source)},
            "public_checkpoint_config": {
                "path": str(checkpoint_config.resolve()),
                "sha256": sha256(checkpoint_config),
            },
        },
        "public_checkpoint_contract": {
            "input_feature_types": input_types,
            "output_feature_types": output_types,
            "action_only_output": action_only_public_output,
            "tokenizer_max_length": public_config.get("tokenizer_max_length"),
            "chunk_size": public_config.get("chunk_size"),
            "n_action_steps": public_config.get("n_action_steps"),
        },
        "source_contract": {
            "pi05_policy_methods": policy_methods,
            "pi05_core_methods": core_methods,
            "semantic_api_identifiers": semantic_names,
            "has_semantic_api": has_semantic_api,
            "has_semantic_output": has_semantic_output,
        },
        "supports_native_semantic_subtask": supports_native_semantic_subtask,
        "verdict": (
            "native_semantic_subtask_channel_present"
            if supports_native_semantic_subtask
            else "no_native_semantic_subtask_channel_in_audited_implementation"
        ),
        "implication": (
            "The audited policy exposes image/language/state conditioning and action-flow output only. "
            "Generic stage or milestone embeddings are experimental conditioning interfaces, not a native "
            "pi0.5 semantic-subtask prediction head. Run the privileged semantic upper bound first; any "
            "learned semantic predictor then requires a separately specified and preflighted interface."
        ),
    }


def render_markdown(result: dict[str, Any]) -> str:
    contract = result["public_checkpoint_contract"]
    source = result["source_contract"]
    return "\n".join(
        [
            "# pi0.5 Semantic-Subtask Interface Audit",
            "",
            f"- Verdict: `{result['verdict']}`",
            f"- Public input feature types: `{', '.join(contract['input_feature_types'])}`",
            f"- Public output feature types: `{', '.join(contract['output_feature_types'])}`",
            f"- Public output is action-only: `{contract['action_only_output']}`",
            f"- Semantic API identifiers in policy/config AST: `{', '.join(source['semantic_api_identifiers']) or 'none'}`",
            f"- Native semantic-subtask channel available: `{result['supports_native_semantic_subtask']}`",
            "",
            "## Interpretation",
            "",
            result["implication"],
            "",
            "This result is an implementation-contract audit, not evidence that semantic subtasks lack control utility.",
            "",
            "## Source Identity",
            "",
            *[
                f"- {name}: `{entry['path']}` (`{entry['sha256']}`)"
                for name, entry in result["sources"].items()
            ],
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-source", type=Path, required=True)
    parser.add_argument("--config-source", type=Path, required=True)
    parser.add_argument("--checkpoint-config", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    args = parser.parse_args()

    result = audit(args.model_source, args.config_source, args.checkpoint_config)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown_out.write_text(render_markdown(result), encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
