#!/usr/bin/env python3
"""Authorize R1 evaluation against an exact frozen source overlay."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import openpi.training.config as training_config

from verify_pi05_r1_protocol import verify


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reverse_eval_launcher_amendment(text: str) -> str:
    replacements = (
        (
            "PROTOCOL=$REPO/lmvla/paper_iclr_lmvla/manifests/pi05_r1_protocol_v1.json\n"
            "VERIFY_REPO=${R1_VERIFY_REPO:-$REPO}\n",
            "PROTOCOL=$REPO/lmvla/paper_iclr_lmvla/manifests/pi05_r1_protocol_v1.json\n",
        ),
        (
            "PROTOCOL_OUTPUT_DIR=${R1_PROTOCOL_OUTPUT_DIR:-$REPO/logs/r1_runtime}\n",
            "",
        ),
        (
            'if [[ "$VERIFY_REPO" != "$REPO" ]]; then\n'
            '  test -s "$VERIFY_REPO/READY"\n'
            "fi\n",
            "",
        ),
        ('mkdir -p "$PROTOCOL_OUTPUT_DIR"\n', ""),
        ('  --repo "$VERIFY_REPO" --protocol "$PROTOCOL" \\\n', '  --repo "$REPO" --protocol "$PROTOCOL" \\\n'),
        (
            '  --output "$PROTOCOL_OUTPUT_DIR/protocol_eval_${CONDITION}_s${SEED}.json"\n',
            '  --output "$REPO/logs/r1/protocol_eval_${CONDITION}_s${SEED}.json"\n',
        ),
    )
    for current, frozen in replacements:
        if text.count(current) != 1:
            raise ValueError(f"unexpected R1 evaluator amendment occurrence: {current!r}")
        text = text.replace(current, frozen, 1)
    return text


def preflight(repo: Path, overlay: Path) -> dict:
    repo = repo.resolve()
    overlay = overlay.resolve()
    protocol = repo / "lmvla/paper_iclr_lmvla/manifests/pi05_r1_protocol_v1.json"
    overlay_audit = json.loads((overlay / "overlay_audit.json").read_text())
    if not overlay_audit.get("passed") or not (overlay / "READY").is_file():
        raise RuntimeError("frozen overlay is not ready")

    config_source = Path(training_config.__file__).resolve()
    if overlay not in config_source.parents:
        raise RuntimeError(f"openpi config did not load from overlay: {config_source}")

    configs = {}
    for name in (
        "pi05_robotwin_a0_public_exact_bj",
        "pi05_predictive_adapter_p1_eval",
        "pi05_r1_crave_eval",
        "pi05_r1_combined_eval",
    ):
        config = training_config.get_config(name)
        if not config.model.pi05 or config.batch_size != 16 or config.num_train_steps != 50_000:
            raise ValueError(f"R1 evaluation recipe drift: {name}")
        configs[name] = {
            "batch_size": config.batch_size,
            "num_train_steps": config.num_train_steps,
            "pi05": config.model.pi05,
        }

    frozen_launcher = (
        repo
        / "lmvla/paper_iclr_lmvla/frozen_sources/pi05_r1_v1"
        / "train_scripts/kai/eval/run_pi05_r1_formal.sh"
    )
    runtime_launcher = repo / "train_scripts/kai/eval/run_pi05_r1_formal.sh"
    normalized = reverse_eval_launcher_amendment(runtime_launcher.read_text())
    if normalized != frozen_launcher.read_text():
        raise ValueError("R1 evaluator contains changes beyond the authorized amendments")

    checkpoints = {
        "crave": repo
        / "kai0/checkpoints/pi05_r1_crave/pi05_r1_crave_seed1000/49999",
        "combined": repo
        / "kai0/checkpoints/pi05_r1_combined/pi05_r1_combined_seed1000/49999",
    }
    for name, checkpoint in checkpoints.items():
        for relative in (
            "_CHECKPOINT_METADATA",
            "params/_METADATA",
            "train_state/_METADATA",
            "assets/robotwin2.0_absolute_meanstd/norm_stats.json",
        ):
            if not (checkpoint / relative).is_file():
                raise FileNotFoundError(f"{name}: {checkpoint / relative}")

    protocol_result = verify(overlay, protocol)
    return {
        "schema_version": 1,
        "protocol": "pi05_r1_frozen_overlay_cpu_preflight_v1",
        "passed": True,
        "overlay": str(overlay),
        "overlay_audit_sha256": sha256(overlay / "overlay_audit.json"),
        "protocol_sha256": protocol_result["protocol_sha256"],
        "config_source": str(config_source),
        "config_sha256": sha256(config_source),
        "configs": configs,
        "runtime_launcher_sha256": sha256(runtime_launcher),
        "frozen_launcher_sha256": sha256(frozen_launcher),
        "launcher_amendment_only": True,
        "checkpoints": {name: str(path) for name, path in checkpoints.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = preflight(args.repo, args.overlay)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    marker = args.overlay / "CPU_PREFLIGHT"
    marker.write_text(
        f"protocol_sha256={result['protocol_sha256']}\n"
        f"audit_sha256={sha256(args.output)}\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
