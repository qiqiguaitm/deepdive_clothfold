#!/usr/bin/env python3
"""Audit and gate the frozen three-seed MT5 temporal-scale 2x2."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any


EXPECTED_SEEDS = (1000, 1001, 1002)


def load_matrix_module():
    path = Path(__file__).with_name("summarize_pi05_confirmatory_matrix.py")
    spec = importlib.util.spec_from_file_location("pi05_confirmatory_matrix", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_seed_paths(values: list[str]) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for value in values:
        seed_text, path_text = value.split("=", 1)
        seed = int(seed_text)
        if seed in result:
            raise ValueError(f"duplicate training seed {seed}")
        result[seed] = Path(path_text)
    if set(result) != set(EXPECTED_SEEDS):
        raise ValueError(f"expected seeds {EXPECTED_SEEDS}, got {sorted(result)}")
    return result


def complementarity_gate(contrasts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    checks = {}
    for name in ("combined_minus_local", "combined_minus_transition"):
        contrast = contrasts[name]
        ci = contrast.get("ci95") if contrast.get("available") else None
        checks[f"{name}_available"] = bool(contrast.get("available"))
        checks[f"{name}_positive"] = bool(contrast.get("available")) and float(
            contrast["point_estimate_macro_delta"]
        ) > 0.0
        checks[f"{name}_ci95_excludes_zero"] = ci is not None and float(ci[0]) > 0.0
    return {"accepted": all(checks.values()), "checks": checks}


def analyze(
    reports: dict[str, dict[int, dict[str, Any]]],
    manifest: dict[str, Any],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    matrix = load_matrix_module()
    audits = {}
    for method, seed_reports in reports.items():
        for seed, report in sorted(seed_reports.items()):
            audit = matrix.audit_report(report, manifest)
            audits[f"{method}:seed{seed}"] = audit
            if not audit["accepted"]:
                raise ValueError(f"protocol audit rejected {method}:seed{seed}: {audit['errors']}")
    pairs = {
        "local_minus_a0": ("local", "a0"),
        "transition_minus_a0": ("transition", "a0"),
        "combined_minus_a0": ("combined", "a0"),
        "combined_minus_local": ("combined", "local"),
        "combined_minus_transition": ("combined", "transition"),
    }
    contrasts = {
        name: matrix.paired_hierarchical_contrast(
            reports[candidate],
            reports[baseline],
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed + index,
        )
        for index, (name, (candidate, baseline)) in enumerate(pairs.items())
    }
    return {
        "complete": True,
        "protocol_audits": audits,
        "contrasts": contrasts,
        "gate": complementarity_gate(contrasts),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    for method in ("a0", "local", "transition", "combined"):
        parser.add_argument(f"--{method}", action="append", required=True, help="SEED=REPORT.json")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260803)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--accepted-marker", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    if protocol.get("version") != 1:
        raise ValueError("unexpected MT5 protocol version")
    reports = {
        method: {
            seed: json.loads(path.read_text())
            for seed, path in parse_seed_paths(getattr(args, method)).items()
        }
        for method in ("a0", "local", "transition", "combined")
    }
    result = analyze(
        reports,
        json.loads(args.manifest.read_text()),
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    result["protocol"] = str(args.protocol.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    args.accepted_marker.unlink(missing_ok=True)
    if result["gate"]["accepted"]:
        args.accepted_marker.parent.mkdir(parents=True, exist_ok=True)
        args.accepted_marker.write_text(f"accepted=true\nresult={args.output.resolve()}\n")


if __name__ == "__main__":
    main()
