from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).with_name(
    "verify_temporal_grounding_tg2_seed_independence.py"
)
SPEC = importlib.util.spec_from_file_location("tg2_seed_independence", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)


def write_orders(root: Path, *, collapse_seeds: bool = False) -> None:
    for seed in verifier.SEEDS:
        effective_seed = 1000 if collapse_seeds else seed
        for arm in verifier.ARMS:
            run_id = f"temporal_grounding_tg2_{arm}_seed{seed}"
            output = root / "logs/temporal_grounding/tg2/data_order" / run_id
            output.mkdir(parents=True)
            for rank in range(4):
                (output / f"rank{rank}.json").write_text(
                    json.dumps(
                        {
                            "arm": arm,
                            "training_seed": seed,
                            "rank": rank,
                            "world_size": 4,
                            "microbatches": 10,
                            "samples": 160,
                            "sha256": f"seed-{effective_seed}-rank-{rank}",
                        }
                    )
                )


def test_seed_independence_accepts_matched_arms_and_distinct_seeds(
    tmp_path: Path,
) -> None:
    write_orders(tmp_path)

    result = verifier.audit(tmp_path)

    assert result["complete"]
    assert result["checks"] == {
        "dataset_order_equal_within_seed": True,
        "dataset_order_distinct_across_seeds": True,
    }


def test_seed_independence_rejects_collapsed_seed_orders(tmp_path: Path) -> None:
    write_orders(tmp_path, collapse_seeds=True)

    with pytest.raises(ValueError, match="do not induce distinct"):
        verifier.audit(tmp_path)
