import hashlib
import json
import os
from pathlib import Path
import subprocess


SCRIPT = (
    Path(__file__).parents[1]
    / "prefetch_temporal_grounding_tg4_eval_checkpoint_to_north.sh"
)
RUN_ID = "temporal_grounding_tg4_auxiliary_only_seed1100"
TASK_ID = f"{RUN_ID}_train"
RUN_NAME = f"test+{RUN_ID}"


def prepare(tmp_path: Path, expected_sha: str | None = None) -> dict[str, Path]:
    repo = tmp_path / "repo"
    model = (
        repo
        / "lmvla/lawam/results/Checkpoints/robotwin"
        / RUN_NAME
        / "final_model/pytorch_model.pt"
    )
    model.parent.mkdir(parents=True)
    model.write_bytes(b"accepted TG4 checkpoint")
    digest = hashlib.sha256(model.read_bytes()).hexdigest()
    acceptance = repo / "acceptance.json"
    acceptance.write_text(
        json.dumps(
            {
                "complete": True,
                "runs": [
                    {
                        "task_id": TASK_ID,
                        "run": (
                            "lmvla/lawam/results/Checkpoints/robotwin/"
                            f"{RUN_NAME}"
                        ),
                        "final_checkpoint_sha256": expected_sha or digest,
                        "final_checkpoint_bytes": model.stat().st_size,
                    }
                ],
            }
        )
    )

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_ssh = fake_bin / "ssh"
    fake_ssh.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "while (($#)); do\n"
        "  case \"$1\" in\n"
        "    -p|-o) shift 2 ;;\n"
        "    -*) shift ;;\n"
        "    *) shift; break ;;\n"
        "  esac\n"
        "done\n"
        "exec bash -c \"$*\"\n"
    )
    fake_ssh.chmod(0o755)
    fake_sync = tmp_path / "sync.sh"
    fake_sync.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "mkdir -p \"$DST\"\n"
        "cp -a \"$SRC/.\" \"$DST/\"\n"
    )
    fake_sync.chmod(0o755)
    return {
        "repo": repo,
        "north": tmp_path / "north",
        "model": model,
        "acceptance": acceptance,
        "output": repo / "prefetch.ok",
        "fake_bin": fake_bin,
        "fake_sync": fake_sync,
    }


def run_prefetch(paths: dict[str, Path]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SCRIPT)],
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PATH": f"{paths['fake_bin']}:{os.environ['PATH']}",
            "REPO": str(paths["repo"]),
            "NORTH_REPO": str(paths["north"]),
            "TG4_ARM": "auxiliary_only",
            "TG4_TRAIN_SEED": "1100",
            "TG4_ACCEPTANCE_MARKER": str(paths["acceptance"]),
            "TG4_PREFETCH_OUTPUT": str(paths["output"]),
            "TG4_NORTH_SYNC": str(paths["fake_sync"]),
        },
    )


def test_prefetch_uploads_then_reuses_a_hash_verified_checkpoint(
    tmp_path: Path,
) -> None:
    paths = prepare(tmp_path)

    first = run_prefetch(paths)

    assert first.returncode == 0, first.stderr
    assert "transfer=uploaded" in paths["output"].read_text()
    remote = (
        paths["north"]
        / ".staging/temporal_grounding_tg4_eval_v1/repo"
        / "lmvla/lawam_local/results/Checkpoints/robotwin"
        / RUN_NAME
        / "final_model/pytorch_model.pt"
    )
    assert remote.read_bytes() == paths["model"].read_bytes()

    paths["output"].unlink()
    second = run_prefetch(paths)

    assert second.returncode == 0, second.stderr
    assert "transfer=reused" in paths["output"].read_text()


def test_prefetch_rejects_a_model_that_disagrees_with_acceptance(
    tmp_path: Path,
) -> None:
    paths = prepare(tmp_path, expected_sha="0" * 64)

    completed = run_prefetch(paths)

    assert completed.returncode != 0
    assert not paths["output"].exists()
