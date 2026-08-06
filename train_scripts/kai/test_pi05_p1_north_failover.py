from __future__ import annotations

from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[2]


def test_pair_launch_is_authorized_before_disjoint_gpu_processes() -> None:
    text = (
        REPO / "train_scripts/kai/run_pi05_p1_north_failover_pair.sh"
    ).read_text()

    authorization_check = text.index(
        'test "$(jq -r \'.launch_authorized\' "$AUTH_AUDIT")" = true'
    )
    a0_launch = text.index("run_arm a0 0,1,2,3 &")
    candidate_launch = text.index("run_arm candidate 4,5,6,7 &")
    assert authorization_check < a0_launch < candidate_launch
    assert "flock -n 9" in text
    assert "pair_training_complete" in text


def test_result_sync_validates_hidden_checkpoint_before_atomic_publish() -> None:
    text = (
        REPO / "train_scripts/kai/sync_pi05_p1_north_failover_results.sh"
    ).read_text()

    incoming = text.index("local incoming=$local_parent/.north-incoming-$final_name")
    metadata = text.index('test -s "$incoming/_CHECKPOINT_METADATA"')
    params = text.index('test -s "$incoming/params/_METADATA"')
    train_state = text.index('test -s "$incoming/train_state/_METADATA"')
    publish = text.index('mv "$incoming" "$final"')
    assert incoming < metadata < params < train_state < publish
    assert "flock -n 9" in text


def test_stage_sync_does_not_consume_artifact_manifest_from_stdin() -> None:
    text = (
        REPO / "train_scripts/kai/sync_pi05_p1_north_failover_stage.sh"
    ).read_text()

    assert 'SSH=(ssh -n -p "$SSH_PORT"' in text
    assert text.count("</dev/null && return 0") == 2
    assert "RSYNC_MAX_ATTEMPTS" in text


def test_north_pair_yaml_is_whole_node_nonretrying_and_stage_gated() -> None:
    path = REPO / "train_scripts/kai/volc/pi05_p1_north_failover_pair_8h20.yaml"
    config = yaml.safe_load(path.read_text())

    assert config["ResourceQueueName"] == "Robot-North-H20"
    assert config["TaskRoleSpecs"][0]["Flavor"] == "ml.hpcpni3ln.45xlarge"
    assert config["Storages"][0] == {
        "Type": "Vepfs",
        "VepfsId": "vepfs-cnbj875793a96d6b",
        "SubPath": "/vis_robot",
        "MountPath": "/vePFS-North-E/vis_robot",
    }
    assert config["RetryOptions"] == {"EnableRetry": False, "MaxRetryTimes": 0}
    entrypoint = config["Entrypoint"]
    assert "north_stage_report.json" in entrypoint
    assert "pi05_p1_north_runtime_preflight.json" in entrypoint
