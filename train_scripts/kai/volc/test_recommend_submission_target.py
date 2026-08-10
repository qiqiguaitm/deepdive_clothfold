import importlib.util
from pathlib import Path
import sys


MODULE_PATH = Path(__file__).with_name("recommend_submission_target.py")
SPEC = importlib.util.spec_from_file_location("recommend_submission_target", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
router = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = router
SPEC.loader.exec_module(router)


def catalog():
    return router.load_json(Path(__file__).with_name("submission_resource_catalog.json"))


def test_catalog_sets_primary_north_gpu_limit_to_25():
    north = catalog()["resources"]["Robot-North-H20"]
    assert north["personal_limit_gpus"] == 25


def snapshot(*, gf1=8, local=2, east=8, north=20, robot_task=32):
    return {
        "timestamp": "2099-01-01T00:00:00Z",
        "resources": {
            "gf1": {"count": 8, "free_count": gf1},
            "local": {"count": 2, "free_count": local},
            "Robot-East-H20": {
                "capacity": 8,
                "active_gpus_all_users": 8 - east,
                "queueing_all_users": [],
            },
            "beijing": {
                "capacity": 56,
                "personal_limit": 25,
                "owned_active_gpus": 25 - north,
                "owned_queued_gpus": 0,
                "active_gpus_all_users": 56 - north,
                "owned_queueing": [],
                "queueing_all_users": [],
                "backup": {"enabled": False, "available": False},
            },
            "robot-task": {
                "capacity": 32,
                "active_gpus_all_users": 32 - robot_task,
                "owned_active_gpus": 32 - robot_task,
                "queueing_all_users": [],
            },
        },
    }


def ranked(gpus, *, locations=(), **capacity):
    return router.rank_targets(
        gpus=gpus,
        catalog=catalog(),
        snapshot=snapshot(**capacity),
        data_locations=set(locations),
    )


def test_one_gpu_prefers_local_then_north_without_location_information():
    assert [item.resource for item in ranked(1)[:2]] == [
        "local",
        "Robot-North-H20",
    ]


def test_four_gpu_static_preference_follows_latest_queue_priority():
    assert [item.resource for item in ranked(4)] == [
        "gf1",
        "Robot-East-H20",
        "Robot-North-H20",
        "robot-task",
    ]


def test_disabled_robot_task_remains_last_and_is_not_runnable():
    live = snapshot(gf1=0, local=0, east=0, north=0, robot_task=32)
    live["resources"]["robot-task"]["submission_enabled"] = False
    results = router.rank_targets(
        gpus=8,
        catalog=catalog(),
        snapshot=live,
        data_locations={"east_shared"},
    )
    robot_task = next(item for item in results if item.resource == "robot-task")
    assert results[-1].resource == "robot-task"
    assert not robot_task.immediately_runnable
    assert robot_task.free_gpus == 0
    assert "temporarily disabled" in robot_task.reason


def test_retired_direct_host_is_never_immediately_runnable():
    live = snapshot(gf1=8, local=0, east=8, north=0, robot_task=0)
    live["resources"]["gf1"].update(
        {
            "submission_enabled": False,
            "retired_reason": "operator retired gf1",
        }
    )
    results = router.rank_targets(
        gpus=8,
        catalog=catalog(),
        snapshot=live,
        data_locations={"east_shared"},
    )

    gf1 = next(item for item in results if item.resource == "gf1")
    assert results[0].resource == "Robot-East-H20"
    assert not gf1.immediately_runnable
    assert gf1.free_gpus == 0
    assert "operator retired gf1" in gf1.reason


def test_eight_gpu_shared_data_prefers_gf1_then_east():
    assert [item.resource for item in ranked(8, locations=("east_shared",))[:2]] == [
        "gf1",
        "Robot-East-H20",
    ]


def test_busy_preferred_resource_moves_behind_immediately_runnable_target():
    results = ranked(8, locations=("east_shared",), gf1=0, east=8)
    assert results[0].resource == "Robot-East-H20"
    assert results[-1].resource == "gf1"


def test_north_local_data_avoids_transfer_for_eight_gpu_request():
    result = ranked(8, locations=("north_shared",))[0]
    assert result.resource == "Robot-North-H20"
    assert not result.transfer_required


def test_sixteen_shared_gpus_prefer_robot_task_over_north_transfer():
    result = ranked(16, locations=("east_shared",))[0]
    assert result.resource == "robot-task"
    assert result.immediately_runnable


def test_strict_locality_excludes_cross_filesystem_targets():
    results = router.rank_targets(
        gpus=8,
        catalog=catalog(),
        snapshot=snapshot(),
        data_locations={"north_shared"},
        strict_locality=True,
    )
    assert [item.resource for item in results] == ["Robot-North-H20"]


def test_path_inference_knows_both_vepfs_mounts():
    assert router.infer_filesystems(
        [
            "/vePFS/tim/workspace/data",
            "/vePFS-North-E/vis_robot/workspace/checkpoint",
        ],
        catalog(),
    ) == {"east_shared", "north_shared"}


def test_north_uses_enabled_backup_profile_when_primary_is_full():
    live = snapshot(north=0)
    live["resources"]["beijing"]["active_gpus_all_users"] = 24
    live["resources"]["beijing"]["backup"] = {
        "enabled": True,
        "submission_enabled": True,
        "available": True,
        "managed_active_gpus": 4,
        "managed_queued_gpus": 0,
        "managed_queueing": [],
        "identity_active_gpus": 4,
        "identity_queued_gpus": 0,
        "identity_queueing": [],
        "personal_limit": 20,
    }
    capacity = router.live_capacity(
        "Robot-North-H20",
        catalog()["resources"]["Robot-North-H20"],
        live,
    )
    assert capacity.credential_profile == "backup"
    assert capacity.free_gpus == 16


def test_north_queued_gpu_quota_moves_behind_an_immediately_runnable_target():
    live = snapshot(gf1=0, east=0, north=20, robot_task=8)
    live["resources"]["beijing"]["owned_queued_gpus"] = 20
    live["resources"]["beijing"]["owned_queueing"] = ["queued"]
    live["resources"]["beijing"]["queueing_all_users"] = ["queued"]
    results = router.rank_targets(
        gpus=8,
        catalog=catalog(),
        snapshot=live,
        data_locations=set(),
    )
    assert results[0].resource == "robot-task"
    north = next(item for item in results if item.resource == "Robot-North-H20")
    assert not north.immediately_runnable
    assert north.free_gpus == 0


def test_north_is_first_when_every_candidate_would_queue():
    live = snapshot(gf1=0, local=0, east=0, north=20, robot_task=0)
    live["resources"]["beijing"]["owned_queued_gpus"] = 20
    live["resources"]["beijing"]["owned_queueing"] = ["queued"]
    live["resources"]["beijing"]["queueing_all_users"] = ["queued"]
    results = router.rank_targets(
        gpus=8,
        catalog=catalog(),
        snapshot=live,
        data_locations={"east_shared"},
    )
    assert results[0].resource == "Robot-North-H20"
    assert not any(item.immediately_runnable for item in results)
