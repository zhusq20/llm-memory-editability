"""Shared admission must handle memory peaks and preserve reservation identities."""

import concurrent.futures
import copy
import json
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from launch_loop_shared_20261023 import latest_snapshot, shared_blockers


def inputs():
    row = dict(uuid="GPU-4", free=19000, used=78000, utilization=100)
    snap = dict(rows={4: row}, process_details=[dict(uuid="GPU-4", pid=99)], containers=[])
    plan = dict(
        gpu_uuids={"4": "GPU-4"},
        required_free_mib={"4": 18000},
        baseline_processes={"GPU-4": [99]},
        baseline_running_ids=["peer"],
    )
    return snap, plan


def container(state="created", cid="reservation"):
    return dict(
        id=cid,
        context="default",
        name="/peer",
        requests=[dict(DeviceIDs=["GPU-4"])],
        state=dict(Status=state, Pid=0, Running=False, StartedAt="0001-01-01T00:00:00Z"),
        restart=dict(Name="no"),
    )


def test_authorized_peer_at_full_utilization_is_allowed_but_memory_peak_is_not():
    snap, plan = inputs()
    assert shared_blockers(snap, 4, plan, []) == []
    snap["rows"][4]["free"] = 17999
    assert "insufficient memory" in shared_blockers(snap, 4, plan, [])[0]


def test_new_gpu_process_or_device_replacement_blocks_admission():
    snap, plan = inputs()
    snap["process_details"].append(dict(uuid="GPU-4", pid=100))
    assert shared_blockers(snap, 4, plan, []) == ["new CUDA process 100"]
    snap["process_details"].pop()
    snap["rows"][4]["uuid"] = "GPU-replacement"
    assert "GPU identity changed" in shared_blockers(snap, 4, plan, [])


def test_baseline_created_record_is_not_deleted_and_transition_reblocks():
    snap, plan = inputs()
    c = container()
    snap["containers"] = [c]
    retained = copy.deepcopy(c)
    assert shared_blockers(snap, 4, plan, [c["id"]]) == []
    assert c == retained
    c["state"].update(Status="running", Running=True, Pid=12)
    assert shared_blockers(snap, 4, plan, [c["id"]])


def test_new_reservation_blocks_and_only_current_created_container_is_ignored():
    snap, plan = inputs()
    snap["containers"] = [container()]
    assert shared_blockers(snap, 4, plan, [])
    assert shared_blockers(snap, 4, plan, [], own_id="reservation") == []


def test_only_preexisting_running_peer_is_shareable():
    snap, plan = inputs()
    snap["containers"] = [container("running", "peer")]
    assert shared_blockers(snap, 4, plan, []) == []
    snap["containers"][0]["state"]["Status"] = "restarting"
    assert shared_blockers(snap, 4, plan, [])


def test_four_admissions_do_not_race_on_the_latest_snapshot_file(tmp_path, monkeypatch):
    import launch_loop_shared_20261023 as shared

    monkeypatch.setattr(shared.resources, "snapshot", lambda: {"complete": True})

    def fixed_temporary_writer(path, data):
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(data))
        time.sleep(0.005)
        temporary.replace(path)

    lock = threading.Lock()
    path = tmp_path / "latest.json"
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [
            pool.submit(latest_snapshot, lock, path, fixed_temporary_writer) for _ in range(12)
        ]
        assert all(f.result() == {"complete": True} for f in futures)
    assert json.loads(path.read_text()) == {"complete": True}
