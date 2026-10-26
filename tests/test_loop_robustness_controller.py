"""Resource admission must never interpret another live project as available."""

import copy

from launch_loop_robustness import FAILED_HEALTH_ID, eligible_gpu, failed_health_check


def test_only_exact_failed_oneoff_health_check_is_ignored():
    container = {
        "Id": FAILED_HEALTH_ID,
        "Name": "/table4-gpu4-health-20261005",
        "State": {
            "Status": "created",
            "Pid": 0,
            "Running": False,
            "StartedAt": "0001-01-01T00:00:00Z",
            "Error": "gpu requires reset",
        },
        "HostConfig": {"RestartPolicy": {"Name": "no"}},
    }
    assert failed_health_check(container)
    for key, value in (("Running", True), ("Pid", 5), ("Status", "running")):
        altered = copy.deepcopy(container)
        altered["State"][key] = value
        assert not failed_health_check(altered)
    container["Id"] = "another-project"
    assert not failed_health_check(container)


def test_external_process_and_reservations_block_even_own_sharing():
    row = {"free": 80000, "used": 15000, "utilization": 99}
    assert not eligible_gpu(row, [], 0)
    assert eligible_gpu(row, [], 1)
    assert not eligible_gpu(row, [], 2)
    assert not eligible_gpu(row, ["active CUDA process 7"], 1)
    assert not eligible_gpu(row, ["/other-project-queued"], 1)
    assert not eligible_gpu({**row, "free": 15000}, [], 1)
    assert eligible_gpu({**row, "used": 0, "utilization": 0}, [], 0)
