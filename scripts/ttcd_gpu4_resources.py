"""User-authorized sharing of GPU4 within the two current TTCD batches only."""

from __future__ import annotations

import contextlib
import fcntl
import os
import time
from pathlib import Path

PERSISTENT = Path("/ossfs/workspace/llm-memory-editability")
COHORTS = {
    "ttcd-successful-base-v1-": (
        PERSISTENT / "ttcd-successful-base-v1/source-v2",
        "/frozen/scripts/run_ttcd_verified_base.py",
    ),
    "ttcd-double-new-eval-v1-": (
        PERSISTENT / "ttcd-double-new-eval-v1/source",
        "/frozen/scripts/run_ttcd_double_new_eval.py",
    ),
}


def _cohort_snapshot(helpers):
    rows, processes = helpers.gpu_snapshot()
    if 4 not in rows or rows[4]["total_mib"] < 90000:
        raise helpers.ResourceUnavailable("The authorized GPU4 capacity is unavailable")
    row = rows[4]
    containers = helpers.running_containers(["lm-memory"])
    own, pids = [], set()
    for container in containers:
        if not helpers.container_uses_gpu(container, 4, row["uuid"]):
            continue
        name = container["Name"].lstrip("/")
        expected = next(
            (value for prefix, value in COHORTS.items() if name.startswith(prefix)), None
        )
        if expected is None:
            raise helpers.ResourceUnavailable(
                f"GPU4 contains a container outside this TTCD cohort: {name}"
            )
        source, script = expected
        mounts = [m for m in container["Mounts"] if m["Destination"] == "/frozen"]
        requests = container["HostConfig"].get("DeviceRequests") or []
        if (
            len(mounts) != 1
            or mounts[0]["RW"]
            or Path(mounts[0]["Source"]).resolve() != source.resolve()
            or script not in container["Config"]["Cmd"]
            or container["Config"].get("User") != "1015:1016"
            or len(requests) != 1
            or requests[0].get("DeviceIDs") != [row["uuid"]]
        ):
            raise helpers.ResourceUnavailable(
                "GPU4 cohort container source/command/ownership differs"
            )
        output = helpers.docker("top", container["Id"], "-eo", "pid").stdout
        known = {int(line.strip()) for line in output.splitlines()[1:] if line.strip().isdigit()}
        pids.update(known)
        own.append({"name": name, "id": container["Id"], "host_pids": sorted(known)})
    unknown = [p["pid"] for p in processes if p["uuid"] == row["uuid"] and p["pid"] not in pids]
    if unknown:
        raise helpers.ResourceUnavailable(
            f"GPU4 has CUDA processes outside the authorized cohort: {unknown}"
        )
    if not own and (row["memory"] > 1024 or row["utilization"] > 5):
        raise helpers.ResourceUnavailable("GPU4 has unaccounted memory/utilization")
    return {"row": row, "containers": own, "authorized_host_pids": sorted(pids)}


def verify_gpu4_cohort(helpers):
    # Container creation/exit and the NVML process snapshot are not atomic.
    # Briefly resnapshot before treating an unrecognized PID as persistent.
    for attempt in range(3):
        try:
            return _cohort_snapshot(helpers)
        except helpers.ResourceUnavailable:
            if attempt == 2:
                raise
            time.sleep(0.25)


def admission(helpers, runtime, phase):
    _, available = helpers.admit_cpu({"runtime": runtime}, phase)
    cohort = verify_gpu4_cohort(helpers)
    row = cohort["row"]
    if row["free_mib"] < 24000:
        raise helpers.ResourceUnavailable(
            "Need24GiB free before starting another TTCD stage on GPU4"
        )
    return {
        **row,
        "gpu": 4,
        "phase": phase,
        "host_available_kib": available,
        "cpu_set": runtime["cpu_sets"][phase],
        "utc": helpers.utc_now(),
        "gpu_sharing": True,
        "authorization": "User explicitly requested concurrent current TTCD runs on GPU4",
        "sharing_scope": "Only the two frozen current TTCD batches; unknown CUDA PIDs are rejected",
        "cohort": cohort["containers"],
    }


@contextlib.contextmanager
def acquire_gpu4_lock():
    directory = Path("/tmp") / f"lm-memory-gpu-locks-{os.getuid()}"
    directory.mkdir(exist_ok=True)
    with (directory / "gpu-4.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield lock
