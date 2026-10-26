"""Authorized GPU8 sharing by the existing Transformer and Loop TTCD arms."""

from __future__ import annotations

import fcntl
import os
import threading
from pathlib import Path

GPU = 8
MINIMUM_FREE_MIB = 32768
BATCH = "ttcd-architecture-comparison-v1"
SOURCE = Path("/ossfs/workspace/llm-memory-editability") / BATCH / "source"
SCRIPT = "/frozen/scripts/run_ttcd_architecture.py"


def cohort(helpers):
    rows, processes = helpers.gpu_snapshot()
    if GPU not in rows or rows[GPU]["total_mib"] < 90000:
        raise helpers.ResourceUnavailable("The authorized GPU8 capacity is unavailable")
    row = rows[GPU]
    containers, known, arms = [], set(), set()
    for container in helpers.running_containers(["lm-memory"]):
        if not helpers.container_uses_gpu(container, GPU, row["uuid"]):
            continue
        name = container["Name"].lstrip("/")
        arm = next(
            (arm for arm in ("transformer", "loop") if name.startswith(f"{BATCH}-{arm}-")),
            None,
        )
        if arm is None or arm in arms:
            raise helpers.ResourceUnavailable("GPU8 contains an unrelated or duplicate arm")
        mounts = [m for m in container["Mounts"] if m["Destination"] == "/frozen"]
        devices = container["HostConfig"].get("DeviceRequests") or []
        if (
            len(mounts) != 1
            or mounts[0]["RW"]
            or Path(mounts[0]["Source"]).resolve() != SOURCE.resolve()
            or SCRIPT not in container["Config"]["Cmd"]
            or container["Config"].get("User") != "1015:1016"
            or len(devices) != 1
            or devices[0].get("DeviceIDs") != [row["uuid"]]
        ):
            raise helpers.ResourceUnavailable("GPU8 container source/command/owner differs")
        output = helpers.docker("top", container["Id"], "-eo", "pid").stdout
        pids = {int(line.strip()) for line in output.splitlines()[1:] if line.strip().isdigit()}
        known.update(pids)
        arms.add(arm)
        containers.append({"name": name, "id": container["Id"], "host_pids": sorted(pids)})
    unknown = [p["pid"] for p in processes if p["uuid"] == row["uuid"] and p["pid"] not in known]
    if unknown:
        raise helpers.ResourceUnavailable(f"GPU8 has outside CUDA processes: {unknown}")
    if not containers and (row["memory"] > 1024 or row["utilization"] > 5):
        raise helpers.ResourceUnavailable("GPU8 has unaccounted memory/utilization")
    return row, containers


def admission(helpers, runtime, arm):
    if arm not in ("transformer", "loop"):
        raise ValueError("Only the two explicitly authorized architecture arms may share GPU8")
    _, available = helpers.admit_cpu({"runtime": runtime}, arm)
    row, containers = cohort(helpers)
    if row["free_mib"] < MINIMUM_FREE_MIB:
        raise helpers.ResourceUnavailable("Need32GiB free before a new GPU8 stage")
    return {
        **row,
        "gpu": GPU,
        "cpu_set": runtime["cpu_sets"][arm],
        "host_available_kib": available,
        "utc": helpers.utc_now(),
        "gpu_sharing": True,
        "minimum_free_mib": MINIMUM_FREE_MIB,
        "authorization": "User requested Loop audit/training on GPU8 alongside Transformer",
        "cohort": containers,
        "wallclock_is_exclusive_gpu_benchmark": False,
    }


class _Lease:
    def __init__(self, pool, arm):
        self.pool, self.arm, self.closed = pool, arm, False

    def close(self):
        if not self.closed:
            self.pool.release(self.arm)
            self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class CohortSlots:
    """One global GPU lock, with independent leases for the two local arms."""

    def __init__(self, path=None):
        self.path = path or Path("/tmp") / f"lm-memory-gpu-locks-{os.getuid()}" / "gpu-8.lock"
        self.guard, self.handle, self.arms = threading.Lock(), None, set()

    def take(self, arm):
        with self.guard:
            if arm not in ("transformer", "loop") or arm in self.arms:
                raise ValueError("Each authorized architecture has at most one active stage")
            if self.handle is None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                handle = self.path.open("a")
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BaseException:
                    handle.close()
                    raise
                self.handle = handle
            self.arms.add(arm)
            return _Lease(self, arm)

    def release(self, arm):
        with self.guard:
            self.arms.remove(arm)
            if not self.arms:
                self.handle.close()
                self.handle = None
