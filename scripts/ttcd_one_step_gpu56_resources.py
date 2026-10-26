"""Observable-process binding and memory reservations for authorized CUDA5/6 sharing."""

from __future__ import annotations

import contextlib
import fcntl
import subprocess
import sys
import threading
import time
from pathlib import Path

BATCH = "ttcd-one-step-training-v1"
ALLOWED = (5, 6)
EXTERNAL_PIDS = {3168776: 5, 3164122: 6, 3165961: 6}
RESERVATIONS_MIB = {"transformer": 28672, "loop": 20480, "discoloop": 20480}
HEADROOM_MIB = 8192


def validate_snapshot(snapshot):
    if (
        snapshot.get("schema") != "ttcd-one-step-gpu56-sharing-v1"
        or snapshot.get("batch") != BATCH
        or snapshot.get("allowed_gpus") != list(ALLOWED)
    ):
        raise ValueError("Only the recorded CUDA5/6 sharing amendment is supported")
    records = snapshot.get("external_processes", [])
    if len(records) != len(EXTERNAL_PIDS) or {r["pid"] for r in records} != set(EXTERNAL_PIDS):
        raise ValueError("Only the three independently observed external PIDs may share")
    for record in records:
        if (
            record["gpu"] != EXTERNAL_PIDS[record["pid"]]
            or not record["uuid"].startswith("GPU-")
            or type(record["starttime"]) is not int
            or type(record["uid"]) is not int
            or record["starttime"] <= 0
            or record["uid"] < 0
            or not record["exe"].startswith("/")
            or record.get("exe_source") != "nvidia-smi process_name"
            or record.get("proc_exe_readable") is not False
        ):
            raise ValueError("External identity must retain its actual observable provenance")


def process_identity(pid, process_name, proc_root=Path("/proc")):
    """Use readable start time/owner and NVML's path; do not claim /proc/exe access."""
    directory = Path(proc_root) / str(pid)
    text = (directory / "stat").read_text()
    _, marker, remainder = text.rpartition(")")
    if not marker or len(remainder.split()) < 20:
        raise ValueError("Malformed process start-time record")
    return {
        "pid": pid,
        "starttime": int(remainder.split()[19]),
        "uid": directory.stat().st_uid,
        "exe": process_name,
        "exe_source": "nvidia-smi process_name",
        "proc_exe_readable": False,
    }


def memory_snapshot():
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,used_gpu_memory,process_name",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    records = {}
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        uuid, pid, memory, name = (part.strip() for part in line.split(",", 3))
        key = (uuid, int(pid))
        if key in records:
            raise ValueError("Duplicate CUDA application in the memory snapshot")
        records[key] = {
            "uuid": uuid,
            "pid": int(pid),
            "used_memory_mib": int(memory),
            "process_name": name,
        }
    return records


def observed_external(snapshot, gpu, row, processes, memory, identity_reader=None):
    if gpu not in ALLOWED:
        raise ValueError("This amendment cannot admit any GPU outside CUDA5/6")
    bindings = {record["pid"]: record for record in snapshot["external_processes"]}
    identity_reader = identity_reader or process_identity
    current = {(p["uuid"], p["pid"]) for p in processes if p["uuid"] == row["uuid"]}
    measured = {key for key in memory if key[0] == row["uuid"]}
    if current != measured:
        raise ValueError("CUDA processes changed between snapshots; take fresh resources")
    external = []
    for _, pid in sorted(current):
        if pid not in bindings:
            continue  # The original cohort validator must account for every other PID.
        expected = bindings[pid]
        observed = memory[(row["uuid"], pid)]
        if expected["gpu"] != gpu or expected["uuid"] != row["uuid"]:
            raise ValueError("An external PID moved to a different GPU identity")
        identity = identity_reader(pid, observed["process_name"])
        if any(identity[field] != expected[field] for field in ("pid", "starttime", "uid", "exe")):
            raise ValueError("An external PID was replaced or changed its observable identity")
        if observed["used_memory_mib"] < 0:
            raise ValueError("The external CUDA allocation is unavailable")
        external.append({**expected, "used_memory_mib": observed["used_memory_mib"]})
    return external


def install(controller, snapshot):
    """Replace only host admission before its GPU acquisition threads begin."""
    validate_snapshot(snapshot)
    if getattr(controller, "_gpu56_sharing_installed", False):
        raise ValueError("The sharing amendment is already installed")
    original_acquire, original_cohort = controller.acquire_resources, controller.gpu_cohort
    for frame in sys._current_frames().values():
        while frame is not None:
            if frame.f_code is original_acquire.__code__:
                raise RuntimeError(
                    "Install before resource acquisition starts; no live-loop hot patch"
                )
            frame = frame.f_back
    resources = controller.resources
    admission_guard = threading.Lock()

    def inspect_cohort(batch, plan, gpu, row, processes, containers, memory):
        try:
            external = observed_external(snapshot, gpu, row, processes, memory)
        except (OSError, ValueError, KeyError) as error:
            raise resources.ResourceUnavailable(str(error)) from error
        known = {(record["uuid"], record["pid"]) for record in external}
        filtered = [p for p in processes if (p["uuid"], p["pid"]) not in known]
        checked_row = {**row, "memory": 0, "utilization": 0} if external else row
        # Retain the original own-container image/source/user/command/output/PID checks.
        members = original_cohort(batch, plan, gpu, checked_row, filtered, containers)
        return members, external

    def gpu_cohort(batch, plan, gpu, row, processes, containers):
        members, _ = inspect_cohort(batch, plan, gpu, row, processes, containers, memory_snapshot())
        return members

    def acquire_resources(plan, arm, stage, update, batch, slots=None):
        if stage in controller.CPU_STAGES:
            return original_acquire(plan, arm, stage, update, batch, slots)
        if arm not in RESERVATIONS_MIB or plan["runtime"].get("gpu_allowed") != list(ALLOWED):
            raise ValueError("Keep the authorized architecture set and CUDA5/6 scope")
        slots = slots or controller.SLOTS
        slots.directory.mkdir(parents=True, exist_ok=True)
        runtime = plan["runtime"]
        deadline = time.monotonic() + runtime["wait_timeout_hours"] * 3600
        while time.monotonic() < deadline:
            held = contextlib.ExitStack()
            try:
                for cpu in runtime["cpu_sets"][arm]:
                    handle = held.enter_context((slots.directory / f"cpu-{cpu}.lock").open("a"))
                    try:
                        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError as error:
                        raise resources.ResourceUnavailable("CPU lease is occupied") from error
                containers, available = resources.admit_cpu(plan, arm)
                rows, processes = resources.gpu_snapshot()
                try:
                    memory = memory_snapshot()
                except (OSError, ValueError, subprocess.SubprocessError) as error:
                    raise resources.ResourceUnavailable(
                        "Cannot measure live CUDA allocations"
                    ) from error
                preferred = list(dict.fromkeys([*runtime["gpu_preferences"][arm], *ALLOWED]))
                reasons = {}
                with admission_guard:
                    with slots.guard:
                        leased_arms = {gpu: set(arms) for gpu, arms in slots.arms.items()}
                    # Spread the first two arms across5/6; then consider permitted sharing.
                    choices = sorted(
                        (g for g in preferred if g in ALLOWED),
                        key=lambda g: len(leased_arms.get(g, set())),
                    )
                    for gpu in choices:
                        lease = None
                        try:
                            if gpu not in rows or rows[gpu]["total_mib"] < 90000:
                                raise resources.ResourceUnavailable(
                                    "Authorized GPU is absent or too small"
                                )
                            row = rows[gpu]
                            members, external = inspect_cohort(
                                batch, plan, gpu, row, processes, containers, memory
                            )
                            existing = leased_arms.get(gpu, set())
                            if arm in existing:
                                raise ValueError("An architecture already holds a GPU lease")
                            member_arms = {
                                name
                                for name in RESERVATIONS_MIB
                                if any(m["name"].startswith(f"{BATCH}-{name}-") for m in members)
                            }
                            if not member_arms <= existing:
                                raise resources.ResourceUnavailable(
                                    "A running own container lacks a reservation lease"
                                )
                            reserve = RESERVATIONS_MIB[arm]
                            previous = sum(RESERVATIONS_MIB[name] for name in existing)
                            external_memory = sum(record["used_memory_mib"] for record in external)
                            if (
                                row["free_mib"] < reserve + HEADROOM_MIB
                                or external_memory + previous + reserve + HEADROOM_MIB
                                > row["total_mib"]
                            ):
                                raise resources.ResourceUnavailable(
                                    "Memory plus running/launching reservations exceed capacity"
                                )
                            lease = slots.take(gpu, arm, runtime["maximum_own_containers_per_gpu"])
                            held.enter_context(lease)
                            return held, {
                                **row,
                                "gpu": gpu,
                                "cpu_set": runtime["cpu_sets"][arm],
                                "host_available_kib": available,
                                "cohort": members,
                                "external_processes": external,
                                "external_memory_mib": external_memory,
                                "existing_reservations_mib": previous,
                                "new_reservation_mib": reserve,
                                "headroom_mib": HEADROOM_MIB,
                                "reserved_architectures": sorted((*existing, arm)),
                                "gpu_sharing": bool(external or existing or members),
                                "wallclock_is_exclusive_gpu_benchmark": False,
                                "identity_exe_source": "nvidia-smi process_name",
                                "proc_exe_readable": False,
                                "utc": resources.utc_now(),
                            }
                        except (resources.ResourceUnavailable, BlockingIOError) as error:
                            if lease is not None:
                                lease.close()
                            reasons[str(gpu)] = str(error)
                raise resources.ResourceUnavailable(str(reasons))
            except resources.ResourceUnavailable as error:
                held.close()
                update({"state": "waiting_for_resources", "stage": stage, "reason": str(error)})
                time.sleep(min(30, max(0, deadline - time.monotonic())))
            except BaseException:
                held.close()
                raise
        raise TimeoutError("No safe CUDA5/6 sharing slot became available; no process was changed")

    controller.gpu_cohort = gpu_cohort
    controller.acquire_resources = acquire_resources
    controller._gpu56_sharing_installed = True
    return {
        "allowed_gpus": list(ALLOWED),
        "reservations_mib": RESERVATIONS_MIB,
        "external_headroom_mib": HEADROOM_MIB,
        "identity_exe_source": "nvidia-smi process_name",
        "proc_exe_readable": False,
        "scientific_source_changed": False,
    }
