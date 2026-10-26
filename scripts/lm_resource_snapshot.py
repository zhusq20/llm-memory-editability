"""Read scheduling fields only; transient Docker errors block admission."""

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ACTIVE = ("created", "running", "paused", "restarting")
FORMAT = (
    '{"id":{{json .Id}},"name":{{json .Name}},'
    '"state":{{json .State}},"requests":{{json .HostConfig.DeviceRequests}},'
    '"restart":{{json .HostConfig.RestartPolicy}}}'
)


class ResourceUnavailable(RuntimeError):
    pass


def verify_code_bundle(directory):
    manifest = json.loads((directory / "code-manifest.json").read_text())
    if not manifest:
        raise ValueError("Empty recovery code manifest")
    for name, expected in manifest.items():
        if Path(name).name != name:
            raise ValueError("Invalid recovery manifest path")
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Recovery code changed: {name}")


def call(args):
    try:
        return subprocess.run(args, text=True, capture_output=True, check=True, timeout=30).stdout
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise ResourceUnavailable(f"Resource query failed: {' '.join(args[:5])}") from exc


def active_ids(context):
    args = ["docker", "--context", context, "ps", "-aq", "--no-trunc"]
    for state in ACTIVE:
        args += ["--filter", f"status={state}"]
    return call(args).split()


def containers(context):
    result = []
    for cid in active_ids(context):
        try:
            record = json.loads(
                call(["docker", "--context", context, "inspect", "--format", FORMAT, cid])
            )
        except ResourceUnavailable:
            # A container can disappear between ps and inspect. Confirm it no
            # longer reserves a device; a daemon/permission error stays closed.
            if cid not in active_ids(context):
                continue
            raise
        if record["state"]["Status"] in ACTIVE:
            result.append(dict(record, context=context))
    return result


def snapshot():
    rows = {}
    for line in call(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    ).splitlines():
        index, uuid, used, free, util = [v.strip() for v in line.split(",")]
        rows[int(index)] = dict(uuid=uuid, used=int(used), free=int(free), utilization=int(util))
    pids = []
    for line in call(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"]
    ).splitlines():
        uuid, pid = [v.strip() for v in line.split(",")]
        pids.append(dict(uuid=uuid, pid=int(pid)))
    return dict(
        utc=datetime.now(timezone.utc).isoformat(),
        rows=rows,
        processes=[p["uuid"] for p in pids],
        process_details=pids,
        containers=containers("default") + containers("lm-memory"),
    )


def released_created(record, released_ids):
    state = record["state"]
    return (
        record["id"] in released_ids
        and state["Status"] == "created"
        and state["Pid"] == 0
        and not state["Running"]
        and state["StartedAt"] == "0001-01-01T00:00:00Z"
        and record["restart"]["Name"] == "no"
    )


def owned_gpu_uuids(snap):
    owned = set()
    for process in snap["process_details"]:
        try:
            if Path(f"/proc/{process['pid']}").stat().st_uid == os.getuid():
                owned.add(process["uuid"])
        except FileNotFoundError:
            continue
    return owned


def knowledge_view(snap, reserved_devices, released_ids=()):
    rows, blocked = snap["rows"], {}
    by_uuid = {r["uuid"]: i for i, r in rows.items()}
    for p in snap["process_details"]:
        if p["uuid"] in by_uuid:
            blocked.setdefault(by_uuid[p["uuid"]], []).append(f"active CUDA process {p['pid']}")
    for c in snap["containers"]:
        if released_created(c, released_ids):
            continue
        for gpu in reserved_devices(c["requests"], rows):
            blocked.setdefault(gpu, []).append(f"{c['context']}:{c['name']}")
    return rows, blocked
