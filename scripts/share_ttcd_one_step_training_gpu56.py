#!/usr/bin/env python3
"""Adopt two running TTCD trains and admit the waiting Transformer with24GiB reserve."""

from __future__ import annotations

import argparse
import contextlib
import datetime
import fcntl
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

BATCH = "ttcd-one-step-training-v1"
ROOT = Path("/ossfs/workspace/llm-memory-editability") / BATCH
HANDOFF = "gpu56-sharing-handoff-20261016/mount-order-repair-v2"
OLD_PID = 3225656
RUNNING = ("loop", "discoloop")
STAGE_MANIFEST_SHA256 = "3d6205d43898f50a83618653f7a798d478034b634b99e07750267538589d8ba9"
RESERVATIONS_MIB = {"transformer": 24576, "loop": 20480, "discoloop": 20480}
HEADROOM_MIB = 8192


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def pid_identity(pid):
    root = Path("/proc") / str(pid)
    suffix = (root / "stat").read_text().rpartition(")")[2].split()
    return {
        "pid": pid,
        "starttime": int(suffix[19]),
        "uid": root.stat().st_uid,
        "cmdline": (root / "cmdline").read_bytes().decode().split("\0")[:-1],
    }


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def load_engine(batch):
    root = batch / "launch-stage-repair-20261016"
    manifest = root / "source-manifest.json"
    if sha256(manifest) != STAGE_MANIFEST_SHA256:
        raise ValueError("Frozen stage recovery manifest changed")
    index = read_json(manifest)
    relative = "scripts/resume_ttcd_one_step_training_stages.py"
    path = root / "source" / relative
    if sha256(path) != index["files"][relative]:
        raise ValueError("Frozen stage recovery bytes changed")
    spec = importlib.util.spec_from_file_location("gpu56_handoff_stage_engine", path)
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    controller, policy, external, _, _ = module.load_frozen(batch)
    policy.RESERVATIONS_MIB["transformer"] = RESERVATIONS_MIB["transformer"]
    module.install_stage_repairs(controller, policy, external)
    return controller, module, policy


def verify_handoff_source(batch):
    root = batch / HANDOFF
    manifest = root / "source-manifest.json"
    index = read_json(manifest)
    required = {
        "scripts/share_ttcd_one_step_training_gpu56.py",
        "tests/test_ttcd_one_step_training_gpu56_handoff.py",
    }
    if index.get("engineering_passed") is not True or not required <= set(index["files"]):
        raise ValueError("Freeze the verified handoff wrapper and tests")
    for relative, expected in index["files"].items():
        path = root / "source" / relative
        if (
            path.is_symlink()
            or not path.resolve().is_relative_to((root / "source").resolve())
            or sha256(path) != expected
        ):
            raise ValueError("Frozen handoff source changed or escaped")
    if sha256(__file__) != index["files"]["scripts/share_ttcd_one_step_training_gpu56.py"]:
        raise ValueError("Execute the exact frozen handoff wrapper")
    return {"path": str(manifest), "sha256": sha256(manifest)}


def progress(batch, arm):
    path = batch / arm / "training-run/training/progress.json"
    result = read_json(path) if path.exists() else {"step": 0}
    step = result["step"]
    if step < 0 or step > 10000:
        raise ValueError("Invalid training progress")
    if step and (
        result["counts"]["base_parameter_updates"] != step
        or result["counts"]["fast_state_updates"] != 128 * step
        or result["counts"]["correction_examples"] != 128 * step
    ):
        raise ValueError("One-step scientific update accounting differs")
    return result


def canonical_mounts(mounts):
    return [
        {key: mount[key] for key in ("Source", "Destination", "RW")}
        for mount in sorted(mounts, key=lambda mount: mount["Destination"])
    ]


def binding(controller, plan, batch, arm):
    name = f"{BATCH}-{arm}-train-output-v2"
    container = controller.resources.inspect_container(name)
    if container is None or not container["State"]["Running"]:
        raise ValueError("A bound model container must still be running")
    rows, cuda = controller.resources.gpu_snapshot()
    devices = container["HostConfig"].get("DeviceRequests") or []
    if len(devices) != 1 or len(devices[0].get("DeviceIDs", [])) != 1:
        raise ValueError("Model GPU assignment differs")
    uuid = devices[0]["DeviceIDs"][0]
    gpu = next((g for g, row in rows.items() if row["uuid"] == uuid), None)
    if gpu not in (5, 6):
        raise ValueError("Only CUDA5/6 model containers may be adopted")
    _, command = controller.build_command(
        plan, arm, "train", batch, Path("/private/token"), {**rows[gpu], "gpu": gpu}
    )
    expected_cmd = command[command.index(plan["runtime"]["image"]) + 1 :]
    config, host = container["Config"], container["HostConfig"]
    manifest_sha = sha256(batch / arm / "source-manifest.json")
    mounts = {m["Destination"]: m for m in container["Mounts"]}
    if (
        config["Image"] != plan["runtime"]["image"]
        or config["User"] != "1015:1016"
        or config["Cmd"] != expected_cmd
        or f"LM_TTCD_SOURCE_MANIFEST_SHA256={manifest_sha}" not in config["Env"]
        or host["CpusetCpus"] != ",".join(map(str, plan["runtime"]["cpu_sets"][arm]))
        or "/frozen" in mounts
        or mounts["/frozen/results"]["RW"]
        or Path(mounts["/output"]["Source"]).resolve() != (batch / arm / "training-run").resolve()
        or not mounts["/output"]["RW"]
        or Path(mounts[f"/frozen/results/{BATCH}/{arm}"]["Source"]).resolve()
        != (batch / arm).resolve()
        or not mounts[f"/frozen/results/{BATCH}/{arm}"]["RW"]
    ):
        raise ValueError("Model image/source/user/command/output/CPU binding differs")
    top = controller.resources.docker("top", container["Id"], "-eo", "pid").stdout
    top_pids = {int(line.strip()) for line in top.splitlines()[1:] if line.strip().isdigit()}
    model_pids = [p["pid"] for p in cuda if p["uuid"] == uuid and p["pid"] in top_pids]
    if not model_pids:
        raise ValueError("No existing CUDA model process was bound")
    return {
        "name": name,
        "id": container["Id"],
        "state_pid": container["State"]["Pid"],
        "started_at": container["State"]["StartedAt"],
        "gpu": gpu,
        "uuid": uuid,
        "cmd": config["Cmd"],
        "image": config["Image"],
        "user": config["User"],
        "source_manifest_sha256": manifest_sha,
        "device_requests": devices,
        "mounts": canonical_mounts(container["Mounts"]),
        "cpu_set": host["CpusetCpus"],
        "state_process": pid_identity(container["State"]["Pid"]),
        "model_processes": [pid_identity(pid) for pid in sorted(model_pids)],
        "wandb": read_json(batch / arm / "training-run/wandb-train.json"),
        "progress": progress(batch, arm),
    }


def check_transformer_waiting(controller, batch):
    state = read_json(batch / "transformer/controller-state.json")
    stage = batch / "transformer/stages/train"
    if state.get("state") != "waiting_for_resources" or state.get("stage") != "train":
        raise ValueError("Transformer must still be waiting before its model launch")
    if {p.name for p in stage.iterdir()} != {"claim.json"}:
        raise ValueError("Transformer train has advanced beyond its pending claim")
    if controller.resources.inspect_container(f"{BATCH}-transformer-train-output-v2") is not None:
        raise ValueError("Transformer model already exists; never submit it again")
    if (batch / "transformer/training-run/training").exists():
        raise ValueError("Transformer scientific state already exists")
    return {"claim_sha256": sha256(stage / "claim.json"), "state": state, "updates": 0}


def capture(controller, batch):
    if not alive(OLD_PID):
        raise RuntimeError("Capture the live old host before stopping its monitoring clients")
    process = read_json(batch / "controller-process.json")
    host = pid_identity(OLD_PID)
    if process["pid"] != OLD_PID or not any(
        "resume_ttcd_one_step_training_stages.py" in value for value in host["cmdline"]
    ):
        raise ValueError("Old host identity/command differs")
    plan = read_json(batch / "transformer/source/configs/one-step-training.json")
    controller.validate_plan(plan)
    for arm in controller.ARCHITECTURES:
        controller.transport.verify_snapshot(batch, arm)
    for path, expected in process["preserved_successes"].items():
        if sha256(path) != expected:
            raise ValueError("Passed prepare/CUDA preflight evidence changed")
    return {
        "batch": BATCH,
        "old_host": host,
        "old_process": process,
        "old_process_sha256": sha256(batch / "controller-process.json"),
        "science_manifest_sha256": sha256(batch / "source-manifest.json"),
        "plan": plan,
        "source_tasks": {
            arm: sha256(batch / arm / "source/configs/task.json")
            for arm in controller.ARCHITECTURES
        },
        "transformer": check_transformer_waiting(controller, batch),
        "models": {arm: binding(controller, plan, batch, arm) for arm in RUNNING},
        "reservations_mib": RESERVATIONS_MIB,
        "headroom_mib": HEADROOM_MIB,
        "model_restart": False,
    }


def verify_snapshot(controller, batch, snapshot):
    if alive(OLD_PID):
        raise RuntimeError("Stop only the old host/client group before adopting models")
    if snapshot["old_host"]["pid"] != OLD_PID or snapshot["batch"] != BATCH:
        raise ValueError("Wrong host handoff snapshot")
    if sha256(batch / "controller-process.json") != snapshot["old_process_sha256"]:
        raise ValueError("Canonical old host record changed")
    if sha256(batch / "source-manifest.json") != snapshot["science_manifest_sha256"]:
        raise ValueError("Science manifest changed")
    current = check_transformer_waiting(controller, batch)
    if current["claim_sha256"] != snapshot["transformer"]["claim_sha256"]:
        raise ValueError("Transformer pending claim changed")
    for arm in controller.ARCHITECTURES:
        controller.transport.verify_snapshot(batch, arm)
        if sha256(batch / arm / "source/configs/task.json") != snapshot["source_tasks"][arm]:
            raise ValueError("Frozen model task changed")
    for arm in RUNNING:
        current = binding(controller, snapshot["plan"], batch, arm)
        before = snapshot["models"][arm]
        for key in before:
            if key != "progress" and current[key] != before[key]:
                raise ValueError(f"Running model identity changed: {arm}/{key}")
        if current["progress"]["step"] < before["progress"]["step"]:
            raise ValueError("Existing model training progress moved backwards")


def adopt_leases(controller, plan, batch, snapshot):
    leases = {}
    try:
        for arm in RUNNING:
            held = contextlib.ExitStack()
            leases[arm] = held
            directory = controller.SLOTS.directory
            directory.mkdir(parents=True, exist_ok=True)
            for cpu in plan["runtime"]["cpu_sets"][arm]:
                handle = held.enter_context((directory / f"cpu-{cpu}.lock").open("a"))
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            held.enter_context(controller.SLOTS.take(snapshot["models"][arm]["gpu"], arm, 3))
        return leases
    except BaseException:
        for held in leases.values():
            held.close()
        raise


def remaining_deadline(started_at, hours=48):
    started = datetime.datetime.fromisoformat(started_at.replace("Z", "+00:00")).timestamp()
    return time.monotonic() + max(0, started + hours * 3600 - time.time())


def monitor_existing(controller, plan, arm, batch, snapshot, held, update):
    bound = snapshot["models"][arm]
    stage, out = batch / arm / "stages/train", batch / arm / "training-run"
    monitor = batch / HANDOFF / arm
    monitor.mkdir(parents=True, exist_ok=True)
    with held, (monitor / "monitor.log").open("x") as log:
        resource = {
            **read_json(stage / "admission.json"),
            "adopted_existing_model": True,
            "current_gpu": bound["gpu"],
            "wallclock_is_exclusive_gpu_benchmark": False,
        }
        update(
            {
                "state": "running",
                "stage": "train",
                "container": bound["name"],
                "container_id": bound["id"],
                "adopted_existing_model": True,
                "resource": resource,
                "gpu_sharing": resource.get("gpu_sharing", True),
                "wandb_links": {"wandb-train": bound["wandb"]["url"]},
            }
        )
        code = controller.resources.monitor_container(
            bound["name"],
            log,
            remaining_deadline(bound["started_at"], plan["runtime"]["timeout_hours"]),
            update,
        )
        controller.write_json(
            stage / "container-state.json", controller.resources.inspect_container(bound["name"])
        )
        controller.write_json(
            stage / "exit.json",
            {"returncode": code, "utc": controller.resources.utc_now(), "host_handoff": HANDOFF},
        )
        if code:
            raise RuntimeError(f"Existing {arm} model exited{code}; never restart it")
        result = out / "result-train.json"
        value = read_json(result)
        if (
            value.get("passed") is False
            or value["step"] != 10000
            or value["counts"]["base_parameter_updates"] != 10000
        ):
            raise ValueError("Existing model did not complete its fixed scientific budget")
        if read_json(out / "wandb-train.json") != bound["wandb"]:
            raise ValueError("The existing model's W&B identity changed")
        controller.write_json(
            stage / "result.json", {"path": str(result), "sha256": sha256(result), "passed": True}
        )
        update(
            {
                "state": "stage_complete",
                "stage": "train",
                "wandb_links": {"wandb-train": bound["wandb"]["url"]},
            }
        )


def install_adopted_arms(controller, snapshot, leases):
    original = controller.run_arm

    def run_arm(plan, arm, batch, token):
        if arm not in RUNNING:
            return original(plan, arm, batch, token)
        state = {
            "architecture": arm,
            "completed_stages": ["prepare", "preflight"],
            "host_handoff": HANDOFF,
        }

        def update(values):
            state.update(values)
            state["utc"] = controller.resources.utc_now()
            controller.write_json(batch / arm / "controller-state.json", state)

        try:
            held = leases.pop(arm)
            try:
                monitor_existing(controller, plan, arm, batch, snapshot, held, update)
            finally:
                held.close()
            state["completed_stages"].append("train")
            for stage in controller.STAGES[controller.STAGES.index("train") + 1 :]:
                controller.run_stage(plan, arm, stage, batch, token, update)
                state["completed_stages"].append(stage)
            update({"state": "complete", "new_base_parameter_updates": 10000})
            return state
        except BaseException:
            update({"state": "needs_attention", "traceback": traceback.format_exc()})
            raise

    controller.run_arm = run_arm


def archive_pending_claim(batch):
    root = batch / HANDOFF / "old-host"
    root.mkdir(exist_ok=False)
    for name in ("controller-process.json", "controller-state.json", "schedule-amendment.json"):
        shutil.copy2(batch / name, root / name)
    stage = batch / "transformer/stages/train"
    if {p.name for p in stage.iterdir()} != {"claim.json"}:
        raise ValueError("Never move a Transformer claim that has started science")
    stage.rename(root / "transformer-pending-train")
    return root


def execute(controller, batch, snapshot, token, source):
    if not token.is_file() or token.stat().st_mode & 0o077:
        raise ValueError("Use the unchanged private W&B token")
    with (batch / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        verify_snapshot(controller, batch, snapshot)
        os.umask(0o022)
        leases = adopt_leases(controller, snapshot["plan"], batch, snapshot)
        try:
            archived = archive_pending_claim(batch)
            install_adopted_arms(controller, snapshot, leases)
            record = {
                **snapshot["old_process"],
                "pid": os.getpid(),
                "old_pid": OLD_PID,
                "utc": controller.resources.utc_now(),
                "host_handoff": source,
                "old_host_archive": str(archived),
                "models_restarted": False,
                "adopted_models": snapshot["models"],
                "reservations_mib": snapshot["reservations_mib"],
                "science_source_changed": False,
                "new_scientific_arms": 0,
            }
            controller.write_json(batch / HANDOFF / "controller-process.json", record)
            controller.write_json(batch / "controller-process.json", record)
            controller.write_json(batch / "schedule-amendment.json", record)
            return controller.execute(snapshot["plan"], batch, token)
        finally:
            for held in leases.values():
                held.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--capture", action="store_true")
    action.add_argument("--execute", action="store_true")
    parser.add_argument("--snapshot", type=Path, default=ROOT / HANDOFF / "handoff-snapshot.json")
    parser.add_argument(
        "--wandb-token-file", type=Path, default=Path.home() / ".cache/llm-memory-ttcd/wandb-token"
    )
    args = parser.parse_args()
    if not (args.capture or args.execute):
        print(
            json.dumps(
                {
                    "capture_first": True,
                    "stop_old_host_only": OLD_PID,
                    "model_restart": False,
                    "transformer_reservation_mib": 24576,
                }
            )
        )
        return 0
    source = verify_handoff_source(ROOT)
    controller, _, _ = load_engine(ROOT)
    if args.capture:
        snapshot = capture(controller, ROOT)
        with args.snapshot.open("x") as stream:
            json.dump(snapshot, stream, indent=2)
        print(
            json.dumps(
                {
                    "snapshot": str(args.snapshot),
                    "sha256": sha256(args.snapshot),
                    "models_restarted": False,
                }
            )
        )
        return 0
    source["snapshot_sha256"] = sha256(args.snapshot)
    return execute(controller, ROOT, read_json(args.snapshot), args.wandb_token_file, source)


if __name__ == "__main__":
    raise SystemExit(main())
