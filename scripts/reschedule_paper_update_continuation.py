#!/usr/bin/env python3
"""Adopt two live continuations; hand off queued arms or use an extra GPU.

Only the old scheduler is replaced. Frozen training code, inputs, optimizer
states, budgets, container identities, and running training processes stay put.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path


def lanes(extra_gpu):
    if extra_gpu is None:
        return {1: ("token-update",), 2: ("token-update-half-lr",)}
    if extra_gpu in (1, 2) or extra_gpu < 0:
        raise ValueError("The extra GPU must differ from the two live lanes")
    return {
        1: ("token-update",),
        2: ("token-update-half-lr", "fixed-update"),
        extra_gpu: ("paper",),
    }


def validate_container(item, launch, command, image, gpu_uuid):
    """Bind adoption to the recorded container, command, runtime, and mounts."""
    if item["Id"] != launch["id"] or item["Name"].lstrip("/") != launch["container"]:
        raise ValueError("Container identity differs from launch receipt")
    if item["Image"] != image or item["Config"]["Image"] != image:
        raise ValueError("Container runtime image changed")
    if item["Config"]["Cmd"] != command[command.index(image) + 1 :]:
        raise ValueError("Container command changed")
    if item["HostConfig"]["CpusetCpus"] != command[command.index("--cpuset-cpus") + 1]:
        raise ValueError("Container CPU allocation changed")
    devices = [
        d
        for r in item["HostConfig"].get("DeviceRequests") or []
        for d in r.get("DeviceIDs") or []
    ]
    if devices != [gpu_uuid]:
        raise ValueError("Container GPU allocation changed")
    actual = {
        m["Destination"]: (m["Source"], m["RW"])
        for m in item["Mounts"]
        if m["Type"] == "bind"
    }
    for i, value in enumerate(command):
        if value == "--mount":
            spec = dict(
                p.split("=", 1) if "=" in p else (p, True)
                for p in command[i + 1].split(",")
            )
            if actual.get(spec["target"]) != (spec["source"], "readonly" not in spec):
                raise ValueError("Container bind mount changed")
    if not item["State"]["Running"] and item["State"]["ExitCode"] != 0:
        raise RuntimeError("Failed training is preserved, never retried")


def validate_old_process(pid, base):
    argv = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    expected = str(base / "source/scripts/launch_paper_update_continuation.py").encode()
    if (
        expected not in argv
        or str(base).encode() not in argv
        or b"--execute" not in argv
    ):
        raise ValueError("PID is not the expected old scheduling process")
    return [a.decode() for a in argv if a]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--mirror", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--new-gpu", type=int)
    mode.add_argument("--handoff-only", action="store_true")
    parser.add_argument(
        "--cpus", default="4,5", help="Two CPU slots for the additional lane"
    )
    parser.add_argument("--wandb-token-file", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    base = args.base.resolve()
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(base / "source/scripts"), str(base / "source/src")]
    import launch_paper_update_continuation as frozen

    common = frozen.original
    schedule = lanes(args.new_gpu)
    cpus = [int(x) for x in args.cpus.split(",")]
    if len(cpus) != 2 or len(set(cpus)) != 2 or set(cpus) & {0, 1, 2, 3}:
        raise ValueError("Use two distinct CPUs apart from the existing two lanes")
    if not set(cpus) <= os.sched_getaffinity(0):
        raise ValueError("Requested CPU slots unavailable")
    output = base / "output"
    intervention = output / (
        "scheduling-handoff-v1"
        if args.handoff_only
        else f"scheduling-intervention-gpu{args.new_gpu}"
    )
    config_relative = f"configs/{frozen.BATCH}.json"
    config = common.read(base / "source" / config_relative)
    frozen.validate_config(config)
    common.verify_frozen(base, config)
    common.credential_archive(args.wandb_token_file)
    original_snapshot = common.snapshot

    def snapshot():
        rows, blocked = original_snapshot()
        by_uuid = {r["uuid"]: i for i, r in rows.items()}
        for context in ("default", "lm-memory"):
            ids = common.docker(
                "ps",
                "-aq",
                "--filter",
                "status=created",
                "--filter",
                "status=paused",
                "--filter",
                "status=restarting",
                context=context,
            ).stdout.split()
            if ids:
                for item in json.loads(
                    common.docker("inspect", *ids, context=context).stdout
                ):
                    for request in item["HostConfig"].get("DeviceRequests") or []:
                        for device in request.get("DeviceIDs") or []:
                            gpu = (
                                int(device) if device.isdigit() else by_uuid.get(device)
                            )
                            if gpu in rows:
                                blocked.setdefault(gpu, []).append(item["Name"])
        return rows, blocked

    common.snapshot = snapshot
    rows, blocked = snapshot()
    row = rows.get(args.new_gpu)
    if not args.handoff_only and (
        not row
        or blocked.get(args.new_gpu)
        or row["used"] > 512
        or row["utilization"] > 5
        or row["free"] < 18000
    ):
        raise ValueError(
            f"Additional GPU is occupied/reserved: {row}, {blocked.get(args.new_gpu)}"
        )
    prior = common.read(output / "controller-state.json")
    process = common.read(output / "controller-process.json")
    old_pid = process["pid"]
    old_argv = validate_old_process(old_pid, base)
    for name in ("paper", "fixed-update"):
        if (
            prior["arms"][name]["state"] != "pending"
            or (output / "execution" / name).exists()
            or (output / "runs" / name).exists()
        ):
            raise ValueError(
                f"{name} already has an execution claim; refuse duplicate submission"
            )
    gpus = {gpu: rows[gpu]["uuid"] for gpu in schedule}
    adopted = {}
    for arm, gpu in (("token-update", 1), ("token-update-half-lr", 2)):
        folder = output / "execution" / arm / "train"
        launch = common.read(folder / "launch.json")
        command = common.read(folder / "command.json")["argv"]
        item = json.loads(common.docker("inspect", launch["container"]).stdout)[0]
        validate_container(item, launch, command, config["runtime"]["image"], gpus[gpu])
        if not item["State"]["Running"]:
            raise ValueError(
                "Initial takeover expects the two existing training containers alive"
            )
        adopted[arm] = {
            "container": item["Name"],
            "id": item["Id"],
            "pid": item["State"]["Pid"],
            "started_at": item["State"]["StartedAt"],
        }
    plan = {
        "utc": common.utc(),
        "old_scheduler_pid": old_pid,
        "old_argv": old_argv,
        "lanes": schedule,
        "new_gpu": row,
        "new_gpu_index": args.new_gpu,
        "new_lane_cpus": cpus if not args.handoff_only else [],
        "handoff_arms": ["paper", "fixed-update"] if args.handoff_only else [],
        "adopted_containers": adopted,
        "freeze_sha256": common.sha(base / "freeze.json"),
        "config_sha256": common.sha(base / "source" / config_relative),
        "source_changed": False,
        "new_budget": 0,
        "authorization": "User requested dispatch of waiting experiments to two identical remote GPUs, then clarified manual remote execution and that local GPUs 3/4 are occupied.",
        "ars_workflow": "experiment-agent/run (inline)",
        "script_sha256": common.sha(Path(__file__)),
    }

    class Controller(common.Controller):
        def __init__(self):
            super().__init__(
                config, base, args.mirror, config_relative, gpus, args.wandb_token_file
            )
            self.state = copy.deepcopy(prior)
            self.cpu_map = {1: [0, 1], 2: [2, 3], args.new_gpu: cpus}
            self.state.update(
                active_supervisor=str(intervention / "process.json"),
                lanes=[
                    {"gpu": g, "arms": list(a), "gpu_uuid": gpus[g]}
                    for g, a in schedule.items()
                ],
            )
            for gpu, names in schedule.items():
                for name in names:
                    self.state["arms"][name].update(gpu=gpu, gpu_uuid=gpus[gpu])
            if args.handoff_only:
                for name in plan["handoff_arms"]:
                    self.state["arms"][name] = {
                        "state": "paused_for_handoff",
                        "destination": "user_managed_remote_two_gpus",
                        "handoff": str(intervention / "handoff-ready.json"),
                    }

        def command(self, name, stage, gpu, uuid):
            label, command, _ = super().command(name, stage, gpu, uuid)
            command[command.index("--cpuset-cpus") + 1] = ",".join(
                map(str, self.cpu_map[gpu])
            )
            old_mount = (
                f"type=bind,source={base / 'data'},target={common.DATA_TARGET},readonly"
            )
            command[command.index(old_mount)] = (
                f"type=bind,source={base / 'data'},target=/inputs,readonly"
            )
            index = command.index("scripts/run_paper_update_gate.py")
            command[index:] = [
                frozen.RUNNER,
                "--stage",
                stage,
                "--config",
                config_relative,
                "--arm",
                name,
                "--output",
                "/output",
            ]
            return label, command, self.cpu_map[gpu]

        def execute(self, name, stage, gpu, uuid):
            folder = output / "execution" / name / stage
            if not folder.exists():
                if stage == "train" and (output / "runs" / name).exists():
                    raise ValueError(
                        "Training output already exists without a launch claim"
                    )
                deadline = time.monotonic() + 300
                while True:
                    current, reservations = snapshot()
                    r = current[gpu]
                    if (
                        r["uuid"] == uuid
                        and not reservations.get(gpu)
                        and r["used"] <= 512
                        and r["utilization"] <= 5
                        and r["free"] >= 18000
                    ):
                        return super().execute(name, stage, gpu, uuid)
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            "GPU failed to become idle; no stage was launched"
                        )
                    time.sleep(5)
            launch = common.read(folder / "launch.json")
            recorded = common.read(folder / "command.json")["argv"]
            _, expected, _ = self.command(name, stage, gpu, uuid)
            if recorded != expected:
                raise ValueError("Existing stage differs from unchanged frozen command")
            started = datetime.fromisoformat(launch["utc"]).timestamp()
            self.update(
                name,
                state="running",
                stage=stage,
                gpu=gpu,
                gpu_uuid=uuid,
                container=launch["container"],
                adopted=True,
            )
            while True:
                item = json.loads(common.docker("inspect", launch["container"]).stdout)[
                    0
                ]
                validate_container(
                    item, launch, recorded, config["runtime"]["image"], uuid
                )
                common.write(
                    folder / "heartbeat.json",
                    {
                        "state": item["State"],
                        "utc": common.utc(),
                        "supervisor_pid": os.getpid(),
                    },
                )
                if not item["State"]["Running"]:
                    if not (folder / "exit.json").exists():
                        common.write(
                            folder / "exit.json",
                            {
                                "code": item["State"]["ExitCode"],
                                "utc": common.utc(),
                                "adopted": True,
                            },
                        )
                    self.mirror_output(force=True)
                    return
                if (
                    time.time() - started
                    > config["runtime"]["stage_timeout_hours"] * 3600
                ):
                    common.write(
                        folder / "timeout.json",
                        {"own_container": launch["container"], "utc": common.utc()},
                    )
                    common.docker("stop", "--time", "30", launch["container"])
                    raise TimeoutError("Inherited stage timeout exceeded")
                usage = (
                    subprocess.run(
                        [
                            "nvidia-smi",
                            f"--id={gpu}",
                            "--query-gpu=memory.used,utilization.gpu",
                            "--format=csv,noheader,nounits",
                        ],
                        check=True,
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )
                    .stdout.strip()
                    .split(",")
                )
                common.write(
                    output / "resources" / f"{name}.json",
                    {
                        "gpu": gpu,
                        "gpu_uuid": uuid,
                        "used_mib": int(usage[0]),
                        "utilization": int(usage[1]),
                        "utc": common.utc(),
                    },
                )
                self.mirror_output()
                time.sleep(config["runtime"]["poll_seconds"])

        def lane(self, gpu, names):
            errors = []
            for name in names:
                try:
                    self.worker(name, gpu, gpus[gpu])
                except Exception as error:
                    errors.append({"arm": name, "error": repr(error)})
                finally:
                    self.mirror_output(force=True)
            return errors

    controller = Controller()
    for name, gpu in (("token-update", 1), ("token-update-half-lr", 2)):
        recorded = common.read(output / "execution" / name / "train/command.json")[
            "argv"
        ]
        if controller.command(name, "train", gpu, gpus[gpu])[1] != recorded:
            raise ValueError("Adopted command differs; old scheduler left untouched")
    if not args.execute:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    # Persist the takeover record before changing the old scheduler process.
    intervention.mkdir(exist_ok=False)
    common.write(intervention / "plan.json", plan)
    common.write(intervention / "previous-controller-state.json", prior)
    common.write(intervention / "previous-controller-process.json", process)
    (intervention / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    controller.mirror_output(force=True)
    validate_old_process(old_pid, base)
    os.kill(
        old_pid, signal.SIGTERM
    )  # Exact scheduler PID only, never a process group/container.
    with (output / "controller.lock").open("a") as lock:
        deadline = time.monotonic() + 15
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() > deadline:
                    raise RuntimeError(
                        "Old scheduler still owns the lock; no new work launched"
                    )
                time.sleep(0.2)
        common.write(
            intervention / "process.json",
            {
                "pid": os.getpid(),
                "utc": common.utc(),
                "argv": sys.argv,
                "replaced_pid": old_pid,
            },
        )
        if args.handoff_only:
            common.write(
                intervention / "handoff-ready.json",
                {
                    "ready": True,
                    "utc": common.utc(),
                    "arms": plan["handoff_arms"],
                    "old_scheduler_stopped": True,
                    "active_containers_unchanged": adopted,
                    "freeze_sha256": plan["freeze_sha256"],
                    "config_sha256": plan["config_sha256"],
                    "remote_training_started": False,
                },
            )
        controller.update(state="scientific_continuation", scheduling_amendment=plan)
        controller.mirror_output(force=True)
        errors = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            futures = [
                pool.submit(controller.lane, gpu, names)
                for gpu, names in schedule.items()
            ]
            for future in concurrent.futures.as_completed(futures):
                try:
                    errors.extend(future.result())
                except Exception as error:
                    errors.append(
                        {"error": repr(error), "traceback": traceback.format_exc()}
                    )
        terminal = (
            "local_arms_complete_handoff_arms_pending"
            if args.handoff_only
            else "training_and_grid_complete_independent_audit_not_requested"
        )
        controller.update(
            state="needs_attention" if errors else terminal,
            errors=errors,
            grid_evaluated=not errors and not args.handoff_only,
        )
        common.write(
            intervention / "exit.json",
            {"code": 1 if errors else 0, "utc": common.utc()},
        )
        controller.mirror_output(force=True)
        if errors:
            raise SystemExit(1)


if __name__ == "__main__":
    os.umask(0o022)
    main()
