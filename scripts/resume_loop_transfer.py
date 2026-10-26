#!/usr/bin/env python3
"""Adopt existing transfer containers and schedule unstarted stages on idle GPUs."""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import fcntl
import json
import os
import socket
import time
from pathlib import Path

import launch_loop_transfer as original


def read(path):
    return json.loads(Path(path).read_text())


def inspect_container(name):
    result = original.command([*original.DOCKER, "inspect", name], text=True, timeout=30)
    return json.loads(result.stdout)[0]


def owned_gpu_uuids(snap):
    """Count this host's user processes, including other LM batches."""
    owned = set()
    raw = original.command(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
        text=True,
        timeout=30,
    ).stdout
    for line in raw.splitlines():
        uuid, pid = [part.strip() for part in line.split(",")]
        try:
            if Path(f"/proc/{pid}").stat().st_uid == os.getuid():
                owned.add(uuid)
        except FileNotFoundError:
            continue
    for container in snap["containers"]:
        state = container["state"]
        if not state["Running"]:
            continue
        try:
            if Path(f"/proc/{state['Pid']}").stat().st_uid != os.getuid():
                continue
        except FileNotFoundError:
            continue
        for request in container["requests"] or []:
            for device in request.get("DeviceIDs") or []:
                owned.add(snap["rows"][int(device)]["uuid"] if device.isdigit() else device)
    return owned


def capacity_available(uuid, occupied, reserved, limit):
    allocated = occupied | reserved
    return uuid in allocated or len(allocated) < limit


def stage_action(directory):
    """An interrupted admission is reusable; any ambiguous launch must stop."""
    if not directory.exists() or not any(directory.iterdir()):
        return "unstarted"
    if (directory / "exit.json").exists():
        if read(directory / "exit.json")["code"] != 0:
            raise RuntimeError(f"Failed stage retained, no automatic retry: {directory}")
        if not (directory / "container.json").exists():
            raise RuntimeError(f"Exit receipt without container identity: {directory}")
        return "complete"
    if (directory / "container.json").exists():
        return "adopt"
    raise RuntimeError(f"Ambiguous launch retained, no duplicate submission: {directory}")


def validate_amendment(base, amendment):
    plan, manifest = read(base / "schedule.json"), read(base / "tasks.json")
    original.validate_schedule(plan, manifest)
    for name in ("schedule.json", "tasks.json"):
        if original.digest(base / name) != amendment["original_sha256"][name]:
            raise ValueError(f"Original whitelist or schedule changed: {name}")
    added = amendment["additional_gpus"]
    if not added or len(set(added)) != len(added) or set(added) & set(plan["gpus"]):
        raise ValueError("Additional GPUs must be distinct from existing lanes")
    if any(g not in range(10) for g in added):
        raise ValueError("Invalid GPU")
    if not 1 <= amendment.get("maximum_local_user_gpus", 4) <= 4:
        raise ValueError("Resource expansion must respect the four-GPU user limit")
    if set(map(str, added)) != set(amendment["additional_cpu_sets"]):
        raise ValueError("Each additional GPU needs its own CPU allocation")
    sets = [set(map(int, s.split(","))) for s in plan["cpu_sets"]]
    sets += [set(map(int, amendment["additional_cpu_sets"][str(g)].split(","))) for g in added]
    if any(a & b for i, a in enumerate(sets) for b in sets[i + 1 :]):
        raise ValueError("CPU allocations overlap")
    return plan


def preflight(base, amendment_path, token, gpu):
    amendment = read(amendment_path)
    plan = validate_amendment(base, amendment)
    output = amendment_path.parent / "preflight" / f"gpu-{gpu}"
    output.mkdir(parents=True, exist_ok=False)
    snap = original.snapshot()
    reasons = original.blockers(snap, gpu, plan["ignored_created_ids"])
    if reasons or snap["rows"][gpu]["uuid"] != amendment["additional_gpu_uuids"][str(gpu)]:
        raise RuntimeError(f"GPU {gpu} unavailable: {reasons}")
    original.write(output / "admission.json", snap)
    argv = read(base / "execution/old-invariant-raw/environment/command.json")["argv"]
    label = f"lqt-1022-rebalance-gpu{gpu}-environment"
    for flag, value in {
        "--name": label,
        "--cpuset-cpus": amendment["additional_cpu_sets"][str(gpu)],
        "--gpus": f"device={amendment['additional_gpu_uuids'][str(gpu)]}",
        "--output": f"/output/preflight/gpu-{gpu}",
    }.items():
        argv[argv.index(flag) + 1] = value
    original.write(output / "command.json", {"argv": argv, "utc": original.utc()})
    created = original.command(argv, text=True, timeout=120).stdout.strip()
    original.write(output / "container.json", {"id": created, "name": label})
    original.command(
        [*original.DOCKER, "cp", "--archive", "-", f"{label}:/"],
        input=original.credential_archive(token),
        timeout=120,
    )
    original.command([*original.DOCKER, "start", label], timeout=120)
    code = int(original.command([*original.DOCKER, "wait", label], text=True).stdout.strip())
    (output / "stdout.log").write_text(
        original.command([*original.DOCKER, "logs", label], text=True).stdout
    )
    original.write(output / "exit.json", {"code": code, "utc": original.utc()})
    if code or not read(base / f"preflight/gpu-{gpu}/complete.json")["passed"]:
        raise RuntimeError(f"GPU {gpu} preflight failed; receipts retained")


class Controller(original.Controller):
    def __init__(self, base, token, amendment_path):
        super().__init__(base, token)
        self.amendment_path = amendment_path
        self.amendment = read(amendment_path)
        validate_amendment(base, self.amendment)
        self.state = copy.deepcopy(read(base / "controller-state.json"))
        self.state.pop("waiting_gpu", None)
        self.state.pop("waiting_reasons", None)
        for gpu in self.amendment["additional_gpus"]:
            self.plan["gpus"].append(gpu)
            self.plan["cpu_sets"].append(self.amendment["additional_cpu_sets"][str(gpu)])
            self.plan["gpu_uuids"][str(gpu)] = self.amendment["additional_gpu_uuids"][str(gpu)]
        self.checked = set()

    def next_stage(self, job):
        for stage in self.tasks[job]["stages"]:
            directory = self.base / "execution" / job / stage
            action = stage_action(directory)
            if action == "complete":
                key = (job, stage)
                if key not in self.checked:
                    container = self.existing_container(job, stage)
                    if container["State"]["Status"] != "exited" or container["State"]["ExitCode"]:
                        raise RuntimeError(f"Exit receipt disagrees with Docker: {job}/{stage}")
                    self.checked.add(key)
                continue
            return stage, action
        return None

    def existing_container(self, job, stage):
        receipt = read(self.base / "execution" / job / stage / "container.json")
        container = inspect_container(receipt["id"])
        expected = f"lqt-1021-{job}-{stage}"
        if container["Name"] != f"/{expected}" or container["Image"] != self.plan["image"]:
            raise RuntimeError(f"Container identity changed: {job}/{stage}")
        argv = container["Config"]["Cmd"]
        if argv[argv.index("--job") + 1] != job or stage not in argv:
            raise RuntimeError(f"Container command changed: {job}/{stage}")
        return container

    def adopt(self, job, stage, container):
        state = container["State"]
        if state["Status"] not in ("running", "exited"):
            raise RuntimeError(f"Refusing to restart existing container: {job}/{stage}")
        if state["Status"] == "running":
            code = int(
                original.command(
                    [*original.DOCKER, "wait", container["Id"]], text=True
                ).stdout.strip()
            )
        else:
            code = state["ExitCode"]
        original.write(
            self.base / "execution" / job / stage / "exit.json",
            {"code": code, "utc": original.utc(), "adopted_without_restart": True},
        )
        if code:
            raise RuntimeError(f"Adopted container failed: {job}/{stage}: {code}")

    def submit_stage(self, pool, job, stage, gpu):
        task = self.tasks[job]
        if task["kind"] == "new":
            if original.digest(Path(task["source"]) / task["config"]) != task["config_sha256"]:
                raise ValueError(f"Frozen configuration changed: {job}")
        directory = self.base / "execution" / job / stage
        if directory.exists():
            if any(directory.iterdir()):
                raise RuntimeError(f"Refusing to overwrite existing stage: {directory}")
            preserved = self.amendment_path.parent / "empty-stage-directories" / job / stage
            preserved.parent.mkdir(parents=True, exist_ok=True)
            directory.rename(preserved)
        return pool.submit(self.execute, job, stage, self.plan["gpus"].index(gpu))

    def run_phase(self, phase, pool):
        active = {}
        while True:
            for job, (future, gpu, stage) in list(active.items()):
                if future.done():
                    try:
                        future.result()
                        self.update(job, state="between_stages", stage=stage, gpu=gpu)
                    except Exception as error:
                        self.update(job, state="failed", error=str(error))
                    del active[job]
            busy = {item[1] for item in active.values()}
            snap = original.snapshot()
            owned = owned_gpu_uuids(snap)
            for job in phase:
                if job in active or self.state["jobs"][job]["state"] in ("complete", "failed"):
                    continue
                try:
                    following = self.next_stage(job)
                    if following is None:
                        self.update(job, state="complete")
                        continue
                    stage, action = following
                    if action == "adopt":
                        container = self.existing_container(job, stage)
                        requests = container["HostConfig"]["DeviceRequests"]
                        devices = [d for r in requests for d in r.get("DeviceIDs", [])]
                        matches = [
                            g
                            for g in self.plan["gpus"]
                            if self.plan["gpu_uuids"][str(g)] in devices
                        ]
                        if len(matches) != 1 or matches[0] in busy:
                            raise RuntimeError(f"Existing GPU assignment ambiguous: {job}/{stage}")
                        gpu = matches[0]
                        future = pool.submit(self.adopt, job, stage, container)
                        self.update(job, state="running", stage=stage, gpu=gpu, adopted=True)
                    else:
                        available = [
                            g
                            for g in self.plan["gpus"]
                            if g not in busy
                            and snap["rows"][g]["uuid"] == self.plan["gpu_uuids"][str(g)]
                            and not original.blockers(snap, g, self.plan["ignored_created_ids"])
                            and capacity_available(
                                self.plan["gpu_uuids"][str(g)],
                                owned,
                                {self.plan["gpu_uuids"][str(b)] for b in busy},
                                self.amendment.get("maximum_local_user_gpus", 4),
                            )
                        ]
                        if not available:
                            continue
                        gpu = available[0]
                        future = self.submit_stage(pool, job, stage, gpu)
                        self.update(job, state="admitting", stage=stage, gpu=gpu)
                    active[job] = future, gpu, stage
                    busy.add(gpu)
                except Exception as error:
                    self.update(job, state="failed", error=str(error))
            if not active and all(
                self.state["jobs"][j]["state"] in ("complete", "failed") for j in phase
            ):
                if any(self.state["jobs"][j]["state"] != "complete" for j in phase):
                    raise RuntimeError(
                        "Phase has failed jobs; later initialization phase stays queued"
                    )
                return
            self.update(active_gpus=sorted(busy), scheduler_heartbeat_utc=original.utc())
            time.sleep(10)

    def run(self):
        locks = []
        for path in [
            self.base / "controller.lock",
            *map(Path, self.plan["source_controller_locks"]),
        ]:
            handle = path.open("a+")
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            locks.append(handle)
        if (self.amendment_path.parent / "controller-process.json").exists():
            raise FileExistsError("Recovery controller already submitted")
        for path, expected in self.plan["frozen_files"].items():
            if original.digest(path) != expected:
                raise ValueError(f"Frozen file changed: {path}")
        if (
            inspect_container(self.amendment["protected_containers"][0]["id"])["Image"]
            != self.plan["image"]
        ):
            raise ValueError("Runtime image changed")
        for gpu in self.amendment["additional_gpus"]:
            receipt = self.amendment_path.parent / "preflight" / f"gpu-{gpu}" / "exit.json"
            if (
                read(receipt)["code"]
                or not read(self.base / f"preflight/gpu-{gpu}/complete.json")["passed"]
            ):
                raise RuntimeError(f"GPU {gpu} preflight is not complete")
        original.credential_archive(self.token)
        process = {
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "utc": original.utc(),
            "schedule_sha256": original.digest(self.base / "schedule.json"),
            "controller_sha256": original.digest(__file__),
            "resumed_from_pid": self.amendment["previous_controller"]["pid"],
            "schedule_amendment": str(self.amendment_path),
        }
        original.write(self.amendment_path.parent / "controller-process.json", process)
        original.write(self.base / "controller-process.json", process)
        self.update(state="running", available_gpu_pool=self.plan["gpus"], resource_expansion=True)
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.plan["gpus"])) as pool:
                for index, phase in enumerate(self.plan["phases"]):
                    self.update(phase=index)
                    self.run_phase(phase, pool)
            self.update(state="complete")
        except Exception as error:
            self.update(state="failed", error=str(error))
            raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("preflight", "run"))
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--wandb-token-file", type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o022)
    if args.mode == "preflight":
        amendment = read(args.amendment)
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(amendment["additional_gpus"])
        ) as pool:
            futures = [
                pool.submit(preflight, args.base, args.amendment, args.wandb_token_file, gpu)
                for gpu in amendment["additional_gpus"]
            ]
            for future in futures:
                future.result()
    else:
        Controller(args.base, args.wandb_token_file, args.amendment).run()


if __name__ == "__main__":
    main()
