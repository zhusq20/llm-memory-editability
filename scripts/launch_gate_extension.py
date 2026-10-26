#!/usr/bin/env python3
"""Schedule the frozen gate extension matrix on independently idle LM GPUs.

Each scientific job owns its GPU through training and its endpoint grid. Other
projects, queued containers and failed attempts are never replaced. Outputs are
written directly to persistent storage, avoiding the nearly full local disk.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import json
import os
import queue
import sys
import threading
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import launch_paper_update_gate as common  # noqa: E402

BATCH = "gate-generalization-extension-v1"
DOCKER = common.DOCKER
read, write, sha, utc = common.read, common.write, common.sha, common.utc
_original_snapshot = common.snapshot
# Read-only inspection on 2026-10-17 established this four-day-old, explicitly
# named "unused" container has never started, has no restart policy, and no
# live controller. The user identified GPU7 as free after its creation. Retain
# the container; if it starts, the active-process snapshot blocks it normally.
INACTIVE_PLACEHOLDER = "ce2c43fca18b65b1c56ba9e586aa9e524a7516edfdba6ecb5c9d47351c5e73e8"


def inactive_placeholder(container):
    state = container.get("State", {})
    return (
        container.get("Id") == INACTIVE_PLACEHOLDER
        and container.get("Name") == "/large-fora-k7-three-sigma-0-to500-20261013-gpu7-unused"
        and state.get("Status") == "created"
        and state.get("StartedAt") == "0001-01-01T00:00:00Z"
        and not state.get("Running")
        and state.get("Pid") == 0
        and container["HostConfig"].get("RestartPolicy", {}).get("Name") == "no"
    )


def snapshot():
    """Protect created/paused reservations as well as active CUDA processes."""
    rows, blocked = _original_snapshot()
    by_uuid = {row["uuid"]: gpu for gpu, row in rows.items()}
    for context in ("default", "lm-memory"):
        identifiers = common.docker(
            "ps",
            "-a",
            "-q",
            "--filter",
            "status=created",
            "--filter",
            "status=paused",
            "--filter",
            "status=restarting",
            context=context,
        ).stdout.split()
        if not identifiers:
            continue
        containers = json.loads(common.docker("inspect", *identifiers, context=context).stdout)
        for container in containers:
            if context == "default" and inactive_placeholder(container):
                continue
            for request in container["HostConfig"].get("DeviceRequests") or []:
                for device in request.get("DeviceIDs") or []:
                    gpu = int(device) if device.isdigit() else by_uuid.get(device)
                    if gpu in rows:
                        blocked.setdefault(gpu, []).append(container["Name"])
    return rows, blocked


def validate(plan):
    if plan["batch"] != BATCH:
        raise ValueError("Unexpected batch")
    jobs = plan["jobs"]
    if len(jobs) != 23 or len({job["id"] for job in jobs}) != len(jobs):
        raise ValueError("Exactly 23 distinct scientific jobs are frozen")
    counts = {
        family: sum(j["family"] == family for j in jobs)
        for family in ("continuation", "confirmation", "wang", "depth")
    }
    if counts != {"continuation": 2, "confirmation": 9, "wang": 3, "depth": 9}:
        raise ValueError("The four experiment families changed")
    if sum(j["steps"] - j["start_step"] for j in jobs) != 12300000:
        raise ValueError("The additional scientific update budget changed")
    if any(j["arm"] == "paper" and j["family"] == "continuation" for j in jobs):
        raise ValueError("The existing P continuation must only be reused")
    if plan["gpu_candidates"] != [5, 7, 4] or plan["maximum_parallel"] != 3:
        raise ValueError("Use only the three requested GPU lanes")
    if plan["independent_audits"] is not False:
        raise ValueError("The existing independent-reload waiver remains in force")


def idle(gpu, uuid, rows, blocked):
    row = rows.get(gpu)
    return bool(
        row
        and row["uuid"] == uuid
        and not blocked.get(gpu)
        and row["used"] <= 512
        and row["utilization"] <= 5
        and row["free"] >= 18000
    )


class JobController(common.Controller):
    def __init__(self, owner, job, gpu, uuid, lane):
        config = read(owner.base / "source" / job["config"])
        super().__init__(
            config, owner.base, owner.output, job["config"], {gpu: uuid}, owner.token_file
        )
        self.owner, self.job, self.lane = owner, job, lane
        self.output = owner.output / "cases" / job["case"]
        self.output.mkdir(parents=True, exist_ok=True)
        self.cpus = owner.cpus[2 * lane : 2 * lane + 2]

    def mirror_output(self, force=False):
        # No local checkpoint mirror: the container already writes persistently.
        return None

    def update(self, name=None, **values):
        self.owner.update(self.job["id"], **values)

    def command(self, name, stage, gpu, uuid):
        label, args, _ = super().command(name, stage, gpu, uuid)
        args[args.index("--cpuset-cpus") + 1] = ",".join(map(str, self.cpus))
        old = f"type=bind,source={self.base / 'data'},target={common.DATA_TARGET},readonly"
        args[args.index(old)] = f"type=bind,source={self.base / 'data'},target=/inputs,readonly"
        runner_index = args.index("scripts/run_paper_update_gate.py")
        args[runner_index:] = [
            self.job["runner"],
            "--stage",
            stage,
            "--config",
            self.config_relative,
            "--arm",
            name,
            "--output",
            "/output",
        ]
        return label, args, self.cpus

    def worker(self, name, gpu, uuid):
        for stage in ("train", "grid"):
            # Drivers can briefly report stale utilization after a process exits.
            # Wait for genuine idle status rather than abandoning a valid grid.
            self.owner.wait_idle(gpu, uuid)
            self.execute(name, stage, gpu, uuid)
        self.update(name, state="training_and_grid_complete_audit_not_requested", grid=True)


class Controller:
    def __init__(self, plan, base, output, token_file, gpus):
        self.plan, self.base, self.output = plan, base, output
        self.token_file, self.gpus = token_file, gpus
        self.cpus = plan["cpus"]
        if len(self.cpus) != 6 or not set(self.cpus) <= set(os.sched_getaffinity(0)):
            raise ValueError("Six admitted CPU affinity slots are required")
        self.mutex = threading.RLock()
        self.pending = queue.Queue()
        for job in plan["jobs"]:
            self.pending.put(job)
        self.state = {
            "batch": BATCH,
            "state": "scheduled",
            "new_scientific_updates": 12300000,
            "independent_audits": "not requested; earlier user waiver retained",
            "jobs": {j["id"]: {"state": "pending", **j} for j in plan["jobs"]},
            "lanes": {},
            "reused_baseline": plan["reused_baseline"],
        }

    def update(self, job=None, **values):
        with self.mutex:
            target = self.state if job is None else self.state["jobs"][job]
            target.update(values)
            self.state["updated_utc"] = utc()
            write(self.output / "controller-state.json", self.state)

    def summarize(self):
        from report_gate_extension import collect

        with self.mutex:
            result = collect(self.plan, self.output)
            self.update(summary_state=result["state"], output_consistency_passed=result["passed"])
            return result

    def wait_idle(self, gpu, uuid, allow_empty=False):
        started = time.monotonic()
        while True:
            rows, blocked = snapshot()
            if idle(gpu, uuid, rows, blocked):
                return True
            if allow_empty and self.pending.empty():
                return False
            if time.monotonic() - started > self.plan["idle_wait_hours"] * 3600:
                raise TimeoutError(f"GPU {gpu} never became independently idle")
            with self.mutex:
                self.state["lanes"][str(gpu)] = {
                    "state": "waiting_for_idle_gpu",
                    "uuid": uuid,
                    "resources": rows.get(gpu),
                    "reservations": blocked.get(gpu, []),
                    "updated_utc": utc(),
                }
                self.update()
            time.sleep(self.plan["poll_seconds"])

    def lane(self, lane, gpu, uuid):
        errors = []
        # Advisory cooperative lease plus actual process/container admission.
        lease = Path("/tmp") / f"lm-memory-gate-extension-{gpu}.lock"
        with lease.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            while not self.pending.empty():
                if not self.wait_idle(gpu, uuid, allow_empty=True):
                    break
                try:
                    job = self.pending.get_nowait()
                except queue.Empty:
                    break
                with self.mutex:
                    self.state["lanes"][str(gpu)] = {
                        "state": "running",
                        "job": job["id"],
                        "uuid": uuid,
                    }
                self.update(job["id"], state="admitted", gpu=gpu, gpu_uuid=uuid)
                try:
                    JobController(self, job, gpu, uuid, lane).worker(job["arm"], gpu, uuid)
                except BaseException as error:
                    details = {
                        "job_id": job["id"],
                        "error": repr(error),
                        "traceback": traceback.format_exc(),
                    }
                    errors.append(details)
                    self.update(job["id"], state="needs_attention", **details)
                finally:
                    self.pending.task_done()
                    try:
                        summary = self.summarize()
                        if not summary["passed"]:
                            errors.append(
                                {
                                    "job_id": job["id"],
                                    "error": "Output consistency failed; see summary.json",
                                }
                            )
                    except Exception as error:
                        errors.append({"job_id": job["id"], "report_error": repr(error)})
        return errors

    def run(self):
        self.update(state="scientific_training")
        self.summarize()
        errors = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.gpus)) as pool:
            futures = [
                pool.submit(self.lane, lane, gpu, uuid)
                for lane, (gpu, uuid) in enumerate(self.gpus.items())
            ]
            for future in concurrent.futures.as_completed(futures):
                try:
                    errors.extend(future.result())
                except BaseException as error:
                    errors.append({"error": repr(error), "traceback": traceback.format_exc()})
        summary = self.summarize()
        local_complete = all(
            summary["cases"][job["case"]]["arms"][job["arm"]]["state"] == "complete"
            for job in self.plan["jobs"]
        )
        passed = not errors and self.pending.empty() and summary["passed"] and local_complete
        self.update(
            state=(
                "complete"
                if summary["state"] == "complete"
                else "local_matrix_complete_reused_baseline_pending"
            )
            if passed
            else "needs_attention",
            errors=errors,
        )
        write(self.output / "controller-exit.json", {"code": 0 if passed else 1, "utc": utc()})
        return passed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan", default=f"configs/{BATCH}.json")
    parser.add_argument("--wandb-token-file", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    base, output = args.base.resolve(), args.output.resolve()
    plan = read(base / "source" / args.plan)
    validate(plan)
    if not args.execute:
        print(json.dumps(plan, ensure_ascii=False))
        return
    common.credential_archive(args.wandb_token_file)
    common.verify_frozen(base, {"runtime": {"image": plan["image"]}})
    if len(plan["engineering_receipts"]) < 13:
        raise ValueError("CPU/data contracts and all eleven CUDA checks are required")
    for path, expected in plan["engineering_receipts"].items():
        receipt = base / path
        if sha(receipt) != expected or read(receipt).get("passed") is not True:
            raise ValueError(f"Engineering validation not passed: {path}")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output / "controller-process.json").exists():
            raise FileExistsError("Do not resubmit this batch; preserve prior controller state")
        # The inherited stage executor must use the same expanded reservation
        # check as the queue, including containers not yet started.
        common.snapshot = snapshot
        rows, _ = snapshot()
        gpus = {gpu: rows[gpu]["uuid"] for gpu in plan["gpu_candidates"]}
        write(
            output / "controller-process.json",
            {
                "pid": os.getpid(),
                "utc": utc(),
                "argv": sys.argv,
                "base": str(base),
                "output": str(output),
                "gpu_uuids": gpus,
                "freeze_sha256": sha(base / "freeze.json"),
                "plan_sha256": sha(base / "source" / args.plan),
            },
        )
        controller = Controller(plan, base, output, args.wandb_token_file, gpus)
        if not controller.run():
            raise SystemExit(1)


if __name__ == "__main__":
    os.umask(0o022)
    main()
