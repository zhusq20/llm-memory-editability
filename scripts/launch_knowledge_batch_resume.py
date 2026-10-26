#!/usr/bin/env python3
"""Adopt the eight existing arms, migrating only at newly committed checkpoints."""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import socket
import subprocess
import time
from pathlib import Path

import accelerate_knowledge_factorial as A
from knowledge_batch_resume import memory_allowance

F, L = A.F, A.L


class Controller(A.Controller):
    def __init__(self, plan, amendment):
        self.plan, self.amendment = plan, amendment
        self.batch = Path(plan["output"])
        self.root = self.batch / "batchsize-20261022"
        self.extension = self.root / "extension-v1"
        self.state = json.loads((self.batch / "controller-state.json").read_text())
        self.reservations = dict(amendment["old_reservations"])
        for arm in F.ARMS:
            job = self.state["jobs"][arm]
            if job["state"] == "running":
                expected = amendment["old_containers"][arm]
                actual = A.inspect(job["container"])
                if actual["id"] != expected["id"] or not actual["Running"]:
                    raise ValueError(f"Old training identity changed: {arm}")
                job.update(migration="waiting_new_checkpoint", container_id=actual["id"])
            elif job["state"] in ("waiting_resource", "pending"):
                job.update(migration="not_started", state="waiting_resource", stage="train")
            else:
                raise ValueError(f"Unexpected initial job state: {arm}")

    def save(self):
        self.state.update(updated_utc=L.utc(), scheduler="batchsize-20261022", state="training")
        L.write(self.batch / "controller-state.json", self.state)

    def start_continuation(self, arm, lane, cap, source=None, previous_step=0):
        job = self.state["jobs"][arm]
        folder = self.root / "execution" / arm / "train"
        if folder.exists():
            raise FileExistsError(f"Preserve existing attempt: {folder}")
        name, argv = F.command(self.plan, "train", arm, lane)
        name = f"kmb-1022-{arm}-train"
        argv[argv.index("--name") + 1] = name
        index = argv.index("scripts/run_knowledge_method_factorial.py")
        argv[index:] = [
            "/speedup/run_knowledge_batch_resume.py",
            "--config",
            f"/project/{self.plan['scientific_config']}",
            "--batch",
            "/batch",
            "--arm",
            arm,
            "--memory-cap-mib",
            str(cap),
            "--previous-observed-step",
            str(previous_step),
        ]
        if source is not None:
            argv += ["--source-checkpoint", f"/batch/runs/{arm}/resume.pt"]
        L.write(folder / "command.json", {"argv": argv, "utc": L.utc()})
        L.call(argv, timeout=120)
        subprocess.run(
            [*L.DOCKER, "cp", "-", f"{name}:/"],
            input=A.extension_archive(self.extension),
            check=True,
            capture_output=True,
        )
        subprocess.run(
            [*L.DOCKER, "cp", "--archive", "-", f"{name}:/"],
            input=L.credential_archive(self.plan["wandb_token_file"]),
            check=True,
            capture_output=True,
        )
        L.call([*L.DOCKER, "start", name], timeout=120)
        actual = A.inspect(name)
        job.update(
            state="running",
            stage="train",
            migration="benchmarking",
            container=name,
            container_id=actual["id"],
            gpu=lane["gpu"],
            gpu_uuid=lane["gpu_uuid"],
            cpus=lane["cpus"],
            started_utc=L.utc(),
        )
        self.reservations[arm] = cap + 6144
        with (folder / "stdout.log").open("x") as handle:
            subprocess.Popen(
                [*L.DOCKER, "logs", "--follow", name], stdout=handle, stderr=subprocess.STDOUT
            )
        self.save()

    def finish(self, arm, job, actual):
        stage = job["stage"]
        folder = self.root / "execution" / arm / stage
        L.write(folder / "exit.json", {"code": actual["ExitCode"], "utc": L.utc()})
        status_path = self.batch / "stage-status" / f"{arm}-{stage}.json"
        status = json.loads(status_path.read_text()) if status_path.exists() else {}
        if actual["ExitCode"] or status.get("state") != "complete":
            job.update(state="needs_attention", error=f"{stage} exited without completion")
        else:
            if stage == "train":
                end = json.loads((self.batch / "runs" / arm / "complete.json").read_text())
                if end.get("exposure") != 1489200 or end.get("smoke") is not False:
                    raise ValueError(f"Exposure endpoint differs: {arm}")
            following = A.next_stage(stage)
            job.setdefault("stages", []).append(stage)
            job.update(state="waiting_resource" if following else "complete", stage=following)
        self.save()

    def start_evaluation(self, arm, lane):
        job = self.state["jobs"][arm]
        stage = job["stage"]
        folder = self.root / "execution" / arm / stage
        if folder.exists():
            raise FileExistsError(folder)
        name, argv = F.command(self.plan, stage, arm, lane)
        name = f"kmb-1022-{arm}-{stage}"
        argv[argv.index("--name") + 1] = name
        L.write(folder / "command.json", {"argv": argv, "utc": L.utc()})
        L.call(argv, timeout=120)
        subprocess.run(
            [*L.DOCKER, "cp", "--archive", "-", f"{name}:/"],
            input=L.credential_archive(self.plan["wandb_token_file"]),
            check=True,
            capture_output=True,
        )
        L.call([*L.DOCKER, "start", name], timeout=120)
        actual = A.inspect(name)
        job.update(
            state="running",
            container=name,
            container_id=actual["id"],
            gpu=lane["gpu"],
            gpu_uuid=lane["gpu_uuid"],
            cpus=lane["cpus"],
        )
        with (folder / "stdout.log").open("x") as handle:
            subprocess.Popen(
                [*L.DOCKER, "logs", "--follow", name], stdout=handle, stderr=subprocess.STDOUT
            )
        self.save()

    def tick(self):
        running = {}
        for arm in F.ARMS:
            job = self.state["jobs"][arm]
            if job["state"] != "running":
                continue
            actual = A.inspect(job["container"])
            if actual["id"] != job["container_id"]:
                raise ValueError("Container identity changed")
            if not actual["Running"]:
                self.finish(arm, job, actual)
                continue
            running[arm] = actual
            selection = self.root / "runs" / arm / "selection.json"
            progress = self.root / "runs" / arm / "progress.json"
            if job.get("migration") == "benchmarking" and selection.exists() and progress.exists():
                selected = json.loads(selection.read_text())
                job.update(
                    migration="continued",
                    batch_size=selected["batch_size"],
                    source_exposure=selected["source_exposure"],
                )
                self.reservations[arm] = math.ceil(selected["pressure_peak_reserved_mib"]) + 6144
                self.save()
        rows, blocked = L.snapshot()
        for base_lane in self.plan["lanes"]:
            gpu = base_lane["gpu"]
            if rows[gpu]["uuid"] != base_lane["gpu_uuid"]:
                raise ValueError("GPU mapping changed")
            peers = [a for a in running if self.state["jobs"][a]["gpu"] == gpu]
            known = {f"lm-memory:/{self.state['jobs'][a]['container']}" for a in peers}
            known.update(f"active CUDA process {running[a]['Pid']}" for a in peers)
            if any(x not in known for x in blocked.get(gpu, [])):
                continue
            if any(self.state["jobs"][a].get("migration") == "benchmarking" for a in peers):
                continue
            # Prefer an arm whose newly committed checkpoint is already available.
            eligible = []
            for arm in peers:
                if self.state["jobs"][arm].get("migration") != "waiting_new_checkpoint":
                    continue
                checkpoint = self.batch / "runs" / arm / "resume.pt"
                if (
                    checkpoint.exists()
                    and checkpoint.stat().st_mtime_ns > self.amendment["checkpoint_mtimes"][arm]
                ):
                    eligible.append(arm)
            if eligible:
                arm = eligible[0]
                job = self.state["jobs"][arm]
                before = json.loads((self.batch / "runs" / arm / "progress.json").read_text())
                original = dict(job)
                L.write(
                    self.root / "handoff" / f"{arm}-before.json",
                    {
                        "job": original,
                        "progress": before,
                        "container": running[arm],
                        "utc": L.utc(),
                    },
                )
                L.call([*L.DOCKER, "stop", "--time", "120", job["container"]], timeout=180)
                stopped = A.inspect(job["container"])
                if stopped["Running"]:
                    raise ValueError("Old training did not stop; never launch a duplicate")
                L.write(
                    self.root / "handoff" / f"{arm}-stopped.json",
                    {
                        "container": stopped,
                        "reason": "user-authorized continuation with larger batches",
                    },
                )
                job.update(state="migrating", original_container=original["container"])
                self.save()
                other = [a for a in peers if a != arm]
                # Refresh memory after releasing the old process's allocation.
                current, _ = L.snapshot()
                total = current[gpu]["used"] + current[gpu]["free"]
                cap = memory_allowance(
                    total, current[gpu]["free"], sum(self.reservations[a] for a in other)
                )
                if cap < self.amendment["old_reservations"][arm] - 6144:
                    raise ValueError("Insufficient verified memory even for the original batch")
                lane = dict(base_lane, cpus=original.get("cpus", base_lane["cpus"]))
                self.start_continuation(
                    arm, lane, cap, self.batch / "runs" / arm / "resume.pt", before["step"]
                )
                return
            if len(peers) >= 2:
                continue
            waiting = [a for a in F.ARMS if self.state["jobs"][a]["state"] == "waiting_resource"]
            if not waiting:
                continue
            arm = waiting[0]
            if self.state["jobs"][arm]["stage"] != "train" and peers:
                # Independent reload and long-context judge stages use an idle card.
                continue
            cap = memory_allowance(
                rows[gpu]["used"] + rows[gpu]["free"],
                rows[gpu]["free"],
                sum(self.reservations[a] for a in peers),
            )
            if cap < self.amendment["old_reservations"][arm] - 6144:
                continue
            occupied = {self.state["jobs"][a].get("cpus", base_lane["cpus"]) for a in peers}
            lane = dict(
                base_lane,
                cpus=A.NEW_CPUS[gpu] if base_lane["cpus"] in occupied else base_lane["cpus"],
            )
            if self.state["jobs"][arm]["stage"] == "train":
                self.start_continuation(arm, lane, cap)
            else:
                self.start_evaluation(arm, lane)
            return

    def run(self):
        self.save()
        while True:
            self.tick()
            states = [self.state["jobs"][a]["state"] for a in F.ARMS]
            if all(x == "complete" for x in states):
                self.state.update(state="complete", finished_utc=L.utc())
                L.write(self.batch / "controller-state.json", self.state)
                return
            if all(x in ("complete", "needs_attention") for x in states):
                raise RuntimeError("Some arms failed; preserve failures without automatic restart")
            time.sleep(10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    batch = Path(plan["output"])
    root = batch / "batchsize-20261022"
    amendment = json.loads((root / "amendment.json").read_text())
    if socket.gethostname() != plan["host"] or not amendment["engineering_passed"]:
        raise ValueError("Host or engineering gate differs")
    A.verify_sources(batch, root / "extension-v1")
    leases = []
    try:
        for path in [
            batch / "controller.lock",
            *[
                Path(f"/tmp/lm-memory-gpu-locks-{os.getuid()}/gpu-{v['gpu']}.lock")
                for v in plan["lanes"]
            ],
        ]:
            lease = path.open("a")
            leases.append(lease)
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        process = {
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "utc": L.utc(),
            "scheduler": "batchsize-20261022",
            "plan": plan,
        }
        L.write(batch / "controller-process.json", process)
        L.write(root / "controller-process.json", process)
        Controller(plan, amendment).run()
    except Exception:
        import traceback

        L.write(
            root / "controller-failure.json", {"utc": L.utc(), "traceback": traceback.format_exc()}
        )
        raise
    finally:
        for lease in leases:
            lease.close()


if __name__ == "__main__":
    main()
