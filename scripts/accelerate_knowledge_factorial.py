#!/usr/bin/env python3
"""Adopt existing containers and admit at most two checked arms per original GPU."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import io
import json
import os
import socket
import subprocess
import tarfile
import time
from pathlib import Path

import launch_knowledge_method_factorial as F

L = F.L
STAGES = ("train", "audit", "evaluate")
NEW_CPUS = {3: "0-9", 6: "24-33", 8: "80-89"}
MARGIN_MIB = 8192


def inspect(name):
    value = L.call(
        [*L.DOCKER, "inspect", "--format", "{{json .Id}} {{json .State}}", name],
        timeout=30,
    ).stdout
    identity, end = json.JSONDecoder().raw_decode(value)
    return {"id": identity, **json.loads(value[end:].strip())}


def admission(row, blockers, known_names, known_pids, reservations, required):
    """Accept only this controller's existing jobs, never another project's GPU."""
    if not row or len(reservations) >= 2:
        return False
    allowed = {f"lm-memory:/{name}" for name in known_names}
    allowed.update(f"active CUDA process {pid}" for pid in known_pids)
    if any(reason not in allowed for reason in blockers):
        return False
    total = row["used"] + row["free"]
    return (
        row["free"] >= required + MARGIN_MIB and sum(reservations) + required + MARGIN_MIB <= total
    )


def next_stage(stage):
    index = STAGES.index(stage) + 1
    return STAGES[index] if index < len(STAGES) else None


def extension_archive(extension):
    """Copy verified bytes across root-squashed storage without exposing its parent."""
    manifest = json.loads((extension / "manifest.json").read_text())
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        directory = tarfile.TarInfo("speedup")
        directory.type, directory.mode = tarfile.DIRTYPE, 0o555
        archive.addfile(directory)
        for name in ["manifest.json", *manifest["sha256"]]:
            if Path(name).name != name:
                raise ValueError("Extension entries must be simple filenames")
            data = (extension / name).read_bytes()
            if (
                name != "manifest.json"
                and hashlib.sha256(data).hexdigest() != manifest["sha256"][name]
            ):
                raise ValueError(f"Extension changed before copy: {name}")
            info = tarfile.TarInfo(f"speedup/{name}")
            info.size, info.mode = len(data), 0o444
            archive.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def verify_sources(batch, extension):
    manifests = [
        (
            batch / "source",
            json.loads((batch / "source-manifest.json").read_text())["source_sha256"],
        ),
        (extension, json.loads((extension / "manifest.json").read_text())["sha256"]),
    ]
    checked = 0
    for base, manifest in manifests:
        if not manifest:
            raise ValueError("Empty source manifest")
        for name, expected in manifest.items():
            if hashlib.sha256((base / name).read_bytes()).hexdigest() != expected:
                raise ValueError(f"Frozen source mismatch: {name}")
            checked += 1
    return checked


class Controller:
    def __init__(self, plan, amendment):
        self.plan = plan
        self.batch = Path(plan["output"])
        self.amendment = amendment
        self.extension = self.batch / amendment["extension_relative"]
        self.state = json.loads((self.batch / "controller-state.json").read_text())
        self.reservations = amendment["memory_reservation_mib"]

    def save(self):
        self.state["updated_utc"] = L.utc()
        self.state["scheduler"] = "speedup-20261022"
        L.write(self.batch / "controller-state.json", self.state)

    def adopt(self):
        for arm, expected in self.amendment["preserved_containers"].items():
            job = self.state["jobs"][arm]
            actual = inspect(job["container"])
            if (
                actual["id"] != expected["id"]
                or actual["Pid"] != expected["Pid"]
                or actual["StartedAt"] != expected["StartedAt"]
                or not actual["Running"]
            ):
                raise ValueError(f"Existing model identity changed: {arm}")
            job["container_id"] = actual["id"]
        for arm in F.ARMS:
            job = self.state["jobs"][arm]
            if job["state"] == "pending":
                if (self.batch / "execution" / arm).exists():
                    raise ValueError(f"Pending arm has an existing execution: {arm}")
                job.update(state="waiting_resource", stage="train")
        self.save()

    def finish(self, arm, job, actual):
        if actual["Status"] != "exited":
            raise ValueError(f"Unexpected container state for {arm}: {actual['Status']}")
        stage = job["stage"]
        folder = self.batch / "execution" / arm / stage
        L.write(
            folder / "exit.json",
            {"code": actual["ExitCode"], "utc": L.utc(), "scheduler": "speedup-20261022"},
        )
        # The adopted stdout followers may still own their original logs.
        with (folder / "speedup-final-stdout.log").open("x") as handle:
            subprocess.run(
                [*L.DOCKER, "logs", job["container"]],
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=True,
            )
        status_path = self.batch / "stage-status" / f"{arm}-{stage}.json"
        status = json.loads(status_path.read_text()) if status_path.exists() else {}
        if actual["ExitCode"] != 0 or status.get("state") != "complete":
            job.update(state="needs_attention", error=f"{stage} did not finish successfully")
        else:
            if stage == "train":
                completed = json.loads((self.batch / "runs" / arm / "complete.json").read_text())
                if completed.get("step") != 372300 or completed.get("smoke") is not False:
                    raise ValueError(f"Training endpoint differs for {arm}")
            job.setdefault("stages", []).append(stage)
            following = next_stage(stage)
            if following is None:
                job.update(state="complete", finished_utc=L.utc())
            else:
                job.update(state="waiting_resource", stage=following)
        self.save()

    def start(self, arm, lane):
        job = self.state["jobs"][arm]
        stage = job["stage"]
        folder = self.batch / "execution" / arm / stage
        if folder.exists():
            raise FileExistsError(f"Existing attempt must not be retried: {folder}")
        name, argv = F.command(self.plan, stage, arm, lane)
        if L.call(
            [*L.DOCKER, "ps", "-a", "--filter", f"name=^/{name}$", "-q"], timeout=30
        ).stdout.strip():
            raise FileExistsError(f"Existing container must not be restarted: {name}")
        if stage == "train":
            index = argv.index("scripts/run_knowledge_method_factorial.py")
            argv[index] = "/speedup/run_knowledge_method_speedup.py"
        L.write(folder / "command.json", {"argv": argv, "utc": L.utc()})
        try:
            L.call(argv, timeout=120)
        except subprocess.CalledProcessError as error:
            L.write(
                folder / "create-failure.json",
                {"code": error.returncode, "stderr": error.stderr, "utc": L.utc()},
            )
            raise
        if stage == "train":
            subprocess.run(
                [*L.DOCKER, "cp", "-", f"{name}:/"],
                input=extension_archive(self.extension),
                check=True,
                capture_output=True,
                timeout=120,
            )
        subprocess.run(
            [*L.DOCKER, "cp", "--archive", "-", f"{name}:/"],
            input=L.credential_archive(self.plan["wandb_token_file"]),
            check=True,
            capture_output=True,
            timeout=120,
        )
        L.call([*L.DOCKER, "start", name], timeout=120)
        actual = inspect(name)
        job.update(
            state="running",
            container=name,
            container_id=actual["id"],
            gpu=lane["gpu"],
            gpu_uuid=lane["gpu_uuid"],
            cpus=lane["cpus"],
            started_utc=L.utc(),
            checkpoint_transport="async_32MiB_per_second" if stage == "train" else "original",
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
            actual = inspect(job["container"])
            if actual["id"] != job["container_id"]:
                raise ValueError(f"Container replaced: {arm}")
            if actual["Running"]:
                running[arm] = actual
            else:
                self.finish(arm, job, actual)
        rows, blocked = L.snapshot()
        for arm in F.ARMS:
            job = self.state["jobs"][arm]
            if job["state"] != "waiting_resource":
                continue
            for base_lane in self.plan["lanes"]:
                gpu = base_lane["gpu"]
                if rows[gpu]["uuid"] != base_lane["gpu_uuid"]:
                    raise ValueError("Physical GPU mapping changed")
                peers = [a for a in running if self.state["jobs"][a]["gpu"] == gpu]
                names = [self.state["jobs"][a]["container"] for a in peers]
                pids = [running[a]["Pid"] for a in peers]
                budgets = [self.reservations[a] for a in peers]
                if not admission(
                    rows[gpu], blocked.get(gpu, []), names, pids, budgets, self.reservations[arm]
                ):
                    continue
                occupied_cpus = {
                    self.state["jobs"][a].get("cpus", base_lane["cpus"]) for a in peers
                }
                cpus = NEW_CPUS[gpu] if base_lane["cpus"] in occupied_cpus else base_lane["cpus"]
                self.start(arm, dict(base_lane, cpus=cpus))
                # Snapshot again before another admission, including newly created CUDA jobs.
                return

    def run(self):
        self.adopt()
        deadline = time.monotonic() + self.plan["runtime"]["admission_timeout_hours"] * 3600
        while time.monotonic() < deadline:
            self.tick()
            states = [self.state["jobs"][arm]["state"] for arm in F.ARMS]
            if all(state == "complete" for state in states):
                self.state.update(state="complete", finished_utc=L.utc())
                self.save()
                L.write(self.batch / "controller-exit-speedup.json", self.state)
                return 0
            if (
                "needs_attention" in states
                and "running" not in states
                and "waiting_resource" not in states
            ):
                raise RuntimeError("One or more stages failed; no automatic retries")
            time.sleep(10)
        raise TimeoutError("Admission deadline reached")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    batch = Path(plan["output"])
    amendment = json.loads((batch / "speedup-20261022/amendment.json").read_text())
    if socket.gethostname() != plan["host"] or plan != amendment["original_plan"]:
        raise ValueError("The frozen launch plan or host differs")
    count = verify_sources(batch, batch / amendment["extension_relative"])
    if not amendment.get("engineering_passed"):
        raise ValueError("Engineering checks must pass before admission")
    if not args.execute:
        print(
            json.dumps(
                {
                    "source_files_verified": count,
                    "memory_reservation_mib": amendment["memory_reservation_mib"],
                }
            )
        )
        return 0
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
        record = {
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "utc": L.utc(),
            "source_files_verified": count,
            "plan": plan,
            "scheduler": "speedup-20261022",
        }
        L.write(batch / "controller-process.json", record)
        L.write(batch / "controller-process-speedup.json", record)
        return Controller(plan, amendment).run()
    except Exception as error:
        L.write(
            batch / "speedup-20261022/controller-failure.json",
            {"error": repr(error), "utc": L.utc()},
        )
        raise
    finally:
        for lease in leases:
            lease.close()


if __name__ == "__main__":
    raise SystemExit(main())
