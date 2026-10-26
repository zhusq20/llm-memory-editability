"""Resume scheduling existing batch-size runs without restarting training."""

import argparse
import fcntl
import hashlib
import json
import math
import os
import socket
import sys
from pathlib import Path

import lm_resource_snapshot as resources


def read(path):
    return json.loads(Path(path).read_text())


def choose_waiting(jobs, arms, peers):
    # A completed arm awaiting an exclusive evaluation must not block a
    # different unstarted training arm that fits beside an existing training.
    candidates = [
        a
        for a in arms
        if jobs[a]["state"] == "waiting_resource" and (not peers or jobs[a]["stage"] == "train")
    ]
    return sorted(candidates, key=lambda a: jobs[a]["stage"] != "train")


def validate_adopted(job, actual):
    if actual["id"] != job["container_id"]:
        raise ValueError("Existing container identity changed")
    if actual["Status"] not in ("running", "exited") or actual.get("OOMKilled"):
        raise ValueError("Ambiguous or failed existing container; no restart")


def build_controller(plan, recovery):
    batch = Path(plan["output"])
    extension = batch / "batchsize-20261022/extension-v1"
    sys.path[:0] = [str(extension), str(batch / "source/scripts"), str(batch / "source/src")]
    import launch_knowledge_batch_resume as old

    A, L, F = old.A, old.L, old.F
    A.verify_sources(batch, extension)

    class Controller(old.Controller):
        def __init__(self):
            self.plan, self.batch = plan, batch
            self.root, self.extension = batch / "batchsize-20261022", extension
            self.amendment = read(self.root / "amendment.json")
            self.state = read(batch / "controller-state.json")
            self.reservations = dict(self.amendment["old_reservations"])
            self.recovery = recovery
            self.checked_running = {}
            for arm in F.ARMS:
                job = self.state["jobs"][arm]
                if job["state"] == "running":
                    actual = A.inspect(job["container"])
                    validate_adopted(job, actual)
                    self.checked_running[arm] = actual
                elif job["state"] == "waiting_resource" and job["stage"] != "train":
                    actual = A.inspect(job["container"])
                    validate_adopted(job, actual)
                    if actual["Running"] or actual["ExitCode"] != 0:
                        raise ValueError(f"Completed predecessor disagrees: {arm}")
                    end = read(batch / "runs" / arm / "complete.json")
                    if end["exposure"] != 1489200 or end["smoke"]:
                        raise ValueError("Training endpoint differs")
                elif job["state"] == "waiting_resource" and job["stage"] == "train":
                    for p in [
                        self.root / "execution" / arm,
                        self.root / "runs" / arm,
                        batch / "runs" / arm / "progress.json",
                    ]:
                        if p.exists():
                            raise ValueError(f"Unstarted arm already has an attempt: {p}")
                elif job["state"] not in ("complete", "needs_attention"):
                    raise ValueError(f"Unsupported state: {arm}: {job}")
                selected = self.root / "runs" / arm / "selection.json"
                if selected.exists():
                    self.reservations[arm] = (
                        math.ceil(read(selected)["pressure_peak_reserved_mib"]) + 6144
                    )

        def save(self):
            self.state.update(updated_utc=L.utc(), scheduler="recovery-20261023", state="training")
            L.write(self.batch / "controller-state.json", self.state)

        def tick(self):
            running = {}
            for arm in F.ARMS:
                job = self.state["jobs"][arm]
                if job["state"] != "running":
                    continue
                actual = A.inspect(job["container"])
                validate_adopted(job, actual)
                if not actual["Running"]:
                    self.finish(arm, job, actual)
                    continue
                running[arm] = actual
                selected = self.root / "runs" / arm / "selection.json"
                if (
                    job.get("migration") == "benchmarking"
                    and selected.exists()
                    and (selected.parent / "progress.json").exists()
                ):
                    s = read(selected)
                    job.update(
                        migration="continued",
                        batch_size=s["batch_size"],
                        source_exposure=s["source_exposure"],
                    )
                    self.reservations[arm] = math.ceil(s["pressure_peak_reserved_mib"]) + 6144
            try:
                snap = resources.snapshot()
            except resources.ResourceUnavailable as exc:
                self.state.update(admission_blocked=str(exc), scheduler_heartbeat_utc=L.utc())
                self.save()
                return
            rows, blocked = resources.knowledge_view(
                snap, L.reserved_devices, plan.get("released_created_ids", [])
            )
            L.write(recovery / "latest-resources.json", snap)
            self.state.pop("admission_blocked", None)
            owned = resources.owned_gpu_uuids(snap)
            self.state.update(
                scheduler_heartbeat_utc=L.utc(),
                gpu_pool=[x["gpu"] for x in plan["lanes"]],
                waiting_reasons={},
            )
            for base_lane in plan["lanes"]:
                gpu = base_lane["gpu"]
                if rows[gpu]["uuid"] != base_lane["gpu_uuid"]:
                    raise ValueError("GPU identity changed")
                peers = [a for a in running if self.state["jobs"][a]["gpu"] == gpu]
                if rows[gpu]["uuid"] not in owned and len(owned) >= 4:
                    self.state["waiting_reasons"][str(gpu)] = ["four user GPUs already occupied"]
                    continue
                known = {f"lm-memory:/{self.state['jobs'][a]['container']}" for a in peers}
                known.update(f"active CUDA process {running[a]['Pid']}" for a in peers)
                foreign = [x for x in blocked.get(gpu, []) if x not in known]
                self.state["waiting_reasons"][str(gpu)] = foreign
                if (
                    foreign
                    or len(peers) >= 2
                    or any(self.state["jobs"][a].get("migration") == "benchmarking" for a in peers)
                ):
                    continue
                # Evaluations are exclusive; never share training with one.
                if any(self.state["jobs"][a]["stage"] != "train" for a in peers):
                    continue
                for arm in choose_waiting(self.state["jobs"], F.ARMS, peers):
                    cap = old.memory_allowance(
                        rows[gpu]["used"] + rows[gpu]["free"],
                        rows[gpu]["free"],
                        sum(self.reservations[a] for a in peers),
                    )
                    if cap < self.amendment["old_reservations"][arm] - 6144:
                        continue
                    occupied = {self.state["jobs"][a].get("cpus") for a in peers}
                    alternate = plan["alternate_cpus"][str(gpu)]
                    lane = dict(
                        base_lane,
                        cpus=alternate if base_lane["cpus"] in occupied else base_lane["cpus"],
                    )
                    if lane["cpus"] in occupied:
                        raise ValueError("CPU lane overlap")
                    if self.state["jobs"][arm]["stage"] == "train":
                        self.start_continuation(arm, lane, cap)
                    else:
                        self.start_evaluation(arm, lane)
                    return
            self.save()

    return Controller(), old


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    plan = read(args.plan)
    batch = Path(plan["output"])
    recovery = args.plan.parent
    if socket.gethostname() != plan["host"] or len(plan["lanes"]) > 4:
        raise ValueError("Host or four-GPU limit differs")
    if len({r["gpu_uuid"] for r in plan["lanes"]}) != len(plan["lanes"]):
        raise ValueError("Repeated GPU")
    if args.execute:
        resources.verify_code_bundle(Path(__file__).resolve().parent)
        if any(r["gpu"] == 4 for r in plan["lanes"]):
            if not read(recovery / "gpu4-preflight/complete.json")["passed"]:
                raise ValueError("GPU4 CUDA preflight has not passed")
    handles = []
    try:
        for p in [
            batch / "controller.lock",
            *[
                Path(f"/tmp/lm-memory-gpu-locks-{os.getuid()}/gpu-{r['gpu']}.lock")
                for r in plan["lanes"]
            ],
        ]:
            p.parent.mkdir(parents=True, exist_ok=True)
            h = p.open("a")
            handles.append(h)
            fcntl.flock(h, fcntl.LOCK_EX | fcntl.LOCK_NB)
        controller, old = build_controller(plan, recovery)
        if not args.execute:
            print(
                json.dumps(
                    {
                        "passed": True,
                        "mode": "read-only",
                        "adopted": list(controller.checked_running),
                        "gpus": [x["gpu"] for x in plan["lanes"]],
                    }
                )
            )
            return
        if (recovery / "controller-process.json").exists():
            raise FileExistsError("Recovery already submitted; preserve this attempt")
        old.L.write(recovery / "before-state.json", controller.state)
        old.L.write(recovery / "adopted-containers.json", controller.checked_running)
        process = dict(
            pid=os.getpid(),
            host=socket.gethostname(),
            utc=old.L.utc(),
            scheduler="recovery-20261023",
            plan=str(args.plan),
            sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        )
        old.L.write(recovery / "controller-process.json", process)
        old.L.write(batch / "controller-process.json", process)
        old.L.write(controller.root / "controller-process.json", process)
        try:
            controller.run()
        except Exception:
            import traceback

            old.L.write(
                recovery / "controller-failure.json",
                dict(utc=old.L.utc(), traceback=traceback.format_exc()),
            )
            raise
    finally:
        for h in handles:
            h.close()


if __name__ == "__main__":
    main()
