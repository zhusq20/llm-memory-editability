#!/usr/bin/env python3
"""Adopt unchanged evaluations; admit audits on idle GPUs or approved bounded sharing."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path("/ossfs/workspace/llm-memory-editability/evaluation-recovery-20261016")
AMENDMENT = ROOT / "approved-gpu7-audit-repair-20261016"
FRACTION = 0.06
MIN_SHARED_FREE_MIB = 32768


def process_identity(pid):
    directory = Path("/proc") / str(pid)
    try:
        stat = (directory / "stat").read_text().rsplit(")", 1)[1].split()
        return {"pid": pid, "uid": directory.stat().st_uid, "start_ticks": int(stat[19])}
    except FileNotFoundError:
        return None


def filtered_snapshot(snapshot, neighbors, identity=process_identity):
    rows, processes = snapshot
    kept = []
    for process in processes:
        pid = process["pid"]
        if pid not in neighbors or identity(pid) != neighbors[pid]:
            kept.append(process)
    return rows, kept


def idle_candidates(rows, processes):
    occupied = {p["uuid"] for p in processes}
    return [
        gpu
        for gpu, row in sorted(rows.items())
        if gpu != 7
        and row["uuid"] not in occupied
        and row["memory"] <= 1024
        and row["utilization"] <= 5
        and row["free_mib"] >= 12000
    ]


def bounded_command(command):
    command = list(command)
    index = command.index("python")
    script = command[index + 1]
    if script not in (
        "/frozen/scripts/run_ttcd_one_step.py",
        "/frozen/scripts/run_ttcd_merge_hops.py",
    ):
        raise ValueError("Only frozen evaluation entrypoints may be memory bounded")
    bootstrap = (
        "import torch,runpy,sys,os;"
        f"torch.cuda.set_per_process_memory_fraction({FRACTION});"
        "sys.argv=sys.argv[1:];sys.path.insert(0,os.path.dirname(sys.argv[0]));"
        "runpy.run_path(sys.argv[0],run_name='__main__')"
    )
    command[index + 1 : index + 1] = ["-c", bootstrap]
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-existing-neighbors", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((AMENDMENT / "source-manifest.json").read_text())
    predecessor = Path(manifest["predecessor"])
    if hashlib.sha256(predecessor.read_bytes()).hexdigest() != manifest["predecessor_sha256"]:
        raise ValueError("The preserved publication controller changed")
    neighbor_path = AMENDMENT / "foreign-processes.json"
    if hashlib.sha256(neighbor_path.read_bytes()).hexdigest() != manifest["neighbors_sha256"]:
        raise ValueError("The reviewed neighbor snapshot changed")
    neighbors = {p["pid"]: p for p in json.loads(neighbor_path.read_text())["processes"]}
    spec = importlib.util.spec_from_file_location("preserved_publication_controller", predecessor)
    sharing = importlib.util.module_from_spec(spec)
    sys.dont_write_bytecode = True
    spec.loader.exec_module(sharing)
    strict_verify = sharing.verify_cohort

    def verify_existing(helpers, controllers, plans):
        # Adoption continues the same containers even if another user joined later.
        snapshot = filtered_snapshot(helpers.gpu_snapshot(), neighbors)
        proxy = SimpleNamespace(**vars(helpers))
        proxy.gpu_snapshot = lambda: snapshot
        return strict_verify(proxy, controllers, plans)

    def admission(base, controllers, plans, plan, arm, stage, update, lock_directory=None):
        helpers = next(iter(controllers.values())).resources
        runtime = plan["runtime"]
        directory = lock_directory or Path("/tmp") / f"lm-memory-gpu-locks-{os.getuid()}"
        directory.mkdir(exist_ok=True)
        deadline = time.monotonic() + runtime["wait_timeout_hours"] * 3600
        while time.monotonic() < deadline:
            held = contextlib.ExitStack()
            try:
                for cpu in runtime["cpu_sets"][arm]:
                    lock = held.enter_context((directory / f"cpu-{cpu}.lock").open("a"))
                    try:
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError as error:
                        raise helpers.ResourceUnavailable("Exclusive CPU lock held") from error
                helpers.admit_cpu(plan, arm)
                if stage in ("prepare", "cloud-audit", "aggregate"):
                    return held, {"device": "cpu", "cpu_set": runtime["cpu_sets"][arm]}
                snapshot = helpers.gpu_snapshot()
                proxy = SimpleNamespace(**vars(helpers))
                proxy.gpu_snapshot = lambda snapshot=snapshot: (
                    filtered_snapshot(snapshot, neighbors)
                    if args.allow_existing_neighbors
                    else snapshot
                )
                try:
                    row, own = strict_verify(proxy, controllers, plans)
                    if len(own) >= 3 or (
                        args.allow_existing_neighbors and row["free_mib"] < MIN_SHARED_FREE_MIB
                    ):
                        raise helpers.ResourceUnavailable("Bounded shared GPU headroom unavailable")
                    return held, {
                        **row,
                        "gpu": 7,
                        "cpu_set": runtime["cpu_sets"][arm],
                        "gpu_sharing": True,
                        "cohort": own,
                        "registered_neighbors_allowed": args.allow_existing_neighbors,
                        "allocator_fraction": FRACTION,
                    }
                except helpers.ResourceUnavailable:
                    pass
                for gpu in idle_candidates(*snapshot):
                    gpu_locks = contextlib.ExitStack()
                    lock = gpu_locks.enter_context((directory / f"gpu-{gpu}.lock").open("a"))
                    try:
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        gpu_locks.close()
                        continue
                    fresh = helpers.gpu_snapshot()
                    if gpu not in idle_candidates(*fresh):
                        gpu_locks.close()
                        continue
                    held.enter_context(gpu_locks)
                    return held, {
                        **fresh[0][gpu],
                        "gpu": gpu,
                        "cpu_set": runtime["cpu_sets"][arm],
                        "gpu_sharing": False,
                        "allocator_fraction": FRACTION,
                    }
                raise helpers.ResourceUnavailable(
                    "Awaiting an idle GPU or approved bounded GPU7 use"
                )
            except helpers.ResourceUnavailable as error:
                held.close()
                update({"state": "waiting_for_resources", "stage": stage, "reason": str(error)})
                time.sleep(5)
            except BaseException:
                held.close()
                raise
        raise TimeoutError("No permitted audit resource became available")

    original_load = sharing.load

    def load():
        base = original_load()
        original_module = base.module

        def module(path, name):
            controller = original_module(path, name)
            build = controller.build_command

            def build_command(plan, arm, stage, batch, token, row):
                label, command = build(plan, arm, stage, batch, token, row)
                return label, bounded_command(command) if "gpu" in row else command

            controller.build_command = build_command
            return controller

        base.module = module
        return base

    sharing.load = load
    sharing.admission = admission
    sharing.verify_cohort = verify_existing
    sharing.HANDOFF = AMENDMENT.name
    sharing.__file__ = __file__
    return sharing.main()


if __name__ == "__main__":
    raise SystemExit(main())
