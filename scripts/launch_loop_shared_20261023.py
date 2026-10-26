"""Resume the frozen Loop whitelist with explicitly authorized GPU sharing."""

import argparse
import concurrent.futures
import copy
import fcntl
import io
import json
import os
import socket
import subprocess
import sys
import tarfile
import threading
import time
from pathlib import Path

import lm_resource_snapshot as resources

PENDING = {
    "many-to-one-loop",
    "many-to-one-discoloop",
    "init-1-ema",
    "init-1-loop",
    "init-1-discoloop",
    "init-2-invariant",
    "init-2-ema",
    "init-2-loop",
    "init-2-discoloop",
}


def read(path):
    return json.loads(Path(path).read_text())


def latest_snapshot(lock, path, writer):
    with lock:
        snap = resources.snapshot()
        writer(path, snap)
        return snap


def shared_blockers(snap, gpu, plan, ignored_created, own_id=None):
    """Sharing permits recorded peers; new reservations and memory pressure still block."""
    row = snap["rows"][gpu]
    reasons = []
    if row["uuid"] != plan["gpu_uuids"][str(gpu)]:
        reasons.append("GPU identity changed")
    if row["free"] < plan["required_free_mib"][str(gpu)]:
        reasons.append("insufficient memory for allocator cap plus CUDA/headroom allowance")
    allowed = set(plan["baseline_processes"].get(row["uuid"], []))
    for process in snap["process_details"]:
        if process["uuid"] == row["uuid"] and process["pid"] not in allowed:
            reasons.append(f"new CUDA process {process['pid']}")
    for c in snap["containers"]:
        if not any(
            str(gpu) in (r.get("DeviceIDs") or []) or row["uuid"] in (r.get("DeviceIDs") or [])
            for r in c["requests"] or []
        ):
            continue
        if c["id"] == own_id or resources.released_created(c, ignored_created):
            continue
        if c["id"] in plan["baseline_running_ids"] and c["state"]["Status"] == "running":
            continue
        reasons.append(f"new or changed reservation {c['context']}:{c['name']}")
    return reasons


def archive_code(source, parent, runtime):
    buffer = io.BytesIO()

    def immutable(item):
        if "__pycache__" in Path(item.name).parts:
            return None
        item.uid = item.gid = 0
        item.uname = item.gname = "root"
        item.mode = 0o555 if item.isdir() else 0o444
        return item

    with tarfile.open(fileobj=buffer, mode="w", dereference=True) as archive:
        archive.add(source, arcname="project", filter=immutable)
        archive.add(parent, arcname="loop-parent-source", filter=immutable)
        control = tarfile.TarInfo("control")
        control.type, control.mode = tarfile.DIRTYPE, 0o555
        control.uid = control.gid = 0
        archive.addfile(control)
        archive.add(runtime, arcname="control/loop_shared_runtime.py", filter=immutable)
    return buffer.getvalue()


def build_controller(base, recovery, plan, token):
    sys.path.insert(0, str(base / "control"))
    import resume_loop_transfer as previous

    original = previous.original
    migration = read(base / "gpu-migration-20261022-13/amendment.json")

    class Controller(previous.Controller):
        def __init__(self):
            original.Controller.__init__(self, base, token)
            self.amendment_path = recovery / "amendment.json"
            self.amendment = {"maximum_local_user_gpus": 4}
            self.plan["gpus"] = plan["gpus"]
            self.plan["cpu_sets"] = plan["cpu_sets"]
            self.plan["gpu_uuids"].update(plan["gpu_uuids"])
            self.plan["ignored_created_ids"] = plan["baseline_created_ids"]
            early = self.plan["phases"][0]
            first = ["many-to-one-loop", "many-to-one-discoloop"]
            self.plan["phases"][0] = first + [j for j in early if j not in first]
            self.state = copy.deepcopy(read(base / "controller-state.json"))
            self.checked = set()
            self.archive_cache = {}
            self.archive_mutex = threading.Lock()

        def existing_container(self, job, stage):
            if stage != "train" or job not in migration["migrations"]:
                return super().existing_container(job, stage)
            expected = migration["migrations"][job]
            receipt = read(base / "execution" / job / stage / "container.json")
            if receipt["id"] != expected["new_container_id"]:
                raise ValueError("Migrated container receipt differs")
            c = previous.inspect_container(receipt["id"])
            if (
                c["Name"] != "/" + expected["new_container_name"]
                or c["Image"] != self.plan["image"]
            ):
                raise ValueError("Migrated training identity changed")
            argv = c["Config"]["Cmd"]
            if argv[argv.index("--job") + 1] != job or "train" not in argv:
                raise ValueError("Migrated command changed")
            return c

        def admit_shared(self, gpu, label, own_id=None):
            while True:
                snap = original.snapshot()
                reasons = shared_blockers(snap, gpu, plan, plan["baseline_created_ids"], own_id)
                if not reasons:
                    original.write(recovery / "admission" / f"{label}.json", snap)
                    return
                self.update(waiting_gpu=gpu, waiting_reasons=reasons)
                time.sleep(10)

        def execute(self, job, stage, lane, preflight=False):
            gpu = self.plan["gpus"][lane]
            engineering = stage == "shared-environment"
            task = self.tasks["many-to-one-loop" if engineering else job]
            output = base / "execution" / job / stage
            if output.exists() and not any(output.iterdir()):
                retained = recovery / "empty-admission-directories" / job / stage
                retained.parent.mkdir(parents=True, exist_ok=True)
                output.rename(retained)
            output.mkdir(parents=True, exist_ok=False)
            label = f"lqs-1023-{job}-{stage}" if engineering else f"lqt-1021-{job}-{stage}"
            self.admit_shared(gpu, f"{job}-{stage}-before-create")
            source = Path(task["source"])
            with self.archive_mutex:
                if source not in self.archive_cache:
                    self.archive_cache[source] = archive_code(
                        source,
                        self.plan["parent_source"],
                        recovery / "loop_shared_runtime.py",
                    )
                archive = self.archive_cache[source]
            argv = [
                *original.DOCKER,
                "create",
                "--name",
                label,
                "--user",
                "0:0",
                "--cpuset-cpus",
                self.plan["cpu_sets"][lane],
                "--cpus",
                "2",
                "--memory",
                "16g",
                "--memory-swap",
                "16g",
                "--shm-size",
                "2g",
                "--tmpfs",
                "/tmp:rw,size=4g",
                "--network",
                "host",
                "--workdir",
                "/project",
                "--gpus",
                f"device={self.plan['gpu_uuids'][str(gpu)]}",
            ]
            env = {
                "PYTHONPATH": "/project/src:/project/scripts",
                "PYTHONDONTWRITEBYTECODE": "1",
                "OMP_NUM_THREADS": "2",
                "MKL_NUM_THREADS": "2",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "XDG_CACHE_HOME": "/tmp/cache",
                "WANDB_CACHE_DIR": "/tmp/wandb-cache",
                "WANDB_CONSOLE": "off",
                "WANDB_SILENT": "true",
                "WANDB_DISABLE_CODE": "true",
                "WANDB_DOCKER": self.plan["image"],
                "LOOP_PARENT_SOURCE": "/loop-parent-source",
                "LM_LOOP_UID": str(os.getuid()),
                "LM_LOOP_GID": str(os.getgid()),
            }
            for key, value in env.items():
                argv += ["--env", f"{key}={value}"]
            for src, dst, mode in [
                (
                    self.plan["persistent_root"],
                    self.plan["persistent_root"],
                    ",readonly",
                ),
                (base, "/output", ""),
            ]:
                argv += ["--mount", f"type=bind,source={src},target={dst}{mode}"]
            profile = f"/output/execution/{job}/{stage}/cuda-peak.json"
            argv += [
                self.plan["image"],
                "python",
                "-u",
                "/control/loop_shared_runtime.py",
                profile,
            ]
            if engineering:
                argv += ["preflight"]
            else:
                argv += [
                    stage,
                    "--plan",
                    str(Path(task["plan"]).relative_to(source)),
                    "--job",
                    job,
                    "--base",
                    "/output",
                ]
            original.write(output / "command.json", {"argv": argv, "utc": original.utc()})
            cid = original.command(argv, text=True, timeout=120).stdout.strip()
            original.write(output / "container.json", {"id": cid, "name": label})
            original.command([*original.DOCKER, "cp", "--archive", "-", f"{cid}:/"], input=archive)
            if stage in ("train", "cloud") or engineering:
                original.command(
                    [*original.DOCKER, "cp", "--archive", "-", f"{cid}:/"],
                    input=original.credential_archive(token),
                    timeout=120,
                )
            self.admit_shared(gpu, f"{job}-{stage}-before-start", cid)
            original.command([*original.DOCKER, "start", cid], timeout=120)
            if not engineering:
                self.update(job, state="running", stage=stage, gpu=gpu, container=label)
            with (output / "stdout.log").open("w") as log:
                follower = subprocess.Popen(
                    [*original.DOCKER, "logs", "--follow", cid],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
                result = original.command([*original.DOCKER, "wait", cid], text=True)
                follower.wait(timeout=60)
            code = int(result.stdout.strip())
            original.write(output / "exit.json", {"code": code, "utc": original.utc()})
            if code:
                raise RuntimeError(f"{label} exited {code}; retained, no automatic retry")

    controller = Controller()
    return controller, previous, original, migration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--recovery", type=Path, required=True)
    parser.add_argument("--wandb-token-file", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    base, recovery = args.base.resolve(), args.recovery.resolve()
    plan = read(recovery / "plan.json")
    if socket.gethostname() != plan["host"] or len(plan["gpus"]) != 4:
        raise ValueError("Host or four-GPU plan differs")
    controller, previous, original, migration = build_controller(
        base, recovery, plan, args.wandb_token_file
    )
    for name, expected in migration["original_sha256"].items():
        if original.digest(base / name) != expected:
            raise ValueError(f"Frozen controller/config changed: {name}")
    for path, expected in controller.plan["frozen_files"].items():
        if original.digest(path) != expected:
            raise ValueError(f"Frozen source changed: {path}")
    for task in controller.tasks.values():
        if (
            task["kind"] == "new"
            and original.digest(Path(task["source"]) / task["config"]) != task["config_sha256"]
        ):
            raise ValueError("Frozen scientific configuration changed")
    pending = {
        j
        for j, t in controller.tasks.items()
        if t["kind"] == "new" and not (base / "execution" / j / "train").exists()
    }
    if pending != PENDING:
        raise ValueError(f"Nine-task whitelist changed: {pending}")
    original.credential_archive(args.wandb_token_file)
    if not args.execute:
        print(json.dumps({"passed": True, "pending": sorted(pending), "gpus": plan["gpus"]}))
        return
    resources.verify_code_bundle(recovery)
    handles = []
    try:
        paths = [
            base / "controller.lock",
            *map(Path, controller.plan["source_controller_locks"]),
        ]
        paths += [
            Path(f"/tmp/lm-memory-gpu-locks-{os.getuid()}/gpu-{g}.lock") for g in plan["gpus"]
        ]
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = path.open("a+")
            handles.append(handle)
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for path in [
            recovery / "controller-process.json",
            base / "recovery-20261023/controller-process.json",
        ]:
            if path.exists():
                raise FileExistsError(f"Recovery already submitted: {path}")
        for job, task in controller.tasks.items():
            for stage in task["stages"]:
                action = previous.stage_action(base / "execution" / job / stage)
                if action == "unstarted":
                    continue
                c = controller.existing_container(job, stage)
                s = c["State"]
                if s["Status"] != "exited" or s["ExitCode"] != 0 or s["OOMKilled"]:
                    raise RuntimeError(f"Existing stage not successfully exited: {job}/{stage}")
        original.write(recovery / "before-state.json", controller.state)
        controller.state.pop("error", None)
        for job in controller.tasks:
            controller.state["jobs"][job]["state"] = "recovering"

        snapshot_lock = threading.Lock()

        def guarded_snapshot():
            while True:
                try:
                    return latest_snapshot(
                        snapshot_lock, recovery / "latest-resources.json", original.write
                    )
                except resources.ResourceUnavailable as exc:
                    controller.update(
                        resource_wait=str(exc), scheduler_heartbeat_utc=original.utc()
                    )
                    time.sleep(10)

        original.snapshot = guarded_snapshot
        original.blockers = lambda snap, gpu, ignored: shared_blockers(snap, gpu, plan, ignored)
        process = dict(
            pid=os.getpid(),
            host=socket.gethostname(),
            utc=original.utc(),
            scheduler=recovery.name,
            controller_sha256=original.digest(__file__),
        )
        original.write(recovery / "controller-process.json", process)
        original.write(base / "controller-process.json", process)
        original.write(
            recovery / "before-schedule-amendment.json", read(base / "schedule-amendment.json")
        )
        original.write(
            base / "schedule-amendment.json",
            dict(
                latest=str(recovery / "amendment.json"),
                scheduler=recovery.name,
                effective_gpu_pool=plan["gpus"],
                maximum_local_user_gpus=4,
                shared_gpu_authorized=True,
                scientific_changes=False,
            ),
        )
        original.write(
            recovery / "amendment.json",
            dict(
                authorization=(
                    "2026-10-23 user explicitly authorizes four shared GPUs "
                    "for the named nine Loop jobs"
                ),
                scientific_changes=False,
                steps_per_job=1500000,
                pending_training=sorted(PENDING),
                initialization_comparisons_last=True,
                maximum_local_user_gpus=4,
                scheduling_plan=plan,
                code_delivery="create-copy-start; root-owned0444/0555",
                reservation_records=(
                    "preserved; baseline never-started records ignored only while unchanged"
                ),
                source_followups="completed source-host models stay deferred; never retrain",
                performance="shared GPU wall time is not exclusive GPU performance",
            ),
        )
        controller.update(
            state="preflight",
            scheduler=recovery.name,
            available_gpu_pool=plan["gpus"],
            active_gpus=[],
            environment_passed=False,
            scheduler_heartbeat_utc=original.utc(),
        )
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                checks = []
                for i, gpu in enumerate(plan["gpus"]):
                    directory = base / "execution" / f"shared-gpu{gpu}" / "shared-environment"
                    if (directory / "exit.json").exists():
                        actual = previous.inspect_container(
                            read(directory / "container.json")["id"]
                        )
                        if (
                            read(directory / "exit.json")["code"] != 0
                            or actual["State"]["Status"] != "exited"
                            or actual["State"]["ExitCode"] != 0
                            or actual["State"]["OOMKilled"]
                            or read(directory / "cuda-peak.json")["stage"] != "preflight"
                        ):
                            raise RuntimeError("Existing engineering result failed or ambiguous")
                        continue
                    checks.append(
                        pool.submit(controller.execute, f"shared-gpu{gpu}", "shared-environment", i)
                    )
                for check in checks:
                    check.result()
                controller.update(state="running", environment_passed=True)
                for index, phase in enumerate(controller.plan["phases"]):
                    controller.update(phase=index)
                    controller.run_phase(phase, pool)
            controller.update(state="complete")
        except Exception as exc:
            controller.update(state="failed", error=str(exc))
            raise
    finally:
        for handle in handles:
            handle.close()


if __name__ == "__main__":
    os.umask(0o022)
    main()
