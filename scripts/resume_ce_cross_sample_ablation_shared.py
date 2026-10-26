"""Scheduling-only adoption: at most three containers of this batch share GPU7."""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import datetime as dt
import fcntl
import importlib.util
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
ROOT = Path(__file__).resolve().parents[1]
BATCH = "ce-cross-sample-ablation-v1"
SOURCE_SHA = "79a5cf4cf4870527c2d6653e39d86a647ea991ae8bcf4204208f4eb36154ad83"
PLAN_SHA = "67331115d2cd5e3cbbdeae4dde2d9c93c964e5f294467c42a8fa928303d16159"
PARENT_SHA = "68bef3953b12e9be33c65e2794b9159417c855dc0e74a79aa146d1a99e86de4f"
GPU = {"index": 7, "uuid": "GPU-462bb54b-05c2-4014-6be9-70d6afec79d0"}
OLD_PID = 1163349
MAXIMUM = 3


def alive(pid):
    path = Path(f"/proc/{pid}/stat")
    if not path.exists():
        return False
    return path.read_text().rsplit(")", 1)[1].strip().split()[0] != "Z"


def epoch(value):
    if not value or value.startswith("0001-"):
        raise ValueError("A created but never started stage cannot be adopted or started")
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def load_controller(source):
    # The loader may cache the target before its body can disable bytecode writes.
    sys.dont_write_bytecode = True
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    spec = importlib.util.spec_from_file_location(
        "_frozen_shared_ablation", source / "scripts/launch_ce_cross_sample_ablation.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Pool:
    def __init__(self, maximum=MAXIMUM):
        self.semaphore = threading.BoundedSemaphore(maximum)
        self.mutex = threading.Lock()
        self.reserved = set()
        self.active = 0
        self.maximum_observed = 0

    def reserve(self, key):
        if not self.semaphore.acquire(blocking=False):
            raise ValueError("Existing batch containers exceed the authorized shared pool")
        with self.mutex:
            self.reserved.add(key)
            self.active += 1
            self.maximum_observed = max(self.maximum_observed, self.active)

    @contextlib.contextmanager
    def slot(self, key):
        with self.mutex:
            adopted = key in self.reserved
            if adopted:
                self.reserved.remove(key)
        if not adopted:
            self.semaphore.acquire()
            with self.mutex:
                self.active += 1
                self.maximum_observed = max(self.maximum_observed, self.active)
        try:
            yield
        finally:
            with self.mutex:
                self.active -= 1
            self.semaphore.release()


class Handoff:
    def __init__(self, controller, plan, batch, source):
        self.controller, self.transport = controller, controller.transport
        self.plan, self.batch, self.source = plan, batch, source
        self.pool = Pool()
        self.original_run = self.transport.run_stage
        self.admission_mutex = threading.Lock()

    def inspect(self, name):
        return json.loads(self.transport.docker("inspect", "--format", "{{json .}}", name).stdout)

    def container_exists(self, name):
        """Only an explicit inspect not-found response establishes safe absence."""
        try:
            self.inspect(name)
        except subprocess.CalledProcessError as error:
            stderr = error.stderr or b""
            message = stderr.lower() if isinstance(stderr, bytes) else stderr.lower().encode()
            if b"no such object:" in message or b"no such container:" in message:
                return False
            raise
        return True

    def validate_container(self, container, arm, stage, receipt):
        expected_name = f"{BATCH}-{arm}-{stage}"
        config, state = container["Config"], container["State"]
        labels = config.get("Labels", {})
        if (
            container["Id"] != receipt["id"]
            or container["Name"].lstrip("/") != expected_name
            or receipt["name"] != expected_name
            or receipt["source_sha256"] != SOURCE_SHA
            or labels.get("lm.batch") != BATCH
            or labels.get("lm.architecture") != arm
            or labels.get("lm.stage") != stage
            or config.get("User") != "1015:1016"
            or config.get("Image") != self.plan["execution"]["image"]
            or f"LM_SOURCE_MANIFEST_SHA256={SOURCE_SHA}" not in config.get("Env", [])
            or config.get("Cmd") != self.controller.command_for(self.plan, stage, arm)
            or state.get("Paused")
            or state.get("Restarting")
        ):
            raise ValueError("Existing container does not exactly match its frozen batch launch")
        requests = container["HostConfig"].get("DeviceRequests") or []
        gpu = receipt.get("gpu")
        if gpu:
            if gpu["uuid"] != GPU["uuid"] or [r.get("DeviceIDs") for r in requests] != [
                [GPU["uuid"]]
            ]:
                raise ValueError("The shared reservation may address only the original GPU7")
        elif requests:
            raise ValueError("A CPU stage unexpectedly requested a GPU")
        return state

    def owned_running(self):
        identifiers = (
            self.transport.docker("ps", "--filter", f"label=lm.batch={BATCH}", "-q")
            .stdout.decode()
            .split()
        )
        allowed, stages = set(), []
        for identifier in identifiers:
            container = self.inspect(identifier)
            labels = container["Config"].get("Labels", {})
            arm, stage = labels.get("lm.architecture"), labels.get("lm.stage")
            if arm not in (
                *self.controller.ARMS,
                "loop-prepare",
                "discoloop-prepare",
                "contracts",
                "report",
            ):
                raise ValueError("An unrecognized container claimed ownership of this batch")
            path = self.batch / arm / f"launch-{stage}.json"
            receipt = json.loads(path.read_text())
            state = self.validate_container(container, arm, stage, receipt)
            if not state["Running"]:
                continue
            if receipt.get("gpu"):
                try:
                    rows = (
                        self.transport.docker("top", identifier, "-eo", "pid")
                        .stdout.decode()
                        .splitlines()[1:]
                    )
                except subprocess.CalledProcessError:
                    current = self.inspect(identifier)
                    current_state = self.validate_container(current, arm, stage, receipt)
                    if not current_state["Running"]:
                        continue
                    raise
                allowed.add(int(state["Pid"]))
                allowed.update(int(row.strip()) for row in rows)
                stages.append((arm, stage))
        return allowed, stages

    def admission(self, gpu):
        if gpu["uuid"] != GPU["uuid"]:
            return {"passed": False, "reason": "another GPU"}
        allowed, stages = self.owned_running()
        actual = set(gpu["compute_pids"])
        return {
            "passed": actual <= allowed and gpu["free_mib"] >= 16384 and len(stages) <= MAXIMUM,
            "actual_compute_pids": sorted(actual),
            "allowed_batch_pids": sorted(allowed),
            "running_batch_stages": stages,
            "free_mib": gpu["free_mib"],
            "external_pids": sorted(actual - allowed),
            "gpu": GPU,
        }

    def shared_idle(self, gpu, plan):
        return self.admission(gpu)["passed"]

    def reserve_existing(self):
        _, stages = self.owned_running()
        for key in stages:
            self.pool.reserve(key)
        return stages

    def adopt(self, arm, stage, receipt, timeout):
        directory = self.batch / arm
        name = receipt["name"]
        timed_out = False
        while True:
            container = self.inspect(name)
            state = self.validate_container(container, arm, stage, receipt)
            started = epoch(state["StartedAt"])
            finished = epoch(state["FinishedAt"]) if not state["Running"] else time.time()
            elapsed = max(0.0, finished - started)
            self.controller.write(
                directory / "controller-state.json",
                {
                    "state": "running",
                    "stage": stage,
                    "container": name,
                    "adopted_existing": True,
                    "elapsed_seconds": elapsed,
                    "shared_gpu": True,
                    "utc_epoch": time.time(),
                },
            )
            if not state["Running"]:
                logs = self.transport.docker("logs", name)
                log_path = directory / f"{stage}.log"
                if log_path.exists():
                    raise FileExistsError(
                        "An unclosed stage already has logs; preserve for manual audit"
                    )
                log_path.write_bytes(logs.stdout + logs.stderr)
                self.controller.write(directory / f"exit-{stage}.json", state)
                self.controller.write(
                    directory / f"duration-{stage}.json",
                    {
                        "elapsed_seconds": elapsed,
                        "timed_out": timed_out,
                        "adopted_existing": True,
                        "shared_gpu": True,
                        "exclusive_gpu_wall_benchmark": False,
                    },
                )
                if timed_out or elapsed > timeout:
                    raise TimeoutError("Adopted stage exhausted the original remaining GPU budget")
                if state["ExitCode"]:
                    raise RuntimeError("Adopted failed stage preserved; never retry")
                return elapsed
            if elapsed > timeout and not timed_out:
                self.transport.docker("stop", "--time", "30", name)
                timed_out = True
            time.sleep(10)

    def dispatch(self, plan, batch, source, manifest, arm, stage, *, gpu=None, timeout=None):
        if manifest != SOURCE_SHA or source != self.source or batch != self.batch:
            raise ValueError(
                "Scheduling handoff cannot alter the frozen scientific source or batch"
            )
        directory = batch / arm
        launch, exit_path = directory / f"launch-{stage}.json", directory / f"exit-{stage}.json"
        if exit_path.exists():
            if not self.transport.completed_stage(directory, stage, manifest):
                raise ValueError("Incomplete exit receipt")
            return json.loads((directory / f"duration-{stage}.json").read_text())["elapsed_seconds"]
        if launch.exists():
            receipt = json.loads(launch.read_text())
            self.controller.write(
                directory / f"adoption-{stage}.json",
                {
                    "original_launch_sha256": self.controller.sha(launch),
                    "name": receipt["name"],
                    "id": receipt["id"],
                    "source_sha256": manifest,
                    "utc_epoch": time.time(),
                    "never_restarted": True,
                },
            )
            return self.adopt(arm, stage, receipt, timeout if timeout is not None else 14400)
        if (directory / "output" / stage).exists():
            raise FileExistsError(
                "Unreceipted stage output exists; never repeat a scientific write"
            )
        name = f"{BATCH}-{arm}-{stage}"
        if self.container_exists(name):
            raise FileExistsError("Unreceipted container exists; never recreate or start it")
        return self.original_run(
            plan, batch, source, manifest, arm, stage, gpu=gpu, timeout=timeout
        )

    def gpu_stage(self, plan, batch, source, manifest, arm, stage, remaining):
        if remaining <= 0:
            raise TimeoutError("The original cumulative arm GPU budget is exhausted")
        directory = batch / arm
        exit_path = directory / f"exit-{stage}.json"
        if exit_path.exists():
            return self.dispatch(plan, batch, source, manifest, arm, stage, timeout=remaining)
        with self.pool.slot((arm, stage)):
            waiting = time.monotonic()
            while True:
                gpu = next(
                    row for row in self.transport.gpu_inventory() if row["uuid"] == GPU["uuid"]
                )
                with self.admission_mutex:
                    receipt = self.admission(gpu)
                    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
                    self.controller.write(directory / f"shared-admission-{stage}.json", receipt)
                if receipt["passed"]:
                    break
                self.controller.write(
                    directory / "controller-state.json",
                    {
                        "state": "waiting_for_shared_gpu7_admission",
                        "stage": stage,
                        "elapsed_wait_seconds": time.monotonic() - waiting,
                        "admission": receipt,
                        "external_sharing_admitted": False,
                    },
                )
                if time.monotonic() - waiting > self.controller.resources(plan).get(
                    "resource_wait_seconds", 14400
                ):
                    raise TimeoutError("Shared pool memory admission did not become available")
                time.sleep(15)
            return self.dispatch(
                plan, batch, source, manifest, arm, stage, gpu=gpu, timeout=remaining
            )


def require_completion(path, source_sha=SOURCE_SHA):
    result = json.loads(path.read_text())
    if result.get("passed") is not True or result.get("source_sha256") != source_sha:
        raise ValueError("A prior completion is not bound to the unchanged frozen source")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch", type=Path, default=Path("/ossfs/workspace/llm-memory-editability") / BATCH
    )
    args = parser.parse_args()
    batch = args.batch
    source = batch / "source-v1"
    controller = load_controller(source)
    plan_path = source / "configs" / f"{BATCH}.json"
    if (
        controller.sha(source / "source-manifest.json") != SOURCE_SHA
        or controller.sha(plan_path) != PLAN_SHA
    ):
        raise ValueError("Frozen source or plan identity differs from the authorized batch")
    plan = json.loads(plan_path.read_text())
    controller.transport.verify_source(source, SOURCE_SHA)
    for relative in (
        "scripts/launch_ce_cross_sample_ablation.py",
        "scripts/launch_ce_cross_sample.py",
    ):
        if controller.sha(ROOT / relative) != controller.sha(source / relative):
            raise ValueError("Local controller/transport differs from its frozen counterpart")
    if plan["execution"]["parent_source_manifest_sha256"] != PARENT_SHA or plan["execution"][
        "persistent_root"
    ] != str(batch):
        raise ValueError("The handoff may not change parent science or output location")
    if (
        plan["execution"]["timeout_seconds_per_arm"] != 43200
        or plan["budget"]["optimizer_updates_total_maximum"] != 20572
    ):
        raise ValueError("The original update or runtime budget changed")
    old = json.loads((batch / "controller-process.json").read_text())
    if (
        old["pid"] != OLD_PID
        or old["source_sha256"] != SOURCE_SHA
        or old["plan_sha256"] != PLAN_SHA
    ):
        raise ValueError("Original controller identity differs")
    if (batch / "completion-manifest.json").exists():
        raise FileExistsError("The original batch is already complete; never resume it")
    preparations = {
        architecture: require_completion(batch / f"{architecture}-prepare/completion-manifest.json")
        for architecture in controller.ARCHITECTURES
    }
    with (batch / "controller.lock").open("a") as controller_lock:
        fcntl.flock(controller_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if alive(OLD_PID):
            raise RuntimeError(
                "The original controller is still alive; root must finish the handoff"
            )
        if (batch / "shared-controller-process.json").exists():
            raise FileExistsError(
                "The scheduling handoff has already been submitted; never duplicate"
            )
        gpu_lock_path = Path(f"/tmp/lm-memory-gpu-locks-{os.getuid()}/gpu-7.lock")
        with gpu_lock_path.open("a") as gpu_lock:
            fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            handoff = Handoff(controller, plan, batch, source)
            adopted = handoff.reserve_existing()
            controller.gpu_stage = handoff.gpu_stage
            controller.transport.run_stage = handoff.dispatch
            controller.transport.idle = handoff.shared_idle
            amendment = {
                "batch": BATCH,
                "source_sha256": SOURCE_SHA,
                "plan_sha256": PLAN_SHA,
                "script_sha256": controller.sha(Path(__file__)),
                "test_sha256": controller.sha(
                    ROOT / "tests/test_ce_cross_sample_ablation_shared.py"
                ),
                "gpu": GPU,
                "maximum_concurrent_own_batch_gpu_containers": MAXIMUM,
                "adopted_existing_stages": adopted,
                "science_unchanged": True,
                "new_optimizer_updates_from_scheduler": 0,
                "external_gpu_sharing": False,
                "exclusive_gpu_wall_benchmark": False,
                "utc_epoch": time.time(),
            }
            controller.write(batch / "shared-gpu7-schedule-amendment.json", amendment)
            controller.write(
                batch / "shared-controller-process.json",
                {"pid": os.getpid(), "replaces_host_controller_pid": OLD_PID, **amendment},
            )
            outcomes, failures = {}, {}
            pending = []
            for arm in controller.ARMS:
                completion = batch / arm / "completion-manifest.json"
                if completion.exists():
                    outcomes[arm] = require_completion(completion)
                else:
                    pending.append(arm)
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=len(pending) or 1) as pool:
                    tasks = {
                        pool.submit(
                            controller.execute_arm, plan, batch, source, SOURCE_SHA, arm
                        ): arm
                        for arm in pending
                    }
                    for future in concurrent.futures.as_completed(tasks):
                        arm = tasks[future]
                        try:
                            outcomes[arm] = future.result()
                        except BaseException as error:
                            failures[arm] = repr(error)
                        controller.write(
                            batch / "controller-state.json",
                            {
                                "state": "running_shared_gpu7",
                                "outcomes": outcomes,
                                "failures": failures,
                                "scheduling_amendment": amendment,
                            },
                        )
                if failures:
                    raise RuntimeError(f"Failed stages preserved without retry: {failures}")
                handoff.dispatch(plan, batch, source, SOURCE_SHA, "report", "report")
                verification_path = batch / "report/output/cloud-verification.json"
                if json.loads(verification_path.read_text()).get("passed") is not True:
                    raise ValueError("Independent report cloud verification did not pass")
                completion = {
                    "state": "complete",
                    "passed": True,
                    "preparations": preparations,
                    "outcomes": outcomes,
                    "source_sha256": SOURCE_SHA,
                    "engineering_updates": sum(
                        x["engineering_updates"] for x in preparations.values()
                    ),
                    "new_foundation_updates": 0,
                    "scientific_updates": sum(x["scientific_updates"] for x in outcomes.values()),
                    "development_optimizer_updates": sum(
                        x["development_optimizer_updates"] for x in outcomes.values()
                    ),
                    "total_optimizer_updates": sum(
                        x["engineering_updates"] for x in preparations.values()
                    )
                    + sum(x["optimizer_updates"] for x in outcomes.values()),
                    "independent_audits_passed": True,
                    "full_cloud_json_verified": True,
                    "report_cloud_verification_sha256": controller.sha(verification_path),
                    "scheduling_amendment": amendment,
                    "maximum_observed_gpu_containers": handoff.pool.maximum_observed,
                    "exclusive_gpu_wall_benchmark": False,
                    "automatic_retry": False,
                }
                controller.write(batch / "completion-manifest.json", completion)
                controller.write(batch / "controller-state.json", completion)
            except BaseException as error:
                controller.write(
                    batch / "controller-state.json",
                    {
                        "state": "needs_attention",
                        "error": repr(error),
                        "outcomes": outcomes,
                        "failures": failures,
                        "scheduling_amendment": amendment,
                        "automatic_retry": False,
                    },
                )
                raise


if __name__ == "__main__":
    main()
