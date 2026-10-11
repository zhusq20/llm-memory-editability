"""A separate execution attempt for the v3 source-base read-only mount repair.

The v3 scientific freeze is retained byte for byte. This helper imports its
controller and changes only execution paths, container mounts/names, and tracking
identity. Original failure receipts and logs remain in their original locations
and are copied to a scheduling-intervention snapshot. No failed task is retried
until the explicit launch of this new attempt.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import netrc
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
ORIGIN = Path("/ossfs/workspace/research-recovery/zero-training-path-overlap-v3")
ATTEMPT_NAME = "source-base-mount-v1"
HELPER_NAME = "resume_path_overlap_mount.py"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def load_controller(origin):
    path = origin / "frozen/source/scripts/execute_path_overlap.py"
    frozen = read(origin / "frozen/freeze.json")
    if sha(path) != frozen["source_files"]["scripts/execute_path_overlap.py"]:
        raise ValueError("Original frozen controller hash changed")
    spec = importlib.util.spec_from_file_location("frozen_v3_controller", path)
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)
    return legacy


def install_wrappers(legacy, origin, attempt):
    """Preserve every scientific function and the inherited selection/power rules."""
    original_source, original_inputs = legacy.source, legacy.input_root
    original_command, original_write = legacy.container_command, legacy.write
    original_resource_check = legacy.resource_check
    evidence_base = Path(read(origin / "selection-adoption.json")["source_base"])
    helper = attempt / "orchestration" / HELPER_NAME
    legacy.source = lambda base: original_source(origin if Path(base) == attempt else base)
    legacy.input_root = lambda base: original_inputs(origin if Path(base) == attempt else base)

    def recorded(path, value):
        path = Path(path)
        if path.name == "run.json" and path.parent.parent == attempt / "runs":
            value = {
                **value,
                "tracking_group": legacy.contract(attempt)["batch"],
                "execution_attempt": ATTEMPT_NAME,
                "spec": {**value["spec"], "execution_attempt": ATTEMPT_NAME},
            }
        original_write(path, value)

    def command(base, run, stage, gpu, cpus):
        old_name, argv = original_command(base, run, stage, gpu, cpus)
        name = old_name.replace("lm-z1-v3-", "lm-z1-v3-mountfix-")
        argv[argv.index("--name") + 1] = name
        entrypoint = str(legacy.source(base) / "scripts/execute_path_overlap.py")
        argv[argv.index(entrypoint)] = str(helper)
        # verify_adoption reads both the original v2 freeze and selection JSON.
        # Mount the full evidence bases, beyond their already mounted inputs.
        extra = []
        for root in (origin, evidence_base):
            extra.extend(["--mount", f"type=bind,source={root},target={root},readonly"])
        # Parent evidence mounts precede the writable attempt bind, preserving
        # the writable child mount even when a runtime honors CLI mount order.
        first_mount = argv.index("--mount")
        argv[first_mount:first_mount] = extra
        argv.extend(["--origin-base", str(origin)])
        return name, argv

    def resources(base, gpus, reserved_cpus=None):
        groups = [list(range(16 + 8 * (gpu - 2), 24 + 8 * (gpu - 2))) for gpu in gpus]
        fixed = [cpu for group in groups for cpu in group]
        if not set(fixed) <= set(os.sched_getaffinity(0)):
            raise RuntimeError("Fixed recovery CPU cores 16-63 are outside host affinity")
        if reserved_cpus is not None and set(reserved_cpus) != set(fixed):
            raise RuntimeError(
                "Recovery worker CPU assignment differs from the fixed 16-63 mapping"
            )
        value = original_resource_check(base, gpus, reserved_cpus=fixed)
        value["cpus"] = groups
        value["scheduling_amendment"] = "Z1 GPU2-7 use CPU16-63; PERK GPU0/1 reserve CPU0-15"
        original_write(Path(base) / f"resource-check-fixed-{time.time_ns()}.json", value)
        return value

    legacy.write = recorded
    legacy.container_command = command
    legacy.resource_check = resources
    return legacy


def original_attempt_evidence(origin, legacy):
    state = read(origin / "controller-state.json")
    if state["state"] != "finished_with_failures" or state.get("active"):
        raise ValueError("Original execution must be terminal before this recovery attempt")
    adoption = legacy.verify_adoption(origin)
    failed = state.get("failed", [])
    if (
        len(failed) != 1
        or failed[0].get("run") != "standard8-phi3.6-wd0.1"
        or failed[0].get("stage") != "select"
    ):
        raise ValueError("This recovery is limited to the one failed terminal selection container")
    failure_log = (origin / "runs/standard8-phi3.6-wd0.1/select.log").read_text()
    missing_freeze = str(Path(adoption["source_base"]) / "frozen/freeze.json")
    if (
        "FileNotFoundError" not in failure_log
        or missing_freeze not in failure_log
        or "verify_adoption" not in failure_log
    ):
        raise ValueError("Original failure was not the authorized provenance mount omission")
    expected = {origin / node["destination"] for node in adoption["nodes"]}
    actual = set((origin / "selection-stage").glob("*/*/selection.json"))
    if actual != expected or len(actual) != 111:
        raise ValueError("The mount failure must have produced zero additional selection nodes")
    cfg = legacy.contract(origin)
    if any(
        (origin / run / f"{node:07d}").exists()
        for run in cfg["source"]["runs"]
        for node in cfg["source"]["nodes"]
    ):
        raise ValueError("Recovery scope assumes no v3 scientific intervention output exists")
    return adoption


def prepare(args):
    origin, attempt = args.origin_base, args.attempt_base
    if attempt.exists():
        raise RuntimeError("Recovery attempt already exists; inspect instead of replacing it")
    legacy = load_controller(origin)
    legacy.verify_freeze(origin)
    adoption = original_attempt_evidence(origin, legacy)
    archive = origin / "scheduling-intervention" / ATTEMPT_NAME
    if archive.exists():
        raise RuntimeError("Scheduling-intervention snapshot already exists")
    evidence = []
    root_files = [
        path
        for path in origin.glob("*")
        if path.is_file()
        and (
            path.suffix == ".log"
            or path.name.startswith(("controller-", "tracking-", "launch-", "execution-"))
        )
    ]
    run_files = [
        path
        for path in (origin / "runs").rglob("*")
        if path.is_file() and path.suffix in {".json", ".log"}
    ]
    tracking_files = list((origin / "tracking-wandb").glob("*/state.json"))
    for path in sorted(set(root_files + run_files + tracking_files)):
        target = archive / "original-attempt" / path.relative_to(origin)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        evidence.append({"path": str(path), "snapshot": str(target), "sha256": sha(path)})
    for folder in ("frozen", "inputs", "orchestration"):
        (attempt / folder).mkdir(parents=True, exist_ok=True)
    (attempt / "frozen/freeze.json").symlink_to(origin / "frozen/freeze.json")
    (attempt / "frozen/source").symlink_to(origin / "frozen/source")
    (attempt / "inputs/grokking-reproduction-v1").symlink_to(
        origin / "inputs/grokking-reproduction-v1"
    )
    (attempt / "input-receipt.json").symlink_to(origin / "input-receipt.json")
    (attempt / "validation").symlink_to(origin / "validation")
    shutil.copy2(origin / "selection-adoption.json", attempt / "selection-adoption.json")
    for node in adoption["nodes"]:
        source = origin / node["destination"]
        target = attempt / node["destination"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        target.chmod(0o444)
    helper = attempt / "orchestration" / HELPER_NAME
    shutil.copy2(Path(__file__), helper)
    helper.chmod(0o444)
    receipt = {
        "scientific_batch": legacy.contract(origin)["batch"],
        "execution_attempt": ATTEMPT_NAME,
        "origin_base": str(origin),
        "attempt_base": str(attempt),
        "origin_freeze_sha256": sha(origin / "frozen/freeze.json"),
        "controller_sha256": sha(origin / "frozen/source/scripts/execute_path_overlap.py"),
        "helper_sha256": sha(helper),
        "selection_adoption_sha256": sha(attempt / "selection-adoption.json"),
        "copied_selection_nodes": 111,
        "failed_attempt_new_selection_nodes": 0,
        "failed_attempt_science_nodes": 0,
        "new_optimizer_updates": 0,
        "change": (
            "Read-only mounts of the complete v2 and v3 evidence bases; "
            "separate orchestration paths, container names and W&B execution identity"
        ),
        "scientific_functions_changed": False,
        "cpu_assignment": {
            str(gpu): list(range(16 + 8 * (gpu - 2), 24 + 8 * (gpu - 2))) for gpu in range(2, 8)
        },
        "failure_evidence": evidence,
        "created_utc": legacy.utc(),
    }
    write(attempt / "recovery-receipt.json", receipt)
    write(archive / "schedule-amendment.json", receipt)
    wrapped = install_wrappers(legacy, origin, attempt)
    cfg = wrapped.contract(attempt)
    for node in adoption["nodes"]:
        wrapped.record_progress(
            attempt,
            node["run"],
            "select",
            node["step"],
            read(attempt / node["destination"]),
            cfg["source"]["nodes"],
            provenance=node,
        )
    print(
        json.dumps(
            {
                "attempt_base": str(attempt),
                "science_batch": cfg["batch"],
                "copied_selection_nodes": 111,
                "scientific_freeze_changed": False,
            }
        ),
        flush=True,
    )


def verify_recovery(args):
    origin, attempt = args.origin_base, args.attempt_base
    receipt = read(attempt / "recovery-receipt.json")
    if receipt["origin_base"] != str(origin) or receipt["attempt_base"] != str(attempt):
        raise ValueError("Recovery source or execution path changed")
    for path, expected in (
        (origin / "frozen/freeze.json", receipt["origin_freeze_sha256"]),
        (origin / "frozen/source/scripts/execute_path_overlap.py", receipt["controller_sha256"]),
        (attempt / "orchestration" / HELPER_NAME, receipt["helper_sha256"]),
        (attempt / "selection-adoption.json", receipt["selection_adoption_sha256"]),
    ):
        if sha(path) != expected:
            raise ValueError(f"Recovery receipt hash mismatch: {path}")
    for item in receipt["failure_evidence"]:
        if sha(item["snapshot"]) != item["sha256"]:
            raise ValueError("Archived original execution evidence changed")
    return install_wrappers(load_controller(origin), origin, attempt)


def launch(args, legacy):
    base = args.attempt_base
    legacy.verify_freeze(base, all_inputs=True)
    with (base / "submission.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (base / "controller-process.json").exists() or (base / "tracking-process.json").exists():
            raise RuntimeError("This recovery attempt has already been submitted")
        validation, audit = (
            read(base / "validation/cuda-validation.json"),
            read(base / "validation/cuda-audit.json"),
        )
        if (
            validation.get("passed") is not True
            or audit.get("passed") is not True
            or audit.get("fresh_process") is not True
        ):
            raise RuntimeError("Inherited original-scale CUDA validation has not passed")
        write(base / "launch-resource-check.json", legacy.resource_check(base, args.gpus))
        auth = netrc.netrc(str(legacy.CREDENTIALS)).authenticators("api.wandb.ai")
        if not auth or not auth[2]:
            raise RuntimeError("W&B credential file has no api.wandb.ai password")
        write(base / "launch-time.json", {"unix_time": time.time(), "created_utc": legacy.utc()})
        write(
            base / "execution-attachment.json",
            {
                "freeze_sha256": sha(base / "frozen/freeze.json"),
                "selection_adoption_sha256": sha(base / "selection-adoption.json"),
                "recovery_receipt_sha256": sha(base / "recovery-receipt.json"),
                "created_utc": legacy.utc(),
            },
        )
        legacy.register_adopted_tracking(base)
        legacy.controller_state(base, "launching", {}, [], [])
        helper = base / "orchestration" / HELPER_NAME
        common = ["--origin-base", str(args.origin_base), "--attempt-base", str(base)]
        env = legacy.host_environment(base)
        command = [sys.executable, "-u", str(helper), "tracker", *common]
        with (base / "tracking.log").open("x") as log:
            tracker = subprocess.Popen(
                command,
                env={**env, "WANDB_API_KEY": auth[2]},
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        write(
            base / "tracking-process.json",
            {"pid": tracker.pid, "command": command, "mode": "online", "created_utc": legacy.utc()},
        )
        deadline = time.monotonic() + 45
        while not (base / "tracking-ready.json").exists():
            if tracker.poll() is not None or time.monotonic() >= deadline:
                failure = {
                    "stage": "tracker_startup",
                    "message": "Tracker did not initialize before recovery controller",
                    "created_utc": legacy.utc(),
                }
                write(base / "controller-failure.json", failure)
                legacy.controller_state(base, "finished_with_failures", {}, [], [failure])
                raise RuntimeError(failure["message"])
            time.sleep(0.1)
        command = [sys.executable, "-u", str(helper), "controller", *common]
        with (base / "controller.log").open("x") as log:
            controller = subprocess.Popen(
                command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
        write(
            base / "controller-process.json",
            {
                "pid": controller.pid,
                "command": command,
                "frozen_scientific_source": True,
                "recovery_receipt_sha256": sha(base / "recovery-receipt.json"),
                "created_utc": legacy.utc(),
            },
        )
        print(
            json.dumps(
                {
                    "controller_pid": controller.pid,
                    "tracker_pid": tracker.pid,
                    "attempt_base": str(base),
                    "scientific_batch": legacy.contract(base)["batch"],
                }
            ),
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=(
            "prepare",
            "verify",
            "resource-check",
            "launch",
            "controller",
            "worker",
            "tracker",
        ),
    )
    parser.add_argument("--origin-base", type=Path, default=ORIGIN)
    parser.add_argument("--attempt-base", "--base", type=Path)
    parser.add_argument("--gpus", type=int, nargs="+", default=list(range(2, 8)))
    parser.add_argument("--run")
    parser.add_argument("--stage", choices=("select", "science", "audit"))
    parser.add_argument("--steps", type=int, nargs="+")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--physical-gpu", type=int)
    args = parser.parse_args()
    args.attempt_base = args.attempt_base or args.origin_base / "execution-attempts" / ATTEMPT_NAME
    if args.action == "prepare":
        prepare(args)
        return
    legacy = verify_recovery(args)
    args.base = args.attempt_base
    if args.action == "launch":
        launch(args, legacy)
    elif args.action == "verify":
        legacy.verify_freeze(args.base, all_inputs=True)
    elif args.action == "resource-check":
        print(json.dumps(legacy.resource_check(args.base, args.gpus)), flush=True)
    elif args.action in ("controller", "worker"):
        getattr(legacy, args.action)(args)
    else:
        legacy.tracker(args.base)


if __name__ == "__main__":
    main()
