"""Run six authorized one-variable ablations with exclusive, dynamic GPU leases."""

from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import importlib.util
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
BATCH = "ce-cross-sample-ablation-v1"
ARCHITECTURES = ("loop", "discoloop")
VARIANTS = ("lower-lr", "carry-adam", "no-bound")
ARMS = tuple(f"{architecture}-{variant}" for architecture in ARCHITECTURES for variant in VARIANTS)
IMAGE = "sha256:9643a7d6aeda1b4c9ac2ac50c2d84474b082d649a87e423bc6e0bdc0994097ee"
PARENT_MANIFEST = "68bef3953b12e9be33c65e2794b9159417c855dc0e74a79aa146d1a99e86de4f"
SPEC = importlib.util.spec_from_file_location(
    "_ce_ablation_transport", ROOT / "scripts/launch_ce_cross_sample.py"
)
transport = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(transport)
transport.BATCH = BATCH
write, sha, resources = transport.write, transport.sha, transport.resources


def arms_for(plan):
    arms = plan["arms"]
    if not isinstance(arms, dict) or set(arms) != set(ARMS):
        raise ValueError("Exactly the six authorized architecture/ablation arms are required")
    for arm, item in arms.items():
        if item["architecture"] != arm.split("-", 1)[0]:
            raise ValueError("Arm architecture does not match its identifier")
        if item["architecture"] not in ARCHITECTURES:
            raise ValueError("Only the existing Loop and DiscoLoop checkpoints are authorized")
    return arms


def extra_files(plan):
    values = plan["execution"].get("extra_files")
    if not values or len(values) != len(set(values)):
        raise ValueError("A unique execution.extra_files list is required")
    paths = tuple(str(transport.safe_relative(value)) for value in values)
    required = {
        "scripts/launch_ce_cross_sample_ablation.py",
        "scripts/run_ce_cross_sample_ablation.py",
        "scripts/report_ce_cross_sample_ablation.py",
        f"configs/{BATCH}.json",
        f"docs/development-artifacts/{BATCH}/protocol.md",
    }
    if not required <= set(paths):
        raise ValueError("Ablation source extras omit required execution files")
    return paths


def private_output(batch, arm):
    transport.validate_batch(batch)
    allowed = (*ARMS, "loop-prepare", "discoloop-prepare", "contracts", "report")
    if arm not in allowed:
        raise ValueError("Output architecture escaped the authorized ablation batch")
    output = batch / arm / "output"
    if output.resolve() != batch.resolve() / arm / "output":
        raise ValueError("Output symlink escaped the authorized ablation batch")
    output.mkdir(mode=0o700, parents=True, exist_ok=True)
    identity = output.stat()
    if (identity.st_uid, identity.st_gid) != (1015, 1016) or identity.st_mode & 0o077:
        raise ValueError(
            "Persistent output must have runtime ownership and private 0700 permissions"
        )
    return output


def command_for(plan, stage, arm):
    allowed = (*ARMS, "loop-prepare", "discoloop-prepare", "contracts", "report")
    if arm not in allowed:
        raise ValueError("Output architecture escaped the authorized ablation batch")
    if stage == "contracts":
        paths = [path for path in extra_files(plan) if path.endswith(".py")]
        tests = plan["execution"].get(
            "contract_tests", [path for path in paths if path.startswith("tests/")]
        )
        if not tests:
            raise ValueError("Ablation contract tests must be frozen")
        commands = [
            ["python", "-m", "pytest", "-q", "-p", "no:cacheprovider", *tests],
            ["ruff", "check", *paths],
            ["ruff", "format", "--check", *paths],
        ]
        return [
            "python",
            "-c",
            "import subprocess; "
            + "; ".join(f"subprocess.run({command!r},check=True)" for command in commands),
        ]
    runtime_batch = transport.runtime_batch(plan)
    if stage == "report":
        return [
            "python",
            "scripts/report_ce_cross_sample_ablation.py",
            "--plan",
            f"configs/{BATCH}.json",
            "--batch",
            str(runtime_batch),
            "--out",
            str(runtime_batch / "report/output"),
        ]
    command = ["python", "scripts/run_ce_cross_sample_ablation.py", stage]
    if stage == "prepare":
        architecture = arm.removesuffix("-prepare")
        if architecture not in ARCHITECTURES or arm != f"{architecture}-prepare":
            raise ValueError("Prepare must address one original architecture")
        command += ["--architecture", architecture]
    else:
        if arm not in ARMS or stage not in ("gate", "audit-gate", "science", "audit", "cloud"):
            raise ValueError("Unrecognized arm stage")
        command += ["--arm", arm]
    return [
        *command,
        "--plan",
        f"configs/{BATCH}.json",
        "--root",
        str(runtime_batch / arm / "output"),
        "--device",
        "cpu" if stage == "cloud" else "cuda:0",
    ]


# These substitutions affect this isolated imported module, never the frozen parent file.
transport.extra_files = extra_files
transport.command_for = command_for
transport.private_output = private_output


def stage_result(batch, arm, stage):
    path = batch / arm / "output" / stage / "result.json"
    return json.loads(path.read_text())


def lease_gpu(plan, directory, stage):
    started = time.monotonic()
    while True:
        gpu, lock = transport.acquire_gpu(plan)
        if gpu is not None:
            return gpu, lock
        write(
            directory / "controller-state.json",
            {
                "state": "waiting_for_exclusive_idle_gpu",
                "stage": stage,
                "elapsed_wait_seconds": time.monotonic() - started,
            },
        )
        if time.monotonic() - started > resources(plan).get("resource_wait_seconds", 14400):
            raise TimeoutError("No exclusive idle GPU became available; external work preserved")
        time.sleep(15)


def gpu_stage(plan, batch, source, manifest, arm, stage, remaining):
    if remaining <= 0:
        raise TimeoutError("The cumulative GPU stage budget is exhausted")
    directory = batch / arm
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    if transport.completed_stage(directory, stage, manifest):
        return json.loads((directory / f"duration-{stage}.json").read_text())["elapsed_seconds"]
    gpu, lock = lease_gpu(plan, directory, stage)
    try:
        prior_path = directory / "resource-history.json"
        prior = json.loads(prior_path.read_text()) if prior_path.exists() else []
        receipt = {
            "stage": stage,
            "gpu": gpu,
            "utc_epoch": time.time(),
            "previous_gpu_uuid": prior[-1]["gpu"]["uuid"] if prior else None,
            "reason": "independent idle-GPU admission for every stage; no external sharing",
        }
        write(directory / f"resource-admission-{stage}.json", receipt)
        write(directory / "resource-history.json", [*prior, receipt])
        if prior and prior[-1]["gpu"]["uuid"] != gpu["uuid"]:
            write(directory / f"resource-amendment-{stage}.json", receipt)
        return transport.run_stage(
            plan, batch, source, manifest, arm, stage, gpu=gpu, timeout=remaining
        )
    finally:
        lock.close()


def run_prepare(plan, batch, source, manifest, architecture):
    arm = f"{architecture}-prepare"
    directory = batch / arm
    try:
        elapsed = gpu_stage(
            plan,
            batch,
            source,
            manifest,
            arm,
            "prepare",
            plan["execution"].get("prepare_timeout_seconds", 43200),
        )
        result = stage_result(batch, arm, "prepare")
        if result.get("passed") is not True:
            raise ValueError("Architecture engineering preparation did not pass")
        completion = {
            "state": "complete",
            "passed": True,
            "architecture": architecture,
            "engineering_updates": 46,
            "scientific_updates": 0,
            "elapsed_gpu_seconds": elapsed,
            "source_sha256": manifest,
        }
        write(directory / "completion-manifest.json", completion)
        write(directory / "controller-state.json", completion)
        return completion
    except BaseException as error:
        write(
            directory / "controller-state.json",
            {"state": "needs_attention", "error": repr(error), "traceback": traceback.format_exc()},
        )
        raise


def execute_arm(plan, batch, source, manifest, arm):
    directory = batch / arm
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    completed, durations = [], {}
    remaining = plan["execution"]["timeout_seconds_per_arm"]
    try:
        for stage in ("gate", "audit-gate"):
            durations[stage] = gpu_stage(plan, batch, source, manifest, arm, stage, remaining)
            remaining -= durations[stage]
            completed.append(stage)
        if stage_result(batch, arm, "audit-gate").get("passed") is not True:
            raise ValueError("Independent gate replay did not pass")
        gate = stage_result(batch, arm, "gate")
        if not isinstance(gate.get("eligible"), bool):
            raise ValueError("Gate result must explicitly declare eligible true or false")
        if gate.get("passed") is not True:
            raise ValueError("Gate stage did not complete its engineering and coverage checks")
        development_updates = gate.get("development_optimizer_updates")
        allowed_development = (640, 1280) if arm.endswith("-lower-lr") else (640,)
        if type(development_updates) is not int or development_updates not in allowed_development:
            raise ValueError("Gate optimizer update count escaped the frozen candidate budget")
        eligible = gate["eligible"]
        if eligible:
            for stage in ("science", "audit"):
                durations[stage] = gpu_stage(plan, batch, source, manifest, arm, stage, remaining)
                remaining -= durations[stage]
                completed.append(stage)
            if stage_result(batch, arm, "science").get("scientific_optimizer_updates") != 2560:
                raise ValueError(
                    "Continuous science optimizer updates differ from the authorized 2560"
                )
            if stage_result(batch, arm, "audit").get("passed") is not True:
                raise ValueError("Independent continuous-state replay did not pass")
        transport.run_stage(plan, batch, source, manifest, arm, "cloud")
        completed.append("cloud")
        cloud = stage_result(batch, arm, "cloud")
        if cloud.get("passed") is not True:
            raise ValueError("The complete eligible or ineligible arm cloud audit did not pass")
        completion = {
            "state": "complete" if eligible else "ineligible",
            "passed": True,
            "eligible": eligible,
            "arm": arm,
            "architecture": plan["arms"][arm]["architecture"],
            "completed_stages": completed,
            "new_foundation_updates": 0,
            "scientific_updates": 2560 if eligible else 0,
            "development_optimizer_updates": development_updates,
            "optimizer_updates": development_updates + (2560 if eligible else 0),
            "independent_audit_passed": True,
            "full_cloud_json_verified": True,
            "gpu_durations_seconds": durations,
            "source_sha256": manifest,
            "automatic_retry": False,
        }
        write(directory / "completion-manifest.json", completion)
        write(directory / "controller-state.json", completion)
        return completion
    except BaseException as error:
        write(
            directory / "controller-state.json",
            {
                "state": "needs_attention",
                "completed_stages": completed,
                "error": repr(error),
                "traceback": traceback.format_exc(),
                "automatic_retry": False,
            },
        )
        raise


def preflight(plan, batch):
    arms_for(plan)
    execution = plan["execution"]
    if plan["batch"] != BATCH or execution["image"] != IMAGE:
        raise ValueError("The authorized batch and immutable LM image must remain fixed")
    if execution["parent_source_manifest_sha256"] != PARENT_MANIFEST:
        raise ValueError("The completed parent source must remain fixed")
    if execution.get("parent_source_file_count") != 444:
        raise ValueError("The completed parent must retain all 444 source files")
    if plan["budget"]["optimizer_updates_total_maximum"] != 20572:
        raise ValueError("The complete authorized optimizer update budget must remain fixed")
    transport.validate_batch(batch)
    free = shutil.disk_usage(batch).free
    if free < execution["minimum_free_persistent_bytes"]:
        raise RuntimeError(
            "Insufficient persistent storage for all arms; existing evidence preserved"
        )
    if shutil.disk_usage("/workspaces").free < resources(plan).get(
        "minimum_free_temporary_bytes", 2 * 1024**3
    ):
        raise RuntimeError("Insufficient temporary storage; no volume-based outputs permitted")
    selected = set()
    cpu_sets = resources(plan)["cpu_sets"]
    for arm in ARMS:
        cpus = set()
        for segment in cpu_sets[arm].split(","):
            start, _, finish = segment.partition("-")
            cpus.update(range(int(start), int(finish or start) + 1))
        if len(cpus) != 4 or selected & cpus or not cpus <= os.sched_getaffinity(0):
            raise ValueError("Six four-core CPU reservations must be available and disjoint")
        selected |= cpus
    for architecture in ARCHITECTURES:
        if cpu_sets[f"{architecture}-prepare"] != cpu_sets[f"{architecture}-lower-lr"]:
            raise ValueError("Prepare may borrow only its architecture's first arm CPU reservation")
    if not transport.TOKEN.is_file():
        raise FileNotFoundError("Existing read-only W&B token is missing")
    write(
        batch / "preflight.json",
        {"persistent_free_bytes": free, "arm_cpus": sorted(selected), "utc_epoch": time.time()},
    )


def run_parallel(function, arguments, names, state_path):
    outcomes, failures = {}, {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(names)) as pool:
        tasks = {pool.submit(function, *arguments, name): name for name in names}
        for future in concurrent.futures.as_completed(tasks):
            name = tasks[future]
            try:
                outcomes[name] = future.result()
            except BaseException:
                failures[name] = traceback.format_exc()
            write(state_path, {"state": "running", "outcomes": outcomes, "failures": failures})
    if failures:
        raise RuntimeError(f"Failed tasks preserved without retry: {sorted(failures)}")
    return outcomes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=ROOT / f"configs/{BATCH}.json")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--freeze-only", action="store_true")
    mode.add_argument("--contracts-only", action="store_true")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    arms_for(plan)
    batch = Path(plan["execution"]["persistent_root"])
    transport.validate_batch(batch)
    batch.mkdir(mode=0o700, parents=True, exist_ok=True)
    with (batch / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (batch / "controller-process.json").exists():
            raise FileExistsError("This authorized ablation has already been submitted")
        source, manifest = transport.freeze(plan, batch, repository=ROOT)
        if args.freeze_only:
            return
        preflight(plan, batch)
        transport.run_stage(plan, batch, source, manifest, "contracts", "contracts")
        write(batch / "contracts.json", {"passed": True, "source_sha256": manifest})
        if args.contracts_only:
            return
        write(
            batch / "controller-process.json",
            {
                "pid": os.getpid(),
                "source_sha256": manifest,
                "source_revision": source.name,
                "plan_sha256": sha(args.plan),
                "submitted_epoch": time.time(),
                "parallel_prepare_architectures": ARCHITECTURES,
                "parallel_arms": ARMS,
                "automatic_retry": False,
            },
        )
        preparations, outcomes = {}, {}
        try:
            preparations = run_parallel(
                run_prepare,
                (plan, batch, source, manifest),
                ARCHITECTURES,
                batch / "controller-state.json",
            )
            outcomes = run_parallel(
                execute_arm, (plan, batch, source, manifest), ARMS, batch / "controller-state.json"
            )
            transport.run_stage(plan, batch, source, manifest, "report", "report")
            verification_path = batch / "report/output/cloud-verification.json"
            if json.loads(verification_path.read_text()).get("passed") is not True:
                raise ValueError("The report's independent full cloud verification did not pass")
            completion = {
                "state": "complete",
                "passed": True,
                "preparations": preparations,
                "outcomes": outcomes,
                "source_sha256": manifest,
                "engineering_updates": sum(
                    item["engineering_updates"] for item in preparations.values()
                ),
                "new_foundation_updates": 0,
                "scientific_updates": sum(item["scientific_updates"] for item in outcomes.values()),
                "development_optimizer_updates": sum(
                    item["development_optimizer_updates"] for item in outcomes.values()
                ),
                "total_optimizer_updates": sum(
                    item["engineering_updates"] for item in preparations.values()
                )
                + sum(item["optimizer_updates"] for item in outcomes.values()),
                "independent_audits_passed": True,
                "full_cloud_json_verified": True,
                "report_cloud_verification_sha256": sha(verification_path),
                "automatic_retry": False,
            }
            write(batch / "completion-manifest.json", completion)
            write(batch / "controller-state.json", completion)
        except BaseException as error:
            write(
                batch / "controller-state.json",
                {
                    "state": "needs_attention",
                    "preparations": preparations,
                    "outcomes": outcomes,
                    "error": repr(error),
                    "traceback": traceback.format_exc(),
                    "automatic_retry": False,
                },
            )
            raise


if __name__ == "__main__":
    main()
