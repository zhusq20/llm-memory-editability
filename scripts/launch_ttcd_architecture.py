#!/usr/bin/env python3
"""Freeze and run the authorized D8/Loop TTCD pipelines on exclusive GPU8/9."""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import fcntl
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BATCH = "ttcd-architecture-comparison-v1"
NEW_ARMS = ("transformer", "loop")
PARENT = "ttcd-successful-base-v1"
SUPPLEMENT = "ttcd-double-new-eval-v1"
PROTOCOL = f"docs/development-artifacts/{BATCH}/protocol.md"
OVERLAYS = (
    "src/llm_memory_editability/ttcd_architecture.py",
    "scripts/run_ttcd_architecture.py",
    "scripts/run_ttcd_architecture_foundation.py",
    "scripts/launch_ttcd_architecture.py",
    "scripts/report_ttcd_architecture.py",
    "src/llm_memory_editability/ttcd_double_new_eval.py",
    "scripts/run_ttcd_double_new_eval.py",
    "tests/test_ttcd_double_new_eval.py",
    "scripts/run_discoloop_restore.py",
    "src/llm_memory_editability/discoloop_restore.py",
    PROTOCOL,
)
INITIAL_SHA256 = {
    "transformer": "20c3c3a0611578eb5a2e4bd15de6f7d825a3bd9442785a51fe9ac642240e33ee",
    "loop": "81e95a16656904a511f901b30966a4b4c4f1de708b67b5712d3e8cd2933fe347",
}
DOUBLE_PANEL_SHA256 = "ade07fa92e528f1b19bbc10965d984f747ed7f7daf66f5eafa254971ac7d08f6"


def load_resources(path=None):
    path = path or ROOT / "scripts/launch_discoloop_paired.py"
    spec = importlib.util.spec_from_file_location("architecture_resource_helpers", path)
    module = importlib.util.module_from_spec(spec)
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = old
    return module


resources = load_resources()
read_json, write_json, sha256 = resources.read_json, resources.write_json, resources.sha256


def validate_plan(plan):
    if plan.get("short_training") is not False:
        raise ValueError("User explicitly cancelled all short training")
    if (
        plan["batch"] != BATCH
        or plan["architectures"] != [*NEW_ARMS, "discoloop"]
        or plan["new_training_architectures"] != list(NEW_ARMS)
        or plan["new_scientific_updates"] != 640000
        or plan["per_new_architecture"] != {"foundation": 300000, "common": 10000, "stream": 10000}
    ):
        raise ValueError("Only the two authorized native300k/common10k/stream10k additions")
    if plan["formal_panel_sha256"] != DOUBLE_PANEL_SHA256 or plan["architecture_configs"] != {
        arm: f"configs/{BATCH}-{arm}.json" for arm in NEW_ARMS
    }:
        raise ValueError("Reuse the exact authorized panel and both architecture configurations")
    runtime = plan["runtime"]
    if runtime["gpus"] != {"transformer": 8, "loop": 9}:
        raise ValueError("Use exactly GPU8 and GPU9; no fallback or sharing")
    if runtime["cpu_sets"] != {"transformer": list(range(40, 44)), "loop": list(range(44, 48))}:
        raise ValueError("Keep new CPU sets disjoint from the original GPU4 cohorts")
    if runtime.get("docker_context") != "lm-memory" or runtime["tracking"] != "online":
        raise ValueError("Use explicit LM Docker and online experiment tracking")
    if runtime["memory_gib"] != 96 or runtime["cpu_threads"] != 4:
        raise ValueError("Each arm requires its reviewed four CPUs and96GiB")
    if runtime["uid"] != 1015 or runtime["gid"] != 1016:
        raise ValueError("Use original persistent output ownership")
    if runtime["minimum_persistent_free_gib"] < 400:
        raise ValueError("Reserve enough for D8's double-size fast states")
    if runtime.get("shared_gpu_allowlist"):
        raise ValueError("This comparison uses exclusive GPUs")
    if not runtime["image"].startswith("sha256:") or len(runtime["image"]) != 71:
        raise ValueError("Use a pinned dependency image")
    if runtime["timeout_hours"] != 48 or runtime["wait_timeout_hours"] != 24:
        raise ValueError("Preserve fixed execution and resource waiting timeouts")
    reuse = plan["reuse"]["discoloop"]
    if reuse != {
        "foundation_batch": "discoloop-success-restore-v1",
        "adaptation_batch": PARENT,
        "arm": "stream",
        "double_new_batch": SUPPLEMENT,
        "double_new_arm": "stream",
        "training_restarted": False,
    }:
        raise ValueError("DiscoLoop must reuse its exact existing stream run")


def stage_plan(arm):
    if arm not in NEW_ARMS:
        raise ValueError("No duplicate DiscoLoop pipeline")
    result = []

    def add(label, family, stage, phase="common", engineering=False):
        result.append(
            dict(label=label, family=family, stage=stage, phase=phase, engineering=engineering)
        )

    # User explicitly cancelled all short training on 2026-10-15.
    # This preflight verifies existing tensors/resources without an optimizer step.
    add("foundation-preflight", "foundation", "preflight")
    for stage in ("train", "audit", "cloud-audit"):
        add(f"foundation-{stage}", "foundation", stage)
    add("adaptation-preflight", "adaptation", "preflight")
    for stage in ("train", "train-audit", "gate", "cloud-audit"):
        add(f"common-{stage}", "adaptation", stage)
    for stage in ("train", "train-audit", "calibrate", "evaluate", "audit", "cloud-audit"):
        add(f"stream-{stage}", "adaptation", stage, "stream")
    for stage in ("double-evaluate", "double-audit", "double-cloud-audit"):
        add(stage, "adaptation", stage, "stream")
    return result


def verify_entries(root, entries):
    for relative, expected in entries.items():
        path = root / relative
        if not path.resolve().is_relative_to(root.resolve()) or sha256(path) != expected:
            raise ValueError(f"Frozen file differs or escapes its root: {relative}")


def engineering_config(config, arm):
    result = copy.deepcopy(config)
    base = f"results/{BATCH}/{arm}"
    result["engineering_only"] = True
    result["foundation"].update(
        checkpoint=f"{base}/foundation/smoke/{arm}/checkpoint-0000032.pt",
        gate_receipt=f"{base}/foundation/smoke-identity.json",
        step=32,
    )
    result["foundation"]["required_spec"]["steps"] = 32
    result["common"].update(
        steps=16,
        checkpoint=f"{base}/engineering/adaptation/common/training/parent.pt",
        gate_receipt=f"{base}/engineering/adaptation/common/gate.json",
    )
    result["training"].update(steps=16, validation_every=16)
    result["models"]["stream"].update(
        checkpoint=f"{base}/engineering/adaptation/stream/training/parent.pt", step=16
    )
    return result


def freeze(batch, plan, config_path):
    if (batch / "source-manifest.json").exists():
        raise FileExistsError("The authorized submission is already frozen; inspect its controller")
    parent, supplement = batch.parent / PARENT, batch.parent / SUPPLEMENT
    original = read_json(parent / "source-v2-manifest.json")
    verify_entries(parent / "source-v2", original["files"])
    verify_entries(parent / "input-mount", original["inputs"])
    original_config = read_json(parent / "source-v2/configs/experiment.json")
    if plan["runtime"]["image"] != original_config["runtime"]["image"]:
        raise ValueError("Use the existing verified DiscoLoop dependency image")
    panel = supplement / "panel.json"
    if sha256(panel) != plan["formal_panel_sha256"]:
        raise ValueError("Reuse the graph-only frozen supplementary panel bytes")
    source = batch / "source"
    source.mkdir(exist_ok=False)
    for relative in original["files"]:
        target = source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(parent / "source-v2" / relative, target)
    additions = [*OVERLAYS]
    additions += [
        str(p.relative_to(ROOT))
        for p in sorted((ROOT / "tests").glob("test_ttcd_architecture*.py"))
    ]
    for relative in additions:
        target = source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        # Existing shared bytes are allowed only when exactly identical.
        if target.exists() and sha256(target) != sha256(ROOT / relative):
            raise ValueError(f"Architecture overlays must not alter parent science: {relative}")
        shutil.copy2(ROOT / relative, target)
    shutil.copy2(config_path, source / "configs/architecture-comparison.json")
    shutil.copy2(config_path, source / f"configs/{BATCH}.json")
    shutil.copy2(
        supplement / "source/configs/double-new-eval.json", source / "configs/double-new-eval.json"
    )
    shutil.copy2(
        supplement / "source/configs/double-new-eval.json",
        source / "configs/ttcd-double-new-eval-v1.json",
    )
    shutil.copy2(panel, source / "configs/double-new-panel.json")
    config_hashes = {}
    for arm in NEW_ARMS:
        config_file = ROOT / plan["architecture_configs"][arm]
        config = read_json(config_file)
        if (
            config["batch"] != BATCH
            or config["architecture"]["name"] != arm
            or set(config["models"]) != {"stream"}
            or config["common"]["steps"] != 10000
            or config["training"]["steps"] != 10000
            or config["runtime"] != plan["runtime"]
        ):
            raise ValueError("Per-architecture settings changed from the authorized matrix")
        for key in (
            "seed",
            "phi",
            "cases",
            "writer",
            "training",
            "reading_gate",
            "capability_gate",
        ):
            if config[key] != original_config[key]:
                raise ValueError(
                    f"Keep the existing Disco stream sampling/objective/settings: {key}"
                )
        expected_continual = {
            **original_config["continual"],
            "panel_source_path": f"results/{BATCH}/preparation/continual-panel.json",
        }
        if config["continual"] != expected_continual:
            raise ValueError("The continual fact/order/evaluation contract must be unchanged")
        shutil.copy2(config_file, source / f"configs/architecture-{arm}.json")
        shutil.copy2(config_file, source / config_file.relative_to(ROOT))
        native = ROOT / f"configs/{BATCH}-{arm}-foundation.json"
        shutil.copy2(native, source / f"configs/foundation-{arm}.json")
        shutil.copy2(native, source / native.relative_to(ROOT))
        initial = batch / "input-mount/initial" / f"{arm}.pt"
        if sha256(initial) != INITIAL_SHA256[arm]:
            raise ValueError(
                "Initial tensor file differs from the checked historical initialization"
            )
        config_hashes[arm] = sha256(config_file)
    # Nested bind targets must exist before the source root becomes read-only.
    (source / "data").mkdir(exist_ok=True)
    for arm in NEW_ARMS:
        (source / "results" / BATCH / arm).mkdir(parents=True, exist_ok=True)
    for path in (source, *source.rglob("*")):
        path.chmod(0o755 if path.is_dir() else 0o644)
    files = resources.source_hashes(source)
    manifest = {
        "schema": "ttcd-architecture-comparison-freeze-v1",
        "utc": resources.utc_now(),
        "config_sha256": sha256(config_path),
        "architecture_config_sha256": config_hashes,
        "parent_source_v2_manifest_sha256": sha256(parent / "source-v2-manifest.json"),
        "parent_immutable_inputs": original["inputs"],
        "initial_sha256": INITIAL_SHA256,
        "double_new_panel_sha256": sha256(panel),
        "files": files,
        "original_scientific_files": original["files"],
        "new_scientific_updates": 640000,
        "discoloop_retrained": False,
    }
    write_json(batch / "source-manifest.json", manifest)
    return manifest


def verify_snapshot(batch):
    manifest = read_json(batch / "source-manifest.json")
    verify_entries(batch / "source", manifest["files"])
    parent = batch.parent / PARENT
    if sha256(parent / "source-v2-manifest.json") != manifest["parent_source_v2_manifest_sha256"]:
        raise ValueError("Original DiscoLoop source identity changed")
    verify_entries(parent / "input-mount", manifest["parent_immutable_inputs"])
    verify_entries(batch / "input-mount/initial", {f"{k}.pt": v for k, v in INITIAL_SHA256.items()})
    return manifest


def output_path(batch, arm, spec):
    base = batch / arm
    if spec["family"] == "foundation":
        return base / "foundation"
    if spec["family"] == "engineering":
        return base / "engineering/fast-reload"
    if spec["stage"].startswith("double-"):
        return base / "double-new"
    if spec["engineering"]:
        return base / "engineering/adaptation" / spec["phase"]
    return base / spec["phase"]


def build_command(plan, arm, spec, batch, token, row):
    runtime = plan["runtime"]
    cpus = runtime["cpu_sets"][arm]
    out = output_path(batch, arm, spec)
    relative_out = out.relative_to(batch)
    name = f"{BATCH}-{arm}-{spec['label']}"
    command = [
        *resources.DOCKER,
        "run",
        "--detach",
        "--name",
        name,
        "--user",
        "1015:1016",
        "--cpus",
        "4",
        "--cpuset-cpus",
        ",".join(map(str, cpus)),
        "--memory",
        "96g",
        "--shm-size",
        "8g",
        "--network",
        "host",
        "--workdir",
        "/frozen",
        "--gpus",
        f"device={row['uuid']}",
    ]
    for host, target, readonly in (
        (batch / "source", "/frozen", True),
        (batch.parent / PARENT / "input-mount/data", "/inputs/data", True),
        (batch / "input-mount/initial", "/inputs/initial", True),
        (batch / arm, f"/frozen/results/{BATCH}/{arm}", False),
    ):
        command += [
            "--mount",
            f"type=bind,source={host},target={target}" + (",readonly" if readonly else ""),
        ]
    command += ["--tmpfs", "/tmp:rw,size=8g"]
    for env in (
        "PYTHONPATH=/frozen/src",
        "PYTHONDONTWRITEBYTECODE=1",
        "PYTHONUNBUFFERED=1",
        "HOME=/tmp/lm-home",
        "XDG_CACHE_HOME=/tmp/lm-home/.cache",
        "XDG_CONFIG_HOME=/tmp/lm-home/.config",
        "OMP_NUM_THREADS=4",
        "MKL_NUM_THREADS=4",
        f"LM_ACTUAL_GPU_INDEX={row['gpu']}",
        f"LM_ACTUAL_GPU_UUID={row['uuid']}",
        f"WANDB_TAGS=ttcd-architecture,{arm},gpu-{row['gpu']}",
    ):
        command += ["--env", env]
    command += ["--mount", f"type=bind,source={token.resolve()},target=/run/secrets/wandb,readonly"]
    command += [
        runtime["image"],
        "bash",
        "-c",
        'mkdir -p "$HOME"\nexport WANDB_API_KEY="$(cat /run/secrets/wandb)"\nexec "$@"',
        "ttcd-architecture-entry",
        "python",
    ]
    if spec["family"] == "foundation":
        command += [
            "/frozen/scripts/run_ttcd_architecture_foundation.py",
            spec["stage"],
            "--config",
            f"/frozen/configs/foundation-{arm}.json",
        ]
    else:
        config_type = "engineering" if spec["engineering"] else "architecture"
        if spec["family"] == "engineering":
            command += ["/frozen/scripts/verify_ttcd_architecture_cuda.py"]
        else:
            command += [
                "/frozen/scripts/run_ttcd_architecture.py",
                spec["stage"],
                "--model",
                spec["phase"],
                "--device",
                "cuda:0",
            ]
        command += ["--config", f"/frozen/configs/{config_type}-{arm}.json"]
        if spec["stage"].startswith("double-"):
            command += [
                "--settings",
                "/frozen/configs/double-new-eval.json",
                "--panel",
                "/frozen/configs/double-new-panel.json",
            ]
    command += ["--out", f"/frozen/results/{BATCH}/{relative_out}"]
    if spec["family"] != "engineering":
        command += ["--tracking", "disabled" if spec["engineering"] else "online"]
    return name, command


def claim_stage(directory, label):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / f"claim-{label}.json").open("x") as stream:
        json.dump({"pid": os.getpid(), "utc": resources.utc_now(), "stage": label}, stream)


def run_cpu_contracts(plan, batch):
    out = batch / "engineering/cpu-contracts"
    claim_stage(out, "cpu-contracts")
    out.chmod(0o755)
    name = f"{BATCH}-cpu-contracts"
    cpus = [*plan["runtime"]["cpu_sets"]["transformer"], *plan["runtime"]["cpu_sets"]["loop"]]
    resources.check_cpu_sets(
        cpus, resources.running_containers(["lm-memory"]), os.sched_getaffinity(0)
    )
    command = [
        *resources.DOCKER,
        "run",
        "--detach",
        "--name",
        name,
        "--user",
        "1015:1016",
        "--cpus",
        "8",
        "--cpuset-cpus",
        ",".join(map(str, cpus)),
        "--memory",
        "16g",
        "--network",
        "none",
        "--workdir",
        "/frozen",
    ]
    for host, target, readonly in (
        (batch / "source", "/frozen", True),
        (batch.parent / PARENT / "input-mount/data", "/frozen/data", True),
        (out, "/checks", False),
    ):
        command += [
            "--mount",
            f"type=bind,source={host},target={target}" + (",readonly" if readonly else ""),
        ]
    command += ["--tmpfs", "/tmp:rw,size=8g"]
    for env in (
        "PYTHONPATH=/frozen/src",
        "PYTHONDONTWRITEBYTECODE=1",
        "OMP_NUM_THREADS=4",
        "MKL_NUM_THREADS=4",
        "HOME=/tmp/lm-home",
    ):
        command += ["--env", env]
    command += [
        plan["runtime"]["image"],
        "python",
        "-m",
        "pytest",
        "-p",
        "no:cacheprovider",
        "tests/test_ttcd_architecture_controller.py",
        "-q",
        "--junitxml=/checks/results.xml",
    ]
    write_json(out / "command.json", command)
    created = resources.docker(*command[3:])
    write_json(out / "container.json", {"name": name, "id": created.stdout.strip()})
    with (out / "container.log").open("x") as log:
        code = resources.monitor_container(name, log, time.monotonic() + 600)
    write_json(out / "container-state.json", resources.inspect_container(name))
    write_json(out / "exit.json", {"code": code, "utc": resources.utc_now()})
    if code:
        raise RuntimeError("Pinned-image CPU contracts failed; no GPU pipeline was started")
    write_json(
        out / "receipt.json",
        {
            "passed": True,
            "scientific_training_updates": 0,
            "gpu_device_requests": False,
            "junit_sha256": sha256(out / "results.xml"),
        },
    )


def admit(plan, arm):
    runtime, gpu = plan["runtime"], plan["runtime"]["gpus"][arm]
    containers = resources.running_containers(["lm-memory"])
    resources.check_cpu_sets(runtime["cpu_sets"][arm], containers, os.sched_getaffinity(0))
    available = resources.host_memory_available_kib()
    if available < runtime["memory_gib"] * 1024**2:
        raise resources.ResourceUnavailable("Host memory is occupied")
    rows, processes = resources.gpu_snapshot()
    if gpu not in rows:
        raise ValueError(f"RequestedGPU{gpu} is absent")
    row = rows[gpu]
    if (
        row["total_mib"] < 90000
        or row["free_mib"] < 80000
        or row["memory"] > runtime["gpu_memory_max_mib"]
        or row["utilization"] > runtime["gpu_utilization_max"]
        or any(p["uuid"] == row["uuid"] for p in processes)
        or any(resources.container_uses_gpu(c, gpu, row["uuid"]) for c in containers)
    ):
        raise resources.ResourceUnavailable(f"GPU{gpu} occupied; no sharing or preemption")
    return {
        **row,
        "gpu": gpu,
        "cpu_set": runtime["cpu_sets"][arm],
        "utc": resources.utc_now(),
        "gpu_sharing": False,
    }


def wait_admission(plan, arm, update):
    deadline = time.monotonic() + plan["runtime"]["wait_timeout_hours"] * 3600
    while time.monotonic() < deadline:
        try:
            return admit(plan, arm)
        except resources.ResourceUnavailable as error:
            update({"state": "waiting_for_resources", "reason": str(error)})
            time.sleep(min(30, max(0, deadline - time.monotonic())))
    raise TimeoutError("Requested exclusive GPU did not become free; no task was preempted")


def run_stage(plan, arm, spec, batch, token, update):
    verify_snapshot(batch)
    row = wait_admission(plan, arm, update)
    stage_dir = batch / arm / "stages" / spec["label"]
    claim_stage(stage_dir, spec["label"])
    out = output_path(batch, arm, spec)
    out.mkdir(parents=True, exist_ok=True)
    out.chmod(0o755)
    name, command = build_command(plan, arm, spec, batch, token, row)
    if resources.inspect_container(name) is not None:
        raise FileExistsError("Stage container already exists; inspect original execution")
    write_json(stage_dir / "admission.json", row)
    write_json(stage_dir / "command.json", command)
    update(
        {
            "state": "running",
            "stage": spec["label"],
            "container": name,
            "gpu": row["gpu"],
            "gpu_uuid": row["uuid"],
        }
    )
    started = resources.docker(*command[3:])
    write_json(
        stage_dir / "container.json",
        {"name": name, "id": started.stdout.strip(), "utc": resources.utc_now()},
    )
    with (stage_dir / "container.log").open("x") as log:
        code = resources.monitor_container(
            name, log, time.monotonic() + plan["runtime"]["timeout_hours"] * 3600, update
        )
    state = resources.inspect_container(name)
    write_json(stage_dir / "container-state.json", state)
    write_json(
        stage_dir / "exit.json", {"code": code, "utc": resources.utc_now(), "stage": spec["label"]}
    )
    if code:
        raise RuntimeError(f"{arm}/{spec['label']} exited{code}; no automatic retry")
    result_path = out / (
        "receipt.json" if spec["family"] == "engineering" else f"result-{spec['stage']}.json"
    )
    result = read_json(result_path)
    if result.get("passed") is False:
        raise ValueError("A mandatory identity or numerical audit failed")
    write_json(
        stage_dir / "result.json",
        {"passed": True, "path": str(result_path), "sha256": sha256(result_path)},
    )
    update({"state": "stage_complete", "stage": spec["label"], "result": str(result_path)})


def run_arm(plan, arm, batch, token):
    directory = batch / arm
    directory.mkdir(parents=True, exist_ok=True)
    directory.chmod(0o755)
    state = {"arm": arm, "state": "starting", "completed_stages": []}

    def update(values):
        state.update(values)
        state["utc"] = resources.utc_now()
        write_json(directory / "controller-state.json", state)

    gpu = plan["runtime"]["gpus"][arm]
    locks = Path("/tmp") / f"lm-memory-gpu-locks-{os.getuid()}"
    locks.mkdir(exist_ok=True)
    with (locks / f"gpu-{gpu}.lock").open("a") as held:
        deadline = time.monotonic() + plan["runtime"]["wait_timeout_hours"] * 3600
        while True:
            try:
                fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError("Exclusive GPU lock wait expired") from None
                update({"state": "waiting_for_gpu_lock", "gpu": gpu})
                time.sleep(30)
        try:
            for spec in stage_plan(arm):
                run_stage(plan, arm, spec, batch, token, update)
                state["completed_stages"].append(spec["label"])
            update({"state": "complete", "new_scientific_updates": 320000})
            return copy.deepcopy(state)
        except BaseException:
            update({"state": "needs_attention", "traceback": traceback.format_exc()})
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / f"configs/{BATCH}.json")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--wandb-token-file", type=Path, default=Path.home() / ".cache/llm-memory-ttcd/wandb-token"
    )
    args = parser.parse_args()
    plan = read_json(args.config)
    validate_plan(plan)
    if not args.execute:
        print(
            json.dumps(
                {"executed": False, "plan": plan, "stages": stage_plan("transformer")}, indent=2
            )
        )
        return 0
    token = args.wandb_token_file
    if not token.is_file() or token.stat().st_mode & 0o077:
        raise ValueError("Use the existing private tracking credential file")
    root = Path(plan["runtime"]["persistent_root"])
    if not root.resolve().is_relative_to(Path("/ossfs/workspace").resolve()) or not root.is_dir():
        raise ValueError("Persistent outputs must be on the LM ossfs mount")
    batch = root / BATCH
    batch.mkdir(exist_ok=True)
    if batch.is_symlink() or not any(p.is_mount() for p in (root, *root.parents) if p != Path("/")):
        raise ValueError("Persistent mount is unavailable or redirected")
    with (batch / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (batch / "controller-process.json").exists():
            raise FileExistsError("Original controller record exists; do not duplicate submission")
        if shutil.disk_usage(batch).free < plan["runtime"]["minimum_persistent_free_gib"] * 1024**3:
            raise RuntimeError("Need at least400GiB persistent free space")
        freeze(batch, plan, args.config)
        write_json(
            batch / "controller-process.json",
            {
                "pid": os.getpid(),
                "utc": resources.utc_now(),
                "config_sha256": sha256(args.config),
                "source_manifest_sha256": sha256(batch / "source-manifest.json"),
                "gpus": plan["runtime"]["gpus"],
                "automatic_retry": False,
            },
        )
        state = {"state": "running", "pid": os.getpid(), "utc": resources.utc_now()}
        write_json(batch / "controller-state.json", state)
        try:
            run_cpu_contracts(plan, batch)
        except BaseException:
            state.update(
                state="needs_attention", traceback=traceback.format_exc(), utc=resources.utc_now()
            )
            write_json(batch / "controller-state.json", state)
            raise
        outcomes, failures = {}, {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = {pool.submit(run_arm, plan, arm, batch, token): arm for arm in NEW_ARMS}
            for future in concurrent.futures.as_completed(futures):
                arm = futures[future]
                try:
                    outcomes[arm] = future.result()
                except BaseException:
                    failures[arm] = traceback.format_exc()
                state.update(outcomes=outcomes, failures=failures, utc=resources.utc_now())
                write_json(batch / "controller-state.json", state)
        if failures:
            state["state"] = "needs_attention"
            write_json(batch / "controller-state.json", state)
            return 1
        state.update(state="waiting_for_reused_discoloop", utc=resources.utc_now())
        write_json(batch / "controller-state.json", state)
        report = batch / "source/scripts/report_ttcd_architecture.py"
        completed = subprocess.run(
            [sys.executable, str(report), "--persistent-root", str(root), "--wait-hours", "48"],
            check=False,
        )
        state.update(
            state="complete" if completed.returncode == 0 else "needs_attention",
            utc=resources.utc_now(),
            report_exit_code=completed.returncode,
        )
        write_json(batch / "controller-state.json", state)
        return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
