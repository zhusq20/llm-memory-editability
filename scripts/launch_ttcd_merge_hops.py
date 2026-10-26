#!/usr/bin/env python3
"""Queue merge/hop evaluations without modifying or rerunning one-step parents."""

from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import importlib.util
import json
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BATCH = "ttcd-merge-hops-v1"
PARENT = "ttcd-one-step-second-hop-v1"
ARCHITECTURES = ("transformer", "loop", "discoloop")
STAGES = ("preflight", "evaluate", "audit", "cloud-audit")
ADDITIONS = (
    "src/llm_memory_editability/ttcd_merge_hop_panel.py",
    "src/llm_memory_editability/ttcd_merge_hops.py",
    "scripts/run_ttcd_merge_hops.py",
    "scripts/launch_ttcd_merge_hops.py",
    "tests/test_ttcd_merge_hop_panel.py",
    "tests/test_ttcd_merge_hops.py",
    "tests/test_ttcd_merge_hops_controller.py",
    f"docs/development-artifacts/{BATCH}/protocol.md",
)


def load_parent_controller():
    spec = importlib.util.spec_from_file_location(
        "merge_hops_parent_controller", ROOT / "scripts/launch_ttcd_one_step.py"
    )
    module = importlib.util.module_from_spec(spec)
    old = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = old
    return module


parent_controller = load_parent_controller()
resources = parent_controller.resources
read_json, write_json, sha256 = resources.read_json, resources.write_json, resources.sha256
safe_path = parent_controller.safe_path
verify_entries = parent_controller.verify_entries
verify_snapshot = parent_controller.verify_snapshot
runtime_archive = parent_controller.runtime_archive
launch_copied_runtime = parent_controller.launch_copied_runtime


def validate_plan(plan):
    if (
        plan["batch"] != BATCH
        or plan["parent_batch"] != PARENT
        or plan["new_base_parameter_updates"] != 0
        or plan["writer_steps"] != 1
        or plan["parent_writer_steps"] != 4
        or plan["parent_phase"] != "stream"
        or plan["methods"] != ["continuous", "independent_merge"]
        or plan["roles"] != ["first", "second"]
        or set(plan["architectures"]) != set(ARCHITECTURES)
    ):
        raise ValueError("Only independent-increment merge and two hop roles on fixed parents")
    for arm, spec in plan["architectures"].items():
        expected = {
            "source_directory": f"{arm}/source",
            "source_manifest": f"{arm}/source-manifest.json",
            "parent_config": "configs/experiment.json"
            if arm == "discoloop"
            else f"configs/architecture-{arm}.json",
            "evaluation_directory": f"{arm}/evaluation",
        }
        if spec != expected:
            raise ValueError("Use each architecture's original immutable source and evaluation")
    runtime = plan["runtime"]
    if (
        runtime["docker_context"] != "lm-memory"
        or runtime["tracking"] != "online"
        or runtime["cpu_threads"] != 4
        or runtime["memory_gib"] != 96
        or runtime["uid"] != 1015
        or runtime["gid"] != 1016
        or runtime["cpu_sets"]
        != {
            "transformer": list(range(48, 52)),
            "loop": list(range(52, 56)),
            "discoloop": list(range(56, 60)),
        }
        or runtime["gpu_preferences"]
        != {
            "transformer": [3, 0, 5],
            "loop": [5, 1, 6],
            "discoloop": [6, 2, 7],
        }
        or runtime["minimum_persistent_free_gib"] < 200
        or runtime["gpu_memory_max_mib"] > 1024
        or runtime["gpu_utilization_max"] > 5
        or runtime["timeout_hours"] != 48
        or runtime["wait_timeout_hours"] != 96
        or not runtime["image"].startswith("sha256:")
        or len(runtime["image"]) != 71
    ):
        raise ValueError("Keep the reviewed disjoint CPU sets and exclusively idle resources")


def freeze(batch, plan, config_path, workspace=ROOT):
    if (batch / "source-manifest.json").exists():
        raise FileExistsError("Original submission is already frozen; no automatic restart")
    parent = safe_path(batch.parent, PARENT)
    parent_index = read_json(safe_path(parent, "source-manifest.json"))
    records = {}
    for arm in ARCHITECTURES:
        spec = plan["architectures"][arm]
        manifest_path = safe_path(parent, spec["source_manifest"])
        if sha256(manifest_path) != parent_index["arms"][arm]:
            raise ValueError("Original per-architecture manifest differs from parent index")
        original = read_json(manifest_path)
        source = safe_path(parent, spec["source_directory"])
        verify_entries(source, original["files"])
        if original["architecture"] != arm or original["parent_config"] != spec["parent_config"]:
            raise ValueError("Frozen parent architecture/configuration mismatch")
        inputs, data = original["parent_inputs"], Path(original["data_directory"])
        verify_entries(data.parent, inputs)
        config = read_json(source / spec["parent_config"])
        if (
            config["runtime"]["image"] != plan["runtime"]["image"]
            or config["writer"]["steps"] != 4
            or config["models"]["stream"]["step"] != 10000
            or sha256(source / "configs/continual-panel.json") != plan["panel_sha256"]
        ):
            raise ValueError("Retain the same dependency image, base model, writer and fact panel")
        directory = safe_path(batch, arm)
        directory.mkdir(exist_ok=False)
        target_source = directory / "source"
        target_source.mkdir()
        for relative in original["files"]:
            target = safe_path(target_source, relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((source / relative).read_bytes())
        for relative in ADDITIONS:
            content = safe_path(workspace, relative).read_bytes()
            target = safe_path(target_source, relative)
            if target.exists():
                if target.read_bytes() != content:
                    raise ValueError(f"Cannot replace parent science: {relative}")
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
        settings = target_source / "configs/merge-hops.json"
        if settings.exists():
            raise FileExistsError("Merge settings must be a new file")
        settings.write_bytes(config_path.read_bytes())
        verify_entries(target_source, original["files"])
        manifest = {
            "schema": "ttcd-merge-hops-source-v1",
            "utc": resources.utc_now(),
            "architecture": arm,
            "parent_batch": PARENT,
            "parent_source_directory": spec["source_directory"],
            "parent_manifest": str(manifest_path),
            "parent_manifest_sha256": sha256(manifest_path),
            "parent_index_sha256": sha256(parent / "source-manifest.json"),
            "parent_files": original["files"],
            "parent_config": spec["parent_config"],
            "parent_inputs": inputs,
            "data_directory": str(data),
            "files": resources.source_hashes(target_source),
            "host_permissions_changed": False,
            "base_parameter_updates": 0,
        }
        write_json(directory / "source-manifest.json", manifest)
        records[arm] = sha256(directory / "source-manifest.json")
    write_json(
        batch / "source-manifest.json",
        {
            "schema": "ttcd-merge-hops-freeze-v1",
            "config_sha256": sha256(config_path),
            "arms": records,
        },
    )
    return records


def parent_receipt(batch, plan, arm):
    parent = safe_path(batch.parent, PARENT)
    evaluation = safe_path(parent, plan["architectures"][arm]["evaluation_directory"])
    required = []
    for stage in ("audit", "cloud-audit"):
        result = safe_path(evaluation, f"result-{stage}.json")
        exit_path = safe_path(parent, f"{arm}/stages/{stage}/exit.json")
        required += [result, exit_path]
        if exit_path.exists() and read_json(exit_path).get("returncode") != 0:
            raise RuntimeError(f"Parent {arm}/{stage} exited unsuccessfully; no retry")
        if result.exists() and read_json(result).get("passed") is not True:
            raise ValueError(f"Parent {arm}/{stage} validation failed")
    if not all(path.is_file() for path in required):
        state_path = safe_path(parent, f"{arm}/controller-state.json")
        if state_path.exists() and read_json(state_path).get("state") in (
            "needs_attention",
            "failed",
        ):
            raise RuntimeError(f"Parent {arm} needs attention before its audit completes")
        return None
    return {
        "passed": True,
        "architecture": arm,
        "parent_batch": PARENT,
        "evaluation_directory": str(evaluation),
        "receipts": {str(path): sha256(path) for path in required},
        "utc": resources.utc_now(),
        "capability_thresholds": None,
    }


def wait_parent(batch, plan, arm, update):
    deadline = time.monotonic() + plan["runtime"]["wait_timeout_hours"] * 3600
    while time.monotonic() < deadline:
        receipt = parent_receipt(batch, plan, arm)
        if receipt:
            write_json(batch / arm / "parent-evaluation.json", receipt)
            return receipt
        update({"state": "waiting_for_parent_evaluation", "parent_batch": PARENT})
        time.sleep(min(30, max(0, deadline - time.monotonic())))
    raise TimeoutError("Parent one-step evaluation did not complete its local and cloud audits")


def acquire_resources(plan, arm, stage, update, lock_directory=None):
    # Pure graph preparation never acquires a GPU and may precede its parent's audit.
    return parent_controller.acquire_resources(
        plan,
        arm,
        "aggregate" if stage == "prepare" else stage,
        update,
        lock_directory,
    )


def build_command(plan, arm, stage, batch, token, row):
    runtime, spec = plan["runtime"], plan["architectures"][arm]
    out = batch / ("aggregate" if stage == "aggregate" else f"{arm}/evaluation")
    name = f"{BATCH}-{arm}-{stage}"
    command = [
        *resources.DOCKER,
        "create",
        "--name",
        name,
        "--user",
        "1015:1016",
        "--cpus",
        "4",
        "--cpuset-cpus",
        ",".join(map(str, runtime["cpu_sets"][arm])),
        "--memory",
        "96g",
        "--shm-size",
        "8g",
        "--network",
        "host",
        "--workdir",
        "/frozen",
    ]
    if "gpu" in row:
        command += ["--gpus", f"device={row['uuid']}"]
    manifest = read_json(batch / arm / "source-manifest.json")
    for host, target, readonly in (
        (Path(manifest["data_directory"]), "/inputs/data", True),
        (Path(manifest["data_directory"]), "/frozen/data", True),
        (batch.parent, "/frozen/results", True),
        (batch.parent / PARENT / spec["evaluation_directory"], "/parent-evaluation", True),
        (out, "/aggregate" if stage == "aggregate" else "/output", False),
        (token.resolve(), "/run/secrets/wandb", True),
    ):
        # Preparation needs only the frozen graph; evaluation may not exist yet.
        if target == "/parent-evaluation" and stage == "prepare" and not host.exists():
            continue
        command += [
            "--mount",
            f"type=bind,source={host},target={target}" + (",readonly" if readonly else ""),
        ]
    if stage == "aggregate":
        command += ["--mount", f"type=bind,source={batch},target=/comparison,readonly"]
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
        f"WANDB_TAGS=ttcd-merge-hops,{arm},{stage}",
        f"LM_TTCD_SOURCE_MANIFEST_SHA256={sha256(batch / arm / 'source-manifest.json')}",
    ):
        command += ["--env", env]
    if "gpu" in row:
        command += [
            "--env",
            f"LM_ACTUAL_GPU_INDEX={row['gpu']}",
            "--env",
            f"LM_ACTUAL_GPU_UUID={row['uuid']}",
        ]
    command += [
        runtime["image"],
        "bash",
        "-c",
        'mkdir -p "$HOME"\nexport WANDB_API_KEY="$(cat /run/secrets/wandb)"\nexec "$@"',
        "ttcd-merge-hops-entry",
        "python",
        "/frozen/scripts/run_ttcd_merge_hops.py",
        stage,
        "--parent-config",
        f"/frozen/{spec['parent_config']}",
        "--settings",
        "/frozen/configs/merge-hops.json",
        "--architecture",
        arm,
        "--parent-evaluation",
        "/parent-evaluation",
        "--out",
        "/aggregate" if stage == "aggregate" else "/output",
        "--tracking",
        "online",
        "--device",
        "cuda:0" if "gpu" in row else "cpu",
    ]
    if stage == "aggregate":
        command += ["--comparison-root", "/comparison"]
    return name, command


def run_stage(plan, arm, stage, batch, token, update):
    archive = runtime_archive(batch, arm)
    directory = batch / ("aggregate-stage" if stage == "aggregate" else f"{arm}/stages/{stage}")
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "claim.json").open("x") as stream:
        json.dump({"stage": stage, "utc": resources.utc_now(), "pid": os.getpid()}, stream)
    out = batch / ("aggregate" if stage == "aggregate" else f"{arm}/evaluation")
    out.mkdir(parents=True, exist_ok=True)
    held, row = acquire_resources(plan, arm, stage, update)
    with held:
        name, command = build_command(plan, arm, stage, batch, token, row)
        if resources.inspect_container(name) is not None:
            raise FileExistsError(
                "Stage container exists; no duplicate submission or automatic retry"
            )
        write_json(directory / "admission.json", row)
        write_json(directory / "command.json", {"argv": command})
        update({"state": "launching", "stage": stage, "container": name, "resource": row})
        try:
            created = launch_copied_runtime(command, archive, directory)
            update({"state": "running", "container_id": created.stdout.strip()})
            with (directory / "container.log").open("x") as log:
                code = resources.monitor_container(
                    name, log, time.monotonic() + plan["runtime"]["timeout_hours"] * 3600, update
                )
            write_json(directory / "container-state.json", resources.inspect_container(name))
            write_json(directory / "exit.json", {"returncode": code, "utc": resources.utc_now()})
            if code:
                raise RuntimeError(f"{arm}/{stage} exited {code}; no automatic retry")
            result_path = out / f"result-{stage}.json"
            result = read_json(result_path)
            if result.get("passed") is False:
                raise ValueError(f"Mandatory {stage} validation failed")
            write_json(
                directory / "result.json",
                {"path": str(result_path), "sha256": sha256(result_path), "passed": True},
            )
            links = {
                path.stem: read_json(path).get("url")
                for path in out.glob("wandb-*.json")
                if read_json(path).get("url")
            }
            update({"state": "stage_complete", "stage": stage, "wandb_links": links})
        except BaseException:
            if not (directory / "exit.json").exists():
                write_json(
                    directory / "exit.json",
                    {
                        "returncode": None,
                        "state": "launch_or_monitor_failed",
                        "traceback": traceback.format_exc(),
                    },
                )
            raise


def run_arm(plan, arm, batch, token):
    state = {"architecture": arm, "state": "starting", "completed_stages": []}

    def update(values):
        state.update(values)
        state["utc"] = resources.utc_now()
        write_json(batch / arm / "controller-state.json", state)

    try:
        run_stage(plan, arm, "prepare", batch, token, update)
        state["completed_stages"].append("prepare")
        endpoint = wait_parent(batch, plan, arm, update)
        update({"parent_endpoint": endpoint})
        for stage in STAGES:
            run_stage(plan, arm, stage, batch, token, update)
            state["completed_stages"].append(stage)
        update({"state": "complete", "new_base_parameter_updates": 0})
        return state
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
                {
                    "executed": False,
                    "plan": plan,
                    "stages": ["prepare", *STAGES],
                    "aggregate": True,
                },
                indent=2,
            )
        )
        return 0
    token = args.wandb_token_file
    if not token.is_file() or token.stat().st_mode & 0o077:
        raise ValueError("Use the existing private W&B credential file")
    root = Path(plan["runtime"]["persistent_root"])
    if not root.is_dir() or not root.resolve().is_relative_to(Path("/ossfs/workspace").resolve()):
        raise ValueError("Outputs must use the LM persistent mount")
    if not any(path.is_mount() for path in (root, *root.parents) if path != Path("/")):
        raise ValueError("Persistent mount is unavailable")
    batch = safe_path(root, BATCH)
    batch.mkdir(exist_ok=True)
    with (batch / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (batch / "controller-process.json").exists():
            raise FileExistsError(
                "Controller already submitted; inspect its state without restarting"
            )
        resources.check_space(batch, plan)
        freeze(batch, plan, args.config)
        write_json(
            batch / "controller-process.json",
            {
                "pid": os.getpid(),
                "utc": resources.utc_now(),
                "config_sha256": sha256(args.config),
                "source_manifest_sha256": sha256(batch / "source-manifest.json"),
                "automatic_retry": False,
                "new_base_parameter_updates": 0,
            },
        )
        state = {"state": "running", "pid": os.getpid(), "outcomes": {}, "failures": {}}

        def update(values):
            state.update(values)
            state["utc"] = resources.utc_now()
            write_json(batch / "controller-state.json", state)

        update({})
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            futures = {pool.submit(run_arm, plan, arm, batch, token): arm for arm in ARCHITECTURES}
            for future in concurrent.futures.as_completed(futures):
                arm = futures[future]
                try:
                    state["outcomes"][arm] = future.result()
                except BaseException:
                    state["failures"][arm] = traceback.format_exc()
                update({})
        if state["failures"]:
            update({"state": "needs_attention"})
            return 1
        try:
            run_stage(plan, "transformer", "aggregate", batch, token, update)
        except BaseException:
            update({"state": "needs_attention", "traceback": traceback.format_exc()})
            return 1
        update({"state": "complete", "new_base_parameter_updates": 0})
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
