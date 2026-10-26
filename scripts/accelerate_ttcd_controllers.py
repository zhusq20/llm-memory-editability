#!/usr/bin/env python3
"""Adopt existing TTCD containers; wrap only future frozen TTCD stage commands.

The default invocation prints a plan. ``--capture`` reads a handoff snapshot to
stdout. This program never signals an old controller or restarts a container.
Execution requires a snapshot captured while the old host controller was stopped,
its subsequent exit, unchanged frozen artifacts and a verified runtime extension.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import datetime as dt
import fcntl
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
PERSISTENT = Path("/ossfs/workspace/llm-memory-editability")
PARENT = "ttcd-successful-base-v1"
ARCHITECTURE = "ttcd-architecture-comparison-v1"
SUPPLEMENT = "ttcd-double-new-eval-v1"
SHARING = "gpu4-parallel-20261015"
TARGETS = {"parent": PARENT, "architecture": ARCHITECTURE, "supplement": SUPPLEMENT}
WRAPPED_SCRIPTS = {
    "run_ttcd_verified_base.py",
    "run_ttcd_architecture.py",
    "run_ttcd_double_new_eval.py",
}


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path, value, *, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive:
        with path.open("x") as stream:
            json.dump(value, stream, indent=2)
            stream.write("\n")
    else:
        temporary = path.with_suffix(path.suffix + ".partial")
        temporary.write_text(json.dumps(value, indent=2) + "\n")
        temporary.replace(path)


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def parse_utc(value):
    result = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("Handoff timestamps must have an explicit timezone")
    return result


def deadline_from_utc(value, *, wall_now=None, monotonic_now=None):
    remaining = (parse_utc(value) - (wall_now or dt.datetime.now(dt.timezone.utc))).total_seconds()
    if remaining <= 0:
        raise TimeoutError("The preserved deadline expired; handoff cannot reset it")
    return (time.monotonic() if monotonic_now is None else monotonic_now) + remaining


def expiration(anchor, hours):
    return (parse_utc(anchor) + dt.timedelta(hours=hours)).isoformat()


def load_module(path, name):
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.dont_write_bytecode = previous


def process_identity(pid):
    try:
        raw = Path(f"/proc/{pid}/stat").read_text()
    except FileNotFoundError:
        return None
    fields = raw[raw.rindex(")") + 2 :].split()
    return {"pid": pid, "state": fields[0], "start_ticks": int(fields[19])}


def verify_speedup(source):
    source = Path(source).resolve()
    manifest_path = source.parent / "source-manifest.json"
    manifest = read_json(manifest_path)
    if manifest.get("numerical_verification", {}).get("passed") is not True:
        raise ValueError("The runtime extension requires passing numerical verification")
    if "scripts/run_ttcd_speedup.py" not in manifest["files"]:
        raise ValueError("The runtime entrypoint must be frozen in its manifest")
    for relative, expected in manifest["files"].items():
        path = source / relative
        if not path.resolve().is_relative_to(source) or sha256(path) != expected:
            raise ValueError(f"Runtime source differs or escapes its root: {relative}")
    receipt = source.parent / "engineering-receipt.json"
    if receipt.exists():
        value = read_json(receipt)
        if value.get("passed") is not True or value.get("source_manifest_sha256") != sha256(
            manifest_path
        ):
            raise ValueError("Engineering receipt does not bind the runtime manifest")
    return manifest_path


def wrap_command(command, image, source, manifest, stage):
    """Keep the exact old script/arguments and all existing Docker options."""
    command = list(command)
    if command[:4] != ["docker", "--context", "lm-memory", "run"]:
        raise ValueError("Only an explicit LM Docker run command can be wrapped")
    image_index = command.index(image)
    python_index = command.index("python", image_index + 1)
    script = command[python_index + 1]
    if any("LM_TTCD_SPEEDUP_" in value for value in command):
        raise ValueError("The future container command was already wrapped")
    if Path(script).name not in WRAPPED_SCRIPTS:
        return command  # Native foundation stages keep their original execution.
    if not script.startswith("/frozen/scripts/"):
        raise ValueError("Only the original frozen TTCD entrypoints can be wrapped")
    command[python_index + 1 : python_index + 1] = [
        "/run/ttcd-speedup/scripts/run_ttcd_speedup.py",
        "--",
    ]
    command[image_index:image_index] = [
        "--mount",
        f"type=bind,source={Path(source).resolve()},target=/run/ttcd-speedup,readonly",
        "--mount",
        f"type=bind,source={Path(manifest).resolve()},target=/run/ttcd-speedup-manifest.json,readonly",
        "--env",
        "LM_TTCD_SPEEDUP_MANIFEST=/run/ttcd-speedup-manifest.json",
        "--env",
        f"LM_TTCD_SPEEDUP_MANIFEST_SHA256={sha256(manifest)}",
        "--env",
        f"LM_TTCD_SPEEDUP_STAGE={stage}",
    ]
    return command


def load_context(target, workspace, persistent):
    batch = persistent / TARGETS[target]
    if target in ("parent", "supplement"):
        amendment = batch / SHARING
        sharing = load_module(amendment / "share_ttcd_gpu4.py", f"speedup_sharing_{target}")
        policy = load_module(amendment / "ttcd_gpu4_resources.py", f"speedup_policy_{target}")
        record = read_json(amendment / "controller-process.json")
        if (
            sha256(amendment / "share_ttcd_gpu4.py") != record["controller_sha256"]
            or sha256(amendment / "ttcd_gpu4_resources.py") != record["resources_sha256"]
        ):
            raise ValueError("The original GPU4 controller/resource bytes changed")
        paths = [
            amendment / "share_ttcd_gpu4.py",
            amendment / "ttcd_gpu4_resources.py",
            amendment / "sharing-amendment.json",
            amendment / "controller-process.json",
        ]
        if target == "parent":
            controller = sharing.load_parent(batch, workspace)
            config_path = batch / "source-v2/configs/experiment.json"
            config = read_json(config_path)
            receipt = batch / "preflight-repair-20261015/engineering-receipt.json"
            controller.verify_repaired_source(batch, config, config_path, receipt)
            controller.controller.require_common_gate(batch, config)
            paths += [batch / "source-v2-manifest.json", config_path, receipt]
            helpers = controller.controller
        else:
            controller = load_module(
                batch / "source/scripts/launch_ttcd_double_new_eval.py", "speedup_supplement"
            )
            controller.ROOT = workspace
            config_path = batch / "source/configs/double-new-eval.json"
            config = read_json(config_path)
            controller.validate_settings(config)
            controller.verify_snapshot(batch, persistent / PARENT, panel_required=True)
            helpers, _ = controller.load_resources(persistent / PARENT, batch)
            receipt = None
            paths += [
                batch / "source-manifest.json",
                config_path,
                batch / "source/configs/experiment.json",
                batch / "panel.json",
                batch / "panel-manifest.json",
                batch / "preparation/result-prepare.json",
                batch / "preparation/exit-prepare.json",
            ]
        return SimpleNamespace(
            target=target,
            batch=batch,
            controller=controller,
            helpers=helpers,
            config=config,
            config_path=config_path,
            receipt=receipt,
            sharing=sharing,
            policy=policy,
            record=record,
            protected=paths,
        )
    controller = load_module(
        batch / "source/scripts/launch_ttcd_architecture.py", "speedup_architecture"
    )
    config_path = batch / "source/configs/architecture-comparison.json"
    config = read_json(config_path)
    controller.validate_plan(config)
    controller.verify_snapshot(batch)
    record = read_json(batch / "controller-process.json")
    return SimpleNamespace(
        target=target,
        batch=batch,
        controller=controller,
        helpers=controller.resources,
        config=config,
        config_path=config_path,
        record=record,
        protected=[batch / "source-manifest.json", config_path, batch / "controller-process.json"],
    )


def active_job(context, arm):
    batch, helpers = context.batch, context.helpers
    if context.target == "parent":
        out = batch / arm
        state_path = out / "controller-state-v2.json"
        state = read_json(state_path)
        stage = state["stage"]
        if stage != "train" or state["state"] != "running":
            raise ValueError("This bounded handoff only adopts the two original running trains")
        claim, command = out / "claim-train.json", out / "command-train.json"
        name, expected_id = state["container"], state["container_id"]
        if (out / "exit-train.json").exists():
            raise FileExistsError("Preserve the existing train exit; do not retry adoption")
        anchor = read_json(batch / "common/exit-cloud-audit.json")["utc"]
        expires = expiration(anchor, context.config["runtime"]["timeout_hours"])
        context.protected.append(batch / "common/exit-cloud-audit.json")
    else:
        state_path = batch / arm / "controller-state.json"
        state = read_json(state_path)
        stage = state["stage"]
        if stage != "foundation-train" or state["completed_stages"] != ["foundation-preflight"]:
            raise ValueError("This bounded handoff only adopts original native foundation trains")
        out = batch / arm / "stages" / stage
        claim, command = out / f"claim-{stage}.json", out / "command.json"
        container_record = read_json(out / "container.json")
        name, expected_id = container_record["name"], container_record["id"]
        if (out / "exit.json").exists():
            raise FileExistsError("Preserve the original foundation stage exit")
        expires = expiration(container_record["utc"], context.config["runtime"]["timeout_hours"])
        context.protected.append(out / "container.json")
        finished = batch / arm / "stages/foundation-preflight"
        receipt = read_json(finished / "result.json")
        if read_json(finished / "exit.json")["code"] != 0 or receipt.get("passed") is not True:
            raise ValueError("The original completed foundation preflight must pass")
        if sha256(receipt["path"]) != receipt["sha256"]:
            raise ValueError("Completed foundation preflight result changed")
        context.protected += [
            finished / "exit.json",
            finished / "result.json",
            Path(receipt["path"]),
        ]
    value = helpers.inspect_container(name)
    if value is None or value["Id"] != expected_id or not value["State"]["Running"]:
        raise ValueError("Snapshot requires the exact original model container to be running")
    if read_json(claim)["pid"] != context.record["pid"]:
        raise ValueError("The training claim belongs to another host controller")
    context.protected += [claim, command]
    return {
        "arm": arm,
        "stage": stage,
        "container": name,
        "container_id": value["Id"],
        "host_model_pid": value["State"]["Pid"],
        "deadline_utc": expires,
        "deadline_policy": "conservative original anchor; no deadline reset or extension",
        "state": state,
    }


def capture_snapshot(context):
    old = process_identity(context.record["pid"])
    if old is None or old["state"] == "Z":
        raise ValueError("Capture the original host controller before terminating it")
    active = []
    wait_expires = None
    if context.target == "supplement":
        state = read_json(context.batch / SHARING / "controller-state.json")
        if state.get("state") not in (
            "waiting_for_parent_completion",
            "waiting_for_parent_gpu4_cohort_lock",
        ):
            raise ValueError("Supplementary handoff only supports its original waiting controller")
        context.sharing.require_unclaimed(
            context.batch, context.controller.ARMS, context.controller.STAGES
        )
        wait_expires = expiration(
            context.record["utc"], context.config["runtime"]["wait_timeout_hours"]
        )
    else:
        arms = ("single", "stream") if context.target == "parent" else ("transformer", "loop")
        active = [active_job(context, arm) for arm in arms]
    return {
        "schema": "ttcd-speedup-handoff-v1",
        "target": context.target,
        "batch": str(context.batch),
        "captured_utc": utc_now(),
        "old_controller": old,
        "active": active,
        "wait_deadline_utc": wait_expires,
        "protected_files": {str(p): sha256(p) for p in context.protected},
        "model_restart": False,
        "preparation_repeated": False,
        "scientific_arguments_changed": False,
        "signals_sent_by_this_program": False,
    }


def validate_handoff(snapshot, context):
    if (
        snapshot.get("schema") != "ttcd-speedup-handoff-v1"
        or snapshot["target"] != context.target
        or Path(snapshot["batch"]).resolve() != context.batch.resolve()
        or snapshot["old_controller"]["state"] not in ("T", "t")
    ):
        raise ValueError("Execution requires a matching snapshot captured after host-only STOP")
    pid = snapshot["old_controller"]["pid"]
    if pid != context.record["pid"]:
        raise ValueError("The recorded old host controller identity differs")
    current = process_identity(pid)
    if current is not None and current["state"] != "Z":
        raise ValueError("The old host controller must exit before its locks are adopted")
    expected_arms = {
        "parent": {"single", "stream"},
        "architecture": {"transformer", "loop"},
        "supplement": set(),
    }[context.target]
    if (
        len(snapshot["active"]) != len(expected_arms)
        or {j["arm"] for j in snapshot["active"]} != expected_arms
    ):
        raise ValueError("The handoff must preserve every original arm exactly once")
    for path, expected in snapshot["protected_files"].items():
        if sha256(path) != expected:
            raise ValueError(f"A protected original artifact changed: {path}")
    for job in snapshot["active"]:
        if context.target == "parent":
            anchor = read_json(context.batch / "common/exit-cloud-audit.json")["utc"]
            expected_stage = "train"
        else:
            anchor = read_json(
                context.batch / job["arm"] / "stages/foundation-train/container.json"
            )["utc"]
            expected_stage = "foundation-train"
        if job["stage"] != expected_stage or job["deadline_utc"] != expiration(
            anchor, context.config["runtime"]["timeout_hours"]
        ):
            raise ValueError("The original stage/deadline cannot be changed during handoff")
        container = context.helpers.inspect_container(job["container"])
        if container is None or container["Id"] != job["container_id"]:
            raise ValueError("An original model container disappeared or was replaced")
        state = container["State"]
        if state["Running"]:
            if state["Pid"] != job["host_model_pid"]:
                raise ValueError("An original model process was restarted")
        elif state.get("Status") != "exited" or state["ExitCode"] != 0:
            raise ValueError("An original model failed; adoption cannot retry it")
        deadline_from_utc(job["deadline_utc"])
    if snapshot["wait_deadline_utc"]:
        if context.target != "supplement" or snapshot["wait_deadline_utc"] != expiration(
            context.record["utc"], context.config["runtime"]["wait_timeout_hours"]
        ):
            raise ValueError("The original waiting deadline cannot be reset")
        deadline_from_utc(snapshot["wait_deadline_utc"])
    elif context.target == "supplement":
        raise ValueError("The original supplementary waiting deadline is required")


@contextlib.contextmanager
def locks(paths):
    with contextlib.ExitStack() as stack:
        for path in paths:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = stack.enter_context(path.open("a"))
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def monitor_adopted(context, job, deadline, directory, update):
    # No admission, command builder, docker run or model mutation on this path.
    with (directory / f"{job['arm']}-adopted.log").open("x") as log:
        return context.helpers.monitor_container(job["container"], log, deadline, update)


def install_parent_build(context, source, manifest):
    repair = context.controller
    original = repair.build_command

    def build(config, phase, stage, batch, token, row):
        name, command = original(config, phase, stage, batch, token, row)
        command = context.sharing.add_resource_metadata(
            command,
            config["runtime"]["image"],
            row,
            batch / SHARING / "sharing-amendment.json",
        )
        return name, wrap_command(
            command, config["runtime"]["image"], source, manifest, f"parent:{phase}:{stage}"
        )

    repair.build_command = build
    context.sharing.guard_monitor(context.helpers, context.policy)


def run_parent_arm(context, job, token, directory):
    repair, helpers, batch = context.controller, context.helpers, context.batch
    phase, config = job["arm"], context.config
    deadline = deadline_from_utc(job["deadline_utc"])
    state = {**job["state"], "speedup_handoff": True, "deadline_utc": job["deadline_utc"]}

    def update(value):
        state.update(value)
        helpers.write_json(
            batch / phase / "controller-state-v2.json", {**state, "updated_utc": utc_now()}
        )

    with locks([batch / phase / "controller.lock"]):
        helpers.require_common_gate(batch, config)
        code = monitor_adopted(context, job, deadline, directory, update)
        write_json(
            batch / phase / "exit-train.json",
            {
                "returncode": code,
                "container": job["container"],
                "container_id": job["container_id"],
                "adopted": True,
                "utc": utc_now(),
            },
            exclusive=True,
        )
        if code:
            update({"state": "needs_attention", "returncode": code})
            return code
        for stage in helpers.ARM_STAGES:
            if stage == "train":
                continue
            repair.verify_repaired_source(batch, config, context.config_path, context.receipt)
            if time.monotonic() >= deadline:
                raise TimeoutError("Original parent phase deadline before its next stage")
            row = context.policy.admission(helpers, config["runtime"], phase)
            code = repair.run_stage(batch, config, phase, stage, token, deadline, row, update)
            if code:
                update({"state": "needs_attention", "returncode": code})
                return code
        update({"state": "complete", "finished_utc": utc_now()})
        return 0


def execute_parent(context, snapshot, token, source, manifest, directory):
    batch = context.batch
    install_parent_build(context, source, manifest)
    with locks(
        [
            batch / p
            for p in ("controller.lock", "resume-controller-v2.lock", "resource-controller.lock")
        ]
    ):
        with context.policy.acquire_gpu4_lock():
            context.policy.verify_gpu4_cohort(context.helpers)
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                tasks = {
                    j["arm"]: pool.submit(run_parent_arm, context, j, token, directory)
                    for j in snapshot["active"]
                }
                exits = {arm: task.result() for arm, task in tasks.items()}
    context.helpers.write_json(
        batch / "controller-state-v2.json",
        {
            "state": "complete" if not any(exits.values()) else "needs_attention",
            "exit_codes": exits,
            "speedup_handoff": True,
            "finished_utc": utc_now(),
        },
    )
    return int(any(exits.values()))


def finish_architecture_stage(context, job, deadline, directory, update):
    controller, batch = context.controller, context.batch
    arm = job["arm"]
    spec = next(s for s in controller.stage_plan(arm) if s["label"] == job["stage"])
    stage_dir = batch / arm / "stages" / spec["label"]
    code = monitor_adopted(context, job, deadline, directory, update)
    write_json(
        stage_dir / "container-state.json", context.helpers.inspect_container(job["container"])
    )
    write_json(
        stage_dir / "exit.json",
        {"code": code, "utc": utc_now(), "stage": spec["label"], "adopted": True},
        exclusive=True,
    )
    if code:
        raise RuntimeError("Original foundation failed; no retry or continuation")
    result_path = controller.output_path(batch, arm, spec) / f"result-{spec['stage']}.json"
    if read_json(result_path).get("passed") is False:
        raise ValueError("Original foundation result failed")
    write_json(
        stage_dir / "result.json",
        {"passed": True, "path": str(result_path), "sha256": sha256(result_path)},
        exclusive=True,
    )


def run_architecture_arm(context, job, token, directory):
    controller, batch, arm = context.controller, context.batch, job["arm"]
    state = dict(job["state"])
    state["completed_stages"] = list(state["completed_stages"])
    state["speedup_handoff"] = True

    def update(values):
        state.update(values)
        controller.write_json(batch / arm / "controller-state.json", {**state, "utc": utc_now()})

    gpu = context.config["runtime"]["gpus"][arm]
    lock = Path("/tmp") / f"lm-memory-gpu-locks-{os.getuid()}" / f"gpu-{gpu}.lock"
    with locks([lock]):
        finish_architecture_stage(
            context, job, deadline_from_utc(job["deadline_utc"]), directory, update
        )
        state["completed_stages"].append(job["stage"])
        stages = controller.stage_plan(arm)
        start = next(i for i, s in enumerate(stages) if s["label"] == job["stage"]) + 1
        for spec in stages[start:]:
            controller.run_stage(context.config, arm, spec, batch, token, update)
            state["completed_stages"].append(spec["label"])
        update({"state": "complete", "new_scientific_updates": 320000})
    return state


def execute_architecture(context, snapshot, token, source, manifest, directory):
    controller, batch = context.controller, context.batch
    original = controller.build_command

    def build(plan, arm, spec, batch, token, row):
        name, command = original(plan, arm, spec, batch, token, row)
        return name, wrap_command(
            command,
            plan["runtime"]["image"],
            source,
            manifest,
            f"architecture:{arm}:{spec['label']}",
        )

    controller.build_command = build
    with locks([batch / "controller.lock"]):
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            tasks = [
                pool.submit(run_architecture_arm, context, job, token, directory)
                for job in snapshot["active"]
            ]
            for task in tasks:
                task.result()
        controller.write_json(
            batch / "controller-state.json",
            {"state": "waiting_for_reused_discoloop", "speedup_handoff": True, "utc": utc_now()},
        )
        code = subprocess.run(
            [
                sys.executable,
                str(batch / "source/scripts/report_ttcd_architecture.py"),
                "--persistent-root",
                str(batch.parent),
                "--wait-hours",
                "48",
            ],
            check=False,
        ).returncode
        controller.write_json(
            batch / "controller-state.json",
            {
                "state": "complete" if code == 0 else "needs_attention",
                "speedup_handoff": True,
                "report_exit_code": code,
                "utc": utc_now(),
            },
        )
        return code


def execute_supplement(context, snapshot, token, source, manifest, directory):
    controller, helpers, policy, batch = (
        context.controller,
        context.helpers,
        context.policy,
        context.batch,
    )
    parent = batch.parent / PARENT
    context.sharing.install_supplement_resources(
        controller, helpers, policy, batch / SHARING / "sharing-amendment.json"
    )
    original = controller.build_command

    def build(settings, arm, stage, batch, parent, token, row=None):
        name, command = original(settings, arm, stage, batch, parent, token, row)
        return name, wrap_command(
            command, settings["runtime"]["image"], source, manifest, f"supplement:{arm}:{stage}"
        )

    controller.build_command = build
    with locks([batch / "controller.lock"]):
        context.sharing.require_unclaimed(batch, controller.ARMS, controller.STAGES)
        base = read_json(batch / "source/configs/experiment.json")
        deadline = deadline_from_utc(snapshot["wait_deadline_utc"])
        controller.wait_dependencies(parent, base, "single", deadline)
        with context.sharing.wait_cohort_lock(
            policy, deadline, lambda value: write_json(directory / "state.json", value)
        ):
            policy.verify_gpu4_cohort(helpers)
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                tasks = [
                    pool.submit(
                        controller.run_arm,
                        context.config,
                        arm,
                        list(controller.STAGES),
                        batch,
                        parent,
                        token,
                        helpers,
                        policy,
                        base,
                    )
                    for arm in controller.ARMS
                ]
                exits = [task.result() for task in tasks]
        write_json(
            batch / SHARING / "controller-state.json",
            {
                "state": "complete" if not any(exits) else "needs_attention",
                "exit_codes": exits,
                "scientific_training_updates": 0,
                "speedup_handoff": True,
            },
        )
    return int(any(exits))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=tuple(TARGETS), required=True)
    parser.add_argument("--workspace", type=Path, default=ROOT)
    parser.add_argument("--persistent-root", type=Path, default=PERSISTENT)
    parser.add_argument(
        "--speedup-source", type=Path, default=PERSISTENT / "ttcd-speedup-v1/source"
    )
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--wandb-token-file", type=Path, default=Path.home() / ".cache/llm-memory-ttcd/wandb-token"
    )
    args = parser.parse_args()
    if args.capture and args.execute:
        parser.error("Capture is read-only and cannot execute a handoff")
    if not args.capture and not args.execute:
        print(
            json.dumps(
                {
                    "target": args.target,
                    "executed": False,
                    "model_restart": False,
                    "signals_sent": False,
                    "wrapper_applies_to_future_TTCD_containers_only": True,
                }
            )
        )
        return 0
    context = load_context(args.target, args.workspace.resolve(), args.persistent_root.resolve())
    if args.capture:
        print(json.dumps(capture_snapshot(context), indent=2))
        return 0
    if args.snapshot is None:
        parser.error("Execution requires the stopped-host snapshot file")
    token = args.wandb_token_file
    if not token.is_file() or token.stat().st_mode & 0o077:
        raise ValueError("Use the existing private credential file without exposing its contents")
    snapshot = read_json(args.snapshot)
    validate_handoff(snapshot, context)
    manifest = verify_speedup(args.speedup_source)
    directory = args.speedup_source.parent / "handoffs" / args.target
    directory.mkdir(parents=True, exist_ok=True)
    write_json(
        directory / "claim.json",
        {
            "pid": os.getpid(),
            "utc": utc_now(),
            "snapshot_sha256": sha256(args.snapshot),
            "source_manifest_sha256": sha256(manifest),
            "host_adapter_sha256": sha256(Path(__file__)),
            "original_controller": snapshot["old_controller"],
            "active_original_models": snapshot["active"],
            "automatic_retry": False,
        },
        exclusive=True,
    )
    execute = {
        "parent": execute_parent,
        "architecture": execute_architecture,
        "supplement": execute_supplement,
    }[args.target]
    try:
        code = execute(context, snapshot, token, args.speedup_source, manifest, directory)
    except BaseException:
        write_json(
            directory / "result.json",
            {"state": "needs_attention", "utc": utc_now(), "automatic_retry": False},
        )
        raise
    write_json(
        directory / "result.json",
        {
            "returncode": code,
            "state": "complete" if code == 0 else "needs_attention",
            "utc": utc_now(),
        },
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
