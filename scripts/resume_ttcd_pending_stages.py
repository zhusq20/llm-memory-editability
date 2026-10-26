"""Recover diagnosed host launch failures and allow future exclusive idle GPUs.

This host-only amendment adopts existing model containers. Scientific source,
arguments, completed stages, live model processes and deadlines are preserved.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import copy
import fcntl
import importlib.util
import json
import os
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

ROOT = Path("/mnt/disk2_from_server2/siqizhu4/llm-memory-editability")
BASE = Path("/ossfs/workspace/llm-memory-editability")
SPEEDUP = BASE / "ttcd-speedup-v1"
AMENDMENT = SPEEDUP / "pending-stage-dependency-repair-v2-20261015"


def module(path, name):
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        result = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(result)
        return result
    finally:
        sys.dont_write_bytecode = previous


def direct_launch_proxy(helpers):
    """The repaired parent uses subprocess.run, rather than helpers.docker."""

    def run(command, **kwargs):
        if list(command[:4]) != ["docker", "--context", "lm-memory", "run"]:
            raise ValueError("Only the original explicit LM launch is intercepted")
        return helpers.docker(*command[3:], check=False)

    # Replace this controller's module reference, never subprocess.run globally.
    return SimpleNamespace(run=run)


def successful_prefix(stages, receipt, result):
    completed, missing = [], False
    for stage in stages:
        exit_path = receipt(stage)
        if not exit_path.exists():
            missing = True
            continue
        value = json.loads(exit_path.read_text())
        code = value.get("code", value.get("returncode"))
        if code != 0:
            missing = True
            continue
        if missing:
            raise ValueError("Successful stages must be a contiguous original prefix")
        output = result(stage)
        if (
            not output.is_file()
            or json.loads(output.read_text()).get("passed") is False
        ):
            raise ValueError("Completed stage lacks a successful result")
        completed.append(stage)
    return completed


def failed_launch_files(out, helpers):
    exit_path = out / "exit-train-audit.json"
    value = helpers.read_json(exit_path)
    command = helpers.read_json(out / "command-train-audit.json")["argv"]
    name = command[command.index("--name") + 1]
    container = helpers.inspect_container(name)
    log = (out / "train-audit.log").read_text()
    mount_failure = value == {"returncode": 126, "launch_failed": True}
    import_failure = (
        value.get("returncode") == 1
        and container is not None
        and container["State"].get("Status") == "exited"
        and container["State"]["ExitCode"] == 1
        and "ImportError: cannot import name 'ttcd_double_new_eval'" in log
        and "eval_module = load_extension" in log
    )
    if mount_failure:
        if container is not None:
            raise ValueError("The failed launch unexpectedly created a model container")
        if "invalid mount config" not in log or str(SPEEDUP / "source") not in log:
            raise ValueError("Failure does not match the diagnosed runtime mount")
    elif not import_failure:
        raise ValueError(
            "Only diagnosed pre-execution mount/import failures are recoverable"
        )
    if (out / "result-train-audit.json").exists():
        raise ValueError("Never retry an executed scientific audit")
    files = [
        out / name
        for name in (
            "claim-train-audit.json",
            "command-train-audit.json",
            "admission-train-audit.json",
            "train-audit.log",
            "exit-train-audit.json",
        )
    ]
    runtime = out / "runtime-acceleration-train-audit.json"
    if import_failure:
        if not runtime.is_file():
            raise ValueError("Import failure requires the original runtime receipt")
        files.append(runtime)
    return files


def context(adapter, target):
    return adapter.load_context(target, ROOT, BASE)


def capture(adapter, ctx):
    record = adapter.read_json(ctx.batch / "controller-process-speedup.json")
    identity = adapter.process_identity(record["pid"])
    if identity is None:
        previous_result = Path(record["snapshot"]).parent / "result.json"
        if ctx.target != "parent" or adapter.read_json(previous_result).get(
            "codes"
        ) != [1, 1]:
            raise ValueError(
                "Only the diagnosed exited parent controller may be adopted"
            )
        identity = {"pid": record["pid"], "state": "exited"}
    elif identity["state"] not in ("T", "t"):
        raise ValueError("Capture requires the stopped current host controller")
    protected = list(ctx.protected)
    jobs = []
    arms = ("single", "stream") if ctx.target == "parent" else ("transformer", "loop")
    for arm in arms:
        if ctx.target == "parent":
            out = ctx.batch / arm
            state = adapter.read_json(out / "controller-state-v2.json")
            stages = list(ctx.helpers.ARM_STAGES)

            def exit_path(stage):
                return out / f"exit-{stage}.json"

            def result_path(stage):
                return out / f"result-{stage}.json"
        else:
            out = ctx.batch / arm
            state = adapter.read_json(out / "controller-state.json")
            specs = ctx.controller.stage_plan(arm)
            stages = [spec["label"] for spec in specs]

            def exit_path(stage):
                return out / "stages" / stage / "exit.json"

            def result_path(stage):
                return out / "stages" / stage / "result.json"

        completed = successful_prefix(stages, exit_path, result_path)
        for stage in completed:
            protected.extend((exit_path(stage), result_path(stage)))
            if ctx.target == "architecture":
                receipt = adapter.read_json(result_path(stage))
                if adapter.sha256(receipt["path"]) != receipt["sha256"]:
                    raise ValueError("Completed architecture result changed")
                protected.append(Path(receipt["path"]))
        if len(completed) == len(stages):
            raise ValueError("Do not replace a completed controller arm")
        stage = stages[len(completed)]
        job = dict(arm=arm, state=state, completed=completed, stage=stage, active=None)
        if ctx.target == "parent" and arm == "single":
            if completed != ["train"] or stage != "train-audit":
                raise ValueError(
                    "Single-fact recovery must follow the original completed train"
                )
            failed = failed_launch_files(out, ctx.helpers)
            protected.extend(failed)
            job["archive"] = [str(path) for path in failed]
        elif state.get("stage") == stage and (
            state.get("state") == "running"
            or (
                ctx.target == "parent"
                and arm == "stream"
                and "GPU4 has CUDA processes outside the authorized cohort"
                in state.get("traceback", "")
            )
        ):
            container = ctx.helpers.inspect_container(state["container"])
            if container is None or not container["State"]["Running"]:
                raise ValueError(
                    "The original active container must still be running at capture"
                )
            job["active"] = {
                "name": state["container"],
                "id": container["Id"],
                "pid": container["State"]["Pid"],
            }
        elif (
            ctx.target != "architecture"
            or arm != "loop"
            or stage != "common-train-audit"
        ):
            raise ValueError("Unexpected waiting stage; no automatic recovery")
        if ctx.target == "parent":
            job["deadline_utc"] = state["deadline_utc"]
        elif job["active"]:
            launch = adapter.read_json(out / "stages" / stage / "container.json")
            job["deadline_utc"] = adapter.expiration(
                launch["utc"], ctx.config["runtime"]["timeout_hours"]
            )
        else:
            # The existing wait started immediately after common-train completed.
            anchor = adapter.read_json(exit_path(completed[-1]))["utc"]
            job["wait_deadline_utc"] = adapter.expiration(
                anchor, ctx.config["runtime"]["wait_timeout_hours"]
            )
        jobs.append(job)
    return {
        "target": ctx.target,
        "old_controller": identity,
        "old_record": record,
        "captured_utc": adapter.utc_now(),
        "jobs": jobs,
        "protected_files": {str(path): adapter.sha256(path) for path in protected},
        "model_restarted": False,
        "scientific_recipe_changed": False,
    }


def validate(adapter, ctx, snapshot):
    if snapshot["target"] != ctx.target:
        raise ValueError("Wrong target snapshot")
    old = adapter.process_identity(snapshot["old_controller"]["pid"])
    if old is not None and old["state"] != "Z":
        raise ValueError("Old host controller must exit before adoption")
    for path, expected in snapshot["protected_files"].items():
        if adapter.sha256(path) != expected:
            raise ValueError(f"Protected original artifact changed: {path}")
    for job in snapshot["jobs"]:
        if job["active"]:
            before = job["active"]
            now = ctx.helpers.inspect_container(before["name"])
            if now is None or now["Id"] != before["id"]:
                raise ValueError("Original active model container changed")
            if now["State"]["Running"]:
                if now["State"]["Pid"] != before["pid"]:
                    raise ValueError("Original model process restarted")
            elif now["State"]["Status"] != "exited" or now["State"]["ExitCode"]:
                raise ValueError("Original model failed; do not retry training")
        adapter.deadline_from_utc(job.get("deadline_utc", job.get("wait_deadline_utc")))


def adopt(adapter, ctx, job, directory, update):
    active = job["active"]
    with (directory / f"{job['arm']}-adopted.log").open("x") as log:
        code = ctx.helpers.monitor_container(
            active["name"], log, adapter.deadline_from_utc(job["deadline_utc"]), update
        )
    if ctx.target == "parent":
        out = ctx.batch / job["arm"]
        receipt = out / f"exit-{job['stage']}.json"
        value = dict(returncode=code, container=active["name"], adopted=True)
        result = out / f"result-{job['stage']}.json"
    else:
        spec = next(
            spec
            for spec in ctx.controller.stage_plan(job["arm"])
            if spec["label"] == job["stage"]
        )
        out = ctx.batch / job["arm"] / "stages" / job["stage"]
        receipt = out / "exit.json"
        value = dict(code=code, stage=job["stage"], adopted=True)
        result = (
            ctx.controller.output_path(ctx.batch, job["arm"], spec)
            / f"result-{spec['stage']}.json"
        )
        adapter.write_json(
            out / "container-state.json", ctx.helpers.inspect_container(active["name"])
        )
    adapter.write_json(receipt, {**value, "utc": adapter.utc_now()}, exclusive=True)
    if code or not result.is_file() or adapter.read_json(result).get("passed") is False:
        raise RuntimeError("Original stage failed; no automatic retry")
    if ctx.target == "architecture":
        adapter.write_json(
            out / "result.json",
            {"passed": True, "path": str(result), "sha256": adapter.sha256(result)},
            exclusive=True,
        )


def exclusive_choice(ctx, arm, deadline, update, original_admit):
    directory = Path("/tmp") / f"lm-memory-gpu-locks-{os.getuid()}"
    directory.mkdir(exist_ok=True)
    while time.monotonic() < deadline:
        rows, _ = ctx.helpers.gpu_snapshot()
        preferences = [ctx.config["runtime"]["gpus"][arm], *sorted(rows)]
        reasons = {}
        for gpu in dict.fromkeys(preferences):
            held = (directory / f"gpu-{gpu}.lock").open("a")
            try:
                fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
                plan = copy.deepcopy(ctx.config)
                plan["runtime"]["gpus"][arm] = gpu
                row = original_admit(plan, arm)
                return held, row
            except (BlockingIOError, ctx.helpers.ResourceUnavailable) as error:
                held.close()
                reasons[str(gpu)] = str(error)
            except BaseException:
                held.close()
                raise
        update(
            {
                "state": "waiting_for_resources",
                "gpu_fallback": "any exclusively idle GPU",
                "reasons": reasons,
            }
        )
        time.sleep(min(30, max(0, deadline - time.monotonic())))
    raise TimeoutError("Original exclusive GPU wait expired")


def wait_parent_admission(ctx, phase, deadline, update):
    """New work waits for the authorized cohort; running models keep monitoring."""
    while time.monotonic() < deadline:
        try:
            return ctx.policy.admission(ctx.helpers, ctx.config["runtime"], phase)
        except ctx.helpers.ResourceUnavailable as error:
            update(
                {
                    "state": "waiting_for_resources",
                    "reason": str(error),
                    "container_running": False,
                }
            )
            time.sleep(min(30, max(0, deadline - time.monotonic())))
    raise TimeoutError(
        "Original parent phase deadline expired before resource admission"
    )


def execute(adapter, transport, ctx, snapshot, directory):
    validate(adapter, ctx, snapshot)
    source = AMENDMENT / "runtime/source"
    manifest = adapter.verify_speedup(source)
    transport.install(ctx.helpers, source, manifest, directory / "transport")
    token = Path.home() / ".cache/llm-memory-ttcd/wandb-token"
    if not token.is_file() or token.stat().st_mode & 0o077:
        raise ValueError("Original tracking credential must remain private")
    local = threading.local()
    if ctx.target == "parent":
        ctx.controller.subprocess = direct_launch_proxy(ctx.helpers)
        original_build = ctx.controller.build_command

        def parent_build(config, phase, stage, batch, token_file, row):
            name, command = original_build(config, phase, stage, batch, token_file, row)
            command = ctx.sharing.add_resource_metadata(
                command,
                config["runtime"]["image"],
                row,
                batch / adapter.SHARING / "sharing-amendment.json",
            )
            command = adapter.wrap_command(
                command,
                config["runtime"]["image"],
                source,
                manifest,
                f"parent:{phase}:{stage}",
            )
            if phase == "single" and stage == "train-audit":
                name += "-dependency-v1"
                command[command.index("--name") + 1] = name
            return name, command

        ctx.controller.build_command = parent_build
    else:
        original_build, original_admit = (
            ctx.controller.build_command,
            ctx.controller.admit,
        )

        def build(plan, arm, spec, batch, token_file, row):
            name, command = original_build(plan, arm, spec, batch, token_file, row)
            return name, adapter.wrap_command(
                command,
                plan["runtime"]["image"],
                source,
                manifest,
                f"architecture:{arm}:{spec['label']}",
            )

        def admitted(plan, arm, update):
            check = copy.deepcopy(plan)
            check["runtime"]["gpus"][arm] = local.row["gpu"]
            return original_admit(check, arm)

        ctx.controller.build_command = build
        ctx.controller.wait_admission = admitted

    def run(job):
        state = dict(job["state"])
        state.pop("traceback", None)
        state["completed_stages"] = list(job["completed"])
        state_path = (
            ctx.batch
            / job["arm"]
            / (
                "controller-state-v2.json"
                if ctx.target == "parent"
                else "controller-state.json"
            )
        )

        def update(values):
            state.update(values)
            stamp = adapter.utc_now()
            state.update(
                updated_utc=stamp, utc=stamp, pending_stage_repair=str(directory)
            )
            adapter.write_json(state_path, state)

        try:
            with adapter.locks([ctx.batch / job["arm"] / "controller.lock"]):
                if job["active"]:
                    update({"state": "running", "stage": job["stage"], "adopted": True})
                    adopt(adapter, ctx, job, directory, update)
                    state["completed_stages"].append(job["stage"])
                if ctx.target == "parent":
                    if job.get("archive"):
                        saved = directory / "failed-attempts" / job["arm"]
                        saved.mkdir(parents=True, exist_ok=False)
                        for path in job["archive"]:
                            path = Path(path)
                            path.rename(saved / path.name)
                    deadline = adapter.deadline_from_utc(job["deadline_utc"])
                    for stage in ctx.helpers.ARM_STAGES[
                        len(state["completed_stages"]) :
                    ]:
                        ctx.controller.verify_repaired_source(
                            ctx.batch, ctx.config, ctx.config_path, ctx.receipt
                        )
                        update({"stage": stage})
                        row = wait_parent_admission(ctx, job["arm"], deadline, update)
                        code = ctx.controller.run_stage(
                            ctx.batch,
                            ctx.config,
                            job["arm"],
                            stage,
                            token,
                            deadline,
                            row,
                            update,
                        )
                        if code:
                            raise RuntimeError(
                                f"Stage {stage} failed; no automatic retry"
                            )
                        state["completed_stages"].append(stage)
                else:
                    specs = ctx.controller.stage_plan(job["arm"])
                    for spec in specs[len(state["completed_stages"]) :]:
                        if job.get("wait_deadline_utc"):
                            deadline = adapter.deadline_from_utc(
                                job.pop("wait_deadline_utc")
                            )
                        else:
                            deadline = (
                                time.monotonic()
                                + ctx.config["runtime"]["wait_timeout_hours"] * 3600
                            )
                        update({"stage": spec["label"], "container_running": False})
                        held, row = exclusive_choice(
                            ctx, job["arm"], deadline, update, original_admit
                        )
                        with held:
                            local.row = row
                            ctx.controller.run_stage(
                                ctx.config, job["arm"], spec, ctx.batch, token, update
                            )
                        state["completed_stages"].append(spec["label"])
                update({"state": "complete", "container_running": False})
            return 0
        except BaseException:
            update(
                {
                    "state": "needs_attention",
                    "traceback": traceback.format_exc(),
                    "automatic_retry": False,
                }
            )
            return 1

    lock_names = ["controller.lock"]
    if ctx.target == "parent":
        lock_names += ["resume-controller-v2.lock", "resource-controller.lock"]
    with adapter.locks([ctx.batch / name for name in lock_names]):
        adapter.write_json(
            directory / "claim.json",
            {"pid": os.getpid(), "utc": adapter.utc_now()},
            exclusive=True,
        )
        adapter.write_json(
            ctx.batch / "controller-process-speedup.json",
            {
                "pid": os.getpid(),
                "utc": adapter.utc_now(),
                "target": ctx.target,
                "source": str(Path(__file__).resolve()),
                "snapshot": str(directory / "snapshot.json"),
                "source_manifest": str(AMENDMENT / "source-manifest.json"),
                "source_manifest_sha256": adapter.sha256(
                    AMENDMENT / "source-manifest.json"
                ),
                "old_pid": snapshot["old_controller"]["pid"],
                "model_restarted": False,
                "scientific_recipe_changed": False,
            },
        )
        top = ctx.batch / (
            "controller-state-v2.json"
            if ctx.target == "parent"
            else "controller-state.json"
        )
        adapter.write_json(
            top, {"state": "running", "pid": os.getpid(), "utc": adapter.utc_now()}
        )
        with contextlib.ExitStack() as stack:
            if ctx.target == "parent":
                stack.enter_context(ctx.policy.acquire_gpu4_lock())
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                codes = list(pool.map(run, snapshot["jobs"]))
        if ctx.target == "architecture" and not any(codes):
            # Preserve the original final report, including the reused DiscoLoop.
            adapter.write_json(
                top, {"state": "waiting_for_reused_discoloop", "utc": adapter.utc_now()}
            )
            codes.append(
                subprocess.run(
                    [
                        sys.executable,
                        str(ctx.batch / "source/scripts/report_ttcd_architecture.py"),
                        "--persistent-root",
                        str(BASE),
                        "--wait-hours",
                        "48",
                    ],
                    check=False,
                ).returncode
            )
        adapter.write_json(
            top,
            {
                "state": "complete" if not any(codes) else "needs_attention",
                "codes": codes,
                "utc": adapter.utc_now(),
                "pending_stage_repair": str(directory),
            },
        )
        adapter.write_json(
            directory / "result.json",
            {"codes": codes, "utc": adapter.utc_now()},
            exclusive=True,
        )
        return int(any(codes))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, choices=("parent", "architecture"))
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.capture and not args.execute:
        print(json.dumps({"executed": False, "model_restarted": False}))
        return 0
    adapter = module(
        SPEEDUP / "source/scripts/accelerate_ttcd_controllers.py", "pending_adapter"
    )
    transport = module(
        SPEEDUP
        / "container-copy-repair-20261015/source/scripts/ttcd_speedup_transport.py",
        "pending_transport",
    )
    ctx = context(adapter, args.target)
    directory = AMENDMENT / args.target
    directory.mkdir(parents=True, exist_ok=True)
    if args.capture:
        snapshot = capture(adapter, ctx)
        adapter.write_json(directory / "snapshot.json", snapshot, exclusive=True)
        print(
            json.dumps(
                {"captured": True, "target": args.target, "model_restarted": False}
            )
        )
        return 0
    frozen = adapter.read_json(AMENDMENT / "source-manifest.json")
    if not frozen.get("engineering_passed"):
        raise ValueError("Passing engineering verification required")
    for relative, expected in frozen["files"].items():
        if adapter.sha256(AMENDMENT / "source" / relative) != expected:
            raise ValueError("Reviewed host amendment source changed")
    return execute(
        adapter,
        transport,
        ctx,
        adapter.read_json(directory / "snapshot.json"),
        directory,
    )


if __name__ == "__main__":
    raise SystemExit(main())
