"""Explicit recovery of zero-update launch failures; unchanged TTCD stage plan."""

from __future__ import annotations

import argparse
import concurrent.futures
import importlib.util
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

ROOT = Path("/mnt/disk2_from_server2/siqizhu4/llm-memory-editability")
BASE = Path("/ossfs/workspace/llm-memory-editability")
SPEEDUP = BASE / "ttcd-speedup-v1"
REPAIR = SPEEDUP / "container-copy-repair-20261015"


def module(path, name):
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


adapter = module(SPEEDUP / "source/scripts/accelerate_ttcd_controllers.py", "original_speedup")
transport = module(Path(__file__).parent / "ttcd_speedup_transport.py", "runtime_transport")


def context(target):
    return adapter.load_context(target, ROOT, BASE)


def capture(ctx):
    record = adapter.read_json(ctx.batch / "controller-process-speedup.json")
    old = adapter.process_identity(record["pid"])
    if old is None or old["state"] not in ("T", "t"):
        raise ValueError("Capture requires a stopped current host controller")
    original = adapter.read_json(SPEEDUP / "handoffs" / ctx.target / "snapshot.json")
    active = []
    if ctx.target == "parent":
        active = [adapter.active_job(ctx, arm) for arm in ("single", "stream")]
        if any(
            (a["container_id"], a["host_model_pid"], a["deadline_utc"])
            != (b["container_id"], b["host_model_pid"], b["deadline_utc"])
            for a, b in zip(active, original["active"], strict=True)
        ):
            raise ValueError("Original models or deadlines changed")
    elif ctx.target == "supplement":
        ctx.sharing.require_unclaimed(ctx.batch, ctx.controller.ARMS, ctx.controller.STAGES)
        adapter.deadline_from_utc(original["wait_deadline_utc"])
    else:
        raise ValueError("Architecture recovery uses completed-foundation validation")
    return {
        "schema": "ttcd-container-copy-handoff-v1",
        "target": ctx.target,
        "captured_utc": adapter.utc_now(),
        "old_controller": old,
        "active": active,
        "wait_deadline_utc": original["wait_deadline_utc"],
        "protected_files": {
            **original["protected_files"],
            **{str(p): adapter.sha256(p) for p in ctx.protected},
        },
        "model_restart": False,
    }


def validate(snapshot, ctx):
    if snapshot["schema"] != "ttcd-container-copy-handoff-v1" or snapshot["target"] != ctx.target:
        raise ValueError("Wrong handoff snapshot")
    if snapshot["old_controller"]["state"] not in ("T", "t"):
        raise ValueError("Host-only stopped snapshot required")
    old = adapter.process_identity(snapshot["old_controller"]["pid"])
    if old is not None and old["state"] != "Z":
        raise ValueError("Current old host controller must exit")
    for path, expected in snapshot["protected_files"].items():
        if adapter.sha256(path) != expected:
            raise ValueError(f"Protected original artifact changed: {path}")
    if ctx.target == "parent":
        observed = [adapter.active_job(ctx, arm) for arm in ("single", "stream")]
        for before, now in zip(snapshot["active"], observed, strict=True):
            for key in ("arm", "container", "container_id", "host_model_pid", "deadline_utc"):
                if before[key] != now[key]:
                    raise ValueError("An original live model changed")
    else:
        ctx.sharing.require_unclaimed(ctx.batch, ctx.controller.ARMS, ctx.controller.STAGES)
        original = adapter.read_json(SPEEDUP / "handoffs/supplement/snapshot.json")
        if snapshot["wait_deadline_utc"] != original["wait_deadline_utc"]:
            raise ValueError("The original supplementary deadline cannot be extended")


def validate_foundations(ctx):
    prior = adapter.read_json(SPEEDUP / "handoffs/architecture/result.json")
    if prior["state"] != "needs_attention" or adapter.process_identity(2133638) is not None:
        raise ValueError("Preserve the failed previous architecture controller")
    ctx.controller.verify_snapshot(ctx.batch)
    for arm in ("transformer", "loop"):
        directory = ctx.batch / arm
        if (directory / "common/training/progress.json").exists():
            raise ValueError("Never restart an existing adaptation train")
        completed = ["foundation-preflight", "foundation-train", "foundation-audit"]
        if arm == "transformer":
            completed.append("foundation-cloud-audit")
        for label in completed:
            stage = directory / "stages" / label
            receipt = adapter.read_json(stage / "result.json")
            if adapter.read_json(stage / "exit.json")["code"] or not receipt["passed"]:
                raise ValueError("Every completed foundation stage must pass")
            if adapter.sha256(receipt["path"]) != receipt["sha256"]:
                raise ValueError("Completed foundation receipt changed")
        identity = adapter.read_json(directory / "foundation/identity.json")
        config = adapter.read_json(ctx.batch / f"source/configs/architecture-{arm}.json")
        checkpoint = directory / f"foundation/{arm}/checkpoint-0300000.pt"
        if (
            not identity["passed"]
            or identity["step"] != 300000
            or adapter.sha256(checkpoint) != identity["checkpoint_sha256"]
        ):
            raise ValueError("Wrong native300k endpoint")
        if config["foundation"]["step"] != 300000:
            raise ValueError("Wrong adaptation starting step")


def archive_launch_failures(ctx):
    # These known failures performed zero adaptation updates. Preserve everything.
    for arm, label in [("loop", "foundation-cloud-audit"), ("transformer", "adaptation-preflight")]:
        old = ctx.batch / arm / "stages" / label
        saved = REPAIR / "failed-attempts" / f"{arm}-{label}"
        saved.parent.mkdir(parents=True, exist_ok=True)
        if saved.exists() or not old.exists():
            raise FileExistsError("This diagnosed repair may only execute once")
        old.rename(saved)
    failure = ctx.batch / "loop/foundation/failure-cloud-audit.json"
    failure.rename(REPAIR / "failed-attempts/loop-failure-cloud-audit.json")


def execute_architecture(ctx, token, source, manifest):
    validate_foundations(ctx)
    original_build = ctx.controller.build_command
    original_monitor = ctx.helpers.monitor_container

    def monitor(name, log, deadline, update=None):
        if name.endswith("-repair-v1"):
            arm = "loop" if "-loop-" in name else "transformer"
            if arm == "loop":
                anchor = adapter.read_json(
                    REPAIR / "failed-attempts/loop-foundation-cloud-audit/container.json"
                )["utc"]
            else:
                anchor = adapter.read_json(
                    REPAIR
                    / "failed-attempts/transformer-adaptation-preflight"
                    / "claim-adaptation-preflight.json"
                )["utc"]
            deadline = min(
                deadline,
                adapter.deadline_from_utc(
                    adapter.expiration(anchor, ctx.config["runtime"]["timeout_hours"])
                ),
            )
        return original_monitor(name, log, deadline, update)

    ctx.helpers.monitor_container = monitor

    def build(plan, arm, spec, batch, token, row):
        name, command = original_build(plan, arm, spec, batch, token, row)
        if (arm, spec["label"]) in {
            ("loop", "foundation-cloud-audit"),
            ("transformer", "adaptation-preflight"),
        }:
            name += "-repair-v1"
            command[command.index("--name") + 1] = name
        return name, adapter.wrap_command(
            command,
            plan["runtime"]["image"],
            source,
            manifest,
            f"architecture:{arm}:{spec['label']}",
        )

    ctx.controller.build_command = build

    def run_arm(arm):
        state = adapter.read_json(ctx.batch / arm / "controller-state.json")
        state["completed_stages"] = list(state["completed_stages"])

        def update(value):
            state.update(value)
            state["utc"] = adapter.utc_now()
            state["container_copy_repair"] = True
            ctx.controller.write_json(ctx.batch / arm / "controller-state.json", state)

        lock = (
            Path("/tmp")
            / f"lm-memory-gpu-locks-{os.getuid()}"
            / f"gpu-{ctx.config['runtime']['gpus'][arm]}.lock"
        )
        try:
            with adapter.locks([lock]):
                start = "foundation-cloud-audit" if arm == "loop" else "adaptation-preflight"
                specs = ctx.controller.stage_plan(arm)
                index = next(i for i, value in enumerate(specs) if value["label"] == start)
                for spec in specs[index:]:
                    ctx.controller.run_stage(ctx.config, arm, spec, ctx.batch, token, update)
                    state["completed_stages"].append(spec["label"])
                update({"state": "complete", "new_scientific_updates": 320000})
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

    with adapter.locks([ctx.batch / "controller.lock"]):
        archive_launch_failures(ctx)
        ctx.controller.write_json(
            ctx.batch / "controller-state.json",
            {
                "state": "running",
                "pid": os.getpid(),
                "utc": adapter.utc_now(),
                "container_copy_repair": True,
            },
        )
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            codes = list(pool.map(run_arm, ("transformer", "loop")))
        if any(codes):
            return 1
        return subprocess.run(
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, choices=("parent", "supplement", "architecture"))
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--snapshot", type=Path)
    args = parser.parse_args()
    if not args.capture and not args.execute:
        print(json.dumps({"executed": False, "host_permissions_changed": False}))
        return 0
    own_manifest = adapter.read_json(REPAIR / "source-manifest.json")
    for relative, expected in own_manifest["files"].items():
        if adapter.sha256(REPAIR / "source" / relative) != expected:
            raise ValueError("The reviewed container-copy recovery source changed")
    ctx = context(args.target)
    if args.capture:
        print(json.dumps(capture(ctx), indent=2))
        return 0
    source = SPEEDUP / "source"
    manifest = adapter.verify_speedup(source)
    directory = REPAIR / "handoffs" / args.target
    directory.mkdir(parents=True, exist_ok=True)
    if args.target != "architecture":
        if args.snapshot is None:
            raise ValueError("Live host replacement requires its stopped-host snapshot")
        snapshot = adapter.read_json(args.snapshot)
        validate(snapshot, ctx)
    else:
        snapshot = None
        validate_foundations(ctx)
    transport.install(ctx.helpers, source, manifest, REPAIR / "transport")
    adapter.write_json(
        directory / "claim.json",
        {
            "pid": os.getpid(),
            "utc": adapter.utc_now(),
            "snapshot": str(args.snapshot),
            "runtime_manifest_sha256": adapter.sha256(manifest),
            "model_restart": False,
            "host_permissions_changed": False,
        },
        exclusive=True,
    )
    token = Path.home() / ".cache/llm-memory-ttcd/wandb-token"
    if not token.is_file() or token.stat().st_mode & 0o077:
        raise ValueError("The original credential must remain private")
    try:
        if args.target == "architecture":
            code = execute_architecture(ctx, token, source, manifest)
        else:
            execute = (
                adapter.execute_parent if args.target == "parent" else adapter.execute_supplement
            )
            code = execute(ctx, snapshot, token, source, manifest, directory)
    except BaseException:
        adapter.write_json(
            directory / "failure.json",
            {"traceback": traceback.format_exc(), "utc": adapter.utc_now()},
        )
        raise
    adapter.write_json(
        directory / "result.json",
        {"code": code, "utc": adapter.utc_now(), "automatic_retry": False},
        exclusive=True,
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
