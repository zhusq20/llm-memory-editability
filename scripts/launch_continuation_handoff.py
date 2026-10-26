#!/usr/bin/env python3
"""Manually launch only the two handed-off pending continuations on a new host."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
from pathlib import Path

ARMS = ("paper", "fixed-update")


def validate_receipt(receipt, freeze_sha, config_sha):
    if (
        receipt.get("ready") is not True
        or receipt.get("old_scheduler_stopped") is not True
        or tuple(receipt.get("arms", [])) != ARMS
        or receipt.get("freeze_sha256") != freeze_sha
        or receipt.get("config_sha256") != config_sha
    ):
        raise ValueError("No matching completed handoff from the original queue")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base", type=Path, required=True, help="Extracted frozen source/data package"
    )
    parser.add_argument(
        "--handoff-root",
        type=Path,
        required=True,
        help="Shared original execution/scheduling-handoff-v1; global single-use claim lives here",
    )
    parser.add_argument(
        "--mirror",
        type=Path,
        required=True,
        help="Shared original execution/remote-handoff-v1",
    )
    parser.add_argument("--gpus", type=int, nargs=2, required=True)
    parser.add_argument("--cpus", default="0,1,2,3")
    parser.add_argument("--wandb-token-file", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    base, mirror, handoff = (
        args.base.resolve(),
        args.mirror.resolve(),
        args.handoff_root.resolve(),
    )
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(base / "source/scripts"), str(base / "source/src")]
    import launch_paper_update_continuation as frozen

    common = frozen.original
    relative = f"configs/{frozen.BATCH}.json"
    config = common.read(base / "source" / relative)
    frozen.validate_config(config)
    validate_receipt(
        common.read(handoff / "handoff-ready.json"),
        common.sha(base / "freeze.json"),
        common.sha(base / "source" / relative),
    )
    if mirror != handoff.parent / "remote-handoff-v1":
        raise ValueError(
            "Use the registered shared remote-handoff-v1 mirror destination"
        )
    cpus = [int(c) for c in args.cpus.split(",")]
    if (
        len(cpus) != 4
        or len(set(cpus)) != 4
        or not set(cpus) <= os.sched_getaffinity(0)
    ):
        raise ValueError("Four distinct available CPU slots are required")
    if len(set(args.gpus)) != 2:
        raise ValueError("Two distinct GPUs required")
    plan = {
        "batch": frozen.BATCH,
        "arms": list(ARMS),
        "gpus": args.gpus,
        "cpus": cpus,
        "start_step": 300000,
        "target_step": 1500000,
        "transferred_updates": 2400000,
        "additional_budget": 0,
        "stages": ["train", "grid"],
        "mirror": str(mirror),
    }
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return
    common.verify_frozen(base, config)
    common.credential_archive(args.wandb_token_file)
    original_snapshot = common.snapshot

    def snapshot():
        rows, blocked = original_snapshot()
        by_uuid = {row["uuid"]: gpu for gpu, row in rows.items()}
        for context in ("default", "lm-memory"):
            ids = common.docker(
                "ps",
                "-aq",
                "--filter",
                "status=created",
                "--filter",
                "status=paused",
                "--filter",
                "status=restarting",
                context=context,
            ).stdout.split()
            if ids:
                for item in json.loads(
                    common.docker("inspect", *ids, context=context).stdout
                ):
                    for request in item["HostConfig"].get("DeviceRequests") or []:
                        for device in request.get("DeviceIDs") or []:
                            gpu = (
                                int(device) if device.isdigit() else by_uuid.get(device)
                            )
                            if gpu in rows:
                                blocked.setdefault(gpu, []).append(item["Name"])
        return rows, blocked

    common.snapshot = snapshot
    rows, blocked = snapshot()
    gpus = common.admit(args.gpus, rows, blocked)
    # Runtime scheduling amendment only; mounted frozen source/config stay byte-identical.
    frozen.LANES = (("paper",), ("fixed-update",))
    output = base / "output"
    output.mkdir(exist_ok=True)
    if any(output.iterdir()) or (mirror.exists() and any(mirror.iterdir())):
        raise FileExistsError(
            "Preserve existing receiver outputs; no automatic restart"
        )
    with (handoff / "remote-launch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # A permanent shared claim prevents another host from submitting these arms.
        with (handoff / "remote-launch-claim.json").open("x") as claim:
            json.dump(
                {
                    **plan,
                    "host": os.uname().nodename,
                    "pid": os.getpid(),
                    "utc": common.utc(),
                },
                claim,
                indent=2,
            )
            claim.write("\n")
        mirror.mkdir(parents=True, exist_ok=True)
        controller = frozen.Controller(
            config, base, mirror, relative, gpus, args.wandb_token_file
        )
        controller.cpus = cpus
        for arm in ("token-update", "token-update-half-lr"):
            controller.state["arms"].pop(arm)
        controller.state.update(
            new_scientific_updates=2400000,
            parent_batch_total_updates=4800000,
            scheduling_only_transfer=True,
        )
        common.write(
            output / "controller-process.json",
            {
                "pid": os.getpid(),
                "utc": common.utc(),
                "argv": sys.argv,
                "host": os.uname().nodename,
                "plan": plan,
                "freeze_sha256": common.sha(base / "freeze.json"),
                "config_sha256": common.sha(base / "source" / relative),
                "script_sha256": common.sha(Path(__file__)),
            },
        )
        common.write(output / "science-schedule.json", plan)
        try:
            passed = controller.run()
        except BaseException as error:
            controller.update(state="needs_attention", error=repr(error))
            common.write(
                output / "controller-exit.json",
                {"code": 1, "utc": common.utc(), "error": repr(error)},
            )
            controller.mirror_output(force=True)
            raise
        common.write(
            output / "controller-exit.json",
            {"code": 0 if passed else 1, "utc": common.utc()},
        )
        controller.mirror_output(force=True)
        if not passed:
            raise SystemExit(1)


if __name__ == "__main__":
    os.umask(0o022)
    main()
