#!/usr/bin/env python3
"""Resume frozen evaluation stages only, preserving training and private directories."""

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

ROOT = Path("/ossfs/workspace/llm-memory-editability")
REPAIR = "evaluation-recovery-20261016"
ONE = "ttcd-one-step-second-hop-v1"
MERGE = "ttcd-merge-hops-v1"
ARMS = ("transformer", "loop", "discoloop")


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


def stage_complete(controller, batch, arm, stage):
    directory = stage_directory(batch, arm, stage)
    receipt = directory / "exit.json"
    if not receipt.exists() or controller.read_json(receipt).get("returncode") != 0:
        return False
    output = canonical_output(batch, arm, stage) / f"result-{stage}.json"
    recorded = controller.read_json(directory / "result.json")
    if (
        not output.is_file()
        or controller.read_json(output).get("passed") is False
        or recorded["sha256"] != controller.sha256(output)
    ):
        raise ValueError("A completed stage lost its original canonical evidence")
    return True


def stage_directory(batch, arm, stage):
    return batch / ("aggregate-stage" if stage == "aggregate" else f"{arm}/stages/{stage}")


def canonical_output(batch, arm, stage):
    return batch / ("aggregate" if stage == "aggregate" else f"{arm}/evaluation")


def public_directory(path, base):
    """Create separate output directories; never relax an existing directory's mode."""
    path, base = Path(path), Path(base)
    if not path.is_relative_to(base) or path.is_symlink():
        raise ValueError("Output transport escaped its dedicated root")
    missing, current = [], path
    while not current.exists():
        missing.append(current)
        current = current.parent
    for item in reversed(missing):
        item.mkdir(mode=0o755)
        item.chmod(0o755)
    return path


def link_evidence(source, destination):
    """Publish completed files without duplicating large fast states or replacing evidence."""
    destination.mkdir(parents=True, exist_ok=True)
    for path in source.rglob("*"):
        relative = path.relative_to(source)
        if relative.parts[0] == "wandb" or path.is_symlink() or not path.is_file():
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if not os.path.samefile(path, target) and path.read_bytes() != target.read_bytes():
                raise ValueError(f"Refusing to replace different evidence: {target}")
        else:
            os.link(path, target)


def redirect_mounts(command, replacements):
    result = list(command)
    found = set()
    for i, value in enumerate(result[:-1]):
        if value != "--mount":
            continue
        fields = dict(part.split("=", 1) for part in result[i + 1].split(",") if "=" in part)
        target = fields.get("target")
        if target in replacements:
            original = fields["source"]
            result[i + 1] = result[i + 1].replace(
                f"source={original},", f"source={replacements[target]},", 1
            )
            found.add(target)
    if found != set(replacements):
        raise ValueError("Every reviewed transport mount must be replaced exactly once")
    return result


def run_stage(controller, plan, batch, arm, stage, token, transport, update):
    if stage not in (
        "prepare",
        "preflight",
        "calibrate",
        "evaluate",
        "audit",
        "cloud-audit",
        "aggregate",
    ):
        raise ValueError("Recovery is strictly evaluation-only; training is prohibited")
    if stage_complete(controller, batch, arm, stage):
        update({"state": "reused", "stage": stage})
        return
    directory = stage_directory(batch, arm, stage)
    if directory.exists():
        # Executed failures require explicit diagnosis; only unstarted launches may be retried.
        if any(canonical_output(batch, arm, stage).glob(f"*{stage}*.json")):
            raise ValueError("An executed failed stage requires a separate diagnosed amendment")
        name = f"{batch.name}-{arm}-{stage}"
        container = controller.resources.inspect_container(name)
        if container is not None:
            raise ValueError("Do not restart or replace an existing model container")
        archived = batch / REPAIR / "failed-attempts" / arm / stage
        archived.parent.mkdir(parents=True, exist_ok=True)
        directory.rename(archived)
    directory.mkdir(parents=True)
    controller.write_json(
        directory / "claim.json",
        {"pid": os.getpid(), "utc": controller.resources.utc_now(), "repair": REPAIR},
    )
    canonical = canonical_output(batch, arm, stage)
    canonical.mkdir(parents=True, exist_ok=True)
    output = public_directory(
        transport / batch.name / ("aggregate" if stage == "aggregate" else f"{arm}/evaluation"),
        transport,
    )
    link_evidence(canonical, output)
    archive = controller.runtime_archive(batch, arm)
    held, row = controller.acquire_resources(plan, arm, stage, update)
    with held:
        name, command = controller.build_command(plan, arm, stage, batch, token, row)
        replacements = {"/aggregate" if stage == "aggregate" else "/output": output}
        if stage == "aggregate":
            replacements["/comparison"] = transport / batch.name
        if batch.name == MERGE:
            replacements["/parent-evaluation"] = transport / ONE / arm / "evaluation"
        command = redirect_mounts(command, replacements)
        if controller.resources.inspect_container(name) is not None:
            raise ValueError("A stage container already exists; no duplicate submission")
        controller.write_json(directory / "admission.json", row)
        controller.write_json(directory / "command.json", {"argv": command})
        update({"state": "launching", "stage": stage, "container": name, "resource": row})
        try:
            created = controller.launch_copied_runtime(command, archive, directory)
            update({"state": "running", "container_id": created.stdout.strip()})
            with (directory / "container.log").open("x") as log:
                code = controller.resources.monitor_container(
                    name, log, time.monotonic() + plan["runtime"]["timeout_hours"] * 3600, update
                )
            controller.write_json(
                directory / "container-state.json", controller.resources.inspect_container(name)
            )
            controller.write_json(
                directory / "exit.json", {"returncode": code, "utc": controller.resources.utc_now()}
            )
            link_evidence(output, canonical)
            if code:
                raise RuntimeError(f"{batch.name}/{arm}/{stage} exited {code}")
            result = canonical / f"result-{stage}.json"
            if controller.read_json(result).get("passed") is False:
                raise ValueError("Mandatory evaluation validation failed")
            controller.write_json(
                directory / "result.json",
                {"path": str(result), "sha256": controller.sha256(result), "passed": True},
            )
            update({"state": "stage_complete", "stage": stage})
        except BaseException as error:
            controller.write_json(
                directory / "recovery-failure.json",
                {"traceback": traceback.format_exc(), "stderr": getattr(error, "stderr", None)},
            )
            raise


def run_arm(controllers, plans, arm, token, transport):
    outcomes = {}
    for name in (ONE, MERGE):
        controller, plan, batch = controllers[name], plans[name], ROOT / name
        state = {"architecture": arm, "repair": REPAIR, "completed_stages": []}

        def update(values, state=state, controller=controller, batch=batch):
            state.update(values)
            state["utc"] = controller.resources.utc_now()
            controller.write_json(batch / arm / "controller-state.json", state)

        try:
            controller.verify_snapshot(batch, arm)
            if name == ONE:
                receipt = controller.endpoint_receipt(batch, plan, arm)
                if receipt is None:
                    raise ValueError("Existing stream10000 training and audit are not complete")
                controller.write_json(batch / arm / "parent-endpoint.json", receipt)
                stages = controller.STAGES
            else:
                receipt = controller.parent_receipt(batch, plan, arm)
                if receipt is None:
                    raise ValueError("Parent one-step local/cloud audits must pass first")
                controller.write_json(batch / arm / "parent-evaluation.json", receipt)
                stages = ("prepare", *controller.STAGES)
            for stage in stages:
                run_stage(controller, plan, batch, arm, stage, token, transport, update)
                state["completed_stages"].append(stage)
            update({"state": "complete", "new_base_parameter_updates": 0})
            outcomes[name] = dict(state)
        except BaseException:
            update({"state": "needs_attention", "traceback": traceback.format_exc()})
            raise
    return outcomes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    controllers = {
        name: module(
            ROOT
            / name
            / "transformer/source/scripts"
            / ("launch_ttcd_one_step.py" if name == ONE else "launch_ttcd_merge_hops.py"),
            f"recovery_{name.replace('-', '_')}",
        )
        for name in (ONE, MERGE)
    }
    helpers = controllers[ONE].resources
    plans = {
        name: helpers.read_json(
            ROOT
            / name
            / "transformer/source/configs"
            / ("one-step.json" if name == ONE else "merge-hops.json")
        )
        for name in controllers
    }
    protected = {}
    for name, controller in controllers.items():
        controller.validate_plan(plans[name])
        for arm in ARMS:
            controller.verify_snapshot(ROOT / name, arm)
        protected[str(ROOT / name / "source-manifest.json")] = helpers.sha256(
            ROOT / name / "source-manifest.json"
        )
    for arm in ARMS:
        receipt = controllers[ONE].endpoint_receipt(ROOT / ONE, plans[ONE], arm)
        if receipt is None:
            raise ValueError("Never recover evaluation before its completed training audit")
        protected[receipt["checkpoint"]] = receipt["checkpoint_sha256"]
        protected[receipt["audit"]] = receipt["audit_sha256"]
    if not args.execute:
        print(json.dumps({"passed": True, "protected": protected, "new_training_updates": 0}))
        return 0
    token = Path.home() / ".cache/llm-memory-ttcd/wandb-token"
    if not token.is_file() or token.stat().st_mode & 0o077:
        raise ValueError("Use the existing private credential file")
    transport = ROOT / REPAIR
    public_directory(transport, transport)
    with (transport / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        batch_locks = []
        for name in controllers:
            batch_lock = (ROOT / name / "controller.lock").open("a")
            fcntl.flock(batch_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            batch_locks.append(batch_lock)
        for container in helpers.running_containers(["lm-memory"]):
            if any(container["Name"].lstrip("/").startswith(name + "-") for name in controllers):
                raise RuntimeError("An existing evaluation is active; never launch a duplicate")
        for name in controllers:
            batch = ROOT / name
            repair = batch / REPAIR
            repair.mkdir(exist_ok=False)
            shutil.copyfile(
                batch / "controller-state.json", repair / "previous-controller-state.json"
            )
            for arm in ARMS:
                shutil.copyfile(
                    batch / arm / "controller-state.json", repair / f"previous-{arm}-state.json"
                )
            helpers.write_json(
                batch / "controller-process-recovery.json",
                {
                    "pid": os.getpid(),
                    "utc": helpers.utc_now(),
                    "source": __file__,
                    "source_sha256": helpers.sha256(__file__),
                    "repair": REPAIR,
                    "new_training_updates": 0,
                },
            )
            helpers.write_json(
                batch / "controller-state.json",
                {
                    "state": "running",
                    "repair": REPAIR,
                    "pid": os.getpid(),
                    "new_base_parameter_updates": 0,
                },
            )
        state = {
            "state": "running",
            "pid": os.getpid(),
            "outcomes": {},
            "failures": {},
            "protected": protected,
            "new_training_updates": 0,
        }

        def update(values):
            state.update(values)
            state["utc"] = helpers.utc_now()
            helpers.write_json(transport / "controller-state.json", state)

        update({})
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            futures = {
                pool.submit(run_arm, controllers, plans, arm, token, transport): arm for arm in ARMS
            }
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
        for name, controller in controllers.items():
            run_stage(
                controller,
                plans[name],
                ROOT / name,
                "transformer",
                "aggregate",
                token,
                transport,
                update,
            )
            helpers.write_json(
                ROOT / name / "controller-state.json",
                {
                    "state": "complete",
                    "repair": REPAIR,
                    "utc": helpers.utc_now(),
                    "new_base_parameter_updates": 0,
                },
            )
        for path, expected in protected.items():
            if helpers.sha256(path) != expected:
                raise ValueError(f"Protected training/source evidence changed: {path}")
        update({"state": "complete", "protected_evidence_unchanged": True})
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
