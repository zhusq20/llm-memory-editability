#!/usr/bin/env python3
"""Adopt the active evaluation and share only this recovery's otherwise idle GPU7."""

from __future__ import annotations

import concurrent.futures
import contextlib
import fcntl
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

ROOT = Path("/ossfs/workspace/llm-memory-editability")
TRANSPORT = ROOT / "evaluation-recovery-20261016"
HANDOFF = "publication-cache-repair-20261016"


def monitor_log(directory):
    return Path(directory) / f"{HANDOFF}-monitor.log"


def completed_publication(source, destination):
    """A host receipt is committed only after the complete outbound publication."""
    for batch in ("ttcd-one-step-second-hop-v1", "ttcd-merge-hops-v1"):
        for arm in ("transformer", "loop", "discoloop"):
            canonical = ROOT / batch / arm / "evaluation"
            output = TRANSPORT / batch / arm / "evaluation"
            if {source, destination} != {canonical, output}:
                continue
            for stage in ("evaluate", "calibrate", "preflight"):
                receipt_path = canonical.parent / "stages" / stage / "result.json"
                if not receipt_path.is_file():
                    continue
                receipt = json.loads(receipt_path.read_text())
                left, right = canonical / f"result-{stage}.json", output / f"result-{stage}.json"
                if not left.is_file() or not right.is_file() or not os.path.samefile(left, right):
                    continue
                if (
                    receipt.get("passed") is not True
                    or hashlib.sha256(left.read_bytes()).hexdigest() != receipt["sha256"]
                ):
                    raise ValueError("Completed publication receipt/result changed")
                return source == canonical, stage == "evaluate"
    return False, False


def publish_evidence(source, destination):
    """Traverse each directory once; link unchanged evidence without repeated mkdir/stat."""
    source, destination = Path(source), Path(destination)
    inbound_complete, immutable_states_linked = completed_publication(source, destination)
    if inbound_complete:
        # The container is the sole producer; canonical files came from this same output.
        return
    try:
        destination.mkdir()
    except FileExistsError:
        if not destination.is_dir() or destination.is_symlink():
            raise ValueError("Evidence destination must be a real directory") from None
    with os.scandir(source) as entries:
        for entry in entries:
            if entry.name == "wandb" or entry.is_symlink():
                continue
            if entry.name == "states" and immutable_states_linked:
                # Frozen audits read/check every state but never replace its files.
                continue
            target = destination / entry.name
            if entry.is_dir(follow_symlinks=False):
                publish_evidence(Path(entry.path), target)
            elif entry.is_file(follow_symlinks=False):
                try:
                    os.link(entry.path, target)
                except FileExistsError:
                    original, existing = entry.stat(follow_symlinks=False), target.stat()
                    if (original.st_dev, original.st_ino) != (existing.st_dev, existing.st_ino):
                        if Path(entry.path).read_bytes() != target.read_bytes():
                            raise ValueError(
                                f"Refusing to replace different evidence: {target}"
                            ) from None


def load():
    path = TRANSPORT / "source/scripts/resume_ttcd_evaluations.py"
    manifest = json.loads((TRANSPORT / "source-manifest.json").read_text())
    import hashlib

    if (
        hashlib.sha256(path.read_bytes()).hexdigest()
        != manifest["files"]["scripts/resume_ttcd_evaluations.py"]
    ):
        raise ValueError("The original recovery wrapper changed")
    spec = importlib.util.spec_from_file_location("original_evaluation_recovery", path)
    module = importlib.util.module_from_spec(spec)
    sys.dont_write_bytecode = True
    spec.loader.exec_module(module)
    return module


def verify_cohort(helpers, controllers, plans):
    rows, processes = helpers.gpu_snapshot()
    row = rows[7]
    containers = helpers.running_containers(["lm-memory"])
    own, pids = [], set()
    for container in containers:
        if not helpers.container_uses_gpu(container, 7, row["uuid"]):
            continue
        name = container["Name"].lstrip("/")
        matched = None
        for batch in controllers:
            for arm in ("transformer", "loop", "discoloop"):
                for stage in ("preflight", "calibrate", "evaluate", "audit"):
                    if name == f"{batch}-{arm}-{stage}":
                        matched = (batch, arm, stage)
        if matched is None:
            raise helpers.ResourceUnavailable("GPU7 contains a task outside this evaluation")
        batch, arm, stage = matched
        controller = controllers[batch]
        expected = controller.read_json(ROOT / batch / arm / "stages" / stage / "command.json")[
            "argv"
        ]
        expected_env = {expected[i + 1] for i, v in enumerate(expected[:-1]) if v == "--env"}
        actual_env = set(container["Config"]["Env"])
        if (
            container["Config"]["User"] != "1015:1016"
            or container["Config"]["Image"] != plans[batch]["runtime"]["image"]
            or not expected_env <= actual_env
            or container["Config"]["Cmd"]
            != expected[expected.index(plans[batch]["runtime"]["image"]) + 1 :]
        ):
            raise ValueError("The registered evaluation container identity changed")
        output = helpers.docker("top", container["Id"], "-eo", "pid").stdout
        pids.update(int(v.strip()) for v in output.splitlines()[1:] if v.strip().isdigit())
        own.append({"name": name, "id": container["Id"], "pid": container["State"]["Pid"]})
    if any(p["uuid"] == row["uuid"] and p["pid"] not in pids for p in processes):
        raise helpers.ResourceUnavailable("GPU7 contains an outside CUDA process")
    if len(own) > 3 or row["free_mib"] < 12000:
        raise helpers.ResourceUnavailable("The bounded three-model GPU7 capacity is unavailable")
    return row, own


def admission(base, controllers, plans, plan, arm, stage, update, lock_directory=None):
    helpers = next(iter(controllers.values())).resources
    runtime = plan["runtime"]
    directory = lock_directory or Path("/tmp") / f"lm-memory-gpu-locks-{os.getuid()}"
    directory.mkdir(exist_ok=True)
    deadline = time.monotonic() + runtime["wait_timeout_hours"] * 3600
    while time.monotonic() < deadline:
        held = contextlib.ExitStack()
        try:
            for cpu in runtime["cpu_sets"][arm]:
                lock = held.enter_context((directory / f"cpu-{cpu}.lock").open("a"))
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as error:
                    raise helpers.ResourceUnavailable("Exclusive CPU lock held") from error
            helpers.admit_cpu(plan, arm)
            if stage in ("prepare", "cloud-audit", "aggregate"):
                return held, {"device": "cpu", "cpu_set": runtime["cpu_sets"][arm]}
            row, own = verify_cohort(helpers, controllers, plans)
            return held, {
                **row,
                "gpu": 7,
                "cpu_set": runtime["cpu_sets"][arm],
                "gpu_sharing": True,
                "cohort": own,
                "utc": helpers.utc_now(),
            }
        except helpers.ResourceUnavailable as error:
            held.close()
            update({"state": "waiting_for_resources", "stage": stage, "reason": str(error)})
            time.sleep(5)
        except BaseException:
            held.close()
            raise
    raise TimeoutError("No safe GPU7 evaluation resource became available")


def adopt_stage(base, original, controller, plan, batch, arm, stage, token, transport, update):
    directory = base.stage_directory(batch, arm, stage)
    if base.stage_complete(controller, batch, arm, stage) or not directory.exists():
        return original(controller, plan, batch, arm, stage, token, transport, update)
    name = f"{batch.name}-{arm}-{stage}"
    container = controller.resources.inspect_container(name)
    if container is None:
        # This is a claimed but unstarted wait, retained separately from the original failure.
        saved = batch / HANDOFF / "unstarted-claims" / arm / stage
        saved.parent.mkdir(parents=True, exist_ok=True)
        directory.rename(saved)
        return original(controller, plan, batch, arm, stage, token, transport, update)
    # Adopt this exact existing container, whether still running or just exited.
    archive = controller.runtime_archive(batch, arm)
    import hashlib

    recorded = controller.read_json(directory / "transport.json")
    if (
        recorded.get("container_id") != container["Id"]
        or recorded["archive_sha256"] != hashlib.sha256(archive).hexdigest()
    ):
        raise ValueError("Adopt only the exact already-launched scientific runtime")
    update(
        {
            "state": "running",
            "stage": stage,
            "container": name,
            "container_id": container["Id"],
            "model_restarted": False,
        }
    )
    with monitor_log(directory).open("x") as log:
        code = controller.resources.monitor_container(
            name, log, time.monotonic() + plan["runtime"]["timeout_hours"] * 3600, update
        )
    output = transport / batch.name / arm / "evaluation"
    canonical = base.canonical_output(batch, arm, stage)
    base.link_evidence(output, canonical)
    controller.write_json(
        directory / "container-state.json", controller.resources.inspect_container(name)
    )
    controller.write_json(
        directory / "exit.json",
        {"returncode": code, "utc": controller.resources.utc_now(), "adopted": True},
    )
    if code:
        raise RuntimeError("Adopted evaluation failed; no automatic retry")
    result = canonical / f"result-{stage}.json"
    if controller.read_json(result).get("passed") is False:
        raise ValueError("Adopted scientific validation failed")
    controller.write_json(
        directory / "result.json",
        {"path": str(result), "sha256": controller.sha256(result), "passed": True},
    )
    update({"state": "stage_complete", "stage": stage})


def main():
    base = load()
    base.link_evidence = publish_evidence
    controllers = {
        name: base.module(
            ROOT
            / name
            / "transformer/source/scripts"
            / ("launch_ttcd_one_step.py" if name == base.ONE else "launch_ttcd_merge_hops.py"),
            "shared_" + name.replace("-", "_"),
        )
        for name in (base.ONE, base.MERGE)
    }
    helpers = controllers[base.ONE].resources
    plans = {
        name: helpers.read_json(
            ROOT
            / name
            / "transformer/source/configs"
            / ("one-step.json" if name == base.ONE else "merge-hops.json")
        )
        for name in controllers
    }
    previous = helpers.read_json(TRANSPORT / "controller-state.json")
    diagnosed_log_collision = (
        set(previous["failures"]) == {"discoloop"}
        and "FileExistsError" in previous["failures"]["discoloop"]
        and "handoff-monitor.log" in previous["failures"]["discoloop"]
    )
    if previous["state"] != "running" or (previous["failures"] and not diagnosed_log_collision):
        raise ValueError("Only the healthy active evaluation may be adopted")
    for controller in controllers.values():
        controller.acquire_resources = (
            lambda plan, arm, stage, update, lock_directory=None: admission(
                base, controllers, plans, plan, arm, stage, update, lock_directory
            )
        )
    original = base.run_stage
    base.run_stage = lambda *args: adopt_stage(base, original, *args)
    base.REPAIR = HANDOFF
    with (TRANSPORT / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        batch_locks = []
        for name in controllers:
            batch_lock = (ROOT / name / "controller.lock").open("a")
            fcntl.flock(batch_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            batch_locks.append(batch_lock)
            directory = ROOT / name / HANDOFF
            directory.mkdir(exist_ok=False)
            shutil.copyfile(
                ROOT / name / "controller-state.json", directory / "previous-state.json"
            )
            helpers.write_json(
                ROOT / name / "controller-process-sharing.json",
                {
                    "pid": os.getpid(),
                    "source": __file__,
                    "source_sha256": helpers.sha256(__file__),
                    "new_training_updates": 0,
                    "model_restarted": False,
                },
            )
        verify_cohort(helpers, controllers, plans)
        state = {
            **previous,
            "pid": os.getpid(),
            "handoff": HANDOFF,
            "outcomes": {},
            "failures": {},
            "gpu": 7,
            "gpu_sharing": True,
        }

        def update(values):
            state.update(values)
            state["utc"] = helpers.utc_now()
            helpers.write_json(TRANSPORT / "controller-state.json", state)

        update({})
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            futures = {
                pool.submit(
                    base.run_arm,
                    controllers,
                    plans,
                    arm,
                    Path.home() / ".cache/llm-memory-ttcd/wandb-token",
                    TRANSPORT,
                ): arm
                for arm in base.ARMS
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
            base.run_stage(
                controller,
                plans[name],
                ROOT / name,
                "transformer",
                "aggregate",
                Path.home() / ".cache/llm-memory-ttcd/wandb-token",
                TRANSPORT,
                update,
            )
            helpers.write_json(
                ROOT / name / "controller-state.json",
                {
                    "state": "complete",
                    "handoff": HANDOFF,
                    "new_base_parameter_updates": 0,
                    "utc": helpers.utc_now(),
                },
            )
        for path, wanted in state["protected"].items():
            if helpers.sha256(path) != wanted:
                raise ValueError("Protected training/source evidence changed")
        update({"state": "complete", "protected_evidence_unchanged": True})
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
