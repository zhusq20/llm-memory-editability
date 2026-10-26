#!/usr/bin/env python3
"""Repair only the zero-update mount failure, preserving frozen science bytes."""

from __future__ import annotations

import fcntl
import importlib.util
import io
import json
import os
import shutil
import subprocess
import tarfile
import time
import traceback
from pathlib import Path

BATCH = "loop-query-control-development-v1"
BASE = Path("/ossfs/workspace/llm-memory-editability") / BATCH
DOCKER = ["docker", "--context", "lm-memory"]
TOKEN = Path.home() / ".cache/llm-memory-ttcd/wandb-token"


def original():
    path = BASE / "source/scripts/launch_loop_query_control.py"
    spec = importlib.util.spec_from_file_location("frozen_query_control_controller", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.verify(BASE)
    return module


def secret_archive():
    # Private input remains private on host; install just one file into the own container.
    if not TOKEN.is_file() or TOKEN.stat().st_mode & 0o077:
        raise ValueError("Original private tracking credential required")
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name in ("run", "run/secrets"):
            info = tarfile.TarInfo(name)
            info.type, info.mode = tarfile.DIRTYPE, 0o755
            archive.addfile(info)
        content = TOKEN.read_bytes()
        info = tarfile.TarInfo("run/secrets/wandb")
        info.size, info.mode, info.uid, info.gid = len(content), 0o400, 1015, 1016
        archive.addfile(info, io.BytesIO(content))
    return buffer.getvalue()


def create_command(module, config, stage, row, attempt="transport-v2"):
    name, args = module.command(config, BASE, stage, TOKEN, row)
    name += f"-{attempt}"
    result = [*DOCKER, "create"]
    index = 4
    while index < len(args):
        value = args[index]
        if value == "--detach":
            index += 1
        elif value == "--name":
            result += [value, name]
            index += 2
        elif value == "--mount" and "target=/run/secrets/wandb," in args[index + 1]:
            index += 2
        elif value == config["runtime"]["image"]:
            # Replace only credential delivery/entry, keeping scientific Python argv.
            result += [
                value,
                "python",
                "-c",
                "import os,sys;from pathlib import Path;"
                "os.environ['WANDB_API_KEY']=Path('/run/secrets/wandb').read_text().strip();"
                "os.execvp(sys.argv[1],sys.argv[1:])",
                *args[index + 5 :],
            ]
            break
        else:
            result.append(value)
            index += 1
    if "run" in result[:4] or not any("target=/output" in a for a in result):
        raise AssertionError("Wrong container transport transformation")
    return name, result


def execute(module, config, stage, row, attempt="transport-v2", builder=None):
    module.verify(BASE)
    name, args = (builder or create_command)(module, config, stage, row, attempt)
    folder = BASE / "execution" / attempt / stage
    folder.mkdir(parents=True, exist_ok=False)
    module.write(
        folder / "command.json",
        {
            "argv": args,
            "utc": module.utc(),
            "frozen_science_manifest_sha256": module.sha(BASE / "freeze.json"),
        },
    )
    created = subprocess.run(args, check=True, capture_output=True, text=True, timeout=60)
    subprocess.run(
        [*DOCKER, "cp", "--archive", "-", f"{name}:/"],
        input=secret_archive(),
        check=True,
        capture_output=True,
        timeout=60,
    )
    module.docker("start", name)
    module.write(
        folder / "launch.json",
        {"container": name, "id": created.stdout.strip(), "utc": module.utc()},
    )
    started = time.monotonic()
    with (folder / "stdout.log").open("x") as log:
        follower = subprocess.Popen(
            [*DOCKER, "logs", "--follow", name], stdout=log, stderr=subprocess.STDOUT
        )
        try:
            while True:
                state = json.loads(module.docker("inspect", name).stdout)[0]["State"]
                module.write(
                    folder / "heartbeat.json",
                    {
                        "container": name,
                        "state": state,
                        "elapsed_seconds": time.monotonic() - started,
                        "utc": module.utc(),
                    },
                )
                if not state["Running"]:
                    module.write(
                        folder / "exit.json", {"code": state["ExitCode"], "utc": module.utc()}
                    )
                    if state["ExitCode"]:
                        raise RuntimeError(
                            f"{stage} failed; retain original results; no auto retry"
                        )
                    break
                if time.monotonic() - started > config["runtime"]["timeout_hours"] * 3600:
                    module.write(
                        folder / "timeout-notice.json", {"own_container": name, "utc": module.utc()}
                    )
                    module.docker("stop", "--time", "30", name)
                    raise TimeoutError("Own container hard timeout")
                time.sleep(15)
        finally:
            try:
                follower.wait(timeout=45)
            except subprocess.TimeoutExpired:
                follower.terminate()
                follower.wait(timeout=10)


def main():
    module = original()
    repair = BASE / "runtime-repair"
    repair.mkdir(exist_ok=True)
    with (BASE / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        old, state = (
            module.read(BASE / "controller-process.json"),
            module.read(BASE / "controller-state.json"),
        )
        try:
            os.kill(old["pid"], 0)
        except ProcessLookupError:
            pass
        else:
            raise RuntimeError("Original controller is still running")
        if state["state"] != "needs_attention" or state["stage"] != "contracts":
            raise ValueError("Repair is restricted to the known pre-container failure")
        if any((BASE / "run").iterdir()):
            raise ValueError("Scientific artifacts already exist; cannot retry training")
        marker = BASE / "controller-process-resume-v2.json"
        if marker.exists():
            raise FileExistsError("Transport repair already claimed")
        shutil.copy2(BASE / "controller-state.json", repair / "failed-controller-state.json")
        shutil.copy2(BASE / "controller-process.json", repair / "failed-controller-process.json")
        snapshot = repair / Path(__file__).name
        shutil.copy2(__file__, snapshot)
        amendment = {
            "old_pid": old["pid"],
            "science_updates_before_repair": 0,
            "reason": "New batch directories inherited 0700; LM daemon cannot traverse source bind",
            "changes": "Only new traversal/output directories now0755; private credential copied "
            "through Docker API as own user0400 instead of inaccessible host bind; "
            "scientific argv/source/input hashes unchanged",
            "manifest_sha256": module.sha(BASE / "freeze.json"),
            "repair_source_sha256": module.sha(snapshot),
            "utc": module.utc(),
            "automatic_scientific_retry": False,
        }
        module.write(repair / "amendment.json", amendment)
        module.write(marker, {"pid": os.getpid(), "utc": module.utc(), **amendment})
        config = module.read(BASE / f"source/configs/{BATCH}.json")
        state = {"state": "running", "transport": "v2", "utc": module.utc()}
        try:
            for stage in ("contracts", "preflight", "train", "audit", "cloud-audit"):
                deadline = time.monotonic() + config["runtime"]["wait_timeout_hours"] * 3600
                row = None
                while row is None:
                    row = module.resources(config)
                    if row is None:
                        state.update(
                            state="waiting_for_idle_resources", stage=stage, utc=module.utc()
                        )
                        module.write(BASE / "controller-state.json", state)
                        if time.monotonic() > deadline:
                            raise TimeoutError("Resource waiting timeout")
                        time.sleep(30)
                module.write(repair / f"resource-admission-{stage}.json", row)
                state.update(state="running", stage=stage, utc=module.utc())
                module.write(BASE / "controller-state.json", state)
                execute(
                    module, config, stage, None if stage in ("contracts", "cloud-audit") else row
                )
            module.compare(config, BASE)
            state.update(
                state="complete",
                new_scientific_updates=300000,
                independent_audit_passed=True,
                cloud_verified=True,
                utc=module.utc(),
            )
            module.write(BASE / "completion-manifest.json", state)
            module.write(BASE / "controller-state.json", state)
        except BaseException as error:
            state.update(
                state="needs_attention",
                error=repr(error),
                traceback=traceback.format_exc(),
                utc=module.utc(),
            )
            module.write(BASE / "controller-state.json", state)
            raise


if __name__ == "__main__":
    main()
