#!/usr/bin/env python3
"""Repair the new NFS traversal directory before any v2 container was created."""

import fcntl
import importlib.util
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
ORIGINAL = Path("/ossfs/workspace/llm-memory-editability/dual-gate-multihop-v1")
BASE = ORIGINAL / "attempts/preflight-repair-v2"


def main():
    os.umask(0o022)
    path = BASE / "source/scripts/launch_dual_gate.py"
    spec = importlib.util.spec_from_file_location("frozen_dual_gate_v2", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = module.read(BASE / "source/configs/dual-gate-multihop-v1.json")
    module.verify(BASE)
    module.verify_baseline(config)
    with (
        (ORIGINAL / "controller.lock").open("a") as original_lock,
        (BASE / "controller.lock").open("a") as lock,
    ):
        for held in (original_lock, lock):
            fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if any(BASE.glob("execution/*/train")):
            raise ValueError("Cannot restart a claimed scientific stage")
        folder = BASE / "execution/all/contracts"
        archive = folder.with_name("contracts-mount-failure-1")
        if archive.exists() or (folder / "launch.json").exists():
            raise ValueError("Preserve existing execution; transport recovery is one-shot")
        if module.docker(
            "ps", "-aq", "--filter", "name=dual-gate-multihop-v1-preflight-repair-v2-"
        ).stdout.strip():
            raise ValueError("The failed attempt unexpectedly created a container")
        permission = (ORIGINAL / "attempts").stat()
        if permission.st_uid != os.getuid():
            raise ValueError("Only this user's new attempt directory may change")
        module.write(
            BASE / "transport-amendment.json",
            {
                "reason": "New attempts directory 0700 blocked NFS Docker traversal",
                "original_mode": oct(permission.st_mode & 0o777),
                "new_mode": "0o755",
                "scientific_updates": 0,
                "model_executions_in_failed_mount_attempt": 0,
                "source_changed": False,
                "freeze_sha256": module.sha(BASE / "freeze.json"),
                "utc": module.utc(),
            },
        )
        (ORIGINAL / "attempts").chmod(0o755)
        folder.rename(archive)
        shutil.copy2(BASE / "controller-state.json", BASE / "controller-state-mount-failure-1.json")
        module.write(
            BASE / "controller-process-transport.json",
            {
                "pid": os.getpid(),
                "utc": module.utc(),
                "argv": sys.argv,
                "freeze_sha256": module.sha(BASE / "freeze.json"),
            },
        )
        controller = module.Controller(config, BASE)
        try:
            controller.run()
        except BaseException as error:
            controller.update(state="needs_attention", error=repr(error))
            raise


if __name__ == "__main__":
    main()
