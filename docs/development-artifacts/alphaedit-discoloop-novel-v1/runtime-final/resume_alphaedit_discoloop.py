"""Resume the frozen AlphaEdit batch after its pre-model bind-mount failure."""

from __future__ import annotations

import fcntl
import hashlib
import importlib.util
import json
import os
import subprocess
import tarfile
import time
from pathlib import Path

BATCH = Path("/ossfs/workspace/llm-memory-editability/alphaedit-discoloop-novel-v1")
VOLUME = "alphaedit-discoloop-novel-v1-private-output"


def persist_stage(stage, launcher, container_name=None, runtime_root="/out", host_root=None):
    """Copy only this stage's files through Docker API into private host storage."""
    destination = host_root or BATCH / "output"
    destination.mkdir(mode=0o700, parents=True, exist_ok=True)
    process = subprocess.Popen(
        [
            "docker",
            "--context",
            "lm-memory",
            "cp",
            f"{container_name or 'alphaedit-discoloop-novel-v1-' + stage}:{runtime_root}/{stage}",
            "-",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    files, skipped = {}, []
    with tarfile.open(fileobj=process.stdout, mode="r|") as archive:
        for member in archive:
            relative = Path(member.name)
            if relative.is_absolute() or ".." in relative.parts or relative.parts[0] != stage:
                raise ValueError("Stage archive escaped its output root")
            target = destination / relative
            if member.isdir():
                target.mkdir(mode=0o700, parents=True, exist_ok=True)
            elif member.isfile():
                target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                source = archive.extractfile(member)
                h = hashlib.sha256()
                if target.exists():
                    for block in iter(lambda source=source: source.read(8 * 1024 * 1024), b""):
                        h.update(block)
                    if launcher.sha(target) != h.hexdigest():
                        raise ValueError(f"Refusing to overwrite differing stage evidence: {target}")
                else:
                    with target.open("xb") as handle:
                        for block in iter(lambda source=source: source.read(8 * 1024 * 1024), b""):
                            handle.write(block)
                            h.update(block)
                files[str(relative)] = h.hexdigest()
            else:
                # W&B convenience symlinks are skipped; regular logs remain saved.
                skipped.append(member.name)
    error = process.stderr.read().decode()
    if process.wait() != 0:
        raise RuntimeError(f"Private stage copy failed: {error}")
    launcher.write(
        BATCH / f"private-copy-{stage}-{container_name or 'default'}.json",
        {
            "files": files,
            "skipped_nonregular": skipped,
            "output_volume": VOLUME,
            "host_permissions_unchanged": True,
        },
    )


def main():
    source = BATCH / "source"
    spec = importlib.util.spec_from_file_location(
        "frozen_alphaedit_launcher", source / "scripts/launch_alphaedit_discoloop.py"
    )
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    plan = json.loads((source / "configs/alphaedit-discoloop-novel-v1.json").read_text())
    original = json.loads((BATCH / "controller-process.json").read_text())
    if launcher.sha(source / "source-manifest.json") != original["manifest_sha256"]:
        raise ValueError("Original source manifest changed")
    with (BATCH / "controller.lock").open("a") as controller_lock:
        fcntl.flock(controller_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (BATCH / "controller-process-resume.json").exists():
            raise FileExistsError("Do not duplicate this resume attempt")
        if (BATCH / "output/development").exists():
            raise ValueError("This recovery is only for a failure before model execution")
        # Keep host modes private; use a Docker volume instead of bind-mounting /out.
        original_docker = launcher.docker
        original_docker("volume", "create", VOLUME)
        initialization = original_docker(
            "run",
            "--name",
            "alphaedit-discoloop-novel-v1-private-init",
            "--user",
            "0:0",
            "--read-only",
            "--cpus",
            "1",
            "--memory",
            "1g",
            "--mount",
            f"type=volume,source={VOLUME},target=/out",
            plan["image"],
            "python",
            "-c",
            "import os; os.chown('/out',1015,1016); os.chmod('/out',0o700)",
        )
        if initialization.returncode:
            raise RuntimeError("Could not initialize private output volume")

        def private_docker(*args, **kwargs):
            args = list(args)
            if args[0] == "create":
                old = f"type=bind,source={BATCH / 'output'},target=/out"
                if old not in args:
                    raise ValueError("Expected exactly the original output bind mount")
                args[args.index(old)] = f"type=volume,source={VOLUME},target=/out"
            return original_docker(*args, **kwargs)

        launcher.docker = private_docker
        launcher.write(
            BATCH / "schedule-amendment.json",
            {
                "reason": "Docker daemon could not traverse new batch/output mode0700",
                "diagnostic": "create-diagnostic.json",
                "previous_model_executions": 0,
                "changed": "private Docker output volume, stage archives copied back by Docker API",
                "output_volume": VOLUME,
                "host_permissions_unchanged": True,
                "permission_expansion_rejected_and_not_executed": True,
                "source_sha256": original["manifest_sha256"],
                "scientific_configuration_changed": False,
                "original_failure_preserved": "controller-failure.json",
            },
        )
        launcher.write(
            BATCH / "controller-process-resume.json",
            {
                "pid": os.getpid(),
                "original_pid": original["pid"],
                "manifest_sha256": original["manifest_sha256"],
            },
        )
        gpu, lock, waiting = None, None, time.monotonic()
        try:
            while gpu is None:
                gpu, lock = launcher.idle_gpu()
                if gpu is None:
                    launcher.write(
                        BATCH / "controller-state.json",
                        {"state": "waiting_for_exclusive_idle_gpu", "resumed": True},
                    )
                    if time.monotonic() - waiting > plan["resource_wait_seconds"]:
                        raise TimeoutError("No exclusive idle GPU became available")
                    time.sleep(30)
            for stage in ("development", "evaluation", "audit", "cloud"):
                launcher.run_stage(BATCH, plan, source, original["manifest_sha256"], stage, gpu)
                persist_stage(stage, launcher)
            launcher.write(
                BATCH / "completion-manifest.json",
                {
                    "passed": True,
                    "source_sha256": original["manifest_sha256"],
                    "base_parameter_updates": 0,
                    "stages": ["development", "evaluation", "audit", "cloud"],
                    "schedule_amendment": "schedule-amendment.json",
                },
            )
            launcher.write(
                BATCH / "controller-state.json",
                {"state": "complete", "passed": True, "gpu": gpu, "resumed": True},
            )
        except BaseException as error:
            launcher.write(BATCH / "controller-failure-resume.json", {"error": repr(error)})
            raise
        finally:
            if lock is not None:
                lock.close()


if __name__ == "__main__":
    main()
