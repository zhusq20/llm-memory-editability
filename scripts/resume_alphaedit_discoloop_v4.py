"""Private-volume transport using the verified LM host-network configuration.

The execution user cannot change root-owned source0444/directories0555. No host
permissions, scientific configuration, frozen model or prior task is changed.
"""

from __future__ import annotations

import fcntl
import importlib.util
import json
import os
import subprocess
import time

from resume_alphaedit_discoloop import BATCH, VOLUME, persist_stage


def run_stage(launcher, plan, source, manifest, stage, gpu):
    name = f"alphaedit-discoloop-novel-v1-{stage}-readout-v4"
    args = [
        "create",
        "--name",
        name,
        "--user",
        "1015:1016",
        "--network",
        "host",
        "--cpuset-cpus",
        "84-87",
        "--cpus",
        "4",
        "--memory",
        "24g",
        "--tmpfs",
        "/tmp:rw,size=2g",
        "--env",
        "PYTHONDONTWRITEBYTECODE=1",
        "--env",
        "PYTHONPATH=/frozen/src:/frozen/scripts",
        "--env",
        f"LM_SOURCE_MANIFEST_SHA256={manifest}",
        "--env",
        "XDG_CACHE_HOME=/tmp/cache",
        "--env",
        "WANDB_CACHE_DIR=/tmp/wandb-cache",
        "--env",
        "WANDB_CONFIG_DIR=/tmp/wandb-config",
        "--env",
        "WANDB_DATA_DIR=/tmp/wandb-data",
        "--mount",
        "type=bind,source=/ossfs/workspace/llm-memory-editability,target=/ossfs/workspace/llm-memory-editability,readonly",
        "--mount",
        f"type=volume,source={VOLUME},target=/out",
        "--mount",
        "type=bind,source=/home/siqizhu4/.cache/llm-memory-ttcd/wandb-token,target=/run/secrets/wandb,readonly",
        "--workdir",
        "/frozen",
    ]
    if stage != "cloud":
        args += ["--gpus", f"device={gpu['uuid']}"]
    args += [
        plan["image"],
        "python",
        "scripts/run_alphaedit_discoloop.py",
        stage,
        "--plan",
        "configs/alphaedit-discoloop-novel-v1.json",
        "--root",
        "/out/attempt-4",
        "--device",
        "cuda:0",
    ]
    created = launcher.docker(*args).stdout.decode().strip()
    copied = subprocess.run(
        ["docker", "--context", "lm-memory", "cp", "-", f"{name}:/"],
        input=launcher.archive(source),
        capture_output=True,
        timeout=60,
    )
    launcher.write(
        BATCH / f"copy-{stage}-readout-v4.json",
        {
            "exit_code": copied.returncode,
            "stderr": copied.stderr.decode(),
            "frozen_source_sha256": manifest,
        },
    )
    copied.check_returncode()
    launcher.docker("start", name)
    launcher.write(
        BATCH / f"launch-{stage}-readout-v4.json",
        {"name": name, "id": created, "gpu": gpu, "argv": args},
    )
    started = time.monotonic()
    while True:
        state = json.loads(launcher.docker("inspect", "--format", "{{json .State}}", name).stdout)
        launcher.write(
            BATCH / "controller-state.json",
            {
                "stage": stage,
                "container": name,
                "container_state": state,
                "elapsed_seconds": time.monotonic() - started,
                "gpu": gpu,
                "transport": "private output volume, full-boundary observer v4",
            },
        )
        if not state["Running"]:
            logs = launcher.docker("logs", name)
            (BATCH / f"{stage}-readout-v4.log").write_bytes(logs.stdout + logs.stderr)
            launcher.write(BATCH / f"exit-{stage}-readout-v4.json", state)
            persist_stage(
                stage,
                launcher,
                name,
                runtime_root="/out/attempt-4",
                host_root=BATCH / "output/attempt-4",
            )
            if state["ExitCode"]:
                raise RuntimeError(f"Stage failed: {stage}; failure preserved")
            return
        if time.monotonic() - started > plan["stage_timeout_seconds"]:
            launcher.docker("stop", "--time", "30", name)
            raise TimeoutError(f"Stage exceeded timeout: {stage}")
        time.sleep(15)


def main():
    source = BATCH / "source-v2"
    spec = importlib.util.spec_from_file_location(
        "frozen_alphaedit_launcher", source / "scripts/launch_alphaedit_discoloop.py"
    )
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    original = json.loads((BATCH / "controller-process.json").read_text())
    plan = json.loads((source / "configs/alphaedit-discoloop-novel-v1.json").read_text())
    manifest = launcher.sha(source / "source-manifest.json")
    if plan["recovery"]["parent_source_sha256"] != original["manifest_sha256"]:
        raise ValueError("Wrong parent scientific source")
    if launcher.sha(source / "source-manifest.json") != manifest:
        raise ValueError("Original frozen source changed")
    with (BATCH / "controller.lock").open("a") as controller_lock:
        fcntl.flock(controller_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (BATCH / "controller-process-resume-v4.json").exists():
            raise FileExistsError("This recovery was already submitted")
        if (BATCH / "output/attempt-4/evaluation").exists():
            raise ValueError("Only recover failures before model execution")
        launcher.docker(
            "run",
            "--name",
            "alphaedit-discoloop-novel-v1-readout-prepare-v2",
            "--user",
            "1015:1016",
            "--read-only",
            "--network",
            "none",
            "--cpuset-cpus",
            "84-87",
            "--cpus",
            "4",
            "--memory",
            "4g",
            "--mount",
            f"type=volume,source={VOLUME},target=/out",
            plan["image"],
            "python",
            "-c",
            (
                "import hashlib; from pathlib import Path; "
                "assert all(hashlib.sha256(Path('/out/attempt-3/development',n).read_bytes())"
                ".digest()"
                "==hashlib.sha256(Path('/out/attempt-4/development',n).read_bytes()).digest() "
                "for n in ('result.json','protection.pt','wandb.json'))"
            ),
        )
        persist_stage(
            "development",
            launcher,
            "alphaedit-discoloop-novel-v1-readout-prepare-v2",
            runtime_root="/out/attempt-4",
            host_root=BATCH / "output/attempt-4",
        )
        launcher.write(
            BATCH / "schedule-amendment-v4.json",
            {
                "reason": (
                    "CUDA slice RMS differed by9.54e-7; "
                    "native full-shape reconstruction is bitwise exact"
                ),
                "source_sha256": manifest,
                "previous_formal_saved_states": 2,
                "changed": (
                    "full-shape observer and resume saved2 states; "
                    "AlphaEdit writer and selectedL2 unchanged"
                ),
                "output": "existing private Docker volume0700 and API persistence",
                "host_permissions_unchanged": True,
                "writer_configuration_changed": False,
                "observation_shape_repaired": True,
                "diagnostic": "boundary-diagnostic.json",
                "parent_source_sha256": original["manifest_sha256"],
                "recovery_script_sha256": launcher.sha(__file__),
            },
        )
        launcher.write(
            BATCH / "controller-process-resume-v4.json",
            {"pid": os.getpid(), "original_pid": original["pid"], "manifest_sha256": manifest},
        )
        gpu, lock, waiting = None, None, time.monotonic()
        try:
            while gpu is None:
                gpu, lock = launcher.idle_gpu()
                if gpu is None:
                    launcher.write(
                        BATCH / "controller-state.json",
                        {"state": "waiting_for_exclusive_idle_gpu", "resumed": "v4"},
                    )
                    if time.monotonic() - waiting > plan["resource_wait_seconds"]:
                        raise TimeoutError("No exclusive idle GPU available")
                    time.sleep(30)
            for stage in ("evaluation", "audit", "cloud"):
                run_stage(launcher, plan, source, manifest, stage, gpu)
            launcher.write(
                BATCH / "completion-manifest.json",
                {
                    "passed": True,
                    "source_sha256": manifest,
                    "base_parameter_updates": 0,
                    "stages": ["development", "evaluation", "audit", "cloud"],
                    "output_root": "output/attempt-4",
                    "schedule_amendments": [
                        "schedule-amendment.json",
                        "schedule-amendment-v2.json",
                        "schedule-amendment-v4.json",
                    ],
                },
            )
            launcher.write(
                BATCH / "controller-state.json",
                {"state": "complete", "passed": True, "gpu": gpu, "resumed": "v4"},
            )
        except BaseException as error:
            launcher.write(BATCH / "controller-failure-resume-v4.json", {"error": repr(error)})
            raise
        finally:
            if lock is not None:
                lock.close()


if __name__ == "__main__":
    main()
