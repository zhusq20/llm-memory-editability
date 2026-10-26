"""Short verification-only sharing with the identified LM job on GPU4."""

from __future__ import annotations

import fcntl
import importlib.util
import json
import os
import signal
import subprocess
from pathlib import Path

from resume_alphaedit_discoloop import BATCH
from resume_alphaedit_discoloop_v5 import run_stage

PEER = "loop-query-control-development-v1-train-capture-v3"
GPU = {"index": 4, "uuid": "GPU-b4302c0e-77f9-f270-ae1b-b95aba3f3c7f"}


def admission(launcher):
    peers = launcher.docker("top", PEER, "-eo", "pid").stdout.decode().splitlines()[1:]
    allowed = {int(p.strip()) for p in peers}
    rows = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    actual = {int(line.split(",")[1]) for line in rows if GPU["uuid"] in line}
    if not actual <= allowed:
        raise RuntimeError("GPU4 contains a process outside the identified LM container")
    free = subprocess.run(
        [
            "nvidia-smi",
            "--id",
            GPU["uuid"],
            "--query-gpu=memory.free",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if int(free) < 16384:
        raise RuntimeError("GPU4 has insufficient free memory for bounded verification")
    return {
        "peer": PEER,
        "resident_pids": sorted(actual),
        "free_mib": int(free),
        "torch_allocator_fraction": 0.03,
        "no_training_or_new_target_fit": True,
    }


def main():
    source = BATCH / "source-v3"
    spec = importlib.util.spec_from_file_location(
        "frozen_alphaedit_launcher", source / "scripts/launch_alphaedit_discoloop.py"
    )
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    plan = json.loads((source / "configs/alphaedit-discoloop-novel-v1.json").read_text())
    pending = json.loads((BATCH / "controller-process-resume-v5.json").read_text())
    if (BATCH / "output/attempt-5/evaluation/result.json").exists():
        raise ValueError("This handoff is only for the waiting controller, not completed work")
    command = Path(f"/proc/{pending['pid']}/cmdline").read_bytes()
    if b"scripts/resume_alphaedit_discoloop_v5.py" not in command:
        raise ValueError("Pending controller process identity differs")
    state = json.loads((BATCH / "controller-state.json").read_text())
    if state.get("state") != "waiting_for_exclusive_idle_gpu":
        raise ValueError("Only hand off a waiting, nonexecuting controller")
    receipt = admission(launcher)
    os.kill(pending["pid"], signal.SIGTERM)
    with (BATCH / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (BATCH / "controller-process-resume-v6.json").exists():
            raise FileExistsError("This verification handoff already exists")
        manifest = launcher.sha(source / "source-manifest.json")
        original_docker = launcher.docker

        def bounded_docker(*args, **kwargs):
            args = list(args)
            if args[0] == "create" and "--gpus" in args:
                image = args.index(plan["image"])
                old_command = args[image + 1 :]
                assert old_command[0] == "python"
                wrapper = (
                    "import torch,sys,runpy; torch.cuda.set_per_process_memory_fraction(0.03); "
                    "sys.argv=sys.argv[1:]; runpy.run_path(sys.argv[0],run_name='__main__')"
                )
                args[image + 1 :] = ["python", "-c", wrapper, *old_command[1:]]
            return original_docker(*args, **kwargs)

        launcher.docker = bounded_docker
        launcher.write(
            BATCH / "schedule-amendment-v6.json",
            {
                "source_sha256": manifest,
                "reason": "Only verification remains; GPU4 has ample memory",
                "admission": receipt,
                "cpu": "84-87; peer80-83",
                "host_permissions_unchanged": True,
                "original_waiting_controller_stopped": pending["pid"],
                "scientific_configuration_changed": False,
                "shared_timing_not_a_benchmark": True,
                "recovery_script_sha256": launcher.sha(__file__),
            },
        )
        launcher.write(
            BATCH / "controller-process-resume-v6.json",
            {"pid": os.getpid(), "manifest_sha256": manifest, "gpu": GPU},
        )
        try:
            for stage in ("evaluation", "audit", "cloud"):
                if stage != "cloud":
                    admission(launcher)
                run_stage(launcher, plan, source, manifest, stage, GPU)
            launcher.write(
                BATCH / "completion-manifest.json",
                {
                    "passed": True,
                    "source_sha256": manifest,
                    "output_root": "output/attempt-5",
                    "stages": ["development", "evaluation", "audit", "cloud"],
                    "base_parameter_updates": 0,
                    "development_reused": True,
                    "all32_original_science_states_reused": True,
                    "schedule_amendment": "schedule-amendment-v6.json",
                },
            )
            launcher.write(
                BATCH / "controller-state.json",
                {"state": "complete", "passed": True, "resumed": "v6", "gpu": GPU},
            )
        except BaseException as error:
            launcher.write(BATCH / "controller-failure-resume-v6.json", {"error": repr(error)})
            raise


if __name__ == "__main__":
    main()
