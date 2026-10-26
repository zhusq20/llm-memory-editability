"""Finish persistence and exact audit after successful complete-record recovery."""

from __future__ import annotations

import fcntl
import importlib.util
import json
import os

from resume_alphaedit_discoloop import BATCH, persist_stage
from resume_alphaedit_discoloop_v5 import run_stage
from resume_alphaedit_discoloop_v6 import GPU, admission


def main():
    source = BATCH / "source-v3"
    spec = importlib.util.spec_from_file_location(
        "frozen_alphaedit_launcher", source / "scripts/launch_alphaedit_discoloop.py")
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    plan = json.loads((source / "configs/alphaedit-discoloop-novel-v1.json").read_text())
    with (BATCH / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (BATCH / "controller-process-resume-v7.json").exists():
            raise FileExistsError("This completion handoff already exists")
        state = json.loads((BATCH / "exit-evaluation-report-v5.json").read_text())
        if state["ExitCode"] != 0:
            raise ValueError("Evaluation recovery must already have succeeded")
        manifest = launcher.sha(source / "source-manifest.json")
        persist_stage("evaluation", launcher, "alphaedit-discoloop-novel-v1-evaluation-report-v5",
                      runtime_root="/out/attempt-5", host_root=BATCH / "output/attempt-5")
        launcher.write(BATCH / "schedule-amendment-v7.json", {
            "reason": "Merge new canonical result into existing identical saved records",
            "source_sha256": manifest, "existing_files_hash_checked_not_overwritten": True,
            "scientific_configuration_changed": False, "host_permissions_unchanged": True,
            "gpu_admission": admission(launcher), "recovery_script_sha256": launcher.sha(__file__),
        })
        launcher.write(BATCH / "controller-process-resume-v7.json", {
            "pid": os.getpid(), "manifest_sha256": manifest, "gpu": GPU})
        original_docker = launcher.docker

        def bounded_docker(*args, **kwargs):
            args = list(args)
            if args[0] == "create" and "--gpus" in args:
                image = args.index(plan["image"])
                old = args[image + 1:]
                wrapper = (
                    "import torch,sys,runpy; torch.cuda.set_per_process_memory_fraction(0.03); "
                    "sys.argv=sys.argv[1:]; runpy.run_path(sys.argv[0],run_name='__main__')"
                )
                args[image + 1:] = ["python", "-c", wrapper, *old[1:]]
            return original_docker(*args, **kwargs)

        launcher.docker = bounded_docker
        try:
            for stage in ("audit", "cloud"):
                if stage == "audit":
                    admission(launcher)
                run_stage(launcher, plan, source, manifest, stage, GPU)
            launcher.write(BATCH / "completion-manifest.json", {
                "passed": True, "source_sha256": manifest, "output_root": "output/attempt-5",
                "stages": ["development", "evaluation", "audit", "cloud"],
                "base_parameter_updates": 0, "development_reused": True,
                "all32_original_science_states_reused": True,
                "schedule_amendments": ["schedule-amendment-v6.json", "schedule-amendment-v7.json"],
            })
            launcher.write(BATCH / "controller-state.json", {
                "state": "complete", "passed": True, "resumed": "v7", "gpu": GPU})
        except BaseException as error:
            launcher.write(BATCH / "controller-failure-resume-v7.json", {"error": repr(error)})
            raise


if __name__ == "__main__":
    main()
