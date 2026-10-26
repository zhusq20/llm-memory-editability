"""Bounded sharing on an already-owned LM GPU; reuse complete development candidates."""

from __future__ import annotations

import fcntl
import functools
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import launch_ce_scope_discoloop as current  # noqa: E402


def admission(launcher, plan):
    allowed = set()
    for name in plan["resources"]["peers"]:
        state = json.loads(launcher.docker("inspect", "--format", "{{json .State}}", name).stdout)
        if state["Running"]:
            rows = launcher.docker("top", name, "-eo", "pid").stdout.decode().splitlines()[1:]
            allowed.update(int(row.strip()) for row in rows)
    uuid = plan["resources"]["gpu_uuid"]
    rows = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    actual = {int(row.split(",")[1]) for row in rows if uuid in row}
    if not actual <= allowed:
        raise RuntimeError("GPU4 has a process outside the two identified LM peer containers")
    free = int(
        subprocess.run(
            [
                "nvidia-smi",
                "--id",
                uuid,
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    if free < 16384:
        raise RuntimeError("Insufficient GPU4 headroom for bounded CE writes")
    return {
        "resident_pids": sorted(actual),
        "free_mib": free,
        "peers": plan["resources"]["peers"],
        "torch_allocator_fraction": 0.06,
    }


def main():
    plan_path = current.ROOT / f"configs/{current.BATCH}-resume-v6.json"
    plan = json.loads(plan_path.read_text())
    batch = Path(plan["output_root"])
    original_plan = json.loads((batch / f"source-v5/configs/{current.BATCH}.json").read_text())
    for key in (
        "ce",
        "learning_rates",
        "minimum_development_retention",
        "checkpoint_sha256",
        "panel_sha256",
        "protection",
        "statistics",
    ):
        if plan[key] != original_plan[key]:
            raise AssertionError("Recovery changed a scientific setting")
    gpu = {"index": plan["resources"]["gpu_index"], "uuid": plan["resources"]["gpu_uuid"]}
    with (batch / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (batch / "controller-process-resume-v6.json").exists():
            raise FileExistsError("Do not duplicate the bounded-sharing handoff")
        prior_exit = json.loads((batch / "exit-development-down.json").read_text())
        if prior_exit["ExitCode"] != 137:
            raise ValueError("Expected preserved development SIGKILL exit")
        parent, source = batch / "source-v5", batch / "source-v6"
        old_manifest = json.loads((parent / "source-manifest.json").read_text())
        source.mkdir(mode=0o700)
        for name, expected in old_manifest["files"].items():
            path = parent / name
            if current.sha(path) != expected:
                raise AssertionError(f"Pre-recovery frozen source changed: {name}")
            dest = source / name
            dest.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            shutil.copyfile(path, dest)
        for name in current.FILES:
            if name.endswith(".py"):
                shutil.copyfile(current.ROOT / name, source / name)
        shutil.copyfile(plan_path, source / f"configs/{current.BATCH}.json")
        shutil.copyfile(plan_path, source / plan_path.relative_to(current.ROOT))
        writer = "src/llm_memory_editability/ce_scope_editing.py"
        if current.sha(parent / writer) != current.sha(source / writer):
            raise AssertionError("Parameter editor changed during scheduling recovery")
        files = {
            str(p.relative_to(source)): current.sha(p)
            for p in sorted(source.rglob("*"))
            if p.is_file() and p != source / "source-manifest.json"
        }
        current.write(
            source / "source-manifest.json",
            {
                **old_manifest,
                "files": files,
                "recovery_parent_sha256": current.sha(parent / "source-manifest.json"),
            },
        )
        manifest = current.sha(source / "source-manifest.json")
        runtime = batch / "runtime-repair"
        shutil.copyfile(__file__, runtime / Path(__file__).name)
        spec = importlib.util.spec_from_file_location(
            "ce_scope_frozen_v6", source / "scripts/launch_ce_scope_discoloop.py"
        )
        launcher = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(launcher)
        receipt = admission(launcher, plan)
        current.write(
            batch / "schedule-amendment-v6.json",
            {
                "reason": (
                    "Exit137 without OOM; possible per-user fifth-GPU rejection; "
                    "guard config/journal not readable"
                ),
                "original_exit": prior_exit,
                "failed_run": plan["recovery_accounting"]["old_run"],
                "changed": (
                    "GPU4 bounded sharing, same CE algorithm and fixed budgets, "
                    "two completed candidates reused"
                ),
                "admission": receipt,
                "scientific_writer_bytes_unchanged": True,
                "source_sha256": manifest,
                "runtime_root": plan["runtime_root"],
                "reused_completed_updates": 320,
                "uncheckpointed_updates_range": [20, 40],
                "formal_updates_before_handoff": 0,
                "host_permissions_unchanged": True,
                "resume_script_sha256": current.sha(__file__),
                "timing_is_shared_not_exclusive": True,
            },
        )
        current.write(
            batch / "controller-process-resume-v6.json",
            {"pid": os.getpid(), "source_sha256": manifest, "gpu": gpu},
        )
        launcher.docker(
            "run",
            "--name",
            current.BATCH + "-reuse-prepare-v6",
            "--user",
            "1015:1016",
            "--network",
            "none",
            "--read-only",
            "--cpus",
            "1",
            "--memory",
            "2g",
            "--mount",
            f"type=volume,source={current.VOLUME},target=/out",
            plan["image"],
            "python",
            "-c",
            (
                "from pathlib import Path; import shutil; "
                "p=Path('/out/attempt-6'); p.mkdir(mode=0o700); "
                "shutil.copytree('/out/prepare',p/'prepare',symlinks=True)"
            ),
        )
        launcher.persist = functools.partial(
            launcher.persist,
            runtime_root=plan["runtime_root"],
            host_root=batch / "output/attempt-6",
        )
        launcher.persist(batch, current.BATCH + "-reuse-prepare-v6", "prepare")
        original_create = launcher.create

        def create(plan, source, manifest, stage, gpu):
            old_name = current.BATCH + "-" + stage
            new_name = old_name + "-v6"
            original_docker = launcher.docker

            def amend(args):
                args = [
                    new_name if arg == old_name else arg.replace(old_name + ":/", new_name + ":/")
                    for arg in args
                ]
                if args[0] == "create":
                    if "--root" in args:
                        args[args.index("--root") + 1] = plan["runtime_root"]
                    if "--gpus" in args:
                        image = args.index(plan["image"])
                        command = args[image + 1 :]
                        wrapper = (
                            "import torch,sys,runpy; "
                            "torch.cuda.set_per_process_memory_fraction(0.06); "
                            "sys.argv=sys.argv[1:]; "
                            "runpy.run_path(sys.argv[0],run_name='__main__')"
                        )
                        args[image + 1 :] = ["python", "-c", wrapper, *command[1:]]
                return args

            def docker(*args, **kwargs):
                return original_docker(*amend(list(args)), **kwargs)

            launcher.docker = docker
            try:
                _, identifier, args = original_create(plan, source, manifest, stage, gpu)
            finally:
                launcher.docker = original_docker
            return new_name, identifier, amend(args)

        launcher.create = create
        try:
            launcher.run_stage(plan, batch, source, manifest, "contracts", None)
            for scope in ("down", "mlp", "all"):
                current.write(
                    batch / f"admission-development-{scope}-v6.json", admission(launcher, plan)
                )
                launcher.run_stage(plan, batch, source, manifest, f"development-{scope}", gpu)
            output = batch / "output/attempt-6"
            current.write(
                batch / "selected-recipes-v6.json",
                {
                    scope: {
                        "development_sha256": current.sha(
                            output / f"development-{scope}/result.json"
                        ),
                        "selected": json.loads(
                            (output / f"development-{scope}/result.json").read_text()
                        )["selected"],
                    }
                    for scope in ("down", "mlp", "all")
                },
            )
            for scope in ("down", "mlp", "all"):
                for kind in ("evaluation", "audit"):
                    current.write(
                        batch / f"admission-{kind}-{scope}-v6.json", admission(launcher, plan)
                    )
                    launcher.run_stage(plan, batch, source, manifest, f"{kind}-{scope}", gpu)
            launcher.run_stage(plan, batch, source, manifest, "cloud", None)
            current.write(
                batch / "completion-manifest.json",
                {
                    "passed": True,
                    "source_sha256": manifest,
                    "scopes": ["down", "mlp", "all"],
                    "formal_edits": 96,
                    "formal_updates": 1920,
                    "unique_development_updates_including_reused": 1920,
                    "extra_uncheckpointed_development_updates_range": [20, 40],
                    "engineering_updates": 6,
                    "audit_optimizer_updates": 0,
                    "alphaedit_reused": True,
                    "new_composition_supervision": 0,
                    "output_root": "output/attempt-6",
                },
            )
            current.write(
                batch / "controller-state.json",
                {"state": "complete", "passed": True, "gpu": gpu, "bounded_shared": True},
            )
        except BaseException as error:
            current.write(batch / "controller-failure-resume-v6.json", {"error": repr(error)})
            raise


if __name__ == "__main__":
    main()
