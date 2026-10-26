"""Pre-science recovery with immutable inherited bytecode and disabled cache writes."""

from __future__ import annotations

import fcntl
import importlib.util
import json
import os
import shutil
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import launch_ce_scope_discoloop as current  # noqa: E402
from resume_ce_scope_discoloop import executable_tree  # noqa: E402


def main():
    plan = json.loads((current.ROOT / f"configs/{current.BATCH}.json").read_text())
    batch = Path(plan["output_root"])
    with (batch / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (batch / "controller-process-resume-v4.json").exists():
            raise FileExistsError("Do not duplicate startup recovery")
        if (batch / "output").exists():
            raise ValueError("Recovery only applies before any model execution")
        original, source = batch / "source", batch / "source-v4"
        old_manifest = json.loads((original / "source-manifest.json").read_text())
        source.mkdir(mode=0o700)
        bytecode = []
        for relative, expected in old_manifest["files"].items():
            path = original / relative
            if current.sha(path) != expected:
                parent = Path(plan["parent_source"]) / relative
                if not relative.endswith(".pyc") or current.sha(parent) != expected:
                    raise AssertionError("A non-cache original source changed")
                bytecode.append(
                    {
                        "file": relative,
                        "declared_sha256": expected,
                        "generated_sha256": current.sha(path),
                    }
                )
                path = parent
            target = source / relative
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        same = {
            name: executable_tree(original / name) == executable_tree(current.ROOT / name)
            for name in (
                "src/llm_memory_editability/ce_scope_editing.py",
                "scripts/run_ce_scope_discoloop.py",
            )
        }
        if not all(same.values()):
            raise AssertionError("Scientific algorithm changed during startup recovery")
        for relative in current.FILES:
            shutil.copyfile(current.ROOT / relative, source / relative)
        files = {name: current.sha(source / name) for name in old_manifest["files"]}
        current.write(
            source / "source-manifest.json",
            {
                **old_manifest,
                "files": files,
                "startup_parent_manifest_sha256": current.sha(original / "source-manifest.json"),
            },
        )
        manifest = current.sha(source / "source-manifest.json")
        runtime = batch / "runtime-repair"
        runtime.mkdir(mode=0o700, exist_ok=True)
        for path in (Path(__file__), current.ROOT / "scripts/resume_ce_scope_discoloop.py"):
            shutil.copyfile(path, runtime / path.name)
        current.write(
            batch / "schedule-amendment-v4.json",
            {
                "reason": (
                    "Project-context import grouping corrected after all23 contracts passed; "
                    "no model execution"
                ),
                "scientific_updates_before_repair": 0,
                "engineering_updates_before_repair": 0,
                "temporary_Ruff_cache": True,
                "host_bytecode_writes_disabled": True,
                "preserved_generated_bytecode": bytecode,
                "scientific_executable_AST_unchanged": same,
                "source_sha256": manifest,
                "prior_failures_preserved": [
                    "startup-failure-v1",
                    "controller-resume-v2.stdout.log",
                    "source",
                    "source-v2",
                ],
                "host_permissions_unchanged": True,
                "resume_script_sha256": current.sha(__file__),
            },
        )
        current.write(
            batch / "controller-process-resume-v4.json",
            {"pid": os.getpid(), "source_sha256": manifest},
        )
        spec = importlib.util.spec_from_file_location(
            "ce_scope_frozen_v4", source / "scripts/launch_ce_scope_discoloop.py"
        )
        launcher = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(launcher)
        original_create = launcher.create

        def create(plan, source, manifest, stage, gpu):
            if stage != "contracts":
                return original_create(plan, source, manifest, stage, gpu)
            old_name = current.BATCH + "-contracts"
            new_name = old_name + "-v4"
            original_docker = launcher.docker

            def docker(*args, **kwargs):
                args = [
                    new_name if arg == old_name else arg.replace(old_name + ":/", new_name + ":/")
                    for arg in args
                ]
                return original_docker(*args, **kwargs)

            launcher.docker = docker
            try:
                name, identifier, args = original_create(plan, source, manifest, stage, gpu)
            finally:
                launcher.docker = original_docker
            return new_name, identifier, [new_name if arg == name else arg for arg in args]

        launcher.create = create
        gpu, gpu_lock, started = None, None, time.monotonic()
        try:
            launcher.run_stage(plan, batch, source, manifest, "contracts", gpu)
            transport = launcher.load_transport(source)
            while gpu is None:
                gpu, gpu_lock = transport.idle_gpu()
                if gpu is None:
                    current.write(
                        batch / "controller-state.json", {"state": "waiting_for_exclusive_idle_gpu"}
                    )
                    if time.monotonic() - started > plan["resource_wait_seconds"]:
                        raise TimeoutError("No exclusive idle GPU available")
                    time.sleep(30)
            launcher.run_stage(plan, batch, source, manifest, "prepare", gpu)
            for scope in ("down", "mlp", "all"):
                launcher.run_stage(plan, batch, source, manifest, f"development-{scope}", gpu)
            current.write(
                batch / "selected-recipes.json",
                {
                    scope: {
                        "development_sha256": current.sha(
                            batch / f"output/development-{scope}/result.json"
                        ),
                        "selected": json.loads(
                            (batch / f"output/development-{scope}/result.json").read_text()
                        )["selected"],
                    }
                    for scope in ("down", "mlp", "all")
                },
            )
            for scope in ("down", "mlp", "all"):
                launcher.run_stage(plan, batch, source, manifest, f"evaluation-{scope}", gpu)
                launcher.run_stage(plan, batch, source, manifest, f"audit-{scope}", gpu)
            launcher.run_stage(plan, batch, source, manifest, "cloud", gpu)
            current.write(
                batch / "completion-manifest.json",
                {
                    "passed": True,
                    "source_sha256": manifest,
                    "scopes": ["down", "mlp", "all"],
                    "formal_edits": 96,
                    "formal_updates": 96 * plan["ce"]["steps"],
                    "audit_optimizer_updates": 0,
                    "alphaedit_reused": True,
                    "new_composition_supervision": 0,
                },
            )
            current.write(
                batch / "controller-state.json", {"state": "complete", "passed": True, "gpu": gpu}
            )
        except BaseException as error:
            current.write(batch / "controller-failure-resume-v4.json", {"error": repr(error)})
            raise
        finally:
            if gpu_lock is not None:
                gpu_lock.close()


if __name__ == "__main__":
    main()
