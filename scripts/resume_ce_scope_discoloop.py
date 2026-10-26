"""Resume before science after preserving the read-only Ruff-cache failure."""

from __future__ import annotations

import ast
import fcntl
import importlib.util
import json
import os
import shutil
import time
from pathlib import Path

import launch_ce_scope_discoloop as current


def executable_tree(path):
    tree = ast.parse(path.read_text())
    tree.body = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
    return ast.dump(tree, include_attributes=False)


def main():
    plan = json.loads((current.ROOT / f"configs/{current.BATCH}.json").read_text())
    batch = Path(plan["output_root"])
    with (batch / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (batch / "controller-process-resume-v2.json").exists():
            raise FileExistsError("This startup handoff already exists")
        failed = json.loads((batch / "exit-contracts.json").read_text())
        if failed["ExitCode"] != 1 or (batch / "output").exists():
            raise ValueError("This repair is only for a pre-science contracts failure")
        original, source = batch / "source", batch / "source-v2"
        unchanged = {}
        for relative in (
            "src/llm_memory_editability/ce_scope_editing.py",
            "scripts/run_ce_scope_discoloop.py",
        ):
            unchanged[relative] = executable_tree(original / relative) == executable_tree(
                current.ROOT / relative
            )
        if not all(unchanged.values()):
            raise AssertionError("Startup repair must not change either scientific algorithm")
        archive = batch / "startup-failure-v1"
        archive.mkdir(mode=0o700)
        for name in (
            "contracts.log",
            "exit-contracts.json",
            "controller-failure.json",
            "controller-state.json",
            "controller.stdout.log",
        ):
            shutil.copyfile(batch / name, archive / name)
        shutil.copytree(original, source)
        for relative in current.FILES:
            shutil.copyfile(current.ROOT / relative, source / relative)
        old_manifest = json.loads((original / "source-manifest.json").read_text())
        files = {
            str(p.relative_to(source)): current.sha(p)
            for p in sorted(source.rglob("*"))
            if p.is_file() and p != source / "source-manifest.json"
        }
        changed = [name for name, sha in files.items() if sha != old_manifest["files"][name]]
        if any(name not in current.FILES for name in changed):
            raise AssertionError("Repair changed inherited source")
        current.write(
            source / "source-manifest.json",
            {
                **old_manifest,
                "files": files,
                "startup_parent_manifest_sha256": current.sha(original / "source-manifest.json"),
            },
        )
        manifest = current.sha(source / "source-manifest.json")
        spec = importlib.util.spec_from_file_location(
            "ce_scope_frozen_v2", source / "scripts/launch_ce_scope_discoloop.py"
        )
        launcher = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(launcher)
        original_create = launcher.create

        def create(plan, source, manifest, stage, gpu):
            if stage != "contracts":
                return original_create(plan, source, manifest, stage, gpu)
            old_name = current.BATCH + "-contracts"
            new_name = old_name + "-v2"
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
        current.write(
            batch / "schedule-amendment-v2.json",
            {
                "reason": "23 CPU contracts passed; Ruff attempted cache under immutable /frozen",
                "scientific_updates_before_repair": 0,
                "engineering_updates_before_repair": 0,
                "changed": (
                    "temporary Ruff cache, project formatting/imports and transport closure binding"
                ),
                "scientific_executable_AST_unchanged": unchanged,
                "changed_source_files": changed,
                "source_sha256": manifest,
                "old_failure": "startup-failure-v1",
                "host_permissions_unchanged": True,
                "resume_script_sha256": current.sha(__file__),
            },
        )
        current.write(
            batch / "controller-process-resume-v2.json",
            {"pid": os.getpid(), "source_sha256": manifest},
        )
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
            current.write(batch / "controller-failure-resume-v2.json", {"error": repr(error)})
            raise
        finally:
            if gpu_lock is not None:
                gpu_lock.close()


if __name__ == "__main__":
    main()
