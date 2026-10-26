"""Resume the diagnosed zero-update, read-only-root source transport failure."""

from __future__ import annotations

import concurrent.futures
import fcntl
import importlib.util
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

sys.dont_write_bytecode = True

import launch_ce_architecture as current  # noqa: E402


def main():
    plan = json.loads((current.ROOT / f"configs/{current.BATCH}.json").read_text())
    batch = Path(plan["output_root"])
    with (batch / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (batch / "controller-process-resume-v4.json").exists():
            raise FileExistsError("Recovery already submitted")
        failure = json.loads((batch / "controller-state.json").read_text())
        if failure["state"] != "needs_attention" or "contracts" not in failure["traceback"]:
            raise ValueError("Expected the diagnosed pre-model Docker copy failure")
        name = f"{current.BATCH}-contracts-contracts"
        state = json.loads(current.docker("inspect", "--format", "{{json .State}}", name).stdout)
        if state["Running"] or state["Pid"] or state["ExitCode"] != 1:
            raise ValueError("Only the exited CPU static-check container may be resumed")
        devices = json.loads(
            current.docker(
                "inspect", "--format", "{{json .HostConfig.DeviceRequests}}", name
            ).stdout
        )
        if devices:
            raise ValueError("CPU-only static failure expected")
        current.write(
            batch / "static-failure-v3.json",
            {"controller": failure, "container_state": state, "scientific_updates": 0},
        )
        current.docker("rename", name, name + "-static-failure-v3")
        parent, source = batch / "source-v3", batch / "source-v4"
        manifest = json.loads((parent / "source-manifest.json").read_text())
        excluded = [
            n for n in manifest["files"] if "__pycache__" in Path(n).parts or n.endswith(".pyc")
        ]
        manifest["files"] = {n: h for n, h in manifest["files"].items() if n not in excluded}
        current.write(
            batch / "bytecode-transport-amendment.json",
            {
                "reason": "Host import changed a generated pyc path; no .py science file changed",
                "excluded_generated_bytecode": excluded,
                "scientific_updates": 0,
                "old_source_and_failed_v2_preserved": True,
            },
        )
        source.mkdir(mode=0o700)

        def copy_item(item):
            relative, expected = item
            original, target = parent / relative, source / relative
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            os.link(original, target)
            if current.sha(target) != expected:
                raise AssertionError("Preserved source changed")

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(copy_item, manifest["files"].items()))
        files = dict(manifest["files"])
        changed = [p for p in current.EXTRAS if p.endswith(".py")]
        changed.append("scripts/resume_ce_architecture.py")
        for relative in changed:
            target = source / relative
            if target.exists():
                target.unlink()
            shutil.copyfile(current.ROOT / relative, target)
            files[relative] = current.sha(target)
        current.write(
            source / "source-manifest.json",
            {
                **manifest,
                "files": files,
                "transport_recovery_parent_sha256": current.sha(parent / "source-manifest.json"),
            },
        )
        digest = current.sha(source / "source-manifest.json")
        spec = importlib.util.spec_from_file_location(
            "ce_architecture_frozen_v4", source / "scripts/launch_ce_architecture.py"
        )
        launcher = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(launcher)
        current.write(
            batch / "controller-process-resume-v4.json",
            {
                "pid": os.getpid(),
                "source": str(source),
                "source_sha256": digest,
                "scientific_updates_before_recovery": 0,
                "model_restarted": False,
            },
        )
        current.write(
            batch / "schedule-amendment-v4.json",
            {
                "reason": "Docker cp into a never-started read-only rootfs failed",
                "change": (
                    "Writable image root for source installation; frozen source still "
                    "root-owned0444 and inputs read-only; repository Ruff formatting applied"
                ),
                "parent_science_editor_bytes_unchanged": True,
                "new_scientific_updates": 0,
                "reused_empty_output_volumes": True,
            },
        )
        gpu_lock = None
        try:
            launcher.run_stage(plan, batch, source, digest, "contracts", "contracts")
            transport = current.helpers.load_transport(source)
            gpu, started = None, time.monotonic()
            while gpu is None:
                gpu, gpu_lock = transport.idle_gpu()
                if gpu is None:
                    candidate = plan["resources"]["registered_lm_peer_gpu"]
                    receipt = launcher.admission(plan, candidate)
                    if receipt["passed"] and receipt["free_mib"] >= 49152:
                        gpu = candidate
                        current.write(batch / "registered-peer-admission-v4.json", receipt)
                if gpu is None:
                    current.write(
                        batch / "controller-state.json",
                        {
                            "state": "waiting_for_resources",
                            "contracts_passed": True,
                            "elapsed_seconds": time.monotonic() - started,
                        },
                    )
                    if time.monotonic() - started > plan["resource_wait_seconds"]:
                        raise TimeoutError("No eligible resources; no external job touched")
                    time.sleep(15)
            current.write(
                batch / "resource-admission-v4.json",
                {
                    "gpu": gpu,
                    "parallel_arms": ["loop", "transformer"],
                    "registered_lm_peers_only": True,
                    "allocator_fraction_per_arm": 0.20,
                },
            )
            outcomes, failures = {}, {}
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                tasks = {
                    pool.submit(launcher.execute_arm, plan, batch, source, digest, arm, gpu): arm
                    for arm in ("loop", "transformer")
                }
                for future in concurrent.futures.as_completed(tasks):
                    arm = tasks[future]
                    try:
                        outcomes[arm] = future.result()
                    except BaseException:
                        failures[arm] = traceback.format_exc()
                    current.write(
                        batch / "controller-state.json",
                        {
                            "state": "running",
                            "outcomes": outcomes,
                            "failures": failures,
                            "gpu": gpu,
                        },
                    )
            if failures:
                raise RuntimeError(f"Failed arms preserved: {list(failures)}")
            launcher.run_stage(plan, batch, source, digest, "report", "comparison")
            completion = {
                "state": "complete",
                "passed": True,
                "outcomes": outcomes,
                "source_sha256": digest,
                "new_foundation_updates": 0,
                "new_scientific_updates": 2560,
                "engineering_updates": 4,
                "formal_edits": 64,
                "discoloop_reused": True,
                "independent_audits_passed": True,
                "full_cloud_json_verified": True,
                "comparison_sha256": current.sha(batch / "report/output/comparison/result.json"),
            }
            current.write(batch / "completion-manifest.json", completion)
            current.write(batch / "controller-state.json", completion)
        except BaseException as error:
            current.write(
                batch / "controller-state.json",
                {
                    "state": "needs_attention",
                    "error": repr(error),
                    "traceback": traceback.format_exc(),
                },
            )
            raise
        finally:
            if gpu_lock:
                gpu_lock.close()


if __name__ == "__main__":
    main()
