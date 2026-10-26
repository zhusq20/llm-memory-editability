"""Continue only after repairing the zero-update CUDA preflight lifetime bug."""

import fcntl
import importlib.util
import os
import shutil
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BATCH = "loop-query-control-development-v1"
BASE = Path("/ossfs/workspace/llm-memory-editability") / BATCH
ATTEMPT = "capture-v3"


def load():
    path = Path(__file__).with_name("resume_loop_query_control_transport.py")
    spec = importlib.util.spec_from_file_location("reviewed_query_transport_v3", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, module.original()


def command(transport, module, config, stage, row, attempt):
    name, args = transport.create_command(module, config, stage, row, attempt)
    if stage == "preflight":
        image_index = args.index(config["runtime"]["image"])
        wrapper = BASE / "runtime-repair/capture-v3/source/verify_loop_query_control_preflight.py"
        args[image_index:image_index] = [
            "--mount",
            f"type=bind,source={wrapper},target=/runtime/preflight.py,readonly",
        ]
        target = args.index("/frozen/scripts/run_loop_query_control.py")
        args[target] = "/runtime/preflight.py"
    return name, args


def main():
    transport, module = load()
    with (BASE / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = module.read(BASE / "controller-state.json")
        old = module.read(BASE / "controller-process-resume-v2.json")
        try:
            os.kill(old["pid"], 0)
        except ProcessLookupError:
            pass
        else:
            raise RuntimeError("Previous controller must have exited")
        if state["state"] != "needs_attention" or state["stage"] != "preflight":
            raise ValueError("Only the diagnosed zero-update preflight failure may be repaired")
        if (BASE / "run/query_control").exists() or (BASE / "run/wandb-train.json").exists():
            raise ValueError("Scientific training already claimed; no duplicate run")
        if module.read(BASE / "execution/transport-v2/contracts/exit.json")["code"] != 0:
            raise ValueError("The frozen CPU contracts must have passed")
        folder = BASE / "runtime-repair/capture-v3"
        folder.mkdir(exist_ok=False)
        source = folder / "source"
        source.mkdir(mode=0o755)
        for name in (
            "resume_loop_query_control_capture.py",
            "resume_loop_query_control_transport.py",
            "verify_loop_query_control_preflight.py",
            "verify_loop_query_control_capture.py",
        ):
            shutil.copy2(ROOT / "scripts" / name, source / name)
        # The daemon only needs traversal of the new readonly amendment directories.
        for path in (BASE / "runtime-repair", folder, source):
            path.chmod(0o755)
        shutil.copy2(BASE / "controller-state.json", folder / "failed-controller-state.json")
        module.write(
            folder / "amendment.json",
            {
                "scientific_updates_before_repair": 0,
                "change": "Only free eager logits/loss references before preflight capture; "
                "formal trainer unchanged",
                "frozen_science_manifest_sha256": module.sha(BASE / "freeze.json"),
                "files": {p.name: module.sha(p) for p in source.iterdir()},
                "exact_formal_path_capture_probe": {
                    "passed": True,
                    "weights_restored": True,
                    "surviving_updates": 0,
                },
                "utc": module.utc(),
            },
        )
        module.write(
            BASE / "controller-process-resume-v3.json",
            {
                "pid": os.getpid(),
                "utc": module.utc(),
                "amendment_sha256": module.sha(folder / "amendment.json"),
            },
        )
        config = module.read(BASE / f"source/configs/{BATCH}.json")
        state = {"state": "running", "attempt": ATTEMPT, "utc": module.utc()}
        try:
            for stage in ("preflight", "train", "audit", "cloud-audit"):
                row = None
                deadline = time.monotonic() + 24 * 3600
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
                module.write(folder / f"resource-admission-{stage}.json", row)
                state.update(state="running", stage=stage, utc=module.utc())
                module.write(BASE / "controller-state.json", state)
                transport.execute(
                    module,
                    config,
                    stage,
                    None if stage == "cloud-audit" else row,
                    ATTEMPT,
                    lambda *args: command(transport, *args),
                )
                if (
                    stage == "preflight"
                    and not module.read(BASE / "run/result-preflight.json")["passed"]
                ):
                    raise ValueError("Repaired full-size preflight must pass before training")
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
