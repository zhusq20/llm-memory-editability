"""Continue the existing Loop TTCD pipeline alongside Transformer on GPU8."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

BASE = Path("/ossfs/workspace/llm-memory-editability")
SPEEDUP = BASE / "ttcd-speedup-v1"
AMENDMENT = SPEEDUP / "gpu8-parallel-20261015"
ENGINE = (
    SPEEDUP
    / "pending-stage-dependency-repair-v2-20261015/source/scripts/resume_ttcd_pending_stages.py"
)
ENGINE_SHA256 = "6772beaed58ea91c75fbe175e29c42747ee0898a6a7185b4e730abb41510b18a"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def module(path, name):
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def install(engine, ctx, policy):
    slots = policy.CohortSlots()
    ctx.controller.admit = lambda plan, arm: policy.admission(ctx.helpers, plan["runtime"], arm)
    original = ctx.controller.build_command

    def build(plan, arm, spec, batch, token, row):
        name, command = original(plan, arm, spec, batch, token, row)
        for i, value in enumerate(command):
            if value.startswith("WANDB_TAGS="):
                command[i] = f"WANDB_TAGS=ttcd-architecture,{arm},gpu-8,shared,user-authorized"
        index = command.index(plan["runtime"]["image"])
        command[index:index] = [
            "--env",
            "LM_GPU_SHARING=1",
            "--env",
            f"LM_GPU8_SCHEDULE_SHA256={sha(AMENDMENT / 'source-manifest.json')}",
        ]
        return name, command

    def choose(context, arm, deadline, update, original_admit):
        while time.monotonic() < deadline:
            lease = None
            try:
                lease = slots.take(arm)
                row = policy.admission(ctx.helpers, ctx.config["runtime"], arm)
                update(
                    {
                        "actual_gpu": 8,
                        "gpu_sharing": True,
                        "reason": None,
                        "gpu_fallback": None,
                        "gpu8_cohort": row["cohort"],
                    }
                )
                return lease, row
            except (BlockingIOError, ctx.helpers.ResourceUnavailable) as error:
                if lease is not None:
                    lease.close()
                update(
                    {
                        "state": "waiting_for_resources",
                        "gpu": 8,
                        "gpu_sharing": True,
                        "reason": str(error),
                    }
                )
                time.sleep(min(30, max(0, deadline - time.monotonic())))
            except BaseException:
                if lease is not None:
                    lease.close()
                raise
        raise TimeoutError("Original resource waiting deadline expired")

    ctx.controller.build_command = build
    engine.exclusive_choice = choose


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=("architecture",), default="architecture")
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.capture and not args.execute:
        print(
            json.dumps(
                {"executed": False, "gpu": 8, "gpu_sharing": True, "new_training_updates": 0}
            )
        )
        return 0
    if sha(ENGINE) != ENGINE_SHA256:
        raise ValueError("The reviewed continuation engine changed")
    frozen = json.loads((AMENDMENT / "source-manifest.json").read_text())
    if not frozen["engineering_passed"] or frozen["engine_sha256"] != ENGINE_SHA256:
        raise ValueError("Passing engineering verification and original engine are required")
    for relative, expected in frozen["files"].items():
        if sha(AMENDMENT / "source" / relative) != expected:
            raise ValueError("GPU8 resource amendment changed after verification")
    engine = module(ENGINE, "gpu8_continuation_engine")
    engine.AMENDMENT = AMENDMENT
    # The engine records this executable wrapper as the canonical recovery entry.
    engine.__file__ = str(Path(__file__).resolve())
    adapter = engine.module(
        SPEEDUP / "source/scripts/accelerate_ttcd_controllers.py", "gpu8_adapter"
    )
    ctx = engine.context(adapter, "architecture")
    directory = AMENDMENT / "architecture"
    directory.mkdir(parents=True, exist_ok=True)
    if args.capture:
        snapshot = engine.capture(adapter, ctx)
        adapter.write_json(directory / "snapshot.json", snapshot, exclusive=True)
        print(json.dumps({"captured": True, "model_restarted": False}))
        return 0
    policy = module(Path(__file__).parent / "ttcd_gpu8_resources.py", "gpu8_resource_policy")
    policy.admission(ctx.helpers, ctx.config["runtime"], "loop")
    install(engine, ctx, policy)
    transport = engine.module(
        SPEEDUP / "container-copy-repair-20261015/source/scripts/ttcd_speedup_transport.py",
        "gpu8_transport",
    )
    return engine.execute(
        adapter, transport, ctx, adapter.read_json(directory / "snapshot.json"), directory
    )


if __name__ == "__main__":
    raise SystemExit(main())
