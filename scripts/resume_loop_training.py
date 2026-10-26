#!/usr/bin/env python3
"""Resume a frozen Loop training function from its complete latest.pt state.

Only initialization/restore and tracking setup change. The original update,
measurement and checkpoint code remains intact and its source hash is pinned.
"""

from __future__ import annotations

import ast
import datetime
import hashlib
import inspect
import json
import os
import sys
from pathlib import Path

FROZEN_TRAINER_SHA256 = "f4071f805238ebc7d98516c842fce0153387b22648ce382d1bd86562e9e6a9d0"


def restore(
    native,
    out,
    config,
    config_path,
    name,
    metadata,
    model,
    optimizer,
    average,
    stream,
    schedule_rng,
    device,
):
    import torch

    if (out / "complete.json").exists():
        raise RuntimeError("Completed training must never be resumed")
    checkpoint = out / "latest.pt"
    expected = os.environ.get("LOOP_RESUME_CHECKPOINT_SHA256")
    if expected and native.digest(checkpoint) != expected:
        raise ValueError("Resume checkpoint hash changed")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    required = {
        "model",
        "ema",
        "optimizer",
        "stream",
        "schedule_rng",
        "counts",
        "cpu_rng",
        "cuda_rng",
        "training_seconds",
        "step",
        "config_sha256",
        "world_sha256",
        "model_sha256",
        "arm",
    }
    if required - saved.keys():
        raise ValueError(f"Incomplete checkpoint: {required - saved.keys()}")
    if saved["config_sha256"] != native.digest(config_path):
        raise ValueError("Checkpoint configuration differs")
    if saved["world_sha256"] != metadata["world_sha256"] or saved["arm"] != name:
        raise ValueError("Checkpoint belongs to a different world or arm")
    step = saved["step"]
    if not 0 < step < config["training"]["steps"]:
        raise ValueError("Resume step must be strictly inside the original budget")
    history = json.loads((out / "learning.json").read_text())
    if history[-1]["step"] != step:
        raise ValueError("Checkpoint and measured learning history disagree")
    model.load_state_dict(saved["model"], strict=True)
    if native.model_digest(model) != saved["model_sha256"]:
        raise ValueError("Restored model hash differs")
    optimizer.load_state_dict(saved["optimizer"])
    if not optimizer.state or any(int(s["step"].item()) != step for s in optimizer.state.values()):
        raise ValueError("Optimizer update counts differ from checkpoint step")
    average.load_state_dict({k: v.to(device) for k, v in saved["ema"].items()})
    stream.load_state_dict(saved["stream"])
    schedule_rng.bit_generator.state = saved["schedule_rng"]
    counts = saved["counts"].copy()
    if counts.shape != (stream.size,):
        raise ValueError("Exposure counts differ from the original data stream")
    receipt = {
        "utc": native.utc(),
        "step": step,
        "budget": config["training"]["steps"],
        "checkpoint_sha256": native.digest(checkpoint),
        "model_sha256": native.model_digest(model),
        "restored": sorted(required),
        "original_update_loop_unchanged": True,
    }
    receipt_path = Path(os.environ.get("LOOP_RESUME_RECEIPT", str(out / "resume.json")))
    if receipt_path.exists():
        raise FileExistsError("This recovery attempt already has a receipt")
    native.write_json(receipt_path, receipt)
    torch.set_rng_state(saved["cpu_rng"])
    if device.type == "cuda":
        if saved["cuda_rng"] is None:
            raise ValueError("CUDA random state missing")
        torch.cuda.set_rng_state(saved["cuda_rng"], device)
    original_run = json.loads((out / "run.json").read_text())
    elapsed = (
        datetime.datetime.now(datetime.timezone.utc)
        - datetime.datetime.fromisoformat(original_run["started_utc"])
    ).total_seconds()
    return (
        history,
        step,
        history[-1]["estimated_matmul_training_flops"],
        history[-1]["executed_rounds"],
        saved["training_seconds"],
        counts,
        elapsed,
    )


def build_resume(native):
    source_path = Path(native.__file__)
    if hashlib.sha256(source_path.read_bytes()).hexdigest() != FROZEN_TRAINER_SHA256:
        raise ValueError("Unexpected frozen trainer source")
    original = inspect.getsource(native.train)
    source = original
    replacements = [
        (
            "out.mkdir(parents=True, exist_ok=False)",
            'assert out.is_dir(), "An existing training directory is required"',
        ),
        (
            'write_json(out / "run.json", run_metadata)',
            'run_metadata = json.loads((out / "run.json").read_text())',
        ),
        (
            "    measure()\n    model.train()\n    for end in nodes[1:]:",
            "    history, step, flops, executed_rounds, training_seconds, counts, elapsed = "
            "_migration_restore(out, config, config_path, name, metadata, model, optimizer, "
            "average, stream, schedule_rng, device)\n"
            "    started = time.perf_counter() - elapsed\n"
            "    nodes = [node for node in nodes if node > step]\n"
            "    model.train()\n    for end in nodes:",
        ),
    ]
    for before, after in replacements:
        if source.count(before) != 1:
            raise ValueError("Frozen training function no longer matches the recovery contract")
        source = source.replace(before, after)
    old_loop = next(
        n
        for n in ast.walk(ast.parse(original))
        if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == "end"
    )
    new_loop = next(
        n
        for n in ast.walk(ast.parse(source))
        if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == "end"
    )
    if ast.dump(ast.Module(body=old_loop.body, type_ignores=[])) != ast.dump(
        ast.Module(body=new_loop.body, type_ignores=[])
    ):
        raise AssertionError("Training update or evaluation loop changed")
    namespace = dict(native.__dict__)
    namespace["_migration_restore"] = lambda *args: restore(native, *args)
    exec(
        compile(source, str(Path(__file__).with_name("generated_resume_train.py")), "exec"),
        namespace,
    )
    return namespace["train"]


def resume_tracker(native, config, name, out, config_path):
    import wandb

    tracking = config["tracking"]
    previous = json.loads((out / "tracking.json").read_text())
    expected_id = f"lnh{native.digest(config_path)[:6]}{list(config['arms']).index(name):02d}"
    if previous["id"] != expected_id:
        raise ValueError("Tracking identity differs")
    run = wandb.init(
        entity=tracking["entity"],
        project=tracking["project"],
        group=config["batch"],
        name=name,
        id=expected_id,
        resume="must",
        config={"experiment": config},
        dir=str(out),
        mode=tracking["mode"],
        save_code=False,
        settings=wandb.Settings(disable_git=True),
    )
    floor = int(os.environ["LOOP_PREVIOUS_LOGGED_STEP"])

    def log(values, step):
        if step > floor:
            run.log(values, step=step)

    log.run = run
    return log


def main():
    sys.path[:0] = ["/project/src", "/project/scripts"]
    sys.path.append(str(Path(os.environ["LOOP_PARENT_SOURCE"]) / "scripts"))
    import run_loop_robustness as runner

    native = runner.native
    native.train = build_resume(native)
    native.tracker_for = lambda *args: resume_tracker(native, *args)
    runner.main()


if __name__ == "__main__":
    os.umask(0o022)
    main()
