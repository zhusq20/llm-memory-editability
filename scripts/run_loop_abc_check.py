#!/usr/bin/env python3
"""Exactly 150 updates per ABC arm, independent reload, and cloud readback."""

import argparse
import copy
import json
import math
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import run_loop_fourway as fourway  # noqa: E402
import run_loop_native_handoff as native  # noqa: E402

from llm_memory_editability.loop_abc_training import abc_loss, auxiliary_depth  # noqa: E402

BATCH = "loop-abc-method-check-150-v1"
_RATE = native.learning_rate
_WRITE = native.write_json
RECORDS = []


def validate(config):
    if config["batch"] != BATCH or list(config["arms"]) != ["A", "B", "C"]:
        raise ValueError("Only the new ABC check is allowed")
    if config["training"]["steps"] != 150 or config["budget"]["total_updates"] != 450:
        raise ValueError("User authorized exactly 150 updates per arm")
    if config["handoff"]["first_round_input"] != "embedding":
        raise ValueError("ABC must all use first-round e")


def initialize(config, name, metadata, device):
    view = copy.deepcopy(config)
    view["arms"][name] = {"handoff": "invariant"}
    model = fourway.initialize(view, name, metadata, device)
    model.abc_arm = name
    model.abc_step = 0
    return model


def backward(model, tokens, positions, labels, training, rng, device):
    model.abc_step += 1
    step = model.abc_step
    depth = auxiliary_depth(training["auxiliary_seed"], step)
    audit = step == 1
    with native.autocast(device):
        result = abc_loss(model, tokens, positions, labels, model.abc_arm, depth, audit)
    result[0].backward()
    record = {
        "step": step,
        "terminal_loss": float(result[1].detach()),
        "auxiliary_depth": depth if model.abc_arm == "C" else None,
        "auxiliary_loss": None if result[2] is None else float(result[2].detach()),
    }
    if audit:
        state = result[4]["first_state"]
        embedding = result[4]["embedding"]
        record["first_state_requires_grad"] = state.requires_grad
        record["first_state_gradient_norm"] = (
            None if state.grad is None else float(state.grad.norm())
        )
        record["injection_gradient_norm"] = float(embedding.grad.norm())
        assert record["injection_gradient_norm"] > 0
        if model.abc_arm == "B":
            assert not state.requires_grad and state.grad is None
        else:
            assert record["first_state_gradient_norm"] > 0
        total = torch.cuda.get_device_properties(device).total_memory / 2**20
        record["peak_reserved_mib"] = torch.cuda.max_memory_reserved(device) / 2**20
        assert record["peak_reserved_mib"] + 6144 < total * 0.16
    if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
        raise FloatingPointError("Nonfinite gradient")
    RECORDS.append(record)
    return result[0].detach(), [], np.full(len(tokens), 8, dtype=np.int64)


def learning_rate(training, step):
    return _RATE({**training, "steps": 1500000}, step)


def flops(model, metadata, depths, slots):
    total = fourway.terminal_step_flops(model, metadata, depths, slots)
    if model.abc_arm == "C":
        total += 6 * len(depths) * slots * model.config.n_embd * metadata["vocab_size"]
    return total


def write(path, value):
    if path.name == "run.json":
        value.update(
            objective={
                "A": "terminal CE at R8, full backward",
                "B": "terminal CE at R8, backward rounds5-8 only",
                "C": "0.9 terminal R8 CE + 0.1 random-depth CE, full backward",
            }[value["arm"]],
            lr_schedule_steps=1500000,
            validation_only=True,
        )
    if path.name == "complete.json":
        value["method_checks"] = RECORDS[0]
        value["peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
        value["peak_reserved_mib"] = torch.cuda.max_memory_reserved() / 2**20
        value["cost_estimate"] = "full-backprop shape upper bound; not exact for B"
        _WRITE(path.parent / "method-steps.json", RECORDS)
    _WRITE(path, value)


@contextmanager
def implementation():
    replacements = {
        "initialize": initialize,
        "validate_config": validate,
        "backward_batch": backward,
        "learning_rate": learning_rate,
        "terminal_step_flops": flops,
        "write_json": write,
    }
    previous = {key: getattr(native, key) for key in replacements}
    try:
        for key, value in replacements.items():
            setattr(native, key, value)
        yield
    finally:
        for key, value in previous.items():
            setattr(native, key, value)


def cloud(config, base, arm):
    import wandb
    from run_loop_recipe_pipeline import reload_matches

    out = base / "runs" / arm
    complete = json.loads((out / "complete.json").read_text())
    grid = json.loads((out / "grid.json").read_text())
    tracking = json.loads((out / "tracking.json").read_text())
    identity = f"{config['tracking']['entity']}/{config['tracking']['project']}/{tracking['id']}"
    remote = wandb.Api(timeout=90).run(identity)
    values = native.learning_metrics(complete["endpoint"])
    checks = {
        "training_step": complete["step"] == 150,
        "reload_step": grid["step"] == 150,
        "checkpoint": grid["checkpoint_sha256"] == complete["final_checkpoint_sha256"],
        "reload_metrics": all(
            reload_matches(metrics, grid["weights"][kind]["answer_and_eos"][r])
            for kind, horizons in complete["endpoint"]["metrics"].items()
            for r, metrics in horizons.items()
        ),
        "cloud_finished": remote.state == "finished",
        "cloud_config": remote.config.get("experiment") == config,
        "cloud_metrics": all(
            isinstance(remote.summary.get(k), (int, float))
            and math.isclose(remote.summary[k], v, rel_tol=1e-6, abs_tol=1e-7)
            for k, v in values.items()
        ),
    }
    report = {
        "passed": all(checks.values()),
        "checks": checks,
        "url": tracking["url"],
        "metrics_checked": len(values),
        "utc": native.utc(),
    }
    _WRITE(out / "cloud-verification.json", report)
    if not report["passed"]:
        raise RuntimeError("Reload or cloud readback failed; preserve all original evidence")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--arm", choices=["A", "B", "C"], required=True)
    parser.add_argument(
        "--stage", choices=["pipeline", "train", "grid", "cloud"], default="pipeline"
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    validate(config)
    os.environ["WANDB_API_KEY"] = Path("/run/secrets/wandb-key").read_text().strip()
    if args.stage == "pipeline":
        directory = args.base / "pipeline" / args.arm
        directory.mkdir(parents=True, exist_ok=False)
        state = {"state": "running", "arm": args.arm, "stages": {}}
        for stage in ["train", "grid", "cloud"]:
            command = [
                sys.executable,
                "-u",
                __file__,
                "--config",
                str(args.config),
                "--base",
                str(args.base),
                "--arm",
                args.arm,
                "--stage",
                stage,
            ]
            with (directory / f"{stage}.log").open("w") as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                state["stages"][stage] = {"pid": process.pid, "started_utc": native.utc()}
                _WRITE(directory / "state.json", state)
                code = process.wait()
            state["stages"][stage].update(exit_code=code, ended_utc=native.utc())
            state["state"] = "running" if code == 0 else "needs_attention"
            _WRITE(directory / "state.json", state)
            if code:
                raise RuntimeError(f"{stage} failed with exit {code}; no automatic retry")
        state.update(state="complete", completed_utc=native.utc())
        _WRITE(directory / "state.json", state)
        return
    if args.stage == "cloud":
        cloud(config, args.base, args.arm)
        return
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(0.16)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda:0")
    with implementation():
        if args.stage == "grid":
            native.grid(config, args.arm, args.base, args.config, device)
        else:
            out = args.base / "runs" / args.arm
            holder = {}

            def track(metadata):
                holder["log"] = native.tracker_for(config, args.arm, out, args.config)
                return holder["log"]

            try:
                native.train(config, args.arm, out, False, args.config, track, device)
            except BaseException:
                if "log" in holder:
                    holder["log"].run.finish(exit_code=1)
                raise
            else:
                holder["log"].run.finish()


if __name__ == "__main__":
    main()
