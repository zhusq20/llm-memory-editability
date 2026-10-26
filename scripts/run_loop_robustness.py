#!/usr/bin/env python3
"""Run frozen invariant robustness jobs and streaming depth evaluations."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
if os.environ.get("LOOP_PARENT_SOURCE"):
    sys.path.append(str(Path(os.environ["LOOP_PARENT_SOURCE"]) / "scripts"))
import run_loop_native_handoff as native  # noqa: E402

from llm_memory_editability.grok_depth import utc, write_json  # noqa: E402
from llm_memory_editability.grokking_reproduction import digest  # noqa: E402
from llm_memory_editability.loop_hop_data import load, pack_rows, panel_names  # noqa: E402
from llm_memory_editability.loop_robustness import (  # noqa: E402
    nested_indices,
    stream_predictions,
    summarize_predictions,
)


def depth(config, name, run, destination, tiers, weights, verify_existing=False):
    destination.mkdir(parents=True, exist_ok=False)
    complete = json.loads((run / "complete.json").read_text())
    checkpoint = run / f"checkpoint-{complete['step']:07d}.pt"
    if digest(checkpoint) != complete["final_checkpoint_sha256"]:
        raise ValueError("Checkpoint hash changed")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    _, panels, metadata = load(config["input_root"])
    device = torch.device("cuda:0")
    model = native.initialize(config, name, metadata, device).eval()
    model.load_state_dict(saved["model"], strict=True)
    average = native.ParameterAverage(model, config["training"]["ema_decay"])
    average.load_state_dict(saved["ema"])
    started = time.monotonic()
    result = {
        "arm": name,
        "weights": weights,
        "checkpoint_sha256": digest(checkpoint),
        "world_sha256": metadata["world_sha256"],
        "tiers": {},
        "generated_eos_evaluated": False,
        "new_training_updates": 0,
    }
    context = average.applied(model) if weights == "ema" else native._nothing()
    with context:
        for tier in tiers:
            size, rounds = tier["rows"], tier["rounds"]
            tier_path = destination / tier["name"]
            tier_path.mkdir()
            tier_result = {}
            for panel in panel_names():
                all_rows = panels[panel]
                indices = (
                    np.arange(len(all_rows))
                    if size is None
                    else nested_indices(panel, len(all_rows), size, 2026102105)
                )
                rows = all_rows[indices]
                tokens, pos, _ = pack_rows(rows, metadata)
                prediction = np.empty((rounds, len(rows)), dtype=np.int32)
                for begin in range(0, len(rows), config["evaluation_batch_size"]):
                    end = min(len(rows), begin + config["evaluation_batch_size"])
                    x = torch.as_tensor(tokens[begin:end], device=device)
                    p = torch.as_tensor(pos[begin:end], device=device)
                    with native.autocast(device):
                        for r, values in enumerate(stream_predictions(model, x, p, rounds)):
                            prediction[r, begin:end] = values.cpu().numpy()
                    write_json(
                        destination / "progress.json",
                        {
                            "tier": tier["name"],
                            "panel": panel,
                            "rows_done": end,
                            "rows": len(rows),
                            "rounds": rounds,
                            "utc": utc(),
                        },
                    )
                if verify_existing and size is None and rounds >= 8:
                    old = np.load(run / "grid" / weights / f"{panel}.npz")
                    if not np.array_equal(old["rows"][indices], rows):
                        raise AssertionError("Replayed query rows differ")
                    for r in (4, 8):
                        if not np.array_equal(prediction[r - 1], old["answer"][r - 1, indices]):
                            mismatches = int(
                                np.count_nonzero(prediction[r - 1] != old["answer"][r - 1, indices])
                            )
                            raise AssertionError(
                                f"Old prediction replay differs: {panel} R{r} {mismatches}"
                            )
                truth = rows[:, -1] + metadata["entity_offset"]
                np.savez_compressed(
                    tier_path / f"{panel}.npz",
                    rows=rows,
                    indices=indices,
                    answer=prediction,
                    truth=truth,
                )
                stat = summarize_predictions(
                    prediction,
                    truth,
                    entity_range=(metadata["entity_offset"], metadata["relation_offset"]),
                )
                write_json(tier_path / f"{panel}.json", stat)
                tier_result[panel] = {
                    "n": len(rows),
                    "rounds": rounds,
                    "r8_accuracy": stat["answer_accuracy"][7],
                    "final_accuracy": stat["answer_accuracy"][-1],
                    "r8_correct_now_wrong": stat["reference_correct_now_wrong"][-1],
                    "r8_wrong_now_correct": stat["reference_wrong_now_correct"][-1],
                    "r8_correct_ever_lost": stat["reference_correct_ever_lost_after_reference"][-1],
                    "r8_correct": stat["reference_correct"],
                }
                print(
                    json.dumps({"tier": tier["name"], "panel": panel, **tier_result[panel]}),
                    flush=True,
                )
            result["tiers"][tier["name"]] = tier_result
            result["wall_seconds"] = time.monotonic() - started
            write_json(destination / "summary.json", result)
            if verify_existing and size is None:
                write_json(
                    destination / "replay.json",
                    {
                        "passed": True,
                        "rounds": [4, 8],
                        "panels": list(panel_names()),
                        "queries": sum(len(v) for v in panels.values()),
                        "utc": utc(),
                    },
                )
    result.update(completed_utc=utc(), old_r4_r8_exact_replay=verify_existing)
    write_json(destination / "complete.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("smoke", "smoke-grid", "train", "grid", "depth", "cloud"))
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--base", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    job = plan["jobs"][args.job]
    config_path = ROOT / job["config"]
    config = json.loads(config_path.read_text())
    if args.stage == "cloud":
        import wandb

        if job["kind"] != "new":
            raise ValueError("Only this batch's new tracking records are checked")
        output = args.base / "cases" / args.job / "runs" / job["arm"]
        tracking = json.loads((output / "tracking.json").read_text())
        server = wandb.Api(timeout=60).run(
            f"{config['tracking']['entity']}/{config['tracking']['project']}/{tracking['id']}"
        )
        checks = {
            "finished": server.state == "finished",
            "step": server.summary.get("_step") == config["training"]["steps"],
            "configuration": server.config.get("experiment") == config,
        }
        write_json(
            output / "cloud-audit.json",
            {
                "passed": all(checks.values()),
                "checks": checks,
                "url": tracking["url"],
                "utc": utc(),
            },
        )
        if not all(checks.values()):
            raise RuntimeError(f"Cloud endpoint verification failed: {checks}")
        return
    name = job["arm"]
    if args.stage == "depth":
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        torch.cuda.set_device(0)
        torch.cuda.set_per_process_memory_fraction(config["runtime"]["gpu_memory_fraction"], 0)
        torch.backends.cuda.matmul.allow_tf32 = True
        run = (
            Path(job["run"])
            if job["kind"] == "reused"
            else args.base / "cases" / args.job / "runs" / name
        )
        tiers = plan["depth_tiers"] if job.get("deep", True) else plan["depth_tiers"][:1]
        depth(
            config,
            name,
            run,
            args.base / "depth" / args.job,
            tiers,
            job.get("weights", "ema"),
            verify_existing=True,
        )
        return
    if job["kind"] != "new":
        raise ValueError("Reused jobs may never train")
    # Reuse the exact existing runner; only the resolved configuration is new.
    native.BATCH = plan["batch"]
    sys.argv = [
        "run_loop_native_handoff.py",
        args.stage,
        "--config",
        str(config_path),
        "--base",
        str(args.base / "cases" / args.job),
        "--arm",
        name,
    ]
    if args.stage.startswith("smoke"):
        sys.argv += ["--limit-rows", "8", "--smoke-grid-rounds", "8"]
    native.main()


if __name__ == "__main__":
    os.umask(0o022)
    main()
