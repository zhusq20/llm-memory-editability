"""Independently replay all retained weights and rescore every saved generation."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from run_grokking_dynamics import ARTIFACTS, RESULTS, config_path, run_name

from llm_memory_editability.grok_depth import EpochStream, write_json
from llm_memory_editability.latent_scaling import (
    build_world,
    compare_metrics,
    compare_predictions,
    construct,
    model_digest,
)
from llm_memory_editability.storage_composition import data_digest, evaluate, file_hash


def audit_run(config, spec, device):
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.set_device(torch.device(device))
    out = RESULTS / config["phase"] / run_name(spec)
    result = json.loads((out / "complete.json").read_text())
    assert result["spec"] == spec
    assert file_hash(out / "latest.pt") == result["checkpoint_sha256"]
    world = build_world(spec)
    assert data_digest(world) == config["data_sha256"] == result["data_sha256"]
    model = construct(spec, device)
    key = f"i{spec['initialization']}-l{spec['layers']}"
    assert model_digest(model) == config["initial_model_sha256"][key]
    assert model_digest(model) == result["initial_model_sha256"]
    history = json.loads((out / "learning.json").read_text())
    assert [row["step"] for row in history] == spec["nodes"]
    tasks = ["common_atomic", "anchor_atomic", "train_composite", "familiar_test", "strict_test"]
    for row in history:
        raw = np.load(out / f"predictions-{row['step']:06d}.npz")
        for task in tasks:
            generated = raw[task + "_generated"]
            answer_ok = generated[:, 0] == world[task][:, -1]
            full_ok = answer_ok & (generated[:, 1] == 5) & (generated[:, 2] == 1)
            np.testing.assert_array_equal(full_ok, raw[task + "_correct"])
            metric = row["metrics"][task]
            assert metric["n"] == len(world[task])
            assert metric["accuracy"] == float(full_ok.mean())
            assert metric["answer_accuracy"] == float(answer_ok.mean())
            assert np.isfinite(raw[task + "_answer_nll"]).all()
            assert abs(float(raw[task + "_answer_nll"].mean()) - metric["answer_nll"]) < 1e-5
        for task in ["familiar_test", "strict_test"]:
            coverage = raw[task + "_coverage"]
            assert float(coverage.mean()) == row["metrics"][task]["atomic_correct_coverage"]
    maximum = 0.0
    checkpoints = []
    for step in spec["checkpoint_nodes"]:
        path = out / f"model-{step:06d}.pt"
        state = torch.load(path, map_location=device, weights_only=False)
        assert state["spec"] == spec and state["step"] == step
        model.load_state_dict(state["model"])
        metrics, predictions = evaluate(model, world, "low", device)
        expected = next(row for row in history if row["step"] == step)
        compare_metrics(metrics, expected["metrics"])
        maximum = max(
            maximum, compare_predictions(predictions, np.load(out / f"predictions-{step:06d}.npz"))
        )
        checkpoints.append({"step": step, "sha256": file_hash(path)})
    exposures = np.load(out / "exposures.npz")
    for i, task in enumerate(["common_atomic", "train_composite", "anchor_atomic"]):
        stream = EpochStream(len(world[task]), spec["stream_seed"] + i)
        counts = np.zeros(len(world[task]), dtype=np.int64)
        total = spec["steps"] * spec["batch_size"] // 3
        for start in range(0, total, 1000000):
            counts += np.bincount(stream.take(min(1000000, total - start)), minlength=len(counts))
        np.testing.assert_array_equal(counts, exposures[f"stratum{i}"])
    record = {
        "passed": True,
        "run": run_name(spec),
        "retained_checkpoints": len(checkpoints),
        "rescored_learning_nodes": len(history),
        "checkpoints": checkpoints,
        "max_nll_difference": maximum,
        "exposures_reconstructed": True,
    }
    write_json(out / "independent-audit.json", record)
    print(json.dumps({k: v for k, v in record.items() if k != "checkpoints"}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    args = parser.parse_args()
    config = json.loads(config_path("development").read_text())
    if args.name:
        spec = next(s for s in config["specs"] if run_name(s) == args.name)
        audit_run(config, spec, args.device)
        return
    import queue

    slots = queue.Queue()
    for gpu in map(int, args.gpus.split(",")):
        slots.put(gpu)

    def launch(spec):
        gpu = slots.get()
        folder = RESULTS / "development" / run_name(spec)
        try:
            with (folder / "independent-audit-process.log").open("a") as log:
                process = subprocess.run(
                    [sys.executable, __file__, "--name", run_name(spec), "--device", f"cuda:{gpu}"],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"),
                    check=False,
                )
            if process.returncode:
                raise RuntimeError(f"Independent audit failed: {folder}")
            return json.loads((folder / "independent-audit.json").read_text())
        finally:
            slots.put(gpu)

    with ThreadPoolExecutor(max_workers=slots.qsize()) as pool:
        audits = list(pool.map(launch, config["specs"]))
    write_json(
        ARTIFACTS / "development/independent-audit.json",
        {
            "passed": True,
            "runs": len(audits),
            "source_sha256": file_hash(__file__),
            "retained_checkpoint_replays": sum(r["retained_checkpoints"] for r in audits),
            "rescored_learning_nodes": sum(r["rescored_learning_nodes"] for r in audits),
            "max_nll_difference": max(r["max_nll_difference"] for r in audits),
            "audits": audits,
        },
    )
    print(json.dumps({"passed": True, "runs": len(audits)}))


if __name__ == "__main__":
    main()
