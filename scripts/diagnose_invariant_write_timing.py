#!/usr/bin/env python3
"""Posthoc, label-free intervention on the identified transient overwrite."""

import argparse
import importlib.util
import json
import os
import time
import types
from pathlib import Path

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location(
        "dip", args.control / "diagnose_invariant_checkpoint_dip.py"
    )
    dip = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dip)
    protocol = json.loads((args.control / "protocol.json").read_text())
    amendment = json.loads((args.control / "timing-amendment.json").read_text())
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    config = json.loads(Path(protocol["config"]).read_text())
    world, panels, metadata = dip.load(config["input_root"])
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.cuda.set_per_process_memory_fraction(0.08)
    device = torch.device("cuda:0")
    model = dip.native.initialize(config, "invariant", metadata, device).eval()
    source = Path(protocol["checkpoint_root"]) / "checkpoint-0700000.pt"
    assert dip.digest(source) == protocol["checkpoint_sha256"]["700000"]
    saved = torch.load(source, map_location="cpu", weights_only=False)
    model.load_state_dict(saved["model"])
    with torch.no_grad():
        for key, parameter in model.named_parameters():
            parameter.copy_(saved["ema"][key])
            parameter.requires_grad_(False)
    state_hash = dip.native.model_digest(model)
    import wandb

    os.environ["WANDB_API_KEY"] = Path("/run/secrets/wandb").read_text().strip()
    run = wandb.init(
        entity="zsqzz",
        project="llm-memory-editability",
        job_type="checkpoint-diagnostic",
        name=protocol["batch"] + "-timing",
        dir=str(out),
        config=amendment,
    )
    dip.write(out / "tracking.json", {"id": run.id, "url": run.url})
    original = model.aggregate_round
    results = []
    before = time.monotonic()
    try:
        for condition in amendment["conditions"]:
            folder = out / condition
            folder.mkdir()
            for panel in dip.PANELS:
                rows = panels[panel]
                x, pos, _ = dip.pack_rows(rows, metadata)
                predictions = []
                for start in range(0, len(rows), protocol["batch_size"]):
                    begin = time.monotonic()
                    end = start + protocol["batch_size"]
                    tokens = torch.as_tensor(x[start:end], device=device)
                    positions = torch.as_tensor(pos[start:end], device=device)
                    hop_count = int(positions[0, 0])
                    counter = [0]

                    def patched(
                        self,
                        state,
                        injection,
                        counter=counter,
                        condition=condition,
                        hop_count=hop_count,
                    ):
                        counter[0] += 1
                        r = counter[0]
                        new, gamma = original(state, injection)
                        chosen = []
                        if condition == "skip_pos2_r3" and r == 3 and hop_count >= 2:
                            chosen = [2]
                        if condition == "skip_next_round_each_position" and 1 <= r - 1 <= hop_count:
                            chosen = [r - 1]
                        if condition == "skip_pos2_r2_control" and r == 2 and hop_count >= 2:
                            chosen = [2]
                        if chosen:
                            # Omit this candidate from N and D; keep the native decode
                            # history used by the next score. No entity labels are read.
                            h, numerator, denominator, probability = new
                            h, numerator, denominator = (
                                h.clone(),
                                numerator.clone(),
                                denominator.clone(),
                            )
                            numerator[:, chosen] = self.beta * state[1][:, chosen]
                            denominator[:, chosen] = self.beta * state[2][:, chosen]
                            h[:, chosen] = numerator[:, chosen] / denominator[:, chosen]
                            gamma = gamma.clone()
                            gamma[:, chosen] = 0
                            new = h, numerator, denominator, probability
                        return new, gamma

                    model.aggregate_round = types.MethodType(patched, model)
                    with torch.no_grad(), dip.native.autocast(device):
                        logits, _, _ = model.trace(tokens, positions, 8)
                    predictions.append(logits[:, :, 0].argmax(-1).cpu().numpy())
                    torch.cuda.synchronize()
                    time.sleep((time.monotonic() - begin) * protocol["yield_factor"])
                answers = np.concatenate(predictions, axis=1)
                np.savez_compressed(folder / f"{panel}.npz", rows=rows, answer=answers)
                metrics = {
                    "condition": condition,
                    "panel": panel,
                    "n": len(rows),
                    "answer_accuracy": (answers == rows[None, :, -1] + metadata["entity_offset"])
                    .mean(1)
                    .tolist(),
                }
                if condition == "native":
                    for r in [4, 8]:
                        old = (
                            Path(protocol["checkpoint_root"])
                            / "predictions"
                            / "0700000"
                            / f"ema-r{r}"
                            / f"{panel}.npz"
                        )
                        with np.load(old) as z:
                            assert np.array_equal(answers[r - 1], z["answer"])
                results.append(metrics)
                dip.write(out / "summary.json", results)
                run.log(
                    {
                        "evaluation_index": len(results),
                        "condition": condition,
                        "panel": panel,
                        "accuracy_r8": metrics["answer_accuracy"][7],
                    }
                )
                print(json.dumps({**metrics, "seconds": time.monotonic() - before}), flush=True)
        model.aggregate_round = original
        assert dip.native.model_digest(model) == state_hash
        assert dip.digest(source) == protocol["checkpoint_sha256"]["700000"]
        dip.write(
            out / "complete.json",
            {
                "passed": True,
                "new_training_steps": 0,
                "prediction_replay_exact": True,
                "model_and_checkpoint_unchanged": True,
                "panels": len(results),
                "seconds": time.monotonic() - before,
            },
        )
        run.finish()
    except BaseException:
        run.finish(exit_code=1)
        raise


if __name__ == "__main__":
    main()
