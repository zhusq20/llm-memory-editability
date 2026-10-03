"""Read existing entity states with the model's original tied readout; fit nothing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from llm_memory_editability.grok_depth import write_json
from llm_memory_editability.representation_alignment import new_model


@torch.no_grad()
def inspect(phase):
    torch.set_num_threads(1)
    root = Path("results/representation-alignment-v1/runs")
    records = []
    for path in sorted(root.glob(phase + "-w*")):
        if not (path / "audit.json").exists():
            continue
        payload = torch.load(path / "model.pt", map_location="cpu", weights_only=False)
        spec = payload["spec"]
        model = new_model(spec, "cpu").eval()
        model.load_state_dict(payload["model"])
        world = np.load(path / "world.npz")
        for task in ("familiar_test", "strict_test"):
            rows = world[task]
            # Only the first-hop prefix; neither following relation nor answer is input.
            prefixes = torch.as_tensor(
                np.column_stack(
                    (
                        np.full(len(rows), 2),
                        rows[:, 0],
                        np.full(len(rows), 3),
                        rows[:, 1],
                    )
                )
            )
            target = torch.as_tensor(rows[:, 2])
            chunks = []
            for start in range(0, len(rows), 256):
                _, state = model(prefixes[start : start + 256], return_bridge=True)
                chunks.append(state)
            state = torch.cat(chunks)
            logits = F.linear(model.ln_final(state), model.token.weight)
            correct = logits.argmax(-1) == target
            cosine = F.cosine_similarity(state, model.token(target))
            records.append(
                {
                    "run": path.name,
                    "world": spec["world"],
                    "initialization": spec["initialization"],
                    "arm": spec["arm"],
                    "task": task,
                    "n": len(rows),
                    "first_hop_original_readout_accuracy": float(correct.float().mean()),
                    "mean_cosine_to_input_embedding": float(cosine.mean()),
                    "cpu_fp32_diagnostic": True,
                }
            )
    target = Path("docs/development-artifacts/representation-alignment-v1") / phase
    target.mkdir(parents=True, exist_ok=True)
    write_json(target / "state-diagnostics.json", records)
    means = {}
    for arm in sorted({row["arm"] for row in records}):
        means[arm] = {}
        for task in ("familiar_test", "strict_test"):
            subset = [row for row in records if row["arm"] == arm and row["task"] == task]
            means[arm][task] = {
                key: float(np.mean([row[key] for row in subset]))
                for key in ("first_hop_original_readout_accuracy", "mean_cosine_to_input_embedding")
            }
    print(json.dumps({"diagnostic_records": len(records), "means": means}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True)
    inspect(parser.parse_args().phase)
