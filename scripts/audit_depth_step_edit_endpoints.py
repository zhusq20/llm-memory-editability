"""Reload parent weights plus saved MLP deltas and regenerate edited endpoints."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from llm_memory_editability.depth_step_mechanism import generate, load_run
from llm_memory_editability.grok_depth import write_json
from llm_memory_editability.latent_scaling import model_digest

ARTIFACTS = Path("docs/development-artifacts/depth-step-v1/edit-followup")
RESULTS = Path("results/depth-step-v1/edit-followup")


def audit(name, device):
    config = json.loads((ARTIFACTS / "config.json").read_text())
    state = next(item for item in config["states"] if item["name"] == name)
    parent, world, _ = load_run(
        state["run_dir"], Path(state["run_dir"]) / state["checkpoint"], device
    )
    records = json.loads((RESULTS / name / "summary.json").read_text())["records"]
    counts, maximum, exact_hashes = 0, 0.0, 0
    parent_hash = model_digest(parent)
    for record in records:
        path = RESULTS / name / f"case{record['original_case_index']:02d}-{record['arm']}-raw.npz"
        saved = np.load(path)
        model = copy.deepcopy(parent)
        delta = torch.as_tensor(saved["mlp_down_weight_delta"], device=device)
        with torch.no_grad():
            model.blocks[0].mlp.down.weight.add_(delta)
        exact_hashes += model_digest(model) == record["final_model_sha256"]
        for key in saved.files:
            if not key.startswith("step200_") or not key.endswith("_rows"):
                continue
            task = key[len("step200_") : -len("_rows")]
            _, actual = generate(model, saved[key], world, device)
            np.testing.assert_array_equal(
                actual["predictions"], saved[f"step200_{task}_predictions"]
            )
            np.testing.assert_allclose(
                actual["answer_probabilities"],
                saved[f"step200_{task}_answer_probabilities"],
                rtol=1e-5,
                atol=1e-5,
            )
            if actual["answer_probabilities"].size:
                maximum = max(
                    maximum,
                    float(
                        np.abs(
                            actual["answer_probabilities"]
                            - saved[f"step200_{task}_answer_probabilities"]
                        ).max()
                    ),
                )
            counts += 1
    if model_digest(parent) != parent_hash:
        raise AssertionError("Reload audit mutated the parent")
    result = {
        "state": name,
        "passed": True,
        "branches": len(records),
        "task_endpoints": counts,
        "all_generated_tokens_exact": True,
        "max_probability_difference": maximum,
        "exact_reconstructed_model_hashes": exact_hashes,
        "scope": "Parent checkpoint + float32 recorded MLP down delta; probability tolerance 1e-5",
    }
    write_json(ARTIFACTS / f"reload-{name}.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = True
    audit(args.state, args.device)
