"""Deterministically replay edits to archive and reload exact final projections.

Float32 differences are not a lossless weight serialization. This verification
repeats the frozen optimizer operation and checks the originally recorded full
model digest before saving the exact projection; it changes no experiment.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from llm_memory_editability.depth_step_edit_calibration import (
    calibrated_edit_one,
    calibration_cases,
)
from llm_memory_editability.depth_step_mechanism import generate, load_run
from llm_memory_editability.grok_depth import write_json
from llm_memory_editability.latent_scaling import model_digest

ARTIFACTS = Path("docs/development-artifacts/depth-step-v1/edit-followup")
RESULTS = Path("results/depth-step-v1/edit-followup")
OriginalAdam = torch.optim.Adam


class CapturedAdam(OriginalAdam):
    final_weight = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.audit_steps = 0

    def step(self, *args, **kwargs):
        result = super().step(*args, **kwargs)
        self.audit_steps += 1
        if self.audit_steps == 200:
            CapturedAdam.final_weight = self.param_groups[0]["params"][0].detach().cpu().clone()
        return result


def audit(name, device):
    started = time.perf_counter()
    config = json.loads((ARTIFACTS / "config.json").read_text())
    state = next(item for item in config["states"] if item["name"] == name)
    parent, world, identity = load_run(
        state["run_dir"], Path(state["run_dir"]) / state["checkpoint"], device
    )
    cases, _ = calibration_cases(world, n_cases=8)
    records = json.loads((RESULTS / name / "summary.json").read_text())["records"]
    maximum = 0.0
    tasks_checked = 0
    torch.optim.Adam = CapturedAdam
    try:
        for record in records:
            index, arm = record["original_case_index"], record["arm"]
            stem = f"case{index:02d}-{arm}"
            saved = np.load(RESULTS / name / f"{stem}-raw.npz")
            replay_record, replay_raw = calibrated_edit_one(
                parent, cases[index], world, device, arm, config["lr"]
            )
            if replay_record["final_model_sha256"] != record["final_model_sha256"]:
                raise AssertionError("Replay did not reproduce the original full-model hash")
            weight = CapturedAdam.final_weight
            checkpoint = {
                "parameter": "blocks.0.mlp.down.weight",
                "weight": weight,
                "parent_checkpoint_sha256": identity["checkpoint_sha256"],
                "final_model_sha256": record["final_model_sha256"],
            }
            target = RESULTS / name / f"{stem}-exact-down.pt"
            if target.exists():
                raise FileExistsError(target)
            torch.save(checkpoint, target)
            restored = torch.load(target, map_location=device, weights_only=False)
            model = copy.deepcopy(parent)
            with torch.no_grad():
                model.blocks[0].mlp.down.weight.copy_(restored["weight"])
            if model_digest(model) != record["final_model_sha256"]:
                raise AssertionError("Exact saved weight reload changed the final model")
            for key in saved.files:
                if key.startswith("step") and key.endswith("_predictions"):
                    np.testing.assert_array_equal(replay_raw[key], saved[key])
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
                tasks_checked += 1
    finally:
        torch.optim.Adam = OriginalAdam
    result = {
        "state": name,
        "passed": True,
        "branches": len(records),
        "verification_replay_updates": 200 * len(records),
        "full_final_model_hashes_exact": True,
        "all_node_generated_tokens_exact": True,
        "reloaded_endpoint_tasks": tasks_checked,
        "max_probability_difference": maximum,
        "wall_seconds": time.perf_counter() - started,
    }
    write_json(ARTIFACTS / f"exact-reload-{name}.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = True
    audit(args.state, args.device)
