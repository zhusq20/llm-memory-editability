"""Restore role-balanced branches in a fresh interpreter and verify raw generation."""

import json
from pathlib import Path

import numpy as np
import torch

from llm_memory_editability.grok_depth import write_json
from llm_memory_editability.grokking_dynamics_mechanism import PARAMETER, safe_generate
from llm_memory_editability.latent_scaling import construct, model_digest
from llm_memory_editability.storage_composition import file_hash


def main():
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.set_device(0)
    device = "cuda:0"
    records = []
    for phase in ["calibration", "development"]:
        path = Path(
            f"docs/development-artifacts/grokking-dynamics-v1/balanced-editor/{phase}/config.json"
        )
        if not path.exists():
            continue
        config = json.loads(path.read_text())
        for path, sha in config["source"].items():
            assert file_hash(path) == sha
        branches, tasks = 0, 0
        for path in sorted(
            Path(f"results/grokking-dynamics-v1/balanced-editor/{phase}").glob("*/case*.json")
        ):
            record = json.loads(path.read_text())
            weight = torch.load(path.with_suffix(".pt"), map_location=device, weights_only=False)
            parent = weight["parent"]
            assert file_hash(parent["checkpoint"]) == parent["sha256"]
            checkpoint = torch.load(parent["checkpoint"], map_location=device, weights_only=False)
            model = construct(checkpoint["spec"], device).eval()
            model.load_state_dict(checkpoint["model"])
            assert model_digest(model) == record["parent_model_sha256"]
            with torch.no_grad():
                dict(model.named_parameters())[PARAMETER].copy_(weight["weight"])
            assert model_digest(model) == record["final_model_sha256"]
            raw = np.load(path.with_suffix(".npz"))
            for task, row_list in record["case"]["tasks"].items():
                rows = np.asarray(row_list, dtype=np.int64)
                _metric, prediction = safe_generate(model, rows, device)
                for key, value in prediction.items():
                    saved = raw[f"step200_{task}_{key}"]
                    if key == "answer_nll":
                        np.testing.assert_allclose(value, saved, rtol=1e-5, atol=1e-5)
                    else:
                        np.testing.assert_array_equal(value, saved)
                tasks += 1
            branches += 1
        expected = (
            len(config["specs"])
            * len(config["nodes"])
            * len(config["cases"])
            * len(config["learning_rates"])
            * 2
        )
        assert branches == expected
        records.append({"phase": phase, "branches": branches, "tasks": tasks, "passed": True})
    write_json(
        Path("docs/development-artifacts/grokking-dynamics-v1/balanced-editor/audit.json"),
        {"passed": True, "records": records, "source_sha256": file_hash(__file__)},
    )
    print(json.dumps(records))


if __name__ == "__main__":
    main()
