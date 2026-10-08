"""CUDA preflight and provenance checks before reusing completed local Loop runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from llm_memory_editability import sequential_transfer as st
from llm_memory_editability.loop_kv_learning import construct
from llm_memory_editability.loop_kv_learning_train import preflight
from llm_memory_editability.loop_learning_train import training_plan
from llm_memory_editability.realworld_composition import evaluate, pack
from llm_memory_editability.realworld_composition_data import write_json


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(config_path, index, out, device):
    config = json.loads(config_path.read_text())
    spec = config["specs"][index]
    reused = next(
        row for row in config["reused_runs"] if row["initialization"] == spec["initialization"]
    )
    old_out = Path(reused["run_dir"])
    repo = Path(config.get("repository", config_path.resolve().parents[1]))
    archived = repo / "docs/development-artifacts/loop-learning-development-v1/source"
    # Exact source equality makes the old Python constructor an archived implementation.
    core = (
        "loop_learning.py",
        "loop_learning_train.py",
        "sequential_transfer.py",
        "grokking_reproduction.py",
        "realworld_composition.py",
    )
    core_hashes = {}
    for name in core:
        relative = Path("src/llm_memory_editability") / name
        assert sha(repo / relative) == sha(archived / relative), name
        core_hashes[name] = sha(archived / relative)
    old_spec = json.loads((old_out / "run.json").read_text())["spec"]
    assert json.loads((old_out / "audit.json").read_text())["passed"] is True
    assert json.loads((old_out / "complete.json").read_text())["independently_reloaded"] is True
    data, tokenizer, old, device = st._setup(old_spec, device)
    records, plan = training_plan(data, spec)
    manifest = st.plan_manifest(records, plan, spec)
    old_manifest = json.loads((old_out / "sampling-plan.json").read_text())
    for key in ("plan_sha256", "multiset_sha256", "counts", "input_tokens", "supervised_tokens"):
        assert manifest[key] == old_manifest[key], key
    kv_spec = {
        **spec,
        "arm": "loop_shared_kv",
        "unique_layers": 2,
        "base_unique_blocks": 2,
        "repeats": 2,
    }
    new = construct(kv_spec, device)
    initial = json.loads((old_out / "architecture.json").read_text())
    from llm_memory_editability.loop_learning import execution_digest

    assert execution_digest(new) == initial["initial_execution_sha256"]
    checkpoint = torch.load(old_out / "latest.pt", map_location="cpu", weights_only=False)
    old.load_state_dict(checkpoint["model"])
    new.load_state_dict(checkpoint["model"])
    del checkpoint
    new.disable_shared = True
    old.eval(), new.eval()
    selected = [records[i] for i in plan[0][:4]]
    tokens, positions, _ = pack(selected, tokenizer.eos_token_id, device)
    with torch.no_grad():
        torch.testing.assert_close(old(tokens, positions), new(tokens, positions), rtol=0, atol=0)
    expected_metrics, expected_raw = evaluate(old, selected, tokenizer, device, batch_size=4)
    actual_metrics, actual_raw = evaluate(new, selected, tokenizer, device, batch_size=4)
    assert actual_raw == expected_raw and actual_metrics == expected_metrics
    del old, new
    torch.cuda.empty_cache()
    result = preflight(spec, out, device)
    checks = {
        "passed": True,
        "frozen_core_source_sha256": core_hashes,
        "reused_run": str(old_out),
        "sampling_plan_exact": True,
        "local_initial_execution_exact": True,
        "disabled_kv_forward_exact": True,
        "disabled_kv_generation_scores_exact": True,
        "old_weights_mutated": False,
        "reused_artifact_sha256": {
            name: sha(old_out / name)
            for name in (
                "run.json",
                "audit.json",
                "complete.json",
                "architecture.json",
                "sampling-plan.json",
                "latest.pt",
                "stage-a.pt",
                "learning.json",
                "predictions-0008000.json",
                "predictions-0012000.json",
            )
        },
        "new_arm_cuda_preflight": result["passed"],
    }
    write_json(out / "reuse-contract.json", checks)
    return {
        "name": spec["name"],
        "passed": True,
        "seconds_per_update": result["seconds_per_update_after_warmup"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    print(json.dumps(validate(args.config, args.index, args.out, "cuda:0")), flush=True)
