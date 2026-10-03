#!/usr/bin/env python3
"""CPU historical compatibility audit; never schedules or initializes a GPU.

Full-width initialization and checkpoint inference use the historical models.
Short AdamW replay uses a small fixture with identical modules and dropout.
CPU FP32 versus saved GPU BF16 NLL is an explicitly tolerant comparison;
old versus new CPU tensors, logits, updates and greedy tokens must be exact.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch
import transformers
from transformers import GPT2TokenizerFast

from llm_memory_editability import parametric_architecture as architecture
from llm_memory_editability import parametric_architecture_train as training
from llm_memory_editability import realworld_composition as historical
from llm_memory_editability.grok_depth import utc
from llm_memory_editability.grokking_reproduction import model_digest


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def equal_state(left, right):
    one, two = left.state_dict(), right.state_dict()
    assert one.keys() == two.keys(), "State keys changed"
    bad = [name for name in one if not torch.equal(one[name], two[name])]
    assert not bad, f"Unequal tensors: {bad[:5]}"
    return len(one)


def fixture_records():
    result = []
    for prefix, target in [
        ([1, 2, 3], [4, 5, 30]),
        ([7, 8], [9, 30]),
        ([11, 30, 12, 13], [14, 15, 16, 30]),
    ]:
        result.append(
            {"encoded": {"prefix": prefix, "target": target, "input": prefix + target[:-1]}}
        )
    return result


def ordinary_replay(config, spec, arm):
    tiny = {**config, "vocab_size": 31, "positions": 32, "hidden_size": 32, "attention_heads": 4}
    recipe = {**spec, "architecture": arm, "microbatch_size": 2}
    old = historical.construct(tiny, recipe, "cpu")
    new = architecture.construct(tiny, recipe, "cpu")
    equal_state(old, new)
    old_optimizer = historical.optimizer_for(old, 5e-5, 0.1)
    new_optimizer = architecture.optimizer_for(new, 5e-5, 0.1)
    rows = fixture_records()
    loss_history = []
    for step in range(3):
        # Nonzero update also exercises bias/norm decay groups and optimizer moments.
        torch.manual_seed(813101 + step)
        first = historical.update(old, old_optimizer, rows, recipe, "cpu", 30)
        torch.manual_seed(813101 + step)
        second = training.update(new, new_optimizer, rows, recipe, "cpu", 30)
        assert first["loss"] == second["answer_ce"]
        assert first["gradient_norm"] == second["gradient_norm"]
        equal_state(old, new)
        for old_p, new_p in zip(old.parameters(), new.parameters(), strict=True):
            a, b = old_optimizer.state[old_p], new_optimizer.state[new_p]
            assert a.keys() == b.keys()
            for name in a:
                assert torch.equal(a[name], b[name])
        loss_history.append(first["loss"])
    return {
        "architecture": arm,
        "width": 32,
        "steps": 3,
        "dropout": tiny["dropout"],
        "microbatch_size": 2,
        "effective_batch_size": 3,
        "exact_model_and_optimizer_equality": True,
        "losses": loss_history,
    }


def audit(repo, output, nll_atol):
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    began = time.monotonic()
    source_paths = [
        repo / "src/llm_memory_editability" / name
        for name in (
            "parametric_architecture.py",
            "parametric_architecture_train.py",
            "realworld_composition.py",
            "grokking_reproduction.py",
            "realworld_composition_data.py",
        )
    ]
    source_paths.append(Path(__file__).resolve())
    sources = {str(p): digest(p) for p in source_paths}
    config_path = repo / "configs/realworld-composition-development-v1.json"
    config = read(config_path)
    data = read(config["data_file"])
    assert digest(config["data_file"]) == config["data_sha256"]
    assert all(digest(p) == h for p, h in config["tokenizer_files"].items())
    report = {
        "passed": False,
        "started_utc": utc(),
        "device": "cpu",
        "environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "transformers": transformers.__version__,
            "cpu_threads": 4,
        },
        "source_files": sources,
        "historical_config_sha256": digest(config_path),
        "data_sha256": config["data_sha256"],
        "tokenizer_hashes_match": True,
        "initialization": [],
        "replay": [],
        "panels": {
            k: {
                "n": len(v),
                "ids_sha256": hashlib.sha256(
                    json.dumps(v, separators=(",", ":")).encode()
                ).hexdigest(),
            }
            for k, v in data["panels"].items()
        },
        "boundaries": [
            "CPU audit does not verify fused CUDA AdamW or BF16 CUDA kernels.",
            "Short update equality uses width32; actual-width checks cover initialization, "
            "forward and retained D8 checkpoint inference.",
            "Four fixed panel examples are a compatibility sample, "
            "not a repeated full endpoint evaluation.",
        ],
    }
    write(output, report)
    arm_names = {
        "development-standard4": "D4",
        "development-standard8": "D8",
        "development-loop4x2": "L4R2",
    }
    for spec in config["runs"]:
        arm = arm_names[spec["name"]]
        old = historical.construct(config["model"], spec, "cpu")
        new = architecture.construct(config["model"], {**spec, "architecture": arm}, "cpu")
        tensor_count = equal_state(old, new)
        run_root = Path(config["results_root"]) / "development" / spec["name"]
        saved = read(run_root / "run.json")
        initial_digest = model_digest(new)
        assert initial_digest == saved["initial_model_sha256"]
        assert historical.tensor_digests(new) == saved["initial_tensor_sha256"]
        old.eval()
        new.eval()
        tokens = torch.tensor([[15496, 11, 995, 50256, 123], [100, 200, 300, 400, 500]])
        positions = torch.tensor([[2, 4], [1, 3]])
        with torch.no_grad():
            left, right = old(tokens, positions), new(tokens, positions)
        assert torch.equal(left, right), f"{arm} actual-width forward differs"
        report["initialization"].append(
            {
                "architecture": arm,
                "parameters": sum(p.numel() for p in new.parameters()),
                "state_tensors": tensor_count,
                "model_sha256": initial_digest,
                "historical_manifest_matches": True,
                "exact_forward_equality": True,
                "run_manifest_sha256": digest(run_root / "run.json"),
            }
        )
        del old, new, left, right
        gc.collect()
        report["replay"].append(ordinary_replay(config["model"], spec, arm))
        write(output, report)
        print(json.dumps({"architecture": arm, "initialization_and_replay": "passed"}), flush=True)

    spec = next(s for s in config["runs"] if s["name"] == "development-standard8")
    run_root = Path(config["results_root"]) / "development" / spec["name"]
    checkpoint_path = run_root / "checkpoint-0032000.pt"
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert state["step"] == 32000
    old = historical.construct(config["model"], spec, "cpu")
    new = architecture.construct(config["model"], {**spec, "architecture": "D8"}, "cpu")
    old.load_state_dict(state["model"])
    new.load_state_dict(state["model"])
    del state
    assert model_digest(new) == read(run_root / "complete.json")["model_sha256"]
    tokenizer = GPT2TokenizerFast.from_pretrained(config["tokenizer"], local_files_only=True)
    frozen = read(run_root / "predictions-0032000.json")
    lookup = {x["id"]: x for key in ("atoms", "train_compositions") for x in data[key]}
    # First two members of each already-frozen panel; no score-based sample choice.
    selected = [
        (key, i) for key in ("atomic", "train_composition") for i in data["panels"][key][:2]
    ]
    rows = [lookup[i] for _, i in selected]
    old_metrics, old_predictions = historical.evaluate(old, rows, tokenizer, "cpu", batch_size=4)
    new_metrics, new_predictions = historical.evaluate(new, rows, tokenizer, "cpu", batch_size=4)
    assert old_metrics == new_metrics and old_predictions == new_predictions
    comparisons = []
    for (group, row_id), pred in zip(selected, new_predictions, strict=True):
        baseline = next(p for p in frozen[group] if p["id"] == row_id)
        gap = abs(pred["nll"] - baseline["nll"])
        assert pred["generated_tokens"] == baseline["generated_tokens"]
        assert pred["eos"] == baseline["eos"]
        assert pred["truncated"] == baseline["truncated"]
        assert gap <= nll_atol, f"Cross-device NLL deviation {gap} exceeds {nll_atol}"
        comparisons.append(
            {
                "group": group,
                "id": row_id,
                "generated_tokens": pred["generated_tokens"],
                "exact_saved_greedy_match": True,
                "cpu_fp32_nll": pred["nll"],
                "saved_gpu_bf16_nll": baseline["nll"],
                "absolute_nll_gap": gap,
            }
        )
    report["D8_development_32000_reload"] = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": digest(checkpoint_path),
        "historical_predictions_sha256": digest(run_root / "predictions-0032000.json"),
        "sample_rule": "first two stored IDs in each original atomic/train_composition panel",
        "exact_old_new_cpu_prediction_and_nll_equality": True,
        "cpu_fp32_vs_saved_gpu_bf16_nll_absolute_tolerance": nll_atol,
        "comparisons": comparisons,
    }
    assert sources == {str(p): digest(p) for p in source_paths}, "Source changed during audit"
    report.update(passed=True, completed_utc=utc(), elapsed_seconds=time.monotonic() - began)
    write(output, report)
    print(
        json.dumps(
            {"passed": True, "report": str(output), "elapsed_seconds": report["elapsed_seconds"]}
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cross-device-nll-atol", type=float, default=0.005)
    args = parser.parse_args()
    output = (
        args.output
        or args.repo
        / "docs/development-artifacts/parametric-architecture-execution-v1"
        / "historical-compatibility.json"
    )
    audit(args.repo.resolve(), output.resolve(), args.cross_device_nll_atol)
