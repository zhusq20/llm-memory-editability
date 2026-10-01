#!/usr/bin/env python3
"""Independently replay saved SSFR artifacts on CPU without optimization."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def audit_world(directory, *, atol=1e-5, rtol=1e-5):
    from llm_memory_editability.hebbian_interface import load_checkpoint, tensor_hash

    directory = Path(directory)
    summary = json.loads((directory / "summary.json").read_text())
    if summary["state"] != "complete":
        raise ValueError("A complete world is required")
    spec, seed = summary["spec"], summary["seed"]
    sys.path.insert(0, str(ROOT / spec["source_path"] / "src"))
    from hebbian.data.synthetics.factsets import BijectiveMapping
    from hebbian.transformer.data import AssociativeRecallBatchGenerator
    from hebbian.transformer.model import GPT, GPTConfig
    from hebbian.transformer.utils import copy_embeddings_to_gpt, insert_mlp_into_gpt

    torch.set_num_threads(1)
    started = time.perf_counter()
    world = dict(np.load(directory / "world.npz"))
    inputs, keys = torch.from_numpy(world["inputs"]), torch.from_numpy(world["key"])
    embeddings = torch.nn.Embedding.from_pretrained(torch.from_numpy(world["embeddings"]))
    mappings = {name: torch.from_numpy(world["mapping_" + name]) for name in ("A", "B")}
    changed = mappings["A"].ne(mappings["B"])
    checks, differences, failures, near_zero_angles = {}, {}, [], []

    def check(name, condition):
        checks[name] = bool(condition)
        if not condition:
            failures.append(name)

    def close(name, actual, expected):
        actual, expected = np.asarray(actual), np.asarray(expected)
        if "angle" in name and expected.size == 1 and abs(float(expected)) <= 1e-3:
            near_zero_angles.append(name)
        matches = actual.shape == expected.shape and np.allclose(
            actual, expected, atol=atol, rtol=rtol, equal_nan=False
        )
        differences[name] = (
            float(np.max(np.abs(actual - expected)))
            if actual.shape == expected.shape and actual.size
            else None
        )
        check(name, matches)

    def parameters(model, non_mlp=False, frozen=False):
        return {
            key: value
            for key, value in model.named_parameters()
            if (not non_mlp or ".mlp." not in key) and (not frozen or not value.requires_grad)
        }

    def aggregates(name, scores, target, changed_rows, recorded):
        prediction = scores.argmax(-1)
        correct = prediction == target
        close(name + "/accuracy_all", correct.float().mean(), recorded["accuracy_all"])
        if changed_rows.any():
            close(
                name + "/accuracy_changed",
                correct[changed_rows].float().mean(),
                recorded["accuracy_changed"],
            )
        else:
            check(name + "/empty_changed_subset", recorded["accuracy_changed"] is None)
        check(name + "/n_all", recorded["n_all"] == len(target))
        check(name + "/n_changed", recorded["n_changed"] == int(changed_rows.sum()))
        return prediction

    check(
        "balanced_keys",
        torch.equal(keys.bincount(), torch.full((spec["num_facts"],), spec["eval_repeats"])),
    )
    check("query_last_no_future_answer", inputs.shape[1] == 2 * spec["junk_len"] + 2)
    check("key_position", torch.equal(inputs[:, spec["junk_len"]], keys))
    check("query_token", bool(inputs[:, -1].eq(spec["num_facts"] + spec["junk_vocab_size"]).all()))
    junk = torch.cat((inputs[:, : spec["junk_len"]], inputs[:, spec["junk_len"] + 1 : -1]), 1)
    check(
        "junk_range",
        bool(
            (
                (junk >= spec["num_facts"]) & (junk < spec["num_facts"] + spec["junk_vocab_size"])
            ).all()
        ),
    )
    check("changed_mapping", np.array_equal(world["changed_mapping"], changed.numpy()))
    check(
        "test_inputs_hash", tensor_hash({"inputs": inputs}) == summary["world"]["test_inputs_hash"]
    )
    memories, model_payloads = {}, {}
    for loss in ("ce", "mse"):
        memories[loss] = {}
        for mapping in ("A", "B"):
            record = summary["memories"][loss][mapping]
            memory, _ = load_checkpoint(directory / record["checkpoint"])
            memories[loss][mapping] = memory
            name = f"memory/{loss}/{mapping}"
            check(name + "/final_hash", tensor_hash(memory.state_dict()) == record["final_hash"])
            arrays = dict(np.load(directory / record["standalone"]["artifact"]))
            with torch.inference_mode():
                outputs = memory(embeddings.weight[: spec["num_facts"]])
                full_scores = F.normalize(outputs, dim=-1) @ embeddings.weight.T
                fact_scores = full_scores[:, : spec["num_facts"]]
                target = mappings[mapping]
                expected = embeddings.weight[target]
                relative = (outputs - expected).norm(dim=-1) / expected.norm(dim=-1).clamp_min(
                    1e-12
                )
                angle = F.cosine_similarity(outputs, expected).clamp(-1, 1).acos()
            close(name + "/scores_full", full_scores.numpy(), arrays["scores"])
            close(name + "/scores_fact", fact_scores.numpy(), arrays["scores_fact"])
            close(
                name + "/scores_fact_raw",
                (outputs @ embeddings.weight[: spec["num_facts"]].T).numpy(),
                arrays["scores_fact_raw"],
            )
            check(name + "/predictions", np.array_equal(full_scores.argmax(-1), arrays["pred"]))
            check(
                name + "/fact_predictions",
                np.array_equal(fact_scores.argmax(-1), arrays["pred_fact"]),
            )
            check(name + "/targets", np.array_equal(target, arrays["target"]))
            check(name + "/changed", np.array_equal(changed, arrays["changed_mapping"]))
            aggregates(name, full_scores, target, changed, record["standalone"]["full_vocab"])
            close(
                name + "/relative_mean",
                relative.mean(),
                record["standalone"]["relative_error_mean"],
            )
            close(name + "/angle_mean", angle.mean(), record["standalone"]["angle_radians_mean"])
            curve = dict(np.load(directory / f"memory-{loss}-{mapping}-curve.npz"))
            check(name + "/fixed_epochs", len(curve["train_loss"]) == spec["mlp_epochs"])
        model_payloads[loss] = load_checkpoint(directory / summary["readers"][loss]["checkpoint"])
    hashes = {
        memory["initial_hash"] for loss in summary["memories"].values() for memory in loss.values()
    }
    check("paired_recorded_memory_initial_hashes", len(hashes) == 1)

    for loss, (model, payload) in model_payloads.items():
        reader_record = summary["readers"][loss]
        prefix = f"reader/{loss}"
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            initial_reader = GPT(GPTConfig(**payload["gpt_config"])).eval()
            copy_embeddings_to_gpt(initial_reader, embeddings)
            insert_mlp_into_gpt(initial_reader, memories[loss]["A"], embeddings)
        for suffix, values, recorded in (
            (
                "initial_non_mlp_hash",
                parameters(initial_reader, non_mlp=True),
                "initial_non_mlp_hash",
            ),
            ("final_non_mlp_hash", parameters(model, non_mlp=True), "final_non_mlp_hash"),
            ("initial_frozen_hash", parameters(initial_reader, frozen=True), "frozen_hash_before"),
            ("final_frozen_hash", parameters(model, frozen=True), "frozen_hash_after"),
        ):
            check(prefix + "/" + suffix, tensor_hash(values) == reader_record[recorded])
        check(
            prefix + "/memory_A_unchanged",
            tensor_hash(model.transformer.h[0].mlp.state_dict())
            == summary["memories"][loss]["A"]["final_hash"],
        )
        conditions = {
            "A": (memories[loss]["A"], "A"),
            "wrong_A_on_B": (memories[loss]["A"], "B"),
            "B_ce": (memories["ce"]["B"], "B"),
            "B_mse": (memories["mse"]["B"], "B"),
        }
        for condition, (memory, mapping) in conditions.items():
            name = prefix + "/" + condition
            record = reader_record["endpoints"][condition]
            arrays = dict(np.load(directory / record["artifact"]))
            insert_mlp_into_gpt(model, memory, embeddings)
            scores, query_inputs = [], []
            hook = memory.register_forward_pre_hook(
                lambda module, args, captured=query_inputs: captured.append(args[0][:, -1].detach())
            )
            try:
                with torch.inference_mode():
                    for batch in inputs.split(spec["batch_size"]):
                        logits, _ = model(batch)
                        scores.append(logits[:, 0])
            finally:
                hook.remove()
            scores, query_inputs = torch.cat(scores), torch.cat(query_inputs)
            targets, changed_rows = mappings[mapping][keys], changed[keys]
            close(name + "/scores", scores.numpy(), arrays["scores"])
            check(name + "/targets", np.array_equal(targets, arrays["target"]))
            check(name + "/keys", np.array_equal(keys, arrays["key"]))
            check(name + "/changed", np.array_equal(changed_rows, arrays["changed_mapping"]))
            predictions = aggregates(name, scores, targets, changed_rows, record)
            check(name + "/predictions", np.array_equal(predictions, arrays["pred"]))
            close(name + "/query_inputs", query_inputs.numpy(), arrays["query_input"])
            expected = embeddings.weight[keys]
            relative = (query_inputs - expected).norm(dim=-1) / expected.norm(dim=-1).clamp_min(
                1e-12
            )
            angle = F.cosine_similarity(query_inputs, expected).clamp(-1, 1).acos()
            close(
                name + "/query_relative_mean", relative.mean(), record["query_relative_error_mean"]
            )
            close(name + "/query_angle_mean", angle.mean(), record["query_angle_radians_mean"])
        insert_mlp_into_gpt(model, memories[loss]["A"], embeddings)
        with torch.inference_mode():
            sample = inputs[:8]
            answer = mappings["A"][keys[:8], None]
            original = model(sample)[0][:, 0]
            full = torch.cat((sample, answer), 1)
            altered = torch.cat((sample, (answer + 1) % spec["num_facts"]), 1)
            full_scores = model.lm_head(model.forward_hidden(full)[:, -2])
            altered_scores = model.lm_head(model.forward_hidden(altered)[:, -2])
        close(prefix + "/causal_append_answer", full_scores.numpy(), original.numpy())
        close(prefix + "/causal_change_future_answer", altered_scores.numpy(), original.numpy())

    mapping = BijectiveMapping(list(range(spec["num_facts"])), mappings["A"].tolist())
    stream_hash = hashlib.sha256()
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed + 10000)
        generator = AssociativeRecallBatchGenerator(
            mapping,
            spec["num_facts"],
            spec["junk_vocab_size"],
            spec["junk_len"],
            spec["junk_len"],
            spec["batch_size"],
            spec["reader_steps"],
            device=torch.device("cpu"),
        )
        for batch, labels in generator:
            stream_hash.update(batch[:, :-1].contiguous().numpy().tobytes())
            stream_hash.update(labels[:, -2].contiguous().numpy().tobytes())
    for loss in ("ce", "mse"):
        check(
            f"reader/{loss}/replayed_stream_hash",
            stream_hash.hexdigest() == summary["readers"][loss]["training_stream_hash"],
        )
        check(
            f"reader/{loss}/final_step",
            summary["readers"][loss]["curve"][-1]["step"] == spec["reader_steps"],
        )
    files = (
        sorted(directory.glob("*.pt"))
        + sorted(directory.glob("*.npz"))
        + [directory / "summary.json"]
    )
    return {
        "passed": not failures,
        "seed": seed,
        "device": "cpu",
        "threads": 1,
        "atol": atol,
        "rtol": rtol,
        "seconds": time.perf_counter() - started,
        "checks": checks,
        "failures": failures,
        "max_absolute_differences": differences,
        "classification": {
            "strict_passed": not failures,
            "near_zero_angle_representation_failures": [
                name for name in failures if name in near_zero_angles
            ],
            "other_failures": [name for name in failures if name not in near_zero_angles],
            "prediction_score_target_aggregate_replay_passed": all(
                result
                for name, result in checks.items()
                if any(term in name for term in ("scores", "predictions", "targets", "accuracy"))
            ),
            "note": (
                "Classification does not override passed or remove any strict failure. "
                "Near-zero means recorded absolute angle <= 1e-3 radians."
            ),
        },
        "initial_memory_evidence": (
            "Four recorded initialization hashes agree; original CUDA initialization "
            "samples are not reconstructed in this CPU-only audit."
        ),
        "scope": (
            "Checkpoint replay, reader initialization, freezing, stream and causality. "
            "No training, tuning, selection or new evaluation contexts."
        ),
        "input_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in files
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("world", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-5)
    args = parser.parse_args()
    report = audit_world(args.world, atol=args.atol, rtol=args.rtol)
    output = args.output or args.world / "independent-audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "checks": len(report["checks"]),
                "failures": report["failures"],
                "output": str(output),
            }
        ),
        flush=True,
    )
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
