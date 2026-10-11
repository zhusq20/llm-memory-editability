"""Zero-update paired phrasing readout of frozen sequential-transfer weights.

This is a development readout. The alternative wording changes query form as
well as token order; scores do not by themselves locate or erase stored facts.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from transformers import GPT2TokenizerFast

from .grok_depth import utc
from .realworld_composition import aggregate, construct, evaluate, model_digest
from .realworld_composition_data import (
    QUESTION_TEMPLATES,
    encode_example,
    sha256,
    write_json,
)
from .sequential_transfer import validate_data

SNAPSHOTS = ("stage_a", "final")
EXACT_FIELDS = (
    "prediction",
    "generated_tokens",
    "canonical_em",
    "alias_em",
    "unambiguous_alias_em",
    "f1",
    "eos",
    "truncated",
    "context_limit_reached",
    "generation_limit_reached",
    "gold",
)


def original_subject(row):
    """Invert the author's rendered question without inferring any fact label."""
    relation = row["edge"][1]
    template = QUESTION_TEMPLATES[relation]
    before, after = template.split("{subject}")
    question = row["question"]
    if not question.startswith(before) or not question.endswith(after):
        raise ValueError(f"Canonical question no longer matches its relation: {row['id']}")
    subject = question[len(before) : len(question) - len(after) if after else None]
    if not subject or template.format(subject=subject) != question:
        raise ValueError(f"Cannot recover the exact original subject: {row['id']}")
    return subject


def phrasing_records(data, tokenizer, templates, limit=None):
    """Preserve all fact IDs, truth, aliases and subsets; change only questions."""
    validate_data(data)
    relations = {row["edge"][1] for row in data["atoms"]}
    if set(templates) != relations:
        raise ValueError("Alternative template coverage must equal all observed relations")
    if any(value.count("{subject}") != 1 for value in templates.values()):
        raise ValueError("Each alternative must contain exactly one subject placeholder")
    training_questions = {row["question"] for row in data["atoms"] + data["train_compositions"]}
    original, alternative = [], []
    for row in data["atoms"]:
        canonical = copy.deepcopy(row)
        encoded = encode_example(tokenizer, row["question"], row["answer"])
        if encoded != row["encoded"]:
            raise ValueError(f"Original question re-encoding differs: {row['id']}")
        changed = copy.deepcopy(row)
        changed["question"] = templates[row["edge"][1]].format(subject=original_subject(row))
        if changed["question"] in training_questions:
            raise ValueError(
                f"Alternative full question appeared in original training: {row['id']}"
            )
        changed["encoded"] = encode_example(tokenizer, changed["question"], changed["answer"])
        if changed["encoded"]["target"] != canonical["encoded"]["target"]:
            raise ValueError("The alternative query changed supervised answer/EOS tokens")
        original.append(canonical)
        alternative.append(changed)
    if limit is not None:
        if not 0 < limit <= len(original):
            raise ValueError("Smoke limit must select a positive fixed prefix of the facts")
        original, alternative = original[:limit], alternative[:limit]
    return {"canonical": original, "heldout": alternative}


def checked_file(path, expected_hash):
    if not expected_hash or sha256(path) != expected_hash:
        raise ValueError(f"Frozen source SHA256 differs: {path}")
    return Path(path)


def load_inputs(config_path, name, limit):
    config = json.loads(Path(config_path).read_text())
    matches = [spec for spec in config["specs"] if spec["name"] == name]
    if len(matches) != 1:
        raise ValueError("Exactly one frozen development run must match --run-name")
    spec = matches[0]
    run_path = checked_file(
        Path(spec["source_run_dir"]) / "run.json", spec["source_run_json_sha256"]
    )
    original_spec = json.loads(run_path.read_text())["spec"]
    if original_spec.get("phase") != "development" or original_spec["name"] != name:
        raise ValueError("Readout is restricted to the named historical development run")
    if config["canonical_minimum_token_match_fraction"] != 1.0:
        raise ValueError("Native canonical reproduction requires exact available outputs")
    source_hashes = {}
    package = Path(__file__).parent
    for item in config["compatible_source_files"]:
        checked_file(item["original_path"], item["sha256"])
        active = package / item["module"]
        checked_file(active, item["sha256"])
        source_hashes[item["module"]] = item["sha256"]
    data_path = checked_file(spec["data_file"], spec["data_sha256"])
    if original_spec["data_sha256"] != spec["data_sha256"]:
        raise ValueError("Readout data differs from the original run's frozen data")
    for item in config["tokenizer_files"]:
        checked_file(Path(spec["tokenizer"]) / item["name"], item["sha256"])
    tokenizer = GPT2TokenizerFast.from_pretrained(spec["tokenizer"], local_files_only=True)
    if len(tokenizer) != original_spec["model"]["vocab_size"]:
        raise ValueError("Tokenizer vocabulary differs from the original model")
    data = json.loads(data_path.read_text())
    pools = phrasing_records(data, tokenizer, config["templates"], limit)
    for snapshot in SNAPSHOTS:
        item = spec["checkpoints"][snapshot]
        checked_file(item["path"], item["sha256"])
        checked_file(item["predictions_file"], item["predictions_sha256"])
        expected_step = original_spec["stage_a_steps"] + (
            original_spec["stage_b_steps"] if snapshot == "final" else 0
        )
        if item["step"] != expected_step:
            raise ValueError("Frozen readout checkpoint step does not match the native endpoint")
    return config, spec, original_spec, data, tokenizer, pools, source_hashes


def load_reader(original_spec, checkpoint_spec, data, device):
    """Load only model parameters. Never construct, restore or call an optimizer."""
    model = construct(original_spec["model"], original_spec, device)
    checkpoint = torch.load(checkpoint_spec["path"], map_location="cpu", weights_only=False)
    if checkpoint["step"] != checkpoint_spec["step"]:
        raise ValueError("Actual checkpoint step differs from the frozen snapshot")
    model.load_state_dict(checkpoint["model"], strict=True)
    model.requires_grad_(False)
    model.eval()
    counts = np.asarray(checkpoint["counts"])
    if len(counts) != len(data["atoms"]) + len(data["train_compositions"]):
        raise ValueError("Historical exposure counters do not cover the original training pool")
    exposures = {row["id"]: int(counts[i]) for i, row in enumerate(data["atoms"])}
    history = {
        "checkpoint_step": int(checkpoint["step"]),
        "training_examples": checkpoint["counters"]["examples"],
        "atomic_exposures": exposures,
        "training_counters": checkpoint["counters"],
    }
    del checkpoint
    return model, history


def evaluate_pool(model, records, tokenizer, device, batch_size):
    """Use the original subset order and scoring implementation, including EOS."""
    by_id = {}
    for subset in ("A", "B"):
        selected = [row for row in records if row["subset"] == subset]
        if selected:
            _, predictions = evaluate(model, selected, tokenizer, device, batch_size)
            by_id.update({row["id"]: row for row in predictions})
    if set(by_id) != {row["id"] for row in records}:
        raise ValueError("Readout lost or duplicated original fact coverage")
    model.eval()
    return [by_id[row["id"]] for row in records]


def native_gate(predictions, native_file):
    """Gate only verifiable native output fields, with NLL reported separately."""
    native = json.loads(Path(native_file).read_text())
    available = native["atomic_A"] + native["atomic_B"]
    lookup = {row["id"]: row for row in available}
    if len(lookup) != len(available):
        raise ValueError("Duplicate native atomic prediction IDs")
    missing = [row["id"] for row in predictions if row["id"] not in lookup]
    if missing:
        raise ValueError(f"Selected atoms lack native endpoint predictions: {missing[:3]}")
    differences, nll_differences, token_available = [], [], 0
    for row in predictions:
        old = lookup[row["id"]]
        fields = [field for field in EXACT_FIELDS if field in old]
        if "prediction" not in fields or "alias_em" not in fields:
            raise ValueError("Native endpoint lacks a verifiable answer/scoring field")
        token_available += "generated_tokens" in fields
        changed = [field for field in fields if row[field] != old[field]]
        if changed:
            differences.append({"id": row["id"], "fields": changed})
        if old.get("nll") is not None and row.get("nll") is not None:
            nll_differences.append(abs(float(row["nll"]) - float(old["nll"])))
    return {
        "passed": not differences,
        "n": len(predictions),
        "native_atomic_pool_n": len(available),
        "token_ids_available_n": token_available,
        "exact_output_match_fraction": 1 - len(differences) / len(predictions),
        "differences": differences,
        "nll_max_absolute_difference_diagnostic": max(nll_differences, default=None),
        "native_selected_alias_em": sum(lookup[r["id"]]["alias_em"] for r in predictions)
        / len(predictions),
        "regenerated_alias_em": sum(r["alias_em"] for r in predictions) / len(predictions),
    }


def enrich(predictions, records, exposures):
    return [
        {
            **prediction,
            "subset": record["subset"],
            "relation": record["edge"][1],
            "question": record["question"],
            "historical_training_exposures": exposures[record["id"]],
            "prompt_tokens": len(record["encoded"]["prefix"]),
            "answer_and_eos_tokens": len(record["encoded"]["target"]),
        }
        for prediction, record in zip(predictions, records, strict=True)
    ]


def paired_metrics(canonical, heldout):
    if [row["id"] for row in canonical] != [row["id"] for row in heldout]:
        raise ValueError("Paired question forms must preserve ordered fact IDs")
    n = len(canonical)
    return {
        "n": n,
        "canonical": aggregate(canonical),
        "heldout": aggregate(heldout),
        "heldout_minus_canonical_alias_em": sum(
            b["alias_em"] - a["alias_em"] for a, b in zip(canonical, heldout, strict=True)
        )
        / n
        if n
        else None,
        "canonical_correct_heldout_wrong_n": sum(
            bool(a["alias_em"]) and not bool(b["alias_em"])
            for a, b in zip(canonical, heldout, strict=True)
        ),
        "canonical_wrong_heldout_correct_n": sum(
            not bool(a["alias_em"]) and bool(b["alias_em"])
            for a, b in zip(canonical, heldout, strict=True)
        ),
    }


def summarize(raw):
    summary = {}
    for snapshot, forms in raw.items():
        predicates = {
            "all": lambda row: True,
            "atomic_A": lambda row: row["subset"] == "A",
            "atomic_B": lambda row: row["subset"] == "B",
            "historically_exposed": lambda row: row["historical_training_exposures"] > 0,
            "historically_unexposed": lambda row: row["historical_training_exposures"] == 0,
        }
        summary[snapshot] = {
            group: paired_metrics(
                [row for row in forms["canonical"] if predicate(row)],
                [row for row in forms["heldout"] if predicate(row)],
            )
            for group, predicate in predicates.items()
        }
    changes = {}
    for form in ("canonical", "heldout"):
        before = {row["id"]: row for row in raw["stage_a"][form]}
        after = raw["final"][form]
        changes[form] = {
            subset: {
                "n": sum(row["subset"] == subset for row in after),
                "final_minus_stage_a_alias_em": sum(
                    row["alias_em"] - before[row["id"]]["alias_em"]
                    for row in after
                    if row["subset"] == subset
                )
                / sum(row["subset"] == subset for row in after)
                if any(row["subset"] == subset for row in after)
                else None,
            }
            for subset in ("A", "B")
        }
    return {"snapshots": summary, "paired_training_change": changes}


def evaluate_snapshots(config, spec, original_spec, data, tokenizer, pools, device, batch, out):
    raw, gates, histories = {}, {}, {}
    for snapshot in SNAPSHOTS:
        model, history = load_reader(original_spec, spec["checkpoints"][snapshot], data, device)
        before = model_digest(model)
        canonical = evaluate_pool(model, pools["canonical"], tokenizer, device, batch)
        gate = native_gate(canonical, spec["checkpoints"][snapshot]["predictions_file"])
        gates[snapshot] = gate
        write_json(out / "native-reproduction-gates.json", gates)
        write_json(out / f"canonical-{snapshot}.json", canonical)
        if not gate["passed"]:
            raise ValueError(f"Exact native reproduction gate failed for {snapshot}")
        heldout = evaluate_pool(model, pools["heldout"], tokenizer, device, batch)
        after = model_digest(model)
        if before != after or any(p.grad is not None for p in model.parameters()):
            raise ValueError("A zero-update readout changed parameters or computed gradients")
        histories[snapshot] = {
            **history,
            "model_sha256_before": before,
            "model_sha256_after": after,
        }
        raw[snapshot] = {
            "canonical": enrich(canonical, pools["canonical"], history["atomic_exposures"]),
            "heldout": enrich(heldout, pools["heldout"], history["atomic_exposures"]),
        }
        del model
        if torch.device(device).type == "cuda":
            torch.cuda.empty_cache()
    return raw, gates, histories


def compare_replay(original, replay, nll_tolerance):
    differences, largest_nll = [], 0.0
    for snapshot in SNAPSHOTS:
        for form in ("canonical", "heldout"):
            previous, fresh = original[snapshot][form], replay[snapshot][form]
            if [row["id"] for row in previous] != [row["id"] for row in fresh]:
                raise ValueError("Replay changed full ordered fact coverage")
            for old, new in zip(previous, fresh, strict=True):
                changed = [field for field in EXACT_FIELDS if old[field] != new[field]]
                nll_diff = abs(float(old["nll"]) - float(new["nll"]))
                largest_nll = max(largest_nll, nll_diff)
                if not math.isfinite(nll_diff) or nll_diff > nll_tolerance:
                    changed.append("nll")
                if changed:
                    differences.append(
                        {"snapshot": snapshot, "form": form, "id": old["id"], "fields": changed}
                    )
    return {
        "passed": not differences,
        "differences": differences,
        "nll_max_difference": largest_nll,
    }


def execute(
    config_path, name, out, device, limit=None, batch_size=None, audit=False, audit_out=None
):
    began = time.perf_counter()
    out = Path(out).resolve()
    if audit:
        prior = json.loads((out / "run.json").read_text())
        if limit is None:
            limit = prior["limit"]
        if batch_size is None:
            batch_size = prior["batch_size"]
    config, spec, original_spec, data, tokenizer, pools, source_hashes = load_inputs(
        config_path, name, limit
    )
    source = Path(spec["source_run_dir"]).resolve()
    if out == source or out.is_relative_to(source) or source.is_relative_to(out):
        raise ValueError("Readout output must be separate from the preserved source run")
    batch = batch_size or original_spec["evaluation_batch_size"]
    if batch < 1:
        raise ValueError("Evaluation batch size must be positive")
    torch.set_num_threads(original_spec.get("cpu_threads", 2))
    if torch.device(device).type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
    if audit:
        metadata = json.loads((out / "run.json").read_text())
        if metadata["config_sha256"] != sha256(config_path):
            raise ValueError("Audit config differs from the completed readout")
        if metadata["readout_module_sha256"] != sha256(__file__):
            raise ValueError("Audit source differs from the completed readout")
        if metadata["limit"] != limit or metadata["batch_size"] != batch:
            raise ValueError("Audit must preserve fixed coverage and evaluation batch size")
        complete = json.loads((out / "complete.json").read_text())
        checked_file(out / "paired-predictions.json", complete["predictions_sha256"])
        checked_file(out / "summary.json", complete["summary_sha256"])
        target = Path(audit_out).resolve() if audit_out else out / "audit"
        if (target / "paired-predictions.json").exists() or (target / "audit.json").exists():
            raise ValueError("Preserve previous audit attempts; use a separate --audit-out")
    else:
        if (out / "run.json").exists() or (out / "complete.json").exists():
            raise ValueError("Use a separate attempt directory; preserve previous readouts")
        target = out
    if target == source or target.is_relative_to(source) or source.is_relative_to(target):
        raise ValueError("Audit output must be separate from preserved source runs")
    target.mkdir(parents=True, exist_ok=True)
    metadata = {
        "name": name,
        "state": "running",
        "phase": "development",
        "smoke": limit is not None,
        "limit": limit,
        "batch_size": batch,
        "config_sha256": sha256(config_path),
        "readout_module_sha256": sha256(__file__),
        "compatible_source_sha256": source_hashes,
        "source_run_dir": str(source),
        "data_sha256": spec["data_sha256"],
        "checkpoint_sha256": {key: value["sha256"] for key, value in spec["checkpoints"].items()},
        "template_family_id": config["template_family_id"],
        "training_updates": 0,
        "new_training_examples": 0,
        "new_training_tokens": 0,
        "new_fact_training_exposures": 0,
        "pid": os.getpid(),
        "device": str(device),
        "created_utc": utc(),
    }
    if not audit:
        write_json(out / "run.json", metadata)
    raw, gates, histories = evaluate_snapshots(
        config, spec, original_spec, data, tokenizer, pools, device, batch, target
    )
    write_json(target / "paired-predictions.json", raw)
    summary = {
        **summarize(raw),
        "name": name,
        "phase": "development",
        "smoke": limit is not None,
        "atom_count": len(pools["canonical"]),
        "native_reproduction_passed": all(value["passed"] for value in gates.values()),
        "training_updates": 0,
        "new_training_examples": 0,
        "new_training_tokens": 0,
        "new_fact_training_exposures": 0,
        "readout_prompt_tokens": sum(
            row["prompt_tokens"]
            for forms in raw.values()
            for rows in forms.values()
            for row in rows
        ),
        "readout_generated_tokens": sum(
            len(row["generated_tokens"])
            for forms in raw.values()
            for rows in forms.values()
            for row in rows
        ),
        "historical_checkpoints": histories,
        "wall_seconds": time.perf_counter() - began,
        "limitations": config["limitations"],
    }
    write_json(target / "summary.json", summary)
    if audit:
        original = json.loads((out / "paired-predictions.json").read_text())
        result = compare_replay(original, raw, config["audit_nll_absolute_tolerance"])
        result.update(
            query_count=4 * len(pools["canonical"]),
            replay_predictions_sha256=sha256(target / "paired-predictions.json"),
            original_predictions_sha256=sha256(out / "paired-predictions.json"),
            new_process_pid=os.getpid(),
            audited_utc=utc(),
        )
        write_json(target / "audit.json", result)
        write_json(out / "audit.json", result)
        if not result["passed"]:
            raise ValueError("Independent full-pool replay differed; audit failure retained")
        return result
    completion = {
        "state": "complete",
        "passed": True,
        "summary_sha256": sha256(out / "summary.json"),
        "predictions_sha256": sha256(out / "paired-predictions.json"),
        "training_updates": 0,
        "smoke": limit is not None,
        "atom_count": len(pools["canonical"]),
        "native_reproduction_passed": True,
        "independent_replay": "pending",
        "completed_utc": utc(),
    }
    write_json(out / "complete.json", completion)
    return completion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--audit-out", type=Path)
    args = parser.parse_args()
    result = execute(
        args.config,
        args.run_name,
        args.out,
        args.device,
        args.limit,
        args.batch_size,
        args.audit,
        args.audit_out,
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
