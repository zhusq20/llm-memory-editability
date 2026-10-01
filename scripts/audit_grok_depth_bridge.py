#!/usr/bin/env python3
"""Independently audit donor truth, saved scores, pairing, matrix and sample forwards.

No donor selector, scorer, tracer or evaluator from the intervention implementation
is imported. Sample reloads use historical SmallGPT's native Block.forward methods.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
import types
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "docs/development-artifacts/grok-depth-bridge-v1"
CONDITIONS = (
    "baseline",
    "identity_r1",
    "different_bridge_r1",
    "same_bridge_r1",
    "different_bridge_h",
    "different_bridge_h_r1",
    "counterfactual_input",
)


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(value, message):
    if not value:
        raise AssertionError(message)


def equal(actual, expected, label):
    require(np.array_equal(actual, expected), label)


def check_dict(actual, expected, label):
    require(actual.keys() == expected.keys(), f"{label}: fields differ")
    for key, value in expected.items():
        observed = actual[key]
        if isinstance(value, dict):
            check_dict(observed, value, f"{label}.{key}")
        elif isinstance(value, float):
            require(
                observed is not None and abs(observed - value) <= 1e-12,
                f"{label}.{key}: {observed} != {value}",
            )
        else:
            require(observed == value, f"{label}.{key}: {observed} != {value}")


def fraction(values, mask):
    return int(np.count_nonzero(values & mask)) / int(mask.sum()) if mask.any() else None


def audit_donors(world, arrays):
    rows = world["test_composite"]
    n = len(rows)
    equal(arrays["original_rows"], rows, "evaluation mother set changed")
    facts = {(int(h), int(r)): int(t) for h, r, t in world["id_atomic"]}
    partitions = {
        split: {tuple(map(int, row)) for row in world[split]}
        for split in ("train_composite", "test_composite", "unused_composite")
    }
    same_membership = Counter()
    same_membership_by_relation = defaultdict(Counter)
    donor_keys = [
        "original_rows",
        "original_bridge",
        "different_donor",
        "same_donor",
        "counterfactual_rows",
        "different_valid",
        "same_valid",
        "same_relation_control",
        "different_nondegenerate",
        "different_candidate_count",
        "same_candidate_count",
        "different_reason",
        "same_reason",
    ]
    for i, (h, r1, r2, t) in enumerate(rows):
        b = facts[h, r1]
        require(arrays["original_bridge"][i] == b and facts[b, r2] == t, "false original path")
        different, same = set(), set()
        for (dh, dr), db in facts.items():
            if (dh, dr) != (h, r1) and db == b and dh != t:
                same.add((dh, dr, db))
            dt = facts.get((db, r2))
            if (
                dr == r1
                and dh != h
                and db != b
                and dt is not None
                and dt != t
                and dh not in (t, dt)
                and (dh, dr, r2, dt) not in partitions["train_composite"]
            ):
                different.add((dh, dr, db))
        for family, candidates in (("different", different), ("same", same)):
            require(
                bool(arrays[family + "_valid"][i]) == bool(candidates), "donor missingness differs"
            )
            require(
                arrays[family + "_candidate_count"][i] == len(candidates), "candidate count differs"
            )
            donor = tuple(map(int, arrays[family + "_donor"][i]))
            if candidates:
                require(donor in candidates, "selected donor violates graph/prefix contract")
                require(arrays[family + "_reason"][i] == "eligible", "eligible reason differs")
                require(t not in donor[:2], "original answer appears in donor prefix")
            else:
                equal(donor, [-1, -1, -1], "missing donor sentinel differs")
                require(
                    arrays[family + "_reason"][i] != "eligible", "missing donor has eligible reason"
                )
        if different:
            dh, dr, db = arrays["different_donor"][i]
            dt = facts[db, r2]
            equal(arrays["counterfactual_rows"][i], [dh, dr, r2, dt], "false counterfactual")
            require(dt not in (dh, dr), "counterfactual answer appears in prefix")
            require(
                (dh, dr, r2, dt) in partitions["test_composite"] | partitions["unused_composite"],
                "counterfactual is not held out",
            )
            require(
                bool(arrays["different_nondegenerate"][i]) == (db != dt and db != t),
                "nondegenerate mask differs",
            )
        else:
            equal(
                arrays["counterfactual_rows"][i],
                [-1] * 4,
                "missing counterfactual sentinel differs",
            )
            require(not arrays["different_nondegenerate"][i], "missing donor marked nondegenerate")
        if same:
            dh, dr, _ = arrays["same_donor"][i]
            preferred = {candidate for candidate in same if candidate[1] == r1}
            if preferred:
                require((dh, dr, b) in preferred, "same-r1 donor preference violated")
            label = "same_r1" if dr == r1 else "different_r1"
            require(arrays["same_relation_control"][i] == label, "same-r1 label differs")
            membership = [
                split for split, values in partitions.items() if (dh, dr, r2, t) in values
            ]
            require(len(membership) == 1, "same-bridge counterfactual partition ambiguous")
            same_membership[membership[0]] += 1
            same_membership_by_relation[label][membership[0]] += 1
        else:
            require(arrays["same_relation_control"][i] == "missing", "missing same relation label")
    content_hash = hashlib.sha256()
    for key in donor_keys:
        content_hash.update(key.encode())
        content_hash.update(arrays[key].tobytes())
    return {
        "all_test_n": n,
        "different_n": int(arrays["different_valid"].sum()),
        "same_n": int(arrays["same_valid"].sum()),
        "paired_donor_content_sha256": content_hash.hexdigest(),
        "same_counterfactual_membership": dict(same_membership),
        "same_counterfactual_membership_by_relation": {
            key: dict(value) for key, value in same_membership_by_relation.items()
        },
    }


def audit_scores(summary, arrays):
    rows = arrays["original_rows"]
    n = len(rows)
    different, same = arrays["different_valid"], arrays["same_valid"]
    original, cf = rows[:, -1], arrays["counterfactual_rows"][:, -1]
    baseline = (arrays["baseline_answer"] == original) & (arrays["baseline_stop"] == 1)
    counterfactual = (
        different
        & (arrays["counterfactual_input_answer"] == cf)
        & (arrays["counterfactual_input_stop"] == 1)
    )
    atomic_correct = {}
    atomic_rows = {
        "different_first": arrays["different_donor"],
        "same_first": arrays["same_donor"],
        "counterfactual_second": np.column_stack((arrays["different_donor"][:, 2], rows[:, 2], cf)),
    }
    for name, facts in atomic_rows.items():
        mask = same if name == "same_first" else different
        key = "atomic_" + name
        equal(arrays[key + "_rows"], facts, "atomic prerequisite facts differ")
        equal(arrays[key + "_valid"], mask, "atomic prerequisite coverage differs")
        answer, stop = arrays[key + "_answer"], arrays[key + "_stop"]
        correct = answer == facts[:, -1]
        atomic_correct[name] = mask & correct & (stop == 1)
        for values in (answer, stop):
            require(values.shape == (n,), "atomic prediction shape differs")
            equal(values[~mask], np.full((~mask).sum(), -1), "missing atomic prediction not masked")
        expected = {
            "n": int(mask.sum()),
            "total_test_n": n,
            "coverage": int(mask.sum()) / n if n else None,
            "answer_accuracy": fraction(correct, mask),
            "complete_accuracy": fraction(correct & (stop == 1), mask),
            "eos_accuracy": fraction(stop == 1, mask),
        }
        check_dict(summary["atomic_preconditions"][name], expected, "atomic." + name)
    groups = {
        "all_test": np.ones(n, dtype=bool),
        "different_donor_available": different,
        "same_donor_available": same,
        "baseline_and_counterfactual_complete_correct": baseline & counterfactual,
        "different_bridge_nondegenerate": arrays["different_nondegenerate"],
        "same_bridge_same_r1": same & (arrays["same_relation_control"] == "same_r1"),
        "same_bridge_different_r1": same & (arrays["same_relation_control"] == "different_r1"),
        "both_counterfactual_atomic_facts_complete_correct": atomic_correct["different_first"]
        & atomic_correct["counterfactual_second"],
    }
    require(set(summary["scores"]) == set(groups), "unexpected or missing subgroup")
    require(tuple(summary["conditions"]) == CONDITIONS, "condition matrix differs")
    for condition in CONDITIONS:
        valid = (
            same
            if condition == "same_bridge_r1"
            else different
            if condition.startswith("different_") or condition == "counterfactual_input"
            else np.ones(n, dtype=bool)
        )
        equal(arrays[condition + "_valid"], valid, "condition coverage differs")
        for field in ("answer", "stop"):
            values = arrays[condition + "_" + field]
            require(values.shape == (n,), "prediction mother-set shape differs")
            equal(values[~valid], np.full((~valid).sum(), -1), "unavailable prediction not masked")
        margins = arrays[condition + "_cf_minus_original_logit"]
        equal(np.isfinite(margins), valid & different, "margin coverage differs")
    for group, mask in groups.items():
        equal(
            arrays["subset_" + group],
            mask,
            "saved subgroup differs from independently recomputed mask",
        )
        expected = {
            "subset_n": int(mask.sum()),
            "total_test_n": n,
            "coverage": int(mask.sum()) / n if n else None,
            "conditions": {},
        }
        for condition in CONDITIONS:
            chosen = mask & arrays[condition + "_valid"]
            cf_chosen = chosen & different
            answer, stop = arrays[condition + "_answer"], arrays[condition + "_stop"]
            expected["conditions"][condition] = {
                "n": int(chosen.sum()),
                "total_test_n": n,
                "coverage": int(chosen.sum()) / n if n else None,
                "original_answer_accuracy": fraction(answer == original, chosen),
                "original_complete_accuracy": fraction((answer == original) & (stop == 1), chosen),
                "eos_accuracy": fraction(stop == 1, chosen),
                "answer_change_rate": fraction(answer != arrays["baseline_answer"], chosen),
                "cf_target_n": int(cf_chosen.sum()),
                "cf_answer_accuracy": fraction(answer == cf, cf_chosen),
                "cf_complete_accuracy": fraction((answer == cf) & (stop == 1), cf_chosen),
            }
        check_dict(summary["scores"][group], expected, "scores." + group)
    for field in ("answer", "stop"):
        equal(
            arrays["baseline_" + field], arrays["identity_r1_" + field], "identity control changed"
        )
        if summary["spec"]["layers"] == 1:
            for condition in CONDITIONS[2:6]:
                mask = arrays[condition + "_valid"]
                equal(
                    arrays[condition + "_" + field][mask],
                    arrays["baseline_" + field][mask],
                    "single-layer negative control changed",
                )
    expected_coverage = {
        "total_test_n": n,
        "different_n": int(different.sum()),
        "same_n": int(same.sum()),
        "different_reasons": dict(Counter(arrays["different_reason"].tolist())),
        "same_reasons": dict(Counter(arrays["same_reason"].tolist())),
        "same_relation_control": dict(Counter(arrays["same_relation_control"].tolist())),
    }
    check_dict(summary["donor_coverage"], expected_coverage, "donor_coverage")
    return {
        "all_groups": len(groups),
        "all_conditions": len(CONDITIONS),
        "all_atomic_prerequisites": len(atomic_rows),
    }


def independent_reload(summary, arrays, device):
    import torch
    from torch.nn import functional as F

    source = Path(summary["source"]["source_dir"])
    metadata = read_json(source / "metadata.json")
    snapshot = source / "source"
    for relative, expected in summary["source"]["source_hashes"].items():
        require(sha256(snapshot / relative) == expected, "historical copied-source hash differs")
    package = "_independent_bridge_audit_" + hashlib.sha256(str(source).encode()).hexdigest()[:12]
    namespace = types.ModuleType(package)
    namespace.__path__ = [str(snapshot / "src/llm_memory_editability")]
    sys.modules[package] = namespace
    training = importlib.import_module(package + ".grok_depth")
    spec = metadata["spec"]
    checkpoint_path = source / f"weights-{summary['step']:07d}.pt"
    require(
        sha256(checkpoint_path) == summary["source"]["checkpoint_sha256"], "checkpoint hash differs"
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    require(
        checkpoint["step"] == summary["step"] and checkpoint["spec"] == spec,
        "checkpoint identity differs",
    )
    model = (
        training.SmallGPT(
            training.ModelConfig(
                vocab_size=2 + spec["entities"] + spec["relations"],
                width=spec["width"],
                layers=spec["layers"],
                heads=spec["heads"],
                context=8,
            ),
            spec["dropout"],
        )
        .to(device)
        .eval()
    )
    model.load_state_dict(checkpoint["model"], strict=True)

    def forward(tokens, donor=None, patch_positions=(), capture=False):
        hidden = model.token(tokens) + model.position(torch.arange(tokens.shape[1], device=device))
        first = None
        for index, block in enumerate(model.blocks):
            hidden = block(hidden)
            if index == 0:
                first = hidden.clone()
                if donor is not None:
                    hidden = hidden.clone()
                    for position in patch_positions:
                        hidden[:, position] = donor[:, position]
        logits = F.linear(model.ln_final(hidden), model.token.weight)
        return (logits, first) if capture else logits

    comparisons = {}
    with torch.no_grad():
        for condition in CONDITIONS:
            # Sample checkpoints, but preserve every condition's historical batch
            # shape: changing TF32 matrix shapes can change rounding and argmaxes.
            selected = np.flatnonzero(arrays[condition + "_valid"])
            if not len(selected):
                comparisons[condition] = {"n": 0}
                continue
            row_key = (
                "counterfactual_rows" if condition == "counterfactual_input" else "original_rows"
            )
            rows = arrays[row_key][selected]
            tokens = torch.as_tensor(rows.copy(), device=device)
            donor, positions = None, ()
            if "bridge_" in condition:
                key = "same_donor" if condition == "same_bridge_r1" else "different_donor"
                prefix = torch.as_tensor(arrays[key][selected, :2].copy(), device=device)
                _, donor = forward(F.pad(prefix, (0, 2), value=0), capture=True)
                donor = donor[:, :2]
                positions = (
                    (0, 1)
                    if condition == "different_bridge_h_r1"
                    else (0,)
                    if condition == "different_bridge_h"
                    else (1,)
                )
            elif condition == "identity_r1":
                _, donor = forward(tokens, capture=True)
                positions = (1,)
            logits = forward(tokens, donor, positions)
            if condition == "baseline":
                require(
                    torch.allclose(logits, model(tokens), atol=1e-5, rtol=1e-5),
                    "native Block path differs from historical forward",
                )
            answer = logits[:, 2].argmax(-1)
            equal(
                answer.cpu().numpy(),
                arrays[condition + "_answer"][selected],
                "independent reloaded answer differs",
            )
            tokens[:, 3] = answer
            stop = forward(tokens, donor, positions)[:, 3].argmax(-1)
            equal(
                stop.cpu().numpy(),
                arrays[condition + "_stop"][selected],
                "independent reloaded generated EOS differs",
            )
            cf_mask = arrays["different_valid"][selected]
            local = np.flatnonzero(cf_mask)
            if len(local):
                ix = torch.as_tensor(local, device=device)
                cf_target = torch.as_tensor(
                    arrays["counterfactual_rows"][selected[local], -1], device=device
                )
                original_target = torch.as_tensor(
                    arrays["original_rows"][selected[local], -1], device=device
                )
                margins = (logits[ix, 2, cf_target] - logits[ix, 2, original_target]).cpu().numpy()
                saved = arrays[condition + "_cf_minus_original_logit"][selected[local]]
                require(
                    np.allclose(margins, saved, atol=2e-4, rtol=2e-4),
                    "reloaded answer margins differ",
                )
                max_delta = float(np.max(np.abs(margins - saved)))
            else:
                max_delta = None
            comparisons[condition] = {
                "n": len(selected),
                "indices": selected.tolist(),
                "predictions_identical": True,
                "margin_max_abs_delta": max_delta,
            }
    return {
        "method": (
            "Historical native Block.forward; independent prefix capture, replacement, "
            "greedy answer then generated EOS; all rows of three sampled checkpoints"
        ),
        "device": device,
        "conditions": comparisons,
    }


def audit_run(path, reload_device=None):
    summary = read_json(path / "summary.json")
    require(summary["state"] == "complete", "model state is incomplete")
    predictions_path = path / "predictions.npz"
    require(sha256(predictions_path) == summary["predictions_sha256"], "prediction hash differs")
    with np.load(predictions_path, allow_pickle=False) as saved:
        arrays = {key: saved[key].copy() for key in saved.files}
    source = Path(summary["source"]["source_dir"])
    require(
        sha256(source / "world.npz") == summary["source"]["world_file_sha256"], "world hash differs"
    )
    with np.load(source / "world.npz", allow_pickle=False) as stored:
        world = {key: stored[key].copy() for key in stored.files}
    require(summary["spec"] == read_json(source / "metadata.json")["spec"], "source spec differs")
    scores = audit_scores(summary, arrays)
    donors = audit_donors(world, arrays)
    with np.load(
        source / f"predictions-{summary['step']:07d}.npz", allow_pickle=False
    ) as historical:
        for field in ("answer", "stop"):
            equal(
                arrays["baseline_" + field],
                historical["test_composite_" + field],
                "historical baseline differs",
            )
    result = {
        "phase": summary["phase"],
        "run_id": summary["run_id"],
        "step": summary["step"],
        "world": summary["spec"]["world_seed"],
        "state": "passed",
        "scores": scores,
        "donors": donors,
        "summary_sha256": sha256(path / "summary.json"),
        "predictions_sha256": sha256(predictions_path),
    }
    if reload_device:
        result["independent_reload"] = independent_reload(summary, arrays, reload_device)
    return result, summary


def audit_aggregate(summaries):
    path = ARTIFACTS / "summary.json"
    if not path.exists():
        return {"state": "not_yet_generated"}
    report = read_json(path)
    worlds = defaultdict(lambda: defaultdict(list))
    for summary in summaries:
        spec = summary["spec"]
        arm = f"d{spec['layers']}-w{spec['width']}"
        for group, content in summary["scores"].items():
            for condition, metrics in content["conditions"].items():
                key = (summary["phase"], arm, summary["step"], group, condition)
                worlds[key][spec["world_seed"]].append(metrics)
        for condition, metrics in summary["atomic_preconditions"].items():
            key = (summary["phase"], arm, summary["step"], "atomic_preconditions", condition)
            worlds[key][spec["world_seed"]].append(metrics)
    require(report["model_states"] == len(summaries), "aggregate model-state count differs")
    require(len(report["groups"]) == len(worlds), "aggregate group count differs")
    for row in report["groups"]:
        key = tuple(row[k] for k in ("phase", "arm", "step", "group", "condition"))
        by_world = worlds[key]
        require(sorted(by_world) == row["worlds"], "aggregate world identities differ")
        for metric, observed in row["metrics"].items():
            world_means = []
            for repeats in by_world.values():
                values = [r[metric] for r in repeats if r[metric] is not None]
                if values:
                    world_means.append(sum(values) / len(values))
            expected = {
                "mean": sum(world_means) / len(world_means) if world_means else None,
                "min": min(world_means) if world_means else None,
                "max": max(world_means) if world_means else None,
                "n_worlds": len(world_means),
            }
            check_dict(observed, expected, "aggregate." + metric)
    return {
        "state": "passed",
        "groups": len(worlds),
        "source_sha256": sha256(path),
        "method": "Independent within-world arithmetic mean, then equal world arithmetic mean",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/grok-depth-bridge-v1.json")
    parser.add_argument("--phase", choices=("all", "development", "confirmation"), default="all")
    parser.add_argument(
        "--reload", action="store_true", help="Reload three predefined endpoint states"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--lock", default=str(ARTIFACTS / "execution-lock-v3.json"))
    args = parser.parse_args()
    config = read_json(ROOT / args.config)
    lock = read_json(args.lock)
    for relative, expected in lock["files"].items():
        require(sha256(ROOT / relative) == expected, "execution lock hash differs: " + relative)
    if args.reload:
        import torch

        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = config["evaluation_tf32"]
        torch.backends.cudnn.allow_tf32 = config["evaluation_tf32"]
        if args.device.startswith("cuda"):
            torch.cuda.set_device(torch.device(args.device))
    reload_states = {
        ("development", "dev-d2-original", 128000),
        ("confirmation", "conf-w143011-s14301-d2", 128000),
        ("confirmation", "conf-w143011-s14301-d1wide", 128000),
    }
    phases = ("development", "confirmation") if args.phase == "all" else (args.phase,)
    expected, observed = set(), set()
    records, summaries, failures = [], [], []
    for phase in phases:
        phase_dir = ROOT / config["output_root"] / phase
        metadata = read_json(phase_dir / "metadata.json")
        require(metadata["config"] == config, "phase configuration differs")
        for relative, value in metadata["files"].items():
            require(
                value == lock["files"][relative] == sha256(phase_dir / "source" / relative),
                "phase frozen source differs",
            )
        for item in config[phase + "_runs"]:
            expected.update((phase, item["run_id"], step) for step in item["steps"])
        require((phase_dir / "batch-summary.json").exists(), "batch completion marker missing")
        batch = read_json(phase_dir / "batch-summary.json")
        require(batch["state"] == "complete", "batch incomplete")
        for path in sorted(phase_dir.glob("*/step-*")):
            key = phase, path.parent.name, int(path.name.removeprefix("step-"))
            observed.add(key)
            try:
                result, summary = audit_run(
                    path, args.device if args.reload and key in reload_states else None
                )
                require(
                    (summary["phase"], summary["run_id"], summary["step"]) == key,
                    "output identity differs",
                )
                require(
                    summary["evaluation_files"] == metadata["files"], "model-state source differs"
                )
                records.append(result)
                summaries.append(summary)
                print(f"Audited {key}: passed", flush=True)
            except Exception as exc:
                failures.append({"identity": key, "error": repr(exc)})
                print(f"Audited {key}: failed {exc!r}", flush=True)
    pairing = defaultdict(set)
    for record in records:
        pairing[record["world"]].add(record["donors"]["paired_donor_content_sha256"])
    require(
        all(len(values) == 1 for values in pairing.values()), "same-world donor pairing differs"
    )
    aggregate = audit_aggregate(summaries) if args.phase == "all" and not failures else None
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "auditor_sha256": sha256(Path(__file__)),
        "passed": expected == observed and not failures,
        "phase": args.phase,
        "matrix": {
            "expected": len(expected),
            "observed": len(observed),
            "missing": sorted(expected - observed),
            "unexpected": sorted(observed - expected),
        },
        "execution_lock_verified": True,
        "same_world_donors_identical": True,
        "runs": records,
        "failures": failures,
        "aggregate": aggregate,
        "independent_reload_states": sum("independent_reload" in r for r in records),
    }
    suffix = "" if args.phase == "all" else "-" + args.phase
    output = ARTIFACTS / f"audit{suffix}.json"
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(
        json.dumps({"passed": result["passed"], "audited": len(records), "output": str(output)}),
        flush=True,
    )
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
