"""Evaluation-only one-step writes on unchanged, independently audited parents.

The original four-step training identity is always used to load a parent.
Only a separate evaluation view changes writer.steps. Full original states,
queries, scoring and replay are retained; second-hop conditions are added from
saved predictions, without choosing facts, checkpoints or hyperparameters.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import torch

from . import ttcd_continual_eval as evaluation
from . import ttcd_verified_base as verified
from .grok_depth import write_json
from .path_overlap import sha256
from .ttcd_editing import load_fast, save_fast, tensor_digest
from .ttcd_experiment import packet_for

BATCH = "ttcd-one-step-second-hop-v1"


def validate_settings(settings):
    if (
        settings["batch"] != BATCH
        or settings["writer_steps"] != 1
        or settings["parent_writer_steps"] != 4
        or settings["new_base_parameter_updates"] != 0
        or settings["parent_phase"] != "stream"
        or set(settings["architectures"]) != {"transformer", "loop", "discoloop"}
    ):
        raise ValueError("Only the authorized three-parent one-step evaluation is supported")


def evaluation_view(parent, settings):
    validate_settings(settings)
    if parent["writer"]["steps"] != settings["parent_writer_steps"]:
        raise ValueError("Declare the actual unchanged parent writer budget")
    result = copy.deepcopy(parent)
    result["writer"]["steps"] = 1
    return result


def parent_engine(parent):
    if "architecture" in parent:
        from .ttcd_architecture import build_engine

        return build_engine(parent)
    return verified


def interfaces(parent, settings, architecture):
    view = evaluation_view(parent, settings)
    if architecture not in settings["architectures"]:
        raise ValueError("Unknown architecture")
    actual = parent.get("architecture", {}).get("name", "discoloop")
    if actual != architecture:
        raise ValueError("Parent architecture differs from the requested evaluation")
    engine = parent_engine(parent)

    def loader(_view, phase, device):
        if phase != "stream":
            raise ValueError("Only the fixed stream endpoint may be reused")
        return engine.load_reader(parent, phase, device)

    def identity(_view, phase, panel):
        return {
            "schema": BATCH,
            "architecture": architecture,
            "settings": settings,
            "evaluation_writer": view["writer"],
            "parent_writer": parent["writer"],
            "parent": engine.evaluation_identity(parent, phase, panel),
            "overlay_sha256": sha256(__file__),
            "new_base_parameter_updates": 0,
        }

    return view, loader, identity


def validate_panel(panel, world):
    edited = set(panel["edited_atoms"])
    if len(edited) != 32 or len(panel["orders"]) != 32:
        raise ValueError("Retain the complete frozen32-fact/32-order panel")
    support = world.support(7.2)
    for fact in panel["facts"]:
        for query in fact["D"]:
            if (
                int(world.second[query]) != fact["atom"]
                or int(world.first[query]) in edited
                or support[query]
            ):
                raise ValueError("D must edit only the second hop of an untrained chain")


def preflight(parent, settings, architecture, out, device):
    """Real endpoint, development facts only, no backbone/optimizer updates."""
    from .grokking_reproduction import encode_rows

    view, loader, _ = interfaces(parent, settings, architecture)
    reader, world = loader(view, "stream", device)
    panel = evaluation._panel(view, world)
    validate_panel(panel, world)
    if sha256(parent["continual"]["panel_path"]) != settings["panel_sha256"]:
        raise ValueError("Frozen formal panel bytes differ")
    if any(p.requires_grad for p in reader.parameters()):
        raise ValueError("Evaluation must freeze all backbone parameters")
    before = tensor_digest(reader.backbone.state_dict())
    arrays = encode_rows(world.atoms[:8], world.metadata)
    tokens, positions, _ = (torch.as_tensor(a, device=device) for a in arrays)
    with torch.no_grad():
        native = reader.backbone(tokens, positions)
        wrapped = reader(tokens, positions, fast=reader.empty_fast())
    torch.testing.assert_close(native, wrapped, rtol=0, atol=0)
    cases = [c for c in evaluation.case_bank(parent) if c["split"] == "calibration"]
    checks = []
    for group in ("id", "ood"):
        case = next(c for c in cases if c["group"] == group)
        fast, history = evaluation.write_fact(
            reader, reader.empty_fast(), view, world, case, device
        )
        if len(history["updates"]) != 1:
            raise AssertionError("A fact must produce exactly one local update")
        packet = torch.tensor([packet_for(view, world, case)], device=device)
        direct, _ = reader.write(
            packet,
            reader.empty_fast(),
            **{k: view["writer"][k] for k in ("teacher_window", "student_window", "learning_rate")},
        )
        if tensor_digest(fast) != tensor_digest(direct):
            raise AssertionError("One-step write differs from one direct writer call")
        path = Path(out) / f"development-{group}.pt"
        save_fast(path, reader, fast, {"development_only": True, "case": case})
        loaded, _ = load_fast(path, reader)
        if tensor_digest(loaded) != tensor_digest(fast):
            raise AssertionError("Fast-state save/reload differs")
        checks.append({"group": group, "case": case["case_id"], "one_step_exact": True})
    if tensor_digest(reader.backbone.state_dict()) != before:
        raise AssertionError("Engineering evaluation changed the backbone")
    return {
        "passed": True,
        "checks": checks,
        "native_A0_exact": True,
        "backbone_unchanged": True,
        "formal_model_queries": 0,
        "new_base_parameter_updates": 0,
        "engineering_fast_updates": 4,
        "parent_writer_steps": 4,
        "evaluation_writer_steps": 1,
    }


def joint(prediction, target="new_answers"):
    lengths = {len(prediction[k]) for k in ("answer", "stop", target)}
    if len(lengths) != 1:
        raise ValueError("Prediction/label lengths differ")
    return [
        int(a) == int(t) and int(s) == int(prediction["_eos"])
        for a, s, t in zip(
            prediction["answer"], prediction["stop"], prediction[target], strict=True
        )
    ]


def scored(prediction, metadata, target="new_answers"):
    # Saved answers are token ids; saved old/new_answers are entity ids.
    value = {
        **prediction,
        target: [int(x) + metadata["entity_offset"] for x in prediction[target]],
        "_eos": metadata["end_marker"],
    }
    return joint(value, target)


def fraction(values):
    values = list(values)
    return {
        "n": len(values),
        "correct": sum(values),
        "accuracy": float(np.mean(values)) if values else None,
    }


def summarize_rows(rows):
    rows = list(rows)
    result = {"all": fraction(r["new_correct"] for r in rows)}
    conditions = {
        "baseline_correct": lambda r: r["baseline_correct"],
        "baseline_correct_and_first_hop": lambda r: r["baseline_correct"]
        and r["baseline_first_correct"],
        "baseline_correct_and_both_current_atoms": lambda r: r["baseline_correct"]
        and r["first_correct"]
        and r["edited_atom_correct"],
    }
    for name, condition in conditions.items():
        selected = [r for r in rows if condition(r)]
        result[name] = {
            **fraction(r["new_correct"] for r in selected),
            "coverage": len(selected) / len(rows) if rows else None,
        }
    result["old_answer_residual"] = fraction(r["old_answer"] for r in rows)
    result["first_hop_accuracy"] = fraction(r["first_correct"] for r in rows)
    return result


def second_hop_analysis(panel, records, metadata):
    """All fixed queries first; baseline-correct conditions are additional readouts."""
    baseline = next(r["readout"] for r in records if r["spec"]["kind"] == "baseline")
    baseline_d = baseline["predictions"]["D"]
    baseline_correct = scored(baseline_d, metadata, "old_answers")
    baseline_first = scored(baseline["predictions"]["prerequisite_atoms"], metadata, "old_answers")
    keys = [tuple(row) for row in baseline_d["rows"]]
    if len(keys) != len(set(keys)):
        raise ValueError("The second-hop panel contains duplicate queries")
    before = dict(zip(keys, zip(baseline_correct, baseline_first, strict=True), strict=True))
    baseline_u = {
        pool: scored(baseline["predictions"][pool], metadata, "old_answers")
        for pool in ("U_atoms", "U_chains")
    }
    single_rows, final_rows, curves, u_curves, single_u = [], [], {}, {}, []
    immediate = {}
    age_rows = {}
    for record in records:
        spec, readout = record["spec"], record["readout"]
        if spec["kind"] not in ("single", "main") or readout is None:
            continue
        predictions = readout["predictions"]
        dc = scored(predictions["D"], metadata)
        first = scored(predictions["prerequisite_atoms"], metadata, "old_answers")
        ec = scored(predictions["E"], metadata)
        old = scored(predictions["D"], metadata, "old_answers")
        current, offset = [], 0
        for position, index in enumerate(spec["indices"]):
            fact = panel["facts"][index]
            age = len(spec["indices"]) - position - 1
            for j in range(offset, offset + len(fact["D"])):
                key = tuple(predictions["D"]["rows"][j])
                original, prerequisite = before[key]
                row = {
                    "query": list(key),
                    "fact": fact["atom"],
                    "new_correct": dc[j],
                    "baseline_correct": original,
                    "baseline_first_correct": prerequisite,
                    "first_correct": first[j],
                    "edited_atom_correct": ec[position],
                    "old_answer": old[j],
                    "rotation": spec["rotation"],
                    "age": age,
                    "t": spec["t"],
                }
                current.append(row)
                if spec["kind"] == "main":
                    anchor = (spec["rotation"], key)
                    if age == 0:
                        immediate[anchor] = (ec[position], dc[j])
                    atom_success, chain_success = immediate[anchor]
                    age_rows.setdefault(age, []).append(
                        {
                            **row,
                            "immediate_atom_correct": atom_success,
                            "immediate_chain_correct": chain_success,
                        }
                    )
            offset += len(fact["D"])
        if offset != len(dc):
            raise ValueError("Per-fact D query coverage differs")
        retention = {}
        for pool, original in baseline_u.items():
            after = scored(predictions[pool], metadata, "old_answers")
            if predictions[pool]["rows"] != baseline["predictions"][pool]["rows"]:
                raise ValueError("Unedited retention questions changed")
            retention[pool] = {
                "all": fraction(after),
                "original_correct": fraction(a for b, a in zip(original, after, strict=True) if b),
                "baseline": fraction(original),
            }
        if spec["kind"] == "single":
            single_rows.extend(current)
            single_u.append(retention)
        else:
            curves.setdefault(spec["t"], []).extend(current)
            u_curves.setdefault(spec["t"], []).append(retention)
            if spec["t"] == len(panel["facts"]):
                final_rows.extend(current)

    def summarize_u(values):
        return {
            pool: {
                measure: {
                    "accuracy": float(np.mean([v[pool][measure]["accuracy"] for v in values]))
                    if values and values[0][pool][measure]["n"]
                    else None,
                    "n_per_state": values[0][pool][measure]["n"] if values else 0,
                    "states": len(values),
                }
                for measure in ("all", "original_correct", "baseline")
            }
            for pool in baseline_u
        }

    result = {
        "schema": BATCH,
        "baseline": {"D": fraction(baseline_correct), "first_hop": fraction(baseline_first)},
        "single": {"D": summarize_rows(single_rows), "U": summarize_u(single_u)},
        "continuous": {
            str(t): {"D": summarize_rows(rows), "U": summarize_u(u_curves[t])}
            for t, rows in sorted(curves.items())
        },
        "by_age": {
            str(age): {
                "D": summarize_rows(rows),
                "previously_successful_D": fraction(
                    r["new_correct"] for r in rows if r["immediate_chain_correct"]
                ),
                "previously_successful_E": fraction(
                    {
                        (r["rotation"], r["t"], r["fact"]): r["edited_atom_correct"]
                        for r in rows
                        if r["immediate_atom_correct"]
                    }.values()
                ),
            }
            for age, rows in sorted(age_rows.items())
        },
        "primary_t": len(panel["facts"]),
        "analysis_unit": (
            "Fixed facts/queries in one graph; rotations and prefixes are repeated measures"
        ),
        "new_base_parameter_updates": 0,
    }
    paired = {
        "baseline": [
            {"query": list(k), "correct": v[0], "first_correct": v[1]} for k, v in before.items()
        ],
        "single": single_rows,
        "continuous_final": final_rows,
    }
    return result, paired


def read_records(out):
    # Match chronological dict insertion order, needed for immediate-write anchors.
    summary = json.loads((Path(out) / "summary.json").read_text())
    return [
        json.loads((Path(out) / "states" / key / "record.json").read_text())
        for key in summary["state_keys"]
    ]


def analyze(out, metadata):
    panel = json.loads((Path(out) / "panel.json").read_text())
    records = read_records(out)
    for record in records:
        history = record["write_history"]
        if history is not None and len(history["updates"]) != 1:
            raise ValueError("Saved transition does not contain exactly one fact update")
    return second_hop_analysis(panel, records, metadata)


def cross_architecture_report(paths, out):
    if set(paths) != {"transformer", "loop", "discoloop"}:
        raise ValueError("All three fixed architectures are required")
    data = {
        name: json.loads((Path(path) / "paired-queries.json").read_text())
        for name, path in paths.items()
    }
    universes = [{tuple(r["query"]) for r in item["baseline"]} for item in data.values()]
    if any(u != universes[0] for u in universes):
        raise ValueError("Architecture query universes differ")
    common = set.intersection(
        *[{tuple(r["query"]) for r in item["baseline"] if r["correct"]} for item in data.values()]
    )
    result = {
        "schema": BATCH,
        "all_queries": len(universes[0]),
        "common_baseline_correct": len(common),
        "coverage": len(common) / len(universes[0]) if universes[0] else None,
        "queries": [list(k) for k in sorted(common)],
        "architectures": {
            name: {
                condition: summarize_rows(r for r in item[condition] if tuple(r["query"]) in common)
                for condition in ("single", "continuous_final")
            }
            for name, item in data.items()
        },
        "empty_common_subset_is_retained": True,
    }
    Path(out).mkdir(parents=True, exist_ok=True)
    write_json(Path(out) / "common-subset.json", result)
    return result
