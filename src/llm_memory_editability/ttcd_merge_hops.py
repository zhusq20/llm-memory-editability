"""Reuse audited one-step states; compare independent increments in both hop roles."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from . import ttcd_continual_eval as original
from . import ttcd_editing as fast_io
from . import ttcd_one_step as one
from .edit_scope_development import evaluate_queries
from .grok_depth import write_json
from .path_overlap import World, sha256
from .ttcd_experiment import Session
from .ttcd_merge_hop_panel import build_panel

BATCH = "ttcd-merge-hops-v1"


def validate_settings(settings):
    if (
        settings["batch"] != BATCH
        or settings["parent_batch"] != one.BATCH
        or settings["writer_steps"] != 1
        or settings["new_base_parameter_updates"] != 0
        or settings["methods"] != ["continuous", "independent_merge"]
        or settings["roles"] != ["first", "second"]
    ):
        raise ValueError("Only the authorized one-step merge and hop comparison is supported")


def prepare(parent, settings, out):
    validate_settings(settings)
    path = Path(parent["continual"]["panel_path"])
    if sha256(path) != settings["panel_sha256"]:
        raise ValueError("Original fact/replacement panel changed")
    world = World(parent["data_dir"])
    panel = build_panel(json.loads(path.read_text()), world, settings, parent["phi"])
    write_json(Path(out) / "panel.json", panel)
    return {
        "passed": True,
        "panel_sha256": sha256(Path(out) / "panel.json"),
        "first_hop_coverage": panel["first_hop_coverage"],
        "model_queries": 0,
        "new_base_parameter_updates": 0,
    }


def source_context(parent, settings, architecture, source, out, device):
    validate_settings(settings)
    source, out = Path(source), Path(out)
    old_settings = json.loads(
        (Path(__file__).resolve().parents[2] / "configs/one-step.json").read_text()
    )
    view, loader, identity = one.interfaces(parent, old_settings, architecture)
    reader, world = loader(view, "stream", device)
    panel = json.loads((out / "panel.json").read_text())
    prepared = build_panel(original._panel(view, world), world, settings, parent["phi"])
    if panel != prepared:
        raise ValueError("Graph-only hop panel changed after preparation")
    summary = json.loads((source / "summary.json").read_text())
    if summary["identity"] != identity(view, "stream", original._panel(view, world)):
        raise ValueError("Reused one-step state identity differs from the exact parent")
    for stage in ("audit", "cloud-audit"):
        if json.loads((source / f"result-{stage}.json").read_text()).get("passed") is not True:
            raise ValueError("Reused states require passing independent and cloud audits")
    if any(p.requires_grad for p in reader.parameters()):
        raise ValueError("All reused backbone parameters must remain frozen")
    return reader, world, view, panel


def add_increment(previous, increment):
    """Out-of-place FP32 sum of immutable deltas; no averaging or base weights."""
    if previous.keys() != increment.keys():
        raise ValueError("Fast matrix keys differ")
    for key, value in previous.items():
        other = increment[key]
        if value.shape != other.shape or value.dtype != other.dtype or value.device != other.device:
            raise ValueError("Fast matrix shape/dtype/device differs")
    return {key: value + increment[key] for key, value in previous.items()}


def state_plan(panel):
    n = len(panel["facts"])
    plans = {
        "baseline": {
            "kind": "baseline",
            "indices": list(range(n)),
            "parent_key": "baseline",
            "t": 0,
            "rotation": None,
        }
    }
    for i in range(n):
        plans[f"single/f{i:03d}"] = {
            "kind": "single",
            "indices": [i],
            "parent_key": f"single/f{i:03d}",
            "t": 1,
            "rotation": None,
        }
    for method in ("continuous", "independent_merge"):
        for rotation, order in enumerate(panel["orders"]):
            for t in range(1, n + 1):
                plans[f"{method}/r{rotation:03d}/t{t:03d}"] = {
                    "kind": method,
                    "indices": order[:t],
                    "t": t,
                    "rotation": rotation,
                    "parent_key": f"main/r{rotation:03d}/t{t:03d}"
                    if method == "continuous"
                    else None,
                }
    return plans


def source_state(reader, source, key):
    relative = Path("states") / key
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Parent state path escaped")
    directory = Path(source) / relative
    record = json.loads((directory / "record.json").read_text())
    fast, provenance = fast_io.load_fast(directory / "fast.pt", reader)
    if (
        record["key"] != key
        or provenance != record["spec"]
        or fast_io.tensor_digest(fast) != record["fast_digest"]
    ):
        raise ValueError("Parent state provenance/digest changed")
    if key != "baseline" and len(record["write_history"]["updates"]) != 1:
        raise ValueError("Parent transition must use one update per fact")
    return fast, record, sha256(directory / "record.json")


@torch.no_grad()
def read_first(reader, fast, world, panel, indices, device):
    before = fast_io.tensor_digest(fast)
    selected = [panel["first_hop"][i] for i in indices]
    queries = [q for entry in selected for q in entry["D"]]
    answers = [a for entry in selected for a in entry["new_answers"]]
    successors = [a for entry in selected for a in entry["successor_atoms"]]
    pools = {
        "D": (world.chains[np.asarray(queries, dtype=np.int64)], answers),
        "other_atoms": (world.atoms[np.asarray(successors, dtype=np.int64)], None),
    }
    predictions = {}
    for name, (rows, targets) in pools.items():
        result = evaluate_queries(
            Session(reader, fast).eval(),
            rows,
            world.metadata,
            torch.device(device),
            new_answers=targets,
            precision="fp32",
        )
        predictions[name] = {k: v.tolist() for k, v in result.items()}
    if fast_io.tensor_digest(fast) != before:
        raise AssertionError("First-hop readout changed fast weights")
    return {"predictions": predictions, "fast_digest_before": before, "fast_digest_after": before}


def identity(parent, settings, architecture, source, out):
    return {
        "schema": BATCH,
        "settings": settings,
        "architecture": architecture,
        "parent_config": parent,
        "parent_summary_sha256": sha256(Path(source) / "summary.json"),
        "parent_audit_sha256": sha256(Path(source) / "result-audit.json"),
        "parent_cloud_sha256": sha256(Path(source) / "result-cloud-audit.json"),
        "panel_sha256": sha256(Path(out) / "panel.json"),
        "overlay_sha256": sha256(__file__),
    }


def preflight(parent, settings, architecture, source, out, device):
    reader, world, view, _ = source_context(parent, settings, architecture, source, out, device)
    before = fast_io.tensor_digest(reader.backbone.state_dict())
    cases = [c for c in original.case_bank(view) if c["split"] == "calibration"][:2]
    increments = []
    for case in cases:
        delta, history = original.write_fact(reader, reader.empty_fast(), view, world, case, device)
        if len(history["updates"]) != 1:
            raise AssertionError("Development write did not use exactly one step")
        increments.append(delta)
    digests = [fast_io.tensor_digest(delta) for delta in increments]
    merged = reader.empty_fast()
    for delta in increments:
        merged = add_increment(merged, delta)
    for key in merged:
        torch.testing.assert_close(
            merged[key], increments[0][key] + increments[1][key], rtol=0, atol=0
        )
    path = Path(out) / "development-merged.pt"
    fast_io.save_fast(path, reader, merged, {"development_only": True})
    loaded, _ = fast_io.load_fast(path, reader)
    if fast_io.tensor_digest(loaded) != fast_io.tensor_digest(merged):
        raise AssertionError("Merged fast state reload differs")
    if digests != [
        fast_io.tensor_digest(delta) for delta in increments
    ] or before != fast_io.tensor_digest(reader.backbone.state_dict()):
        raise AssertionError("Merging mutated donors or backbone")
    return {
        "passed": True,
        "development_one_step_writes": 2,
        "new_base_parameter_updates": 0,
        "formal_model_queries": 0,
        "sum_exact": True,
        "donors_and_backbone_unchanged": True,
    }


def evaluate(parent, settings, architecture, source, out, device, log=None):
    started = time.monotonic()
    out, source = Path(out), Path(source)
    reader, world, view, panel = source_context(parent, settings, architecture, source, out, device)
    initial = fast_io.tensor_digest(reader.backbone.state_dict())
    plans = state_plan(panel)
    donors = [
        source_state(reader, source, f"single/f{i:03d}")[0] for i in range(len(panel["facts"]))
    ]
    donor_digests = [fast_io.tensor_digest(delta) for delta in donors]
    records, merged, previous_rotation = [], None, None
    for key, spec in plans.items():
        directory = out / "states" / key
        directory.mkdir(parents=True, exist_ok=False)
        if spec["kind"] == "independent_merge":
            if spec["rotation"] != previous_rotation:
                merged, previous_rotation = reader.empty_fast(), spec["rotation"]
            merged = add_increment(merged, donors[spec["indices"][-1]])
            fast = merged
            fast_io.save_fast(directory / "fast.pt", reader, fast, spec)
            base = original.read_state(
                reader,
                fast,
                world,
                panel,
                spec["indices"],
                [panel["facts"][i]["new_tail"] for i in spec["indices"]],
                device,
            )
            source_sha = None
        else:
            fast, reference, source_sha = source_state(reader, source, spec["parent_key"])
            base = reference["readout"]
        first = read_first(reader, fast, world, panel, spec["indices"], device)
        record = {
            "key": key,
            "spec": spec,
            "readout": base,
            "first": first,
            "fast_digest": fast_io.tensor_digest(fast),
            "parent_record_sha256": source_sha,
        }
        write_json(directory / "record.json", record)
        records.append(record)
        write_json(
            out / "evaluation-progress.json",
            {"states": len(records), "expected": len(plans), "last": key},
        )
        if log:
            log(len(records), record)
    if donor_digests != [
        fast_io.tensor_digest(delta) for delta in donors
    ] or initial != fast_io.tensor_digest(reader.backbone.state_dict()):
        raise AssertionError("Evaluation mutated donors or backbone")
    summary, paired = analyze(panel, records, world.metadata)
    summary.update(
        identity=identity(parent, settings, architecture, source, out),
        state_keys=list(plans),
        states=len(plans),
        new_merged_states=len(panel["orders"]) * len(panel["facts"]),
        reused_one_step_donors=len(donors),
        new_training_updates=0,
        wall_seconds=time.monotonic() - started,
        compute=dict(reader.compute),
        base_parameters=sum(p.numel() for p in reader.parameters()),
        fast_elements=sum(v.numel() for v in donors[0].values()),
    )
    write_json(out / "summary.json", summary)
    write_json(out / "paired-queries.json", paired)
    return summary


def ratio(rows, condition=lambda r: True):
    rows = list(rows)
    selected = [r for r in rows if condition(r)]
    by_fact = {}
    for row in selected:
        by_fact.setdefault(row["fact"], []).append(row["new_correct"])
    return {
        **one.fraction(r["new_correct"] for r in selected),
        "coverage": len(selected) / len(rows) if rows else None,
        "facts": len(by_fact),
        "fact_equal_accuracy": float(np.mean([np.mean(v) for v in by_fact.values()]))
        if by_fact
        else None,
    }


def summarize_rows(rows):
    rows = list(rows)
    return {
        "all": ratio(rows),
        "baseline_correct": ratio(rows, lambda r: r["baseline_correct"]),
        "baseline_correct_and_other_fact": ratio(
            rows, lambda r: r["baseline_correct"] and r["baseline_other_correct"]
        ),
        "baseline_correct_and_both_current_atoms": ratio(
            rows,
            lambda r: r["baseline_correct"] and r["other_correct"] and r["edited_atom_correct"],
        ),
        "old_answer_residual": one.fraction(r["old_answer"] for r in rows),
    }


def role_rows(record, panel, role, before, metadata):
    spec = record["spec"]
    predictions = (
        record["readout"]["predictions"] if role == "second" else record["first"]["predictions"]
    )
    other = "prerequisite_atoms" if role == "second" else "other_atoms"
    dc, oc, old = (
        one.scored(predictions["D"], metadata),
        one.scored(predictions[other], metadata, "old_answers"),
        one.scored(predictions["D"], metadata, "old_answers"),
    )
    ec = one.scored(record["readout"]["predictions"]["E"], metadata)
    rows, offset = [], 0
    for position, index in enumerate(spec["indices"]):
        entry = panel["facts"][index] if role == "second" else panel["first_hop"][index]
        for j in range(offset, offset + len(entry["D"])):
            query = tuple(predictions["D"]["rows"][j])
            original_correct, other_before = before[query]
            rows.append(
                {
                    "query": list(query),
                    "fact": panel["facts"][index]["atom"],
                    "new_correct": dc[j],
                    "old_answer": old[j],
                    "other_correct": oc[j],
                    "edited_atom_correct": ec[position],
                    "baseline_correct": original_correct,
                    "baseline_other_correct": other_before,
                    "rotation": spec["rotation"],
                    "t": spec["t"],
                    "age": len(spec["indices"]) - position - 1,
                }
            )
        offset += len(entry["D"])
    if offset != len(dc):
        raise ValueError("Role query/fact coverage differs")
    return rows


def analyze(panel, records, metadata):
    baseline = next(r for r in records if r["spec"]["kind"] == "baseline")
    before, baselines = {}, {}
    for role in ("first", "second"):
        pred = (
            baseline["first"]["predictions"]
            if role == "first"
            else baseline["readout"]["predictions"]
        )
        other = "other_atoms" if role == "first" else "prerequisite_atoms"
        dc, oc = (
            one.scored(pred["D"], metadata, "old_answers"),
            one.scored(pred[other], metadata, "old_answers"),
        )
        keys = [tuple(q) for q in pred["D"]["rows"]]
        if len(keys) != len(set(keys)):
            raise ValueError("Duplicate role queries")
        before[role] = dict(zip(keys, zip(dc, oc, strict=True), strict=True))
        baselines[role] = one.fraction(dc)
    base_u = {
        pool: one.scored(baseline["readout"]["predictions"][pool], metadata, "old_answers")
        for pool in ("U_atoms", "U_chains")
    }
    collected = {role: {} for role in before}
    u, e, ages, immediate = {}, {}, {}, {}
    paired = {
        "baseline": {
            role: [{"query": list(k), "correct": v[0]} for k, v in values.items()]
            for role, values in before.items()
        },
        "roles": {
            role: {method: [] for method in ("single", "continuous", "independent_merge")}
            for role in before
        },
    }
    for record in records:
        spec = record["spec"]
        method = spec["kind"]
        if method == "baseline":
            continue
        t = spec["t"]
        for role in before:
            rows = role_rows(record, panel, role, before[role], metadata)
            collected[role].setdefault((method, t), []).extend(rows)
            if method == "single" or t == len(panel["facts"]):
                paired["roles"][role][method].extend(rows)
        correct_e = one.scored(record["readout"]["predictions"]["E"], metadata)
        e.setdefault((method, t), []).extend(correct_e)
        for position, index in enumerate(spec["indices"]):
            age = len(spec["indices"]) - position - 1
            if method != "single":
                anchor = (method, spec["rotation"], index)
                if age == 0:
                    immediate[anchor] = correct_e[position]
                ages.setdefault((method, age), []).append((immediate[anchor], correct_e[position]))
        for pool, eligible in base_u.items():
            pred = record["readout"]["predictions"][pool]
            if pred["rows"] != baseline["readout"]["predictions"][pool]["rows"]:
                raise ValueError("U query panel changed")
            after = one.scored(pred, metadata, "old_answers")
            item = u.setdefault((method, t, pool), {"all": [], "retained": []})
            item["all"].extend(after)
            item["retained"].extend(a for b, a in zip(eligible, after, strict=True) if b)
    summary = {
        "schema": BATCH,
        "baseline_D": baselines,
        "first_hop_coverage": panel["first_hop_coverage"],
        "roles": {
            role: {
                method: {
                    str(t): summarize_rows(rows) for (m, t), rows in values.items() if m == method
                }
                for method in ("single", "continuous", "independent_merge")
            }
            for role, values in collected.items()
        },
        "E": {
            method: {str(t): one.fraction(values) for (m, t), values in e.items() if m == method}
            for method in ("single", "continuous", "independent_merge")
        },
        "U": {
            method: {
                str(t): {
                    pool: {measure: one.fraction(v) for measure, v in values.items()}
                    for (m, k, pool), values in u.items()
                    if m == method and k == t
                }
                for t in sorted({k for m, k, _ in u if m == method})
            }
            for method in ("single", "continuous", "independent_merge")
        },
        "E_by_age": {
            method: {
                str(age): {
                    "all": one.fraction(a for _, a in values),
                    "previously_successful": one.fraction(a for b, a in values if b),
                }
                for (m, age), values in ages.items()
                if m == method
            }
            for method in ("continuous", "independent_merge")
        },
        "primary_t": len(panel["facts"]),
        "analysis_unit": "Fixed facts in one graph; rotations/prefixes are repeated measures",
    }
    return summary, paired


def audit(parent, settings, architecture, source, out, device):
    out, source = Path(out), Path(source)
    reader, world, view, panel = source_context(parent, settings, architecture, source, out, device)
    plans = state_plan(panel)
    summary = json.loads((out / "summary.json").read_text())
    if summary["identity"] != identity(parent, settings, architecture, source, out) or summary[
        "state_keys"
    ] != list(plans):
        raise ValueError("Evaluation identity/state matrix changed")
    actual = {
        str(p.parent.relative_to(out / "states")) for p in (out / "states").rglob("record.json")
    }
    if actual != set(plans):
        raise ValueError("Incomplete or extra states")
    donors = []
    for i, fact in enumerate(panel["facts"]):
        delta, _, _ = source_state(reader, source, f"single/f{i:03d}")
        rebuilt, history = original.write_fact(
            reader, reader.empty_fast(), view, world, fact, device
        )
        if len(history["updates"]) != 1 or fast_io.tensor_digest(rebuilt) != fast_io.tensor_digest(
            delta
        ):
            raise AssertionError("Independent donor is not one update from A0")
        donors.append(delta)
    records, merged, previous_rotation = [], None, None
    for key, spec in plans.items():
        directory = out / "states" / key
        recorded = json.loads((directory / "record.json").read_text())
        if recorded["spec"] != spec or recorded["key"] != key:
            raise ValueError("State provenance changed")
        if spec["kind"] == "independent_merge":
            if spec["rotation"] != previous_rotation:
                merged, previous_rotation = reader.empty_fast(), spec["rotation"]
            merged = add_increment(merged, donors[spec["indices"][-1]])
            fast, provenance = fast_io.load_fast(directory / "fast.pt", reader)
            if provenance != spec or fast_io.tensor_digest(fast) != fast_io.tensor_digest(merged):
                raise AssertionError("Merged state is not the declared sum of independent donors")
            base = original.read_state(
                reader,
                fast,
                world,
                panel,
                spec["indices"],
                [panel["facts"][i]["new_tail"] for i in spec["indices"]],
                device,
            )
            original._compare_readouts(base, recorded["readout"])
        else:
            fast, reference, source_sha = source_state(reader, source, spec["parent_key"])
            if (
                source_sha != recorded["parent_record_sha256"]
                or reference["readout"] != recorded["readout"]
            ):
                raise ValueError("Audited reused predictions changed")
        if fast_io.tensor_digest(fast) != recorded["fast_digest"]:
            raise AssertionError("Recorded state digest differs")
        original._compare_nested(
            read_first(reader, fast, world, panel, spec["indices"], device), recorded["first"]
        )
        records.append(recorded)
    analysis, paired = analyze(panel, records, world.metadata)
    original._compare_nested(analysis, {k: summary[k] for k in analysis})
    original._compare_nested(paired, json.loads((out / "paired-queries.json").read_text()))
    return {
        "passed": True,
        "states": len(plans),
        "independent_donors_rewritten": len(donors),
        "merged_transitions_and_tokens_replayed": True,
        "first_hop_reads_replayed": True,
        "second_hop_control_reused_from_audited_parent": True,
        "new_base_parameter_updates": 0,
    }


def paired_difference(continuous, merged):
    def indexed(rows):
        result = {(tuple(r["query"]), r["rotation"]): r for r in rows}
        if len(result) != len(rows):
            raise ValueError("Duplicate method query/order pairs")
        return result

    old, new = indexed(continuous), indexed(merged)
    if old.keys() != new.keys():
        raise ValueError("Methods must cover identical query/order pairs")
    result = {}
    for condition in ("all", "baseline_correct"):
        differences = []
        for key, reference in old.items():
            if reference["baseline_correct"] != new[key]["baseline_correct"]:
                raise ValueError("Methods disagree about baseline eligibility")
            if condition == "all" or reference["baseline_correct"]:
                differences.append(int(new[key]["new_correct"]) - int(reference["new_correct"]))
        result[condition] = {
            "n": len(differences),
            "merge_minus_continuous_pp": 100 * float(np.mean(differences)) if differences else None,
            "improved": differences.count(1),
            "regressed": differences.count(-1),
            "unchanged": differences.count(0),
        }
    return result


def cross_architecture_report(paths, out):
    if set(paths) != {"transformer", "loop", "discoloop"}:
        raise ValueError("All three architectures are required")
    values = {
        name: json.loads((Path(path) / "paired-queries.json").read_text())
        for name, path in paths.items()
    }
    reports = {}
    for role in ("first", "second"):
        universes = [
            {tuple(r["query"]) for r in data["baseline"][role]} for data in values.values()
        ]
        if any(u != universes[0] for u in universes):
            raise ValueError("Architecture query panels differ")
        common = set.intersection(
            *[
                {tuple(r["query"]) for r in data["baseline"][role] if r["correct"]}
                for data in values.values()
            ]
        )
        reports[role] = {
            "all_queries": len(universes[0]),
            "common_baseline_correct": len(common),
            "coverage": len(common) / len(universes[0]) if universes[0] else None,
            "architectures": {
                name: {
                    method: summarize_rows(r for r in rows if tuple(r["query"]) in common)
                    for method, rows in data["roles"][role].items()
                }
                for name, data in values.items()
            },
        }
    result = {
        "schema": BATCH,
        "roles": reports,
        "new_base_parameter_updates": 0,
        "all_results": {
            name: json.loads((Path(path) / "summary.json").read_text())
            for name, path in paths.items()
        },
        "paired_method_differences": {
            name: {
                role: paired_difference(
                    data["roles"][role]["continuous"], data["roles"][role]["independent_merge"]
                )
                for role in ("first", "second")
            }
            for name, data in values.items()
        },
    }
    write_json(Path(out) / "comparison.json", result)
    return result
