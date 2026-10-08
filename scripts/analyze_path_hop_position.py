"""Pre-registered hop-position analysis of the MQuAKE and RippleEdits reproductions.

The contract is configs/path-hop-analysis-v1.json, frozen before any hop-position
outcome was read. ``counts`` reads benchmark data only. ``analyze`` reads only a
completed reproduction run whose independent replay audit passed; it reuses the
reproductions' frozen scores and computes no model outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from llm_memory_editability.mquake_reproduction import phase_scores
from llm_memory_editability.paper_editing_runtime import exact_alias_match
from llm_memory_editability.ripple_reproduction import AXES

PROJECT = Path(__file__).resolve().parents[1]
CONFIG = PROJECT / "configs/path-hop-analysis-v1.json"
RIPPLE_HOP_AXES = {"CI": "Compositionality_I", "CII": "Compositionality_II"}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mean(values):
    return sum(values) / len(values) if values else None


# MQuAKE ---------------------------------------------------------------------


def mquake_edit_positions(case):
    """1-based hop of each requested rewrite in the new chain (reproduction's matching rule)."""
    chain = list(zip(case["orig"]["new_triples"], case["orig"]["new_triples_labeled"], strict=True))
    positions = []
    for request in case["requested_rewrite"]:
        hits = [
            index + 1
            for index, (triple, labels) in enumerate(chain)
            if labels[0] == request["subject"]
            and triple[1] == request["relation_id"]
            and triple[2] == request["target_new"]["id"]
        ]
        assert len(hits) == 1, (case["case_id"], request["subject"], hits)
        positions.append(hits[0])
    return positions


def mquake_group(case):
    """Primary grouping: single-edit cases, edit at hop 1 or at a later hop."""
    positions = mquake_edit_positions(case)
    if len(positions) != 1:
        return None
    return "first" if positions[0] == 1 else "later"


def mquake_counts(cases):
    cells = {}
    for case in cases:
        positions = mquake_edit_positions(case)
        if len(positions) == 1:
            key = f"{len(case['single_hops'])}-hop/edited-hop-{positions[0]}"
        else:
            key = f"{len(case['single_hops'])}-hop/{len(positions)}-edits/" + (
                "includes-hop-1" if 1 in positions else "no-hop-1"
            )
        cells[key] = cells.get(key, 0) + 1
    return dict(sorted(cells.items()))


def mquake_row(case, record):
    assert record["case_id"] == case["case_id"]
    assert (
        record["case_sha256"]
        == hashlib.sha256(json.dumps(case, sort_keys=True).encode()).hexdigest()
    )
    baseline = phase_scores(record["baseline"]["predictions"])
    edited = phase_scores(record["edited"]["predictions"])
    old = [case["answer"], *case["answer_alias"]]
    new = [case["new_answer"], *case["new_answer_alias"]]
    distinct = not (set(old) & set(new))
    return {
        "hops": len(case["single_hops"]),
        "positions": mquake_edit_positions(case),
        "group": mquake_group(case),
        "direct": edited["multi"],
        "cot": edited["cot"],
        "baseline_direct": baseline["multi"],
        "prerequisites": baseline["multi"] and edited["instance"] and all(edited["edit"]),
        "old_answer": any(
            exact_alias_match(row["answer_text"], old)
            for row in record["edited"]["predictions"]["multi"]
        )
        if distinct
        else None,
    }


def mquake_gaps(rows, strata):
    """Equal-weight mean over hop-count strata of first-minus-later accuracy."""
    result = {}
    for metric in ("direct", "cot"):
        per_stratum = []
        for hops in strata:
            first = [r[metric] for r in rows if r["hops"] == hops and r["group"] == "first"]
            later = [r[metric] for r in rows if r["hops"] == hops and r["group"] == "later"]
            if not first or not later:
                return None
            per_stratum.append(mean(first) - mean(later))
        result[metric] = mean(per_stratum)
    result["interaction"] = result["direct"] - result["cot"]
    return result


def bootstrap(rows, statistic, spec, strata_key=None):
    """Percentile interval; cases are resampled jointly within each stratum."""
    rng = random.Random(spec["seed"])
    groups = {}
    for row in rows:
        groups.setdefault(row[strata_key] if strata_key else None, []).append(row)
    if not groups:
        return {}
    draws = {}
    for _ in range(spec["iterations"]):
        sample = [rng.choice(members) for members in groups.values() for _ in members]
        value = statistic(sample)
        if value is None:
            continue
        for key, number in value.items():
            draws.setdefault(key, []).append(number)
    low, high = spec["interval"]
    intervals = {}
    for key, values in draws.items():
        values.sort()
        intervals[key] = {
            "low": values[int(low * (len(values) - 1))],
            "high": values[int(high * (len(values) - 1))],
            "valid_draws": len(values),
        }
    return intervals


def cell_summary(rows):
    summary = {}
    for hops in sorted({r["hops"] for r in rows}):
        for group in ("first", "later"):
            cell = [r for r in rows if r["hops"] == hops and r["group"] == group]
            if not cell:
                continue
            retained = [r["old_answer"] for r in cell if r["old_answer"] is not None]
            summary[f"{hops}-hop/{group}"] = {
                "cases": len(cell),
                "direct": mean([r["direct"] for r in cell]),
                "cot": mean([r["cot"] for r in cell]),
                "baseline_direct": mean([r["baseline_direct"] for r in cell]),
                "old_answer_retained": mean(retained),
                "old_answer_cases": len(retained),
            }
    return summary


def analyze_mquake(config, run_dir, data_file):
    spec = config["mquake"]
    assert sha(data_file) == spec["data_sha256"]
    cases = read(data_file)[: spec["max_cases"]]
    rows = [
        mquake_row(case, read(run_dir / "cases" / f"{index:05d}" / "record.json"))
        for index, case in enumerate(cases, 1)
    ]
    single = [r for r in rows if r["group"] is not None]
    strata = spec["primary_hop_counts"]
    primary = [r for r in single if r["hops"] in strata]
    prerequisite = [r for r in primary if r["prerequisites"]]
    multi = {}
    for r in rows:
        if r["group"] is None:
            key = "includes-hop-1" if 1 in r["positions"] else "no-hop-1"
            multi.setdefault(key, []).append(r["direct"])
    return {
        "primary_full_pool": {
            "estimate": mquake_gaps(primary, strata),
            "interval": bootstrap(
                primary, lambda s: mquake_gaps(s, strata), config["bootstrap"], "hops"
            ),
            "cells": cell_summary(primary),
        },
        "secondary_prerequisite_subset": {
            "coverage": len(prerequisite) / len(primary),
            "estimate": mquake_gaps(prerequisite, strata),
            "interval": bootstrap(
                prerequisite, lambda s: mquake_gaps(s, strata), config["bootstrap"], "hops"
            ),
            "cells": cell_summary(prerequisite),
        },
        "descriptive_other_single_edit": cell_summary(
            [r for r in single if r["hops"] not in strata]
        ),
        "descriptive_multi_edit_direct": {
            key: {"cases": len(values), "direct": mean(values)} for key, values in multi.items()
        },
    }


# RippleEdits ----------------------------------------------------------------


def ripple_structure(case, axis):
    """Per test: does the edited fact sit at the hop that this axis is meant to probe?"""
    subject = case["edit"]["subject_id"]
    checks = []
    for test in case[RIPPLE_HOP_AXES[axis]]:
        queries = test["test_queries"]
        two_hop = all(q["query_type"] == "two_hop" for q in queries)
        if axis == "CI":
            valid = two_hop and all(q["subject_id"] == subject for q in queries)
        else:
            valid = two_hop and all(
                subject in q["target_ids"] and q["subject_id"] != subject for q in queries
            )
        checks.append(valid)
    return checks


def ripple_counts(cases):
    counts = {}
    for axis in RIPPLE_HOP_AXES:
        checks = [ok for case in cases for ok in ripple_structure(case, axis)]
        counts[axis] = {
            "cases": sum(bool(case[RIPPLE_HOP_AXES[axis]]) for case in cases),
            "tests": len(checks),
            "structure_valid_tests": sum(checks),
        }
    counts["cases_with_both"] = sum(
        bool(case["Compositionality_I"] and case["Compositionality_II"]) for case in cases
    )
    return counts


def ripple_case(record):
    """Passed/executed counts of structure-valid tests, only where the edit succeeded."""
    counts = {}
    if record.get("excluded"):
        return counts
    for axis, field in RIPPLE_HOP_AXES.items():
        assert AXES[field] == axis
        result = record["axes"][axis]["result"]
        if not result["edit_success"]:
            continue
        valid = ripple_structure(record["case"], axis)
        passed = sum(valid[i] for i in result["outcomes"].get("PASSED", []))
        failed = sum(valid[i] for i in result["outcomes"].get("FAILED", []))
        if passed + failed:
            counts[axis] = (passed, passed + failed)
    return counts


def ripple_gap(sample):
    accuracy = {}
    for axis in RIPPLE_HOP_AXES:
        passed = sum(case[axis][0] for case in sample if axis in case)
        total = sum(case[axis][1] for case in sample if axis in case)
        if not total:
            return None
        accuracy[axis] = passed / total
    return {**accuracy, "CI_minus_CII": accuracy["CI"] - accuracy["CII"]}


def analyze_ripple(config, run_dir, data_file, data_name):
    spec = config["rippleedits"]
    assert sha(data_file) == spec["data_sha256"][data_name]
    cases = read(data_file)
    by_case = []
    for index, case in enumerate(cases, 1):
        record = read(run_dir / "cases" / f"{index:05d}" / "record.json")
        assert record["source_index"] == index - 1 and record["case"] == case
        counts = ripple_case(record)
        if counts:
            by_case.append(counts)
    paired = [
        c["CI"][0] / c["CI"][1] - c["CII"][0] / c["CII"][1]
        for c in by_case
        if set(c) == set(RIPPLE_HOP_AXES)
    ]
    return {
        "primary_test_level": {
            "estimate": ripple_gap(by_case),
            "interval": bootstrap(by_case, ripple_gap, config["bootstrap"]),
            "executed_valid_tests": {
                axis: sum(c[axis][1] for c in by_case if axis in c) for axis in RIPPLE_HOP_AXES
            },
            "cases": len(by_case),
        },
        "descriptive_within_case": {"cases": len(paired), "mean_CI_minus_CII": mean(paired)},
    }


# Entry point ----------------------------------------------------------------


def require_audited(run_dir):
    complete, audit = run_dir / "complete.json", run_dir / "audit.json"
    if not complete.exists() or not audit.exists() or not read(audit).get("passed"):
        raise SystemExit(
            f"{run_dir}: complete.json and a passed audit.json are required; "
            "the frozen analysis does not read interim records"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["counts", "analyze"])
    parser.add_argument("--benchmark", choices=["mquake", "rippleedits"], required=True)
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = read(CONFIG)
    data_name = args.data_file.name
    if args.action == "counts":
        cases = read(args.data_file)
        result = mquake_counts(cases) if args.benchmark == "mquake" else ripple_counts(cases)
    else:
        assert args.run_dir and args.output, "analyze needs --run-dir and --output"
        require_audited(args.run_dir)
        run_name = read(args.run_dir / "run.json")["spec"]["name"]
        assert run_name in config[args.benchmark]["runs"], run_name
        if args.benchmark == "mquake":
            result = analyze_mquake(config, args.run_dir, args.data_file)
        else:
            result = analyze_ripple(config, args.run_dir, args.data_file, data_name)
        result = {
            "contract": "path-hop-analysis-v1",
            "contract_sha256": sha(CONFIG),
            "run": run_name,
            "data_sha256": sha(args.data_file),
            **result,
        }
    text = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
