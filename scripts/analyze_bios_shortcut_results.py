"""Aggregate the complete P3 matrix without changing the fixed-person cohorts.

Keeps world, initialization, organization and query relation in the endpoint cells;
world-level tables are descriptive summaries, not independent-query inference.
"""

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "results/bios-mechanism-dev-v1/p3-summary"
SOURCES = (
    "audit.json",
    "sources.json",
    "paired-learning.csv",
    "paired-two-step.csv",
    "learning-strata.csv",
    "learning-metrics.csv",
    "organization-effects.csv",
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(source, name):
    with (source / name).open() as stream:
        return list(csv.DictReader(stream))


def write(output, name, rows):
    with (output / name).open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def key(row):
    return tuple(row[k] for k in ("world", "seed", "condition", "chain", "cohort"))


def mean(values):
    present = [float(value) for value in values if value is not None and value != ""]
    return sum(present) / len(present) if present else None


def aggregate(cells, keys):
    groups = defaultdict(list)
    for row in cells:
        for world in ("all", row["world"]):
            identity = {**row, "world": world}
            groups[tuple(identity[k] for k in keys)].append(row)
    result = []
    metrics = (
        "low_direct",
        "high_direct",
        "direct_change",
        "low_two_step",
        "high_two_step",
        "two_step_change",
        "low_gap",
        "high_gap",
        "gap_change",
        "low_actual_on_conflict",
        "high_actual_on_conflict",
    )
    for values, rows in sorted(groups.items()):
        result.append(
            {
                **dict(zip(keys, values, strict=True)),
                "cases": len(rows),
                "n": sum(row["n"] for row in rows),
                **{metric: mean(row[metric] for row in rows) for metric in metrics},
                "positive_direct_cells": sum(row["direct_change"] > 0 for row in rows),
                "negative_direct_cells": sum(row["direct_change"] < 0 for row in rows),
                "min_direct_change": min(row["direct_change"] for row in rows),
                "max_direct_change": max(row["direct_change"] for row in rows),
            }
        )
    return result


def analyze(source=DEFAULT):
    source = Path(source)
    audit = json.loads((source / "audit.json").read_text())
    expected = {
        "complete": True,
        "complete_pairs": 12,
        "learning_pairs": 72,
        "two_step_pairs": 24,
        "edit_cases": 0,
        "missing": [],
        "incomplete_organization_blocks": [],
    }
    if any(audit.get(k) != v for k, v in expected.items()):
        raise ValueError("P3 interpretation requires the complete audited paired matrix")
    hashes = {name: sha(source / name) for name in SOURCES}
    metadata = source / "interpretation-audit.json"
    if metadata.exists() and json.loads(metadata.read_text())["sources_hashes"] != hashes:
        raise ValueError("P3 source summaries changed after interpretation")
    paired = [
        r
        for r in read(source, "paired-learning.csv")
        if r["step"] == "15360" and r["split"] == "heldout"
    ]
    two = {key(r): r for r in read(source, "paired-two-step.csv") if r["split"] == "heldout"}
    attraction = {
        (key(r), r["phase"]): r
        for r in read(source, "learning-strata.csv")
        if r["step"] == "15360" and r["split"] == "heldout" and r["cohort_kind"] == "fixed"
    }
    cells = []
    for row in paired:
        other = two[key(row)]
        if row["n"] != other["n"]:
            raise ValueError("Direct/two-step fixed-person denominators differ")
        low, high = float(row["low_accuracy"]), float(row["high_accuracy"])
        low_two, high_two = float(other["low_accuracy"]), float(other["high_accuracy"])
        cells.append(
            {
                **{k: row[k] for k in ("world", "seed", "condition", "chain", "cohort")},
                "n": int(row["n"]),
                "low_direct": low,
                "high_direct": high,
                "direct_change": high - low,
                "low_two_step": low_two,
                "high_two_step": high_two,
                "two_step_change": high_two - low_two,
                "low_gap": low_two - low,
                "high_gap": high_two - high,
                "gap_change": (high_two - high) - (low_two - low),
                **{
                    f"{phase}_actual_on_conflict": float(
                        attraction[key(row), phase]["actual_on_conflict_rate"]
                    )
                    if attraction[key(row), phase]["actual_on_conflict_rate"]
                    else None
                    for phase in ("low", "high")
                },
            }
        )
    if len(cells) != 96 or len({key(row) for row in cells}) != 96:
        raise ValueError("Endpoint cells must retain all 24 cases and four fixed cohorts")
    output = {
        "endpoint-cells.csv": cells,
        "endpoint-by-world.csv": aggregate(cells, ("world", "cohort")),
        "endpoint-by-organization.csv": aggregate(cells, ("world", "condition", "chain", "cohort")),
    }
    base = [r for r in read(source, "learning-metrics.csv") if r["step"] == "15360"]
    if len(base) != 24:
        raise ValueError("Expected all 12 model pairs in both prevalence conditions")
    metrics = (
        "base_accuracy",
        "base_nll",
        "company_membership",
        "project_membership",
        "company_default",
        "project_default",
        "company_actual",
        "project_actual",
    )
    groups = defaultdict(list)
    for row in base:
        for world in ("all", row["world"]):
            groups[world, row["phase"]].append(row)
    output["base-facts-by-world.csv"] = [
        {
            "world": world,
            "phase": phase,
            "models": len(rows),
            **{metric: mean(row[metric] for row in rows) for metric in metrics},
        }
        for (world, phase), rows in sorted(groups.items())
    ]
    groups = defaultdict(dict)
    for row in read(source, "organization-effects.csv"):
        if row["step"] != "15360" or row["split"] != "heldout":
            continue
        identity = tuple(row[k] for k in ("world", "seed", "method", "cohort"))
        groups[identity][row["phase"]] = row
    interactions = []
    for (world, seed, method, cohort), phases in sorted(groups.items()):
        if set(phases) != {"low", "high"}:
            raise ValueError("Incomplete organization interaction pair")
        interactions.append(
            {
                "world": world,
                "seed": seed,
                "method": method,
                "cohort": cohort,
                **{
                    f"{phase}_{metric}": float(phases[phase][metric])
                    for phase in ("low", "high")
                    for metric in (
                        "matching_effect",
                        "company_mean_benefit",
                        "project_mean_benefit",
                    )
                },
            }
        )
    output["organization-endpoint-cells.csv"] = interactions
    indexed = {
        (r["world"], r["cohort"], r["condition"], r["chain"]): r
        for r in output["endpoint-by-organization.csv"]
    }
    unmatched = []
    for world in ("all", "0", "1"):
        for cohort in ("all", "original_exception", "newly_exception", "remaining_ordinary"):
            for phase in ("low", "high"):
                metric = f"{phase}_direct"
                company = (
                    indexed[world, cohort, "company", "project"][metric]
                    - indexed[world, cohort, "neither", "project"][metric]
                )
                project = (
                    indexed[world, cohort, "project", "company"][metric]
                    - indexed[world, cohort, "neither", "company"][metric]
                )
                unmatched.append(
                    {
                        "world": world,
                        "cohort": cohort,
                        "phase": phase,
                        "company_unmatched_benefit": company,
                        "project_unmatched_benefit": project,
                        "mean_unmatched_benefit": (company + project) / 2,
                    }
                )
    output["organization-unmatched-effects.csv"] = unmatched
    for name, digest in hashes.items():
        if sha(source / name) != digest:
            raise ValueError(f"P3 source changed during interpretation: {name}")
    for name, rows in output.items():
        write(source, name, rows)
    metadata.write_text(
        json.dumps(
            {
                "complete": True,
                "sources_hashes": hashes,
                "script_sha256": sha(Path(__file__)),
                "endpoint_cells": 96,
                "models_per_phase": 12,
                "interpretation": "Fixed-person heldout cohorts and full original matrix. "
                "Actual-match rates are undefined when no actual/default conflict exists, "
                "not zero. World/seed/query-relation cells are retained; "
                "no independent-query uncertainty claim. "
                "A smaller two-step gap alone is insufficient if two-step accuracy deteriorates.",
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"complete": True, "endpoint_cells": len(cells), "outputs": list(output)}))


def analyze_components(source=DEFAULT):
    """Measure prerequisite facts on the same heldout people, without filtering QA."""
    import numpy as np

    from llm_memory_editability.bios_cross import CHAINS, CONDITIONS, make_cross_world

    source = Path(source)
    if not json.loads((source / "audit.json").read_text())["complete"]:
        raise ValueError("Component analysis requires a complete source audit")
    hashes = {"audit.json": sha(source / "audit.json")}
    rows = []
    for w in (0, 1):
        world = make_cross_world(w, ROOT / "data/bios-organization-v1")
        for seed in (0, 1):
            for condition in CONDITIONS:
                name = f"queries/world-{w}-seed-{seed}-{condition}-learning-15360.npz"
                path = source / name
                hashes[name] = sha(path)
                with np.load(path) as saved:
                    a = dict(saved)
                np.testing.assert_array_equal(a["low_truth"], world.answers)
                np.testing.assert_array_equal(a["derived_ids"], world.derived_ids)
                for phase in ("low", "high"):
                    expected = (a[f"{phase}_prediction"] == a[f"{phase}_truth"]) & a[
                        f"{phase}_ended"
                    ]
                    np.testing.assert_array_equal(expected, a[f"{phase}_correct"])
                for chain, label in enumerate(CHAINS):
                    heldout = world.person[world.heldout_ids[chain]]
                    for cohort, code in (
                        ("all", None),
                        ("original_exception", 1),
                        ("newly_exception", 2),
                        ("remaining_ordinary", 0),
                    ):
                        people = (
                            heldout
                            if code is None
                            else heldout[a["person_cohort"][chain, heldout] == code]
                        )
                        n = len(people)
                        expected_n = {None: 1024, 1: 64, 2: 448, 0: 512}[code]
                        if n != expected_n:
                            raise ValueError("Fixed-person component denominator changed")
                        independent = np.flatnonzero(
                            np.isin(world.relation, [3, 4, 5, 6]) & np.isin(world.person, people)
                        )
                        if len(independent) != 4 * n:
                            raise ValueError("Independent-attribute fact count changed")
                        for phase in ("low", "high"):
                            correct = a[f"{phase}_correct"]
                            membership = correct[world.membership_ids[chain, people]]
                            default = correct[
                                world.root_ids[chain, world.memberships[chain, people]]
                            ]
                            rows.append(
                                {
                                    "world": str(w),
                                    "seed": seed,
                                    "condition": condition,
                                    "chain": label,
                                    "cohort": cohort,
                                    "phase": phase,
                                    "n": n,
                                    "membership": float(membership.mean()),
                                    "default": float(default.mean()),
                                    "both_prerequisites": float((membership & default).mean()),
                                    "actual": float(
                                        correct[world.actual_ids[chain, people]].mean()
                                    ),
                                    "independent_attributes": float(correct[independent].mean()),
                                }
                            )
    if len(rows) != 192:
        raise ValueError("Missing fixed-person component cells")
    previous = source / "component-audit.json"
    if previous.exists() and json.loads(previous.read_text())["sources_hashes"] != hashes:
        raise ValueError("Component source identity changed")
    for name, digest in hashes.items():
        if sha(source / name) != digest:
            raise ValueError("Component source changed during analysis")
    groups = defaultdict(list)
    for row in rows:
        for w in ("all", row["world"]):
            groups[w, row["cohort"], row["phase"]].append(row)
    metrics = ("membership", "default", "both_prerequisites", "actual", "independent_attributes")
    aggregated = [
        {
            "world": w,
            "cohort": cohort,
            "phase": phase,
            "cases": len(group),
            "n": sum(r["n"] for r in group),
            **{metric: mean(r[metric] for r in group) for metric in metrics},
        }
        for (w, cohort, phase), group in sorted(groups.items())
    ]
    write(source, "component-endpoint-cells.csv", rows)
    write(source, "components-by-world.csv", aggregated)
    previous.write_text(
        json.dumps(
            {
                "complete": True,
                "cells": len(rows),
                "sources_hashes": hashes,
                "script_sha256": sha(Path(__file__)),
                "interpretation": (
                    "Same fixed heldout people in low/high conditions; no filtering by "
                    "correctness. Default facts are indexed using the true organization "
                    "only for separate knowledge measurement, never supplied as the "
                    "bridge in autonomous two-step inference."
                ),
            },
            indent=2,
        )
        + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT)
    args = parser.parse_args()
    analyze(args.source)
    analyze_components(args.source)


if __name__ == "__main__":
    main()
