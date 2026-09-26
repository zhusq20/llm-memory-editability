"""Audit saved edit predictions and describe failures without selecting successes."""

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .bios_data import N_BASE, RELATIONS, load_world, write_json

RETENTION_GROUPS = ("0", "1", "2", "3", "2_base", "2_derived")


def read(path):
    return json.loads(path.read_text())


def failure_flags(point, accuracy=0.95, damage=0.01):
    rates = [point["U_heldout_destruction"][g]["rate"] for g in RETENTION_GROUPS]
    return {
        "E": point["E_root"] < 1 or point["E_member"] < accuracy,
        "D": point["D"] < accuracy,
        "U": any(r is not None and r > damage for r in rates),
        "unevaluable": any(r is None for r in rates),
    }


def failure_signature(flags):
    if flags["unevaluable"]:
        return "unevaluable"
    return "+".join(k for k in ("E", "D", "U") if flags[k]) or "pass"


def propagation_errors(world, sets, prediction):
    ids = sets["D"]
    value, ended = prediction["prediction"][ids], prediction["ended"][ids]
    target = sets["target"][ids]
    actual = sets["target"][2112 + world.person[ids]]
    correct = (value == target) & ended
    remaining = ~correct
    counts = {"correct": int(correct.sum())}
    for name, mask in (
        ("nontermination", ~ended),
        ("old_default", value == world.answers[ids]),
        ("personal_city_instead_of_company_default", value == actual),
    ):
        counts[name] = int((remaining & mask).sum())
        remaining &= ~mask
    counts["other_wrong_value"] = int(remaining.sum())
    assert sum(counts.values()) == len(ids)
    strata = {}
    old_exception = world.exceptions[world.person[ids]]
    for name, mask in (
        ("old_exception", old_exception),
        ("updated_personal_equals_default", ~old_exception & (actual == target)),
        ("updated_personal_differs_from_default", ~old_exception & (actual != target)),
    ):
        strata[name] = {"n": int(mask.sum()), "correct": int(correct[mask].sum())}
    return {"counts": counts, "strata": strata}


def retention_masks(sets, old_correct, view="heldout"):
    eligible = old_correct.copy()
    if view != "full":
        selected = np.zeros(len(eligible), dtype=bool)
        selected[sets[view]] = True
        eligible &= selected
    masks = {str(g): eligible & (sets["strata"] == g) for g in range(4)}
    masks["2_base"] = masks["2"] & (np.arange(len(eligible)) < N_BASE)
    masks["2_derived"] = masks["2"] & (np.arange(len(eligible)) >= N_BASE)
    return masks


def retention_counts(sets, correct, old_correct, view="heldout"):
    return {
        group: {
            "known": int(mask.sum()),
            "broken": int((mask & ~correct).sum()),
            "rate": float((~correct[mask]).mean()) if mask.any() else None,
        }
        for group, mask in retention_masks(sets, old_correct, view).items()
    }


def dump_csv(path, rows):
    if not rows:
        return
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize_group(cases):
    n = len(cases)
    result = {"n": n}
    for field in ("E", "E_root", "E_member", "D", "U_accuracy", "E_target_value_nll"):
        result[f"mean_{field}"] = float(np.mean([c["final"][field] for c in cases]))
    result.update(
        final_pass=sum(c["final"]["joint_pass"] is True for c in cases),
        ever_pass=sum(c["first_observed_joint_step"] is not None for c in cases),
        regressed=sum(
            c["first_observed_joint_step"] is not None and not c["final"]["joint_pass"]
            for c in cases
        ),
        signatures=dict(Counter(c["failure_signature"] for c in cases)),
        **{f"fail_{k}": sum(c["failure_flags"][k] for c in cases) for k in ("E", "D", "U")},
    )
    for g in RETENTION_GROUPS:
        rates = [c["final"]["U_heldout_destruction"][g]["rate"] for c in cases]
        result[f"fail_U{g}"] = sum(r is not None and r > 0.01 for r in rates)
        result[f"mean_U{g}_damage"] = float(np.mean([r for r in rates if r is not None]))
    result["D_error_counts"] = dict(
        sum((Counter(c["propagation"]["counts"]) for c in cases), Counter())
    )
    result["mean_initial_new_answer_nll"] = float(np.mean([c["initial_nll"] for c in cases]))
    return result


def run(args):
    root, out = Path(args.root), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    worlds = {i: load_world(f"data/bios-work-v1/world-{i}") for i in (0, 1)}
    cases, audit, timelines = [], [], []
    lookup = {}
    for run_dir in sorted(root.glob("world-*-seed-*-*")):
        config = read(run_dir / "config.json")
        world = worlds[config["world_seed"]]
        for dirname, window in (("edits", 3), ("edits-window0", 0)):
            completed = read(run_dir / dirname / "complete.json")
            assert completed["status"] == "complete" and len(completed["cases"]) == 16
            for path in sorted((run_dir / dirname).glob("*/complete.json")):
                c = read(path)
                assert c["status"] == "complete" and c["final"]["step"] == 512
                assert c["window_zero_based"] == list(range(window, window + 3))
                sets = dict(np.load(path.parent / "sets.npz"))
                pred = dict(np.load(path.parent / "predictions-512.npz"))
                correct = (pred["prediction"] == sets["target"]) & pred["ended"]
                np.testing.assert_array_equal(correct, pred["correct"])
                retention = retention_counts(sets, correct, sets["old_correct"])
                assert retention == c["final"]["U_heldout_destruction"]
                for name, ids in (
                    ("E", sets["E"]),
                    ("D", sets["D"]),
                    ("E_root", sets["E"][world.relation[sets["E"]] == 1]),
                    ("E_member", sets["E"][world.relation[sets["E"]] == 2]),
                ):
                    assert float(correct[ids].mean()) == c["final"][name]
                trajectory = read(path.parent / "trajectory.json")
                for point in trajectory:
                    flags = failure_flags(point)
                    expected = None if flags["unevaluable"] else not any(flags.values())
                    assert point["joint_pass"] == expected
                flags = failure_flags(c["final"])
                damaged_by_relation = {}
                for group, mask in retention_masks(sets, sets["old_correct"]).items():
                    broken = mask & ~correct
                    damaged_by_relation[group] = {
                        name: int((broken & (world.relation == relation)).sum())
                        for relation, name in enumerate((*RELATIONS, "derived_default_city"))
                    }
                    assert sum(damaged_by_relation[group].values()) == int(broken.sum())
                method = f"FT-{c['scope'].upper()}"
                if c["branch"]:
                    method += f"+{c['branch']}"
                meta = {
                    "run": run_dir.name,
                    "world": world.seed,
                    "seed": config["seed"],
                    "order": config["order"],
                    "window": window,
                    "method": method,
                    "kind": c["kind"],
                    "support": c["support"],
                }
                c.update(
                    meta,
                    directory=str(path.parent),
                    failure_flags=flags,
                    failure_signature=failure_signature(flags),
                    heldout_damage_by_relation=damaged_by_relation,
                    propagation=propagation_errors(world, sets, pred),
                    replay_retention=retention_counts(sets, correct, sets["old_correct"], "replay"),
                    initial_nll=trajectory[0]["E_target_value_nll"],
                    ever_E=any(not failure_flags(p)["E"] for p in trajectory),
                    ever_D=any(not failure_flags(p)["D"] for p in trajectory),
                    ever_U=any(not failure_flags(p)["U"] for p in trajectory),
                    last_failure_free_E_step=max(
                        (p["step"] for p in trajectory if not failure_flags(p)["E"]), default=None
                    ),
                    down_calibration=c["final"].get("down_calibration"),
                )
                cases.append(c)
                lookup[(run_dir.name, window, method, c["kind"], c["support"])] = (c, sets, pred)
                audit.append(
                    {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                )
                timelines.extend(
                    {**meta, **p, "failure_flags": failure_flags(p)} for p in trajectory
                )
    assert len(cases) == 256
    groups = defaultdict(list)
    for c in cases:
        groups[(c["window"], c["method"], c["order"], c["kind"])].append(c)
    aggregates = [
        {"window": w, "method": m, "order": o, "kind": k, **summarize_group(cs)}
        for (w, m, o, k), cs in sorted(groups.items())
    ]
    common, duplicates, branch_pairs = [], [], []
    for (run_name, window, method, kind, support), (case, sets, pred) in lookup.items():
        if case["order"] == "SA":
            other, other_sets, other_pred = lookup[
                (run_name[:-2] + "AS", window, method, kind, support)
            ]
            for field in (
                "E",
                "D",
                "target",
                "heldout",
                "replay",
                "strata",
                "edit_sampling",
                "replay_sampling",
            ):
                np.testing.assert_array_equal(sets[field], other_sets[field])
            known = sets["old_correct"] & other_sets["old_correct"]
            sa = retention_counts(sets, pred["correct"], known)
            ass = retention_counts(sets, other_pred["correct"], known)
            common.append(
                {
                    "world": case["world"],
                    "seed": case["seed"],
                    "window": window,
                    "method": method,
                    "kind": kind,
                    "support": support,
                    "SA": sa,
                    "AS": ass,
                }
            )
        if method == "FT-ALL" and window == 3:
            _, _, duplicate = lookup[(run_name, 0, method, kind, support)]
            duplicates.append(
                {
                    "run": run_name,
                    "kind": kind,
                    "support": support,
                    "identical_final_predictions": bool(
                        all(np.array_equal(pred[k], duplicate[k]) for k in pred)
                    ),
                }
            )
        if case["branch"]:
            baseline, _, baseline_pred = lookup[(run_name, window, "FT-MLP", kind, support)]
            branch_pairs.append(
                {
                    "run": run_name,
                    "window": window,
                    "method": method,
                    "kind": kind,
                    "support": support,
                    **{f"delta_{k}": case["final"][k] - baseline["final"][k] for k in ("E", "D")},
                    "baseline_joint": baseline["final"]["joint_pass"],
                    "branch_joint": case["final"]["joint_pass"],
                    "changed_predictions_vs_baseline": int(
                        (pred["prediction"] != baseline_pred["prediction"]).sum()
                    ),
                    "max_value_nll_difference_vs_baseline": float(
                        np.max(np.abs(pred["value_nll"] - baseline_pred["value_nll"]))
                    ),
                }
            )
    sensitivity = []
    for (window, method, order, kind), cs in sorted(groups.items()):
        for accuracy in (0.90, 0.95, 0.99):
            for damage in (0.0, 0.01, 0.02, 0.05):
                sensitivity.append(
                    {
                        "window": window,
                        "method": method,
                        "order": order,
                        "kind": kind,
                        "accuracy_threshold": accuracy,
                        "damage_threshold": damage,
                        "n": len(cs),
                        "final_pass": sum(
                            not any(failure_flags(c["final"], accuracy, damage).values())
                            for c in cs
                        ),
                    }
                )
    quality = []
    for world in (0, 1):
        for seed in (0, 1):
            sa, ass = [
                read(root / f"world-{world}-seed-{seed}-{order}" / "complete.json")["final"]
                for order in ("SA", "AS")
            ]
            quality.append(
                {
                    "world": world,
                    "seed": seed,
                    "base_difference_pp": 100 * (sa["base_accuracy"] - ass["base_accuracy"]),
                    "derived_difference_pp": 100
                    * (sa["derived_accuracy"] - ass["derived_accuracy"]),
                    "passed": bool(
                        min(
                            sa["base_accuracy"],
                            ass["base_accuracy"],
                            sa["derived_accuracy"],
                            ass["derived_accuracy"],
                        )
                        >= 0.99
                        and abs(sa["base_accuracy"] - ass["base_accuracy"]) <= 0.005
                        and abs(sa["derived_accuracy"] - ass["derived_accuracy"]) <= 0.005
                        and abs(sa["value_nll"] - ass["value_nll"]) <= 0.1
                    ),
                }
            )
    quality_keys = {(q["world"], q["seed"]) for q in quality if q["passed"]}
    quality_aggregates = []
    for (w, m, o, k), cs in sorted(groups.items()):
        subset = [c for c in cs if (c["world"], c["seed"]) in quality_keys]
        quality_aggregates.append(
            {"window": w, "method": m, "order": o, "kind": k, **summarize_group(subset)}
        )
    report = {
        "phase": "symbolic development; descriptive; not independent edit replicates",
        "cases": cases,
        "groups": aggregates,
        "common_known_retention": common,
        "branch_paired_support0": branch_pairs,
        "ft_all_window_duplicates": duplicates,
        "quality_pairs": quality,
        "quality_subset_groups": quality_aggregates,
        "sensitivity": sensitivity,
        "source_audit": audit,
        "expected_cases": 256,
        "audited_cases": len(cases),
    }
    write_json(out / "failure-analysis.json", report)
    write_json(out / "edit-trajectories.json", timelines)
    dump_csv(
        out / "failure-summary.csv",
        [{k: v for k, v in row.items() if not isinstance(v, dict)} for row in aggregates],
    )
    dump_csv(out / "threshold-sensitivity.csv", sensitivity)
    dump_csv(out / "branch-paired-support0.csv", branch_pairs)
    dump_csv(
        out / "case-failures.csv",
        [
            {
                **{
                    k: c[k]
                    for k in (
                        "run",
                        "world",
                        "seed",
                        "order",
                        "window",
                        "method",
                        "kind",
                        "support",
                        "failure_signature",
                        "first_observed_joint_step",
                        "initial_nll",
                    )
                },
                **{k: c["final"][k] for k in ("E", "E_root", "E_member", "D", "joint_pass")},
            }
            for c in cases
        ],
    )
    print(
        json.dumps(
            {
                "audited_cases": len(cases),
                "groups": len(aggregates),
                "quality_pairs_passed": len(quality_keys),
                "FT_ALL_identical_duplicate_cases": sum(
                    d["identical_final_predictions"] for d in duplicates
                ),
            }
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="results/bios-dev-v1")
    parser.add_argument("--output", default="results/bios-dev-v1/failure-analysis-20260926")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
