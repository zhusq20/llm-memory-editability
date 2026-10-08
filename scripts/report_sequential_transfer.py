"""Summarize partial sequential-transfer batches without choosing on B composition."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

POOLS = ("atomic_A", "atomic_B", "train_composition", "AA", "BA", "AB", "BB")
HISTORY_ORDER = (
    "sequential_composition",
    "sequential_atomic",
    "joint",
    "sequential_composition_sham",
)


def read(path, default=None):
    return json.loads(Path(path).read_text()) if Path(path).exists() else default


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def snapshot_reporter(out):
    """Keep the exact analysis source beside its derived report."""
    script = Path(__file__).resolve()
    tests = script.parents[1] / "tests/test_sequential_transfer_report.py"
    sources = {}
    for original in (script, tests):
        if not original.exists():
            continue
        destination = Path(out) / "source" / original.name
        payload = original.read_bytes()
        if destination.exists() and destination.read_bytes() != payload:
            raise FileExistsError("Use a new report directory for changed analysis source")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
        sources[str(destination)] = hashlib.sha256(payload).hexdigest()
    return sources


def metric(record, pool, predictions=None):
    if record is None:
        return None
    value = record.get("metrics", {}).get(pool, {})
    result = {
        "n": value.get("n", 0),
        "answer_accuracy": value.get("answer_accuracy", value.get("alias_em")),
        "canonical_em": value.get("canonical_em"),
        "unambiguous_alias_em": value.get("unambiguous_alias_em"),
        "nll": value.get("nll"),
        "correct": None,
        "counts_verified_from_predictions": False,
    }
    if predictions is not None and pool in predictions:
        raw = predictions[pool]
        if len(raw) != result["n"]:
            raise ValueError(f"Raw prediction denominator disagrees for {pool}")
        if any(item["alias_em"] not in (0, 1) for item in raw):
            raise ValueError("Exact-match scores must be binary")
        correct = sum(int(item["alias_em"]) for item in raw)
        if raw and abs(correct / len(raw) - result["answer_accuracy"]) > 1e-12:
            raise ValueError(f"Raw prediction accuracy disagrees for {pool}")
        result.update(correct=correct, counts_verified_from_predictions=True)
    return result


def preservation(before, after, pool="AA"):
    """Track actual old-correct questions rather than equating net change with retention."""
    if before is None or after is None or pool not in before or pool not in after:
        return None
    first, last = (
        {item["id"]: int(item["alias_em"]) for item in raw[pool]} for raw in (before, after)
    )
    if set(first) != set(last) or len(first) != len(before[pool]):
        raise ValueError("Preservation requires identical, unique question IDs")
    old = sum(first.values())
    retained = sum(first[key] and last[key] for key in first)
    return {
        "n": len(first),
        "stage_a_correct": old,
        "endpoint_correct": sum(last.values()),
        "retained_correct": retained,
        "lost_correct": old - retained,
        "gained_correct": sum(not first[key] and last[key] for key in first),
        "retention_of_stage_a_correct": retained / old if old else None,
    }


def hierarchical_values(rows, get_value):
    """Descriptive equal-split mean after initialization averaging; no independence claim."""
    by_split = defaultdict(list)
    for row in rows:
        by_split[row["split"]].append(get_value(row))
    splits = {}
    for split, values in sorted(by_split.items()):
        available = [value for value in values if value is not None]
        splits[split] = {
            "mean": sum(available) / len(available) if available else None,
            "available_initializations": len(available),
            "expected_initializations": len(values),
            "complete": len(available) == len(values),
        }
    available = [item["mean"] for item in splits.values() if item["mean"] is not None]
    complete = bool(splits) and all(item["complete"] for item in splits.values())
    mean = sum(available) / len(available) if available else None
    return {
        "mean": mean if complete else None,
        "available_mean": mean,
        "complete": complete,
        "available_splits": len(available),
        "expected_splits": len(splits),
        "by_split": splits,
    }


def aggregate_cell(rows, scope, pool):
    def cell(row):
        return row[scope].get(pool)

    result = hierarchical_values(
        rows, lambda row: cell(row)["answer_accuracy"] if cell(row) else None
    )
    observed = [cell(row) for row in rows if cell(row) is not None]
    counted = [value for value in observed if value["counts_verified_from_predictions"]]
    result.update(
        pooled_correct=sum(value["correct"] for value in counted) if counted else None,
        pooled_prediction_n=sum(value["n"] for value in counted) if counted else None,
        counts_verified_runs=len(counted),
        counts_complete=len(counted) == len(rows),
        unique_question_count=None,
    )
    return result


def aggregate_results(rows, pairs):
    histories = {}
    for name in HISTORY_ORDER:
        selected = [row for row in rows if row["history"] == name]
        if not selected:
            continue
        histories[name] = {
            "expected_runs": len(selected),
            "stage_a": {pool: aggregate_cell(selected, "stage_a", pool) for pool in POOLS},
            "endpoint": {pool: aggregate_cell(selected, "endpoint", pool) for pool in POOLS},
            "endpoint_autonomous": {
                pool: aggregate_cell(selected, "endpoint_autonomous", pool)
                for pool in ("AA", "BA", "AB", "BB")
            },
            "AA_change": hierarchical_values(selected, lambda row: row["AA_change"]),
            "BB_change": hierarchical_values(selected, lambda row: row["BB_change"]),
            "AA_old_correct_retention": hierarchical_values(
                selected,
                lambda row: row["AA_preservation"]["retention_of_stage_a_correct"]
                if row["AA_preservation"]
                else None,
            ),
        }
        preserved = [row["AA_preservation"] for row in selected if row["AA_preservation"]]
        histories[name]["AA_pooled_transitions"] = (
            {
                key: sum(item[key] for item in preserved)
                for key in (
                    "n",
                    "stage_a_correct",
                    "endpoint_correct",
                    "retained_correct",
                    "lost_correct",
                    "gained_correct",
                )
            }
            if preserved
            else None
        )
    contrasts = {}
    for comparison in (
        "sequential_composition_minus_atomic",
        "joint_minus_sequential_composition",
        "sequential_composition_minus_sham",
        "autonomous_minus_direct",
    ):
        contrasts[comparison] = {
            pool: hierarchical_values(
                pairs, lambda row, key=pool, comparison=comparison: row[comparison].get(key)
            )
            for pool in POOLS
        }
    contrasts["AA_change_B_minus_sham"] = hierarchical_values(
        pairs, lambda row: row["AA_change_B_minus_sham"]
    )
    contrasts["BB_change_B_minus_sham"] = hierarchical_values(
        pairs, lambda row: row["BB_change_B_minus_sham"]
    )
    return histories, contrasts


def accuracy(record, pool):
    value = metric(record, pool)
    return value["answer_accuracy"] if value else None


def difference(left, right):
    return left - right if left is not None and right is not None else None


def readiness_evidence(spec, stage_a, endpoint, audit_passed):
    """Descriptive operating evidence, deliberately excluding BA, AB and BB."""
    return {
        "name": spec["name"],
        "history": spec["history"],
        "stage_a_recorded": stage_a is not None,
        "endpoint_recorded": endpoint is not None,
        "independent_audit_passed": audit_passed,
        "stage_a": {
            pool: metric(stage_a, pool) for pool in ("atomic_A", "train_composition", "AA")
        },
        "endpoint": {pool: metric(endpoint, pool) for pool in ("atomic_A", "atomic_B", "AA")},
        "AA_change": difference(accuracy(endpoint, "AA"), accuracy(stage_a, "AA")),
        "selection_rule": "Descriptive only; no performance threshold or automatic selection",
    }


def summarize(config, root):
    root = Path(root)
    specs = config.get("specs", config.get("runs", []))
    defaults = config.get("defaults", {})
    rows, pairs, readiness, groups = [], [], [], defaultdict(dict)
    for selected in specs:
        spec = {**defaults, **selected}
        out = root / "runs" / spec["name"]
        history = read(out / "learning.json", [])
        total = spec["stage_a_steps"] + spec["stage_b_steps"]
        stage_a = next((row for row in history if row["step"] == spec["stage_a_steps"]), None)
        endpoint = next((row for row in history if row["step"] == total), None)
        stage_raw = read(out / f"predictions-{spec['stage_a_steps']:07d}.json") if stage_a else None
        final_raw = read(out / f"predictions-{total:07d}.json") if endpoint else None
        latest = max(history, key=lambda row: row["step"]) if history else None
        metadata = read(out / "run.json", {})
        audit = read(out / "audit.json", {})
        status = read(out / "status.json", {})
        sampling = read(out / "sampling-plan.json", metadata.get("sampling_manifest", {}))
        data_identity = spec.get("data_sha256", spec.get("data_file_sha256", spec["data_file"]))
        split = spec.get("split", Path(spec["data_file"]).stem)
        row = {
            "name": spec["name"],
            "split": split,
            "data_identity": data_identity,
            "initialization": spec["initialization"],
            "history": spec["history"],
            "state": status.get("state", "not_started"),
            "latest_step": latest["step"] if latest else None,
            "target_step": total,
            "stage_a_step": spec["stage_a_steps"],
            "stage_a": {pool: metric(stage_a, pool, stage_raw) for pool in POOLS},
            "endpoint": {pool: metric(endpoint, pool, final_raw) for pool in POOLS},
            "endpoint_autonomous": {
                pool: metric(endpoint, pool + "_autonomous", final_raw)
                for pool in ("AA", "BA", "AB", "BB")
            },
            "AA_change": difference(accuracy(endpoint, "AA"), accuracy(stage_a, "AA")),
            "BB_change": difference(accuracy(endpoint, "BB"), accuracy(stage_a, "BB")),
            "AA_preservation": preservation(stage_raw, final_raw),
            "initial_model_sha256": metadata.get("initial_model_sha256"),
            "multiset_sha256": sampling.get("multiset_sha256"),
            "planned_examples": sampling.get("examples"),
            "planned_supervised_tokens": sampling.get("supervised_tokens"),
            "independent_audit_passed": audit.get("passed") is True,
            "has_failure_record": (out / "failure.json").exists(),
        }
        rows.append(row)
        readiness.append(
            readiness_evidence(spec, stage_a, endpoint, row["independent_audit_passed"])
        )
        key = (split, data_identity, spec["initialization"])
        if spec["history"] in groups[key]:
            raise ValueError(f"Duplicate history in paired group: {key}, {spec['history']}")
        groups[key][spec["history"]] = row
    for (split, identity, initialization), arms in sorted(groups.items()):
        initial = [row["initial_model_sha256"] for row in arms.values()]
        combo = arms.get("sequential_composition")
        atomic = arms.get("sequential_atomic")
        joint = arms.get("joint")
        sham = arms.get("sequential_composition_sham")

        def value(row, pool):
            cell = row["endpoint"].get(pool) if row else None
            return cell["answer_accuracy"] if cell else None

        matched = None
        if combo and joint and combo["multiset_sha256"] and joint["multiset_sha256"]:
            matched = all(
                combo[field] == joint[field]
                for field in (
                    "multiset_sha256",
                    "planned_examples",
                    "planned_supervised_tokens",
                )
            )
        pairs.append(
            {
                "split": split,
                "data_identity": identity,
                "initialization": initialization,
                "histories": sorted(arms),
                "initial_hashes_all_available": all(initial),
                "initial_hashes_identical": len(set(initial)) == 1 if all(initial) else None,
                "joint_sequential_multiset_identical": matched,
                "sequential_composition_minus_atomic": {
                    pool: difference(value(combo, pool), value(atomic, pool)) for pool in POOLS
                },
                "joint_minus_sequential_composition": {
                    pool: difference(value(joint, pool), value(combo, pool)) for pool in POOLS
                },
                "sequential_composition_minus_sham": {
                    pool: difference(value(combo, pool), value(sham, pool)) for pool in POOLS
                },
                "autonomous_minus_direct": {
                    pool: difference(
                        combo["endpoint_autonomous"][pool]["answer_accuracy"]
                        if combo and combo["endpoint_autonomous"].get(pool)
                        else None,
                        value(combo, pool),
                    )
                    for pool in POOLS
                },
                "AA_change_with_B": combo["AA_change"] if combo else None,
                "AA_change_A_only_sham": sham["AA_change"] if sham else None,
                "AA_change_B_minus_sham": difference(
                    combo["AA_change"] if combo else None,
                    sham["AA_change"] if sham else None,
                ),
                "BB_before_with_B": combo["stage_a"]["BB"]["answer_accuracy"]
                if combo and combo["stage_a"]["BB"]
                else None,
                "BB_before_A_only_sham": sham["stage_a"]["BB"]["answer_accuracy"]
                if sham and sham["stage_a"]["BB"]
                else None,
                "BB_change_with_B": combo["BB_change"] if combo else None,
                "BB_change_A_only_sham": sham["BB_change"] if sham else None,
                "BB_change_B_minus_sham": difference(
                    combo["BB_change"] if combo else None,
                    sham["BB_change"] if sham else None,
                ),
            }
        )
    histories, contrasts = aggregate_results(rows, pairs)
    return {
        "batch": config.get("batch", root.name),
        "results_root": str(root),
        "expected_runs": len(specs),
        "registered_runs": sum(row["latest_step"] is not None for row in rows),
        "audited_runs": sum(row["independent_audit_passed"] for row in rows),
        "runs": rows,
        "paired_comparisons": pairs,
        "history_aggregates": histories,
        "contrast_aggregates": contrasts,
        "aggregation": {
            "order": "Mean across initializations within each split, then equal mean across splits",
            "source_graphs": 1,
            "splits_are_independent_worlds": False,
            "pooled_counts_unit": "Correct prediction occurrences across all runs; "
            "repeated questions are counted again",
            "uncertainty": "No query- or seed-based confidence interval; "
            "per-split and per-seed results retained",
        },
        "development_readiness_evidence": readiness,
        "interpretation": [
            "All rates retain actual denominators; missing endpoints remain null.",
            "Pairs share a data split and initialization; queries and seeds are not "
            "independent worlds.",
            "Readiness evidence uses A/B atomic learning, A training composition and AA only.",
            "BA/AB/BB are reported as outcomes and never feed readiness or recipe selection.",
            "Joint stage-A denotes the common schedule boundary, not an A-only learning stage.",
            "A-only sham quantifies forgetting under continued atomic training; "
            "no universal accuracy threshold is imposed.",
            "The sham never learns B; its BB score is a background comparison, "
            "not a matched-B-knowledge control.",
            "BB gain from adding B is evaluated before/after and against the same-parent "
            "A-only continuation; raw BB alone does not establish new-fact incorporation.",
            "Aggregate means average initializations first and splits second; "
            "these splits share one real graph.",
            "Pooled counts count prediction occurrences, including repeated questions "
            "across seeds and overlapping splits.",
        ],
    }


def percent(value):
    return "—" if value is None else f"{100 * value:.2f}%"


def render(summary):
    lines = [
        "# Sequential knowledge transfer",
        "",
        f"Registered {summary['registered_runs']}/{summary['expected_runs']}; "
        f"independently audited {summary['audited_runs']}.",
        "",
        "Endpoint rates; missing endpoints remain blank. Each row is one split and initialization.",
        "",
        "| Split / initialization | History | State / step | A atoms | B atoms "
        "| AA | BA | AB | BB | AA change |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    aggregate_lines = [
        "",
        "Initialization means are averaged equally across data splits from one real graph.",
        "",
        "| History | A atoms | B atoms | BB before | BB after | BB two calls "
        "| AA before | AA after | Old-correct AA retained |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in summary["history_aggregates"].items():

        def display(cell):
            text = percent(cell["mean"])
            if cell.get("counts_complete"):
                text += f" ({cell['pooled_correct']}/{cell['pooled_prediction_n']})"
            return text

        cells = [row["endpoint"][pool] for pool in ("atomic_A", "atomic_B")]
        cells.extend((row["stage_a"]["BB"], row["endpoint"]["BB"]))
        cells.extend(
            (row["endpoint_autonomous"]["BB"], row["stage_a"]["AA"], row["endpoint"]["AA"])
        )
        aggregate_lines.append(
            f"| {name} | "
            + " | ".join(display(cell) for cell in cells)
            + " | "
            + percent(row["AA_old_correct_retention"]["mean"])
            + " |"
        )
    aggregate_lines.extend(
        [
            "",
            "Counts above are prediction occurrences, including repeated questions across runs.",
            "The A-only sham never learns B; its BB score measures background answering.",
            "",
            "| Paired contrast | BB difference (percentage points) "
            "| AA difference (percentage points) |",
            "|---|---:|---:|",
        ]
    )
    for name in (
        "sequential_composition_minus_atomic",
        "joint_minus_sequential_composition",
        "sequential_composition_minus_sham",
        "autonomous_minus_direct",
    ):
        values = [summary["contrast_aggregates"][name][pool]["mean"] for pool in ("BB", "AA")]
        aggregate_lines.append(
            "| "
            + name
            + " | "
            + " | ".join("—" if value is None else f"{100 * value:+.2f}" for value in values)
            + " |"
        )
    # Keep the aggregate table ahead of the full run table without removing per-seed evidence.
    lines[5:5] = aggregate_lines
    for row in summary["runs"]:
        values = []
        for pool in ("atomic_A", "atomic_B", "AA", "BA", "AB", "BB"):
            cell = row["endpoint"][pool]
            values.append(percent(cell["answer_accuracy"] if cell else None))
        lines.append(
            f"| {row['split']} / {row['initialization']} | {row['history']} | "
            f"{row['state']} / {row['latest_step']} | "
            + " | ".join(values)
            + f" | {percent(row['AA_change'])} |"
        )
    lines.extend(
        [
            "",
            "Readiness review uses these observed values, without a universal pass threshold:",
            "",
            "| Run | Stage A atoms | Stage A training composition | Stage A AA "
            "| Final A atoms | Final B atoms | AA change |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["development_readiness_evidence"]:
        cells = [row["stage_a"][pool] for pool in ("atomic_A", "train_composition", "AA")]
        cells += [row["endpoint"][pool] for pool in ("atomic_A", "atomic_B")]
        values = [percent(cell["answer_accuracy"] if cell else None) for cell in cells]
        lines.append(
            f"| {row['name']} | " + " | ".join(values) + f" | {percent(row['AA_change'])} |"
        )
    lines.extend(["", *["- " + item for item in summary["interpretation"]], ""])
    return "\n".join(lines)


def save_figure(summary, out):
    """Two direct behavioral outcomes, with split variation and raw prediction counts."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    histories = [name for name in HISTORY_ORDER if name in summary["history_aggregates"]]
    complete = all(
        summary["history_aggregates"][name]["endpoint"][pool]["complete"]
        for name in histories
        for pool in ("AA", "BB")
    )
    if not complete or summary["audited_runs"] != summary["expected_runs"]:
        return {
            "written": False,
            "reason": "Awaiting all configured endpoints and independent audits",
        }
    labels = {
        "sequential_composition": "Prior composition\nthen new facts",
        "sequential_atomic": "Prior atoms\nthen new facts",
        "joint": "Joint\nexposure",
        "sequential_composition_sham": "A-only\ncontinuation",
    }
    colors = ["#3269A8", "#9CA6B0", "#329B84", "#D5A147"]
    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.titlesize": 11,
        }
    ):
        fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.9), sharey=True)
        for axis, pool, title in zip(
            axes, ("BB", "AA"), ("New + new facts (BB)", "Old + old facts (AA)"), strict=True
        ):
            for x, name in enumerate(histories):
                row = summary["history_aggregates"][name]
                cell = row["endpoint"][pool]
                value = 100 * cell["mean"]
                axis.bar(x, value, width=0.64, color=colors[x], alpha=0.9, zorder=2)
                split_values = [item["mean"] * 100 for item in cell["by_split"].values()]
                offsets = [
                    (i - (len(split_values) - 1) / 2) * 0.10 for i in range(len(split_values))
                ]
                axis.scatter(
                    [x + offset for offset in offsets],
                    split_values,
                    s=17,
                    color="#27323D",
                    marker="o",
                    zorder=4,
                    label="Split means" if x == 0 else None,
                )
                reference = row["stage_a"][pool]
                if reference["mean"] is not None:
                    axis.scatter(
                        [x],
                        [100 * reference["mean"]],
                        marker="_",
                        s=180,
                        color="#573839",
                        zorder=5,
                        label="Before stage B" if x == 0 else None,
                    )
                if pool == "BB" and row["endpoint_autonomous"]["BB"]["mean"] is not None:
                    axis.scatter(
                        [x],
                        [100 * row["endpoint_autonomous"]["BB"]["mean"]],
                        marker="D",
                        s=43,
                        edgecolors="#AF3929",
                        facecolors="none",
                        zorder=5,
                        label="Two model calls" if x == 0 else None,
                    )
                count = (
                    f"{cell['pooled_correct']}/{cell['pooled_prediction_n']}"
                    if cell["counts_complete"]
                    else "counts pending"
                )
                axis.text(
                    x,
                    4 if value >= 30 else max(value, max(split_values)) + 3,
                    f"{value:.1f}%\n{count}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="white" if value >= 30 else "#171C22",
                    zorder=6,
                )
            axis.set_title(title, loc="left", weight="bold", pad=31)
            axis.set_xticks(range(len(histories)), [labels[name] for name in histories])
            axis.set_ylim(0, 114)
            axis.set_yticks((0, 25, 50, 75, 100))
            axis.grid(axis="y", color="#E1E5E9", linewidth=0.7, zorder=0)
            axis.legend(
                frameon=False,
                fontsize=8,
                loc="lower left",
                bbox_to_anchor=(0, 1.005),
                ncol=3,
                columnspacing=0.9,
                handletextpad=0.45,
            )
        axes[0].set_ylabel("Exact answer accuracy (%)")
        fig.suptitle(
            "Using newly learned knowledge and retaining earlier composition",
            x=0.07,
            y=0.985,
            ha="left",
            fontsize=13,
            weight="bold",
        )
        fig.text(
            0.07,
            0.03,
            "Bars: mean over initializations, then splits. "
            "Points: split means from one real graph.\n"
            "Counts: correct prediction occurrences / total, including repeats across runs. "
            "A-only continuation never receives new B facts.\n"
            "For joint exposure, the before-stage-B marker denotes the same schedule boundary.",
            fontsize=8,
            va="bottom",
        )
        fig.subplots_adjust(left=0.07, right=0.985, bottom=0.25, top=0.80, wspace=0.16)
        paths = [
            Path(out) / "sequential-transfer-summary.pdf",
            Path(out) / "sequential-transfer-summary.png",
        ]
        for path in paths:
            fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    return {"written": True, "files": [str(path) for path in paths]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--figure", action="store_true", help="Write figure after all independent audits"
    )
    args = parser.parse_args()
    config = read(args.config)
    root = args.root or Path(config["results_root"])
    out = args.out or root / "report"
    summary = summarize(config, root)
    summary["config_sha256"] = hashlib.sha256(args.config.read_bytes()).hexdigest()
    summary["report_source_sha256"] = snapshot_reporter(out)
    write(out / "summary.json", summary)
    (out / "report.md").write_text(render(summary))
    if args.figure:
        figure = save_figure(summary, out)
        write(out / "figure-status.json", figure)
    print(
        json.dumps(
            {
                "out": str(out),
                "expected_runs": summary["expected_runs"],
                "audited_runs": summary["audited_runs"],
            }
        )
    )


if __name__ == "__main__":
    main()
