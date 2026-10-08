"""Report fixed-endpoint, paired old-atom versus full-old-data replay experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

ARMS = ("atom_replay", "full_replay")
POOLS = ("atomic_A", "atomic_B", "train_composition", "AA", "BA", "AB", "BB")
COMPOSITIONS = ("AA", "BA", "AB", "BB")
TRANSITIONS = ("AA", "BB", "train_composition")


def read(path, default=None):
    return json.loads(Path(path).read_text()) if Path(path).exists() else default


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(value):
    if value is None:
        return None
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def raw_scores(raw, pool):
    if raw is None or pool not in raw:
        return None
    scores = {}
    for item in raw[pool]:
        if item["id"] in scores or item["alias_em"] not in (0, 1):
            raise ValueError("Predictions require unique IDs and binary exact-match scores")
        scores[item["id"]] = int(item["alias_em"])
    return scores


def metric(record, pool, predictions=None):
    if record is None or pool not in record.get("metrics", {}):
        return None
    value = record["metrics"][pool]
    scores = raw_scores(predictions, pool)
    accuracy = value.get("answer_accuracy", value.get("alias_em"))
    result = {
        "n": value["n"],
        "answer_accuracy": accuracy,
        "canonical_em": value.get("canonical_em"),
        "unambiguous_alias_em": value.get("unambiguous_alias_em"),
        "nll": value.get("nll"),
        "correct": None,
        "counts_verified_from_predictions": scores is not None,
    }
    if scores is not None:
        correct = sum(scores.values())
        if len(scores) != result["n"]:
            raise ValueError(f"Prediction denominator disagrees for {pool}")
        if scores and (accuracy is None or abs(correct / len(scores) - accuracy) > 1e-12):
            raise ValueError(f"Prediction accuracy disagrees for {pool}")
        result["correct"] = correct
    return result


def preservation(before, after, pool):
    first, last = raw_scores(before, pool), raw_scores(after, pool)
    if first is None or last is None:
        return None
    if set(first) != set(last):
        raise ValueError(f"Before/after prediction IDs disagree for {pool}")
    old = sum(first.values())
    retained = sum(first[key] and last[key] for key in first)
    return {
        "n": len(first),
        "before_correct": old,
        "after_correct": sum(last.values()),
        "retained_correct": retained,
        "lost_correct": old - retained,
        "gained_correct": sum(not first[key] and last[key] for key in first),
        "retention_of_before_correct": retained / old if old else None,
    }


def difference(left, right):
    return left - right if left is not None and right is not None else None


def rate(row, scope, pool):
    cell = row[scope].get(pool) if row else None
    return cell["answer_accuracy"] if cell else None


def hierarchical_values(rows, get_value):
    """Average seeds within split, then splits; incomplete matrices have no final mean."""
    by_split = defaultdict(list)
    for row in rows:
        by_split[row["split"]].append(get_value(row))
    splits = {}
    for split, values in sorted(by_split.items()):
        observed = [value for value in values if value is not None]
        splits[split] = {
            "mean": sum(observed) / len(observed) if observed else None,
            "available_initializations": len(observed),
            "expected_initializations": len(values),
            "complete": len(observed) == len(values),
        }
    observed = [item["mean"] for item in splits.values() if item["mean"] is not None]
    complete = bool(splits) and all(item["complete"] for item in splits.values())
    mean = sum(observed) / len(observed) if observed else None
    return {
        "mean": mean if complete else None,
        "available_mean": mean,
        "complete": complete,
        "by_split": splits,
    }


def aggregate_cell(rows, scope, pool):
    result = hierarchical_values(rows, lambda row: rate(row, scope, pool))
    cells = [row[scope][pool] for row in rows if row[scope].get(pool)]
    counted = [cell for cell in cells if cell["counts_verified_from_predictions"]]
    result.update(
        pooled_correct=sum(cell["correct"] for cell in counted) if counted else None,
        pooled_prediction_n=sum(cell["n"] for cell in counted) if counted else None,
        counts_verified_runs=len(counted),
        counts_complete=len(counted) == len(rows),
    )
    return result


def aggregate_transitions(rows, pool):
    items = [row["transitions"][pool] for row in rows if row["transitions"][pool]]
    result = hierarchical_values(
        rows,
        lambda row: row["transitions"][pool]["retention_of_before_correct"]
        if row["transitions"][pool]
        else None,
    )
    result["counts_complete"] = len(items) == len(rows)
    result["pooled"] = (
        {
            key: sum(item[key] for item in items)
            for key in (
                "n",
                "before_correct",
                "after_correct",
                "retained_correct",
                "lost_correct",
                "gained_correct",
            )
        }
        if items
        else None
    )
    return result


def same_available(values):
    return len(set(values)) == 1 if len(values) == 2 and all(values) else None


def paired_comparison(arms):
    atomic, full = arms.get("atom_replay"), arms.get("full_replay")
    checks = {
        key: same_available([row.get(field) for row in (atomic, full) if row])
        for key, field in (
            ("parent_checkpoint_identical", "parent_checkpoint_sha256"),
            ("parent_model_identical", "parent_model_sha256"),
            ("new_B_stream_identical", "new_stream_sha256"),
            ("stage_a_predictions_identical", "stage_a_predictions_sha256"),
        )
    }
    valid = False if False in checks.values() else (True if all(checks.values()) else None)
    result = {"pairing_checks": checks, "pairing_verified": valid}
    for scope in ("endpoint", "endpoint_autonomous"):
        pools = POOLS if scope == "endpoint" else COMPOSITIONS
        result[scope + "_full_minus_atomic"] = {
            pool: difference(rate(full, scope, pool), rate(atomic, scope, pool))
            if valid is True
            else None
            for pool in pools
        }
    result["change_full_minus_atomic"] = {
        pool: difference(full["changes"][pool], atomic["changes"][pool]) if valid is True else None
        for pool in POOLS
    }
    return result


def summarize(config, root):
    root = Path(root)
    specs = config.get("specs", config.get("runs", []))
    rows, groups = [], defaultdict(dict)
    for selected in specs:
        spec = {**config.get("defaults", {}), **selected}
        out = root / "runs" / spec["name"]
        metadata = read(out / "run.json", {})
        parent_dir = Path(spec["parent_run_dir"])
        parent = read(parent_dir / "run.json", {}).get("spec", {})
        merged = {**parent, **metadata.get("parent_spec", {}), **spec}
        stage_a_step = spec["original_stage_a_steps"]
        total = stage_a_step + spec["stage_b_steps"]
        learning = read(out / "learning.json", [])
        stage_a = next((item for item in learning if item["step"] == stage_a_step), None)
        stage_source = "branch_recomputed" if stage_a else None
        stage_raw = read(out / f"predictions-{stage_a_step:07d}.json") if stage_a else None
        if stage_a is None:
            stage_a = next(
                (
                    item
                    for item in read(parent_dir / "learning.json", [])
                    if item["step"] == stage_a_step
                ),
                None,
            )
            stage_source = "parent_recorded" if stage_a else None
            stage_raw = (
                read(parent_dir / f"predictions-{stage_a_step:07d}.json") if stage_a else None
            )
        endpoint = next((item for item in learning if item["step"] == total), None)
        final_raw = read(out / f"predictions-{total:07d}.json") if endpoint else None
        latest = max(learning, key=lambda item: item["step"]) if learning else None
        sampling = read(out / "sampling-plan.json", {})
        arm = spec["replay_arm"]
        if arm not in ARMS:
            raise ValueError(f"Unknown replay arm: {arm}")
        data_file = merged.get("data_file", "unknown")
        split = merged.get("split", Path(data_file).stem)
        identity = merged.get("data_sha256", merged.get("data_file_sha256", data_file))
        audit = read(out / "audit.json", {})
        row = {
            "name": spec["name"],
            "arm": arm,
            "split": split,
            "initialization": merged["initialization"],
            "data_identity": identity,
            "stage_a_step": stage_a_step,
            "target_step": total,
            "target_branch_updates": spec["stage_b_steps"],
            "latest_step": latest["step"] if latest else None,
            "latest_branch_step": latest.get("branch_step") if latest else None,
            "state": read(out / "status.json", {}).get("state", "not_started"),
            "stage_a_source": stage_source,
            "stage_a": {pool: metric(stage_a, pool, stage_raw) for pool in POOLS},
            "endpoint": {pool: metric(endpoint, pool, final_raw) for pool in POOLS},
            "endpoint_autonomous": {
                pool: metric(endpoint, pool + "_autonomous", final_raw) for pool in COMPOSITIONS
            },
            "transitions": {pool: preservation(stage_raw, final_raw, pool) for pool in TRANSITIONS},
            "parent_run_dir": str(parent_dir),
            "parent_checkpoint_sha256": spec.get("parent_checkpoint_sha256"),
            "parent_model_sha256": metadata.get("parent_model_sha256"),
            "stage_a_predictions_sha256": digest(stage_raw),
            "new_stream_sha256": sampling.get("new_stream_sha256"),
            "sampling_manifest": sampling,
            "endpoint_exposures": endpoint.get("exposures") if endpoint else None,
            "endpoint_wall_seconds": endpoint.get("wall_seconds") if endpoint else None,
            "independent_audit_passed": audit.get("passed") is True,
            "completion_recorded": (out / "complete.json").exists(),
            "has_failure_record": (out / "failure.json").exists(),
        }
        row["changes"] = {
            pool: difference(rate(row, "endpoint", pool), rate(row, "stage_a", pool))
            for pool in POOLS
        }
        key = (split, identity, merged["initialization"])
        if arm in groups[key]:
            raise ValueError(f"Duplicate replay arm within pair: {key}")
        groups[key][arm] = row
        rows.append(row)
    pairs = [
        {
            "split": split,
            "data_identity": identity,
            "initialization": initialization,
            "arms": sorted(arms),
            **paired_comparison(arms),
        }
        for (split, identity, initialization), arms in sorted(groups.items())
    ]
    aggregates = {}
    for arm in ARMS:
        selected = [row for row in rows if row["arm"] == arm]
        if not selected:
            continue
        aggregates[arm] = {
            "expected_runs": len(selected),
            **{
                scope: {pool: aggregate_cell(selected, scope, pool) for pool in pools}
                for scope, pools in (
                    ("stage_a", POOLS),
                    ("endpoint", POOLS),
                    ("endpoint_autonomous", COMPOSITIONS),
                )
            },
            "changes": {
                pool: hierarchical_values(selected, lambda row, p=pool: row["changes"][p])
                for pool in POOLS
            },
            "transitions": {pool: aggregate_transitions(selected, pool) for pool in TRANSITIONS},
        }
    contrasts = {
        key: {
            pool: hierarchical_values(pairs, lambda row, k=key, p=pool: row[k][p]) for pool in pools
        }
        for key, pools in (
            ("endpoint_full_minus_atomic", POOLS),
            ("endpoint_autonomous_full_minus_atomic", COMPOSITIONS),
            ("change_full_minus_atomic", POOLS),
        )
    }
    return {
        "batch": config.get("batch", root.name),
        "phase": config.get("phase", specs[0].get("phase") if specs else None),
        "results_root": str(root),
        "expected_runs": len(specs),
        "registered_runs": sum(row["latest_step"] is not None for row in rows),
        "endpoint_runs": sum(row["endpoint"]["BB"] is not None for row in rows),
        "audited_runs": sum(row["independent_audit_passed"] for row in rows),
        "runs": rows,
        "paired_comparisons": pairs,
        "arm_aggregates": aggregates,
        "contrast_aggregates": contrasts,
        "aggregation": {
            "order": "Average initializations within each split, then weight splits equally",
            "source_graphs": 1,
            "splits_are_independent_worlds": False,
            "pooled_counts_unit": "Prediction occurrences, including repeats across runs",
            "uncertainty": "Report every split and initialization; no independence-based CI",
        },
        "development_readiness_evidence": [
            {
                "name": row["name"],
                "audit_passed": row["independent_audit_passed"],
                "endpoint": {
                    pool: row["endpoint"][pool]
                    for pool in ("atomic_A", "atomic_B", "train_composition", "AA")
                },
            }
            for row in rows
        ],
        "interpretation": [
            "Both replay arms use the same parent weights, B examples and optimizer recipe; "
            "the old replay distribution differs.",
            "Fixed endpoints only; an intermediate checkpoint never substitutes for an endpoint.",
            "BB, BA and AB are reported outcomes, never inputs to recipe selection.",
            "Before/after BB measures net learning; the matched atomic replay contrast "
            "isolates the effect of replaying the full old training distribution.",
            "AA retention counts the questions correct before stage B that remain correct; "
            "gains on different questions do not erase losses.",
            "Autonomous two-call evaluation is externally decomposed and reported separately "
            "from direct compositional answering.",
            "The paired extension reuses three previously observed splits of one real graph; "
            "it is not a new independent-world confirmation.",
            "Pooled counts include repeated questions across seeds and overlapping splits; "
            "the reported mean weights splits equally.",
            "Training composition is a rehearsal target; held-out AA is reported separately.",
        ],
    }


def percent(value):
    return "—" if value is None else f"{100 * value:.2f}%"


def display(cell, aggregate=False):
    if cell is None:
        return "—"
    result = percent(cell["mean"] if aggregate else cell["answer_accuracy"])
    if cell.get("counts_complete" if aggregate else "counts_verified_from_predictions"):
        correct = cell["pooled_correct" if aggregate else "correct"]
        total = cell["pooled_prediction_n" if aggregate else "n"]
        result += f" ({correct}/{total})"
    return result


def render(summary):
    lines = [
        "# Sequential replay: fixed-endpoint paired comparison",
        "",
        f"Batch: {summary['batch']}; phase: {summary['phase']}. "
        f"Endpoints {summary['endpoint_runs']}/{summary['expected_runs']}; "
        f"independent audits {summary['audited_runs']}/{summary['expected_runs']}.",
        "",
        "Means average initializations within split, then weight splits equally. "
        "Parent-stage results are inherited or recomputed, not newly trained here.",
        "",
        "| Arm | A atoms | B atoms | Training AA before → after | Held-out AA before → after "
        "| BB before → after | BB two model calls |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, row in summary["arm_aggregates"].items():
        values = [display(row["endpoint"][pool], True) for pool in ("atomic_A", "atomic_B")]
        values += [
            display(row["stage_a"][pool], True) + " → " + display(row["endpoint"][pool], True)
            for pool in ("train_composition", "AA", "BB")
        ]
        values.append(display(row["endpoint_autonomous"]["BB"], True))
        lines.append(f"| {arm} | " + " | ".join(values) + " |")
    lines += [
        "",
        "Counts are correct prediction occurrences / all occurrences, including "
        "repeated questions. Percentages are equal-split means and need not equal "
        "the pooled count ratio.",
        "",
        "| Arm / pool | Old-correct retained | Lost | Newly correct | "
        "Mean fraction of old-correct retained |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm, row in summary["arm_aggregates"].items():
        for pool in TRANSITIONS:
            transition = row["transitions"][pool]
            counts = transition["pooled"] if transition["counts_complete"] else None
            values = (
                [
                    f"{counts['retained_correct']}/{counts['before_correct']}",
                    str(counts["lost_correct"]),
                    str(counts["gained_correct"]),
                ]
                if counts
                else ["—"] * 3
            )
            lines.append(
                f"| {arm} / {pool} | " + " | ".join(values) + f" | {percent(transition['mean'])} |"
            )
    lines += [
        "",
        "| Full replay minus atomic replay | A atoms | B atoms | Training AA "
        "| Held-out AA | BA | AB | BB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in ("endpoint_full_minus_atomic", "change_full_minus_atomic"):
        values = [summary["contrast_aggregates"][key][pool]["mean"] for pool in POOLS]
        lines.append(
            f"| {key} (percentage points) | "
            + " | ".join("—" if value is None else f"{100 * value:+.2f}" for value in values)
            + " |"
        )
    lines += [
        "",
        "| Split / initialization | Arm | B updates | A atoms | B atoms | "
        "Training AA | AA before → after | BA | AB | BB before → after | BB two calls |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["runs"]:
        values = [
            display(row["endpoint"][pool]) for pool in ("atomic_A", "atomic_B", "train_composition")
        ]
        values.append(display(row["stage_a"]["AA"]) + " → " + display(row["endpoint"]["AA"]))
        values += [display(row["endpoint"][pool]) for pool in ("BA", "AB")]
        values.append(display(row["stage_a"]["BB"]) + " → " + display(row["endpoint"]["BB"]))
        values.append(display(row["endpoint_autonomous"]["BB"]))
        lines.append(
            f"| {row['split']} / {row['initialization']} | {row['arm']} | "
            f"{row['latest_branch_step']} / {row['target_branch_updates']} | "
            + " | ".join(values)
            + " |"
        )
    lines += [
        "",
        "| Split / initialization | Pairing verified | AA difference (pp) | BB difference (pp) |",
        "|---|---|---:|---:|",
    ]
    for pair in summary["paired_comparisons"]:
        values = [pair["endpoint_full_minus_atomic"][pool] for pool in ("AA", "BB")]
        lines.append(
            f"| {pair['split']} / {pair['initialization']} | "
            f"{pair['pairing_verified']} | "
            + " | ".join("—" if value is None else f"{100 * value:+.2f}" for value in values)
            + " |"
        )
    lines += ["", *["- " + item for item in summary["interpretation"]], ""]
    return "\n".join(lines)


def snapshot_reporter(out):
    script = Path(__file__).resolve()
    sources = {}
    for source in (script, script.parents[1] / "tests/test_sequential_replay_report.py"):
        if not source.exists():
            continue
        target = Path(out) / "source" / source.name
        payload = source.read_bytes()
        if target.exists() and target.read_bytes() != payload:
            raise FileExistsError("Use a new report directory for changed analysis source")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        sources[str(target)] = hashlib.sha256(payload).hexdigest()
    return sources


def save_figure(summary, out):
    """Display descriptive split variation, without treating splits as independent worlds."""
    if (
        not summary["expected_runs"]
        or summary["endpoint_runs"] != summary["expected_runs"]
        or summary["audited_runs"] != summary["expected_runs"]
        or not all(pair["pairing_verified"] is True for pair in summary["paired_comparisons"])
    ):
        return {"written": False, "reason": "Awaiting fixed endpoints, audits and verified pairs"}
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {"atom_replay": "Replay old atoms", "full_replay": "Replay full old data"}
    colors = {"atom_replay": "#517C9D", "full_replay": "#32866E"}
    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    ):
        fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.7), sharey=True)
        for axis, pool, title in zip(
            axes,
            ("AA", "BB"),
            ("Retaining earlier combinations (AA)", "Composing new facts (BB)"),
            strict=True,
        ):
            for x, arm in enumerate(ARMS):
                row = summary["arm_aggregates"][arm]
                cell = row["endpoint"][pool]
                after = cell["mean"] * 100
                before = row["stage_a"][pool]["mean"] * 100
                axis.bar(x, after, width=0.58, color=colors[arm], zorder=2)
                splits = [item["mean"] * 100 for item in cell["by_split"].values()]
                offsets = [(i - (len(splits) - 1) / 2) * 0.10 for i in range(len(splits))]
                axis.scatter(
                    [x + offset for offset in offsets],
                    splits,
                    color="#172E3D",
                    edgecolors="white",
                    linewidths=0.5,
                    s=30,
                    zorder=4,
                    label="Split means after replay" if x == 0 else None,
                )
                axis.scatter(
                    [x],
                    [before],
                    marker="_",
                    color="#99362E",
                    s=550,
                    linewidths=2.2,
                    zorder=5,
                    label="Before replay" if x == 0 else None,
                )
                text_y = max([after, before, *splits]) + 5
                axis.text(
                    x,
                    text_y,
                    f"{after:.1f}%\n{cell['pooled_correct']}/{cell['pooled_prediction_n']}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
            axis.set_xticks(range(2), [labels[arm] for arm in ARMS])
            axis.set_xlim(-0.55, 1.55)
            axis.set_ylim(0, 119)
            axis.set_yticks((0, 25, 50, 75, 100))
            axis.set_title(title, loc="left", fontsize=11, weight="bold", pad=15)
            axis.grid(axis="y", linewidth=0.6, color="#DFE5E8", zorder=0)
        axes[0].set_ylabel("Exact answer accuracy (%)")
        axes[0].legend(frameon=False, loc="upper left", fontsize=8)
        fig.suptitle(
            "Does replaying earlier combinations help new knowledge transfer?",
            x=0.08,
            y=0.99,
            ha="left",
            fontsize=12,
            weight="bold",
        )
        split_count = len(
            next(iter(summary["arm_aggregates"].values()))["endpoint"]["AA"]["by_split"]
        )
        phase_text = (
            "Previously observed splits; this is a paired extension. "
            if summary["phase"] == "paired_extension"
            else "Development data. "
        )
        fig.text(
            0.08,
            0.025,
            "Bars: mean across initializations, then equally across "
            f"{split_count} split(s) of the same real graph. No independence-based CI.\n"
            "Counts: correct prediction occurrences / all occurrences, including repeats. "
            "Both arms learn the same B examples.\n"
            + phase_text
            + "Two-call results and actual old-correct retention are reported separately.",
            fontsize=7.7,
            va="bottom",
        )
        fig.subplots_adjust(left=0.08, right=0.98, bottom=0.24, top=0.82, wspace=0.14)
        paths = [
            Path(out) / f"sequential-replay-summary.{extension}" for extension in ("png", "pdf")
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
    parser.add_argument("--figure", action="store_true")
    args = parser.parse_args()
    config = read(args.config)
    root = args.root or Path(config.get("results_root", Path("results") / config["batch"]))
    out = args.out or root / "report"
    summary = summarize(config, root)
    summary["config_sha256"] = hashlib.sha256(args.config.read_bytes()).hexdigest()
    summary["report_source_sha256"] = snapshot_reporter(out)
    write(out / "summary.json", summary)
    (out / "report.md").write_text(render(summary))
    if args.figure:
        write(out / "figure-status.json", save_figure(summary, out))
    print(
        json.dumps(
            {
                "out": str(out),
                "endpoints": summary["endpoint_runs"],
                "audited_runs": summary["audited_runs"],
            }
        )
    )


if __name__ == "__main__":
    main()
