"""Report Loop knowledge-learning comparisons, including incomplete batches.

Intermediate evaluation panels and full stage endpoints remain separate. Seeds
are repeated fits of a dataset, never counted as independent knowledge worlds.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ARMS = ("shared", "untied", "mlp_shared", "shallow")
POOLS = ("atomic_A", "atomic_B", "train_composition", "AA", "BA", "AB", "BB")
COSTS = (
    "examples",
    "supervised_tokens",
    "effective_input_tokens",
    "executed_input_tokens",
    "estimated_matmul_training_flops",
    "training_seconds",
    "wall_seconds",
    "peak_gpu_memory_bytes",
)
COLORS = dict(zip(ARMS, ("#2868A7", "#D17A26", "#398364", "#8A659E"), strict=True))


def read(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else default


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def identity(spec):
    """Use the intended Loop design, not shallow's executed depth, for pairing."""
    return {
        "data_sha256": spec.get("data_sha256", spec.get("data_file")),
        "initialization": spec["initialization"],
        "hidden_size": spec.get("model", {}).get("hidden_size", spec.get("hidden_size")),
        "base_blocks": spec.get(
            "base_unique_blocks", spec.get("base_blocks", spec.get("unique_layers"))
        ),
        "repeats": spec.get("repeats"),
        "sampling_seed": spec.get("sampling_seed"),
        "stage_a_steps": spec["stage_a_steps"],
        "stage_b_steps": spec["stage_b_steps"],
        "batch_size": spec.get("batch_size"),
        "history": spec.get("history"),
        "replay_source": spec.get("replay_source"),
    }


def arm_name(spec):
    arm = spec.get("arm", spec.get("sharing", spec.get("architecture")))
    if arm not in ARMS:
        raise ValueError(f"Unknown Loop arm: {arm!r}")
    return arm


def metric(record, pool, predictions=None):
    if record is None or pool not in record.get("metrics", {}):
        return None
    raw_metric = record["metrics"][pool]
    cell = {
        **raw_metric,
        "answer_accuracy": raw_metric.get("answer_accuracy", raw_metric.get("alias_em")),
        "correct": None,
        "counts_verified_from_predictions": False,
    }
    if predictions is not None and pool in predictions:
        rows = predictions[pool]
        if len({item["id"] for item in rows}) != len(rows):
            raise ValueError(f"Duplicate prediction IDs in {pool}")
        if len(rows) != cell.get("n"):
            raise ValueError(f"Prediction denominator disagrees for {pool}")
        if any(item["alias_em"] not in (0, 1) for item in rows):
            raise ValueError("Exact-match scores must be binary")
        correct = sum(int(item["alias_em"]) for item in rows)
        accuracy = cell["answer_accuracy"]
        if rows and (accuracy is None or abs(correct / len(rows) - accuracy) > 1e-12):
            raise ValueError(f"Prediction accuracy disagrees for {pool}")
        cell.update(correct=correct, counts_verified_from_predictions=True)
    return cell


def transitions(before, after, pool):
    """Count per-question retention, losses and gains on an identical full pool."""
    if before is None or after is None or pool not in before or pool not in after:
        return None
    maps = []
    for predictions in (before, after):
        rows = predictions[pool]
        if any(item["alias_em"] not in (0, 1) for item in rows):
            raise ValueError("Exact-match scores must be binary")
        lookup = {item["id"]: int(item["alias_em"]) for item in rows}
        if len(lookup) != len(rows):
            raise ValueError(f"Duplicate prediction IDs in {pool}")
        maps.append(lookup)
    first, last = maps
    if first.keys() != last.keys():
        raise ValueError(f"Full before/after question IDs differ for {pool}")
    old = sum(first.values())
    retained = sum(first[key] and last[key] for key in first)
    gained = sum(not first[key] and last[key] for key in first)
    n = len(first)
    return {
        "n": n,
        "stage_a_correct": old,
        "endpoint_correct": sum(last.values()),
        "retained_correct": retained,
        "lost_correct": old - retained,
        "gained_correct": gained,
        "neither_correct": n - old - gained,
        "stage_a_correct_coverage": old / n if n else None,
        "retention_of_stage_a_correct": retained / old if old else None,
        "net_accuracy_change": (gained - (old - retained)) / n if n else None,
    }


def difference(first, second):
    return first - second if first is not None and second is not None else None


def accuracy(row, boundary, pool):
    cell = row[boundary].get(pool)
    return cell["answer_accuracy"] if cell else None


def summarize(config, root):
    root = Path(root)
    rows, curves, groups = [], [], defaultdict(dict)
    for spec in config["specs"]:
        out = root / "runs" / spec["name"]
        history = read(out / "learning.json", [])
        steps = [item["step"] for item in history]
        if steps != sorted(set(steps)):
            raise ValueError(f"Evaluation steps are not unique and increasing: {spec['name']}")
        metadata = read(out / "run.json", {})
        if "spec" in metadata and metadata["spec"] != spec:
            raise ValueError(f"Run spec differs from frozen config: {spec['name']}")
        stage_a_step = spec["stage_a_steps"]
        final_step = stage_a_step + spec["stage_b_steps"]
        stage_a = next((item for item in history if item["step"] == stage_a_step), None)
        endpoint = next((item for item in history if item["step"] == final_step), None)
        first_raw = read(out / f"predictions-{stage_a_step:07d}.json")
        last_raw = read(out / f"predictions-{final_step:07d}.json")
        saved_endpoint = read(out / "endpoint.json")
        if endpoint and saved_endpoint is not None and endpoint["metrics"] != saved_endpoint:
            raise ValueError(f"Endpoint differs from learning record: {spec['name']}")
        audit = read(out / "audit.json", {})
        status = read(out / "status.json", {})
        architecture = read(out / "architecture.json", {})
        sampling = read(out / "sampling-plan.json", {})
        arm = arm_name(spec)
        paired_identity = identity(spec)
        full_pools = set(POOLS)
        for boundary in (stage_a, endpoint):
            if boundary:
                full_pools.update(boundary.get("metrics", {}))
        row = {
            "name": spec["name"],
            "arm": arm,
            **paired_identity,
            "latest_step": steps[-1] if steps else None,
            "expected_final_step": final_step,
            "state": status.get("state", "unregistered" if not metadata else "pending"),
            "registered": bool(metadata),
            "independent_audit_passed": audit.get("passed") is True,
            "complete_artifact": (out / "complete.json").exists(),
            "has_failure_record": (out / "failure.json").exists(),
            "stage_a": {pool: metric(stage_a, pool, first_raw) for pool in sorted(full_pools)},
            "endpoint": {pool: metric(endpoint, pool, last_raw) for pool in sorted(full_pools)},
            "transitions": {pool: transitions(first_raw, last_raw, pool) for pool in POOLS},
            "parameters": metadata.get("parameters", architecture.get("parameters")),
            "architecture": architecture,
            "initial_model_sha256": metadata.get("initial_model_sha256"),
            "plan_sha256": sampling.get("plan_sha256"),
            "multiset_sha256": sampling.get("multiset_sha256"),
            "cost_at_stage_a": {key: stage_a.get(key) if stage_a else None for key in COSTS},
            "cost_at_endpoint": {key: endpoint.get(key) if endpoint else None for key in COSTS},
            "cost_at_latest": {key: history[-1].get(key) if history else None for key in COSTS},
        }
        row["change_during_B"] = {
            pool: difference(accuracy(row, "endpoint", pool), accuracy(row, "stage_a", pool))
            for pool in POOLS
        }
        row["complete"] = (
            row["complete_artifact"]
            and row["independent_audit_passed"]
            and endpoint is not None
            and not row["has_failure_record"]
        )
        rows.append(row)
        group_key = json.dumps(paired_identity, sort_keys=True)
        if arm in groups[group_key]:
            raise ValueError(f"Duplicate paired arm {arm}: {group_key}")
        groups[group_key][arm] = row
        for item in history:
            scope = "full" if item["step"] in (stage_a_step, final_step) else "panel"
            if item.get("evaluation_scope", scope) != scope:
                raise ValueError(f"Unexpected evaluation scope: {spec['name']}, {item['step']}")
            raw = read(out / f"predictions-{item['step']:07d}.json")
            for pool in sorted(item.get("metrics", {})):
                cell = metric(item, pool, raw)
                curves.append(
                    {
                        "name": spec["name"],
                        "arm": arm,
                        **paired_identity,
                        "step": item["step"],
                        "stage": "A" if item["step"] <= stage_a_step else "B",
                        "evaluation_scope": scope,
                        "pool": pool,
                        **cell,
                        **{key: item.get(key) for key in COSTS},
                    }
                )
    paired = []
    for key, arms in sorted(groups.items()):
        shared = arms.get("shared")
        for name in ARMS[1:]:
            other = arms.get(name)
            if shared is None or other is None:
                continue
            item = {
                **json.loads(key),
                "contrast": "shared_minus_" + name,
                "runs": [shared["name"], other["name"]],
                "both_complete": shared["complete"] and other["complete"],
                "sampling_plan_identical": (
                    shared["plan_sha256"] == other["plan_sha256"]
                    if shared["plan_sha256"] and other["plan_sha256"]
                    else None
                ),
                "initial_parameter_hash_equality_is_not_function_equality": True,
            }
            for boundary in ("stage_a", "endpoint"):
                item[boundary] = {
                    pool: difference(
                        accuracy(shared, boundary, pool), accuracy(other, boundary, pool)
                    )
                    for pool in POOLS
                }
            item["change_during_B"] = {
                pool: difference(shared["change_during_B"][pool], other["change_during_B"][pool])
                for pool in POOLS
            }
            for field in ("retention_of_stage_a_correct", "stage_a_correct_coverage"):
                left, right = shared["transitions"]["AA"], other["transitions"]["AA"]
                item["AA_" + field] = difference(
                    left[field] if left else None, right[field] if right else None
                )
            paired.append(item)
    complete = bool(rows) and all(row["complete"] for row in rows)
    return {
        "batch": config.get("batch", root.name),
        "results_root": str(root),
        "state": "complete" if complete else "partial",
        "expected_runs": len(rows),
        "registered_runs": sum(row["registered"] for row in rows),
        "completed_runs": sum(row["complete"] for row in rows),
        "audited_runs": sum(row["independent_audit_passed"] for row in rows),
        "data_hashes": sorted({row["data_sha256"] for row in rows}),
        "runs": rows,
        "paired_comparisons": paired,
        "curves": curves,
        "interpretation": [
            "Each run is one fit; seeds and dataset partitions are not independent worlds.",
            "Panel curves retain actual denominators and are not connected to full endpoints.",
            "Both stage endpoints use full fixed pools; missing endpoints remain null.",
            "BB at stage A is the baseline; final BB alone does not measure newly acquired use.",
            "Before/after changes without an A-only continuation do not isolate B's causal effect.",
            "AA retention uses each model's stage-A correct set; its coverage is reported.",
            "Steps, data exposure, tokens, training FLOPs, time and parameters are distinct costs.",
            "Shallow executes fewer blocks; its per-example compute is not matched to Loop.",
            "Shared and untied state-dict hashes need not match despite equal initial functions.",
            "Independent reload and W&B cloud completion are separate checks.",
        ],
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }


def display(cell):
    if cell is None or cell.get("answer_accuracy") is None:
        return "—"
    count = (
        f"{cell['correct']}/{cell['n']}" if cell.get("correct") is not None else f"n={cell['n']}"
    )
    return f"{100 * cell['answer_accuracy']:.2f}% ({count})"


def percentage(value):
    return "—" if value is None else f"{100 * value:.2f}%"


def render(summary):
    lines = [
        "# Loop knowledge learning and use",
        "",
        f"Status: **{summary['state']}**. Registered {summary['registered_runs']}/"
        f"{summary['expected_runs']}; complete with reload {summary['completed_runs']}. "
        "Incomplete runs are retained, with missing full endpoints left blank.",
        "",
        "Each row below uses a full stage endpoint. Seeds are repeated training runs on the "
        "same data, not new knowledge worlds. Stage A ends before new B training begins.",
        "",
        "| Run | Boundary | A atoms | B atoms | AA | BA | AB | BB |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["runs"]:
        for boundary in ("stage_a", "endpoint"):
            cells = [
                display(row[boundary].get(pool)) for pool in POOLS if pool != "train_composition"
            ]
            lines.append(f"| {row['name']} | {boundary} | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "AA parent-correct retention uses actual question IDs, with each model's selection "
            "coverage. BB gain is the net change from its stage-A baseline, "
            "not its final accuracy.",
            "",
            "| Run | AA parent-correct coverage | AA retained / parent-correct | AA lost | "
            "AA newly correct | BB before | BB after | BB change (pp) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["runs"]:
        item = row["transitions"]["AA"]
        cells = (
            [
                percentage(item["stage_a_correct_coverage"]),
                f"{item['retained_correct']}/{item['stage_a_correct']} "
                f"({percentage(item['retention_of_stage_a_correct'])})",
                str(item["lost_correct"]),
                str(item["gained_correct"]),
            ]
            if item
            else ["—"] * 4
        )
        delta = row["change_during_B"]["BB"]
        cells += [
            display(row["stage_a"]["BB"]),
            display(row["endpoint"]["BB"]),
            "—" if delta is None else f"{100 * delta:+.2f}",
        ]
        lines.append(f"| {row['name']} | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "Paired differences below are percentage points. They match initialization, data, "
            "width, base blocks, repetitions and training budgets. Missing endpoints stay blank.",
            "",
            "| Contrast | Initialization | Width | Both complete | Final AA difference | "
            "Final BB difference | Difference in BB change |",
            "|---|---:|---:|---|---:|---:|---:|",
        ]
    )
    for row in summary["paired_comparisons"]:
        values = [row["endpoint"]["AA"], row["endpoint"]["BB"], row["change_during_B"]["BB"]]
        lines.append(
            f"| {row['contrast']} | {row['initialization']} | {row['hidden_size']} | "
            f"{row['both_complete']} | "
            + " | ".join("—" if value is None else f"{100 * value:+.2f}" for value in values)
            + " |"
        )
    lines.extend(
        [
            "",
            "Resource values below are cumulative through the latest recorded evaluation. "
            "Completed runs include both learning stages. The CSV/JSON preserve stage-A "
            "costs and every intermediate panel denominator separately.",
            "",
            "| Run | State / step | Parameters | Supervised tokens | Estimated training FLOPs | "
            "Training seconds | Wall seconds |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["runs"]:
        values = [row["parameters"]] + [
            row["cost_at_latest"][key]
            for key in (
                "supervised_tokens",
                "estimated_matmul_training_flops",
                "training_seconds",
                "wall_seconds",
            )
        ]
        lines.append(
            f"| {row['name']} | {row['state']} / {row['latest_step']} | "
            + " | ".join("—" if value is None else f"{value:,.3g}" for value in values)
            + " |"
        )
    lines += ["", *["- " + line for line in summary["interpretation"]], ""]
    return "\n".join(lines)


def write_csv(path, rows):
    fields = sorted({key for row in rows for key in row})
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def export_csv(summary, out):
    out = Path(out)
    write_csv(out / "curves.csv", summary["curves"])
    endpoints, comparisons = [], []
    for row in summary["runs"]:
        for boundary in ("stage_a", "endpoint"):
            for pool, cell in row[boundary].items():
                endpoints.append(
                    {
                        "name": row["name"],
                        "arm": row["arm"],
                        "initialization": row["initialization"],
                        "hidden_size": row["hidden_size"],
                        "base_blocks": row["base_blocks"],
                        "repeats": row["repeats"],
                        "data_sha256": row["data_sha256"],
                        "boundary": boundary,
                        "pool": pool,
                        "complete": row["complete"],
                        "parameters": row["parameters"],
                        **(cell or {}),
                        **row["cost_at_stage_a" if boundary == "stage_a" else "cost_at_endpoint"],
                    }
                )
    for pair in summary["paired_comparisons"]:
        for boundary in ("stage_a", "endpoint", "change_during_B"):
            for pool, value in pair[boundary].items():
                comparisons.append(
                    {
                        "contrast": pair["contrast"],
                        "initialization": pair["initialization"],
                        "hidden_size": pair["hidden_size"],
                        "base_blocks": pair["base_blocks"],
                        "repeats": pair["repeats"],
                        "data_sha256": pair["data_sha256"],
                        "both_complete": pair["both_complete"],
                        "boundary": boundary,
                        "pool": pool,
                        "difference": value,
                    }
                )
    write_csv(out / "endpoints.csv", endpoints)
    write_csv(out / "paired-comparisons.csv", comparisons)


def save_figures(summary, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    out = Path(out)
    paths = []
    axes_specs = (
        ("step", "Optimizer updates", "steps"),
        ("supervised_tokens", "Supervised tokens", "tokens"),
        ("estimated_matmul_training_flops", "Estimated training FLOPs", "flops"),
    )
    pools = ("atomic_A", "atomic_B", "AA", "BA", "AB", "BB")
    seeds = sorted({row["initialization"] for row in summary["runs"]})
    styles = {seed: ("-", "--", ":", "-.")[i % 4] for i, seed in enumerate(seeds)}
    with plt.rc_context({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False}):
        for cost, label, suffix in axes_specs:
            fig, axes = plt.subplots(2, 3, figsize=(13.2, 8), sharey=True)
            legend = {}
            for axis, pool in zip(axes.flat, pools, strict=True):
                for run in summary["runs"]:
                    values = [
                        item
                        for item in summary["curves"]
                        if item["name"] == run["name"]
                        and item["pool"] == pool
                        and item.get(cost) is not None
                        and item.get("answer_accuracy") is not None
                    ]
                    for scope in ("panel", "full"):
                        # Do not connect a pre-B panel to a post-B panel: the full
                        # boundary between them has a different question pool.
                        for stage in ("A", "B"):
                            selected = [
                                item
                                for item in values
                                if item["evaluation_scope"] == scope and item["stage"] == stage
                            ]
                            if not selected:
                                continue
                            name = (
                                f"{run['arm']} h{run['hidden_size']} seed {run['initialization']}"
                            )
                            line = axis.plot(
                                [item[cost] for item in selected],
                                [100 * item["answer_accuracy"] for item in selected],
                                color=COLORS[run["arm"]],
                                alpha=0.75,
                                linewidth=1,
                                linestyle=styles[run["initialization"]]
                                if scope == "panel"
                                else "none",
                                marker="." if scope == "panel" else "D",
                                markersize=4,
                                fillstyle="full" if scope == "panel" else "none",
                                label=name,
                            )[0]
                            legend.setdefault(name, line)
                axis.set_title(pool)
                axis.set_xlabel(label)
                axis.set_ylim(-2, 102)
                axis.grid(alpha=0.18)
                axis.ticklabel_format(axis="x", style="sci", scilimits=(-3, 4))
            axes[0, 0].set_ylabel("Exact answer accuracy (%)")
            axes[1, 0].set_ylabel("Exact answer accuracy (%)")
            handles = list(legend.values()) + [
                Line2D([], [], color="#333333", marker="D", linestyle="none", fillstyle="none")
            ]
            labels = list(legend) + ["Full endpoint (not panel)"]
            fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8)
            fig.suptitle(f"{summary['batch']} — {summary['state']}; seeds reuse the same data")
            fig.tight_layout(rect=(0, 0.12, 1, 0.95))
            path = out / f"learning-{suffix}.png"
            fig.savefig(path, dpi=180)
            plt.close(fig)
            paths.append(str(path))
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    config = read(args.config)
    root = args.root or Path(config["results_root"])
    out = args.out or root / "report"
    out.mkdir(parents=True, exist_ok=True)
    summary = summarize(config, root)
    summary["config_sha256"] = hashlib.sha256(args.config.read_bytes()).hexdigest()
    summary["report_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    write(out / "summary.json", summary)
    (out / "report.md").write_text(render(summary))
    export_csv(summary, out)
    figures = [] if args.no_figures else save_figures(summary, out)
    write(out / "figure-status.json", {"state": summary["state"], "files": figures})
    print(
        json.dumps({"out": str(out), "state": summary["state"], "runs": summary["expected_runs"]})
    )


if __name__ == "__main__":
    main()
