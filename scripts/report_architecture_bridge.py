#!/usr/bin/env python3
"""Read-only outcome aggregation; write summary/audit/plot without selecting runs.

Evaluation and development are separate. Missing/excluded cases stay in the
audit. Bootstrap samples case groups, keeping paired conditions together; layers
are reported separately. Learning U is used only for reporting, never selection.
An incomplete run produces an explicitly partial report, not a completion claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
METRICS = ("em", "f1", "alias_em", "margin")
BEHAVIORS = ("unassisted", "first_hop", "second_hop", "oracle_input")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def metric(row, name):
    value = row.get("scores", {}).get("margin") if name == "margin" else row.get(name)
    if isinstance(value, (float, int, bool)) and np.isfinite(value):
        return float(value)
    return None


def average(values):
    values = [
        value for value in values if isinstance(value, (int, float, bool)) and np.isfinite(value)
    ]
    return float(np.mean(values)) if values else None


def cluster_interval(values, groups, repetitions=2000, seed=142935):
    """Case-weighted mean and paired cluster percentile CI, resampling groups."""
    if len(values) != len(groups):
        raise ValueError("One group is required per case value.")
    buckets = defaultdict(list)
    for value, group in zip(values, groups, strict=True):
        if not np.isfinite(value):
            raise ValueError("Nonfinite case differences cannot enter a bootstrap.")
        buckets[str(group)].append(float(value))
    result = {"estimate": average(values), "n_cases": len(values), "n_groups": len(buckets)}
    if len(buckets) < 2:
        return {**result, "ci95": [None, None], "interval_status": "insufficient_groups"}
    sums = np.array([sum(buckets[key]) for key in sorted(buckets)], dtype=np.float64)
    counts = np.array([len(buckets[key]) for key in sorted(buckets)], dtype=np.float64)
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(buckets), size=(repetitions, len(buckets)))
    draws = sums[sampled].sum(axis=1) / counts[sampled].sum(axis=1)
    return {**result, "ci95": np.quantile(draws, [0.025, 0.975]).tolist(), "interval_status": "ok"}


def aggregate_predictions(predictions):
    result = {"n": len(predictions)}
    for name in METRICS:
        values = [metric(row, name) for row in predictions]
        result[name] = average(values)
        result[name + "_n"] = sum(value is not None for value in values)
    result["ended_eos"] = average(
        [float(row["ended_eos"]) for row in predictions if "ended_eos" in row]
    )
    result["correct_mean_logp"] = average(
        [row.get("scores", {}).get("correct", {}).get("mean_logp") for row in predictions]
    )
    result["correct_sum_logp"] = average(
        [row.get("scores", {}).get("correct", {}).get("sum_logp") for row in predictions]
    )
    return result


def contrast(cases, layer, coefficients, name, repetitions, seed):
    values, groups = [], []
    for case in cases:
        indexed = {(row["layer"], row["condition"]): row for row in case.get("interventions", [])}
        rows = [indexed.get((layer, condition)) for condition in coefficients]
        if any(row is None for row in rows):
            continue
        terms = [metric(row, name) for row in rows]
        if any(value is None for value in terms):
            continue
        values.append(
            sum(
                coefficient * value
                for coefficient, value in zip(coefficients.values(), terms, strict=True)
            )
        )
        groups.append(case["group"])
    return cluster_interval(values, groups, repetitions, seed)


def prediction_errors(prediction, eos, budget):
    errors = []
    for name in METRICS:
        value = metric(prediction, name)
        if value is None or (name != "margin" and not 0 <= value <= 1):
            errors.append("invalid_" + name)
    tokens = prediction.get("tokens", [])
    if not tokens or len(tokens) > budget:
        errors.append("invalid_generation_length")
    if eos is not None and tokens and bool(prediction.get("ended_eos")) != (tokens[-1] == eos):
        errors.append("ended_eos_mismatch")
    scores = prediction.get("scores", {})
    for role in ("correct", "competitor"):
        score = scores.get(role, {})
        target = score.get("tokens", [])
        if not target or (eos is not None and target[-1] != eos):
            errors.append(role + "_missing_eos_target")
        total, mean = score.get("sum_logp"), score.get("mean_logp")
        if total is None or mean is None or not np.isfinite(total) or not np.isfinite(mean):
            errors.append(role + "_nonfinite_logp")
        elif target and not np.isclose(total / len(target), mean, atol=1e-6, rtol=1e-6):
            errors.append(role + "_sum_mean_inconsistent")
    if all(role in scores and "mean_logp" in scores[role] for role in ("correct", "competitor")):
        difference = scores["correct"]["mean_logp"] - scores["competitor"]["mean_logp"]
        if not np.isclose(difference, scores.get("margin", np.nan), atol=1e-6, rtol=1e-6):
            errors.append("margin_inconsistent")
    return errors


def expected_eos(root, model):
    path = root / model["path"] / "config.json"
    if not path.exists():
        return None
    config = read(path)
    value = config.get("text_config", config).get("eos_token_id")
    return value if isinstance(value, int) else None


def build_report(root=ROOT):
    root = Path(root)
    config_path = root / "configs/architecture-bridge-v1.json"
    config = read(config_path)
    data = root / "data/architecture-bridge-v1"
    artifacts = root / "docs/development-artifacts/architecture-bridge-v1"
    results = root / "results/architecture-bridge-v1"
    checks, loaded, learning = [], defaultdict(list), defaultdict(list)
    result_files = {}

    def check(name, passed, kind="integrity", detail=None):
        checks.append({"name": name, "passed": bool(passed), "kind": kind, "detail": detail})

    check("frozen_cases_present", (data / "cases.json").exists(), "completion")
    cases = read(data / "cases.json") if (data / "cases.json").exists() else []
    pools = read(data / "learning.json") if (data / "learning.json").exists() else {}
    case_index = {(case["dataset"], case["id"]): case for case in cases}
    check("frozen_case_ids_unique", len(case_index) == len(cases))
    expected_per_model = len(config["datasets"]) * sum(config["cases_per_dataset"].values())
    check("frozen_case_count", len(cases) == expected_per_model, "completion")
    if cases:
        for dataset in config["datasets"]:
            for split, count in config["cases_per_dataset"].items():
                check(
                    f"frozen_count:{dataset}:{split}",
                    sum(case["dataset"] == dataset and case["split"] == split for case in cases)
                    == count,
                )
    lock_path = artifacts / "data-lock.json"
    check("data_lock_present", lock_path.exists(), "completion")
    if lock_path.exists():
        lock = read(lock_path)
        for name, path in [
            ("config", config_path),
            ("cases", data / "cases.json"),
            ("learning", data / "learning.json"),
        ]:
            check(
                f"frozen_hash:{name}", path.exists() and digest(path) == lock.get(name + "_sha256")
            )
    execution_path = artifacts / "execution-lock.json"
    check("execution_lock_present", execution_path.exists(), "completion")
    if execution_path.exists():
        execution = read(execution_path)
        for name, expected in execution.get("files", {}).items():
            path = root / name
            check("execution_source_hash:" + name, path.exists() and digest(path) == expected)
        if lock_path.exists():
            check(
                "execution_data_lock_hash", execution.get("data_lock_sha256") == digest(lock_path)
            )
    learning_config = config["local_learning"]
    if pools:
        group_sets = {role: {row["group"] for row in pools[role]} for role in ("E", "R", "U")}
        check(
            "learning_E_R_U_group_disjoint",
            not (
                group_sets["E"] & group_sets["R"]
                or group_sets["E"] & group_sets["U"]
                or group_sets["R"] & group_sets["U"]
            ),
        )
        for role, count in [
            ("E", learning_config["episodes"]),
            ("R", learning_config["keep_R"]),
            ("U", learning_config["unseen_U"]),
        ]:
            check("learning_pool_count:" + role, len(pools[role]) == count)
    else:
        check("learning_pools_present", False, "completion")
    exclusions, missing_cases, missing_episodes = [], [], []
    for model, model_config in config["models"].items():
        eos = expected_eos(root, model_config)
        check(f"{model}:eos_config_available", eos is not None, "completion")
        observed_keys = []
        for path in sorted((results / model / "cases").glob("*.json")):
            result_files[str(path.relative_to(root))] = digest(path)
            try:
                result = read(path)
            except (ValueError, OSError) as error:
                check(f"{model}:read:{path.name}", False, detail=str(error))
                continue
            key = (result.get("dataset"), result.get("id"))
            check(f"{model}:known_case:{path.name}", key in case_index)
            if key not in case_index:
                continue
            if key in observed_keys:
                check(f"{model}:duplicate_case:{path.name}", False)
                continue
            observed_keys.append(key)
            reference = case_index[key]
            label = f"{model}:{key[0]}:{key[1]}"
            metadata_ok = all(
                result.get(field) == reference.get(field)
                for field in ("split", "group", "exclusion_reason")
            )
            check(label + ":metadata", metadata_ok)
            rows = result.get("interventions", [])
            actual = [(row.get("layer"), row.get("condition")) for row in rows]
            expected = (
                {
                    (layer, condition)
                    for layer in model_config["layers"]
                    for condition in config["conditions"]
                }
                if reference.get("exclusion_reason") is None
                else set()
            )
            check(
                label + ":condition_rows",
                len(actual) == len(expected) and set(actual) == expected,
                "completion",
                {"expected": len(expected), "observed": len(actual)},
            )
            check(label + ":condition_rows_unique", len(actual) == len(set(actual)))
            check(label + ":no_unregistered_condition", not (set(actual) - expected))
            behaviors = list(BEHAVIORS if expected else BEHAVIORS[:3])
            for behavior in behaviors:
                check(label + ":behavior:" + behavior, behavior in result, "completion")
            all_predictions = [
                (behavior, result[behavior]) for behavior in behaviors if behavior in result
            ]
            all_predictions += [(f"L{row['layer']}:{row['condition']}", row) for row in rows]
            for condition, prediction in all_predictions:
                errors = prediction_errors(prediction, eos, config["generation_tokens"])
                check(label + ":prediction:" + condition, not errors, detail=errors or None)
            indexed = {(row["layer"], row["condition"]): row for row in rows}
            for layer in model_config["layers"]:
                baseline, identity = (
                    indexed.get((layer, "baseline")),
                    indexed.get((layer, "identity")),
                )
                if baseline is not None and identity is not None:
                    margins = [metric(row, "margin") for row in (baseline, identity)]
                    equal = (
                        baseline.get("tokens") == identity.get("tokens")
                        and None not in margins
                        and abs(margins[0] - margins[1]) <= 1e-5
                    )
                    check(label + f":identity:L{layer}", equal)
                for channel in ("mlp", "state", "joint"):
                    correct = indexed.get((layer, channel + "_correct"))
                    random = indexed.get((layer, channel + "_random"))
                    if correct is None or random is None:
                        continue
                    norm_keys = [key for key in correct if key.endswith("_delta_norm")]
                    matched = bool(norm_keys) and all(
                        key in random
                        and np.isclose(correct[key], random[key], atol=1e-6, rtol=1e-5)
                        for key in norm_keys
                    )
                    check(label + f":random_norm_match:L{layer}:{channel}", matched)
            if metadata_ok:
                loaded[model].append(result)
        missing = sorted(set(case_index) - set(observed_keys))
        missing_cases += [
            {"model": model, "dataset": dataset, "id": identifier}
            for dataset, identifier in missing
        ]
        check(
            model + ":case_results_complete",
            len(observed_keys) == expected_per_model and not missing,
            "completion",
            {"expected": expected_per_model, "observed": len(observed_keys)},
        )
        exclusions += [
            {
                "model": model,
                "dataset": case["dataset"],
                "split": case["split"],
                "id": case["id"],
                "reason": case["exclusion_reason"],
                "result_present": (case["dataset"], case["id"]) in observed_keys,
            }
            for case in cases
            if case.get("exclusion_reason")
        ]
        for episode in range(learning_config["episodes"]):
            folder = results / model / "learning" / str(episode)
            path = folder / "complete.json"
            if not path.exists():
                missing_episodes.append({"model": model, "episode": episode})
                continue
            result_files[str(path.relative_to(root))] = digest(path)
            result = read(path)
            label = f"{model}:learning:{episode}"
            check(label + ":episode_identity", result.get("episode") == episode)
            check(
                label + ":target_identity",
                bool(pools) and result.get("target_id") == pools["E"][episode]["id"],
            )
            check(label + ":original_restored", result.get("restored_original") is True)
            check(
                label + ":endpoint_weight_present", (folder / "endpoint.pt").exists(), "completion"
            )
            steps = [node.get("step") for node in result.get("nodes", [])]
            check(label + ":nodes_complete", steps == learning_config["nodes"], "completion")
            training_steps = [row.get("step") for row in result.get("training", [])]
            check(
                label + ":training_steps",
                training_steps == list(range(1, learning_config["steps"] + 1)),
                "completion",
            )
            for node in result.get("nodes", []):
                check(
                    label + f":step{node['step']}:roles_present",
                    all(role in node for role in ("E", "D", "R", "U")),
                    "completion",
                )
                for role in ("R", "U"):
                    expected_ids = [row["id"] for row in pools.get(role, [])]
                    check(
                        label + f":step{node['step']}:{role}_ids",
                        [row.get("id") for row in node.get(role, [])] == expected_ids,
                    )
                for role in ("E", "D", "R", "U"):
                    predictions = (
                        [node[role]] if role in ("E", "D") and role in node else node.get(role, [])
                    )
                    for index, prediction in enumerate(predictions):
                        errors = prediction_errors(prediction, eos, config["generation_tokens"])
                        check(
                            label + f":step{node['step']}:{role}:{index}",
                            not errors,
                            detail=errors or None,
                        )
            learning[model].append(result)
        found_episodes = {
            path.parent.name for path in (results / model / "learning").glob("*/complete.json")
        }
        check(
            model + ":learning_episode_set",
            found_episodes == {str(index) for index in range(learning_config["episodes"])},
            "completion",
        )
    summaries, behaviors, contrasts, learning_summary, learning_changes = [], [], [], [], []
    for model, model_config in config["models"].items():
        for dataset in config["datasets"]:
            for split in config["cases_per_dataset"]:
                selected = [
                    case for case in cases if case["dataset"] == dataset and case["split"] == split
                ]
                present = [
                    case
                    for case in loaded[model]
                    if case["dataset"] == dataset and case["split"] == split
                ]
                eligible = [case for case in present if case.get("exclusion_reason") is None]
                expected_eligible = sum(case.get("exclusion_reason") is None for case in selected)
                common = {
                    "model": model,
                    "dataset": dataset,
                    "split": split,
                    "expected_selected": len(selected),
                    "expected_eligible": expected_eligible,
                }
                for population, items in [
                    ("all_selected", present),
                    ("eligible_interventions", eligible),
                ]:
                    for condition in BEHAVIORS:
                        if condition == "oracle_input" and population == "all_selected":
                            continue
                        predictions = [case[condition] for case in items if condition in case]
                        behaviors.append(
                            {
                                **common,
                                "population": population,
                                "condition": condition,
                                **aggregate_predictions(predictions),
                            }
                        )
                for layer in model_config["layers"]:
                    for condition in config["conditions"]:
                        predictions = [
                            row
                            for case in eligible
                            for row in case.get("interventions", [])
                            if row.get("layer") == layer and row.get("condition") == condition
                        ]
                        summaries.append(
                            {
                                **common,
                                "layer": layer,
                                "condition": condition,
                                **aggregate_predictions(predictions),
                            }
                        )
                    specifications = []
                    for channel in ("mlp", "state", "joint"):
                        for control in ("baseline", f"{channel}_wrong", f"{channel}_random"):
                            specifications.append(
                                (
                                    f"{channel}_correct_minus_{control}",
                                    {f"{channel}_correct": 1, control: -1},
                                )
                            )
                    for donor in ("correct", "wrong", "random"):
                        specifications.append(
                            (
                                f"interaction_{donor}",
                                {
                                    f"joint_{donor}": 1,
                                    f"mlp_{donor}": -1,
                                    f"state_{donor}": -1,
                                    "baseline": 1,
                                },
                            )
                        )
                    for comparison, coefficients in specifications:
                        for name in METRICS:
                            interval = contrast(
                                eligible,
                                layer,
                                coefficients,
                                name,
                                config["bootstrap"],
                                config["seed"],
                            )
                            contrasts.append(
                                {
                                    **common,
                                    "layer": layer,
                                    "comparison": comparison,
                                    "coefficients": coefficients,
                                    "metric": name,
                                    **interval,
                                }
                            )
        for step in learning_config["nodes"]:
            for role in ("E", "D", "R", "U"):
                by_episode = []
                for episode in learning[model]:
                    node = next((node for node in episode["nodes"] if node["step"] == step), None)
                    if node is None or role not in node:
                        continue
                    predictions = [node[role]] if role in ("E", "D") else node[role]
                    by_episode.append(
                        {
                            **aggregate_predictions(predictions),
                            "kl": average([row.get("kl") for row in predictions]),
                            "parameter_delta_norm": node.get("parameter_delta_norm"),
                        }
                    )
                learning_summary.append(
                    {
                        "model": model,
                        "step": step,
                        "role": role,
                        "unit": "episode; first average R/U cases within each episode",
                        "n_episodes": len(by_episode),
                        "n_case_observations": sum(row["n"] for row in by_episode),
                        **{
                            name: average([row.get(name) for row in by_episode])
                            for name in (*METRICS, "kl", "parameter_delta_norm")
                        },
                    }
                )
        for role in ("E", "D", "R", "U"):
            for name in (*METRICS, "kl"):
                changes, groups = [], []
                for episode in learning[model]:
                    nodes = {node["step"]: node for node in episode["nodes"]}
                    if (
                        0 not in nodes
                        or learning_config["steps"] not in nodes
                        or role not in nodes[0]
                        or role not in nodes[learning_config["steps"]]
                    ):
                        continue
                    means = []
                    for step in (0, learning_config["steps"]):
                        predictions = (
                            [nodes[step][role]] if role in ("E", "D") else nodes[step][role]
                        )
                        means.append(
                            average(
                                [
                                    row.get("kl") if name == "kl" else metric(row, name)
                                    for row in predictions
                                ]
                            )
                        )
                    if None not in means:
                        changes.append(means[1] - means[0])
                        groups.append(episode["target_id"])
                learning_changes.append(
                    {
                        "model": model,
                        "role": role,
                        "metric": name,
                        "comparison": "final_minus_initial",
                        "unit": "target episode",
                        **cluster_interval(changes, groups, config["bootstrap"], config["seed"]),
                    }
                )
    integrity_failures = [
        check for check in checks if not check["passed"] and check["kind"] == "integrity"
    ]
    incomplete = [
        check for check in checks if not check["passed"] and check["kind"] == "completion"
    ]
    complete = not integrity_failures and not incomplete
    status = "complete" if complete else "invalid" if integrity_failures else "partial"
    counts = {
        "expected_case_results": len(config["models"]) * expected_per_model,
        "observed_case_results": sum(len(value) for value in loaded.values()),
        "expected_eligible_condition_rows": sum(
            sum(case.get("exclusion_reason") is None for case in cases)
            * len(model["layers"])
            * len(config["conditions"])
            for model in config["models"].values()
        ),
        "observed_condition_rows": sum(
            len(case.get("interventions", [])) for values in loaded.values() for case in values
        ),
        "expected_learning_episodes": len(config["models"]) * learning_config["episodes"],
        "observed_learning_episodes": sum(len(value) for value in learning.values()),
    }
    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "report_script_sha256": digest(Path(__file__)),
        "result_sha256": result_files,
        "config_sha256": digest(config_path),
    }
    summary = {
        "experiment": config["experiment"],
        "status": status,
        "complete": complete,
        "counts": counts,
        "primary_split": "evaluation",
        "bootstrap": {
            "replicates": config["bootstrap"],
            "seed": config["seed"],
            "unit": "source case group; paired conditions kept together; layers separate",
            "interval": "95% percentile; exploratory, no multiplicity correction",
            "one_group": "CI omitted",
        },
        "interpretation": (
            "Oracle-assisted paired diagnostics. Family score differences are not causal "
            "architecture effects. No U-based selection or fitting. "
            "Partial tables do not imply completion."
        ),
        "interventions": summaries,
        "behavior": behaviors,
        "paired_contrasts": contrasts,
        "learning": learning_summary,
        "learning_changes": learning_changes,
        "provenance": provenance,
    }
    audit = {
        "experiment": config["experiment"],
        "status": status,
        "complete": complete,
        "counts": counts,
        "integrity_passed": not integrity_failures,
        "checks": checks,
        "integrity_failures": integrity_failures,
        "incomplete_requirements": incomplete,
        "excluded_cases": exclusions,
        "missing_case_results": missing_cases,
        "missing_learning_episodes": missing_episodes,
        "not_verified": [
            "semantic correctness of every benchmark annotation",
            "causal specificity of oracle donor interventions",
            "unrecorded model internals; this is an artifact audit",
        ],
        "provenance": provenance,
    }
    return summary, audit, config


def plot_summary(summary, config, destination):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = list(config["models"])
    conditions = [condition for condition in config["conditions"] if condition != "identity"]
    fig, axes = plt.subplots(
        len(config["datasets"]) + 1, len(models), figsize=(15, 12), squeeze=False
    )
    labels = [
        condition.replace("_correct", "+").replace("_wrong", " wrong").replace("_random", " rand")
        for condition in conditions
    ]
    for column, model in enumerate(models):
        for row, dataset in enumerate(config["datasets"]):
            ax = axes[row, column]
            plotted = False
            for layer in config["models"][model]["layers"]:
                selected = {
                    item["condition"]: item
                    for item in summary["interventions"]
                    if item["model"] == model
                    and item["dataset"] == dataset
                    and item["split"] == "evaluation"
                    and item["layer"] == layer
                }
                values = [selected[condition]["em"] for condition in conditions]
                if any(value is not None for value in values):
                    ax.plot(
                        range(len(conditions)),
                        [np.nan if value is None else value for value in values],
                        marker="o",
                        linewidth=1,
                        label=f"Layer {layer}; n={selected['baseline']['n']}",
                    )
                    plotted = True
            ax.set(
                title=f"{model} / {dataset} / evaluation",
                ylabel="Canonical exact match",
                ylim=(-0.03, 1.03),
            )
            ax.set_xticks(range(len(conditions)), labels, rotation=45, ha="right", fontsize=8)
            ax.grid(axis="y", alpha=0.2)
            if plotted:
                ax.legend(fontsize=8)
            else:
                ax.text(
                    0.5, 0.5, "No completed evaluation cases", ha="center", transform=ax.transAxes
                )
        ax = axes[-1, column]
        plotted = False
        for role in ("E", "D", "R", "U"):
            selected = sorted(
                [
                    item
                    for item in summary["learning"]
                    if item["model"] == model and item["role"] == role
                ],
                key=lambda item: item["step"],
            )
            if any(item["em"] is not None for item in selected):
                ax.plot(
                    [item["step"] for item in selected],
                    [np.nan if item["em"] is None else item["em"] for item in selected],
                    marker="o",
                    label=role,
                )
                plotted = True
        ax.set(
            title=f"{model} / local learning",
            xlabel="SGD step",
            ylabel="Episode mean exact match",
            ylim=(-0.03, 1.03),
        )
        ax.grid(alpha=0.2)
        if plotted:
            ax.legend()
        else:
            ax.text(0.5, 0.5, "No completed learning episodes", ha="center", transform=ax.transAxes)
    counts = summary["counts"]
    fig.suptitle(
        f"Architecture bridge: {summary['status'].upper()} | "
        f"cases {counts['observed_case_results']}/{counts['expected_case_results']} | "
        f"episodes {counts['observed_learning_episodes']}/{counts['expected_learning_episodes']}\n"
        "Oracle-assisted diagnostics; layers paired within cases; development reported separately",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.root / "docs/development-artifacts/architecture-bridge-v1"
    summary, audit, config = build_report(args.root)
    write(output / "summary.json", summary)
    write(output / "audit.json", audit)
    plot_summary(summary, config, output / "summary.png")
    print(
        json.dumps(
            {
                "status": summary["status"],
                "complete": summary["complete"],
                **summary["counts"],
                "integrity_failures": len(audit["integrity_failures"]),
            }
        )
    )


if __name__ == "__main__":
    main()
