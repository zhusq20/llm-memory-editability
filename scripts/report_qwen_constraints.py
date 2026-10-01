#!/usr/bin/env python3
"""Audit and descriptive summaries; never treat prompts as independent trained models."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path.cwd()
ART = ROOT / "docs/development-artifacts/qwen-constraints-v1"
RESULTS = ROOT / "results/qwen-constraints-v1"


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def median(values):
    return float(np.median(list(values)))


def rows_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize():
    lock = read(ART / "lock.json")
    cfg = lock["config"]
    jobs = [f"layer-{layer}-{fmt}" for layer in cfg["layers"] for fmt in cfg["formats"]]
    supplement = ART / "final-layer-control-lock.json"
    if supplement.exists():
        jobs.extend(read(supplement)["jobs"])
    audit, geometry_rows, effect_rows, summary, route_rows, baseline_rows = [], [], [], [], [], []
    for job in jobs:
        folder = ART / job
        complete = read(folder / "completion.json")
        geometry = read(folder / "geometry.json")
        baseline = read(folder / "baseline.json")
        predictions = read(folder / "predictions.json")["rows"]
        path = RESULTS / job / "interventions.jsonl"
        actual = [json.loads(line) for line in path.read_text().splitlines()]
        assert len(actual) == len(predictions) == complete["interventions"] == 396
        assert digest(path) == complete["interventions_sha256"]
        assert digest(folder / "predictions.json") == complete["predictions_sha256"]
        for prediction, measured in zip(predictions, actual, strict=True):
            assert all(prediction[k] == measured[k] for k in prediction)
        assert (
            len(
                {
                    (v["case_id"], v["group"], v["method"], v["metric"], v["route"], v["step"])
                    for v in actual
                }
            )
            == 396
        )
        extra_count = 0
        extra_seconds = 0.0
        extra_folder = ART / "module-last" / job
        if (ART / "module-last-control-lock.json").exists():
            extra_complete = read(extra_folder / "completion.json")
            extra_path = RESULTS / "module-last" / job / "interventions.jsonl"
            extra_actual = [json.loads(line) for line in extra_path.read_text().splitlines()]
            extra_predictions = read(extra_folder / "predictions.json")["rows"]
            assert len(extra_actual) == len(extra_predictions) == 72
            assert digest(extra_path) == extra_complete["interventions_sha256"]
            assert digest(extra_folder / "predictions.json") == extra_complete["predictions_sha256"]
            for prediction, measured in zip(extra_predictions, extra_actual, strict=True):
                assert all(prediction[k] == measured[k] for k in prediction)
            assert extra_complete["restored_parameter_hash"] == complete["restored_parameter_hash"]
            assert extra_complete["restoration_max_margin_error"] == 0.0
            actual.extend(extra_actual)
            extra_count = len(extra_actual)
            extra_seconds = extra_complete["elapsed_seconds"]
            geometry["rows"].extend(
                {"kind": "module_last", "group": "down", "keep_count": 64, **v}
                for v in read(extra_folder / "geometry.json")["rows"]
            )
        zero_rows = read(folder / "zero-update-calibration.json")["rows"]
        assert len(zero_rows) == 36
        zero_lookup = {(v["case_id"], v["route"]): v for v in zero_rows}
        for value in actual:
            zero = zero_lookup[(value["case_id"], value["route"])]
            value["raw_actual_target"] = value["actual_target"]
            value["actual_target"] -= zero["target_zero_shift"]
            value["target_linear_error"] = value["actual_target"] - value["predicted_target"]
            for role, measures in value["roles"].items():
                corrected = np.asarray(measures["margin_changes"]) - np.asarray(
                    zero["roles"][role]["margin_changes"]
                )
                measures["raw_margin_rms"] = measures["margin_rms"]
                measures["margin_changes"] = corrected.tolist()
                measures["margin_rms"] = float(np.sqrt(np.mean(corrected**2)))
                measures["margin_max"] = float(np.max(np.abs(corrected)))
            if value["predicted_R"] is not None:
                value["R_prediction_rms_error"] = float(
                    np.sqrt(
                        np.mean(
                            (
                                np.array(value["roles"]["R"]["margin_changes"])
                                - np.array(value["predicted_R"])
                            )
                            ** 2
                        )
                    )
                )
        spectrum = geometry["weight_spectrum"]
        rows = geometry["rows"]
        for row in rows:
            geometry_rows.append({"job": job, **row})
        for role in ["E", "R", "U", "W"]:
            chosen = [v for v in baseline if v["role"] == role]
            baseline_rows.append(
                dict(
                    job=job,
                    role=role,
                    count=len(chosen),
                    first_token_accuracy=float(np.mean([v["first_token_correct"] for v in chosen])),
                    median_margin=median(v["margin"] for v in chosen),
                )
            )
        for group in cfg["groups"]:
            functional = [
                v
                for v in rows
                if v["kind"] == "functional" and v["group"] == group and v["keep_count"] == 64
            ]
            position = [v for v in rows if v["kind"] == "positions" and v["group"] == group]
            item = dict(
                job=job,
                group=group,
                down_rank=spectrum["ranks"]["1e-06"],
                down_condition=spectrum["condition"],
                functional_retained_energy=median(v["retained_energy"] for v in functional),
                functional_cost_factor=median(v["equal_target_cost_factor"] for v in functional),
                max_functional_projection_leak=max(v["relative_keep_residual"] for v in functional),
                last_full_cosine=median(v["last_full_cosine"] for v in position),
                earlier_predicted_fraction=median(v["earlier_response_fraction"] for v in position),
            )
            if group == "down":
                module = [
                    v
                    for v in rows
                    if v["kind"] == "module"
                    and v["keep_count"] == 64
                    and v["svd_rtol"] == cfg["feature_svd_rtol"]
                ]
                item.update(
                    module_retained_energy=median(v["retained_energy"] for v in module),
                    module_cost_factor=median(v["equal_target_cost_factor"] for v in module),
                    module_feature_rank=module[0]["rank"],
                    module_features=module[0]["token_features"],
                    module_cost_over_functional=median(
                        m["equal_target_cost_factor"] / f["equal_target_cost_factor"]
                        for m, f in zip(module, functional, strict=True)
                    ),
                )
                module_last = [v for v in rows if v["kind"] == "module_last"]
                if module_last:
                    item.update(
                        module_last_retained_energy=median(
                            v["retained_energy"] for v in module_last
                        ),
                        module_last_cost_factor=median(
                            v["equal_target_cost_factor"] for v in module_last
                        ),
                        module_last_max_feature_leak=max(
                            v["relative_last_feature_leak"] for v in module_last
                        ),
                    )
            summary.append(item)
        for value in actual:
            effect = {k: v for k, v in value.items() if k not in ["roles", "predicted_R"]}
            effect["job"] = job
            for role, measurements in value["roles"].items():
                effect.update(
                    {f"{role}_{k}": v for k, v in measurements.items() if k != "margin_changes"}
                )
            effect_rows.append(effect)
        lookup = {
            (v["case_id"], v["group"], v["method"], v["metric"], v["route"], v["step"]): v
            for v in actual
        }
        for v in actual:
            if v["group"] == "down" and v["method"] == "gradient" and v["route"] == "all":
                last = lookup[(v["case_id"], "down", "gradient", "equal_norm", "last", v["step"])]
                earlier = lookup[
                    (v["case_id"], "down", "gradient", "equal_norm", "earlier", v["step"])
                ]
                assert (
                    abs(
                        last["predicted_target"]
                        + earlier["predicted_target"]
                        - v["predicted_target"]
                    )
                    < 2e-6
                )
                route_rows.append(
                    dict(
                        job=job,
                        case_id=v["case_id"],
                        step=v["step"],
                        all_actual=v["actual_target"],
                        last_actual=last["actual_target"],
                        earlier_actual=earlier["actual_target"],
                        earlier_actual_fraction=earlier["actual_target"] / v["actual_target"],
                        earlier_predicted_fraction=earlier["predicted_target"]
                        / v["predicted_target"],
                        nonadditivity=v["actual_target"]
                        - last["actual_target"]
                        - earlier["actual_target"],
                    )
                )
        derivative = read(folder / "derivative-checks.json")
        audit.append(
            dict(
                job=job,
                interventions=len(actual),
                original_conditions=396,
                last_feature_followup_conditions=extra_count,
                prediction_hash_matches=True,
                raw_hash_matches=True,
                all_conditions_present=True,
                original_parameters_restored=True,
                restoration_margin_error=complete["restoration_max_margin_error"],
                max_chain_rule_relative_error=derivative["max_relative_error"],
                max_zero_update_margin_offset=max(
                    v["roles"][role]["margin_max"] for v in zero_rows for role in ["R", "U", "W"]
                ),
                zero_update_kl_floor=median(v["roles"]["R"]["kl_mean"] for v in zero_rows),
                matched_batch_margin_correction=True,
                capped=sum(v["capped"] for v in actual),
                elapsed_seconds=complete["elapsed_seconds"],
                last_feature_followup_seconds=extra_seconds,
            )
        )
    aggregates = []
    keys = sorted(
        {
            (v["job"], v["group"], v["method"], v["metric"], v["route"], v["step"])
            for v in effect_rows
        }
    )
    for key in keys:
        names = ["job", "group", "method", "metric", "route", "step"]
        selected = [v for v in effect_rows if tuple(v[n] for n in names) == key]
        row = dict(zip(names, key, strict=True))
        row["n"] = len(selected)
        for metric in [
            "actual_target",
            "predicted_target",
            "delta_norm",
            "relative_delta_norm",
            "R_margin_rms",
            "U_margin_rms",
            "W_margin_rms",
            "R_kl_mean",
            "U_kl_mean",
            "W_kl_mean",
            "R_first_token_accuracy",
            "U_first_token_accuracy",
        ]:
            row[metric] = median(v[metric] for v in selected)
        row["absolute_prediction_error"] = median(abs(v["target_linear_error"]) for v in selected)
        row["relative_prediction_error"] = median(
            abs(v["target_linear_error"]) / max(abs(v["predicted_target"]), 1e-8) for v in selected
        )
        row["target_positive_count"] = sum(v["actual_target"] > 0 for v in selected)
        aggregates.append(row)
    rows_csv(ART / "geometry.csv", geometry_rows)
    rows_csv(ART / "effects.csv", effect_rows)
    rows_csv(ART / "routes.csv", route_rows)
    rows_csv(ART / "baseline.csv", baseline_rows)
    rows_csv(ART / "summary.csv", summary)
    rows_csv(ART / "aggregate-effects.csv", aggregates)
    save(ART / "summary.json", dict(geometry=summary, effects=aggregates, baseline=baseline_rows))
    save(
        ART / "completion-audit.json",
        dict(
            jobs=audit,
            total_interventions=len(effect_rows),
            independent_pretrained_models=1,
            scope=cfg["scope"],
            selection_sha256=digest(ART / "selection.json"),
            all_pass=True,
        ),
    )
    plot(summary, route_rows)
    print(
        json.dumps(
            dict(
                jobs=len(jobs),
                interventions=len(effect_rows),
                audit="passed",
                down=[v for v in summary if v["group"] == "down"],
            ),
            indent=2,
        )
    )


def plot(summary, routes):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    jobs = list(dict.fromkeys(v["job"] for v in summary))
    values = [next(v for v in summary if v["job"] == job and v["group"] == "down") for job in jobs]
    labels = [job.replace("layer-", "L").replace("-", " ") for job in jobs]
    x = np.arange(len(jobs))
    axes[0].bar(
        x - 0.25,
        [v["functional_cost_factor"] for v in values],
        0.25,
        label="64 final-output margins",
        color="#3574aa",
    )
    axes[0].bar(
        x + 0.25,
        [v["module_cost_factor"] for v in values],
        0.25,
        label="All MLP vectors in R",
        color="#d68a36",
    )
    axes[0].bar(
        x,
        [v["module_last_cost_factor"] for v in values],
        0.25,
        label="64 last-position MLP vectors",
        color="#43866d",
    )
    axes[0].set_ylabel("Median local parameter cost / unconstrained")
    axes[0].set_title("Local cost depends on what is protected")
    axes[0].set_yscale("log")
    axes[0].set_ylim(0.85, 30)
    axes[0].set_yticks([1, 2, 5, 10, 20], ["1", "2", "5", "10", "20"])
    axes[0].axhline(1, color="gray", lw=0.7)
    axes[0].legend(fontsize=8)
    measured = [
        median(v["earlier_actual_fraction"] for v in routes if v["job"] == job and v["step"] == 0.1)
        for job in jobs
    ]
    axes[1].bar(x, measured, 0.65, color="#43866d", label="Actual earlier-position-only effect")
    axes[1].scatter(
        x,
        [v["earlier_predicted_fraction"] for v in values],
        color="#292e35",
        s=24,
        label="First-order prediction",
        zorder=3,
    )
    axes[1].set_ylabel("Earlier-only / full target margin change")
    axes[1].set_title("Earlier-position changes affect the final output")
    axes[1].set_ylim(-0.03, 1.04)
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.set_xticks(x, labels, rotation=35, ha="right")
    fig.suptitle("Qwen3-0.6B-Base | 12 paired targets | one pretrained checkpoint", fontsize=12)
    fig.tight_layout()
    fig.savefig(ART / "constraints.png", dpi=180)
    fig.savefig(ART / "constraints.pdf")
    plt.close(fig)


if __name__ == "__main__":
    summarize()
