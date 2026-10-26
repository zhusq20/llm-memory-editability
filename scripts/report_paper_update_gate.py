#!/usr/bin/env python3
"""Compare audited continuous-update gates with the read-only paper baseline.

This report only reads existing results. It performs no model evaluation,
training, or network publication; all eleven panels and R1--4 are retained.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from llm_memory_editability.grok_depth import utc  # noqa: E402
from llm_memory_editability.grokking_reproduction import digest  # noqa: E402
from llm_memory_editability.loop_hop_data import panel_names  # noqa: E402
from llm_memory_editability.paper_update_contract import verify_baseline  # noqa: E402

ARTIFACTS = ("complete.json", "audit.json", "grid.json", "grid-audit.json")


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def same_metrics(left, right):
    """Compare the native metrics, allowing grid-only common-premise fields."""
    return all(
        key in right
        and (
            right[key] is None
            if value is None
            else right[key] is not None and np.isclose(value, right[key], atol=1e-6, rtol=1e-6)
        )
        for key, value in left.items()
    )


def verified_condition(config, name, folder, config_sha256, baseline_counts):
    payload = {filename: read(folder / filename) for filename in ARTIFACTS}
    complete, audit, grid, grid_audit = [payload[filename] for filename in ARTIFACTS]
    steps = config["training"]["steps"]
    structure = config["arms"][name]
    origin = config["input_origin"]
    identity = f"Unverified {name}"
    require(
        complete["state"] == "complete"
        and complete["arm"] == name
        and complete["step"] == complete["budget"] == steps
        and complete["engineering_only"] is False
        and complete["config_sha256"] == config_sha256
        and complete["world_sha256"] == origin["world_sha256"]
        and complete["endpoint"]["step"] == steps,
        f"{identity}: training endpoint identity or budget changed",
    )
    require(
        audit["passed"] is True
        and audit["arm"] == name
        and audit["step"] == steps
        and audit["exposure_match"] is True
        and set(audit["panels"]) == set(panel_names())
        and all(
            row["passed"] is True and row["answer_and_eos_ids_exact"] is True
            for row in audit["panels"].values()
        ),
        f"{identity}: independent endpoint audit failed or is incomplete",
    )
    require(
        grid["arm"] == name
        and grid["step"] == steps
        and grid["training_repeats"] == structure["repeats"]
        and grid["unique_layers"] == structure["unique_layers"]
        and grid["engineering_only"] is False
        and grid["config_sha256"] == config_sha256
        and grid["world_sha256"] == origin["world_sha256"]
        and grid["panels_sha256"] == origin["panels_sha256"]
        and grid["model_sha256"] == complete["model_sha256"]
        and grid["initial_model_sha256"] == complete["initial_model_sha256"],
        f"{identity}: grid/checkpoint/config/data identity changed",
    )
    repeats = [str(value) for value in range(1, structure["repeats"] + 1)]
    expected_findings = {f"r{value}/{panel}" for value in repeats for panel in panel_names()}
    require(
        grid_audit["passed"] is True
        and grid_audit["model_sha256"] == complete["model_sha256"]
        and grid_audit["grid_sha256"] == digest(folder / "grid.json")
        and set(grid_audit["findings"]) == expected_findings
        and all(value is True for value in grid_audit["findings"].values()),
        f"{identity}: independent inference-grid audit failed or changed",
    )
    require(set(grid["metrics"]) == set(repeats), f"{identity}: incomplete inference grid")
    endpoint_metrics = complete["endpoint"]["metrics"]
    require(set(endpoint_metrics) == set(panel_names()), f"{identity}: missing endpoint panel")
    for repeat in repeats:
        require(
            set(grid["metrics"][repeat]) == set(panel_names()),
            f"{identity}: missing panel at R{repeat}",
        )
    for panel in panel_names():
        require(
            same_metrics(endpoint_metrics[panel], audit["panels"][panel]["metrics"])
            and same_metrics(
                endpoint_metrics[panel], grid["metrics"][str(structure["repeats"])][panel]
            ),
            f"{identity}: endpoint/audit/grid metrics differ for {panel}",
        )
    checkpoint = folder / f"checkpoint-{steps:07d}.pt"
    require(
        digest(checkpoint) == complete["final_checkpoint_sha256"],
        f"{identity}: endpoint checkpoint bytes changed",
    )
    with np.load(folder / "exposure.npz") as saved:
        require(
            np.array_equal(saved["counts"], baseline_counts),
            f"{identity}: sample exposure differs from the reused paper baseline",
        )
    return {
        **payload,
        "reused": False,
        "sample_exposure_equal_to_paper": True,
        "files": {
            filename: digest(folder / filename)
            for filename in (*ARTIFACTS, "exposure.npz", checkpoint.name)
        },
    }


def build_report(config_path, base, baseline_dir=None):
    config_path, base = Path(config_path), Path(base)
    config = read(config_path)
    verification = verify_baseline(config, baseline_dir=baseline_dir)
    baseline = Path(verification["baseline_run"])
    with np.load(baseline / "exposure.npz") as saved:
        baseline_counts = saved["counts"].copy()
    conditions = {
        "paper-reused": {
            **{filename: read(baseline / filename) for filename in ARTIFACTS},
            "reused": True,
            "sample_exposure_equal_to_paper": True,
            "files": {filename: digest(baseline / filename) for filename in ARTIFACTS},
        }
    }
    config_sha256 = digest(config_path)
    for name in config["arms"]:
        conditions[name] = verified_condition(
            config, name, base / "runs" / name, config_sha256, baseline_counts
        )
    rows = []
    baseline_grid = conditions["paper-reused"]["grid.json"]["metrics"]
    for name, condition in conditions.items():
        grid = condition["grid.json"]["metrics"]
        require(set(grid) == set(baseline_grid), f"{name}: inference budgets differ from baseline")
        for repeat, metrics in grid.items():
            for panel in panel_names():
                reference = baseline_grid[repeat][panel]
                require(
                    set(metrics[panel]) == set(reference) and metrics[panel]["n"] == reference["n"],
                    f"{name}: panel metrics or coverage differ from baseline at R{repeat}/{panel}",
                )
                rows.append(
                    {
                        "condition": name,
                        "reused": condition["reused"],
                        "training_step": condition["complete.json"]["step"],
                        "inference_repeats": int(repeat),
                        "panel": panel,
                        **metrics[panel],
                    }
                )
    repeats = str(next(iter(config["arms"].values()))["repeats"])
    primary = {
        name: condition["grid.json"]["metrics"][repeats]["ood_3"]["answer_accuracy"]
        for name, condition in conditions.items()
    }
    fixed = next(name for name, arm in config["arms"].items() if arm["update_mode"] == "fixed")
    token = next(name for name, arm in config["arms"].items() if arm["update_mode"] == "token")
    report = {
        "batch": config["batch"],
        "config_sha256": config_sha256,
        "primary": config["primary"],
        "primary_answer_accuracy": primary,
        "primary_difference_percentage_points": {
            "fixed_minus_paper": 100 * (primary[fixed] - primary["paper-reused"]),
            "token_minus_fixed": 100 * (primary[token] - primary[fixed]),
            "token_minus_paper": 100 * (primary[token] - primary["paper-reused"]),
        },
        "conditions": conditions,
        "baseline_verification": verification,
        "new_scientific_updates": sum(
            condition["complete.json"]["step"]
            for condition in conditions.values()
            if not condition["reused"]
        ),
        "baseline_new_updates": 0,
        "all_local_audits_passed": True,
        "cloud_verification": "Not performed by this local report; inspect per-arm cloud audits.",
        "scope": "One previously observed development graph and initialization; no new evaluation.",
        "utc": utc(),
    }
    return report, rows


def write_report(base, report, rows):
    base = Path(base)
    destinations = [base / "comparison.json", base / "comparison.csv"]
    if any(path.exists() for path in destinations):
        raise FileExistsError("Preserve the existing comparison.json and comparison.csv")
    leading = ["condition", "reused", "training_step", "inference_repeats", "panel"]
    fields = leading + sorted({key for row in rows for key in row} - set(leading))
    with destinations[0].open("x") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    with destinations[1].open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    report, rows = build_report(args.config, args.base, args.baseline)
    write_report(args.base, report, rows)
    print(json.dumps(report["primary_difference_percentage_points"], allow_nan=False))


if __name__ == "__main__":
    main()
