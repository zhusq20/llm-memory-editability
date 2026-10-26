#!/usr/bin/env python3
"""Summarize frozen endpoints and paired worlds without selecting checkpoints."""

from __future__ import annotations

import argparse
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ARMS = ("paper", "fixed-update", "token-update")
PAIRS = {"token_minus_fixed": "fixed-update", "token_minus_paper": "paper"}
COST = (
    "examples",
    "effective_input_tokens",
    "executed_input_tokens",
    "supervised_tokens",
    "estimated_matmul_training_flops",
    "training_seconds",
    "wall_seconds",
)


def read(path):
    return json.loads(Path(path).read_text())


def thresholds(nodes):
    result = {}
    for target in (0.50, 0.75, 0.90, 0.95):
        found, previous = None, None
        for row in nodes:
            if row["primary_score"] >= target:
                found = {
                    "lower_step": previous,
                    "upper_step": row["step"],
                    "left_censored": previous is None,
                }
                break
            previous = row["step"]
        result[str(int(target * 100))] = found
    return result


def inspect_run(folder, arm, budget, panel, score, repeats):
    result = {"path": str(folder), "state": "pending", "passed": None, "nodes": []}
    try:
        case_root = folder.parent.parent
        for stage in ("train", "grid"):
            failure = case_root / "failures" / f"{arm}-{stage}.json"
            if failure.exists():
                raise ValueError(f"{stage} failure recorded: {read(failure)}")
            exit_path = case_root / "execution" / arm / stage / "exit.json"
            if exit_path.exists() and read(exit_path)["code"] != 0:
                raise ValueError(f"{stage} container exited unsuccessfully: {read(exit_path)}")
        learning = read(folder / "learning.json") if (folder / "learning.json").exists() else []
        steps = [row["step"] for row in learning]
        if steps != sorted(set(steps)):
            raise ValueError("Learning nodes are duplicated or out of order")
        result["nodes"] = [
            {
                "step": row["step"],
                "primary_score": row["metrics"][panel][score],
                **{key: row[key] for key in COST if key in row},
            }
            for row in learning
        ]
        if any(not 0 <= row["primary_score"] <= 1 for row in result["nodes"]):
            raise ValueError("Learning score must be finite and between zero and one")
        result["threshold_first_observed_intervals"] = thresholds(result["nodes"])
        if not (folder / "complete.json").exists():
            if (folder / "failure.json").exists():
                raise ValueError(f"Run failure recorded: {read(folder / 'failure.json')}")
            return result
        complete = read(folder / "complete.json")
        if complete["state"] != "complete" or complete["arm"] != arm:
            raise ValueError("Completion arm/state mismatch")
        if complete["step"] != budget or complete["budget"] != budget:
            raise ValueError("Completion budget differs from the frozen plan")
        endpoint = complete["endpoint"]
        primary = endpoint["metrics"][panel][score]
        if not learning or learning[-1] != endpoint or endpoint["step"] != budget:
            raise ValueError("Learning curve and final endpoint disagree")
        result.update(
            step=budget,
            primary_score=primary,
            endpoint_metrics=endpoint["metrics"],
            world_sha256=complete["world_sha256"],
            model_sha256=complete["model_sha256"],
            initial_model_sha256=complete.get("initial_model_sha256"),
            config_sha256=complete["config_sha256"],
            arm=arm,
            cost={key: endpoint[key] for key in COST if key in endpoint},
            parameters=complete.get("parameters"),
            state="awaiting_grid",
        )
        if not (folder / "grid.json").exists():
            return result
        grid = read(folder / "grid.json")
        if any(not 0 <= metrics[panel][score] <= 1 for metrics in grid["metrics"].values()):
            raise ValueError("Grid score must be finite and between zero and one")
        for key in ("world_sha256", "model_sha256", "config_sha256", "step", "arm"):
            if grid[key] != complete[key]:
                raise ValueError(f"Endpoint/grid identity mismatch: {key}")
        if grid["training_repeats"] != repeats:
            raise ValueError("Grid training horizon differs from the fixed primary horizon")
        if not np.isclose(grid["metrics"][str(repeats)][panel][score], primary, atol=1e-7, rtol=0):
            raise ValueError("Fixed-horizon grid score differs from the final endpoint")
        with np.load(folder / "exposure.npz", allow_pickle=False) as exposure:
            counts = exposure["counts"]
            if counts.ndim != 1 or np.any(counts < 0) or counts.sum() != endpoint["examples"]:
                raise ValueError("Exposure counts disagree with endpoint examples")
        result.update(
            state="complete",
            passed=True,
            inference_scores={r: metrics[panel][score] for r, metrics in grid["metrics"].items()},
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        result.update(state="failed", passed=False, failure=f"{type(error).__name__}: {error}")
    return result


def compare(first, second):
    result = {"state": "pending", "passed": None, "difference_pp": None}
    if first["state"] == "failed" or second["state"] == "failed":
        return {
            **result,
            "state": "failed",
            "passed": False,
            "failure": "A paired run failed validation",
        }
    if first["state"] != "complete" or second["state"] != "complete":
        return result
    try:
        if first["world_sha256"] != second["world_sha256"] or first["step"] != second["step"]:
            raise ValueError("Paired world hash or endpoint step differs")
        if (
            Path(first["path"]).parent == Path(second["path"]).parent
            and first["config_sha256"] != second["config_sha256"]
        ):
            raise ValueError("Local paired arms used different frozen case configurations")
        if (
            second["arm"] == "fixed-update"
            and first["initial_model_sha256"] is not None
            and first["initial_model_sha256"] != second["initial_model_sha256"]
        ):
            raise ValueError("Fixed/token initial model hashes differ")
        with np.load(Path(first["path"]) / "exposure.npz", allow_pickle=False) as a:
            with np.load(Path(second["path"]) / "exposure.npz", allow_pickle=False) as b:
                if not np.array_equal(a["counts"], b["counts"]):
                    raise ValueError("Paired training exposure arrays differ")
        other = {row["step"]: row for row in second["nodes"]}
        nodes = []
        for row in first["nodes"]:
            if row["step"] not in other:
                continue
            matched = other[row["step"]]
            for key in (
                "examples",
                "effective_input_tokens",
                "executed_input_tokens",
                "supervised_tokens",
            ):
                if key in row and key in matched and row[key] != matched[key]:
                    raise ValueError(f"Paired node exposure differs: {row['step']} {key}")
            nodes.append(
                {
                    "step": row["step"],
                    "difference_pp": 100 * (row["primary_score"] - matched["primary_score"]),
                }
            )
        result.update(
            state="complete",
            passed=True,
            exposure_exact=True,
            nodes=nodes,
            difference_pp=100 * (first["primary_score"] - second["primary_score"]),
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        result.update(state="failed", passed=False, failure=f"{type(error).__name__}: {error}")
    return result


def collect(plan, output):
    """Write an atomic summary; pending baselines remain visible and unscored."""
    plan = read(plan) if isinstance(plan, (str, Path)) else plan
    output = Path(output)
    controller = (
        read(output / "controller-state.json")
        if (output / "controller-state.json").exists()
        else {}
    )
    cases = {}
    for job in plan["jobs"]:
        case = cases.setdefault(job["case"], {"family": job["family"], "jobs": {}})
        case["jobs"][job["arm"]] = job
    for name, case in cases.items():
        family = case["family"]
        panel, score, repeats = (
            ("test_oo", "accuracy", 2) if family == "wang" else ("ood_3", "answer_accuracy", 4)
        )
        jobs = case.pop("jobs")
        case["primary"] = {"panel": panel, "metric": score, "inference_repeats": repeats}
        case["arms"] = {}
        for arm in ARMS:
            if arm == "paper" and family == "continuation":
                reused = plan["reused_baseline"]
                folder, budget = Path(reused["root"]) / "runs" / "paper", reused["steps"]
            else:
                folder, budget = output / "cases" / name / "runs" / arm, jobs[arm]["steps"]
            case["arms"][arm] = inspect_run(folder, arm, budget, panel, score, repeats)
            job_id = jobs.get(arm, {}).get("id", f"{name}--{arm}")
            status = controller.get("jobs", {}).get(job_id, {})
            if status.get("state") in {"needs_attention", "failed"}:
                case["arms"][arm].update(
                    state="failed", passed=False, failure=f"Controller recorded failure: {status}"
                )
        case["comparisons"] = {
            key: compare(case["arms"]["token-update"], case["arms"][arm])
            for key, arm in PAIRS.items()
        }
    families = {}
    for family in ("confirmation", "depth"):
        selected = {name: case for name, case in cases.items() if case["family"] == family}
        families[family] = {}
        for comparison in PAIRS:
            per_world = {name: case["comparisons"][comparison] for name, case in selected.items()}
            completed = [
                row["difference_pp"] for row in per_world.values() if row["state"] == "complete"
            ]
            all_done = len(completed) == len(selected) and bool(selected)
            worlds = [
                case["arms"]["token-update"]["world_sha256"]
                for case in selected.values()
                if case["comparisons"][comparison]["state"] == "complete"
            ]
            distinct = len(worlds) == len(set(worlds))
            families[family][comparison] = {
                "independent_unit": "knowledge graph plus initialization pair",
                "per_world": per_world,
                "planned_worlds": len(selected),
                "completed_worlds": len(completed),
                "state": "failed" if not distinct else "complete" if all_done else "pending",
                "failure": None
                if distinct
                else "Repeated graph hashes cannot count as independent worlds",
                "mean_difference_pp": float(np.mean(completed)) if all_done and distinct else None,
                "sample_sd_pp": float(np.std(completed, ddof=1))
                if all_done and distinct and len(completed) > 1
                else None,
            }
    all_states = [
        row["state"]
        for case in cases.values()
        for group in ("arms", "comparisons")
        for row in case[group].values()
    ]
    failed_families = any(
        row["state"] == "failed" for group in families.values() for row in group.values()
    )
    if failed_families:
        all_states.append("failed")
    result = {
        "batch": plan["batch"],
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "passed": "failed" not in all_states,
        "state": "failed"
        if "failed" in all_states
        else "complete"
        if all(state == "complete" for state in all_states)
        else "pending",
        "cases": cases,
        "families": families,
        "scope": "Fixed registered endpoints and horizons; no best-checkpoint selection. "
        "Means/SD use independent paired worlds, never queries. No statistical significance claim.",
        "threshold_note": "First observed crossing bracket between scheduled measurements; "
        "null means not observed yet, not proven impossible.",
    }
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=output, prefix=".summary-", suffix=".json", delete=False
    ) as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
        temporary = Path(stream.name)
    temporary.replace(output / "summary.json")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = collect(args.plan, args.output)
    print(json.dumps({"state": result["state"], "passed": result["passed"]}))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
