#!/usr/bin/env python3
"""Compare three already audited TTCD endpoints without loading any model."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

BATCH = "ttcd-architecture-comparison-v1"
ARMS = ("transformer", "loop", "discoloop")
PANEL_SHA256 = "ade07fa92e528f1b19bbc10965d984f747ed7f7daf66f5eafa254971ac7d08f6"
PANELS = ("atomic_id", "atomic_ood", "train_ii", "test_ii", "test_io", "test_oi", "test_oo")


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def layout(root):
    root = Path(root)
    rows = {}
    for arm in ARMS:
        if arm == "discoloop":
            base = root / "ttcd-successful-base-v1"
            foundation = root / "discoloop-success-restore-v1"
            native = foundation / "discoloop"
            supplement = root / "ttcd-double-new-eval-v1/stream"
            panel = root / "ttcd-double-new-eval-v1/panel.json"
            double_prefix = ""
        else:
            base = root / BATCH / arm
            foundation, native = base / "foundation", base / "foundation" / arm
            supplement = base / "double-new"
            panel = root / BATCH / "source/configs/double-new-panel.json"
            double_prefix = "double-"
        common, stream = base / "common", base / "stream"
        rows[arm] = {
            "panel": panel,
            "native_complete": native / "complete.json",
            "native_audit": native / "audit.json",
            "foundation_cloud": foundation / "result-cloud-audit.json",
            "common_gate": common / "result-gate.json",
            "common_cloud": common / "result-cloud-audit.json",
            "train": stream / "result-train.json",
            "train_audit": stream / "result-train-audit.json",
            "calibrate": stream / "result-calibrate.json",
            "original_evaluate": stream / "result-evaluate.json",
            "original_audit": stream / "result-audit.json",
            "original_cloud": stream / "result-cloud-audit.json",
            "supplement_evaluate": supplement / f"result-{double_prefix}evaluate.json",
            "supplement_audit": supplement / f"result-{double_prefix}audit.json",
            "supplement_cloud": supplement / f"result-{double_prefix}cloud-audit.json",
            "stream_progress": stream / "training/progress.json",
        }
    return rows


def assert_no_active_failure(rows):
    """Preserved old attempts do not override a newer active v2 controller."""
    directories = {p.parent for row in rows.values() for p in row.values()}
    for directory in directories:
        preferred = directory / "controller-state-v2.json"
        if not preferred.exists():
            preferred = directory / "controller-state.json"
        if preferred.exists():
            state = read(preferred)
            if state.get("state") in ("needs_attention", "failed", "calibration_needs_attention"):
                raise RuntimeError(
                    f"An active reused/comparison stage needs attention: {preferred}"
                )
        for failure in directory.glob("failure-*.json"):
            stage = failure.name.removeprefix("failure-").removesuffix(".json")
            if not (directory / f"result-{stage}.json").exists():
                raise RuntimeError(f"An original stage failed; no automatic rerun: {failure}")


def readiness(rows):
    assert_no_active_failure(rows)
    missing = [str(path) for row in rows.values() for path in row.values() if not path.is_file()]
    return sorted(set(missing))


def verify_cloud(path):
    cloud = read(path)
    if cloud.get("passed") is not True or cloud.get("full_json_exact") is not True:
        raise ValueError(f"Canonical cloud result verification is required: {path}")
    records = cloud.get("runs", [])
    if not records:
        raise ValueError(f"Cloud receipt has no original stages: {path}")
    seen = set()
    for record in records:
        stage = record["stage"]
        if stage in seen:
            raise ValueError("Duplicate stage in cloud verification receipt")
        seen.add(stage)
        result = path.parent / f"result-{stage}.json"
        expected = record.get("result_sha256", record.get("sha256"))
        if not expected or sha(result) != expected:
            raise ValueError(f"Cloud receipt does not bind the local original stage: {result}")
        if record.get("state", "finished") != "finished":
            raise ValueError("A recorded cloud stage was not finished")
    return seen


def _check_numeric(expected, actual):
    if expected is None or actual is None:
        return expected is actual
    return math.isclose(expected, actual, rel_tol=1e-12, abs_tol=1e-12)


def collect(root):
    rows = layout(root)
    missing = readiness(rows)
    if missing:
        raise FileNotFoundError("Audited endpoints are incomplete: " + ", ".join(missing[:6]))
    input_hashes = {str(path): sha(path) for row in rows.values() for path in row.values()}
    results, panel_reference = {}, None
    for arm, paths in rows.items():
        if sha(paths["panel"]) != PANEL_SHA256:
            raise ValueError(f"The exact authorized supplementary panel changed: {arm}")
        panel = read(paths["panel"])
        if panel_reference is None:
            panel_reference = panel
        elif panel != panel_reference:
            raise ValueError("Architectures used different raw graph panels")
        panel_digest = hashlib.sha256(json.dumps(panel, sort_keys=True).encode()).hexdigest()
        if len(panel["facts"]) != 32 or len(panel["pairs"]) != 16 or len(panel["orders"]) != 32:
            raise ValueError("The original full facts/pairs/rotations must be retained")
        supplement, audit = read(paths["supplement_evaluate"]), read(paths["supplement_audit"])
        if (
            audit.get("passed") is not True
            or audit.get("answer_and_EOS_exact") is not True
            or audit.get("write_states_exact") is not True
            or audit.get("summary_recomputed") is not True
            or supplement["identity"] != audit["identity"]
            or supplement["identity"]["panel_sha256"] != panel_digest
            or supplement["states"] != audit["states"]
        ):
            raise ValueError(f"Double-new audit does not bind this endpoint and panel: {arm}")
        if supplement["primary_t"] != 32:
            raise ValueError("Primary comparison must remain the fixed t32 endpoint")
        primary = supplement["primary"]
        pair_values = list(primary["pair_means"].values())
        if len(pair_values) != 16 or any(v is None for v in pair_values):
            raise ValueError("All fixed pairs must contribute to the primary endpoint")
        if not _check_numeric(sum(pair_values) / 16, primary["two_new_accuracy"]):
            raise ValueError("Primary accuracy is not pair-equal")
        native, native_audit = read(paths["native_complete"]), read(paths["native_audit"])
        if (
            native.get("state") != "complete"
            or native.get("step") != 300000
            or native.get("engineering_only") is not False
            or native_audit.get("passed") is not True
            or native_audit.get("step") != 300000
        ):
            raise ValueError(f"Original native foundation was not independently completed: {arm}")
        native_metrics = native["endpoint"]["metrics"]
        if set(native_metrics) != set(PANELS):
            raise ValueError("All seven native foundation panels are required")
        for key in ("train_audit", "calibrate", "original_audit", "common_gate"):
            if read(paths[key]).get("passed") is not True:
                raise ValueError(f"Mandatory original stage/audit is incomplete: {arm}/{key}")
        train, train_audit = read(paths["train"]), read(paths["train_audit"])
        if (
            train["step"] != 10000
            or train["counts"]["base_parameter_updates"] != 10000
            or train["checkpoint_sha256"] != train_audit["checkpoint_sha256"]
            or train["identity"] != train_audit["identity"]
            or supplement["identity"]["parent"]["checkpoint_sha256"] != train["checkpoint_sha256"]
        ):
            raise ValueError("Reused/new stream checkpoint is not source-bound to its audit")
        original, original_audit = read(paths["original_evaluate"]), read(paths["original_audit"])
        if original["identity"] != original_audit["identity"]:
            raise ValueError("Original continual summary and audit identities differ")
        cloud_stages = {
            key: verify_cloud(paths[key])
            for key in ("foundation_cloud", "common_cloud", "original_cloud", "supplement_cloud")
        }
        required = {
            "foundation_cloud": {"train", "audit"},
            "common_cloud": {"train", "train-audit", "gate"},
            "original_cloud": {"train", "train-audit", "calibrate", "evaluate", "audit"},
            "supplement_cloud": {"evaluate", "audit"}
            if arm == "discoloop"
            else {"double-evaluate", "double-audit"},
        }
        if any(not required[key] <= cloud_stages[key] for key in required):
            raise ValueError("Cloud completion omits a required original stage")
        progress = read(paths["stream_progress"])
        if progress["step"] != 10000 or set(progress["native_metrics"]) != set(PANELS):
            raise ValueError("All seven terminal stream native panels must be reported")
        layers, repeats = (8, 1) if arm == "transformer" else (4, 2)
        results[arm] = {
            "reused": arm == "discoloop",
            "foundation_native_metrics": native_metrics,
            "stream_native_metrics": progress["native_metrics"],
            "primary_t32": primary,
            "curves": supplement["curves"],
            "by_fact_age": supplement["by_fact_age"],
            "by_pair_completion_age": supplement["by_pair_completion_age"],
            "by_constituent_age_and_gap": supplement["by_constituent_age_and_gap"],
            "omission_and_isolated_pair_controls": supplement[
                "omission_and_isolated_pair_controls"
            ],
            "original_continual_primary": original["primary"],
            "checkpoint_sha256": train["checkpoint_sha256"],
            "common_quality": read(paths["common_gate"]),
            "calibration_quality": read(paths["calibrate"]),
            "cost": {
                "native_training_seconds": native["training_seconds"],
                "stream_training_seconds": train["training_seconds"],
                "stream_counts": train["counts"],
                "stream_compute": train["compute"],
                "base_parameters": native["parameters"],
                "physical_layers": layers,
                "repeats": repeats,
                "fast_elements": layers * 3072 * 768,
                "fast_bytes_fp32": layers * 3072 * 768 * 4,
            },
            "source_paths": {key: str(value) for key, value in paths.items()},
            "cloud_stages": {key: sorted(value) for key, value in cloud_stages.items()},
        }
    for path, digest in input_hashes.items():
        if sha(path) != digest:
            raise ValueError("A source result changed during read-only aggregation")
    return {
        "passed": True,
        "batch": BATCH,
        "utc": datetime.now(timezone.utc).isoformat(),
        "panel_sha256": PANEL_SHA256,
        "architectures": results,
        "source_sha256": input_hashes,
        "new_model_queries": 0,
        "new_training_updates": 0,
        "analysis_unit": (
            "Same observed Wang world and fixed16 pairs;32 rotations are repeated measures"
        ),
        "conditional_cohorts": (
            "Each model's own immediate-correct/baseline-correct cohort; denominators preserved"
        ),
        "capacity_warning": (
            "D8 has twice the persistent fast-state elements of Loop/Disco; "
            "base parameters and compute also differ"
        ),
    }


def markdown(report):
    def pct(value):
        return "—" if value is None else f"{100 * value:.2f}%"

    lines = [
        "# 三种架构的TTCD持续写入比较",
        "",
        "同一知识图、同一32个更正键及16条双新组合；32个循环顺序为重复测量。",
        "",
        "| 架构 | 32次写入后的双新组合 | 先前原子即时正确后的保持 | "
        "原正确旧原子保持 | 原正确旧组合保持 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for arm, result in report["architectures"].items():
        primary = result["primary_t32"]
        retained = primary["earlier_atoms_retention_from_immediate"]
        label = {"transformer": "普通D8", "loop": "Loop4×2", "discoloop": "DiscoLoop4×2（复用）"}[
            arm
        ]
        lines.append(
            f"| {label} | {pct(primary['two_new_accuracy'])} | "
            f"{pct(retained['accuracy'])}（{retained['eligible']}/{retained['total']}） | "
            f"{pct(primary['U']['U_atoms']['retained'])} | "
            f"{pct(primary['U']['U_chains']['retained'])} |"
        )
    lines += [
        "",
        "主指标保留全部16对事实。条件保持按各模型自己的原正确集合计算，完整覆盖率、"
        "逐年龄结果、写入顺序与间隔、七池基础能力及原始来源哈希保存在comparison.json。",
        "",
        "D8有8份快矩阵，Loop与Disco各4份；参数、状态容量和计算量不同，不能将结果解释为等成本比较。"
        "本轮仅一张已观察图和一个初始化，不推断通用架构优势。",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--persistent-root", type=Path, required=True)
    parser.add_argument("--wait-hours", type=float, default=0)
    args = parser.parse_args()
    if not 0 <= args.wait_hours <= 48:
        raise ValueError("Read-only aggregation wait must be bounded by48 hours")
    output = args.persistent_root / BATCH
    output.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + args.wait_hours * 3600
    try:
        while missing := readiness(layout(args.persistent_root)):
            write(
                output / "report-state.json",
                {
                    "state": "waiting_for_audited_cloud_endpoints",
                    "missing": missing,
                    "new_model_queries": 0,
                    "new_training_updates": 0,
                },
            )
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "Required audited/cloud endpoints did not complete within wait budget"
                )
            time.sleep(min(30, max(0.0, deadline - time.monotonic())))
        report = collect(args.persistent_root)
        if (output / "comparison.json").exists() or (output / "comparison.md").exists():
            raise FileExistsError("Preserve the existing architecture comparison report")
        write(output / "comparison.json", report)
        (output / "comparison.md").write_text(markdown(report))
        write(
            output / "report-state.json",
            {
                "state": "complete",
                "passed": True,
                "comparison_sha256": sha(output / "comparison.json"),
                "new_model_queries": 0,
            },
        )
    except Exception as error:
        write(
            output / "report-failure.json",
            {
                "state": "needs_attention",
                "error": str(error),
                "new_model_queries": 0,
                "new_training_updates": 0,
            },
        )
        raise


if __name__ == "__main__":
    main()
