#!/usr/bin/env python3
"""Five-condition comparison with the three prespecified differences."""

import argparse
import json
from pathlib import Path

import numpy as np

from llm_memory_editability.grok_depth import utc, write_json
from llm_memory_editability.grokking_reproduction import digest


def build(config, base, baseline):
    for name, expected in config["baseline"]["files"].items():
        if digest(baseline / name) != expected:
            raise ValueError(f"Read-only F changed: {name}")
    diagnostic = json.loads((base / "activation-diagnostic.json").read_text())
    if not diagnostic["passed"]:
        raise ValueError("Fixed diagnostic validation failed")
    baseline_replay = json.loads((base / "baseline-target-replay.json").read_text())
    if not baseline_replay["passed"]:
        raise ValueError("Target-machine F replay failed")
    with np.load(baseline / "exposure.npz") as saved:
        counts = saved["counts"].copy()
    report = {
        "batch": config["batch"],
        "primary": config["primary"],
        "conditions": {},
        "new_scientific_updates": 1200000,
        "baseline_new_updates": 0,
        "diagnostic_sha256": digest(base / "activation-diagnostic.json"),
        "baseline_target_replay": baseline_replay,
        "limits": config["limits"],
        "utc": utc(),
        "baseline_cloud_status": "Parent historical cloud-summary failure is preserved; "
        "this report separately uploads the reused local audited evidence.",
    }
    for name in ("F", *config["arms"]):
        folder = baseline if name == "F" else base / "runs" / name
        payload = {
            n: json.loads((folder / n).read_text())
            for n in ("complete.json", "audit.json", "grid.json", "grid-audit.json")
        }
        complete = payload["complete.json"]
        with np.load(folder / "exposure.npz") as saved:
            matched = np.array_equal(counts, saved["counts"])
        if (
            not matched
            or not payload["audit.json"]["passed"]
            or not payload["grid-audit.json"]["passed"]
            or complete["step"] != 300000
            or complete["engineering_only"]
        ):
            raise ValueError(f"Unaudited/mismatched endpoint: {name}")
        if name != "F":
            cloud = json.loads((folder / "cloud-audit.json").read_text())
            if not cloud["passed"]:
                raise ValueError(f"Cloud audit failed: {name}")
            payload["cloud-audit.json"] = cloud
        report["conditions"][name] = {
            **payload,
            "exposure_exact_with_F": matched,
            "complete_sha256": digest(folder / "complete.json"),
            "grid_sha256": digest(folder / "grid.json"),
        }
    primary = {
        n: value["grid.json"]["metrics"]["4"]["ood_3"]["answer_accuracy"]
        for n, value in report["conditions"].items()
    }
    report["primary_difference_percentage_points"] = {
        "R-F": 100 * (primary["R"] - primary["F"]),
        "N-F": 100 * (primary["N"] - primary["F"]),
        "C-A": 100 * (primary["C"] - primary["A"]),
    }
    return report


def main():
    import wandb

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    destination = args.base / "comparison.json"
    if destination.exists():
        raise FileExistsError(destination)
    config = json.loads(args.config.read_text())
    report = build(config, args.base, args.baseline)
    write_json(destination, report)
    lines = [
        "# DiscoLoop内部残差与轮间交接比较",
        "",
        "固定300k、推理4轮；单开发图/初始化。F复用，新增四条训练。",
        "",
        "|条件|图B两跳|图B三跳（主指标）|图B四跳|",
        "|---|---:|---:|---:|",
    ]
    for name, row in report["conditions"].items():
        metrics = row["grid.json"]["metrics"]["4"]
        values = [f"{100 * metrics[f'ood_{k}']['answer_accuracy']:.2f}%" for k in (2, 3, 4)]
        lines.append("|" + "|".join([name, *values]) + "|")
    lines += ["", "预定主指标差值（百分点）："]
    lines += [f"- {k}: {v:+.2f}" for k, v in report["primary_difference_percentage_points"].items()]
    lines += [
        "",
        "完整原子/图A/图B、条件覆盖、R1–4网格见comparison.json；固定位置统计见activation-diagnostic.json。低方差不能单独证明组合改进。",
    ]
    (args.base / "report.md").write_text("\n".join(lines) + "\n")
    run = wandb.init(
        entity="zhusq20",
        project="llm-memory-editability",
        group=config["batch"],
        name="five-condition-report",
        id=f"nh-report-{digest(args.config)[:10]}",
        dir=str(args.base),
        mode="online",
        config={"experiment": config},
        settings=wandb.Settings(disable_git=True),
    )
    run.summary.update(
        {
            "comparison_sha256": digest(destination),
            "all_local_audits_passed": True,
            "new_scientific_updates": 1200000,
            "fixed_baseline_new_updates": 0,
            **report["primary_difference_percentage_points"],
        }
    )
    for filename in ("comparison.json", "activation-diagnostic.json", "report.md"):
        run.save(str(args.base / filename), base_path=str(args.base), policy="now")
    identity, url = run.id, run.url
    run.finish()
    remote = wandb.Api(timeout=60).run(f"zhusq20/llm-memory-editability/{identity}")
    matches = {}
    for filename in ("comparison.json", "activation-diagnostic.json", "report.md"):
        saved = remote.file(filename).download(
            root=str(args.base / "report-cloud-download"), replace=False
        )
        matches[filename] = digest(saved.name) == digest(args.base / filename)
    passed = all(matches.values()) and remote.state == "finished"
    write_json(
        args.base / "report-cloud-audit.json",
        {
            "passed": passed,
            "files": matches,
            "url": url,
            "utc": utc(),
        },
    )
    if not passed:
        raise AssertionError("Full comparison/diagnostic cloud roundtrip failed")


if __name__ == "__main__":
    main()
