#!/usr/bin/env python3
"""Compare four independently audited arms to the read-only reused baseline."""

import argparse
import json
from pathlib import Path

import numpy as np

from llm_memory_editability.grok_depth import utc, write_json
from llm_memory_editability.grokking_reproduction import digest


def main():
    import wandb

    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    for name, expected in config["baseline"]["files"].items():
        if digest(args.baseline / name) != expected:
            raise ValueError("Read-only baseline identity changed")
    report = {
        "batch": config["batch"],
        "primary": config["primary"],
        "conditions": {},
        "new_scientific_updates": 1200000,
        "baseline_new_updates": 0,
        "novel_fact_updates": 0,
        "utc": utc(),
    }
    with np.load(args.baseline / "exposure.npz") as saved:
        baseline_counts = saved["counts"].copy()
    for name, folder in {
        "fixed-reused": args.baseline,
        **{n: args.base / "runs" / n for n in config["arms"]},
    }.items():
        complete = json.loads((folder / "complete.json").read_text())
        audit = json.loads((folder / "audit.json").read_text())
        grid = json.loads((folder / "grid.json").read_text())
        grid_audit = json.loads((folder / "grid-audit.json").read_text())
        with np.load(folder / "exposure.npz") as saved:
            exposure_exact = np.array_equal(saved["counts"], baseline_counts)
        if (
            not audit["passed"]
            or not grid_audit["passed"]
            or not exposure_exact
            or complete["step"] != 300000
        ):
            raise ValueError(f"Unverified endpoint: {name}")
        report["conditions"][name] = {
            "complete": complete,
            "audit": audit,
            "grid": grid,
            "grid_audit": grid_audit,
            "exposure_exact_with_fixed_baseline": exposure_exact,
            "files": {
                p: digest(folder / p)
                for p in ("complete.json", "audit.json", "grid.json", "grid-audit.json")
            },
        }
    destination = args.base / "comparison.json"
    write_json(destination, report)
    lines = [
        "# 五组轮间门控多跳比较",
        "",
        "固定300k、推理4轮；单开发图/初始化。固定DiscoLoop复用，新增知识实验0。",
        "",
        "|条件|图B两跳|图B三跳（主指标）|图B四跳|",
        "|---|---:|---:|---:|",
    ]
    for name, row in report["conditions"].items():
        metrics = row["grid"]["metrics"]["4"]
        values = [f"{100 * metrics[f'ood_{hop}']['answer_accuracy']:.2f}%" for hop in (2, 3, 4)]
        lines.append("|" + "|".join([name, *values]) + "|")
    (args.base / "report.md").write_text("\n".join(lines) + "\n")
    run = wandb.init(
        entity="zhusq20",
        project="llm-memory-editability",
        group=config["batch"],
        name="five-condition-report",
        id=f"dg-report-{digest(args.config)[:10]}",
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
        }
    )
    run.save(str(destination), base_path=str(args.base), policy="now")
    url, identity = run.url, run.id
    run.finish()
    remote = wandb.Api(timeout=60).run(f"zhusq20/llm-memory-editability/{identity}")
    downloaded = remote.file(destination.name).download(
        root=str(args.base / "report-cloud-download"), replace=False
    )
    passed = digest(downloaded.name) == digest(destination) and remote.state == "finished"
    write_json(
        args.base / "report-cloud-audit.json",
        {"passed": passed, "url": url, "comparison_sha256": digest(destination), "utc": utc()},
    )
    if not passed:
        raise AssertionError("Full five-condition cloud report verification failed")


if __name__ == "__main__":
    main()
