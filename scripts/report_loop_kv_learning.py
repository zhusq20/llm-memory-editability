"""Three-model learning curves and paired endpoints, reusing audited old runs."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from report_loop_learning import COSTS, POOLS, difference, metric, read, transitions, write

LABELS = {
    "standard": "普通四层Transformer",
    "loop_local": "两层Loop计算两轮",
    "loop_shared_kv": "同Loop加首轮共享KV",
}


def summarize(config):
    root = Path(config["results_root"])
    sources = [(s["arm"], root / "runs" / s["name"], s, False) for s in config["specs"]]
    for reused in config["reused_runs"]:
        path = Path(reused["run_dir"])
        sources.append(("loop_local", path, read(path / "run.json")["spec"], True))
    rows, curves = [], []
    for arm, path, spec, reused in sources:
        metadata = read(path / "run.json", {})
        if metadata and metadata["spec"] != spec:
            raise ValueError(f"Frozen spec differs: {path}")
        history = read(path / "learning.json", [])
        steps = [item["step"] for item in history]
        if steps != sorted(set(steps)):
            raise ValueError(f"Duplicate or unordered nodes: {path}")
        boundaries = [spec["stage_a_steps"], spec["stage_a_steps"] + spec["stage_b_steps"]]
        endpoints = [next((h for h in history if h["step"] == n), None) for n in boundaries]
        raw = [read(path / f"predictions-{n:07d}.json") for n in boundaries]
        endpoint_json = read(path / "endpoint.json")
        if endpoints[1] and endpoint_json != endpoints[1]["metrics"]:
            raise ValueError(f"Endpoint metrics differ: {path}")
        audit = read(path / "audit.json", {})
        complete = read(path / "complete.json", {})
        pools = set(POOLS)
        for endpoint in endpoints:
            if endpoint:
                pools.update(endpoint["metrics"])
        row = {
            "name": spec["name"],
            "arm": arm,
            "label": LABELS[arm],
            "run_dir": str(path),
            "reused": reused,
            "initialization": spec["initialization"],
            "data_sha256": spec["data_sha256"],
            "stage_a": {p: metric(endpoints[0], p, raw[0]) for p in sorted(pools)},
            "endpoint": {p: metric(endpoints[1], p, raw[1]) for p in sorted(pools)},
            "transitions": {p: transitions(raw[0], raw[1], p) for p in POOLS},
            "architecture": read(path / "architecture.json", {}),
            "sampling_plan_sha256": read(path / "sampling-plan.json", {}).get("plan_sha256"),
            "initial_model_sha256": metadata.get("initial_model_sha256"),
            "latest_step": read(path / "status.json", {}).get("step"),
            "costs": {
                label: {k: h.get(k) if h else None for k in COSTS}
                for label, h in zip(("stage_a", "endpoint"), endpoints, strict=True)
            },
            "complete_with_reload": bool(
                endpoints[1]
                and complete.get("independently_reloaded")
                and audit.get("passed")
                and not (path / "failure.json").exists()
            ),
        }
        rows.append(row)
        for item in history:
            predictions = read(path / f"predictions-{item['step']:07d}.json")
            curves.append(
                {
                    "name": spec["name"],
                    "arm": arm,
                    "initialization": spec["initialization"],
                    "step": item["step"],
                    "evaluation_scope": "full" if item["step"] in boundaries else "panel",
                    "metrics": {p: metric(item, p, predictions) for p in item["metrics"]},
                    "costs": {k: item.get(k) for k in COSTS},
                }
            )
    pairs = []
    for seed in sorted({row["initialization"] for row in rows}):
        group = {r["arm"]: r for r in rows if r["initialization"] == seed}
        local = group["loop_local"]
        for arm in ("standard", "loop_shared_kv"):
            other = group[arm]
            if local["sampling_plan_sha256"] and other["sampling_plan_sha256"]:
                if local["sampling_plan_sha256"] != other["sampling_plan_sha256"]:
                    raise ValueError("Paired sampling plans differ")
            comparison = {
                "initialization": seed,
                "contrast": arm + "_minus_loop_local",
                "both_complete": local["complete_with_reload"] and other["complete_with_reload"],
            }
            for boundary in ("stage_a", "endpoint"):
                comparison[boundary] = {
                    p: difference(
                        other[boundary][p]["answer_accuracy"] if other[boundary][p] else None,
                        local[boundary][p]["answer_accuracy"] if local[boundary][p] else None,
                    )
                    for p in POOLS
                }
            for pool in ("BB", "AA"):
                left, right = other["transitions"][pool], local["transitions"][pool]
                comparison[pool + "_net_gain_difference"] = difference(
                    left["net_accuracy_change"] if left else None,
                    right["net_accuracy_change"] if right else None,
                )
            left, right = other["transitions"]["AA"], local["transitions"]["AA"]
            comparison["AA_retention_difference"] = difference(
                left["retention_of_stage_a_correct"] if left else None,
                right["retention_of_stage_a_correct"] if right else None,
            )
            pairs.append(comparison)
    return {
        "batch": config["batch"],
        "expected_new_runs": 4,
        "reused_runs": 2,
        "complete_new_runs": sum(r["complete_with_reload"] and not r["reused"] for r in rows),
        "state": "complete" if all(r["complete_with_reload"] for r in rows) else "in_progress",
        "runs": rows,
        "paired_comparisons": pairs,
        "curves": curves,
        "limitations": [
            "Two seeds on one development split are not independent worlds.",
            "Final BB and its net increase are separate outcomes.",
            "No B-free continuation is included in this batch.",
            "Parameters and FLOPs are reported separately.",
            "Panel nodes and full endpoints have different denominators.",
            "Scientific completion, independent reload and cloud audit are separate.",
        ],
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }


def render(summary):
    def score(cell):
        return (
            "—"
            if cell is None
            else f"{100 * cell['answer_accuracy']:.2f}% ({cell['correct']}/{cell['n']})"
        )

    lines = [
        "# 三种模型的旧知识学习与新知识使用",
        "",
        f"状态：{summary['state']}；新增完成 {summary['complete_new_runs']}/4；复用旧Loop两条。",
        "",
        "| 模型 | 初始化 | 端点 | 旧原子 | 新原子 | AA | BA | AB | BB |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["runs"]:
        for boundary in ("stage_a", "endpoint"):
            values = [score(row[boundary].get(p)) for p in POOLS if p != "train_composition"]
            lines.append(
                f"| {row['label']} | {row['initialization']} | {boundary} | "
                + " | ".join(values)
                + " |"
            )
    lines.extend(
        [
            "",
            "BB净增及AA逐题保持、覆盖、成本和完整学习曲线见summary.json。",
            "两个初始化来自同一开发划分。空缺终点尚未完成，不能当作零分。",
            "共享KV增加注意力计算，普通模型增加独立参数；分别报告成本。",
        ]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    out = args.out or Path(config["results_root"]) / "report"
    summary = summarize(config)
    write(out / "summary.json", summary)
    (out / "report.md").write_text(render(summary))
    print(json.dumps({"state": summary["state"], "new_complete": summary["complete_new_runs"]}))
