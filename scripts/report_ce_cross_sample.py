"""Summarize only fully audited and cloud-verified continuous CE experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llm_memory_editability.ce_cross_sample_summary import compare_architectures  # noqa: E402


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def percentage(value):
    return "NA" if value is None else f"{100 * value:.2f}%"


def pooled_old(summary, category):
    values = [
        summary["streams"][stream]["endpoint"]["old"]["categories"][category][
            "originally_correct_retained"
        ]
        for stream in ("forward", "reverse")
    ]
    correct, n = sum(v["correct"] for v in values), sum(v["n"] for v in values)
    return {"correct": correct, "n": n, "accuracy": correct / n if n else None}


def main():
    import wandb

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "result.json").exists() or (args.out / "failure.json").exists():
        raise FileExistsError("Never replace an existing comparison or its failed evidence")
    plan, panel = read(args.plan), None
    panel = read(ROOT / plan["panel"])
    summaries, receipts, hashes, table = {}, {}, {}, {}
    for architecture in ("transformer", "loop", "discoloop"):
        root = args.batch / architecture / "output"
        science, audit, cloud = (
            read(root / stage / "result.json") for stage in ("evaluate", "audit", "cloud")
        )
        if (
            not science["passed"]
            or not audit["passed"]
            or not cloud["passed"]
            or science["scientific_optimizer_updates"] != 4480
            or audit["scientific_optimizer_updates"] != 0
            or audit["replayed_states"] != 257
            or not audit["summary_recomputed"]
            or audit["evaluate_result_sha256"] != sha(root / "evaluate/result.json")
            or not cloud["canonical_full_json_exact"]
        ):
            raise ValueError("Every architecture must complete science, independent replay and W&B")
        for receipt in cloud["runs"]:
            if receipt["result_sha256"] != sha(root / receipt["stage"] / "result.json"):
                raise ValueError("A cloud verification receipt no longer binds the actual result")
        summaries[architecture], receipts[architecture] = science, cloud
        hashes[architecture] = {
            stage: sha(root / stage / "result.json")
            for stage in ("prepare", "evaluate", "audit", "cloud")
        }
        table[architecture] = {
            "new_atoms": {
                "correct": sum(
                    science["streams"][s]["endpoint"]["written_atoms"]["correct"]
                    for s in ("forward", "reverse")
                ),
                "n": 128,
            },
            "double_new": science["primary_endpoint"]["double_new"],
            "both_atoms_and_composition": science["primary_endpoint"]["both_atoms_and_composition"],
            "composition_given_both_atoms": science["primary_endpoint"][
                "composition_given_both_atoms"
            ],
            "isolated_double_new": science["primary_endpoint"]["isolated_double_new"],
            "old_atoms": pooled_old(science, "atoms"),
            "old_compositions": pooled_old(science, "compositions"),
        }
    comparison = compare_architectures(summaries, panel, plan["evaluation"])
    result = {
        "passed": True,
        "batch": plan["batch"],
        "summaries": summaries,
        "comparison": comparison,
        "table": table,
        "verified_inputs": hashes,
        "cloud_receipts": receipts,
        "new_scientific_updates": 13440,
        "new_foundation_updates": 0,
        "new_composition_supervision": 0,
        "independent_replay_passed": True,
        "full_cloud_json_verified": True,
        "worlds": 1,
        "initializations_per_architecture": 1,
        "source_sha256": sha(ROOT / "source-manifest.json"),
    }
    lines = [
        "# 跨样本连续学习双新组合与旧知识保持",
        "",
        "三架构复用原生300k，每架构两条64事实连续流。以下终点为32对×两种顺序；"
        "两种顺序是重复测量。所有科学状态独立重载及W&B完整JSON核验通过。",
        "",
        "| 模型 | 已写原子 | 双新组合 | 两原子及组合均正确 | 两原子正确时组合 | "
        "独立两事实组合 | 旧原子保持 | 旧组合保持 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for architecture, row in table.items():
        atomic = row["new_atoms"]
        conditional = row["composition_given_both_atoms"]
        cells = [
            architecture,
            f"{atomic['correct']}/{atomic['n']}",
            percentage(row["double_new"]["accuracy"]),
            percentage(row["both_atoms_and_composition"]["accuracy"]),
            f"{percentage(conditional['accuracy'])}（覆盖{conditional['coverage_n']}/{conditional['full_n']}）",
            percentage(row["isolated_double_new"]["accuracy"]),
            percentage(row["old_atoms"]["accuracy"]),
            percentage(row["old_compositions"]["accuracy"]),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        "旧保持分母是固定旧题在两个终点状态下的原正确题/状态对，不是新增独立事实。"
        "七池分项、零覆盖NA、逐前缀曲线、年龄保持、三架构共同原子前提子集、配对bootstrap与全部状态哈希见result.json。",
        "",
        "新增科学更新13,440，基础和writer外层训练0；新组合答案监督0，使用旧输出KL保护。"
        "单图、每架构一个初始化，基础组合能力及参数/计算不同；结果只评价当前配方，不能唯一归因参数共享或通道。",
        "",
        "原始预测与MLP状态保存在本批持久目录；本报告由AI辅助整理。",
    ]
    (args.out / "report.md").write_text("\n".join(lines) + "\n")
    save(args.out / "result.json", result)
    os.environ["WANDB_API_KEY"] = Path("/run/secrets/wandb").read_text().strip()
    run = wandb.init(
        entity="zhusq20",
        project="llm-memory-editability",
        name=plan["batch"] + "/comparison",
        group=plan["batch"],
        job_type="comparison",
        config=plan,
        dir=str(args.out),
        mode="online",
        settings=wandb.Settings(disable_git=True),
    )
    save(args.out / "wandb.json", {"id": run.id, "url": run.url})
    run.summary["result_sha256"] = sha(args.out / "result.json")
    run.summary["scientific_optimizer_updates"] = 0
    run.summary["new_scientific_updates_total"] = 13440
    for architecture, row in table.items():
        run.summary[architecture + "/double_new"] = row["double_new"]["accuracy"]
        run.summary[architecture + "/old_composition_retention"] = row["old_compositions"][
            "accuracy"
        ]
    run.save(str(args.out / "result.json"), base_path=str(args.out), policy="now")
    run.save(str(args.out / "report.md"), base_path=str(args.out), policy="now")
    identifier, url = run.id, run.url
    run.finish()
    deadline = time.monotonic() + 300
    while True:
        remote = wandb.Api(timeout=60).run(f"zhusq20/llm-memory-editability/{identifier}")
        if remote.state == "finished":
            break
        if time.monotonic() >= deadline:
            raise TimeoutError("Report W&B run has not reached finished")
        time.sleep(5)
    if remote.summary.get("result_sha256") != sha(args.out / "result.json"):
        raise AssertionError("Report W&B result identity differs")
    actual = remote.file("result.json").download(root=str(args.out / "cloud"), replace=False)
    if sha(actual.name) != sha(args.out / "result.json"):
        raise AssertionError("Report complete W&B JSON differs")
    save(
        args.out / "cloud-verification.json",
        {
            "passed": True,
            "id": identifier,
            "url": url,
            "canonical_full_json_exact": True,
            "result_sha256": sha(args.out / "result.json"),
            "scientific_optimizer_updates": 0,
        },
    )
    print(json.dumps({"passed": True, "wandb": url, "table": table}), flush=True)


if __name__ == "__main__":
    main()
