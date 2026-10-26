"""Publish the audited three-architecture atomic-CE endpoint comparison."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_alphaedit_discoloop as alpha  # noqa: E402

from llm_memory_editability.ce_architecture import paired_report  # noqa: E402
from llm_memory_editability.grok_depth import write_json  # noqa: E402


def fraction(value):
    accuracy = "NA" if value["accuracy"] is None else f"{100 * value['accuracy']:.2f}%"
    return f"{value['correct']}/{value['n']}（{accuracy}）"


def main():
    alpha.verify_source()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    args.out.mkdir(parents=True, exist_ok=False)
    for path, expected in plan["bound_inputs"].items():
        if alpha.digest(path) != expected:
            raise AssertionError("Reused immutable input changed")
    if (
        not json.loads(Path(plan["discoloop_audit"]).read_text())["passed"]
        or not json.loads(Path(plan["discoloop_cloud"]).read_text())["passed"]
    ):
        raise AssertionError("Reused DiscoLoop requires independent and cloud audits")
    roots = {"discoloop": Path(plan["discoloop_evaluation"])}
    sources = {
        plan["discoloop_audit"]: alpha.digest(plan["discoloop_audit"]),
        plan["discoloop_cloud"]: alpha.digest(plan["discoloop_cloud"]),
    }
    budgets, selected, costs = {}, {}, {}
    for arm in ("loop", "transformer"):
        root = Path(plan["output_root"]) / arm / "output"
        roots[arm] = root / "evaluation-mlp"
        stages = {}
        for stage in (
            "prepare",
            "development-mlp",
            "evaluation-mlp",
            "audit-mlp",
            "cloud",
        ):
            path = root / stage / "result.json"
            stages[stage] = json.loads(path.read_text())
            if stages[stage]["passed"] is not True:
                raise AssertionError("Every arm stage must pass before reporting")
            sources[str(path)] = alpha.digest(path)
        counts = {k: v["scientific_optimizer_updates"] for k, v in stages.items()}
        if counts != {
            "prepare": 0,
            "development-mlp": 640,
            "evaluation-mlp": 640,
            "audit-mlp": 0,
            "cloud": 0,
        }:
            raise AssertionError("Architecture edit budget differs from the frozen plan")
        if stages["audit-mlp"]["state_sha256"] != stages["evaluation-mlp"]["state_sha256"]:
            raise AssertionError("Independent audit did not replay all exact saved states")
        budgets[arm] = counts
        selected[arm] = stages["development-mlp"]["selected"]
        costs[arm] = {
            k: stages["evaluation-mlp"][k]
            for k in (
                "trainable_parameters",
                "exposures_per_fact",
                "compute",
                "wall_seconds",
                "peak_cuda_allocated_bytes",
            )
        }
    records = {}
    for arm, folder in roots.items():
        paths = sorted(folder.glob("evaluation-novel-*.json"))
        records[arm] = [json.loads(p.read_text()) for p in paths]
        sources.update({str(p): alpha.digest(p) for p in paths})
    result = paired_report(records, plan["statistics"])
    for arm in ("loop", "transformer"):
        if any(
            record["write_info"]["scope"] != "mlp"
            or record["write_info"]["updated_composite_supervision"] != 0
            for record in records[arm]
        ):
            raise AssertionError("Formal edit scope or supervision differs")
        for own, original in zip(records[arm], records["discoloop"], strict=True):
            for key in (
                "steps",
                "new_atomic_exposures",
                "new_supervised_tokens",
                "old_atomic_exposures",
                "old_composite_exposures",
                "old_distillation_slots",
            ):
                if own["write_info"][key] != original["write_info"][key]:
                    raise AssertionError("Formal edit exposure differs from DiscoLoop")
    result.update(
        passed=True,
        sources=sources,
        selected_recipes=selected,
        costs=costs,
        budgets=budgets,
        native_metrics={
            arm: value["native_metrics"] for arm, value in plan["architectures"].items()
        },
        new_foundation_updates=0,
        scientific_optimizer_updates=0,
        new_scientific_updates=2560,
        new_composition_supervision=0,
        discoloop_reused=True,
        independent_worlds=1,
        initializations_per_architecture=1,
        limits=plan["interpretation"],
    )
    os.environ["WANDB_API_KEY"] = Path("/run/secrets/wandb").read_text().strip()
    import wandb

    run = wandb.init(
        entity="zhusq20",
        project="llm-memory-editability",
        name=plan["batch"] + "/comparison",
        group=plan["batch"],
        job_type="comparison",
        dir=str(args.out),
        config={
            "facts": 32,
            "new_foundation_updates": 0,
            "new_scientific_updates": 2560,
        },
        settings=wandb.Settings(disable_git=True),
    )
    write_json(args.out / "wandb.json", {"id": run.id, "url": run.url})
    write_json(args.out / "result.json", result)
    run.summary["result_sha256"] = alpha.digest(args.out / "result.json")
    run.summary["scientific_optimizer_updates"] = 0
    run.save(str(args.out / "result.json"), base_path=str(args.out), policy="now")
    run.finish()
    remote = wandb.Api(timeout=60).run(f"zhusq20/llm-memory-editability/{run.id}")
    if remote.state != "finished":
        raise AssertionError("Comparison cloud run not finished")
    downloaded = remote.file("result.json").download(
        root=str(args.out / "cloud-verification"), replace=False
    )
    if alpha.digest(downloaded.name) != alpha.digest(args.out / "result.json"):
        raise AssertionError("Complete comparison cloud JSON differs")
    write_json(
        args.out / "cloud-receipt.json",
        {"passed": True, "full_json_exact": True, "id": run.id, "url": run.url},
    )
    lines = [
        "# 新增事实整个MLP原子CE：三架构比较",
        "",
        "普通Loop与普通Transformer复用原生300k基础，新增基础/外层训练0；每模型开发640与正式640编辑更新。DiscoLoop复用原独立编辑与审计。所有方法均无新增两跳答案监督。",
        "",
        "|架构|新原子|新→旧组合|旧→新组合|原正确旧原子保持|原正确旧组合保持|",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm, summary in result["architectures"].items():
        after, old = summary["conditions"]["after"], summary["extended_old_retained"]
        lines.append(
            "|"
            + "|".join(
                [
                    arm,
                    fraction(after["novel_atom"]),
                    fraction(after["first_hop"]),
                    fraction(after["second_hop"]),
                    fraction(old["atoms"]),
                    fraction(old["compositions"]),
                ]
            )
            + "|"
        )
    lines += [
        "",
        "每条事实独立从基础编辑；未验证累计持续写入。旧题在32状态重复，不是独立知识。完整JSON包含原子写成与旧前提保持的各模型条件子集及三架构共同子集、覆盖率、事实配对bootstrap及写前/写后旧池读数。",
        "",
        "基础组合能力、总/可编辑参数和计算不同；该比较评价当前配方下的整体行为，不能唯一归因写入传播接口。单图、每架构一个初始化。资源共享时间不作独占效率基准。",
        "",
        f"W&B报告：{run.url}",
    ]
    (args.out / "report.md").write_text("\n".join(lines) + "\n")
    print(
        json.dumps({"passed": True, "wandb": run.url, "architectures": list(records)}),
        flush=True,
    )


if __name__ == "__main__":
    main()
