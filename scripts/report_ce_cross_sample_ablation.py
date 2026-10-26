"""Report only independently replayed, cloud-verified continuous CE ablations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llm_memory_editability.ce_cross_sample_ablation_summary import (  # noqa: E402
    compare_with_control,
    summarize_streams,
)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for part in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def verify_source():
    path = ROOT / "source-manifest.json"
    if sha(path) != os.environ.get("LM_SOURCE_MANIFEST_SHA256"):
        raise ValueError("Frozen report source manifest differs")
    for name, expected in read(path)["files"].items():
        candidate = ROOT / name
        if (
            candidate.is_symlink()
            or not candidate.resolve().is_relative_to(ROOT.resolve())
            or sha(candidate) != expected
        ):
            raise ValueError("A frozen source file changed or escaped its root")


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


def verify_cloud(root, cloud, expected_stages, stage_roots=None):
    if cloud.get("passed") is not True or cloud.get("canonical_full_json_exact") is not True:
        raise ValueError("A complete canonical W&B JSON verification is required")
    receipts = cloud.get("runs", [])
    if len(receipts) != len(expected_stages) or {r["stage"] for r in receipts} != set(
        expected_stages
    ):
        raise ValueError("Cloud receipts do not cover exactly the completed stages")
    for receipt in receipts:
        stage_root = (stage_roots or {}).get(receipt["stage"], root)
        if receipt["result_sha256"] != sha(stage_root / receipt["stage"] / "result.json"):
            raise ValueError("Cloud receipt no longer binds its local stage result")


def load_records(root, expected_hashes, *, continuous_only=False):
    paths = sorted((root / "records").glob("*.json"))
    if {p.name for p in paths} != set(expected_hashes):
        raise ValueError("Saved record coverage differs from the bound result")
    records = []
    for path in paths:
        if sha(path) != expected_hashes[path.name]:
            raise ValueError("A saved scientific readout changed")
        record = read(path)
        if not continuous_only or record["stream"] in ("baseline", "forward", "reverse"):
            records.append(record)
    return records


def verify_gate(root):
    gate, audit = (read(root / stage / "result.json") for stage in ("gate", "audit-gate"))
    if gate.get("passed") is not True or audit.get("passed") is not True:
        raise ValueError("Write gate and its independent replay must both pass execution checks")
    valid = gate.get("gate_valid")
    if not isinstance(valid, bool) or gate.get("eligible") != valid:
        raise ValueError("Gate must declare consistent boolean gate_valid and eligible")
    if audit.get("gate_valid") != valid:
        raise ValueError("Independent gate replay disagrees about write validity")
    if audit.get("gate_result_sha256") != sha(root / "gate/result.json"):
        raise ValueError("Independent gate receipt no longer binds the actual gate")
    if audit.get("scientific_optimizer_updates") != 0:
        raise ValueError("Independent gate audit must not perform scientific updates")
    if gate.get("scientific_optimizer_updates") != 0:
        raise ValueError("Development write gate must not contain scientific updates")
    candidates = gate.get("candidates", [])
    if (
        len(candidates) not in (1, 2)
        or gate.get("development_optimizer_updates") != 640 * len(candidates)
        or audit.get("replayed_states") != 32 * len(candidates)
        or audit.get("summary_recomputed") is not True
    ):
        raise ValueError("Independent gate replay must cover every complete candidate")
    return gate, audit, valid


def verify_new_arm(root, prepare_root):
    """Failed write gates are explicit unrun outcomes, never zero accuracy."""
    gate, gate_audit, valid = verify_gate(root)
    cloud = read(root / "cloud/result.json")
    prepare = read(prepare_root / "prepare/result.json")
    if prepare.get("passed") is not True or prepare.get("scientific_optimizer_updates") != 0:
        raise ValueError("Architecture engineering prepare must pass without scientific updates")
    if (
        prepare.get("engineering_optimizer_updates") != 46
        or prepare.get("restored_theta0") is not True
    ):
        raise ValueError("All 46 engineering updates and exact foundation restoration are required")
    stages = ["prepare", "gate", "audit-gate"]
    science = audit = None
    if valid:
        stages += ["science", "audit"]
        science, audit = (read(root / stage / "result.json") for stage in ("science", "audit"))
        if (
            science.get("passed") is not True
            or science.get("gate_valid") is not True
            or science.get("scientific_optimizer_updates") != 2560
            or science.get("gate_result_sha256") != sha(root / "gate/result.json")
            or audit.get("passed") is not True
            or audit.get("gate_valid") is not True
            or audit.get("scientific_optimizer_updates") != 0
            or audit.get("replayed_states") != 129
            or audit.get("summary_recomputed") is not True
            or audit.get("science_result_sha256") != sha(root / "science/result.json")
        ):
            raise ValueError("Valid arm needs all 2560 updates and independent 129-state replay")
    elif (root / "science").exists() or (root / "audit").exists():
        raise ValueError("An invalid write gate must not enter science or scientific audit")
    verify_cloud(root, cloud, stages, {"prepare": prepare_root})
    if cloud.get("gate_valid") != valid:
        raise ValueError("Cloud verification disagrees about write validity")
    return {
        "status": "complete" if valid else "write_gate_failed_science_not_run",
        "gate_valid": valid,
        "scientific_optimizer_updates": 2560 if valid else 0,
        "gate": gate,
        "prepare": prepare,
        "gate_audit": gate_audit,
        "science": science,
        "audit": audit,
        "cloud": cloud,
        "stage_hashes": {
            stage: sha((prepare_root if stage == "prepare" else root) / stage / "result.json")
            for stage in [*stages, "cloud"]
        },
    }


def verify_control(reference, architecture):
    root = Path(reference["root"])
    completion_path = Path(reference["completion_manifest"])
    if sha(completion_path) != reference["completion_manifest_sha256"]:
        raise ValueError("Original completion manifest changed")
    completion = read(completion_path)
    outcome = completion.get("outcomes", {}).get(architecture, completion)
    if (
        completion.get("passed") is not True
        or completion.get("state") != "complete"
        or outcome.get("passed") is not True
        or outcome.get("state") != "complete"
        or outcome.get("independent_replay_passed") is not True
        or outcome.get("full_cloud_json_verified") is not True
        or outcome.get("scientific_updates") != 4480
    ):
        raise ValueError("Original control has not completed its frozen full matrix")
    science_path = root / "evaluate/result.json"
    if sha(science_path) != reference["evaluate_result_sha256"]:
        raise ValueError("Original control result changed")
    science, audit, cloud = (
        read(root / stage / "result.json") for stage in ("evaluate", "audit", "cloud")
    )
    if (
        science.get("passed") is not True
        or science.get("architecture") != architecture
        or science.get("scientific_optimizer_updates") != 4480
        or audit.get("passed") is not True
        or audit.get("scientific_optimizer_updates") != 0
        or audit.get("replayed_states") != 257
        or audit.get("summary_recomputed") is not True
        or audit.get("evaluate_result_sha256") != sha(science_path)
    ):
        raise ValueError("Original control's independent replay is incomplete or mismatched")
    verify_cloud(root, cloud, ("prepare", "evaluate", "audit"))
    return {
        "root": str(root),
        "science": science,
        "audit": audit,
        "cloud": cloud,
        "completion_manifest_sha256": sha(completion_path),
        "stage_hashes": {
            stage: sha(root / stage / "result.json")
            for stage in ("prepare", "evaluate", "audit", "cloud")
        },
    }


def table_row(summary):
    primary = summary["primary_endpoint"]
    return {
        "just_written_atoms": primary["just_written_atoms"],
        "new_atoms": primary["all_written_atoms"],
        "initially_correct_new_atoms_retained": primary["initially_correct_new_atoms_retained"],
        "double_new": primary["double_new"],
        "both_atoms_and_composition": primary["both_atoms_and_composition"],
        "composition_given_both_atoms": primary["composition_given_both_atoms"],
        "old_atoms": pooled_old(summary, "atoms"),
        "old_compositions": pooled_old(summary, "compositions"),
    }


def build_report(plan, batch, panel):
    summaries, controls, outcomes, comparisons, table = {}, {}, {}, {}, {}
    original_report = Path(plan["control_batch"]) / "report/output"
    if sha(original_report / "result.json") != plan["control_report_sha256"]:
        raise ValueError("Original comparison result changed")
    original_comparison = read(original_report / "result.json")
    if original_comparison.get("passed") is not True:
        raise ValueError("Original full comparison did not pass")
    report_cloud = read(original_report / "cloud-verification.json")
    if (
        report_cloud.get("passed") is not True
        or report_cloud.get("canonical_full_json_exact") is not True
        or report_cloud.get("result_sha256") != plan["control_report_sha256"]
    ):
        raise ValueError("Original comparison has no bound complete W&B JSON verification")
    for architecture in ("loop", "discoloop"):
        receipt = verify_control(plan["controls"][architecture], architecture)
        records = load_records(
            Path(receipt["root"]) / "evaluate",
            receipt["science"]["record_hashes"],
            continuous_only=True,
        )
        controls[architecture] = receipt
        summary = summarize_streams(records, panel, plan["evaluation"])
        summaries[architecture + "-control"] = summary
        table[architecture + "-control"] = {"status": "existing_control", **table_row(summary)}
        variants = {}
        for arm, settings in plan["arms"].items():
            if settings["architecture"] != architecture:
                continue
            root = batch / arm / "output"
            outcome = verify_new_arm(root, batch / (architecture + "-prepare") / "output")
            outcomes[arm] = outcome
            if not outcome["gate_valid"]:
                table[arm] = {
                    "status": outcome["status"],
                    "gate_valid": False,
                    "double_new": None,
                    "old_atoms": None,
                    "old_compositions": None,
                }
                continue
            records = load_records(root / "science", outcome["science"]["record_hashes"])
            new_summary = summarize_streams(records, panel, plan["evaluation"])
            # The audit independently re-computed this same continuous-only summary.
            for key, value in new_summary.items():
                if outcome["science"].get(key) != value:
                    raise ValueError(f"{arm}: recomputed statistics differ at {key}")
            summaries[arm] = variants[arm] = new_summary
            table[arm] = {"status": "complete", "gate_valid": True, **table_row(new_summary)}
        comparisons[architecture] = compare_with_control(
            summary, variants, panel, plan["evaluation"]
        )
    if set(outcomes) != set(plan["arms"]):
        raise ValueError("Not all planned arms belong to the two authorized architectures")
    return {
        "passed": True,
        "batch": plan["batch"],
        "status": "complete"
        if all(o["gate_valid"] for o in outcomes.values())
        else "complete_with_write_gate_failures",
        "summaries": summaries,
        "comparisons": comparisons,
        "table": table,
        "arm_outcomes": outcomes,
        "verified_controls": controls,
        "original_comparison_cloud_verification": report_cloud,
        "existing_transformer_reference": original_comparison["table"]["transformer"],
        "created_date": datetime.now(ZoneInfo("America/Chicago")).date().isoformat(),
        "new_scientific_updates": sum(o["scientific_optimizer_updates"] for o in outcomes.values()),
        "new_development_updates": sum(
            o["gate"]["development_optimizer_updates"] for o in outcomes.values()
        ),
        "new_engineering_updates": sum(
            next(
                o["prepare"]["engineering_optimizer_updates"]
                for arm, o in outcomes.items()
                if plan["arms"][arm]["architecture"] == architecture
            )
            for architecture in ("loop", "discoloop")
        ),
        "new_foundation_updates": 0,
        "new_writer_outer_updates": 0,
        "new_composition_supervision": 0,
        "independent_replay_passed": True,
        "full_cloud_json_verified": True,
        "worlds": 1,
        "initializations_per_architecture": 1,
    }


def markdown_report(result):
    valid_count = sum(o["gate_valid"] for o in result["arm_outcomes"].values())
    completion_text = (
        f"本批{valid_count}臂完成正式读出、独立重载及W&B完整JSON核验；"
        if valid_count
        else "本批没有臂通过写入门槛，正式阶段未运行；"
    )
    completion_text += "写入失败的开发状态及其独立重载和云端核验仍完整保留。"
    lines = [
        "# 连续学习新事实的单变量消融",
        "",
        "## Material Passport",
        "",
        "- Origin Skill: academic-research-suite experiment-agent",
        "- Origin Mode: run",
        "- Origin Date: " + result["created_date"],
        "- Verification Status: VERIFIED",
        "- Version Label: ce-cross-sample-ablation-v1",
        "",
        "Loop与DiscoLoop各比较降低学习率、跨事实保留Adam状态、移除参数改动上限。"
        "原配方只复算已有连续流，不重跑；每新增有效臂为两条64事实流、2560次更新。"
        + completion_text,
        "",
        "| 配方 | 刚写原子 | 终点原子 | 初正确新原子保持 | 双新组合 | 两原子＋组合 | "
        "两原子正确时组合 | 旧原子保持 | 旧组合保持 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in result["table"].items():
        if row["status"] == "write_gate_failed_science_not_run":
            lines.append(
                f"| {name}（写入门槛失败，正式未跑） | NA | NA | NA | NA | NA | NA | NA | NA |"
            )
            continue
        conditional = row["composition_given_both_atoms"]
        cells = [
            name,
            f"{row['just_written_atoms']['correct']}/{row['just_written_atoms']['n']}",
            f"{row['new_atoms']['correct']}/{row['new_atoms']['n']}",
            percentage(row["initially_correct_new_atoms_retained"]["accuracy"]),
            percentage(row["double_new"]["accuracy"]),
            percentage(row["both_atoms_and_composition"]["accuracy"]),
            f"{percentage(conditional['accuracy'])}（覆盖{conditional['coverage_n']}/{conditional['full_n']}）",
            f"{percentage(row['old_atoms']['accuracy'])}（{row['old_atoms']['correct']}/{row['old_atoms']['n']}）",
            f"{percentage(row['old_compositions']['accuracy'])}（{row['old_compositions']['correct']}/{row['old_compositions']['n']}）",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    reference = result["existing_transformer_reference"]
    lines += [
        "",
        "普通Transformer沿用已完成参照：终点新原子"
        f"{reference['new_atoms']['correct']}/{reference['new_atoms']['n']}，"
        f"双新组合{percentage(reference['double_new']['accuracy'])}，"
        f"旧组合原正确保持{percentage(reference['old_compositions']['accuracy'])}。",
        "",
        "| 变体相对同架构原配方 | 双新差值/95%区间（pp） | 新原子差值/95%区间（pp） | "
        "共同原子前提上的组合差值/95%区间（pp） | 共同覆盖 |",
        "|---|---:|---:|---:|---:|",
    ]
    for comparison in result["comparisons"].values():
        for arm, variant in comparison["variants"].items():
            delta = variant["paired_differences"]["variant_minus_control"]
            conditional = variant["common_atomic_prerequisites"]["variant"]
            common_delta = variant["common_prerequisite_paired_differences"][
                "variant_minus_control"
            ]

            def interval(value):
                if value["mean"] is None:
                    return "NA"
                lo, hi = value["ci95_pair_bootstrap"]
                return f"{100 * value['mean']:+.2f} [{100 * lo:+.2f}, {100 * hi:+.2f}]"

            cells = [
                arm,
                interval(delta["double_new"]),
                interval(delta["all_written_atoms"]),
                interval(common_delta),
                f"{conditional['coverage_n']}/{conditional['full_n']}",
            ]
            lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        "双新终点以32个事实对为单位，先平均两个顺序；未闭合链不进入组合保持。"
        "旧保持按各模型原正确题计数，七池及原子/组合分类另列，原正确为0时为NA。"
        "重复查询、两个顺序及时间点均不作为独立知识图。",
        "",
        "每个变体相对同架构原配方的配对分差、95%事实对bootstrap区间、共同两原子前提子集及覆盖，"
        "逐事实刚写/终点保持、年龄曲线、混合组合和必要旧原子、严格两次单跳参考均见result.json。"
        "所有预定变体保留，不按结果选最优臂。",
        "",
        f"实际新增科学更新{result['new_scientific_updates']:,}；新增基础训练和writer外层训练均为0，"
        "新组合答案监督为0。仅一张已观察图、每架构一个已有初始化；"
        "区间描述固定面板对事实对重采样的敏感性，不能代表跨图或初始化稳定性。",
        f"开发写入更新{result['new_development_updates']:,}，工程更新{result['new_engineering_updates']:,}，"
        "独立审计更新为0；写入门槛只按当前事实学习选择，不使用组合或保持结果选择。",
        "",
        "原始预测和参数状态保留在批次持久目录；本报告由AI辅助整理。",
    ]
    return "\n".join(lines) + "\n"


def publish_and_verify(plan, out, result):
    import wandb

    os.environ["WANDB_API_KEY"] = Path("/run/secrets/wandb").read_text().strip()
    run = wandb.init(
        entity="zhusq20",
        project="llm-memory-editability",
        name=plan["batch"] + "/comparison",
        group=plan["batch"],
        job_type="comparison",
        config=plan,
        dir=str(out),
        mode="online",
        settings=wandb.Settings(disable_git=True),
    )
    save(out / "wandb.json", {"id": run.id, "url": run.url})
    try:
        run.summary["result_sha256"] = sha(out / "result.json")
        run.summary["scientific_optimizer_updates"] = 0
        run.summary["new_scientific_updates_total"] = result["new_scientific_updates"]
        for arm, row in result["table"].items():
            run.summary[arm + "/status"] = row["status"]
            if row["double_new"] is not None:
                run.summary[arm + "/double_new"] = row["double_new"]["accuracy"]
                run.summary[arm + "/old_composition_retention"] = row["old_compositions"][
                    "accuracy"
                ]
        run.save(str(out / "result.json"), base_path=str(out), policy="now")
        run.save(str(out / "report.md"), base_path=str(out), policy="now")
        identifier, url = run.id, run.url
        run.finish()
    except BaseException:
        run.finish(exit_code=1)
        raise
    deadline = time.monotonic() + 300
    while True:
        remote = wandb.Api(timeout=60).run(f"zhusq20/llm-memory-editability/{identifier}")
        if remote.state == "finished":
            break
        if time.monotonic() >= deadline:
            raise TimeoutError("Comparison W&B run has not reached finished")
        time.sleep(5)
    if (
        remote.summary.get("result_sha256") != sha(out / "result.json")
        or remote.summary.get("scientific_optimizer_updates") != 0
        or remote.summary.get("new_scientific_updates_total") != result["new_scientific_updates"]
    ):
        raise AssertionError("Comparison W&B identity or update counts differ")
    actual = remote.file("result.json").download(root=str(out / "cloud"), replace=False)
    if sha(actual.name) != sha(out / "result.json"):
        raise AssertionError("Comparison complete cloud JSON differs from local evidence")
    receipt = {
        "passed": True,
        "id": identifier,
        "url": url,
        "canonical_full_json_exact": True,
        "result_sha256": sha(out / "result.json"),
        "scientific_optimizer_updates": 0,
    }
    save(out / "cloud-verification.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "result.json").exists() or (args.out / "failure.json").exists():
        raise FileExistsError("Never replace a completed or failed comparison")
    try:
        verify_source()
        plan = read(args.plan)
        base_plan_path = ROOT / plan["base_plan"]
        if sha(base_plan_path) != plan["base_plan_sha256"]:
            raise ValueError("Original frozen base plan changed")
        base_plan = read(base_plan_path)
        panel_path = ROOT / base_plan["panel"]
        if sha(panel_path) != base_plan["panel_sha256"]:
            raise ValueError("Frozen evaluation panel changed")
        result = build_report(plan, args.batch, read(panel_path))
        result["source_sha256"] = sha(ROOT / "source-manifest.json")
        save(args.out / "result.json", result)
        (args.out / "report.md").write_text(markdown_report(result))
        receipt = publish_and_verify(plan, args.out, result)
        print(
            json.dumps({"passed": True, "wandb": receipt["url"], "table": result["table"]}),
            flush=True,
        )
    except BaseException as error:
        save(args.out / "failure.json", {"error": repr(error), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
