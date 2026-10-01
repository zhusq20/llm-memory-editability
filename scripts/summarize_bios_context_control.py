"""Audit CPU-only paired context controls; full fixed cohorts, no mastery filtering.

This script never trains or reads model weights. Isolated-document attention is
compared with the original low-exception, unrestricted-context training matrix.
Intermediate direct paths and NLL are observed; autonomous two-step is available
at the final checkpoint in the frozen context producer.
"""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from summarize_bios_shortcut_control import (
    read_arrays,
    read_json,
    score_direct,
    score_two_step,
    write_csv,
)

from llm_memory_editability.bios_context_control import (
    CHECKPOINTS,
    context_sources,
    fact_causal_mask,
    validate_study,
)
from llm_memory_editability.bios_cross import (
    CHAINS,
    CONDITIONS,
    documents,
    make_cross_world,
    qa_schedule,
)
from llm_memory_editability.bios_cross_continue import expected_exposure
from llm_memory_editability.bios_cross_train import learning_metrics, source_hashes
from llm_memory_editability.bios_data import array_hash, write_json

ROOT = Path(__file__).resolve().parents[1]
MECHANISM = ROOT / "results/bios-mechanism-dev-v1"
PHASES = ("unrestricted", "isolated")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prior_audit_complete(audit):
    if "complete" in audit:
        return audit["complete"] and audit.get("passed", True) and not audit.get("errors", [])
    # P1 intentionally uses a different completion schema.
    return (
        audit.get("state") == "complete"
        and audit.get("passed_available_predictions") is True
        and not audit.get("missing")
        and all(
            audit.get(done) == expected
            for done, expected in (
                ("complete_models", 24),
                ("complete_new_edits", 192),
                ("complete_reference_edits", 96),
                ("prediction_checkpoints_recomputed", 1152),
                ("weights_and_optimizer_states_verified", 192),
            )
        )
    )


def audit_launch(root, ledger):
    gate = read_json(root / "launch-gate.json", ledger)
    if not gate["no_edits"] or gate["full_matrix"] != {
        "world": [0, 1],
        "seed": [0, 1],
        "organization": list(CONDITIONS),
        "width": 256,
        "steps": 15360,
        "models": 12,
    }:
        raise ValueError("Conditional launch gate matrix changed")
    for section in ("sources_sha256", "audits_sha256"):
        for name, expected in gate[section].items():
            path = ROOT / name
            if digest(path) != expected:
                raise ValueError(f"Conditional launch gate input changed: {name}")
            ledger[str(path)] = expected
            if name.endswith("/audit.json") and not prior_audit_complete(
                json.loads(path.read_text())
            ):
                raise ValueError(f"Conditional launch input is incomplete: {name}")
    launch = read_json(root / "launch-contract.json", ledger)
    if launch["sources"] != context_sources() or launch["edit_cases"] != 0:
        raise ValueError("Dispatcher source / edit contract mismatch")
    if launch["learning_runs"] != 12 or len(launch["jobs"]) != 12:
        raise ValueError("Dispatcher matrix is incomplete")
    identities = {(j["world"], j["seed"], j["condition"]) for j in launch["jobs"]}
    if identities != {(w, s, c) for w in (0, 1) for s in (0, 1) for c in CONDITIONS}:
        raise ValueError("Dispatcher matrix identities changed")
    if launch["runner_sha256"] != digest(ROOT / "scripts/run_bios_context_control.py"):
        raise ValueError("Dispatcher code changed")
    if launch["config_sha256"] != digest(ROOT / "configs/bios-context-control-v1.json"):
        raise ValueError("Dispatcher config changed")


def audit_run(original, isolated, world, identity, ledger):
    configs = [read_json(path / "config.json", ledger) for path in (original, isolated)]
    launch = read_json(isolated / "launch-contract.json", ledger)
    done = read_json(isolated / "complete.json", ledger)
    if (
        done.get("status") != "complete"
        or done.get("phase") != "P3-context-control-learning-only"
        or done.get("learning_steps") != 15360
        or done.get("edit_cases") != 0
    ):
        raise ValueError("Context producer completion identity mismatch")
    low, high = configs
    validate_study(launch["study"])
    for key in ("world", "seed", "condition"):
        if any(obj[key] != identity[key] for obj in (low, high, launch)):
            raise ValueError(f"Run path / producer identity mismatch: {key}")
    for key in (
        "model",
        "initial_sha256",
        "documents_sha256",
        "qa_sha256",
        "prompts_sha256",
        "truth_sha256",
        "torch",
        "numpy",
        "precision",
    ):
        if low[key] != high[key]:
            raise ValueError(f"Paired training control mismatch: {key}")
    if high["model"]["width"] != 256 or world.seed != identity["world"]:
        raise ValueError("Wrong context world or model width")
    if any(conf["sources"] != source_hashes() for conf in configs):
        raise ValueError("Frozen training source changed")
    if launch["sources"] != context_sources():
        raise ValueError("Context producer source changed")
    if high["study"]["context_contract"] != launch["sources"]:
        raise ValueError("Context source contract mismatch")
    intervention = {
        "training_document_attention": "causal within each six-token fact only",
        "absolute_positions": "unchanged 0 through 59; never reset within facts",
        "document_tokens_batches_and_supervision": "unchanged",
        "QA_training_attention": "original causal attention",
        "all_inference_attention": "original causal attention",
        "mask_sha256": array_hash(fact_causal_mask().numpy()),
    }
    if launch["intervention"] != intervention:
        raise ValueError("Unexpected attention intervention")
    if high["study"]["context_intervention"] != intervention:
        raise ValueError("Training attention provenance mismatch")
    for key in (
        "width",
        "layers",
        "heads",
        "steps",
        "checkpoints",
        "lr",
        "documents_per_step",
        "facts_per_document",
        "QA_per_chain_per_step",
        "document_QA_weights",
    ):
        if any(conf["study"][key] != launch["study"][key] for conf in configs):
            raise ValueError(f"Training budget changed: {key}")
    expected = {
        "documents": documents(world, identity["condition"]),
        "qa": qa_schedule(world, 15360),
    }
    for path in (original, isolated):
        saved = read_arrays(path / "schedule.npz", ledger)
        for name, value in expected.items():
            np.testing.assert_array_equal(saved[name], value)
            if high[f"{name}_sha256"] != array_hash(value):
                raise ValueError(f"Training schedule hash mismatch: {name}")
    expected_hashes = {
        "documents_sha256": array_hash(expected["documents"]),
        "qa_sha256": array_hash(expected["qa"]),
        "prompts_sha256": array_hash(world.prompts),
        "truth_sha256": array_hash(world.answers),
    }
    if launch["data"] != expected_hashes:
        raise ValueError("Context data contract mismatch")
    if any(high[key] != value for key, value in expected_hashes.items()):
        raise ValueError("Context scoring world mismatch")
    if any((isolated / "edits").glob("*")):
        raise ValueError("Context learning control unexpectedly contains edits")
    return high, expected


def masks(world, chain):
    ids = world.derived_ids[chain]
    for split, pool in (
        ("all", ids),
        ("trained", world.train_ids[chain]),
        ("heldout", world.heldout_ids[chain]),
    ):
        for cohort, mask in (
            ("all", np.ones(len(ids), dtype=bool)),
            ("original_exception", world.exceptions[chain]),
            ("ordinary", ~world.exceptions[chain]),
        ):
            yield split, cohort, np.isin(ids, pool) & mask


def path_rows(identity, world, chain, phase, arrays, method):
    ids = world.derived_ids[chain]
    a = (
        {k: arrays[k][ids] for k in ("prediction", "ended", "correct")}
        if method == "direct"
        else arrays
    )
    actual = world.answers[world.actual_ids[chain]]
    truth = world.answers[ids]
    conflict = actual != truth
    rows = []
    for split, cohort, take in masks(world, chain):
        n = int(take.sum())
        actual_wrong = a["ended"] & (a["prediction"] == actual) & conflict
        correct = int(a["correct"][take].sum())
        wrong_actual = int(actual_wrong[take].sum())
        row = {
            **identity,
            "phase": phase,
            "method": method,
            "chain": CHAINS[chain],
            "split": split,
            "cohort": cohort,
            "n": n,
            "correct": correct,
            "accuracy": correct / n,
            "wrong_actual": wrong_actual,
            "wrong_actual_per_query": wrong_actual / n,
            "other_error": n - correct - wrong_actual,
            "termination_error": int((~a["ended"][take]).sum()),
            "conflict_n": int((take & conflict).sum()),
            "overlap_n": int((take & ~conflict).sum()),
            "nll": float(arrays["value_nll"][ids[take]].mean()) if method == "direct" else None,
        }
        if method == "two_step":
            direct, after = a["direct_correct"][take], a["correct"][take]
            row.update(
                bridge_valid=int(a["bridge_valid"][take].sum()),
                bridge_correct=int(a["bridge_correct"][take].sum()),
                bridge_valid_rate=float(a["bridge_valid"][take].mean()),
                bridge_accuracy=float(a["bridge_correct"][take].mean()),
                direct_only=int((direct & ~after).sum()),
                two_step_only=int((~direct & after).sum()),
                direct_accuracy=float(direct.mean()),
                two_step_gap=float(after.mean() - direct.mean()),
            )
        rows.append(row)
    return rows


def component_rows(identity, world, chain, phase, arrays):
    groups = world.memberships[chain]
    root = world.root_ids[chain][groups]
    membership = arrays["correct"][world.membership_ids[chain]]
    default = arrays["correct"][root]
    rows = []
    for split, cohort, take in masks(world, chain):
        row = {
            **identity,
            "phase": phase,
            "chain": CHAINS[chain],
            "split": split,
            "cohort": cohort,
            "n": int(take.sum()),
            "both_prerequisites": float((membership & default)[take].mean()),
        }
        for label, query_ids in (
            ("membership", world.membership_ids[chain]),
            ("root", root),
            ("actual", world.actual_ids[chain]),
        ):
            row[f"{label}_accuracy"] = float(arrays["correct"][query_ids[take]].mean())
            row[f"{label}_nll"] = float(arrays["value_nll"][query_ids[take]].mean())
        rows.append(row)
    return rows


def pair_rows(identity, world, chain, original, isolated, method):
    ids = world.derived_ids[chain]
    low = original["correct"][ids] if method == "direct" else original["correct"]
    high = isolated["correct"][ids] if method == "direct" else isolated["correct"]
    rows = []
    for split, cohort, take in masks(world, chain):
        a, b = low[take], high[take]
        rows.append(
            {
                **identity,
                "method": method,
                "chain": CHAINS[chain],
                "split": split,
                "cohort": cohort,
                "n": int(take.sum()),
                "unrestricted_accuracy": float(a.mean()),
                "isolated_accuracy": float(b.mean()),
                "isolated_minus_unrestricted": float(b.mean() - a.mean()),
                "unrestricted_only": int((a & ~b).sum()),
                "isolated_only": int((~a & b).sum()),
                "both_correct": int((a & b).sum()),
                "both_wrong": int((~a & ~b).sum()),
            }
        )
    return rows


def organization_rows(rows):
    keys = ("world", "seed", "step", "phase", "method", "split", "cohort")
    groups = defaultdict(dict)
    for row in rows:
        groups[tuple(row[k] for k in keys)][row["condition"], row["chain"]] = row["accuracy"]
    effects, missing = [], []
    for identity, values in sorted(groups.items()):
        labels = dict(zip(keys, identity, strict=True))
        if set(values) != {(c, chain) for c in CONDITIONS for chain in CHAINS}:
            missing.append(labels)
            continue
        effects.append(
            {
                **labels,
                "matching_effect": 0.5
                * (
                    values["company", "company"]
                    - values["project", "company"]
                    + values["project", "project"]
                    - values["company", "project"]
                ),
                **{
                    f"{org}_mean_benefit": 0.5
                    * sum(values[org, chain] - values["neither", chain] for chain in CHAINS)
                    for org in ("company", "project")
                },
                "company_unmatched_benefit": values["company", "project"]
                - values["neither", "project"],
                "project_unmatched_benefit": values["project", "company"]
                - values["neither", "company"],
            }
        )
    pair_keys = tuple(k for k in keys if k != "phase")
    paired = defaultdict(dict)
    for row in effects:
        paired[tuple(row[k] for k in pair_keys)][row["phase"]] = row
    interactions = []
    for identity, phases in sorted(paired.items()):
        if set(phases) != set(PHASES):
            missing.append(dict(zip(pair_keys, identity, strict=True)))
            continue
        metrics = [k for k in phases[PHASES[0]] if k not in keys]
        interactions.append(
            {
                **dict(zip(pair_keys, identity, strict=True)),
                **{
                    f"isolated_minus_unrestricted_{k}": phases["isolated"][k]
                    - phases["unrestricted"][k]
                    for k in metrics
                },
            }
        )
    return effects, interactions, missing


def verify_two_step_source(path, original, world, step, ledger):
    provenance = read_json(path / "complete.json", ledger)
    if not provenance["complete"] or provenance["oracle_bridging"]:
        raise ValueError("Invalid unrestricted autonomous two-step provenance")
    identity = provenance["identity"]
    if Path(identity["run"]).resolve() != original.resolve():
        raise ValueError("Wrong unrestricted two-step source run")
    expected = {
        "config_sha256": ledger[str(original / "config.json")],
        "direct_predictions_sha256": ledger[str(original / f"predictions-{step}.npz")],
        "answer_sha256": array_hash(world.answers),
    }
    if any(identity[k] != v for k, v in expected.items()):
        raise ValueError("Unrestricted two-step identity mismatch")


def aggregate(rows, keys, metrics):
    """Case means only; retain both individual worlds plus a labeled pooled mean."""
    groups = defaultdict(list)
    for row in rows:
        for world in (str(row["world"]), "all"):
            normalized = {**row, "world": world}
            groups[tuple(normalized[k] for k in keys)].append(row)
    result = []
    for identity, group in sorted(groups.items()):
        row = {**dict(zip(keys, identity, strict=True)), "cases": len(group)}
        if "n" in group[0]:
            row["n"] = sum(r["n"] for r in group)
        for metric in metrics:
            values = [r.get(metric) for r in group if r.get(metric) is not None]
            row[metric] = float(np.mean(values)) if values else None
        if "isolated_minus_unrestricted" in group[0]:
            delta = [r["isolated_minus_unrestricted"] for r in group]
            row.update(
                positive_cases=sum(v > 0 for v in delta),
                negative_cases=sum(v < 0 for v in delta),
                min_change=min(delta),
                max_change=max(delta),
            )
        result.append(row)
    return result


def write_report(output, audit, tables):
    lines = ["# 文档内跨事实注意力阻断：配对开发实验", ""]
    if not audit["complete"]:
        lines.append("审计尚未完整；不发布效果结论。")
        (output / "report.md").write_text("\n".join(lines) + "\n")
        return

    def cell(name, **keys):
        rows = [r for r in tables[name] if all(r[k] == v for k, v in keys.items())]
        if len(rows) != 1:
            raise ValueError(f"Report cell missing or duplicated: {name}/{keys}")
        return rows[0]

    lines += [
        "正式状态：12/12 模型配对、72/72 学习节点配对、24/24 终点两步链配对；"
        "所有读取的来源文件在分析结束后重新核验哈希。原启动门控及其 P0/P1/P2/P3 "
        "审计身份已验证。未读取权重，未启动额外训练或编辑。",
        "",
        "## 比较定义",
        "",
        "约 758 万参数，2 个世界 × 2 个初始化 × 3 种组织方式。使用原低例外率世界；"
        "同一人员、事实、派生目标、初始化、文档张量、绝对位置、监督位置、QA 日程、"
        "步数与样本暴露保持配对。仅在训练文档中删除跨六-token 事实的注意力边，"
        "QA 训练及所有推理仍使用原始完整因果注意力。原始条件标为 unrestricted，"
        "阻断条件标为 isolated。",
        "",
        "全部结果按完整值和 EOS 评分。以下是模型—查询链单元等权平均，"
        "两世界均值含 24 个单元，每世界 12 个；基础总准确率和 NLL 每模型计一次。"
        "全体人员均保留，未按训练后掌握程度筛选。重复人员与查询不是独立重复实验，"
        "无 query 级置信区间。两个复用世界属于开发证据。",
        "",
        "## 终点直接查询",
        "",
        "| 世界 | 固定 heldout 人员 | 原始准确率 | 阻断准确率 | 阻断 − 原始 |",
        "|---|---|---:|---:|---:|",
    ]
    for world in ("all", "0", "1"):
        for cohort in ("all", "ordinary", "original_exception"):
            row = cell(
                "paired-summary.csv",
                world=world,
                step=15360,
                method="direct",
                split="heldout",
                cohort=cohort,
            )
            lines.append(
                f"| {world} | {cohort} | {100 * row['unrestricted_accuracy']:.4f}% | "
                f"{100 * row['isolated_accuracy']:.4f}% | "
                f"{100 * row['isolated_minus_unrestricted']:+.4f} pp |"
            )
    lines += [
        "",
        "## 原例外的答案、基础知识与终点两步",
        "",
        "| 世界 | 条件 | direct | 错答 actual/全部查询 | 其他错误 | actual 事实 | "
        "成员事实 | 同人对应根事实 | 自主两步 | 两步 − direct |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for world in ("all", "0", "1"):
        for phase in PHASES:
            keys = dict(
                world=world, step=15360, phase=phase, split="heldout", cohort="original_exception"
            )
            direct = cell("path-summary.csv", **keys, method="direct")
            two = cell("path-summary.csv", **keys, method="two_step")
            facts = cell("component-summary.csv", **keys)
            values = [
                direct["accuracy"],
                direct["wrong_actual_per_query"],
                1 - direct["accuracy"] - direct["wrong_actual_per_query"],
                facts["actual_accuracy"],
                facts["membership_accuracy"],
                facts["root_accuracy"],
                two["accuracy"],
                two["two_step_gap"],
            ]
            lines.append(
                f"| {world} | {phase} | " + " | ".join(f"{100 * v:.4f}" for v in values) + " |"
            )
    lines += [
        "",
        "上表前七个数值列单位为 %，最后一列单位为 pp。other 包括终止错误，"
        "完整终止计数保留在 path-strata.csv；普通人 actual/default 标签重合另列 overlap_n，"
        "不把重合答案强行解释为 actual 路径。根事实采用真值组织索引独立基础查询，"
        "按人员加权，同一组织的 root 可以重复计入，不能将人员数当作独立 root 数；"
        "仅用于知识测量，从未送入自主两步。两步第一跳由模型生成；额外推理调用"
        "改变计算量，错误组织也可能共享城市，因此两步正确不直接证明内部组合路径。",
        "",
        "## 基础知识水平及中间节点",
        "",
        "| 世界 | 节点 | 条件 | 全体基础事实准确率 | 基础事实 value NLL |",
        "|---|---:|---|---:|---:|",
    ]
    for world in ("all", "0", "1"):
        for step in (5120, 10240, 15360):
            for phase in PHASES:
                row = cell("base-summary.csv", world=world, step=step, phase=phase)
                lines.append(
                    f"| {world} | {step} | {phase} | {100 * row['base_accuracy']:.4f}% | "
                    f"{row['base_nll']:.6f} |"
                )
    lines += [
        "",
        "所有六个节点（0/1280/2560/5120/10240/15360）的 direct 答案分解、"
        "NLL、同人基础事实及组织交互完整保留在 CSV。自主两步仅在 15360 终点测量；"
        "没有把 5120/10240 节点的直接预测冒充两步结果。",
        "",
        "## 组织方式 × 测试关系",
        "",
        "| 世界 | 人员 | 条件 | 匹配交互 | 公司组织两链均益 | 项目组织两链均益 | "
        "公司组织不匹配链收益 | 项目组织不匹配链收益 |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for world in ("all", "0", "1"):
        for cohort in ("all", "original_exception"):
            for phase in PHASES:
                row = cell(
                    "organization-summary.csv",
                    world=world,
                    step=15360,
                    phase=phase,
                    method="direct",
                    split="heldout",
                    cohort=cohort,
                )
                values = [
                    row[k]
                    for k in (
                        "matching_effect",
                        "company_mean_benefit",
                        "project_mean_benefit",
                        "company_unmatched_benefit",
                        "project_unmatched_benefit",
                    )
                ]
                lines.append(
                    f"| {world} | {cohort} | {phase} | "
                    + " | ".join(f"{100 * v:+.4f}" for v in values)
                    + " |"
                )
    lines += [
        "",
        "上表单位均为 pp。匹配交互为 0.5×[(公司组织公司查询−项目组织公司查询)"
        "+(项目组织项目查询−公司组织项目查询)]。收益均减去 neither 的对应查询；"
        "不匹配收益分别只看公司组织的项目查询和项目组织的公司查询。先在相同"
        "世界/初始化内计算交互，再等权平均；逐 block 值保留在组织 CSV 中。",
        "",
        "## 解释边界",
        "",
        "注意力阻断干预训练计算，同时可能改变优化难度与知识习得水平。必须将"
        "直接准确率、基础知识/NLL、actual 答案吸引、两步能力和全体 heldout 代价"
        "同时解释：若两步差距缩小伴随基础知识或两步能力下降，不能称为选择性"
        "路径修复。阻断后残留的组织差异仍可能来自批次梯度相关性、文档槽位"
        "绝对位置或优化；即使效应减弱，也不能唯一定位某个内部电路。",
        "",
        "精确来源和物理路径见 audit.json、sources.json；启动前因果对比及"
        "门控证据来自被哈希锁定的 launch-gate.json。",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n")


def summarize(isolated_root, original_root, two_step_root, output, require_complete=False):
    isolated_root, original_root, two_step_root, output = [
        Path(p).resolve() for p in (isolated_root, original_root, two_step_root, output)
    ]
    if (isolated_root / "width-256").exists():
        isolated_root /= "width-256"
    if (two_step_root / "width-256").exists():
        two_step_root /= "width-256"
    for source in (isolated_root, original_root, two_step_root):
        if source == output or source in output.parents or output in source.parents:
            raise ValueError("Summary output overlaps raw sources")
    ledger, missing, paths, paired, components, metrics = {}, [], [], [], [], []
    audit_launch(
        isolated_root.parent if isolated_root.name == "width-256" else isolated_root, ledger
    )
    counts = {"complete_pairs": 0, "learning_pairs": 0, "two_step_pairs": 0}
    for w in (0, 1):
        world = make_cross_world(w, ROOT / "data/bios-organization-v1")
        for seed in (0, 1):
            for condition in CONDITIONS:
                name = f"world-{w}-seed-{seed}-{condition}"
                original, isolated = original_root / name, isolated_root / name
                if not (isolated / "complete.json").exists():
                    missing.append(str(isolated / "complete.json"))
                    continue
                identity = {"world": w, "seed": seed, "condition": condition}
                conf, schedule = audit_run(original, isolated, world, identity, ledger)
                timelines = [
                    {p["step"]: p for p in read_json(path / "learning.json", ledger)}
                    for path in (original, isolated)
                ]
                terminal = None
                for step in CHECKPOINTS:
                    arrays = [
                        score_direct(
                            read_arrays(path / f"predictions-{step}.npz", ledger), world.answers
                        )
                        for path in (original, isolated)
                    ]
                    exposure = expected_exposure(
                        world, schedule["documents"], schedule["qa"], step, conf["study"]["lr"]
                    )
                    for key, value in zip(("exposure", "slots", "weighted"), exposure, strict=True):
                        np.testing.assert_array_equal(arrays[0][key], arrays[1][key])
                        np.testing.assert_allclose(arrays[1][key], value, rtol=1e-11, atol=1e-12)
                    if step == 0:
                        for key in ("prediction", "ended", "correct", "value_nll"):
                            np.testing.assert_array_equal(arrays[0][key], arrays[1][key])
                    node = {**identity, "step": step}
                    for phase, a, timeline in zip(PHASES, arrays, timelines, strict=True):
                        measured = learning_metrics(world, a)
                        for key, value in measured.items():
                            observed = timeline[step][key]
                            if (value is None and observed is not None) or (
                                value is not None and not np.isclose(value, observed)
                            ):
                                raise ValueError(
                                    f"Prediction/timeline mismatch: {phase}/{step}/{key}"
                                )
                        metrics.append({**node, "phase": phase, **measured})
                        for chain in range(2):
                            paths.extend(path_rows(node, world, chain, phase, a, "direct"))
                            components.extend(component_rows(node, world, chain, phase, a))
                    for chain in range(2):
                        paired.extend(pair_rows(node, world, chain, *arrays, "direct"))
                    counts["learning_pairs"] += 1
                    if step == 15360:
                        terminal = arrays
                low_two = two_step_root / name / "learning-15360"
                verify_two_step_source(low_two, original, world, 15360, ledger)
                for chain, chain_name in enumerate(CHAINS):
                    arrays = [
                        score_two_step(
                            read_arrays(low_two / f"{chain_name}.npz", ledger),
                            world,
                            chain,
                            terminal[0],
                        ),
                        score_two_step(
                            read_arrays(isolated / f"two-step-{chain_name}.npz", ledger),
                            world,
                            chain,
                            terminal[1],
                        ),
                    ]
                    node = {**identity, "step": 15360}
                    for phase, a in zip(PHASES, arrays, strict=True):
                        paths.extend(path_rows(node, world, chain, phase, a, "two_step"))
                    paired.extend(pair_rows(node, world, chain, *arrays, "two_step"))
                    counts["two_step_pairs"] += 1
                counts["complete_pairs"] += 1
    effects, interactions, incomplete = organization_rows(paths)
    tables = {
        "path-strata.csv": paths,
        "paired-paths.csv": paired,
        "component-strata.csv": components,
        "learning-metrics.csv": metrics,
        "organization-effects.csv": effects,
        "organization-interactions.csv": interactions,
        "path-summary.csv": aggregate(
            paths,
            ("world", "step", "phase", "method", "split", "cohort"),
            (
                "accuracy",
                "wrong_actual_per_query",
                "nll",
                "bridge_accuracy",
                "bridge_valid_rate",
                "direct_accuracy",
                "two_step_gap",
            ),
        ),
        "paired-summary.csv": aggregate(
            paired,
            ("world", "step", "method", "split", "cohort"),
            ("unrestricted_accuracy", "isolated_accuracy", "isolated_minus_unrestricted"),
        ),
        "component-summary.csv": aggregate(
            components,
            ("world", "step", "phase", "split", "cohort"),
            (
                "both_prerequisites",
                "membership_accuracy",
                "membership_nll",
                "root_accuracy",
                "root_nll",
                "actual_accuracy",
                "actual_nll",
            ),
        ),
        "base-summary.csv": aggregate(
            metrics, ("world", "step", "phase"), ("base_accuracy", "base_nll", "mean_heldout")
        ),
        "organization-summary.csv": aggregate(
            effects,
            ("world", "step", "phase", "method", "split", "cohort"),
            (
                "matching_effect",
                "company_mean_benefit",
                "project_mean_benefit",
                "company_unmatched_benefit",
                "project_unmatched_benefit",
            ),
        ),
        "interaction-summary.csv": aggregate(
            interactions,
            ("world", "step", "method", "split", "cohort"),
            tuple(
                f"isolated_minus_unrestricted_{m}"
                for m in (
                    "matching_effect",
                    "company_mean_benefit",
                    "project_mean_benefit",
                    "company_unmatched_benefit",
                    "project_unmatched_benefit",
                )
            ),
        ),
    }
    # All sources are small predictions/configuration/schedules, never model weights.
    for path, expected in ledger.items():
        if digest(Path(path)) != expected:
            raise ValueError(f"Source changed during summary: {path}")
    complete = counts == {"complete_pairs": 12, "learning_pairs": 72, "two_step_pairs": 24}
    audit = {
        "complete": complete and not missing and not incomplete,
        **counts,
        "missing": missing,
        "incomplete_organization_blocks": incomplete,
        "model_weights_read": False,
        "edit_cases": 0,
        "direct_steps": list(CHECKPOINTS),
        "autonomous_two_step_steps": [15360],
        "intermediate_two_step_status": "Not produced by the frozen context trainer",
        "source_roots": {
            "isolated": str(isolated_root),
            "unrestricted": str(original_root),
            "unrestricted_two_step": str(two_step_root),
        },
        "script_sha256": digest(Path(__file__)),
        "helper_sha256": digest(Path(__file__).with_name("summarize_bios_shortcut_control.py")),
        "cohorts": "Fixed original people; all/ordinary/original_exception; no mastery filtering",
        "root_measure": "Person-weighted corresponding root; repeated people can share one root. "
        "True organization indexes separate fact scoring only, never the inference bridge.",
        "inference": "Original unrestricted attention; full value plus EOS; model-generated bridge",
        "uncertainty": "No query-level confidence intervals; two development worlds reused",
    }
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in tables.items():
        if rows:
            write_csv(output / name, rows)
        else:
            (output / name).write_text("")
    write_json(output / "sources.json", ledger)
    write_json(output / "audit.json", audit)
    write_report(output, audit, tables)
    if require_complete and not audit["complete"]:
        raise ValueError(f"Context matrix incomplete: {counts}, missing {len(missing)} runs")
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isolated-root", type=Path, default=MECHANISM / "p3-context")
    parser.add_argument(
        "--original-root", type=Path, default=ROOT / "results/bios-cross-scale-dev-v1/width-256"
    )
    parser.add_argument("--two-step-root", type=Path, default=MECHANISM / "p0/two-step/width-256")
    parser.add_argument("--output", type=Path, default=MECHANISM / "p3-context-summary")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            summarize(
                args.isolated_root,
                args.original_root,
                args.two_step_root,
                args.output,
                args.require_complete,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
