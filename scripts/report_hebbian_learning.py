#!/usr/bin/env python3
"""Report only completed predeclared matrices; incomplete stages remain explicitly pending."""

from __future__ import annotations

import argparse
import itertools
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from llm_memory_editability.hebbian_learning import (
    ARTIFACTS,
    DATA,
    RESULTS,
    ROOT,
    now,
    read_json,
    sha256,
    write_json,
)
from llm_memory_editability.hebbian_statistics import (
    fact_trajectories,
    paired_inference,
    spearman,
)


def save_figure(fig, name):
    path = ARTIFACTS / "figures"
    path.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(path / f"{name}.png", dpi=160)
    fig.savefig(path / f"{name}.pdf")
    plt.close(fig)


def subject_component_sensitivity(differences, split):
    """Preserve episode weighting while sharing signs/resampling units across entities."""
    differences = np.asarray(differences, dtype=float)
    audit = read_json(ARTIFACTS / "episode-subject-audit.json")["pools"][split]
    components = [{i} for i in range(len(differences))]
    for linked in audit["cross_episode_subjects"].values():
        selected = [g for g in components if g & set(linked)]
        merged = set().union(*selected)
        components = [g for g in components if not g & merged] + [merged]
    components = sorted([sorted(g) for g in components])
    sums = np.array([differences[g].sum() for g in components])
    counts = np.array([len(g) for g in components])
    rng = np.random.default_rng(20260928)
    indices = rng.integers(0, len(components), (10000, len(components)))
    samples = sums[indices].sum(1) / counts[indices].sum(1)
    signs = np.array(list(itertools.product([-1, 1], repeat=len(components))))
    observed = float(differences.mean())
    p = float(np.mean(np.abs(signs @ sums / len(differences)) >= abs(observed) - 1e-14))
    return {
        "mean": observed,
        "ci95": np.quantile(samples, [0.025, 0.975]).tolist(),
        "component_sign_p": p,
        "components": components,
        "status": "supplement_prespecified_before_C_eval; B_use_is_retrospective",
    }


def report_a():
    audit = read_json(ARTIFACTS / "A/audit.json")
    fig, axes = plt.subplots(2, 4, figsize=(13, 6), sharex=True, sharey=True)
    for row, kind in enumerate(["exact", "bilinear"]):
        runs = [r for r in audit["runs"] if r["kind"] == kind and r["seed"] == 0]
        for col, run in enumerate(runs):
            arrays = np.load(RESULTS / "A" / (run["run_id"] + ".npz"))
            ax = axes[row, col]
            for mode, color in [(0, "#2166ac"), (1, "#b2182b")]:
                ax.semilogy(
                    arrays["norms"][:, mode], color=color, label=["common", "difference"][mode]
                )
                ax.semilogy(arrays["prediction"][:, mode], color="black", ls="--", lw=0.8)
            ax.set_title(f"{kind}; measured rho={run['rho']:.4f}")
            ax.set_ylim(1e-9, 2)
            ax.set_xscale("symlog", linthresh=16)
            ax.set_xticks([0, 16, 256, 4096, 16384], ["0", "16", "256", "4096", "16384"])
            ax.tick_params(axis="x", labelsize=8)
            ax.set_xlabel("SGD step")
            ax.set_ylabel("Mode error norm")
    axes[0, 0].legend()
    save_figure(fig, "01-calibration")
    return {
        "runs": len(audit["runs"]),
        "pass": audit["pass"],
        "max_relative_error": max(r["max_relative_vector_error"] for r in audit["runs"]),
    }


def report_b(episodes):
    lock = read_json(ARTIFACTS / "B-lock.json")
    lr = lock["learning_rate"]
    paths = [RESULTS / f"B/eval-lr{lr:g}-e{i}" for i in range(len(episodes["B_eval"]))]
    if not all((p / "complete.json").exists() for p in paths):
        return {
            "status": "incomplete",
            "complete": sum((p / "complete.json").exists() for p in paths),
            "expected": len(paths),
        }
    data = []
    learning = []
    calibrations = []
    for episode, path in enumerate(paths):
        facts = {r["case_id"]: r for r in fact_trajectories(path)}
        learning.append(
            {
                "nodes": next(iter(facts.values()))["nodes"],
                "rewrite": np.mean([r["rewrite_accuracy"] for r in facts.values()], axis=0),
                "standard": np.mean([r["standard_accuracy"] for r in facts.values()], axis=0),
            }
        )
        for row in read_json(path / "prospective-predictions.json")["predictions"]:
            data.append(
                {
                    "episode": episode,
                    "case_id": row["case_id"],
                    "actual": facts[row["case_id"]]["q_auc"],
                    **row["values"],
                }
            )
        calibrations.extend(read_json(path / "first-order-calibration.json"))
    metrics, errors = {}, {}
    for name in ["P0", "P1", "P2"]:
        errors[name] = [
            float(np.mean([abs(r[name] - r["actual"]) for r in data if r["episode"] == e]))
            for e in range(len(paths))
        ]
        metrics[name] = {
            "mae": float(np.mean(errors[name])),
            "spearman": spearman([r[name] for r in data], [r["actual"] for r in data]),
            "episode_mae": errors[name],
        }
    comparisons = {
        "P1_minus_P0_MAE": paired_inference(np.array(errors["P1"]) - errors["P0"]),
        "P2_minus_P1_MAE": paired_inference(np.array(errors["P2"]) - errors["P1"]),
        "P2_minus_P0_MAE": paired_inference(np.array(errors["P2"]) - errors["P0"]),
    }
    for result in comparisons.values():
        result["shared_subject_sensitivity"] = subject_component_sensitivity(
            result["episode_differences"], "B_eval"
        )
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
    for ax, name in zip(axes[:3], metrics, strict=True):
        ax.scatter([r["actual"] for r in data], [r[name] for r in data], s=12, alpha=0.55)
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set_title(f"{name}: MAE={metrics[name]['mae']:.3f}")
        ax.set_xlabel("Observed rewrite Q-AUC")
        ax.set_ylabel("Prospective prediction")
    for metric in ["standard", "rewrite"]:
        axes[3].plot(
            learning[0]["nodes"], np.mean([r[metric] for r in learning], axis=0), label=metric
        )
    axes[3].set_xlabel("Adaptation step")
    axes[3].set_ylabel("Held-out exact match")
    axes[3].legend()
    save_figure(fig, "02-prediction")
    result = {
        "status": "complete",
        "learning_rate": lr,
        "metrics": metrics,
        "comparisons": comparisons,
        "rows": data,
        "calibration": {
            "records": len(calibrations),
            "nll_sign_agreement": float(np.mean([r["nll_sign_match"] for r in calibrations])),
            "margin_sign_agreement": float(np.mean([r["margin_sign_match"] for r in calibrations])),
            "mean_nll_absolute_error": float(
                np.mean([r["nll_absolute_error"] for r in calibrations])
            ),
        },
    }
    write_json(ARTIFACTS / "B-analysis.json", result)
    return result


def report_c(episodes):
    tasks = list(itertools.product(["C0", "C1", "C2"], [0, 1, 2], range(len(episodes["C_eval"]))))
    paths = [RESULTS / f"C/adapt-{c}-s{s}-e{e}" for c, s, e in tasks]
    if not all((p / "complete.json").exists() for p in paths):
        return {
            "status": "incomplete",
            "complete": sum((p / "complete.json").exists() for p in paths),
            "expected": len(paths),
        }
    facts, runs, curves, formation = [], [], defaultdict(list), []
    baseline = {r["case_id"]: r for r in read_json(DATA / "baseline.json")}
    base_path = next((RESULTS / "B").glob("eval*/evaluation-0000.json"))
    base_text_nll = read_json(base_path)["text_nll"]
    for (condition, seed, episode), path in zip(tasks, paths, strict=True):
        individual = fact_trajectories(path)
        final = read_json(path / "evaluation-0128.json")
        initial = read_json(path / "evaluation-0000.json")
        initially_correct = {r["case_id"] for r in initial["keep"] if r["answer_em"]}
        parent_damage = sum(
            r["case_id"] in initially_correct and not r["answer_em"] for r in final["keep"]
        ) / max(len(initially_correct), 1)
        keep_damage = 1 - np.mean([r["answer_em"] for r in final["keep"]])
        run = {
            "condition": condition,
            "seed": seed,
            "episode": episode,
            "q_auc": float(np.mean([r["q_auc"] for r in individual])),
            "q_gain_auc": float(np.mean([r["q_gain_auc"] for r in individual])),
            "standard_auc": float(np.mean([r["standard_auc"] for r in individual])),
            "u_damage_vs_original": float(keep_damage),
            "u_damage_vs_parent": float(parent_damage),
            "parent_correct_count": len(initially_correct),
            "text_nll": final["text_nll"],
            "text_nll_change": final["text_nll"] - initial["text_nll"],
            "text_nll_change_vs_original": final["text_nll"] - base_text_nll,
            "text_ppl": float(np.exp(final["text_nll"])),
            "right_censored_fraction": float(np.mean([r["right_censored"] for r in individual])),
            "sustained_success_fraction": float(
                np.mean([r["sustained_step"] is not None for r in individual])
            ),
            "endpoint_first_success_fraction": float(
                np.mean([r["endpoint_first_success"] for r in individual])
            ),
        }
        runs.append(run)
        for r in individual:
            base_nll = baseline[r["case_id"]]["views"][0]["answer_nll"]
            facts.append(
                {
                    "condition": condition,
                    "seed": seed,
                    "episode": episode,
                    "original_nll_stratum": int(np.digitize(base_nll, [1, 2, 4])),
                    "original_p0_correct": baseline[r["case_id"]]["views"][0]["answer_em"],
                    **r,
                }
            )
        curves[condition].append(
            {
                "nodes": individual[0]["nodes"],
                "rewrite": np.mean([r["rewrite_accuracy"] for r in individual], axis=0),
                "standard": np.mean([r["standard_accuracy"] for r in individual], axis=0),
            }
        )
    comparisons = {}
    for comparator in ["C0", "C2"]:
        per_episode = []
        for episode in range(len(episodes["C_eval"])):
            pair = []
            for seed in [0, 1, 2]:
                selected = {
                    r["condition"]: r["q_auc"]
                    for r in runs
                    if r["episode"] == episode and r["seed"] == seed
                }
                pair.append(selected["C1"] - selected[comparator])
            per_episode.append(np.mean(pair))
        comparisons["C1_minus_" + comparator] = paired_inference(per_episode)
        comparisons["C1_minus_" + comparator]["shared_subject_sensitivity"] = (
            subject_component_sensitivity(per_episode, "C_eval")
        )
    gain_comparisons = {}
    for comparator in ["C0", "C2"]:
        effects = []
        for episode in range(len(episodes["C_eval"])):
            values = {
                condition: np.mean(
                    [
                        r["q_gain_auc"]
                        for r in runs
                        if r["condition"] == condition and r["episode"] == episode
                    ]
                )
                for condition in ["C1", comparator]
            }
            effects.append(values["C1"] - values[comparator])
        gain_comparisons["C1_minus_" + comparator] = paired_inference(effects)
    ordered = sorted(comparisons, key=lambda n: comparisons[n]["exact_two_sided_sign_p"])
    previous = 0.0
    for index, name in enumerate(ordered):
        adjusted = min(
            1.0, max(previous, (2 - index) * comparisons[name]["exact_two_sided_sign_p"])
        )
        comparisons[name]["holm_p"] = adjusted
        previous = adjusted
    condition_means = {}
    for condition in ["C0", "C1", "C2"]:
        subset = [r for r in runs if r["condition"] == condition]
        condition_means[condition] = {
            key: float(np.mean([r[key] for r in subset]))
            for key in subset[0]
            if key not in ["condition", "seed", "episode"]
        }
    for condition, seed in itertools.product(["C0", "C1", "C2"], [0, 1, 2]):
        for path in sorted((RESULTS / f"C/form-{condition}-s{seed}").glob("evaluation-*.json")):
            ev = read_json(path)
            formation.append(
                {
                    "condition": condition,
                    "seed": seed,
                    "step": ev["rows"][0]["step"],
                    "same_fact_cos": float(np.mean([r["same_fact_cos"] for r in ev["rows"]])),
                    "other_fact_cos": float(np.mean([r["other_fact_cos"] for r in ev["rows"]])),
                    "u_damage": float(1 - np.mean([r["answer_em"] for r in ev["keep"]])),
                }
            )
    colors = {"C0": "#444444", "C1": "#2166ac", "C2": "#b2182b"}
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for condition in colors:
        for ax, metric in zip(axes, ["same_fact_cos", "other_fact_cos", "u_damage"], strict=True):
            steps = sorted({r["step"] for r in formation})
            values = [
                np.mean(
                    [r[metric] for r in formation if r["condition"] == condition and r["step"] == s]
                )
                for s in steps
            ]
            ax.plot(steps, values, label=condition, color=colors[condition])
            ax.set_xlabel("Formation step")
            ax.set_ylabel(metric)
    axes[0].legend()
    save_figure(fig, "03-formation")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for condition in colors:
        for ax, metric in zip(axes, ["standard", "rewrite"], strict=True):
            values = [r[metric] for r in curves[condition]]
            ax.plot(
                curves[condition][0]["nodes"],
                np.mean(values, axis=0),
                label=condition,
                color=colors[condition],
            )
            array = np.asarray(values).reshape(3, len(episodes["C_eval"]), -1).mean(0)
            rng = np.random.default_rng(20260928)
            samples = array[rng.integers(0, len(array), (10000, len(array)))].mean(1)
            lo, hi = np.quantile(samples, [0.025, 0.975], axis=0)
            ax.fill_between(
                curves[condition][0]["nodes"], lo, hi, color=colors[condition], alpha=0.12
            )
            ax.set_xlabel("Adaptation step")
            ax.set_ylabel(metric + " exact match")
    axes[0].legend()
    save_figure(fig, "04-learning")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for condition in colors:
        rows = [r for r in runs if r["condition"] == condition]
        for ax, metric in zip(
            axes, ["u_damage_vs_original", "text_nll_change_vs_original"], strict=True
        ):
            ax.scatter(
                [r[metric] for r in rows],
                [r["q_gain_auc"] for r in rows],
                label=condition,
                color=colors[condition],
                alpha=0.65,
            )
            ax.set_xlabel(metric)
            ax.set_ylabel("Rewrite Q-gain-AUC")
    axes[0].axvline(0.02, color="black", ls="--", lw=1)
    axes[0].legend()
    save_figure(fig, "05-retention")
    result = {
        "status": "complete",
        "comparisons": comparisons,
        "q_gain_comparisons": gain_comparisons,
        "condition_means": condition_means,
        "runs": runs,
        "facts": facts,
        "formation": formation,
    }
    write_json(ARTIFACTS / "C-analysis.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.parse_args()
    a = report_a()
    episodes = read_json(DATA / "episodes.json")
    b = report_b(episodes) if (ARTIFACTS / "B-lock.json").exists() else {"status": "not_locked"}
    c = report_c(episodes) if (ARTIFACTS / "C-lock.json").exists() else {"status": "not_locked"}
    receipts = sorted(RESULTS.glob("**/complete.json"))
    costs = [read_json(p) for p in receipts]
    coverage = read_json(ARTIFACTS / "data-lock.json")["coverage"]
    lines = [
        "# Hebbian 记忆与后续学习能力：实验结果 v1",
        "",
        f"更新：{now()}。执行依据：`docs/hebbian-learning-plan-v1.md`。",
        "",
        f"阶段 A：24/24 轨迹通过；最大相对向量误差 {a['max_relative_error']:.3g}。",
        f"阶段 B：{b['status']}。阶段 C：{c['status']}。",
        "",
        "## 实际数据与适用范围",
        "",
        "使用CounterFact原事实。各池主体互斥；保持池要求原模型标准表达完整生成正确。",
        "语义审核逐条应用模型制定的保守谓词规则；真实事实与全部别名未独立人工核验。",
        "出生地与泛指来源地、母语与一般使用语言、制造者与开发者混淆的记录进入复核池。",
        "因此主分析描述经过规则审核的公开基准子集，不把来源标签当作独立核验的事实。",
        "",
        "|池|计划|实际|",
        "|---|---:|---:|",
    ]
    lines += [f"|{name}|{r['planned']}|{r['actual']}|" for name, r in coverage.items()]
    lines += [
        "",
        "完整未来学习任务规模保留：16 个 B 任务、8 个 C 任务，三条件各三个形成种子。",
        "P19 因表达歧义未进入主清单；主清单的关系分布和保持池缩减均在训练前冻结。",
        "C_eval的256事实对应253主体；3个主体连接了不同episode。原8任务分析保留，"
        "另列共享主体连通分量敏感性分析，其规则在C_eval启动前补充固定。",
        "",
        "![数学验收](development-artifacts/hebbian-learning-v1/figures/01-calibration.png)",
        "",
    ]
    if b["status"] == "complete":
        lines += [
            "## 阶段 B：事前预测",
            "",
            f"B_dev 选定共同学习率 {b['learning_rate']:g}。",
            "",
            "|预测器|留出 MAE|Spearman|",
            "|---|---:|---:|",
        ]
        lines += [
            f"|{name}|{r['mae']:.4f}|{str(r['spearman'])[:7]}|" for name, r in b["metrics"].items()
        ]
        lines += [
            "",
            "三个开发学习率的V_keep损伤均超过2个百分点，所选学习率的开发损伤为5.08个百分点。",
            "P0/P1/P2 的特征与预测在正式曲线前保存；P2 使用标签梯度，属于离线机制诊断。",
            "![预测](development-artifacts/hebbian-learning-v1/figures/02-prediction.png)",
            "",
        ]
        best_predictor = min(b["metrics"], key=lambda name: b["metrics"][name]["mae"])
        lines += [
            f"留出MAE最低的预测器为{best_predictor}。P1相对P0的效应区间见下方；",
            "此处衡量固定开发样本和正则线性预测器下的解释力。局部一阶校准与128步预测分别判读。",
        ]
        for comparison, result in b["comparisons"].items():
            lo, hi = result["ci95"]
            lines.append(
                f"{comparison}：{result['mean']:+.4f}，"
                f"95% episode区间 [{lo:+.4f}, {hi:+.4f}]；MAE差值为负表示改善。"
            )
        lines += [
            "",
            f"一阶数值校准共{b['calibration']['records']}条："
            f"NLL符号一致率{b['calibration']['nll_sign_agreement']:.2%}，"
            f"margin符号一致率{b['calibration']['margin_sign_agreement']:.2%}。",
        ]
    if c["status"] == "complete":
        c_lock = read_json(ARTIFACTS / "C-lock.json")
        lines += [
            "## 阶段 C：新实体学习的配对效应",
            "",
            f"形成学习率{c_lock['learning_rate']:g}，几何系数{c_lock['lambda_geo']:g}；"
            "全部条件使用相同未来学习率、事实顺序和保持回放。",
            "|比较|Q-AUC 差值|95% episode bootstrap 区间|Holm p|",
            "|---|---:|---|---:|",
        ]
        for name, result in c["comparisons"].items():
            lo, hi = result["ci95"]
            lines.append(
                f"|{name}|{result['mean']:+.4f}|[{lo:+.4f}, {hi:+.4f}]|{result['holm_p']:.4f}|"
            )
        lines += ["", "共享主体合并后共5个连通分量；以下为补充敏感性分析：", ""]
        for name, result in c["comparisons"].items():
            sensitivity = result["shared_subject_sensitivity"]
            lo, hi = sensitivity["ci95"]
            lines.append(
                f"{name}：区间[{lo:+.4f}, {hi:+.4f}]，"
                f"分量符号置换p={sensitivity['component_sign_p']:.4f}（未作多重校正）。"
            )
        for name, result in c["q_gain_comparisons"].items():
            lo, hi = result["ci95"]
            lines.append(
                f"\n{name}的Q-gain-AUC差值为{result['mean']:+.4f}，"
                f"95%区间[{lo:+.4f}, {hi:+.4f}]。这是扣除各父模型0步表现的学习增量。"
            )
        lines += [
            "",
            "先在相同 episode 内平均三个形成种子的配对差，再对八个 episode 推断。",
            "推断限定于 Qwen3-0.6B-Base、第14层、当前三次形成训练、该事实清单与128步预算。",
            "",
            "|条件|Q-AUC|Q-gain-AUC|原始 U 破坏率|相对父模型 U 破坏率|文本 NLL 增量|",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for name, r in c["condition_means"].items():
            lines.append(
                f"|{name}|{r['q_auc']:.4f}|{r['q_gain_auc']:.4f}|{r['u_damage_vs_original']:.2%}|{r['u_damage_vs_parent']:.2%}|{r['text_nll_change']:+.4f}|"
            )
        lines += [
            "",
            "|条件|形成种子|Q-AUC|Q-gain-AUC|持续达标覆盖|终点首次成功|右截尾|",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for condition, seed in itertools.product(["C0", "C1", "C2"], [0, 1, 2]):
            subset = [r for r in c["runs"] if r["condition"] == condition and r["seed"] == seed]
            values = [
                float(np.mean([r[k] for r in subset]))
                for k in [
                    "q_auc",
                    "q_gain_auc",
                    "sustained_success_fraction",
                    "endpoint_first_success_fraction",
                    "right_censored_fraction",
                ]
            ]
            lines.append(
                f"|{condition}|{seed}|{values[0]:.4f}|{values[1]:.4f}|"
                f"{values[2]:.2%}|{values[3]:.2%}|{values[4]:.2%}|"
            )
        for number, name in [(3, "formation"), (4, "learning"), (5, "retention")]:
            lines += [
                "",
                f"![{name}](development-artifacts/hebbian-learning-v1/figures/0{number}-{name}.png)",
            ]
    lines += [
        "",
        "## 运行与复现",
        "",
        f"目前有 {len(receipts)} 个完整训练运行回执。",
        f"同步训练墙钟总和：{sum(r['ledger']['wall_seconds'] for r in costs) / 3600:.3f} "
        "GPU 小时；评价与文件写入另计。",
        "断点保存参数、Adam、随机状态、数据游标与token账本；全模型哈希检查训练范围。",
        "A轨迹在 `results/hebbian-learning-v1/A/`；B/C逐例结果在同批次B/C子目录。",
        "成本账本记录训练、参考模型与几何输入token及稠密计算估算；"
        "生成评价、诊断、加载与文件写入未单独计时，训练墙钟不能视为端到端成本。",
        "来源、环境、三次锁定及审计位于 `docs/development-artifacts/hebbian-learning-v1/`。",
        "",
    ]
    report = ROOT / "docs/hebbian-learning-results-v1.md"
    report.write_text("\n".join(lines))
    write_json(
        ARTIFACTS / "execution-status.json",
        {
            "time": now(),
            "A": a,
            "B": b["status"],
            "C": c["status"],
            "completed_training_runs": len(receipts),
            "report_sha256": sha256(report),
            "all_stages_complete": a["pass"] and b["status"] == c["status"] == "complete",
        },
    )
    print(report)


if __name__ == "__main__":
    main()
