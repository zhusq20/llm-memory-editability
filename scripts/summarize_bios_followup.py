"""Recompute paired E93 outcomes and the fixed-state gradient comparisons."""

import argparse
import csv
import itertools
import json
from collections import defaultdict

import numpy as np
import torch

from llm_memory_editability import bios_context_gradient as gradient
from llm_memory_editability import bios_mechanism_edit as editing
from llm_memory_editability.bios_cross import CHAINS, edit_pair, make_cross_world
from llm_memory_editability.bios_data import array_hash, write_json
from llm_memory_editability.bios_followup import (
    ROOT,
    load_config,
    mixture_spec,
    production_hashes,
    verify_lock,
    verify_receipt,
)
from llm_memory_editability.bios_organization_train import state_hash

METRICS = (
    "D_probe_conflict_heldout_accuracy",
    "D_heldout_accuracy",
    "D_accuracy",
    "E_changed_root_accuracy",
    "E_changed_actual_accuracy",
    "U_full_rate",
    "U_unseen_rate",
    "U_unseen_strata_0_rate",
    "U_unseen_strata_1_rate",
    "fixed_person_new_actual_accuracy",
    "fixed_person_old_default_accuracy",
)


def flatten(record, prefix=""):
    result = {}
    for key, value in record.items():
        name = f"{prefix}_{key}" if prefix else key
        if isinstance(value, dict):
            result.update(flatten(value, name))
        else:
            result[name] = value
    return result


def write_csv(path, rows):
    if not rows:
        raise ValueError("Cannot publish empty result table")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def average(values):
    valid = [value for value in values if value is not None]
    return float(np.mean(valid)) if valid else None


def assert_metrics(recorded, expected):
    for key, value in expected.items():
        actual = recorded[key]
        if isinstance(value, dict):
            assert_metrics(actual, value)
        elif value is None:
            assert actual is None, key
        elif isinstance(value, float):
            assert np.isclose(value, actual, atol=1e-12, rtol=1e-12), key
        else:
            assert value == actual, key


def signature_metrics(world, chain, pair, arrays):
    ids = np.intersect1d(pair["conflict_D"], world.heldout_ids[chain])
    actual_ids = world.actual_ids[chain, world.person[ids]]
    labels = {
        "new_default": pair["exception"][ids],
        "new_actual": pair["exception"][actual_ids],
        "old_default": world.answers[ids],
    }
    hits = {
        key: (arrays["prediction"][ids] == values) & arrays["ended"][ids]
        for key, values in labels.items()
    }
    counts = np.stack(list(hits.values())).sum(0)
    result = {f"fixed_person_{key}_accuracy": float(value.mean()) for key, value in hits.items()}
    result.update(
        fixed_person_n=len(ids),
        fixed_person_ambiguous_hits=int((counts > 1).sum()),
        fixed_person_other_answers=int(((counts == 0) & arrays["ended"][ids]).sum()),
        fixed_person_termination_errors=int((~arrays["ended"][ids]).sum()),
    )
    return result


def score_edits(config, out):
    rows, sources, reused, new = [], {}, 0, 0
    for world_id, seed, condition in itertools.product(
        config["worlds"], config["seeds"], config["conditions"]
    ):
        name = f"world-{world_id}-seed-{seed}-{condition}"
        parent = ROOT / config["parent_root"] / name
        fresh = ROOT / config["output"] / "editing" / name
        if not (fresh / "complete.json").exists():
            raise ValueError("Incomplete parent: " + name)
        parent_config = json.loads((parent / "config.json").read_text())
        parent_hash = json.loads((parent / "learning-complete.json").read_text())["model_sha256"]
        world = make_cross_world(world_id)
        with np.load(parent / "predictions-15360.npz") as data:
            old_correct = data["correct"].copy()
        for chain, chain_name in enumerate(CHAINS):
            pair = edit_pair(world, chain)
            choices = editing.sampling_streams(world, chain, config["edit"]["edit_steps"])
            for kind, weight in itertools.product(("coherent", "exception"), config["root_mass"]):
                arm = kind if weight == "uniform" else f"{kind}-{weight}"
                spec = mixture_spec(world, chain, arm)
                historical = weight == "uniform" or (kind == "exception" and weight == "root500")
                if weight == "uniform":
                    source = parent / "edits" / f"{chain_name}-{kind}-mlp"
                elif historical:
                    source = (
                        ROOT
                        / config["balanced_root"]
                        / name
                        / "edits"
                        / f"{chain_name}-class-balanced-exception"
                    )
                else:
                    source = fresh / "edits" / f"{chain_name}-{arm}"
                complete = json.loads((source / "complete.json").read_text())
                if weight != "uniform":
                    contract = json.loads((source / "contract.json").read_text())
                    assert complete["contract_sha256"] == editing._contract_hash(contract)
                    assert contract["parent"]["model_sha256"] == parent_hash
                    assert contract["sources"] == (
                        editing.source_hashes() if historical else production_hashes()
                    )
                    for key in (
                        "edit_steps",
                        "edit_checkpoints",
                        "edit_lr",
                        "retention_kl",
                        "batch_size",
                    ):
                        assert contract["study"][key] == config["edit"][key], key
                    assert contract["old_correct_sha256"] == array_hash(old_correct)
                    for key, expected in spec.items():
                        assert contract["sets_sha256"][key] == array_hash(expected), key
                    assert contract["edit_sampling_sha256"] == array_hash(choices[0])
                    assert contract["replay_sampling_sha256"] == array_hash(choices[1])
                else:
                    for key in ("edit_steps", "edit_checkpoints", "edit_lr", "retention_kl"):
                        assert parent_config["study"][key] == config["edit"][key]
                with np.load(source / "sets.npz") as saved:
                    check = {
                        "old_correct": old_correct,
                        "edit_sampling": choices[0],
                        "replay_sampling": choices[1],
                    }
                    if weight == "uniform":
                        check.update(E=spec["S"], replay=spec["replay"])
                        check[kind] = spec["target"]
                    else:
                        check.update(spec)
                    for key, expected in check.items():
                        assert np.array_equal(saved[key], expected), (source, key)
                trajectories = json.loads((source / "trajectory.json").read_text())
                points = {point["step"]: point for point in trajectories}
                for step in config["edit"]["edit_checkpoints"]:
                    path = source / f"predictions-{step}.npz"
                    with np.load(path) as saved:
                        arrays = {key: saved[key] for key in ("prediction", "ended", "correct")}
                    assert np.array_equal(
                        arrays["correct"],
                        (arrays["prediction"] == spec["target"]) & arrays["ended"],
                    ), path
                    metrics = editing.arm_metrics(world, spec, arrays, old_correct)
                    if weight != "uniform":
                        assert_metrics(points[step], metrics)
                    rows.append(
                        {
                            "world": world_id,
                            "seed": seed,
                            "condition": condition,
                            "chain": chain_name,
                            "kind": kind,
                            "weight": weight,
                            "alpha": config["root_mass"][weight],
                            "step": step,
                            "reused": historical,
                            "source": str(source.relative_to(ROOT)),
                            **flatten(metrics),
                            **signature_metrics(world, chain, pair, arrays),
                        }
                    )
                    sources[str(path.relative_to(ROOT))] = editing.file_hash(path)
                sources[str((source / "sets.npz").relative_to(ROOT))] = editing.file_hash(
                    source / "sets.npz"
                )
                sources[str((source / "complete.json").relative_to(ROOT))] = editing.file_hash(
                    source / "complete.json"
                )
                reused += int(historical)
                new += int(not historical)
    assert (reused, new, len(rows)) == (72, 120, 768)
    write_csv(out / "editing-all-nodes.csv", rows)
    endpoint = [row for row in rows if row["step"] == 512]
    write_csv(out / "editing-endpoint.csv", endpoint)
    return rows, sources


def paired_effects(rows):
    key_fields = ("world", "seed", "condition", "chain", "kind", "step")
    reference = {tuple(r[k] for k in key_fields): r for r in rows if r["weight"] == "uniform"}
    paired = []
    for row in rows:
        if row["weight"] == "uniform":
            continue
        baseline = reference[tuple(row[k] for k in key_fields)]
        pair = {key: row[key] for key in (*key_fields, "weight", "alpha")}
        for metric in METRICS:
            a, b = row[metric], baseline[metric]
            pair[f"delta_{metric}"] = a - b if a is not None and b is not None else None
        paired.append(pair)
    return paired


def grouped(rows, fields, metrics):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in fields)].append(row)
    return [
        {
            **dict(zip(fields, key, strict=True)),
            "cases": len(values),
            **{metric: average([v[metric] for v in values]) for metric in metrics},
        }
        for key, values in sorted(groups.items())
    ]


def organization_effects(rows):
    cells = {}
    for row in rows:
        key = tuple(row[k] for k in ("world", "seed", "kind", "weight", "step"))
        cells.setdefault(key, {})[(row["condition"], row["chain"])] = row
    effects = []
    for key, values in sorted(cells.items()):
        if len(values) != 6:
            raise ValueError("Missing organization/chain cell")
        output = dict(zip(("world", "seed", "kind", "weight", "step"), key, strict=True))
        for metric in METRICS:
            cc, pc, pp, cp, nc, np_ = (
                values[pair][metric]
                for pair in (
                    ("company", "company"),
                    ("project", "company"),
                    ("project", "project"),
                    ("company", "project"),
                    ("neither", "company"),
                    ("neither", "project"),
                )
            )
            if None in (cc, pc, pp, cp, nc, np_):
                match = mismatch = None
            else:
                match = ((cc - pc) + (pp - cp)) / 2
                mismatch = ((pc - nc) + (cp - np_)) / 2
            output["matching_" + metric] = match
            output["mismatched_minus_neither_" + metric] = mismatch
        effects.append(output)
    return effects


def update_interactions(effects):
    """Within-block exception minus coherent matching effects, on identical people."""
    keys = ("world", "seed", "weight", "step")
    control = {tuple(r[k] for k in keys): r for r in effects if r["kind"] == "coherent"}
    result = []
    for row in effects:
        if row["kind"] != "exception":
            continue
        reference = control[tuple(row[k] for k in keys)]
        difference = {k: row[k] for k in keys}
        for metric in (
            "matching_D_probe_conflict_heldout_accuracy",
            "matching_D_heldout_accuracy",
        ):
            difference["exception_minus_coherent_" + metric] = row[metric] - reference[metric]
        result.append(difference)
    return result


def validate_new_weights(config):
    """Only new products: preserve frozen parameters and full optimizer endpoints."""
    checked, files = 0, {}
    for world, seed, condition in itertools.product(
        config["worlds"], config["seeds"], config["conditions"]
    ):
        name = f"world-{world}-seed-{seed}-{condition}"
        parent = ROOT / config["parent_root"] / name / "model-15360.pt"
        baseline = torch.load(parent, map_location="cpu", weights_only=False)["model"]
        directory = ROOT / config["output"] / "editing" / name / "edits"
        for chain, arm in itertools.product(CHAINS, config["edit"]["arms"]):
            source = directory / f"{chain}-{arm}"
            complete = json.loads((source / "complete.json").read_text())
            saved = torch.load(source / "model-final.pt", map_location="cpu", weights_only=False)
            assert saved["step"] == 512
            assert state_hash(saved["model"]) == complete["final_model_sha256"]
            for parameter, value in baseline.items():
                editable = any(parameter.startswith(f"blocks.{layer}.mlp.") for layer in (3, 4, 5))
                if not editable:
                    assert torch.equal(value, saved["model"][parameter]), (source, parameter)
            del saved
            resume = torch.load(source / "resume.pt", map_location="cpu", weights_only=False)
            assert resume["step"] == 512
            assert resume["contract_sha256"] == complete["contract_sha256"]
            assert state_hash(resume["model"]) == complete["final_model_sha256"]
            assert len(resume["optimizer"]["state"]) == 12
            assert all(int(state["step"]) == 512 for state in resume["optimizer"]["state"].values())
            assert len(resume["update_stats"]) == 512
            del resume
            for filename in ("model-final.pt", "resume.pt"):
                path = source / filename
                files[str(path.relative_to(ROOT))] = {
                    "bytes": path.stat().st_size,
                    "sha256": editing.file_hash(path),
                }
            checked += 1
        del baseline
    assert checked == 120
    return {"new_edit_weights_verified": checked, "files": files}


def score_gradients(config, out):
    rows, modules, sources, sham, predictor_errors = [], [], {}, [], []
    for world, seed in itertools.product(config["worlds"], config["seeds"]):
        name = f"world-{world}-seed-{seed}"
        fresh = ROOT / config["output"] / "gradient" / name
        parent = ROOT / config["gradient_parent"] / name
        complete = json.loads((fresh / "complete.json").read_text())
        assert complete["lock_sha256"] == verify_lock(config)
        if (fresh / "sham-check.json").exists():
            sham.append(max(json.loads((fresh / "sham-check.json").read_text()).values()))
        for step in config["gradient"]["states"]:
            actual, uniform = {}, {}
            for condition in config["conditions"]:
                old = parent / f"step-{step}-open-{condition}"
                new = fresh / f"step-{step}-uniform-{condition}"
                for path in (old, new):
                    verify_receipt(path)
                    sources[str((path / "receipt.json").relative_to(ROOT))] = editing.file_hash(
                        path / "receipt.json"
                    )
                actual[condition], uniform[condition] = (
                    gradient.arm_data(old),
                    gradient.arm_data(new),
                )
                for kind in ("answer", "eos"):
                    qkv = uniform[condition][kind]["blocks.0.attention.qkv.weight"]
                    assert not np.count_nonzero(qkv[: 2 * (qkv.shape[0] // 3)])
                for predictor, values in actual[condition]["predictions"].items():
                    error = gradient.vector_metrics(
                        values, uniform[condition]["predictions"][predictor]
                    )["relative_error"]
                    assert error is not None and error < 1e-6
                    predictor_errors.append(error)
            for first, second in gradient.CONTRASTS:
                for kind in ("answer", "eos", "combined"):
                    kinds = ("answer", "eos") if kind == "combined" else (kind,)
                    r = {
                        n: sum(actual[first][k][n] - actual[second][k][n] for k in kinds)
                        for n in actual[first]["answer"]
                    }
                    u = {
                        n: sum(uniform[first][k][n] - uniform[second][k][n] for k in kinds)
                        for n in uniform[first]["answer"]
                    }
                    meta = {
                        "world": world,
                        "seed": seed,
                        "step": step,
                        "first": first,
                        "second": second,
                        "kind": kind,
                    }
                    modules.append({**meta, "modules": gradient.module_differences(r, u)})
                    comparisons = {"actual_vs_uniform": (r[gradient.TARGET], u[gradient.TARGET])}
                    for predictor in ("position", "token", "shuffled"):
                        p = sum(
                            actual[first]["predictions"][f"{predictor}/{k}"]
                            - actual[second]["predictions"][f"{predictor}/{k}"]
                            for k in kinds
                        )
                        comparisons[f"uniform_vs_{predictor}"] = (u[gradient.TARGET], p)
                        if predictor == "position":
                            comparisons["actual_vs_position"] = (r[gradient.TARGET], p)
                    for comparison, (left, right) in comparisons.items():
                        rows.append(
                            {
                                **meta,
                                "comparison": comparison,
                                **gradient.vector_metrics(left, right),
                            }
                        )
    assert len(rows) == 4 * 3 * 3 * 3 * 5
    assert len(sham) == 1 and max(sham) <= config["gradient"]["sham_relative_tolerance"]
    write_csv(out / "gradient-comparisons.csv", rows)
    write_json(out / "gradient-module-differences.json", modules)
    primary = [r for r in rows if r["kind"] == "answer" and r["second"] == "neither"]
    summary = grouped(primary, ("step", "comparison"), ("cosine", "relative_error", "norm_ratio"))
    write_csv(out / "gradient-summary.csv", summary)
    return (
        summary,
        sources,
        {
            "sham_max_relative_error": max(sham),
            "predictor_max_relative_error": max(predictor_errors),
            "uniform_arms": 36,
        },
    )


def report(config, out, rows, gradients):
    endpoint = [r for r in rows if r["step"] == 512]
    overall = grouped(endpoint, ("kind", "weight", "alpha"), METRICS)
    write_csv(out / "editing-overall.csv", overall)
    blocks = grouped(rows, ("world", "seed", "kind", "weight", "alpha", "step"), METRICS)
    write_csv(out / "editing-blocks.csv", blocks)
    write_csv(out / "editing-paired.csv", paired_effects(rows))
    effects = organization_effects(rows)
    write_csv(out / "organization-effects.csv", effects)
    write_csv(out / "update-type-interactions.csv", update_interactions(effects))
    effect_metrics = [k for k in effects[0] if k.startswith(("matching_", "mismatched_"))]
    effect_summary = grouped(
        [r for r in effects if r["step"] == 512], ("kind", "weight"), effect_metrics
    )
    write_csv(out / "organization-summary.csv", effect_summary)
    lines = [
        "# v2.12 组织×编辑监督配比与第一层均匀注意力",
        "",
        "两个已见开发世界、两个初始化；世界内配对、等权平均，无确认性显著性检验。",
        "共192个编辑条件：复用72、新增120，均为E93、固定512步；另新增36个均匀注意力测量臂及1个sham。",
        "",
        "## 固定512步编辑终点",
        "",
        "同人D指原例外任务冲突留出人员；一致更新使用同一人员参照。U仅以编辑前正确且真值不变者为分母。",
        "",
        "| 更新 | 默认CE总权重 | 同人D | 全留出D | 默认E | 个人E | "
        "全池U损伤 | 未见U损伤 | 局部旧例外U损伤（未见） |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in sorted(overall, key=lambda r: (r["kind"], r["alpha"])):
        values = [
            r[m]
            for m in (
                "D_probe_conflict_heldout_accuracy",
                "D_heldout_accuracy",
                "E_changed_root_accuracy",
                "E_changed_actual_accuracy",
                "U_full_rate",
                "U_unseen_rate",
                "U_unseen_strata_0_rate",
            )
        ]
        lines.append(
            f"| {r['kind']} | {100 * r['alpha']:.3f}% | "
            + " | ".join(f"{100 * v:.3f}%" if v is not None else "不可估计" for v in values)
            + " |"
        )
    lines += [
        "",
        "## 组织×关系匹配效应",
        "",
        "单位：百分点；同人D与全部留出D分开。",
        "",
        "| 更新 | 权重 | 同人D匹配效应 | 全留出D匹配效应 | 不匹配关系同人D−中性 |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for r in sorted(effect_summary, key=lambda r: (r["kind"], config["root_mass"][r["weight"]])):
        values = [
            r[m]
            for m in (
                "matching_D_probe_conflict_heldout_accuracy",
                "matching_D_heldout_accuracy",
                "mismatched_minus_neither_D_probe_conflict_heldout_accuracy",
            )
        ]
        lines.append(
            f"| {r['kind']} | {r['weight']} | "
            + " | ".join(f"{100 * v:+.3f}" for v in values)
            + " |"
        )
    lines += [
        "",
        "## 真实梯度与第一层均匀注意力",
        "",
        "R为原真实梯度差，U为均匀第一层后的梯度差，P为原统计预测；均无拟合。下表为答案部分、两组织各减neither。",
        "",
        "| 状态 | 比较 | 余弦 | 相对误差 | 预测/参照范数比 |",
        "| ---: | --- | ---: | ---: | ---: |",
    ]
    for r in gradients:
        lines.append(
            f"| {r['step']} | {r['comparison']} | {r['cosine']:.6f} | "
            f"{r['relative_error']:.6f} | {r['norm_ratio']:.6f} |"
        )
    lines += [
        "",
        "完整CSV保留全部配对案例、0/32/128/512节点、U分母、回答候选重合、三组织×两查询关系，以及梯度EOS和combined对比。组织/查询/初始化均不是新增独立世界。",
        "",
    ]
    (out / "report.md").write_text("\n".join(lines))
    return overall, blocks, effect_summary


def plots(out, rows, gradients):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    overall = grouped([r for r in rows if r["step"] == 512], ("kind", "alpha"), METRICS)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.7), constrained_layout=True)
    for kind, color in (("coherent", "#2271b2"), ("exception", "#d55e00")):
        values = sorted([r for r in overall if r["kind"] == kind], key=lambda r: r["alpha"])
        x = [100 * r["alpha"] for r in values]
        for ax, metric in zip(
            axes,
            ("D_probe_conflict_heldout_accuracy", "D_heldout_accuracy", "U_unseen_rate"),
            strict=True,
        ):
            ax.plot(x, [100 * r[metric] for r in values], "o-", color=color, label=kind)
            ax.set_xlabel("Root-fact CE mass (%)")
            ax.grid(alpha=0.2)
    for ax, title in zip(
        axes,
        (
            "Same-person heldout propagation",
            "All heldout propagation",
            "Unseen known-knowledge damage",
        ),
        strict=True,
    ):
        ax.set_title(title)
        ax.set_ylabel("Percent")
    axes[0].legend()
    fig.savefig(out / "edit-mixture.png", dpi=180)
    fig.savefig(out / "edit-mixture.pdf")
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    trajectory = grouped(rows, ("kind", "weight", "alpha", "step"), METRICS)
    for index, kind in enumerate(("coherent", "exception")):
        for weight in ("uniform", "root125", "root250", "root500"):
            values = sorted(
                [r for r in trajectory if r["kind"] == kind and r["weight"] == weight],
                key=lambda r: r["step"],
            )
            for ax, metric in zip(
                axes[index], ("D_probe_conflict_heldout_accuracy", "U_unseen_rate"), strict=True
            ):
                ax.plot(
                    [r["step"] for r in values],
                    [100 * r[metric] for r in values],
                    "o-",
                    label=f"root mass {100 * values[0]['alpha']:.1f}%",
                )
                ax.set_xlabel("Edit steps")
                ax.set_ylabel("Percent")
                ax.grid(alpha=0.2)
        axes[index, 0].set_title(kind + ": same-person propagation")
        axes[index, 1].set_title(kind + ": unseen known-U damage")
    axes[0, 0].legend(fontsize=8)
    fig.savefig(out / "edit-trajectories.png", dpi=180)
    fig.savefig(out / "edit-trajectories.pdf")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True)
    for comparison, label, color in (
        ("actual_vs_uniform", "Actual vs uniform first layer", "#2271b2"),
        ("uniform_vs_position", "Uniform vs statistics", "#d55e00"),
        ("actual_vs_position", "Actual vs statistics", "#009e73"),
    ):
        values = sorted(
            [r for r in gradients if r["comparison"] == comparison], key=lambda r: r["step"]
        )
        for ax, metric in zip(axes, ("cosine", "relative_error"), strict=True):
            ax.plot(
                [r["step"] for r in values],
                [r[metric] for r in values],
                "o-",
                label=label,
                color=color,
            )
            ax.set_xlabel("Common trajectory step")
            ax.set_title(metric.replace("_", " "))
            ax.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    axes[1].axhline(1, color="gray", linestyle=":", linewidth=1)
    fig.savefig(out / "uniform-attention.png", dpi=180)
    fig.savefig(out / "uniform-attention.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plots", action="store_true")
    parser.add_argument("--verify-new-weights", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    config = load_config()
    lock_hash = verify_lock(config)
    out = ROOT / config["artifacts"]
    rows, edit_sources = score_edits(config, out)
    gradients, gradient_sources, controls = score_gradients(config, out)
    report(config, out, rows, gradients)
    if args.plots:
        plots(out, rows, gradients)
    if args.verify_new_weights:
        write_json(out / "new-weight-audit.json", validate_new_weights(config))
    write_json(out / "analysis-sources.json", {**edit_sources, **gradient_sources})
    audit = {
        "complete": True,
        "completed_at": gradient.stamp(),
        "lock_sha256": lock_hash,
        "new_edits": 120,
        "reused_edits": 72,
        "edit_nodes": len(rows),
        **controls,
        "analysis_source_sha256": editing.file_hash(__file__),
        "scope": (
            "Saved predictions independently rescored, paired sets/streams/runtime "
            "and gradient receipts checked; no re-forward of all historical edits"
        ),
    }
    write_json(out / "audit.json", audit)
    print(json.dumps(audit))


if __name__ == "__main__":
    main()
