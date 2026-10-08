"""Export saved Loop gradient geometry and actual updates without running models.

Missing frozen nodes are reported as partial. Local training-record gradients
and optimizer updates on mixed training batches remain separate observations;
neither is used to predict performance or assert a cause of generalization.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

ROLES = ("old_atoms", "new_atoms", "old_compositions")
IDENTITY_FIELDS = (
    "run",
    "arm",
    "seed",
    "data_sha256",
    "hidden_size",
    "base_unique_blocks",
    "logical_repeats",
    "executed_blocks",
    "step",
    "stage",
)
GROUP_FIELDS = (
    *IDENTITY_FIELDS,
    "role",
    "component",
    "base_block",
    "occurrences",
    "actually_shared",
    "record_count",
    "encoded_records_sha256",
    "reference_loss",
    "summed_gradient_norm",
    "sum_individual_squared_norms",
    "twice_pairwise_inner_product_sum",
    "squared_norm_ratio",
    "shared_gradient_maximum_absolute_error",
    "full_gradient_reconstruction_maximum_absolute_error",
    "forward_maximum_absolute_error",
)
OCCURRENCE_FIELDS = (
    *IDENTITY_FIELDS,
    "role",
    "component",
    "base_block",
    "executed_block",
    "loop_iteration",
    "gradient_norm",
)
PAIR_FIELDS = (
    *IDENTITY_FIELDS,
    "role",
    "component",
    "base_block",
    "first_executed_block",
    "second_executed_block",
    "first_loop_iteration",
    "second_loop_iteration",
    "first_gradient_norm",
    "second_gradient_norm",
    "dot_product",
    "cosine",
    "actually_shared",
)
UPDATE_FIELDS = (
    *IDENTITY_FIELDS,
    "role",
    "component",
    "unique_module_name",
    "stored_block",
    "base_block",
    "applies_to_executed_blocks",
    "actual_update_l2",
    "learning_rate",
    "gradient_norm_before_clipping",
)


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def identity(spec, step):
    return {
        "run": spec["name"],
        "arm": spec["arm"],
        "seed": spec["initialization"],
        "data_sha256": spec["data_sha256"],
        "hidden_size": spec["model"]["hidden_size"],
        "base_unique_blocks": spec["base_unique_blocks"],
        "logical_repeats": spec["repeats"],
        "executed_blocks": spec["base_unique_blocks"]
        * (1 if spec["arm"] == "shallow" else spec["repeats"]),
        "step": step,
        "stage": "initial" if step == 0 else "A" if step <= spec["stage_a_steps"] else "B",
    }


def stats(values):
    values = [value for value in values if value is not None]
    return (
        {"n": len(values), "mean": mean(values), "min": min(values), "max": max(values)}
        if values
        else {
            "n": 0,
            "mean": None,
            "min": None,
            "max": None,
        }
    )


def summarize(config, root):
    tables = {
        "gradient-groups": [],
        "occurrence-gradients": [],
        "gradient-pairs": [],
        "actual-updates": [],
    }
    runs, hashes, source_files = [], defaultdict(set), []
    for spec in config["specs"]:
        run = root / "runs" / spec["name"]
        metadata = read(run / "run.json")
        if metadata and metadata["spec"] != spec:
            raise ValueError(f"Run spec differs from supplied config: {spec['name']}")
        expected = sorted(set(spec.get("diagnostic_nodes", [])))
        end = spec["stage_a_steps"] + spec["stage_b_steps"]
        expected_updates = sorted(
            {n + offset for n in expected for offset in (0, 1) if 0 < n + offset <= end}
        )
        found, found_updates, missing_roles = [], [], {}
        for path in sorted(run.glob("diagnostics-*.json")):
            source_files.append(
                {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            )
            node = read(path)
            step = node["step"]
            if step in found or int(path.stem.rsplit("-", 1)[1]) != step:
                raise ValueError(f"Duplicate or inconsistent diagnostic step: {path}")
            found.append(step)
            missing = sorted(set(ROLES) - node["roles"].keys())
            if missing:
                missing_roles[str(step)] = missing
            for role, value in node["roles"].items():
                if value["arm"] != spec["arm"]:
                    raise ValueError(f"Diagnostic arm differs from run: {path}")
                hashes[(spec["data_sha256"], role)].add(value["encoded_records_sha256"])
                common = {**identity(spec, step), "role": role}
                reconstruction = value["gradient_reconstruction"]["maximum_absolute_error"]
                for group in value["groups"]:
                    local = {
                        **common,
                        "component": group["component"],
                        "base_block": group["base_block"],
                    }
                    indices, iterations = group["executed_block_indices"], group["loop_iterations"]
                    norms = group["occurrence_gradient_norms"]
                    if not len(indices) == len(iterations) == len(norms):
                        raise ValueError(f"Inconsistent occurrence dimensions: {path}")
                    tables["gradient-groups"].append(
                        {
                            **local,
                            "occurrences": len(indices),
                            "actually_shared": group["actually_shared_across_occurrences"],
                            "record_count": value["record_count"],
                            "encoded_records_sha256": value["encoded_records_sha256"],
                            "reference_loss": value["reference_loss"],
                            **{
                                key: group[key]
                                for key in (
                                    "summed_gradient_norm",
                                    "sum_individual_squared_norms",
                                    "twice_pairwise_inner_product_sum",
                                    "squared_norm_ratio",
                                )
                            },
                            "shared_gradient_maximum_absolute_error": group[
                                "shared_gradient_maximum_absolute_error"
                            ],
                            "full_gradient_reconstruction_maximum_absolute_error": reconstruction,
                            "forward_maximum_absolute_error": value[
                                "forward_maximum_absolute_error"
                            ],
                        }
                    )
                    for i, block in enumerate(indices):
                        tables["occurrence-gradients"].append(
                            {
                                **local,
                                "executed_block": block,
                                "loop_iteration": iterations[i],
                                "gradient_norm": norms[i],
                            }
                        )
                        for j in range(i + 1, len(indices)):
                            tables["gradient-pairs"].append(
                                {
                                    **local,
                                    "first_executed_block": block,
                                    "second_executed_block": indices[j],
                                    "first_loop_iteration": iterations[i],
                                    "second_loop_iteration": iterations[j],
                                    "first_gradient_norm": norms[i],
                                    "second_gradient_norm": norms[j],
                                    "dot_product": group["gradient_dot_products"][i][j],
                                    "cosine": group["gradient_cosines"][i][j],
                                    "actually_shared": group["actually_shared_across_occurrences"],
                                }
                            )
        for path in sorted(run.glob("actual-update-*.json")):
            source_files.append(
                {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            )
            value = read(path)
            step = value["step"]
            if step in found_updates or int(path.stem.rsplit("-", 1)[1]) != step:
                raise ValueError(f"Duplicate or inconsistent actual-update step: {path}")
            found_updates.append(step)
            for name, norm in value["unique_block_update_l2"].items():
                parts = name.split(".")
                if len(parts) != 4 or parts[:2] != ["transformer", "h"]:
                    raise ValueError(f"Unexpected unique module name: {name}")
                block, module = int(parts[2]), parts[3]
                shared = spec["arm"] == "shared" or (
                    spec["arm"] == "mlp_shared" and module == "mlp"
                )
                occurrences = (
                    [block + r * spec["base_unique_blocks"] for r in range(spec["repeats"])]
                    if shared
                    else [block]
                )
                tables["actual-updates"].append(
                    {
                        **identity(spec, step),
                        "role": "training_mixture",
                        "component": "attention" if module == "attn" else module,
                        "unique_module_name": name,
                        "stored_block": block,
                        "base_block": block % spec["base_unique_blocks"],
                        "applies_to_executed_blocks": json.dumps(occurrences),
                        "actual_update_l2": norm,
                        "learning_rate": value["learning_rate"],
                        "gradient_norm_before_clipping": value["gradient_norm_before_clipping"],
                    }
                )
        missing_nodes = sorted(set(expected) - set(found))
        missing_updates = sorted(set(expected_updates) - set(found_updates))
        runs.append(
            {
                **identity(spec, 0),
                "state": "partial"
                if missing_nodes or missing_updates or missing_roles
                else "complete",
                "expected_diagnostic_nodes": expected,
                "observed_diagnostic_nodes": found,
                "missing_diagnostic_nodes": missing_nodes,
                "missing_roles": missing_roles,
                "expected_actual_update_nodes": expected_updates,
                "observed_actual_update_nodes": found_updates,
                "missing_actual_update_nodes": missing_updates,
                "independent_completion_marker_present": (run / "complete.json").exists(),
            }
        )
    aggregates = defaultdict(lambda: {"cosines": [], "norm_ratios": [], "summed_norms": []})
    keys = (
        "role",
        "arm",
        "hidden_size",
        "base_unique_blocks",
        "logical_repeats",
        "step",
        "component",
    )
    for row in tables["gradient-pairs"]:
        aggregates[tuple(row[key] for key in keys)]["cosines"].append(row["cosine"])
    for row in tables["gradient-groups"]:
        bucket = aggregates[tuple(row[key] for key in keys)]
        bucket["norm_ratios"].append(row["squared_norm_ratio"])
        bucket["summed_norms"].append(row["summed_gradient_norm"])
    update_keys = (
        "run",
        "arm",
        "seed",
        "hidden_size",
        "base_unique_blocks",
        "logical_repeats",
        "step",
        "component",
    )
    update_squares = defaultdict(float)
    for row in tables["actual-updates"]:
        update_squares[tuple(row[key] for key in update_keys)] += row["actual_update_l2"] ** 2
    summary = {
        "batch": config.get("batch", root.name),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "state": "complete"
        if runs and all(row["state"] == "complete" for row in runs)
        else "partial",
        "runs": runs,
        "row_counts": {key: len(value) for key, value in tables.items()},
        "fixed_batch_hashes": [
            {
                "data_sha256": data,
                "role": role,
                "sha256": sorted(values),
                "same_batch_everywhere": len(values) == 1,
            }
            for (data, role), values in sorted(hashes.items())
        ],
        "maximum_forward_error": max(
            (row["forward_maximum_absolute_error"] for row in tables["gradient-groups"]),
            default=None,
        ),
        "maximum_gradient_reconstruction_error": max(
            (
                row["full_gradient_reconstruction_maximum_absolute_error"]
                for row in tables["gradient-groups"]
            ),
            default=None,
        ),
        "descriptive_aggregates": [
            {
                **dict(zip(keys, key, strict=True)),
                **{name: stats(values) for name, values in bucket.items()},
            }
            for key, bucket in sorted(aggregates.items())
        ],
        "actual_update_component_norms": [
            {
                **dict(zip(update_keys, key, strict=True)),
                "role": "training_mixture",
                "unique_component_update_l2": math.sqrt(value),
            }
            for key, value in sorted(update_squares.items())
        ],
        "source_files": source_files,
        "limits": [
            "Gradients use fixed caller-selected training batches, with dropout and autocast "
            "disabled; per-batch record counts are retained in the CSV.",
            "Repeated seeds, blocks and call positions reuse one dataset; aggregate counts "
            "are measurements, not independent worlds.",
            "Cosines exclude LayerNorm; the full gradient identity check includes "
            "LayerNorm and embeddings.",
            "Actual updates are AdamW changes on mixed training batches and include clipping, "
            "optimizer state and weight decay; these differ from role-specific "
            "diagnostic gradients.",
            "Shared-module update norms count each unique parameter once; independent models "
            "have more such coordinates.",
            "The statistics describe current local geometry and observed updates. They do not "
            "predict accuracy or establish a causal explanation of learning efficiency.",
            "Diagnostic coverage completion, training completion, independent audit, "
            "and W&B synchronization are separate states.",
        ],
    }
    json.dumps(summary, allow_nan=False)
    return summary, tables


def render(summary):
    lines = [
        f"# {summary['batch']}：Loop 梯度与实际更新记录",
        "",
        f"诊断覆盖状态：**{summary['state']}**。原始 JSON 只读；本报告未运行模型或重新训练。",
        "",
        f"最大前向重建误差：{summary['maximum_forward_error']}；最大梯度求和重建误差：{summary['maximum_gradient_reconstruction_error']}。",
        "",
        "## 文件与覆盖",
        "",
        "| 文件 | 行数 |",
        "|---|---:|",
        *[f"| [{key}.csv]({key}.csv) | {count} |" for key, count in summary["row_counts"].items()],
        "",
        "| 运行 | 状态 | 缺少诊断节点 | 缺少更新节点 | 独立完成标记 |",
        "|---|---|---|---|---|",
        *[
            f"| {row['run']} | {row['state']} | {row['missing_diagnostic_nodes']} "
            f"| {row['missing_actual_update_nodes']} "
            f"| {row['independent_completion_marker_present']} |"
            for row in summary["runs"]
        ],
        "",
        "## 各节点的描述统计",
        "",
        "均值按种子、基础块及调用位置汇总；这些计数不是独立世界数。"
        "`shallow` 没有跨调用位置的梯度余弦。",
        "",
        "| 角色 | 架构 | 宽度 | 基础块 | 重复数 | 步数 | 模块 | 调用间余弦均值 [最小, 最大] "
        "| 梯度求和平方范数比均值 |",
        "|---|---|---:|---:|---:|---:|---|---|---:|",
    ]
    for row in summary["descriptive_aggregates"]:
        cos = row["cosines"]
        formatted = (
            f"{cos['mean']:.4f} [{cos['min']:.4f}, {cos['max']:.4f}]"
            if cos["n"]
            else "无跨调用比较"
        )
        ratio = row["norm_ratios"]["mean"]
        ratio_text = f"{ratio:.4f}" if ratio is not None else "未定义"
        lines.append(
            f"| {row['role']} | {row['arm']} | {row['hidden_size']} "
            f"| {row['base_unique_blocks']} | {row['logical_repeats']} "
            f"| {row['step']} | {row['component']} "
            f"| {formatted} | {ratio_text} |"
        )
    lines.extend(
        [
            "",
            "平方范数比为 ‖Σgᵢ‖² / Σ‖gᵢ‖²；1 表示交叉内积净和为零，0 表示完全抵消，"
            "相同非零梯度给出调用次数。共享参数的梯度求和恒等式只对真实绑定的模块适用。",
            "",
            "## 解释边界",
            "",
        ]
    )
    lines.extend(f"- {limit}" for limit in summary["limits"])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    config = read(args.config)
    root = args.root or Path(config.get("results_root", f"results/{config['batch']}"))
    out = args.out or root / "diagnostic-report"
    summary, tables = summarize(config, root)
    summary["config_sha256"] = hashlib.sha256(args.config.read_bytes()).hexdigest()
    summary["report_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    out.mkdir(parents=True, exist_ok=True)
    for name, fields in zip(
        tables, (GROUP_FIELDS, OCCURRENCE_FIELDS, PAIR_FIELDS, UPDATE_FIELDS), strict=True
    ):
        with (out / f"{name}.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(tables[name])
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    (out / "report.md").write_text(render(summary))
    print(
        json.dumps(
            {"out": str(out), "state": summary["state"], "row_counts": summary["row_counts"]}
        )
    )


if __name__ == "__main__":
    main()
