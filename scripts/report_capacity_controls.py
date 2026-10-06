"""Summarize all audited results, retaining incomplete/failed matrix entries."""

import argparse
import csv
import json
from pathlib import Path


def report(root):
    config = json.loads((root / "frozen-config.json").read_text())
    records, missing = [], []
    for spec in config["runs"]:
        path = root / "runs" / spec["name"]
        if not (path / "audit.json").exists():
            missing.append({"name": spec["name"], "failed": (path / "failure.json").exists()})
            continue
        audit = json.loads((path / "audit.json").read_text())
        if not audit.get("passed"):
            missing.append({"name": spec["name"], "failed": True})
            continue
        m = audit["metrics"]
        row = {
            "name": spec["name"],
            "heads_n": spec["heads_n"],
            "arm": "old_weights" if spec.get("evaluation_only") else spec["training_mode"],
            "parent": spec.get("parent"),
            "metrics": m,
            "density": m["information"]["data_bits_per_parameter"],
            "learned_density": m["information"]["learned_bits_per_parameter"],
            "atomic": m["atomic"]["accuracy"],
            "train_composition": m["train_composition"]["accuracy"],
        }
        for pool in ("II", "IO", "OI", "OO"):
            for method in ("", "serial_", "wrong_bridge_", "oracle_bridge_"):
                row[method + pool] = m[method + pool]["accuracy"]
            row[pool + "_both_coverage"] = m[pool]["both_atomic_coverage"]
        records.append(row)
    result = {
        "batch": config["batch"],
        "expected": len(config["runs"]),
        "audited": len(records),
        "missing_or_failed": missing,
        "independent_worlds": 1,
        "initializations": 1,
        "runs": records,
        "interpretation_limits": [
            "External two-call recall is not implicit composition or an internal repair.",
            "Atomic replay matches examples/updates/nominal matmul FLOPs, not atomic exposure.",
            "Atomic-only matches base factual exposure, not total compute.",
            "Continuation changes duration; the paired arm changes late weight decay only.",
            "No parameter capacity bound, scaling exponent, or independent-world confirmation.",
        ],
    }
    (root / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    fields = [k for k in records[0] if k != "metrics"] if records else ["name"]
    with (root / "summary.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    # Each old endpoint uses the identical world as its paired new controls.
    paired = []
    for heads in (256, 1024, 4096):
        base_name = "recall-calibration-long" if heads == 256 else f"recall-load-h{heads}"
        base = next((r for r in records if r["name"] == base_name), None)
        if base is None:
            base = next(
                (
                    r
                    for r in records
                    if r["heads_n"] == heads and r["arm"] == "mixed" and not r["parent"]
                ),
                None,
            )
        if base:
            for arm in ("atomic", "atomic_replay"):
                row = next((r for r in records if r["name"] == f"{arm}-h{heads}"), None)
                if row is None:
                    row = next(
                        (
                            r
                            for r in records
                            if r["heads_n"] == heads and r["arm"] == arm and not r["parent"]
                        ),
                        None,
                    )
                if row:
                    paired.append(
                        {
                            "heads_n": heads,
                            "arm": arm,
                            "atomic_delta": row["atomic"] - base["atomic"],
                            "learned_density_delta": row["learned_density"]
                            - base["learned_density"],
                            "serial_II_delta": row["serial_II"] - base["serial_II"],
                        }
                    )
    result["paired_training_comparisons"] = paired
    (root / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# 存储、连续调用与学习控制",
        "",
        f"完成重载审计 {len(records)}/{len(config['runs'])}；一个开发世界、一个初始化。",
        "",
        "| 运行 | bits/参数 | 单跳 | 直接II | 自产桥II | 错桥II | 自产桥OO |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in records:
        lines.append(
            f"| {r['name']} | {r['density']:.3f} | {r['atomic']:.2%} | {r['II']:.2%} "
            f"| {r['serial_II']:.2%} | {r['wrong_bridge_II']:.2%} | {r['serial_OO']:.2%} |"
        )
    lines += [
        "",
        "自产桥使用模型自行生成的实体及公开拆题规则，增加一次调用；oracle单列。它不证明网络内部已学会组合，也不单独证明串扰机制。",
        "",
        "原子复习对照与旧混合训练匹配样本总数、更新数及名义矩阵计算，但多了原子曝光；原子单独训练匹配原子曝光，总计算较少。两个对照共同解释训练配方成本。",
        "",
        "后续先依据原生组合的完整学习轨迹判断学习前提；在其建立前，不拟合组合容量尺度律。",
    ]
    (root / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"audited": len(records), "expected": len(config["runs"])}))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    report(parser.parse_args().root)
