"""Fixed512 H5 contrasts by world/initialization, with raw outcome counts."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

SCOPES = ("mlp", "all", "all-freeze-embedding")
METRICS = (
    "E",
    "E_default",
    "E_actual",
    "D_heldout",
    "D_conflict_heldout",
    "U_full_damage",
    "U_heldout_damage",
    "U_full_0_rate",
    "U_full_coverage",
)
COUNTS = (
    "E_count",
    "E_n",
    "E_default_count",
    "E_default_n",
    "E_actual_count",
    "E_actual_n",
    "D_conflict_heldout_count",
    "D_conflict_heldout_n",
    "U_full_broken",
    "U_full_known",
    "U_full_n",
    "U_full_0_broken",
    "U_full_0_known",
)


def csv_write(path, rows):
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows, keys):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    output = []
    for key, group in sorted(grouped.items()):
        row = dict(zip(keys, key, strict=True))
        row["cases"] = len(group)
        for metric in METRICS:
            values = [r[metric] for r in group if r[metric] is not None]
            row[metric] = mean(values) if values else None
            row[metric + "_valid_cases"] = len(values)
        for count in COUNTS:
            row[count] = sum(int(r[count]) for r in group)
        output.append(row)
    return output


def analyze(source):
    output = Path(source).resolve() / "summary"
    audit = json.loads((output / "audit.json").read_text())
    if not audit["complete"] or audit["new_cases"] != 48 or audit["frozen_readout_checks"] != 48:
        raise ValueError("H5 must be fully audited, including every frozen readout")
    rows = []
    for row in csv.DictReader((output / "editing.csv").open()):
        if row["step"] != "512":
            continue
        for key, value in row.items():
            try:
                row[key] = float(value) if value else None
            except ValueError:
                pass
        rows.append(row)
    if len(rows) != 144:
        raise ValueError("H5 analysis must retain all48 new and96 reference fixed endpoints")
    blocks = aggregate(rows, ("width", "world", "seed", "scope"))
    worlds = aggregate(rows, ("width", "world", "scope"))
    overall = aggregate(rows, ("width", "scope"))
    organizations = aggregate(rows, ("width", "world", "condition", "scope"))
    group = defaultdict(dict)
    for row in blocks:
        group[row["width"], row["world"], row["seed"]][row["scope"]] = row
    paired = []
    for (width, world, seed), scopes in sorted(group.items()):
        if set(scopes) != set(SCOPES) or any(r["cases"] != 6 for r in scopes.values()):
            raise ValueError("Incomplete H5 world/initialization block")
        row = dict(width=width, world=world, seed=seed, cases_per_scope=6)
        for metric in METRICS:
            for reference in ("mlp", "all"):
                a, b = scopes[SCOPES[-1]][metric], scopes[reference][metric]
                row[f"freeze_minus_{reference}_{metric}"] = (
                    a - b if a is not None and b is not None else None
                )
        paired.append(row)
    for filename, data in (
        ("endpoint-world-seed.csv", blocks),
        ("endpoint-world.csv", worlds),
        ("endpoint-overall.csv", overall),
        ("endpoint-organization.csv", organizations),
        ("endpoint-paired-world-seed.csv", paired),
    ):
        csv_write(output / filename, data)
    result = dict(
        blocks=blocks,
        worlds=worlds,
        overall=overall,
        paired_blocks=paired,
        fixed_step=512,
        limitations=[
            "Two reused development worlds; initialization is not an independent world.",
            "Freezes both tied input embedding and output readout; not output-only.",
            "Parameter count and clipped gradient set differ across scopes.",
            "All same-parent E93 edits retained; no best-step or learned-parent selection.",
        ],
    )
    (output / "endpoint-analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        "# H5 固定终点：逐世界/初始化",
        "",
        "| Width | World | Seed | Scope | E | Conflict heldout D | U damage | "
        "D correct/n | U broken/known |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in blocks:
        values = [100 * row[k] for k in ("E", "D_conflict_heldout", "U_full_damage")]
        lines.append(
            f"| {int(row['width'])} | {int(row['world'])} | {int(row['seed'])} | {row['scope']} | "
            + " | ".join(f"{v:.3f}%" for v in values)
            + f" | {row['D_conflict_heldout_count']}/{row['D_conflict_heldout_n']} | "
            + f"{row['U_full_broken']}/{row['U_full_known']} |"
        )
    lines += [
        "",
        "百分比是案例等权平均；合计整数仅交代分母，比例不必等于该平均值。"
        "干预冻结绑定输入/输出而非单独读出；参数量及梯度裁剪集合也变化。",
        "",
    ]
    (output / "endpoint-report.md").write_text("\n".join(lines))
    print(json.dumps(dict(blocks=len(blocks), paired_blocks=len(paired), endpoint_cases=len(rows))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="results/bios-mechanism-dev-v1/h5-readout-control")
    analyze(parser.parse_args().source)
