"""Independently verify and summarize the fixed-person same-weight diagnosis."""

import csv
import json
from collections import defaultdict
from pathlib import Path

import torch

from llm_memory_editability.bios_original_data import answer, date, question
from llm_memory_editability.bios_original_diagnostics import operation_prompt, parse_date
from llm_memory_editability.bios_original_train import normalize
from llm_memory_editability.grok_depth import write_json

ROOT = Path("docs/development-artifacts/bios-same-weight-v1")


def main():
    lock = json.loads((ROOT / "lock.json").read_text())
    rows, checks = [], 0
    for endpoint in lock["endpoints"]:
        world, condition, stage = (endpoint[k] for k in ("world", "condition", "stage"))
        path = ROOT / "endpoints" / f"w{world}-{condition}-{stage}.json"
        data = json.loads(path.read_text())
        assert data["endpoint"] == endpoint
        people = json.loads(Path(f"data/bios-original-v1/world-{world}/people.json").read_text())
        stats = defaultdict(lambda: dict(n=0, correct=0, eos_correct=0))
        for row in data["native"]:
            n = len(row["target_ids"])
            correct = row["generated_ids"][:n] == row["target_ids"]
            assert correct == row["attribute_exact"]
            t = stats["native_" + row["attribute"]]
            t["n"] += 1
            t["correct"] += int(correct)
            t["eos_correct"] += int(row["attribute_and_boundary_exact"])
            checks += 1
        if stage == "pretrain":
            old = json.loads(
                Path(
                    "docs/development-artifacts/bios-original-trajectory-v1/nodes/"
                    f"world-{world}-init-1427-{condition}-pass-540.json"
                ).read_text()
            )
            for new, previous in zip(data["native"], old["predictions"], strict=True):
                assert new["person_id"] == previous["person_id"]
                assert new["attribute_exact"] == previous["attribute_exact"]
                assert abs(new["full_attribute_nll"] - previous["full_attribute_nll"]) < 1e-5
                checks += 1
        extraction = {
            (r["person_id"], r["view"]): r for r in data["predictions"] if r["mode"] == "qa_date"
        }
        for row in data["predictions"]:
            p = people[row["person_id"]]
            mode, view = row["mode"], row["view"]
            task = mode[3:] if mode.startswith("qa_") else "parity"
            assert row["target"] == answer(p, task)
            if mode.startswith("qa_") or mode == "direct_parity":
                assert row["prompt"] == question(p, task, view)
            elif mode == "oracle_parity":
                assert row["prompt"] == operation_prompt("parity", date(p))
            else:
                supplied = extraction[p["id"], view]["generated"]
                assert row["generated_date"] == supplied
                assert row["prompt"] == operation_prompt("parity", supplied)
                parsed = parse_date(supplied)
                assert row["input_parseable"] == (parsed is not None)
                assert row["input_month_correct"] == (
                    parsed is not None and parsed[1] == p["month"]
                )
            correct = normalize(row["generated"]) == normalize(row["target"])
            assert correct == row["correct"]
            key = f"{mode}/view{view}"
            t = stats[key]
            t["n"] += 1
            t["correct"] += int(correct)
            t["eos_correct"] += int(correct and row["eos"])
            if mode == "autonomous_parity":
                t["parseable_n"] = t.get("parseable_n", 0) + int(row["input_parseable"])
                t["correct_month_n"] = t.get("correct_month_n", 0) + int(row["input_month_correct"])
                t["parseable_correct"] = t.get("parseable_correct", 0) + int(
                    row["input_parseable"] and correct
                )
            checks += 1
        for key, value in stats.items():
            assert value["n"] == data["totals"][key]["n"]
            assert value["correct"] == data["totals"][key]["correct"]
            rows.append(
                dict(
                    world=world,
                    condition=condition,
                    stage=stage,
                    metric=key,
                    **value,
                    accuracy=value["correct"] / value["n"],
                )
            )
    fields = sorted({key for row in rows for key in row})
    with (ROOT / "scores.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    groups = defaultdict(list)
    for row in rows:
        groups[row["condition"], row["stage"], row["metric"]].append(row)
    summary = [
        dict(
            condition=k[0],
            stage=k[1],
            metric=k[2],
            worlds=len(v),
            accuracy=sum(r["accuracy"] for r in v) / len(v),
            n=sum(r["n"] for r in v),
            correct=sum(r["correct"] for r in v),
        )
        for k, v in groups.items()
    ]
    base_checks = []
    for world in (142701, 142702):
        for condition in ("S", "M", "MP"):
            selected = {
                e["stage"]: e
                for e in lock["endpoints"]
                if e["world"] == world and e["condition"] == condition
            }
            base = torch.load(
                selected["pretrain"]["checkpoint"], map_location="cpu", weights_only=False
            )["model"]
            for stage in ("adapt", "task"):
                adapted = torch.load(
                    selected[stage]["checkpoint"], map_location="cpu", weights_only=False
                )["model"]
                assert all(torch.equal(value, adapted[key]) for key, value in base.items())
                base_checks.append(
                    dict(
                        world=world,
                        condition=condition,
                        stage=stage,
                        unchanged_base_tensors=len(base),
                    )
                )
    write_json(ROOT / "base-parameter-audit.json", dict(status="passed", comparisons=base_checks))
    write_json(
        ROOT / "summary.json",
        dict(
            status="complete",
            endpoints=18,
            checks=checks,
            predictions=18 * 384,
            summary=summary,
            independent_worlds=2,
            people_per_world=32,
            base_parameter_comparisons=len(base_checks),
            frozen_base_unchanged=True,
        ),
    )
    for row in summary:
        if row["metric"] in (
            "native_date",
            "native_birthcity",
            "qa_date/view0",
            "direct_parity/view0",
            "autonomous_parity/view0",
        ):
            print(row)


if __name__ == "__main__":
    main()
