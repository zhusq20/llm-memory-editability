"""Descriptive reporting after frozen evaluation; no recipe selection or writes."""

import hashlib
import itertools
import json
from pathlib import Path


def rate(values):
    values = list(values)
    return {
        "correct": sum(values),
        "n": len(values),
        "accuracy": sum(values) / len(values) if values else None,
    }


def derive(folder):
    summary = json.loads((folder / "result.json").read_text())
    records = [json.loads(p.read_text()) for p in sorted(folder.glob("evaluation-novel-*.json"))]
    if len(records) != 32:
        raise AssertionError("All32 formal facts must be retained")
    retained = {"atoms": [], "compositions": []}
    by_state = []
    for record in records:
        local = {"atoms": [], "compositions": []}
        for pool in record["before"]["extended_old"]:
            key = "atoms" if pool.startswith("atomic") else "compositions"
            b = record["before"]["extended_old"][pool]["final_correct"]
            a = record["after"]["extended_old"][pool]["final_correct"]
            kept = [x for x, was_correct in zip(a, b, strict=True) if was_correct]
            retained[key].extend(kept)
            local[key].extend(kept)
        by_state.append(
            {"case_id": record["fact"]["case_id"], **{k: rate(v) for k, v in local.items()}}
        )
    return {
        "summary_sha256": hashlib.sha256((folder / "result.json").read_bytes()).hexdigest(),
        "conditions": summary["conditions"],
        "conditional": summary["conditional"],
        "atomic_answer_only": rate(
            r["after"]["novel_atom"]["answer"][0] == r["after"]["novel_atom"]["expected"][0]
            for r in records
        ),
        "pooled_original_correct_retention": {k: rate(v) for k, v in retained.items()},
        "worst_single_edit": {k: min(by_state, key=lambda r: r[k]["accuracy"]) for k in retained},
        "edits_with_any_old_loss": sum(
            any(r[k]["correct"] < r[k]["n"] for k in retained) for r in by_state
        ),
        "by_old_pool": summary["extended_retention"],
        "selected_learning_rate": summary.get("selected_learning_rate"),
        "trainable_parameters": summary.get("trainable_parameters", 9437184),
        "formal_optimizer_updates": summary.get("scientific_optimizer_updates"),
        "wall_seconds_shared": summary["wall_seconds"]
        if "scientific_optimizer_updates" in summary
        else None,
        "summary_recovery_wall_seconds": summary["wall_seconds"]
        if "recovered_complete_records" in summary
        else None,
        "cost_scope": (
            "CE stage includes initialization, writing, reading and hashes on shared GPU"
            if "scientific_optimizer_updates" in summary
            else "AlphaEdit summary recovery cost; not original writing cost"
        ),
        "write_GPU_milliseconds": sum(
            r["write_info"]["gpu_optimizer_milliseconds"] for r in records
        )
        if "gpu_optimizer_milliseconds" in records[0].get("write_info", {})
        else None,
        "maximum_absolute_parameter_change": max(
            r["write_info"]["maximum_absolute_parameter_change"] for r in records
        )
        if "maximum_absolute_parameter_change" in records[0].get("write_info", {})
        else None,
        "records": records,
    }


def main():
    base = Path("/ossfs/workspace/llm-memory-editability/ce-scope-discoloop-novel-v1")
    complete = json.loads((base / "completion-manifest.json").read_text())
    if not complete["passed"]:
        raise ValueError("Formal evaluation and independent audits must be completed")
    root = base / complete["output_root"]
    methods = {
        "alphaedit": base.parent / "alphaedit-discoloop-novel-v1/output/attempt-5/evaluation",
        **{s: root / f"evaluation-{s}" for s in ("down", "mlp", "all")},
    }
    reports = {name: derive(folder) for name, folder in methods.items()}
    common = {}
    for name in ("down", "mlp", "all"):
        left, right = reports["alphaedit"]["records"], reports[name]["records"]
        if [r["fact"] for r in left] != [r["fact"] for r in right]:
            raise AssertionError("Paired cases changed")
        pairs = [
            (a, b)
            for a, b in zip(left, right, strict=True)
            if a["after"]["novel_atom"]["final_correct"][0]
            and b["after"]["novel_atom"]["final_correct"][0]
        ]
        common[name] = {
            "shared_atomic_success_facts": len(pairs),
            **{
                role: {
                    method: rate(
                        value
                        for pair in pairs
                        for value in pair[index]["after"][role]["final_correct"]
                    )
                    for method, index in (("alphaedit", 0), (name, 1))
                }
                for role in ("first_hop", "second_hop")
            },
        }
    matched = {}
    for left_name, right_name in itertools.combinations(reports, 2):
        pairs = [
            (a, b)
            for a, b in zip(
                reports[left_name]["records"], reports[right_name]["records"], strict=True
            )
            if a["after"]["novel_atom"]["final_correct"][0]
            and b["after"]["novel_atom"]["final_correct"][0]
        ]
        matched[f"{left_name}_vs_{right_name}"] = {
            "shared_atomic_success_facts": len(pairs),
            **{
                role: {
                    method: rate(
                        value
                        for pair in pairs
                        for value in pair[index]["after"][role]["final_correct"]
                    )
                    for method, index in ((left_name, 0), (right_name, 1))
                }
                for role in ("first_hop", "second_hop")
            },
        }
    for report in reports.values():
        del report["records"]
    result = {
        "descriptive_posthoc": True,
        "new_recipe_selection": False,
        "methods": reports,
        "shared_atomic_success_with_alphaedit": common,
        "shared_atomic_success_pairwise": matched,
        "scientific_updates_unique": 3840,
        "actual_updates_including_failed_range": [3860, 3880],
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    destination = Path("results/ce-scope-discoloop-novel-v1/derived-readouts.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                name: {
                    "atomic": r["conditions"]["after"]["novel_atom"]["accuracy"],
                    "first": r["conditions"]["after"]["first_hop"]["accuracy"],
                    "second": r["conditions"]["after"]["second_hop"]["accuracy"],
                    "old": r["pooled_original_correct_retention"],
                }
                for name, r in reports.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
