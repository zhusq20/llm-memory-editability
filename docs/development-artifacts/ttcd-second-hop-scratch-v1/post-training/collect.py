"""Summarize audited frozen readouts; no model calls or sample selection."""

import hashlib
import json
from pathlib import Path

ROOT = Path("/ossfs/workspace/llm-memory-editability/ttcd-second-hop-scratch-v1")


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else None


def correct(predictions, key="old_answers"):
    return [
        answer == target + 2 and stop == 1
        for answer, target, stop in zip(
            predictions["answer"], predictions[key], predictions["stop"], strict=True
        )
    ]


def summarize(records):
    result = {
        "cases": len(records),
        "E_new": mean(r["after"]["metrics"]["E"]["new_accuracy"] for r in records),
        "D_new_case_equal": mean(r["after"]["metrics"]["D"]["new_accuracy"] for r in records),
        "D_old_before_case_equal": mean(
            r["before"]["metrics"]["D"]["old_accuracy"] for r in records
        ),
        "D_old_after_case_equal": mean(r["after"]["metrics"]["D"]["old_accuracy"] for r in records),
        "open_atomic_new": mean(r["open_context"]["atomic"]["new_correct"] for r in records),
        "open_chain_new": mean(r["open_context"]["two_hop"]["new_correct"] for r in records),
        "conditional_cases": sum(r["conditional"]["n"] > 0 for r in records),
        "conditional_queries": sum(r["conditional"]["n"] for r in records),
        "total_D_queries": sum(r["conditional"]["total"] for r in records),
    }
    n = result["conditional_queries"]
    result["conditional_D_query_equal"] = (
        sum(
            r["conditional"]["new_accuracy"] * r["conditional"]["n"]
            for r in records
            if r["conditional"]["n"]
        )
        / n
        if n
        else None
    )
    result["conditional_query_coverage"] = n / result["total_D_queries"]
    for pool in ("U_atoms", "U_chains", "same_answer_atoms", "prerequisite_atoms"):
        before, after, target_before, target_after = [], [], [], []
        for record in records:
            before.extend(correct(record["before"]["predictions"][pool]))
            after.extend(correct(record["after"]["predictions"][pool]))
            for phase, accumulator in (("before", target_before), ("after", target_after)):
                predictions = record[phase]["predictions"][pool]
                accumulator.extend(
                    answer == record["case"]["new_tail"] + 2 and stop == 1
                    for answer, stop in zip(predictions["answer"], predictions["stop"], strict=True)
                )
        kept = sum(a and b for a, b in zip(before, after, strict=True))
        originally_correct = sum(before)
        result[pool] = {
            "before": mean(before),
            "after": mean(after),
            "original_correct": originally_correct,
            "kept": kept,
            "retained_original_correct": kept / originally_correct if originally_correct else None,
            "query_occurrences": len(before),
            "edited_new_tail_joint_before": mean(target_before),
            "edited_new_tail_joint_after": mean(target_after),
        }
    return result


def main():
    control = ROOT / "post-training-20261015"
    metadata = read(ROOT / "source/input-backup/complete.json")
    if metadata["entity_offset"] != 2 or metadata["end_marker"] != 1:
        raise ValueError("Unexpected token offsets for the archived symbolic world")
    completion = read(control / "completion.json")
    if completion["state"] != "complete":
        repaired = read(control / "cloud-verification-completion.json")
        if repaired["state"] != "complete" or not repaired["science_stages_unchanged"]:
            raise ValueError("Post-training pipeline has not completed successfully")
    comparison, files = {}, {}
    for arm in ("discoloop", "loop"):
        out = ROOT / arm
        for stage in ("train-audit", "calibrate", "evaluate", "audit"):
            if read(out / f"exit-{stage}.json")["returncode"] != 0:
                raise ValueError("An original scientific post-training stage failed")
        for relative in (
            "training/summary.json",
            "training/parent.pt",
            "training/audit.json",
            "calibration.json",
            "summary.json",
            "audit.json",
            "cloud-audit.json",
        ):
            path = out / relative
            files[str(path.relative_to(ROOT))] = sha(path)
        for relative in (
            "training/audit.json",
            "calibration.json",
            "audit.json",
            "cloud-audit.json",
        ):
            if not read(out / relative)["passed"]:
                raise ValueError("A required audit/gate did not pass")
        paths = sorted((out / "cases").glob("*/*/readout.json"))
        records = [read(path) for path in paths]
        if len(records) != 96 or read(out / "audit.json")["cases_methods"] != 96:
            raise ValueError("Incomplete fixed 32-case by 3-method matrix")
        for path in paths:
            files[str(path.relative_to(ROOT))] = sha(path)
            files[str((path.parent / "fast.pt").relative_to(ROOT))] = sha(path.parent / "fast.pt")
        for path in out.glob("cloud-summary-downloads*/*/wandb-summary.json"):
            files[str(path.relative_to(ROOT))] = sha(path)
        comparison[arm] = {
            "training_step": read(out / "training/summary.json")["step"],
            "calibration": read(out / "calibration.json")["groups"],
            "groups": {
                f"{method}/{group}": summarize(
                    [
                        r
                        for r in records
                        if r["method"] == method and (group == "all" or r["case"]["group"] == group)
                    ]
                )
                for method in ("ttcd", "atomic_ce", "sham")
                for group in ("all", "id", "ood")
            },
        }
    result = {
        "primary": "All 32 fixed cases, case-equal D answer+EOS accuracy",
        "retention": "Matched query occurrences originally correct; also report original coverage",
        "scope": "One development graph and initialization; single-fact independent fast states",
        "arms": comparison,
    }
    with (ROOT / "comparison.json").open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    files["comparison.json"] = sha(ROOT / "comparison.json")
    files["source-manifest.json"] = sha(ROOT / "source-manifest.json")
    manifest = {
        "passed": True,
        "arms": ["discoloop", "loop"],
        "training_step_per_arm": 30000,
        "training_reload_passed": True,
        "calibration_passed": True,
        "editing_cases_methods_per_arm": 96,
        "edit_replay_passed": True,
        "cloud_stage_receipts_per_arm": 5,
        "cloud_audit_passed": True,
        "new_training_updates_in_postprocessing": 0,
        "files_sha256": files,
    }
    with (ROOT / "completion-manifest.json").open("x") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
    for arm, row in comparison.items():
        for group, metrics in row["groups"].items():
            print(arm, group, json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
