#!/usr/bin/env python3
"""Collect completed arms without rescoring, filtering cases, or changing endpoints."""

import argparse
import json
from pathlib import Path

ARMS = ("pd_lora", "sft_lora", "sft_full", "sft_down")


def collect(batch):
    output = {"batch": batch.name, "arms": {}, "complete": True}
    table = [
        "# Qwen knowledge-method diagnosis — development results",
        "",
        "Natural HotpotQA method transplant; 32 cases, not a full paper reproduction.",
        "PD and SFT use different teacher-answer temperatures. EM is not the paper's LLM score.",
        "",
        "| Arm | Closed EM before → after | Context EM before → after | Old correct retained |",
        "| --- | --- | --- | --- |",
    ]
    for arm in ARMS:
        root = batch / "runs" / arm
        paths = {
            "evaluation": root / "eval/evaluate.json",
            "audit": root / "audit/audit.json",
            "training": root / "complete.json",
            "manifest": root / "manifest.json",
        }
        if not all(path.is_file() for path in paths.values()):
            output["complete"] = False
            output["arms"][arm] = {"status": "incomplete"}
            table.append(f"| {arm} | pending | pending | pending |")
            continue
        records = {key: json.loads(path.read_text()) for key, path in paths.items()}
        if not records["audit"]["passed"] or records["training"]["smoke"]:
            raise ValueError(f"{arm}: audit did not pass or engineering result was supplied")
        if records["training"]["step"] != records["manifest"]["steps"]:
            raise ValueError(f"{arm}: training budget incomplete")
        summary = records["evaluation"]["summary"]
        output["arms"][arm] = {
            "status": "complete",
            "summary": summary,
            "teacher_target_fit": records["evaluation"]["teacher_target_fit"],
            "training": records["training"],
            "audit": records["audit"],
            "trainable_parameters": records["manifest"]["trainable_parameters"],
            "data_sha256": records["manifest"]["data_sha256"],
        }
        c, x, r = summary["closed"], summary["context"], summary["retention"]
        table.append(
            f"| {arm} | {c['before']:.2%} → {c['after']:.2%} "
            f"| {x['before']:.2%} → {x['after']:.2%} "
            f"| {r['retained_n']}/{r['initially_correct_n']} |"
        )
    hashes = {row["data_sha256"] for row in output["arms"].values() if "data_sha256" in row}
    if len(hashes) > 1:
        raise ValueError("Arms used different training files")
    table += [
        "",
        "All case-level generations and fitting diagnostics remain in each arm's eval file.",
        "Completion here includes independent reload, not independent W&B cloud verification.",
    ]
    return output, "\n".join(table) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=Path, required=True)
    args = parser.parse_args()
    summary, report = collect(args.batch)
    for name, text in (
        ("summary.json", json.dumps(summary, ensure_ascii=False, indent=2) + "\n"),
        ("report.md", report),
    ):
        temporary = args.batch / (name + ".tmp")
        temporary.write_text(text)
        temporary.replace(args.batch / name)


if __name__ == "__main__":
    main()
