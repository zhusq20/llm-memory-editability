#!/usr/bin/env python3
"""Run separate paper-style generation or regrade existing immutable answers."""

import argparse
import json
from pathlib import Path

from llm_memory_editability import knowledge_method_paper_eval as P


def main():
    parser = argparse.ArgumentParser(
        description="Paper grading protocol with configured Llama3 or Qwen3-4B no-thinking judge."
    )
    parser.add_argument("--stage", choices=("regrade-old", "grade-json", "evaluate"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--batch", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--records", type=Path)
    parser.add_argument("--arm")
    parser.add_argument("--judge-model", choices=(P.JUDGE_MODEL, P.QWEN_JUDGE_MODEL))
    parser.add_argument("--judge-model-path", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.judge_model:
        config["judge_model"] = args.judge_model
    if args.judge_model_path:
        config["judge_model_path"] = str(args.judge_model_path)
    if args.stage == "regrade-old":
        if args.batch is None or args.output is None:
            parser.error("regrade-old requires --batch and a separate --output")
        if args.output.resolve().is_relative_to(args.batch.resolve()):
            parser.error("Old-batch regrading output must be outside the frozen batch")
        result = P.regrade_old(config, args.batch, args.output)
    elif args.stage == "grade-json":
        if args.records is None or args.output is None:
            parser.error("grade-json requires --records and --output")
        if args.records.suffix == ".jsonl":
            records = [json.loads(line) for line in args.records.read_text().splitlines()]
        else:
            records = json.loads(args.records.read_text())
        result = P.grade_records(records, config, args.output)
    else:
        if args.batch is None or args.arm is None:
            parser.error("evaluate requires --batch and --arm")
        result = P.evaluate(config, args.batch, args.arm)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
