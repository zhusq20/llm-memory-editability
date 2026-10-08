"""Run, independently audit, or inspect fixed write-target edit comparisons."""

import argparse
import json
from pathlib import Path

from llm_memory_editability import write_target_editing as experiment
from llm_memory_editability.grok_depth import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--spec", type=Path, required=True)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--device", default="cuda:0")
    audit = sub.add_parser("audit")
    audit.add_argument("--out", type=Path, required=True)
    audit.add_argument("--device", default="cuda:0")
    diagnostic = sub.add_parser("diagnostics")
    diagnostic.add_argument("--existing-dir", type=Path, required=True)
    diagnostic.add_argument("--out", type=Path, required=True)
    diagnostic.add_argument("--device", default="cuda:0")
    decision = sub.add_parser("development-decision")
    decision.add_argument("--run-dirs", nargs="+", type=Path, required=True)
    decision.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "run":
        result = experiment.run(json.loads(args.spec.read_text()), args.out, args.device)
    elif args.command == "audit":
        result = experiment.audit(args.out, args.device)
    elif args.command == "diagnostics":
        result = experiment.diagnostics(args.existing_dir, args.out, args.device)
    else:
        result = experiment.development_decision(args.run_dirs)
        write_json(args.out, result)
    print(
        json.dumps(
            {key: value for key, value in result.items() if key not in {"branches", "records"}}
        )
    )


if __name__ == "__main__":
    main()
