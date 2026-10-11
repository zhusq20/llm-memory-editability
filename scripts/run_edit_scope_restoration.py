#!/usr/bin/env python3
"""Read or independently replay fixed saved joint-edit restoration shards."""

import argparse
import json
from pathlib import Path

from llm_memory_editability.edit_scope_restoration import ROLES, audit_shard, run_shard


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--source-batch", type=Path, required=True)
    run.add_argument("--role", choices=ROLES, required=True)
    run.add_argument("--learning-rate", type=float, choices=(1e-5, 1e-4), required=True)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--device", default="cuda:0")
    run.add_argument("--limit", type=int, choices=(1,))
    audit = sub.add_parser("audit")
    audit.add_argument("--out", type=Path, required=True)
    audit.add_argument("--audit-out", type=Path, required=True)
    audit.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.command == "audit":
        result = audit_shard(args.out, args.audit_out, args.device)
    else:
        result = run_shard(
            args.source_batch,
            args.role,
            args.learning_rate,
            args.out,
            args.device,
            limit=args.limit,
        )
    print(
        json.dumps(
            {
                "case_count": result["case_count"],
                "optimization_updates": 0,
                "seconds": result["seconds"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
