#!/usr/bin/env python3
"""Run or independently replay a development-only U1 write-scope arm."""

import argparse
import json
from pathlib import Path

from llm_memory_editability.edit_scope_development import ARMS, audit_arm, run_arm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--audit-dir", type=Path)
    args = parser.parse_args()
    if args.audit_dir:
        result = audit_arm(args.config, args.audit_dir, args.device)
        print(json.dumps({"passed": result["passed"], "checked": result["checked"]}), flush=True)
    else:
        if args.arm is None or args.out is None:
            parser.error("--arm and --out are required for an execution")
        result = run_arm(args.config, args.arm, args.out, args.device, args.smoke)
        print(
            json.dumps(
                {
                    "arm": args.arm,
                    "endpoints": result["endpoints_completed"],
                    "seconds": result["wall_seconds"],
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
