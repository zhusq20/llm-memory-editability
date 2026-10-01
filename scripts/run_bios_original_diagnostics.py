"""Prepare immutable samples and independently evaluate existing bioS endpoints."""

import argparse
import json
import time
import traceback
from pathlib import Path

from llm_memory_editability.bios_original_data import load_json, write_json
from llm_memory_editability.bios_original_diagnostics import DEFAULT_ROOT, prepare, run_endpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--config", default="configs/bios-original-development-v1.json")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    runner = sub.add_parser("run")
    runner.add_argument("--endpoint")
    runner.add_argument("--world", type=int)
    runner.add_argument("--device", default="cuda:0")
    runner.add_argument("--memory-fraction", type=float, default=0.045)
    args = parser.parse_args()
    if args.command == "prepare":
        plan = prepare(args.config, args.root)
        print(json.dumps(dict(endpoints=len(plan["endpoints"]), people=plan["count"])))
    else:
        plan = load_json(args.root + "/plan.json")
        endpoints = [
            e
            for e in plan["endpoints"]
            if (args.endpoint is None or e["id"] == args.endpoint)
            and (args.world is None or e["world"] == args.world)
        ]
        if not endpoints:
            raise ValueError("No selected endpoints")
        for endpoint in endpoints:
            try:
                result = run_endpoint(args.root, endpoint["id"], args.device, args.memory_fraction)
            except Exception:
                write_json(
                    Path(args.root) / "failures" / f"{endpoint['id']}-{time.time_ns()}.json",
                    dict(endpoint=endpoint["id"], traceback=traceback.format_exc()),
                )
                raise
            print(
                json.dumps(
                    dict(
                        endpoint=endpoint["id"],
                        seconds=result["seconds"],
                        audit=result["audit"],
                        peak_allocated_bytes=result["peak_allocated_bytes"],
                    )
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
