"""Run or independently audit the fixed 50% old-data continuation recipe."""

import argparse
import json
from pathlib import Path

from llm_memory_editability import sequential_replay


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "audit"))
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.action == "run":
        if args.spec is None:
            parser.error("run requires --spec")
        result = sequential_replay.run(json.loads(args.spec.read_text()), args.out, args.device)
    else:
        result = sequential_replay.audit(args.out, args.device)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
