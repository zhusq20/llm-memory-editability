"""Train or independently reload/audit one frozen multihop scaling run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm_memory_editability.multihop_scaling import audit, train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "audit"))
    parser.add_argument("--spec-file", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.mode == "train":
        if args.spec_file is None:
            parser.error("train requires --spec-file")
        value = json.loads(args.spec_file.read_text())
        spec = value.get("spec", value)
        source = value.get("source", spec.get("source"))
        if not isinstance(source, dict):
            parser.error("spec-file must include a frozen source dictionary")
        train(spec, args.out, source, args.device)
    else:
        print(json.dumps(audit(args.out, args.device)), flush=True)


if __name__ == "__main__":
    main()
