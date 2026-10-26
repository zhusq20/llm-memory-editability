"""Read back every phase's complete W&B JSON without running a model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import wandb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    api = wandb.Api(timeout=60)
    records = []
    for phase, updates in (("development", 32), ("evaluation", 128), ("audit", 0)):
        directory = args.root / phase
        local = json.loads((directory / "summary.json").read_text())
        tracking = json.loads((directory / "wandb.json").read_text())
        run = api.run(f"zhusq20/llm-memory-editability/{tracking['id']}")
        if run.state != "finished" or not local["passed"]:
            raise AssertionError(f"Phase not successfully finished: {phase}")
        for key, expected in (("slow_parameter_updates", 0), ("fast_updates", updates)):
            if run.summary.get(key) != expected or local[key] != expected:
                raise AssertionError(f"Phase count mismatch: {phase}/{key}")
        destination = args.out / phase
        destination.mkdir()
        downloaded = run.file("summary.json").download(root=str(destination), replace=False)
        content = Path(downloaded.name).read_bytes()
        if json.loads(content) != local:
            raise AssertionError(f"Full cloud JSON differs: {phase}")
        records.append(
            {
                "phase": phase,
                "url": tracking["url"],
                "state": run.state,
                "full_json_exact": True,
                "downloaded_sha256": hashlib.sha256(content).hexdigest(),
                "slow_parameter_updates": 0,
                "fast_updates": updates,
            }
        )
    result = {"passed": True, "records": records, "model_forwards": 0}
    (args.out / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
