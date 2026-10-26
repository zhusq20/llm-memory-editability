"""Compare Qwen3-1.7B E3 at its calibrated rate (v2) with the same E3 at 1.5e-4."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from summarize_read_interface_v2 import e3

PERSISTENT = Path(
    "/mnt/disk2_from_server2/siqizhu4/lm-memory-persistent/workspace/llm-memory-editability"
)
RUNS = {
    "v2_calibrated": PERSISTENT / "knowledge-read-interface-v2/runs/dev/qwen3-1.7b",
    "lr_1.5e-4": PERSISTENT / "e3-learning-rate-control-v1/runs/dev/qwen3-1.7b",
}
ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "results/e3-learning-rate-control-v1")
    args = parser.parse_args()
    result = {name: e3(run) for name, run in RUNS.items()}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(result, indent=1, default=str) + "\n")
    for name, arms in result.items():
        for arm, m in arms.items():
            if arm.startswith("paired"):
                continue
            final = m["final"]
            print(
                name,
                arm,
                f"lr={m['lr']}",
                f"U0={final['U0']:.2f}",
                f"U2_direct={json.dumps(final['U2_direct'])}",
                f"U3_direct={final['U3_direct']:.2f}",
                f"causal={json.dumps(m['mechanism']['causal'])}",
            )


if __name__ == "__main__":
    main()
