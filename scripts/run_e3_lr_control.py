"""E3 on Qwen3-1.7B at Qwen3.5-2B's calibrated rate (1.5e-4), separating rate from model.

Runs the unchanged knowledge-read-interface-v2 E3 stage with a fixed calibration file and its
own W&B run, so the original v2 runs are never resumed or appended to.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import run_read_interface_v2 as v2

BATCH = "e3-learning-rate-control-v1"


class Tracker(v2.Tracker):
    def __init__(self, model_name, fact_set, run):
        self.run = None
        if not os.environ.get("WANDB_API_KEY"):
            return
        import wandb

        root = Path(__file__).resolve().parents[1]
        defaults = json.loads((root / "configs/experiment-tracking-defaults.json").read_text())
        self.run = wandb.init(
            entity=defaults["entity"],
            project=defaults["project"],
            group=BATCH,
            name=f"{BATCH}-{model_name}-{fact_set}",
            dir=str(run),
            config={
                "model": model_name,
                "fact_set": fact_set,
                "e3_lr": 1.5e-4,
                "base": v2.BATCH_NAME,
            },
        )


v2.Tracker = Tracker

if __name__ == "__main__":
    v2.main()
