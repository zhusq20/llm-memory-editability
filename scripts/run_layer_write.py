"""knowledge-read-interface-followup-v1, part 2: M2 data written into interface-layer MLPs (m6i)
or downstream-layer MLPs (m6d) only.

Facts, released-model evaluations and stage-0 calibration are copied from the development run, so
the comparison uses the same facts and the same interface band. Each arm: inject -> evaluate ->
transplant (T1-T4) -> bridge readout. Completed outputs are skipped on restart.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
import traceback
from pathlib import Path

import torch
from run_read_interface import evaluate_tag

from llm_memory_editability import read_interface_bridge as B
from llm_memory_editability import read_interface_model as M
from llm_memory_editability import read_interface_stages as S

BATCH_NAME = "knowledge-read-interface-followup-v1"
ARMS = {
    "qwen3-1.7b": {"m6i": list(range(4, 16)), "m6d": list(range(16, 28))},
    "qwen3.5-2b": {"m6i": list(range(6, 15)), "m6d": list(range(15, 24))},
}
LRS = [1e-5, 3e-5]
INPUTS = [
    "facts.json",
    "screening.json",
    "eval/base.json",
    "eval/context.json",
    "calibration/swap.json",
    "calibration/knockout.json",
]


class Tracker:
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
            group=BATCH_NAME,
            name=f"{BATCH_NAME}-layer-write-{model_name}-{fact_set}",
            id=f"krif-write-{model_name.replace('.', '')}-{fact_set}",
            resume="allow",
            dir=str(run),
            config={
                "model": model_name,
                "fact_set": fact_set,
                "part": "layer-write",
                "arms": ARMS[model_name],
                "lrs": LRS,
                "data": "M2 varied single-hop examples + KL replay",
            },
        )

    def log(self, values, step=None):
        if self.run:
            self.run.log(values)

    def summary(self, values):
        if self.run:
            self.run.summary.update(values)

    def finish(self):
        if self.run:
            self.run.finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=sorted(M.MODELS))
    parser.add_argument("--prev-run", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--fact-set", required=True)
    parser.add_argument("--arms", default="m6i,m6d")
    args = parser.parse_args()
    prev, run = Path(args.prev_run), Path(args.run)
    for name in INPUTS:
        if not (run / name).exists():
            (run / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(prev / name, run / name)
    status = S.read(run / "status.json") if (run / "status.json").exists() else {}
    status["inputs_from"] = str(prev)
    tracker = Tracker(args.model, args.fact_set, run)
    try:
        base, tok = M.load(args.model)
        for p in base.parameters():
            p.requires_grad_(False)
        work, _ = M.load(args.model)
        for arm in args.arms.split(","):
            layers = ARMS[args.model][arm]
            if not (run / "inject" / f"{arm}.json").exists():
                for lr in LRS:
                    B.reset(work, base)
                    summary = S.inject(
                        args.model,
                        run,
                        arm,
                        work,
                        tok,
                        lr,
                        tracker=tracker,
                        teacher=base,
                        mlp_layers=layers,
                    )
                    attempt = run / "inject" / f"{arm}-lr{lr:g}.json"
                    shutil.copyfile(run / "inject" / f"{arm}.json", attempt)
                    if summary["stopped"] == "target":
                        break
                status.setdefault("stages", {})[f"inject_{arm}"] = {
                    "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "lr": summary["lr"],
                    "steps": summary["steps"],
                    "stopped": summary["stopped"],
                }
                S.write(run / "status.json", status)
            else:
                B.reset(work, base)
                B.load_injected(work, run / "weights" / f"{arm}.pt")
            work.eval()
            for p in work.parameters():
                p.requires_grad_(False)
            evaluate_tag(args.model, run, arm, work, tok, tracker)
            if not (run / "transplant" / f"{arm}.json").exists():
                S.transplant(args.model, run, arm, base, work, tok)
            if not (run / "bridge" / f"{arm}.json").exists():
                result = B.readout(work, tok, S.facts_from(run))
                result.update({"method": arm, "model": args.model, "fact_set": args.fact_set})
                S.write(run / "bridge" / f"{arm}.json", result)
            status.setdefault("stages", {})[f"done_{arm}"] = {
                "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            }
            S.write(run / "status.json", status)
            torch.cuda.empty_cache()
        status["complete"] = True
        S.write(run / "status.json", status)
    except Exception:
        status["error"] = traceback.format_exc()
        S.write(run / "status.json", status)
        raise
    finally:
        tracker.finish()


if __name__ == "__main__":
    main()
