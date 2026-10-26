"""knowledge-read-interface-development-v1: run the stage-0/1/2 pipeline for one model.

Stages: screen -> eval base/context -> calibrate -> for each method: inject -> eval -> transplant.
Completed stages (their output JSON exists) are skipped on restart. Training stages across
containers serialize on a shared lock file so two models never train on one GPU at once.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
import time
import traceback
from pathlib import Path

import torch

from llm_memory_editability import read_interface_model as M
from llm_memory_editability import read_interface_stages as S

BATCH_NAME = "knowledge-read-interface-development-v1"
LR = {"m1": 3e-6, "m2": 3e-6, "m3": 3e-6, "m5": 1e-5}
WINDOWS = {
    "qwen3-1.7b": [list(range(4, 9)), list(range(8, 13)), list(range(12, 17))],
    "qwen3.5-2b": [list(range(3, 8)), list(range(7, 12)), list(range(11, 16))],
}
LAMBDAS = [5000.0, 15000.0]


class Tracker:
    def __init__(self, model_name, run, suffix="v1"):
        self.run = None
        if not os.environ.get("WANDB_API_KEY"):
            return
        import wandb

        defaults = json.loads(
            (
                Path(__file__).resolve().parents[1] / "configs/experiment-tracking-defaults.json"
            ).read_text()
        )
        self.run = wandb.init(
            entity=defaults["entity"],
            project=defaults["project"],
            group=BATCH_NAME,
            name=f"{BATCH_NAME}-{model_name}-{suffix}",
            config={
                "model": model_name,
                **M.MODELS[model_name],
                "lr": LR,
                "memit_windows": WINDOWS[model_name],
                "memit_lambdas": LAMBDAS,
            },
            dir=str(run),
            resume="allow",
            id=f"kri-{model_name.replace('.', '')}-{suffix}",
        )
        self.offset = 0

    def log(self, values, step=None):
        if self.run:
            self.run.log(values)

    def summary(self, values):
        if self.run:
            self.run.summary.update(values)

    def finish(self):
        if self.run:
            self.run.finish()


@contextlib.contextmanager
def gpu_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def stage(run, name, status):
    status.setdefault("stages", {})[name] = {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }
    S.write(run / "status.json", status)


def fresh(model, model_name):
    """Reset a model in place to the released weights."""
    reference, _ = M.load(model_name, device="cpu")
    with torch.no_grad():
        for (n, p), (m, q) in zip(
            model.named_parameters(), reference.named_parameters(), strict=True
        ):
            assert n == m
            p.copy_(q.to(p.device))
    del reference


def evaluate_tag(model_name, run, tag, model, tok, tracker, context=False):
    if (run / "eval" / f"{tag}.json").exists():
        return
    S.evaluate(model_name, run, tag, model, tok, context=context)
    data = S.read(run / "eval" / f"{tag}.json")
    by = {}
    for r in data["uses"]:
        if r.get("split") == "calibration":
            continue
        key = f"{tag}/{r['use'].split('.')[0]}_{r['mode']}"
        by.setdefault(key, []).append(r["strict"])
    tracker.summary({k: sum(v) / len(v) for k, v in by.items()})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=sorted(M.MODELS))
    parser.add_argument("--run", required=True)
    parser.add_argument("--lock", required=True)
    parser.add_argument("--wikitext", required=True)
    parser.add_argument("--methods", default="m1,m2,m5,m3,m4")
    parser.add_argument("--wandb-suffix", default="v1")
    args = parser.parse_args()
    run = Path(args.run)
    run.mkdir(parents=True, exist_ok=True)
    status = S.read(run / "status.json") if (run / "status.json").exists() else {}
    tracker = Tracker(args.model, run, args.wandb_suffix)
    try:
        base, tok = M.load(args.model)
        if not (run / "facts.json").exists():
            S.screen(args.model, run, base, tok)
            stage(run, "screen", status)
        facts = S.facts_from(run)
        tracker.summary({"facts": len(facts), "subjects": len({f.h for f in facts})})
        evaluate_tag(args.model, run, "base", base, tok, tracker)
        evaluate_tag(args.model, run, "context", base, tok, tracker, context=True)
        stage(run, "eval_base_context", status)
        if not (run / "calibration" / "knockout.json").exists():
            S.calibrate(args.model, run, base, tok)
            stage(run, "calibrate", status)
        work = None
        for method in args.methods.split(","):
            weights = run / "weights" / f"{method}.pt"
            if not (run / "inject" / f"{method}.json").exists():
                if work is None:
                    work, _ = M.load(args.model)
                else:
                    fresh(work, args.model)
                with gpu_lock(Path(args.lock)):
                    if method in ("m4", "m4b"):
                        S.run_memit(
                            args.model,
                            run,
                            work,
                            tok,
                            args.wikitext,
                            WINDOWS[args.model],
                            LAMBDAS,
                            batch=10 if method == "m4b" else None,
                        )
                    else:
                        S.inject(
                            args.model,
                            run,
                            method,
                            work,
                            tok,
                            LR[method],
                            tracker=tracker,
                            teacher=base,
                        )
                stage(run, f"inject_{method}", status)
                injected = work
            else:
                injected, _ = M.load(args.model, weights=weights)
            injected.eval()
            for p in injected.parameters():
                p.requires_grad_(False)
            evaluate_tag(args.model, run, method, injected, tok, tracker)
            if not (run / "transplant" / f"{method}.json").exists():
                S.transplant(args.model, run, method, base, injected, tok)
                stage(run, f"transplant_{method}", status)
            if injected is not work:
                del injected
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
