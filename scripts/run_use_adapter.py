"""use-adapter-development-v1: one model, fact sets in order (dev, second).

Per fact set: P-0 (released model + saved E3 Structural writes) is evaluated; P-U and P-P each
start from the released weights with the same frozen writes, train a fixed number of steps
(E4-U's actual count for that model and fact set) and are evaluated with the E3/E4 code. Inputs
come read-only from the knowledge-read-interface-v2 run; completed outputs are skipped on
restart. No model weights are saved.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
import traceback
from collections import defaultdict
from pathlib import Path

import torch

from llm_memory_editability import read_interface_e3 as E3
from llm_memory_editability import read_interface_model as M
from llm_memory_editability import read_interface_stages as S
from llm_memory_editability import read_interface_v2 as V2
from llm_memory_editability import use_adapter as U

INPUTS = ["facts.json", "screening.json", "eval/base.json", "e4/screen.json"]


def log(message):
    print(time.strftime("%H:%M:%S"), message, flush=True)


class Tracker:
    def __init__(self, model_name, sets, out_dir):
        self.run = None
        if not os.environ.get("WANDB_API_KEY"):
            return
        import wandb

        root = Path(__file__).resolve().parents[1]
        defaults = json.loads((root / "configs/experiment-tracking-defaults.json").read_text())
        self.run = wandb.init(
            entity=defaults["entity"],
            project=defaults["project"],
            group=U.BATCH_NAME,
            name=f"{U.BATCH_NAME}-{model_name}",
            id=f"ua1-{model_name.replace('.', '')}",
            resume="allow",
            dir=str(out_dir),
            config={
                "model": model_name,
                "fact_sets": sets,
                "arms": list(U.ARMS),
                "lr": U.LR,
                "half_batch": U.HALF,
                "seed": U.SEED,
                "base": "released model + saved E3 Structural writes (frozen)",
                "steps": "E4-U actual steps per model x fact set",
            },
        )

    def log(self, values):
        if self.run:
            self.run.log(values)

    def summary(self, values):
        if self.run:
            self.run.summary.update(values)

    def finish(self):
        if self.run:
            self.run.finish()


def done(run, status, name, **info):
    status.setdefault("stages", {})[name] = {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **info,
    }
    S.write(run / "status.json", status)


def quick(uses, facts, split):
    """Fact-mean strict U2 (country) correctness on one split, for live tracking only."""
    by = defaultdict(list)
    ids = {f.fid for f in facts if f.split == split}
    for r in uses:
        if r["fid"] in ids and r["use"].startswith("U2"):
            by[(r["mode"], r["fid"])].append(r["strict"])
    out = {}
    for mode in ("direct", "cot"):
        vals = [sum(v) / len(v) for (m, _), v in by.items() if m == mode]
        out[mode] = sum(vals) / len(vals) if vals else None
    return out


def consistency(run, v2_run, order):
    """P-0 regenerates the saved E3 Structural evaluation: agreement of final answers."""
    ids = {f.fid for f in order}
    old = {
        (r["fid"], r["use"], r["mode"]): r
        for r in S.read(v2_run / "eval" / "e3-structural.json")["uses"]
        if r["fid"] in ids
    }
    new = {
        (r["fid"], r["use"], r["mode"]): r
        for r in S.read(run / "eval" / "p0.json")["uses"]
        if r["fid"] in ids
    }
    keys = sorted(set(old) & set(new))
    return {
        "rows": len(keys),
        "same_answer": sum(old[k]["answer"] == new[k]["answer"] for k in keys) / len(keys),
        "same_strict": sum(old[k]["strict"] == new[k]["strict"] for k in keys) / len(keys),
        "strict_saved": sum(old[k]["strict"] for k in keys) / len(keys),
        "strict_p0": sum(new[k]["strict"] for k in keys) / len(keys),
    }


def evaluate(model_name, run, arm, model, tok, facts, order):
    if not (run / "eval" / f"{arm}.json").exists():
        S.evaluate(model_name, run, arm, model, tok)
    if not (run / "ops" / f"{arm}.json").exists():
        S.write(run / "ops" / f"{arm}.json", {"records": U.op_eval(model, tok, facts)})
    if not (run / "unrelated" / f"{arm}.json").exists():
        rows = E3.retention_rows(run, order)
        S.write(run / "unrelated" / f"{arm}.json", {"records": V2.answer_rows(model, tok, rows)})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=sorted(M.MODELS))
    parser.add_argument("--sets", default="dev,second")
    parser.add_argument("--v2-root", required=True)
    parser.add_argument("--out-root", required=True)
    args = parser.parse_args()
    sets = args.sets.split(",")
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    tracker = Tracker(args.model, sets, out_root)
    student = teacher = tok = None
    try:
        for fact_set in sets:
            v2_run = Path(args.v2_root) / fact_set / args.model
            run = out_root / fact_set / args.model
            for name in INPUTS:
                if not (run / name).exists():
                    (run / name).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(v2_run / name, run / name)
            status = S.read(run / "status.json") if (run / "status.json").exists() else {}
            status["inputs_from"] = str(v2_run)
            if all((run / sub / f"{arm}.json").exists()
                   for arm in U.ARMS for sub in ("eval", "ops", "unrelated")):  # fmt: skip
                log(f"{fact_set}: complete, skipped")
                continue
            facts = S.facts_from(run)
            order = U.e3_facts(v2_run)
            e3_info = S.read(v2_run / "e3" / "structural.json")
            e4u = S.read(v2_run / "inject" / "e4u.json")
            steps, replay = e4u["steps"], e4u["replay_pool"]
            frozen = U.load_adapters(v2_run / "e3" / "structural-adapters.pt")
            assert len(frozen[0]) == len(order) == 50
            status["design"] = {
                "e3_lr": e3_info["lr"],
                "e3_facts": len(order),
                "practiced_facts": [f.fid for f in U.practiced(order)],
                "steps": steps,
                "replay_pool": len(replay),
                "examples": {arm: len(U.arm_examples(order, arm)) for arm in U.TRAINED},
            }
            S.write(run / "status.json", status)
            if student is None:
                teacher, tok = M.load(args.model)
                student, _ = M.load(args.model)
                for model in (teacher, student):
                    for p in model.parameters():
                        p.requires_grad_(False)
            E3.uninstall(student)
            U.reset_base(student, teacher)
            U.attach(student, frozen)
            for arm in U.ARMS:
                outputs = ("eval", "ops", "unrelated")
                if all((run / sub / f"{arm}.json").exists() for sub in outputs):
                    continue
                if arm in U.TRAINED:
                    U.reset_base(student, teacher)
                    # weights are not saved: an interrupted arm is retrained from the start
                    for sub in ("train", "eval", "ops", "unrelated"):
                        (run / sub / f"{arm}.json").unlink(missing_ok=True)
                    examples = U.arm_examples(order, arm)

                    def check(model, order=order, tok=tok):
                        return U.u0_teacher_forced(model, tok, order)

                    def track(row, fact_set=fact_set, arm=arm):
                        tracker.log({f"{fact_set}/{arm}/{k}": v for k, v in row.items()})

                    log(f"{fact_set} {arm}: train {steps} steps on {len(examples)} examples")
                    summary = U.train(student, teacher, tok, examples, replay, steps,
                                      log=track, check=check)  # fmt: skip
                    S.write(run / "train" / f"{arm}.json", summary)
                    done(run, status, f"train_{arm}", steps=steps, seconds=summary["seconds"])
                evaluate(args.model, run, arm, student, tok, facts, order)
                torch.cuda.empty_cache()
                uses = S.read(run / "eval" / f"{arm}.json")["uses"]
                q = quick(uses, order, "test")
                tracker.summary({f"{fact_set}/{arm}/test_U2_{m}": v for m, v in q.items()})
                done(run, status, f"eval_{arm}", test_u2=q)
                log(f"{fact_set} {arm}: F_test U2 {q}")
                if arm == "p0":
                    status["p0_vs_saved_e3"] = consistency(run, v2_run, order)
                    S.write(run / "status.json", status)
                    log(f"{fact_set} P-0 vs saved E3: {status['p0_vs_saved_e3']}")
            status["complete"] = True
            S.write(run / "status.json", status)
            torch.cuda.empty_cache()
    except Exception:
        log(traceback.format_exc())
        raise
    finally:
        tracker.finish()


if __name__ == "__main__":
    main()
