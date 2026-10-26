"""knowledge-read-interface-v2: R2/R3 measurement fixes, E1 subject-state write upper bound,
E3 write-form comparison and E4 usage transfer for one model and one fact set.

Facts, screening, released-model evaluations, stage-0 calibration, the M2 weights (E4-0), the M3
teacher replies (E3) and the KL replay pool come from the development batch run of the same model
and fact set. Completed outputs are skipped on restart. The second fact set reuses the E3 learning
rates calibrated on the development set.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import time
import traceback
from pathlib import Path

import torch

from llm_memory_editability import read_interface_bridge as B
from llm_memory_editability import read_interface_data as D
from llm_memory_editability import read_interface_e3 as E3
from llm_memory_editability import read_interface_model as M
from llm_memory_editability import read_interface_stages as S
from llm_memory_editability import read_interface_v2 as V2
from llm_memory_editability import read_interface_v2_data as V

BATCH_NAME = "knowledge-read-interface-v2"
E4_LR = 3e-6
INPUTS = [
    "facts.json",
    "screening.json",
    "eval/base.json",
    "eval/context.json",
    "calibration/swap.json",
    "calibration/knockout.json",
    "eval/m2.json",
    "transplant/m2.json",
]
STAGES = "r2,r3,e3cal,e1,e4,e3"


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
            name=f"{BATCH_NAME}-{model_name}-{fact_set}",
            id=f"kri2-{model_name.replace('.', '')}-{fact_set}",
            resume="allow",
            dir=str(run),
            config={
                "model": model_name,
                "fact_set": fact_set,
                "e1_layers": V2.E1_LAYERS[model_name],
                "e1": {
                    "steps": V2.E1_STEPS,
                    "clamp": V2.E1_CLAMP,
                    "kl": V2.E1_KL_WEIGHT,
                    "step_fraction": V2.E1_STEP_FRACTION,
                    "templates_per_step": V2.E1_TEMPLATES_PER_STEP,
                },
                "e3": {
                    "facts": E3.N_FACTS,
                    "epochs": E3.EPOCHS,
                    "batch": E3.BATCH,
                    "temperature": E3.TEMPERATURE,
                    "lrs": E3.LRS,
                },
                "e4_lr": E4_LR,
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


def log(message):
    print(time.strftime("%H:%M:%S"), message, flush=True)


def done(run, status, name, **info):
    status.setdefault("stages", {})[name] = {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **info,
    }
    S.write(run / "status.json", status)


def e3_order(facts):
    main = sorted([f for f in facts if f.split != "calibration"], key=lambda f: f.fid)
    rng = random.Random(E3.SEED)
    rng.shuffle(main)
    return main[: E3.N_FACTS]


def e4_extra(facts, arm):
    adapt = [f for f in facts if f.split == "adapt"]
    make = V.practice_examples if arm == "e4u" else V.paraphrase_examples
    return [{"user": u, "assistant": a} for f in adapt for u, a in make(f)]


def op_eval(model, tok, facts, context=False):
    rows = []
    for f in facts:
        if f.split == "calibration":
            continue
        for r in V.op_queries(f):
            row = {**r, "split": f.split}
            if context:
                row["system"] = "Fact: " + D.context_statement(f)
            rows.append(row)
    return V2.answer_rows(model, tok, rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=sorted(M.MODELS))
    parser.add_argument("--fact-set", required=True, choices=["dev", "second"])
    parser.add_argument("--prev-run", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--stages", default=STAGES)
    parser.add_argument("--e3-calibration-from", default=None)
    args = parser.parse_args()
    prev, run = Path(args.prev_run), Path(args.run)
    stages = args.stages.split(",")
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
        facts = S.facts_from(run)
        info = S.read(run / "facts.json")

        if "r2" in stages and not (run / "r2" / "distractor.json").exists():
            out = V2.r2(base, tok, facts)
            S.write(run / "r2" / "distractor.json", out)
            done(run, status, "r2", seconds=out["seconds"])
            log("R2 done")

        if "r3" in stages and not (run / "r3" / "natural.json").exists():
            out = V2.r3(base, tok, info["known"])
            S.write(run / "r3" / "natural.json", out)
            done(run, status, "r3", seconds=out["seconds"])
            log("R3 done")

        replies = S.read(prev / "inject" / "m3.json")["teacher_replies"]
        general = S.read(prev / "inject" / "m2.json")["replay_pool"]

        if "e3cal" in stages:
            student = None
            for arm in ("structural", "synaptic"):
                path = run / "e3" / f"calibration-{arm}.json"
                if path.exists():
                    continue
                if student is None:
                    student, _ = M.load(args.model)
                    for p in student.parameters():
                        p.requires_grad_(False)
                    teacher = E3.Teacher(base, tok)
                calib = [f for f in facts if f.split == "calibration"]
                keep = E3.calibration_keep_rows(run, calib)
                lr, sweep = E3.calibrate(
                    base, student, tok, teacher, arm, calib, replies, general, keep, log
                )
                E3.uninstall(student)
                rule = "max U0 + keep, ties -> smaller lr"
                S.write(path, {"arm": arm, "chosen_lr": lr, "sweep": sweep, "rule": rule})
                done(run, status, f"e3cal_{arm}", lr=lr)
                tracker.summary({f"e3/{arm}/lr": lr})
            if student is not None:
                del student, teacher
                torch.cuda.empty_cache()

        if "e1" in stages:
            for layer in V2.E1_LAYERS[args.model]:
                V2.run_e1(base, tok, facts, layer, run / "e1", log)
            done(run, status, "e1")

        if "e4" in stages:
            e4 = run / "e4"
            if not (e4 / "screen.json").exists():
                rows = V.screen_rows(info["values"])
                S.write(e4 / "screen.json", {"rows": V2.answer_rows(base, tok, rows)})
            for tag, ctx in (("base", False), ("context", True)):
                if not (e4 / f"ops-{tag}.json").exists():
                    S.write(e4 / f"ops-{tag}.json", {"records": op_eval(base, tok, facts, ctx)})
            work = None
            if not (e4 / "ops-m2.json").exists():
                work, _ = M.load(args.model)
                B.load_injected(work, prev / "weights" / "m2.pt")
                for p in work.parameters():
                    p.requires_grad_(False)
                S.write(e4 / "ops-m2.json", {"records": op_eval(work, tok, facts)})
            for arm in ("e4u", "e4p"):
                if (e4 / f"ops-{arm}.json").exists():
                    continue
                if work is None:
                    work, _ = M.load(args.model)
                weights = run / "weights" / f"{arm}.pt"
                if not (run / "inject" / f"{arm}.json").exists():
                    B.reset(work, base)
                    extra = e4_extra(facts, arm)
                    summary = S.inject(
                        args.model,
                        run,
                        arm,
                        work,
                        tok,
                        E4_LR,
                        tracker=tracker,
                        teacher=base,
                        extra_examples=extra,
                    )
                    done(
                        run,
                        status,
                        f"inject_{arm}",
                        steps=summary["steps"],
                        stopped=summary["stopped"],
                    )
                else:
                    B.reset(work, base)
                    B.load_injected(work, weights)
                work.eval()
                for p in work.parameters():
                    p.requires_grad_(False)
                if not (run / "eval" / f"{arm}.json").exists():
                    S.evaluate(args.model, run, arm, work, tok)
                if not (run / "transplant" / f"{arm}.json").exists():
                    S.transplant(args.model, run, arm, base, work, tok)
                S.write(e4 / f"ops-{arm}.json", {"records": op_eval(work, tok, facts)})
                done(run, status, f"e4_{arm}")
                log(f"E4 {arm} done")
            if work is not None:
                del work
                torch.cuda.empty_cache()

        if "e3" in stages:
            order = e3_order(facts)
            student = None
            for arm in ("structural", "synaptic"):
                final = run / "e3" / f"{arm}.json"
                if final.exists():
                    continue
                source = Path(args.e3_calibration_from or run / "e3") / f"calibration-{arm}.json"
                while not source.exists():
                    log(f"waiting for {source}")
                    time.sleep(60)
                lr = S.read(source)["chosen_lr"]
                if student is None:
                    student, _ = M.load(args.model)
                    for p in student.parameters():
                        p.requires_grad_(False)
                    teacher = E3.Teacher(base, tok)
                wrappers = E3.install(student, arm)
                progress = run / "e3" / f"{arm}-progress.pt"
                progress.parent.mkdir(parents=True, exist_ok=True)
                state = torch.load(progress, weights_only=False) if progress.exists() else {}
                state["save"] = lambda st, path=progress: torch.save(
                    {k: v for k, v in st.items() if k != "save"}, path
                )

                def after(i, fact, order=order, student=student, arm=arm):
                    out = E3.immediate(student, tok, fact, order[: i + 1])
                    tracker.log({f"e3/{arm}/u0_written": out["u0_written"], "e3/fact": i + 1})
                    return out

                reports = E3.write_sequence(
                    student,
                    tok,
                    teacher,
                    wrappers,
                    arm,
                    lr,
                    order,
                    replies,
                    general,
                    log,
                    after,
                    state,
                )
                tag = f"e3-{arm}"
                if not (run / "eval" / f"{tag}.json").exists():
                    S.evaluate(args.model, run, tag, student, tok)
                unrelated = E3.retention_rows(run, order)
                kept = V2.answer_rows(student, tok, unrelated)
                transplant = E3.transplant_same_layer(base, student, tok, order)
                mech = E3.mechanism(base, student, tok, wrappers, order, unrelated)
                torch.save(
                    [[{k: v.cpu() for k, v in fr.items()} for fr in w.frozen] for w in wrappers],
                    run / "e3" / f"{arm}-adapters.pt",
                )
                n_params = sum(v.numel() for w in wrappers for fr in w.frozen for v in fr.values())
                S.write(
                    final,
                    {
                        "arm": arm,
                        "lr": lr,
                        "order": [f.fid for f in order],
                        "trainable_parameters": n_params,
                        "reports": reports,
                        "unrelated": kept,
                        "transplant": transplant,
                        "mechanism": mech,
                    },
                )
                E3.uninstall(student)
                done(run, status, f"e3_{arm}", lr=lr)
                log(f"E3 {arm} done")

        status["complete"] = all(
            k in status.get("stages", {}) for k in ("e1", "e3_structural", "e3_synaptic")
        )
        S.write(run / "status.json", status)
    except Exception:
        status["error"] = traceback.format_exc()
        S.write(run / "status.json", status)
        raise
    finally:
        tracker.finish()


if __name__ == "__main__":
    main()
