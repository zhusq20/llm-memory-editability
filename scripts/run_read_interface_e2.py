"""knowledge-read-interface-v2 E2 (user choice (a), 2026-10-20): known capitals with an existing
relation ("twin city") whose value the released model does not know, next to one fictional
relation per subject. Readouts as in the development batch (base/context, M2, MEMIT, T1-T4),
plus the R2 distractor context and the E1 subject-state write at the E1 layers.

Screening, the stage-0 calibration and the MEMIT window/lambda come from the development batch
run of the same model. Completed outputs are skipped on restart.
"""

from __future__ import annotations

import argparse
import shutil
import time
import traceback
from dataclasses import asdict
from pathlib import Path

import torch
from run_read_interface_v2 import Tracker, done, log

from llm_memory_editability import read_interface_bridge as B
from llm_memory_editability import read_interface_model as M
from llm_memory_editability import read_interface_stages as S
from llm_memory_editability import read_interface_v2 as V2
from llm_memory_editability import read_interface_v2_data as V

INPUTS = ["screening.json", "calibration/swap.json", "calibration/knockout.json"]
M2_LR = 3e-6


@torch.no_grad()
def priors(model, tok, known):
    rows = []
    for h in V.e2_candidates(known):
        for i, t in enumerate(V.E2_PRIOR):
            rows.append({"h": h, "i": i, "prompt": t.format(h=h) + V.D.DIRECT, "gold": h,
                         "type": "city", "mode": "direct", "exclude": [h]})  # fmt: skip
    answered = V2.answer_rows(model, tok, rows)
    out = {}
    for r in answered:
        out.setdefault(r["h"], [None] * len(V.E2_PRIOR))[r["i"]] = r["first"]
    return out, answered


def memit_fixed(model_name, run, model, tok, wikitext, chosen):
    """One batch MEMIT edit of all facts with the development batch's window and lambda."""
    facts = S.facts_from(run)
    t0 = time.time()
    moments, n_tokens = S.second_moments(model, tok, chosen["window"], wikitext)
    report = S.memit(
        model_name, run, model, tok, chosen["window"], chosen["lambda"], moments, facts
    )
    acc = S.u0_accuracy(model, tok, facts)
    state = {n: p.detach().cpu() for n, p in model.named_parameters() if "down_proj" in n}
    (run / "weights").mkdir(parents=True, exist_ok=True)
    torch.save(state, run / "weights" / "m4.pt")
    S.write(run / "inject" / "m4.json", {"method": "m4", "chosen": chosen, "sweep": None,
                                         "batch": len(facts), "layers_report": report,
                                         "u0_teacher_forced": acc, "moment_tokens": n_tokens,
                                         "seconds": time.time() - t0})  # fmt: skip


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=sorted(M.MODELS))
    parser.add_argument("--prev-run", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--wikitext", required=True)
    args = parser.parse_args()
    V.register_e2_relations()
    prev, run = Path(args.prev_run), Path(args.run)
    for name in INPUTS:
        if not (run / name).exists():
            (run / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(prev / name, run / name)
    status = S.read(run / "status.json") if (run / "status.json").exists() else {}
    status["inputs_from"] = str(prev)
    tracker = Tracker(args.model, "e2", run)
    try:
        base, tok = M.load(args.model)
        for p in base.parameters():
            p.requires_grad_(False)
        if not (run / "facts.json").exists():
            known = S.read(prev / "facts.json")["known"]
            prior, answered = priors(base, tok, known)
            facts, subjects, values, stable = V.build_e2_facts(known, prior)
            S.write(run / "priors.json", {"first_city": prior, "records": answered,
                                          "stable_dropped": stable})  # fmt: skip
            S.write(run / "facts.json", {"seed": 20261024, "subjects": subjects,
                                         "values": values, "known": known,
                                         "facts": [asdict(f) for f in facts]})  # fmt: skip
            done(run, status, "facts", subjects=len(subjects), dropped=len(stable))
        facts = S.facts_from(run)
        for tag, ctx in (("base", False), ("context", True)):
            if not (run / "eval" / f"{tag}.json").exists():
                S.evaluate(args.model, run, tag, base, tok, context=ctx)
        if not (run / "r2" / "distractor.json").exists():
            S.write(run / "r2" / "distractor.json", V2.r2(base, tok, facts))
        done(run, status, "eval_base_context_r2")
        log("E2 base, context and distractor evaluations done")
        for layer in V2.E1_LAYERS[args.model]:
            V2.run_e1(base, tok, facts, layer, run / "e1", log, variants=("plain",))
        done(run, status, "e1")
        work, _ = M.load(args.model)
        chosen = S.read(prev / "inject" / "m4.json")["chosen"]
        for method in ("m2", "m4"):
            if not (run / "inject" / f"{method}.json").exists():
                B.reset(work, base)
                if method == "m2":
                    S.inject(args.model, run, method, work, tok, M2_LR, tracker=tracker,
                             teacher=base)  # fmt: skip
                else:
                    memit_fixed(args.model, run, work, tok, args.wikitext, chosen)
            else:
                B.reset(work, base)
                B.load_injected(work, run / "weights" / f"{method}.pt")
            work.eval()
            for p in work.parameters():
                p.requires_grad_(False)
            if not (run / "eval" / f"{method}.json").exists():
                S.evaluate(args.model, run, method, work, tok)
            if not (run / "transplant" / f"{method}.json").exists():
                S.transplant(args.model, run, method, base, work, tok)
            done(run, status, f"e2_{method}")
            log(f"E2 {method} done")
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
