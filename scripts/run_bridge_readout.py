"""knowledge-read-interface-followup-v1, part 1: bridge readout (zero training).

Reads facts and injected weights of one development run (read-only) and writes one JSON per
method under --out. Completed methods are skipped on restart.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path

import torch

from llm_memory_editability import read_interface_bridge as B
from llm_memory_editability import read_interface_model as M
from llm_memory_editability import read_interface_stages as S

BATCH_NAME = "knowledge-read-interface-followup-v1"


def tracker(model_name, fact_set, out):
    if not os.environ.get("WANDB_API_KEY"):
        return None
    import wandb

    root = Path(__file__).resolve().parents[1]
    defaults = json.loads((root / "configs/experiment-tracking-defaults.json").read_text())
    return wandb.init(
        entity=defaults["entity"],
        project=defaults["project"],
        group=BATCH_NAME,
        name=f"{BATCH_NAME}-bridge-{model_name}-{fact_set}",
        id=f"krif-bridge-{model_name.replace('.', '')}-{fact_set}",
        resume="allow",
        dir=str(out),
        config={"model": model_name, "fact_set": fact_set, "part": "bridge-readout", "training": 0},
    )


def curve(records, source, key, fids):
    by = {}
    for r in records:
        if r["source"] == source and r["fid"] in fids:
            by.setdefault(r["layer"], {}).setdefault(r["fid"], []).append(r[key])
    return {
        layer: sum(sum(v) / len(v) for v in f.values()) / len(f) for layer, f in sorted(by.items())
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=sorted(M.MODELS))
    parser.add_argument("--prev-run", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--fact-set", required=True)
    parser.add_argument("--methods", default="base,m1,m2,m3,m5,m4")
    args = parser.parse_args()
    prev, out = Path(args.prev_run), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    status = S.read(out / "status.json") if (out / "status.json").exists() else {}
    run = tracker(args.model, args.fact_set, out)
    try:
        facts = S.facts_from(prev)
        base, tok = M.load(args.model)
        for p in base.parameters():
            p.requires_grad_(False)
        work = None
        for method in args.methods.split(","):
            target = out / f"{method}.json"
            if target.exists():
                continue
            t0 = time.time()
            if method == "base":
                model, loaded = base, 0
            else:
                if work is None:
                    work, _ = M.load(args.model)
                    for p in work.parameters():
                        p.requires_grad_(False)
                B.reset(work, base)
                loaded = B.load_injected(work, prev / "weights" / f"{method}.pt")
                model = work
            result = B.readout(model, tok, facts, with_fact=method != "base")
            result.update(
                {
                    "method": method,
                    "model": args.model,
                    "fact_set": args.fact_set,
                    "loaded_tensors": loaded,
                    "seconds": time.time() - t0,
                }
            )
            S.write(target, result)
            status.setdefault("methods", {})[method] = {
                "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "seconds": result["seconds"],
            }
            S.write(out / "status.json", status)
            if run:
                main_ids = {f.fid for f in facts if f.split != "calibration"}
                for source in ("bridge", "explicit"):
                    for layer, rate in curve(result["decode"], source, "h", main_ids).items():
                        run.log({f"{method}/decode_h_{source}": rate, "layer": layer})
            torch.cuda.empty_cache()
        status["complete"] = True
        S.write(out / "status.json", status)
    except Exception:
        status["error"] = traceback.format_exc()
        S.write(out / "status.json", status)
        raise
    finally:
        if run:
            run.finish()


if __name__ == "__main__":
    main()
