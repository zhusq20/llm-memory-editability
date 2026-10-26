"""write-mode-readout-v1 (zero training): G2 write-difference maps and G3 in-context read depth on
Qwen3-1.7B / Qwen3.5-2B, plus the G1 table, summary and figures
(protocol: docs/development-artifacts/write-mode-readout-v1/protocol.md).

GPU stage (read-interface image):
    python scripts/run_write_mode_readout.py --model M --krid /krid --out /batch
Summary stage (plotting image, CPU):
    python scripts/run_write_mode_readout.py --summary --krid /krid --out /batch
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BATCH = "write-mode-readout-v1"
MODELS = ("qwen3-1.7b", "qwen3.5-2b")
SETS = {"dev": "runs", "second": "runs-confirm"}
METHODS = ("m1", "m2", "m3", "m5", "m4")
WINDOW = {"qwen3-1.7b": 5, "qwen3.5-2b": 4}


def log(message):
    print(time.strftime("%H:%M:%S"), message, flush=True)


def clean(value):
    import numpy as np

    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if hasattr(value, "tolist") and not isinstance(value, (float, int)):
        return clean(value.tolist())
    if isinstance(value, (np.floating, float)):
        x = float(value)
        return round(x, 6) if math.isfinite(x) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(clean(value), indent=1, ensure_ascii=False))
    tmp.replace(path)


def tracker(out, name, config):
    if not os.environ.get("WANDB_API_KEY"):
        log("W&B disabled (no key)")
        return None
    import wandb

    defaults = json.loads((ROOT / "configs/experiment-tracking-defaults.json").read_text())
    run = wandb.init(
        entity=defaults["entity"],
        project=defaults["project"],
        group=BATCH,
        name=f"{BATCH}-{name}",
        id=f"wmr-v1-{name}",
        resume="allow",
        dir=str(out),
        config=config,
    )
    log(f"W&B run: {run.url}")
    return run


# ---------------------------------------------------------------- GPU stage


def save_g2(path, results, rows):
    import numpy as np

    arrays, index = {}, []
    for r, res in zip(rows, results, strict=True):
        index.append({"fid": r["fact"].fid, "use": r["row"]["use"], "groups": sorted(res)})
    for g in sorted({g for res in results for g in res}):
        for use in sorted({r["row"]["use"] for r in rows}):
            sel = [res[g] for r, res in zip(rows, results, strict=True)
                   if r["row"]["use"] == use and g in res]  # fmt: skip
            fids = [r["fact"].fid for r, res in zip(rows, results, strict=True)
                    if r["row"]["use"] == use and g in res]  # fmt: skip
            if sel:
                arrays[f"{use}|{g}"] = np.stack(sel).astype(np.float32)
                arrays[f"{use}|{g}|fids"] = np.array(fids)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return index


def run_model(name, krid, out, run):
    import torch

    from llm_memory_editability import read_interface_data as D
    from llm_memory_editability import read_interface_model as M
    from llm_memory_editability import write_mode_readout as W

    base, tok = M.load(name)
    base.requires_grad_(False)
    work = None
    previous = []
    for tag, rel in SETS.items():
        src = krid / rel / name
        dst = out / "runs" / name / tag
        facts = [D.Fact(**f) for f in json.loads((src / "facts.json").read_text())["facts"]]
        systems = W.distractor_systems(facts)
        controls = W.control_systems(facts, systems)
        write(dst / "systems.json", {"systems": systems, "controls": controls})

        cities = W.cities_of(facts)
        ent, ans, first = W.entity_vectors(base, tok, cities)
        city_index = {c: i for i, c in enumerate(cities)}
        write(dst / "cities.json", {"cities": cities, "first_tokens": first})

        jobs = [("context", None), ("g3", None)]
        jobs += [(m, src / "weights" / f"{m}.pt") for m in METHODS]
        for method, weights in jobs:
            if method == "g3":
                if (dst / "g3.json").exists():
                    continue
                t0 = time.time()
                items = W.g3_items(tok, facts, systems)
                prefix = f"{name}/{tag}"
                res = W.g3_run(
                    base, tok, items, WINDOW[name], log=lambda m, p=prefix: log(f"{p} {m}")
                )
                res["seconds"] = time.time() - t0
                write(dst / "g3.json", res)
                log(f"{name}/{tag}: G3 done in {res['seconds']:.0f}s")
                if run:
                    run.log({f"{tag}/g3_seconds": res["seconds"]})
                continue
            path = dst / f"g2-{method}.npz"
            if path.exists():
                continue
            t0 = time.time()
            if method == "context":
                rows = W.g2_context_rows(tok, facts, systems, controls)
                results = W.g2_compare(base, base, tok, rows, city_index, ent, ans)
            else:
                if work is None:
                    work, _ = M.load(name)
                    work.requires_grad_(False)
                state = torch.load(weights, map_location="cpu", weights_only=True)
                previous = W.reset_and_load(work, base, state, previous)
                del state
                rows = W.g2_param_rows(tok, facts)
                results = W.g2_compare(base, work, tok, rows, city_index, ent, ans)
            index = save_g2(path, results, rows)
            write(dst / f"g2-{method}.json", {"rows": index, "seconds": time.time() - t0})
            log(f"{name}/{tag}: G2 {method} done in {time.time() - t0:.0f}s")
            if run:
                run.log({f"{tag}/g2_{method}_seconds": time.time() - t0})
            torch.cuda.empty_cache()
        if work is not None and previous:
            previous = W.reset_and_load(work, base, {}, previous)
    write(out / "runs" / name / "complete.json", {"utc": time.time()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODELS)
    ap.add_argument("--krid", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()
    if args.summary:
        from write_mode_readout_summary import summarize

        paths = summarize(args.krid, args.out, ROOT)
        run = tracker(args.out, "summary", {"batch": BATCH, "stage": "summary"})
        if run:
            import wandb

            run.log({f"figures/{p.stem}": wandb.Image(str(p)) for p in paths})
            run.summary.update(json.loads((args.out / "summary.json").read_text())["flat"])
            run.finish()
        return
    config = {"batch": BATCH, "model": args.model, "zero_training": True}
    run = tracker(args.out, args.model, {**config, "window": WINDOW[args.model]})
    t0 = time.time()
    run_model(args.model, args.krid, args.out, run)
    log(f"{args.model}: all stages done in {time.time() - t0:.0f}s")
    if run:
        run.summary["seconds"] = time.time() - t0
        run.finish()


if __name__ == "__main__":
    main()
