"""bios-write-address-v1 (zero training): name versus context as the address of bioS attributes,
on the saved bioS-original pretraining checkpoints (protocol:
docs/development-artifacts/bios-write-address-v1/protocol.md).

GPU stage (read-interface image): python scripts/run_bios_address.py --old /old --out /batch
Figure stage (plotting image):
    python scripts/run_bios_address.py --old /old --out /batch --figures
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BATCH = "bios-write-address-v1"
RUN_ID = "bwa-v1-main"
WORLDS = (142701, 142702)
CONDITIONS = ("S", "M", "MP")
READ_PASSES = (30, 90, 180, 360, 540)
WRITE_PASSES = (0, 30, 90, 180, 360, 540)
OWN = {"S": "single", "M": "canonical", "MP": "permuted"}  # each model's training texts
SOURCE = "data/bios-source/data-synthetic-pretrain/Capo-bioS-bioR/Capo-bioS-bioR.py"


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
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(clean(value), indent=1, ensure_ascii=False))
    tmp.replace(path)


def tracker(out, config, run_id=RUN_ID):
    if not os.environ.get("WANDB_API_KEY"):
        log("W&B disabled (no key)")
        return None
    import wandb

    defaults = json.loads((ROOT / "configs/experiment-tracking-defaults.json").read_text())
    run = wandb.init(
        entity=defaults["entity"],
        project=defaults["project"],
        group=BATCH,
        name=BATCH if run_id == RUN_ID else run_id,
        id=run_id,
        resume="allow",
        dir=str(out),
        config=config,
    )
    log(f"W&B run: {run.url}")
    return run


def checkpoint(old, world, condition, n):
    return (
        old / "results/bios-original-development-v1" / f"world-{world}" / "init-1427" / condition
        / "pretrain" / f"pass-{n:03d}.pt"
    )  # fmt: skip


def read_stage(world_obj, model, rows):
    from llm_memory_editability import bios_address as BA

    scored = BA.score(world_obj, rows, BA.greedy(model, rows))
    agg = {}
    for p in BA.PROMPTS:
        agg[p] = {}
        for attr in [None, *BA.ATTRS[1:]]:
            sel = [s for r, s in zip(rows, scored, strict=True)
                   if r["prompt"] == p and (attr is None or r["attr"] == attr)]  # fmt: skip
            key = attr or "all"
            agg[p][key] = {w: sum(s[w] for s in sel) / len(sel) for w in ("a", "b", "c")} | {
                "n": len(sel)
            }
    return agg, scored


def write_stage(model, rows_by_set, per_batch=60):
    import torch

    from llm_memory_editability import bios_address as BA

    out = {}
    for name, rows in rows_by_set.items():
        shares, slots, attrs, cos_sum, n_facts = [], [], [], 0, 0
        for s in range(0, len(rows), per_batch):
            part = rows[s : s + per_batch]
            k, d, lab = BA.write_batch(model, part)
            shares.append(BA.group_shares(k, d, lab).cpu())  # (L, B, G)
            c, n = BA.cross_variant_cosine(k, d, lab, part)
            cos_sum = cos_sum + c.cpu() * n
            n_facts += n
            slots += [r["slot"] for r in part]
            attrs += [r["attr"] for r in part]
            del k, d, lab
        sh = torch.cat(shares, 1)  # (L, N, G)
        slots_t = torch.tensor(slots)
        out[name] = dict(
            share_by_layer={g: sh[:, :, i].mean(1) for i, g in enumerate(BA.GROUPS)},
            share_mean={g: float(sh[:, :, i].mean()) for i, g in enumerate(BA.GROUPS)},
            share_by_slot={
                str(j): {g: float(sh[:, slots_t == j, i].mean()) for i, g in enumerate(BA.GROUPS)}
                for j in range(6)
                if bool((slots_t == j).any())
            },
            cross_variant_cosine=dict(
                name=(cos_sum / max(n_facts, 1))[:, 0], value=(cos_sum / max(n_facts, 1))[:, 1]
            ),
            facts=n_facts,
            rows=len(rows),
        )
    return out


def run(old, out, n_people, n_write, worlds=WORLDS):
    import torch

    from llm_memory_editability import bios_address as BA

    device = "cuda"
    data = old / "data/bios-original-v1"
    results = {}
    for world in worlds:
        w = BA.World(data / f"world-{world}", old / SOURCE, data / "tokenizer/tokenizer.json")
        people = list(range(n_people))
        rows = BA.read_prompts(w, people)
        grad_people = list(range(n_write))
        rows_by_set = {
            "canonical": BA.biography_rows(w, grad_people, range(5), permute=False),
            "permuted": BA.biography_rows(w, grad_people, range(5), permute=True),
            "single": BA.biography_rows(w, grad_people, [0], permute=False),
        }
        log(f"world {world}: {len(rows)} read prompts, write rows "
            f"{ {k: len(v) for k, v in rows_by_set.items()} }")  # fmt: skip
        for cond in CONDITIONS:
            for n in sorted(set(READ_PASSES) | set(WRITE_PASSES)):
                key = f"{world}/{cond}/{n:03d}"
                path = out / "runs" / str(world) / cond / f"pass-{n:03d}.json"
                if path.exists():
                    results[key] = json.loads(path.read_text())
                    continue
                t0 = time.time()
                model = BA.load_model(checkpoint(old, world, cond, n), device)
                res = {}
                if n in READ_PASSES:
                    res["read"], scored = read_stage(w, model, rows)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        [dict(person=r["person"], partner=r["partner"], attr=r["attr"],
                              prompt=r["prompt"], **s) for r, s in zip(rows, scored, strict=True)],
                        path.with_suffix(".rows.pt"),
                    )  # fmt: skip
                if n in WRITE_PASSES:
                    res["write"] = write_stage(model, rows_by_set)
                res["seconds"] = time.time() - t0
                write(path, res)
                results[key] = json.loads(json.dumps(clean(res)))
                if "read" in res:
                    r = res["read"]
                    log(
                        f"{key}: conflict A/B {r['conflict']['all']['a']:.3f}/"
                        f"{r['conflict']['all']['b']:.3f}, context-only A "
                        f"{r['context_only']['all']['a']:.3f}, name-only B "
                        f"{r['name_only']['all']['b']:.3f}, original A "
                        f"{r['original']['all']['a']:.3f}"
                    )
                del model
                torch.cuda.empty_cache()
    return results


def predictions(results):
    import numpy as np

    out = {}
    for world in WORLDS:
        s, mp = (results[f"{world}/{c}/540"]["read"] for c in ("S", "MP"))
        out[f"{world}/B1_S_conflict_follows_context"] = (
            s["conflict"]["all"]["a"] > s["conflict"]["all"]["b"]
        )
        out[f"{world}/B1_MP_conflict_follows_name"] = (
            mp["conflict"]["all"]["b"] > mp["conflict"]["all"]["a"]
        )
        out[f"{world}/B2_context_only_S_above_MP"] = (
            s["context_only"]["all"]["a"] > mp["context_only"]["all"]["a"]
        )
        out[f"{world}/B3_name_only_MP_above_S"] = (
            mp["name_only"]["all"]["b"] > s["name_only"]["all"]["b"]
        )
        own = {
            (c, n): results[f"{world}/{c}/{n:03d}"]["write"][OWN[c]]["share_mean"]["name"]
            for c in ("S", "MP")
            for n in (90, 540)
        }
        for n in (540, 90):
            key = f"{world}/B4_name_write_share_MP_above_S_pass{n:03d}"
            out[key] = own[("MP", n)] > own[("S", n)]
        for c in CONDITIONS:
            cv = results[f"{world}/{c}/540"]["write"]["canonical"]["cross_variant_cosine"]
            name, val = np.array(cv["name"], float), np.array(cv["value"], float)
            out[f"{world}/B5_{c}_name_cosine_above_value_layers"] = (
                f"{int(np.sum(name > val))}/{len(name)}"
            )
            out[f"{world}/B5_{c}"] = bool(np.sum(name > val) > len(name) / 2)
    return out


def figures(out, results):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    colors = {"S": "#4C72B0", "M": "#DD8452", "MP": "#55A868"}
    # 1. conflict outcome at the endpoint, per condition, both worlds.
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2), sharey=True)
    panels = [
        ("original", "a", "original: A's value"),
        ("conflict", "b", "conflict: B's (name)"),
        ("conflict", "a", "conflict: A's (context)"),
        ("context_only", "a", "context only: A's"),
    ]
    for ax, (p, who, title) in zip(axes, panels, strict=True):
        for i, c in enumerate(("S", "M", "MP")):
            vals = [results[f"{w}/{c}/540"]["read"][p]["all"][who] for w in WORLDS]
            ax.bar(i, np.mean(vals), color=colors[c])
            ax.scatter([i] * len(vals), vals, color="k", s=12, zorder=3)
        ax.set_xticks(range(3), ["S", "M", "MP"])
        ax.set_title(title)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("fraction of greedy continuations")
    fig.tight_layout()
    p = fig_dir / "read-address-endpoint.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    paths.append(p)
    # 2. trajectory of conflict -> name over passes.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, (p, who, title) in zip(axes, [("conflict", "b", "conflict: follows name"),
                                          ("conflict", "a", "conflict: follows context"),
                                          ("name_only", "b", "name only: B's value")],
                                   strict=True):  # fmt: skip
        for c in ("S", "M", "MP"):
            ys = [np.mean([results[f"{w}/{c}/{n:03d}"]["read"][p]["all"][who] for w in WORLDS])
                  for n in READ_PASSES]  # fmt: skip
            ax.plot(READ_PASSES, ys, marker="o", color=colors[c], label=c)
        ax.set_title(title)
        ax.set_xlabel("pretraining pass")
    axes[0].legend()
    fig.tight_layout()
    p = fig_dir / "read-address-trajectory.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    paths.append(p)
    # 3. write shares by layer (own texts) and cross-variant cosine.
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    own = OWN
    for c in ("S", "M", "MP"):
        for g, ls in (("name", "-"), ("value", "--")):
            ys = np.mean([results[f"{w}/{c}/540"]["write"][own[c]]["share_by_layer"][g]
                          for w in WORLDS], 0)  # fmt: skip
            axes[0].plot(ys, ls, color=colors[c], label=f"{c} {g}")
        for g, ls in (("name", "-"), ("value", "--")):
            cv = [results[f"{w}/{c}/540"]["write"]["canonical"]["cross_variant_cosine"][g]
                  for w in WORLDS]  # fmt: skip
            ys = np.array(cv, dtype=float).mean(0)  # null (zero name write, last layer) -> NaN
            axes[1].plot(ys, ls, color=colors[c], label=f"{c} {g}")
        ys = [np.mean([results[f"{w}/{c}/{n:03d}"]["write"][own[c]]["share_mean"]["name"]
                       for w in WORLDS]) for n in WRITE_PASSES]  # fmt: skip
        axes[2].plot(WRITE_PASSES, ys, marker="o", color=colors[c], label=c)
    axes[0].set_title("write share by layer (own texts, pass 540)")
    axes[1].set_title("cross-variant cosine of the write (pass 540)")
    axes[2].set_title("name-group write share over training")
    for ax in axes:
        ax.legend(fontsize=7)
    axes[0].set_xlabel("layer")
    axes[1].set_xlabel("layer")
    axes[2].set_xlabel("pretraining pass")
    fig.tight_layout()
    p = fig_dir / "write-address.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    paths.append(p)
    return paths


def collect(out):
    results = {}
    for world in WORLDS:
        for cond in CONDITIONS:
            for n in sorted(set(READ_PASSES) | set(WRITE_PASSES)):
                path = out / "runs" / str(world) / cond / f"pass-{n:03d}.json"
                results[f"{world}/{cond}/{n:03d}"] = json.loads(path.read_text())
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--people", type=int, default=1000)
    ap.add_argument("--write-people", type=int, default=128)
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--worlds", type=int, nargs="+", default=list(WORLDS),
                    help="helper containers compute a subset; the full run reuses their files")
    args = ap.parse_args()
    if args.figures:
        results = collect(args.out)
        paths = figures(args.out, results)
        run_ = tracker(args.out, {"batch": BATCH, "stage": "figures"})
        if run_:
            import wandb

            run_.log({f"figures/{p.stem}": wandb.Image(str(p)) for p in paths})
            run_.finish()
        log(f"figures: {[p.name for p in paths]}")
        return
    run_ = tracker(
        args.out,
        {
            "batch": BATCH,
            "people": args.people,
            "write_people": args.write_people,
            "worlds": args.worlds,
            "zero_training": True,
        },
        RUN_ID if tuple(args.worlds) == WORLDS else f"{RUN_ID}-w{'-'.join(map(str, args.worlds))}",
    )
    results = run(args.old, args.out, args.people, args.write_people, tuple(args.worlds))
    if tuple(args.worlds) != WORLDS:
        log("subset run complete")
        return
    pred = predictions(results)
    endpoint = {
        f"{w}/{c}": {p: results[f"{w}/{c}/540"]["read"][p]["all"] for p in
                     ("original", "conflict", "context_only", "name_only")}
        for w in WORLDS for c in CONDITIONS
    }  # fmt: skip
    write(args.out / "summary.json", dict(predictions=pred, endpoint=endpoint))
    log(json.dumps(pred, indent=1))
    if run_:
        run_.summary.update({k: v for k, v in pred.items()})
        for key, v in endpoint.items():
            for p, s in v.items():
                run_.summary.update({f"{key}/{p}/{w}": s[w] for w in ("a", "b", "c")})
        run_.finish()


if __name__ == "__main__":
    main()
