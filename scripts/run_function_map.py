"""knowledge-reasoning-map-v1 (zero training): entity / question / joint shares of every
position x layer in Qwen3-1.7B and Qwen3.5-2B, plus where an answer-supervised write lands
(protocol: docs/development-artifacts/knowledge-reasoning-map-v1/protocol.md).

GPU stage (read-interface image):  python scripts/run_function_map.py --out /batch
Figure stage (plotting image, CPU): python scripts/run_function_map.py --out /batch --figures
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BATCH = "knowledge-reasoning-map-v1"
RUN_ID = "krmap-v1-main"
MODELS = ("qwen3-1.7b", "qwen3.5-2b")
EXAMPLE = ("France", "Japan", "Brazil", "Egypt")  # pre-registered single example


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


def tracker(out, config):
    if not os.environ.get("WANDB_API_KEY"):
        log("W&B disabled (no key)")
        return None
    import wandb

    defaults = json.loads((ROOT / "configs/experiment-tracking-defaults.json").read_text())
    run = wandb.init(
        entity=defaults["entity"],
        project=defaults["project"],
        group=BATCH,
        name=BATCH,
        id=RUN_ID,
        resume="allow",
        dir=str(out),
        config=config,
    )
    log(f"W&B run: {run.url}")
    return run


def run_model(name, out):
    import torch

    from llm_memory_editability import function_map as FM
    from llm_memory_editability import read_interface_model as M

    t0 = time.time()
    model, tok = M.load(name)
    model.requires_grad_(False)
    n_layers = len(M.layers(model))
    entities = FM.single_token_entities(tok)
    ids, start, e_pos, r_pos, labels = FM.encode_grid(tok, entities)
    E, Q, T = ids.shape
    log(f"{name}: {E} single-token entities, {Q} questions, {T} tokens, span from {start}")
    flat = ids.view(E * Q, T)

    texts = FM.generate(model, tok, flat)
    questions = FM.question_texts()
    ok = torch.tensor(
        [
            FM.correct(texts[i * Q + j], e, questions[j][0])
            for i, e in enumerate(entities)
            for j in range(Q)
        ]
    ).view(E, Q)
    known = [e for i, e in enumerate(entities) if bool(ok[i].all())]
    log(f"{name}: accuracy {ok.float().mean():.3f}; all eight correct for {len(known)} entities")

    states, _ = FM.forward_states(model, flat, start)
    P = T - start
    shares = {"all": {}, "known": {}}
    known_idx = torch.tensor([entities.index(e) for e in known])
    for q, x in states.items():
        x = x.view(n_layers, E, Q, P, -1)
        for subset, sel in (("all", None), ("known", known_idx)):
            per_layer = [
                FM.two_way_shares(x[layer] if sel is None else x[layer][sel])
                for layer in range(n_layers)
            ]
            shares[subset][q] = {
                k: torch.stack([s[k] for s in per_layer]).numpy()
                for k in ("entity", "question", "joint", "relation", "phrasing")
            }

    ex = [e for e in EXAMPLE if e in entities]
    ex_idx = [entities.index(e) for e in ex]
    q0 = [j for j, (r, t) in enumerate(questions) if t == 0]
    example = {}
    for q in ("mlp_address", "residual"):
        x = states[q].view(n_layers, E, Q, P, -1)
        mats = {"entity": [], "last": []}
        for layer in range(n_layers):
            for key, p in (("entity", e_pos - start), ("last", P - 1)):
                v = x[layer][ex_idx][:, q0, p].reshape(len(ex_idx) * len(q0), -1)
                mats[key].append(FM.similarity_matrix(v).numpy())
        example[q] = mats
    del states

    targets_map = FM.counterfactual_targets(entities)
    targets = [targets_map[(e, questions[j][0])] for e in entities for j in range(Q)]
    write_share = FM.write_shares(model, tok, flat, targets, start)  # (L, N, P + 1)
    ws = write_share.view(n_layers, E, Q, P + 1)
    write_mean = {
        "all": ws.mean((1, 2)).numpy(),
        "known": ws[:, known_idx].mean((1, 2)).numpy(),
    }

    result = dict(
        model=name,
        revision=M.MODELS[name]["revision"],
        layer_kinds=M.layer_kinds(model),
        n_layers=n_layers,
        entities=entities,
        known=known,
        questions=[f"{r}/{t}" for r, t in questions],
        tokens=T,
        span_start=start,
        entity_position=e_pos - start,
        relation_position=r_pos - start,
        labels=labels,
        accuracy=float(ok.float().mean()),
        accuracy_by_relation={
            r: float(ok[:, [j for j, (rr, _) in enumerate(questions) if rr == r]].float().mean())
            for r in FM.RELATIONS
        },
        shares=shares,
        write_share=write_mean,
        example=dict(entities=ex, questions=[questions[j][0] for j in q0], matrices=example),
        seconds=time.time() - t0,
    )
    write(out / "runs" / name / "summary.json", result)
    write(
        out / "runs" / name / "generations.json",
        [
            dict(entity=e, question=f"{questions[j][0]}/{questions[j][1]}", text=texts[i * Q + j],
                 correct=bool(ok[i, j]), counterfactual=targets[i * Q + j])
            for i, e in enumerate(entities)
            for j in range(Q)
        ],
    )  # fmt: skip
    del model
    torch.cuda.empty_cache()
    return result


def band_means(res, subset="all", q="mlp_address"):
    """Pre-registered readouts: means over the first two thirds / last third of the layers at the
    entity token and at the last prompt token."""
    import numpy as np

    L = res["n_layers"]
    early, late = range(0, (2 * L) // 3), range((2 * L) // 3, L)
    s = {k: np.array(v, dtype=float) for k, v in res["shares"][subset][q].items()}
    e, last = res["entity_position"], len(res["labels"]) - 1
    ws = np.array(res["write_share"][subset], dtype=float)
    out = {}
    for band, rows in (("early", early), ("late", late)):
        rows = list(rows)
        for where, p in (("entity", e), ("last", last)):
            for k in ("entity", "question", "joint"):
                out[f"{where}/{band}/{k}"] = float(np.nanmean(s[k][rows, p]))
        out[f"write/{band}/entity"] = float(np.mean(ws[rows, e]))
        out[f"write/{band}/last"] = float(np.mean(ws[rows, last]))
        out[f"write/{band}/later_reply"] = float(np.mean(ws[rows, -1]))
    return out


def predictions(res):
    """K1-K4 of the protocol, on the MLP address of all entities."""
    import numpy as np

    L = res["n_layers"]
    early, late = list(range(0, (2 * L) // 3)), list(range((2 * L) // 3, L))
    s = {k: np.array(v, dtype=float) for k, v in res["shares"]["all"]["mlp_address"].items()}
    e, last = res["entity_position"], len(res["labels"]) - 1
    ws = np.array(res["write_share"]["all"], dtype=float)
    k1 = int(np.sum(s["question"][early, e] < 0.1))
    k2 = int(np.sum(s["question"][early, last] > s["entity"][early, last]))
    k3 = (float(np.nanmean(s["joint"][late, last])), float(np.nanmean(s["joint"][early, last])))
    k4 = int(np.sum(ws[late, last] + ws[late, -1] > 2 * ws[late, e]))
    return dict(
        K1_entity_token_question_share_below_0p1=f"{k1}/{len(early)} early layers",
        K1=bool(k1 > len(early) / 2),
        K2_last_token_question_above_entity=f"{k2}/{len(early)} early layers",
        K2=bool(k2 > len(early) / 2),
        K3_last_token_joint_late_vs_early=k3,
        K3=bool(k3[0] > k3[1]),
        K4_write_last_plus_reply_above_twice_entity=f"{k4}/{len(late)} late layers",
        K4=bool(k4 > len(late) / 2),
    )


def figures(out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    paths = []
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for name in MODELS:
        res = json.loads((out / "runs" / name / "summary.json").read_text())
        labels = [repr(t)[1:-1] for t in res["labels"]]
        labels[res["entity_position"]] = "E:" + labels[res["entity_position"]]
        labels[res["relation_position"]] = "R:" + labels[res["relation_position"]]
        L = res["n_layers"]
        for q in ("mlp_address", "residual"):
            s = res["shares"]["all"][q]
            panels = [("entity", "Blues"), ("question", "Oranges"), ("joint", "Purples")]
            fig, axes = plt.subplots(1, 4, figsize=(22, 6), sharey=True)
            for ax, (k, cmap) in zip(axes, panels, strict=False):  # fourth axis: write share
                im = ax.imshow(np.array(s[k], dtype=float), origin="lower", aspect="auto",
                               cmap=cmap, vmin=0, vmax=1)  # fmt: skip
                ax.set_title(f"{k} share")
                ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=7)
                fig.colorbar(im, ax=ax, fraction=0.04)
            ws = np.array(res["write_share"]["all"], dtype=float)
            im = axes[3].imshow(ws, origin="lower", aspect="auto", cmap="Greys", vmin=0)
            axes[3].set_title("answer-supervised write share")
            axes[3].set_xticks(range(len(labels) + 1), labels + ["reply"], rotation=90, fontsize=7)
            fig.colorbar(im, ax=axes[3], fraction=0.04)
            axes[0].set_ylabel("layer")
            axes[0].set_yticks(range(0, L, 2))
            fig.suptitle(f"{name}: {q} ({len(res['entities'])} countries x 8 questions)")
            fig.tight_layout()
            p = fig_dir / f"map-{name}-{q}.png"
            fig.savefig(p, dpi=130)
            plt.close(fig)
            paths.append(p)
        ex = res["example"]
        n = len(ex["entities"]) * len(ex["questions"])
        tick = [f"{e[:3]}/{r[:4]}" for e in ex["entities"] for r in ex["questions"]]
        fig, axes = plt.subplots(1, 2, figsize=(13, 6))
        for ax, (key, layer) in zip(
            axes, (("entity", L // 3), ("last", (2 * L) // 3)), strict=True
        ):
            m = np.array(ex["matrices"]["mlp_address"][key][layer], dtype=float)
            ax.imshow(m, cmap="RdBu_r", vmin=-1, vmax=1)
            ax.set_xticks(range(n), tick, rotation=90, fontsize=7)
            ax.set_yticks(range(n), tick, fontsize=7)
            where = "entity token" if key == "entity" else "last prompt token"
            ax.set_title(f"MLP address similarity, {where}, layer {layer}")
        fig.tight_layout()
        p = fig_dir / f"example-{name}.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        paths.append(p)
    paths.append(similarity_curves(out, plt, np, fig_dir))
    return paths


def similarity_curves(out, plt, np, fig_dir):
    """Post-hoc companion of the single example: for the four example countries, the mean
    cosine between MLP addresses of the same entity (different relations) and of the same
    relation (different entities), at every layer, at the entity token and the last token."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharey=True)
    for row, name in enumerate(MODELS):
        res = json.loads((out / "runs" / name / "summary.json").read_text())
        ex = res["example"]["matrices"]["mlp_address"]
        n_e, n_r = len(res["example"]["entities"]), len(res["example"]["questions"])
        ent = np.repeat(np.arange(n_e), n_r)
        rel = np.tile(np.arange(n_r), n_e)
        off = ~np.eye(n_e * n_r, dtype=bool)
        same_e = (ent[:, None] == ent[None]) & off
        same_r = (rel[:, None] == rel[None]) & off
        for col, key in enumerate(("entity", "last")):
            mats = [np.array(m, dtype=float) for m in ex[key]]
            ax = axes[row, col]
            ax.plot([m[same_e].mean() for m in mats], color="#1f77b4", label="same entity")
            ax.plot([m[same_r].mean() for m in mats], color="#ff7f0e", label="same relation")
            ax.axhline(0, color="0.7", lw=0.8)
            where = "entity token" if key == "entity" else "last prompt token"
            ax.set_title(f"{name}: {where}")
            ax.set_xlabel("layer")
    axes[0, 0].set_ylabel("mean cosine (centred MLP address)")
    axes[1, 0].set_ylabel("mean cosine (centred MLP address)")
    axes[0, 0].legend()
    fig.suptitle("France, Japan, Brazil, Egypt x capital, continent, language, currency")
    fig.tight_layout()
    p = fig_dir / "example-similarity-by-layer.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--figures", action="store_true")
    args = ap.parse_args()
    out = args.out
    if args.figures:
        paths = figures(out)
        run = tracker(out, {"batch": BATCH, "stage": "figures"})
        if run:
            import wandb

            run.log({f"figures/{p.stem}": wandb.Image(str(p)) for p in paths})
            run.finish()
        log(f"figures: {[p.name for p in paths]}")
        return
    run = tracker(out, {"batch": BATCH, "models": MODELS, "zero_training": True})
    summary = {}
    for name in MODELS:
        if (out / "runs" / name / "summary.json").exists():
            res = json.loads((out / "runs" / name / "summary.json").read_text())
            log(f"{name}: reuse completed summary")
        else:
            res = run_model(name, out)
            res = json.loads(json.dumps(clean(res)))
        summary[name] = dict(
            entities=len(res["entities"]),
            known=len(res["known"]),
            accuracy=res["accuracy"],
            accuracy_by_relation=res["accuracy_by_relation"],
            bands_all=band_means(res, "all"),
            bands_known=band_means(res, "known"),
            bands_residual=band_means(res, "all", "residual"),
            predictions=predictions(res),
            seconds=res["seconds"],
        )
        log(f"{name}: {summary[name]['predictions']}")
        if run:
            run.summary.update({f"{name}/{k}": v for k, v in summary[name]["bands_all"].items()})
            run.summary.update({f"{name}/{k}": v for k, v in summary[name]["predictions"].items()})
    write(out / "summary.json", summary)
    if run:
        run.finish()


if __name__ == "__main__":
    main()
