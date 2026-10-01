#!/usr/bin/env python3
"""Full-answer, subgroup, and paired-world analysis with all conditions retained."""

from __future__ import annotations

import argparse
import csv
import itertools
import os

import numpy as np
import torch
from run_hebbian_future import evaluate_chain

from llm_memory_editability.bios_model import CausalLM, ModelConfig
from llm_memory_editability.hebbian_followup import ART, CONFIG, DATA, RESULTS, fixed_world
from llm_memory_editability.hebbian_future import (
    ROOT,
    auc,
    bootstrap_mean,
    digest,
    read,
    ridge_predict,
    unit,
    write,
)


def save_csv(name, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (ART / name).open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def softmax(x):
    x = np.exp(x - x.max(-1, keepdims=True))
    return x / x.sum(-1, keepdims=True)


def real_report():
    cfg = read(CONFIG)
    facts = read(DATA / "facts.json")
    rows = read(RESULTS / "real/measurements.json")
    n = len(facts)
    assert [(r["fact"], r["view"]) for r in rows] == list(itertools.product(range(n), range(4)))
    refs = rows[::4]
    queries = [r for r in rows if r["view"]]
    fact_index = np.repeat(np.arange(n), 3)
    dev = np.array([r["role"] == "dev" for r in queries])
    plain = np.array([r["view"] in (1, 2) for r in queries])
    reference_scores = np.array([r["logp_mean"] for r in refs])
    reference_probs = softmax(reference_scores)
    correct = np.array([r["candidates"].index(r["target"]) for r in refs])
    ref_margins = reference_scores[np.arange(n), correct] - np.max(
        np.where(np.eye(4)[correct].astype(bool), -np.inf, reference_scores), 1
    )
    relations = sorted({r["relation"] for r in refs})
    base = np.array(
        [
            [
                float(r["choice_mean"] == r["target"]),
                float(r["choice_sum"] == r["target"]),
                reference_probs[i, correct[i]],
                ref_margins[i],
                r["logp_mean"][correct[i]],
                r["logp_sum"][correct[i]],
                r["answer_tokens"][correct[i]],
                -float(np.sum(reference_probs[i] * np.log(reference_probs[i]))),
                r["entropy"],
                r["top1_probability"],
                float(r["parsed"]["status"] == "recognized"),
                float(r["parsed"]["entity"] == r["target"]),
                len(r["prompt"]),
                *[float(r["relation"] == rel) for rel in relations],
            ]
            for i, r in enumerate(refs)
        ]
    )
    base = np.c_[np.repeat(base, 3, axis=0), [len(r["prompt"]) for r in queries]]
    recognized = np.array([r["parsed"]["status"] == "recognized" for r in queries])
    targets = {
        "candidate_mean_error": (
            np.array([r["choice_mean"] != r["target"] for r in queries], float),
            np.ones(n * 3, bool),
        ),
        "candidate_sum_error": (
            np.array([r["choice_sum"] != r["target"] for r in queries], float),
            np.ones(n * 3, bool),
        ),
        "recognized_entity_error": (
            np.array([r["parsed"]["entity"] != r["target"] for r in queries], float),
            recognized,
        ),
        "unrecognized_generation": ((~recognized).astype(float), np.ones(n * 3, bool)),
    }
    classes = sorted({entity for r in facts for entity in r["candidates"]})
    class_index = {entity: i for i, entity in enumerate(classes)}
    query_candidates = np.array([[class_index[e] for e in r["candidates"]] for r in queries])
    truth_choice = np.array([r["candidates"].index(r["target"]) for r in queries])
    bank = np.zeros((n, len(classes)))
    for i, r in enumerate(refs):
        bank[i, [class_index[e] for e in r["candidates"]]] = reference_probs[i]
    baselines, prediction_rows, agreements = {}, [], []
    stored_predictions = {}

    def measure(name, design):
        for style, style_mask in (("plain", plain), ("wrapped", ~plain)):
            for outcome, (y, eligible) in targets.items():
                train = dev & eligible & style_mask
                test = ~dev & eligible & style_mask
                if train.sum() == 0 or test.sum() == 0:
                    prediction_rows.append(
                        {
                            "method": name,
                            "style": style,
                            "outcome": outcome,
                            "status": "no_eligible_examples",
                            "n_dev": int(train.sum()),
                            "n_eval": int(test.sum()),
                        }
                    )
                    continue
                p = ridge_predict(design[train], y[train], design, cfg["real"]["prediction_ridge"])
                loss = (p - y) ** 2
                subject_loss = np.array(
                    [loss[test & (fact_index == i)].mean() for i in np.unique(fact_index[test])]
                )
                record = {
                    "method": name,
                    "style": style,
                    "outcome": outcome,
                    "n_dev": int(train.sum()),
                    "n_eval": int(test.sum()),
                    "eval_brier": float(subject_loss.mean()),
                    "eval_auc": auc(y[test], p[test]),
                }
                key = style + "/" + outcome
                if name == "canonical_baseline":
                    baselines[key] = loss
                else:
                    differences = np.array(
                        [
                            (baselines[key] - loss)[test & (fact_index == i)].mean()
                            for i in np.unique(fact_index[test])
                        ]
                    )
                    record["brier_gain"] = float(differences.mean())
                    record["ci_low"], record["ci_high"] = bootstrap_mean(
                        differences, cfg["seed"], cfg["real"]["bootstrap"]
                    )
                prediction_rows.append(record)
                stored_predictions[name + "/" + key] = p

    measure("canonical_baseline", base)
    features = np.load(RESULTS / "real/features.npz")
    own = np.arange(n * 3)
    for layer in cfg["real"]["layers"]:
        for kind in ("x", "phi"):
            f = features[f"{kind}-{layer}"]
            reference, query = unit(f[:, 0]), unit(f[:, 1:].reshape(n * 3, -1))
            cross, gram = query @ reference.T, reference @ reference.T
            values, vectors = np.linalg.eigh(gram)
            for ridge in [None, *cfg["real"]["ridges"]]:
                kernel = (
                    cross
                    if ridge is None
                    else (cross @ vectors / (np.maximum(values, 0) + ridge)) @ vectors.T
                )
                for shuffle in [False, True] if ridge == 0.01 else [False]:
                    name = (
                        f"{kind}-L{layer}-"
                        + ("cosine" if ridge is None else f"ridge{ridge}")
                        + ("-shuffled" if shuffle else "")
                    )
                    used_bank = bank.copy()
                    if shuffle:
                        rng = np.random.default_rng(cfg["seed"])
                        for relation in relations:
                            ids = [i for i, r in enumerate(refs) if r["relation"] == relation]
                            used_bank[ids] = bank[rng.permutation(ids)]
                    predicted_scores = (kernel @ used_bank)[own[:, None], query_candidates]
                    predicted_choice = predicted_scores.argmax(1)
                    copy_choice = np.repeat(reference_scores.argmax(1), 3)
                    self_weight = kernel[own, fact_index]
                    other = kernel.copy()
                    other[own, fact_index] = -np.inf
                    competitors = predicted_scores.copy()
                    competitors[own, truth_choice] = -np.inf
                    stats = np.c_[
                        self_weight,
                        other.max(1),
                        self_weight - other.max(1),
                        (kernel.argmax(1) == fact_index).astype(float),
                        predicted_scores[own, truth_choice] - competitors.max(1),
                        (predicted_choice == truth_choice).astype(float),
                        (predicted_choice == copy_choice).astype(float),
                    ]
                    measure(name, np.c_[base, stats])
                    actual_choice = np.array(
                        [r["candidates"].index(r["choice_mean"]) for r in queries]
                    )
                    error = ~dev & plain & (actual_choice != truth_choice)
                    agreement = {
                        "method": name,
                        "n_errors": int(error.sum()),
                        "wrong_answer_agreement": float(
                            (predicted_choice[error] == actual_choice[error]).mean()
                        ),
                        "copy_wrong_answer_agreement": float(
                            (copy_choice[error] == actual_choice[error]).mean()
                        ),
                        "overall_agreement": float(
                            (predicted_choice[~dev & plain] == actual_choice[~dev & plain]).mean()
                        ),
                    }
                    agreements.append(agreement)
        print("real analysis layer", layer, flush=True)
    summary = []
    for view in range(4):
        subset = [r for r in rows if r["view"] == view and r["role"] == "eval"]
        rec = [r for r in subset if r["parsed"]["status"] == "recognized"]
        summary.append(
            {
                "view": view,
                "n": len(subset),
                "candidate_mean_accuracy": float(
                    np.mean([r["choice_mean"] == r["target"] for r in subset])
                ),
                "candidate_sum_accuracy": float(
                    np.mean([r["choice_sum"] == r["target"] for r in subset])
                ),
                "strict_accuracy": float(np.mean([r["strict_correct"] for r in subset])),
                "recognized": len(rec),
                "recognized_correct": sum(r["parsed"]["entity"] == r["target"] for r in rec),
                "recognized_wrong": sum(r["parsed"]["entity"] != r["target"] for r in rec),
                "ambiguous": sum(r["parsed"]["status"] == "ambiguous" for r in subset),
            }
        )
    save_csv("real-prediction.csv", prediction_rows)
    save_csv("real-answer-agreement.csv", agreements)
    write(
        ART / "real-summary.json",
        {"by_view": summary, "predictions": prediction_rows, "agreements": agreements},
    )
    np.savez_compressed(RESULTS / "real/predictions.npz", **stored_predictions)


def chains_report():
    cfg = read(CONFIG)["chains"]
    rows, meta = [], []
    for directory in sorted((RESULTS / "chains").glob("w*-i*-r*-d*")):
        m = read(directory / "complete.json")
        meta.append(m)
        for row in read(directory / "learning.json"):
            rows.append(
                {
                    **{k: m[k] for k in ("world", "init", "rho", "architecture")},
                    "step": row["step"],
                    **row["summary"],
                }
            )
    save_csv("chain-learning.csv", rows)
    metrics = [
        "member_accuracy",
        "root_accuracy",
        "home_accuracy",
        "held_composite",
        "held_common_conflict_composite",
        "held_two_step",
        "held_conflict_home_copy",
    ]
    end = [r for r in rows if r["step"] == cfg["steps"]]
    aggregates, effects = [], []
    for arch, rho in itertools.product(cfg["architectures"], cfg["correlations"]):
        part = [r for r in end if r["architecture"] == arch["name"] and r["rho"] == rho]
        aggregates.append(
            {
                "architecture": arch["name"],
                "rho": rho,
                "n": len(part),
                **{k: float(np.mean([r[k] for r in part])) for k in metrics},
            }
        )
    for arch in cfg["architectures"]:
        for metric in metrics:
            per_world = []
            seed_effects = []
            for world in cfg["worlds"]:
                per_seed = []
                for init in cfg["initializations"]:
                    part = {
                        r["rho"]: r
                        for r in end
                        if r["architecture"] == arch["name"]
                        and r["world"] == world
                        and r["init"] == init
                    }
                    per_seed.append(part[0.25][metric] - part[0.75][metric])
                per_world.append(float(np.mean(per_seed)))
                seed_effects.append(per_seed)
            effects.append(
                {
                    "architecture": arch["name"],
                    "metric": metric,
                    "mean": float(np.mean(per_world)),
                    "world_effects": per_world,
                    "seed_effects": seed_effects,
                    "ci": bootstrap_mean(per_world, 912, 2000),
                }
            )
    interaction = {}
    for metric in metrics:
        shallow = next(
            r for r in effects if r["architecture"] == "d2w135" and r["metric"] == metric
        )
        deep = next(r for r in effects if r["architecture"] == "d4w96" and r["metric"] == metric)
        delta = np.array(shallow["world_effects"]) - deep["world_effects"]
        interaction[metric] = {
            "mean": float(delta.mean()),
            "world_effects": delta.tolist(),
            "ci": bootstrap_mean(delta, 912, 2000),
        }
    write(
        ART / "chain-summary.json",
        {
            "completed": len(meta),
            "aggregates": aggregates,
            "effects": effects,
            "matched_budget_interaction": interaction,
            "parameters": {m["architecture"]: m["parameters"] for m in meta},
            "seconds_sum": sum(m["seconds"] for m in meta),
            "input_tokens": sum(m["input_tokens"] for m in meta),
            "flops_estimate": sum(m["flops_estimate"] for m in meta),
        },
    )
    save_csv("chain-endpoints.csv", aggregates)


def audit():
    torch.set_num_threads(4)
    cfg = read(CONFIG)["chains"]
    checks, endpoints, initial = {}, [], {}
    for name, value in read(ART / "lock.json")["files"].items():
        checks[name] = digest(ART / "source" / name) == value
    directories = sorted((RESULTS / "chains").glob("w*-i*-r*-d*"))
    for directory in directories:
        m = read(directory / "complete.json")
        initial.setdefault((m["world"], m["init"], m["architecture"]), set()).add(m["initial_hash"])
        for name, value in m["files"].items():
            checks[str(directory.relative_to(ROOT) / name)] = digest(directory / name) == value
        cp = torch.load(
            directory / f"model-{cfg['steps']}.pt", map_location="cpu", weights_only=False
        )
        model = CausalLM(ModelConfig(**cp["config"])).eval()
        model.load_state_dict(cp["model"])
        w = fixed_world(m["world"], m["rho"], cfg)
        actual = evaluate_chain(model, w, "cpu")
        expected = read(directory / "learning.json")[-1]
        equal = all(
            actual["rows"][k][field] == expected["rows"][k][field]
            for k in actual["rows"]
            for field in ("prediction", "exact", "correct")
        )
        opt_steps = {int(v["step"]) for v in cp["optimizer"]["state"].values()}
        endpoints.append(
            {
                "name": directory.name,
                "behavior_equal": equal,
                "optimizer_steps_ok": opt_steps == {cfg["steps"]},
            }
        )
    checks["all_108_models"] = len(directories) == 108
    checks["paired_initialization"] = all(len(v) == 1 for v in initial.values())
    for name, value in read(RESULTS / "real/complete.json")["files"].items():
        checks["real/" + name] = digest(RESULTS / "real" / name) == value
    passed = all(checks.values()) and all(
        e["behavior_equal"] and e["optimizer_steps_ok"] for e in endpoints
    )
    write(ART / "audit.json", {"passed": passed, "checks": checks, "endpoints": endpoints})
    assert passed


def plot():
    os.environ.setdefault("MPLCONFIGDIR", str(ART / "matplotlib-cache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c = read(ART / "chain-summary.json")
    r = read(ART / "real-summary.json")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for arch in read(CONFIG)["chains"]["architectures"]:
        rows = [x for x in c["aggregates"] if x["architecture"] == arch["name"]]
        for axis, key in zip(
            axes[:2], ["held_composite", "held_common_conflict_composite"], strict=True
        ):
            axis.plot(
                [x["rho"] for x in rows], [x[key] for x in rows], marker="o", label=arch["name"]
            )
            axis.set_xlabel("Home/HQ agreement in training")
            axis.set_ylim(0, 1)
            axis.legend(fontsize=8)
    axes[0].set_title("All held-out composition")
    axes[1].set_title("Same people AND competing answers")
    rows = [
        x
        for x in r["predictions"]
        if x["style"] == "plain"
        and x["outcome"] == "candidate_mean_error"
        and "ridge0.01" in x["method"]
        and "shuffled" not in x["method"]
    ]
    axes[2].barh(range(len(rows)), [x["brier_gain"] for x in rows])
    axes[2].set_yticks(range(len(rows)), [x["method"] for x in rows], fontsize=8)
    axes[2].axvline(0, color="black", lw=0.5)
    axes[2].set_title("Full-answer Brier improvement")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(ART / f"overview.{ext}", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["real", "chains", "audit", "plot"])
    args = p.parse_args()
    {"real": real_report, "chains": chains_report, "audit": audit, "plot": plot}[args.stage]()
