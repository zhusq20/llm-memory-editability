#!/usr/bin/env python3
"""Post-observation controls: matched input whitening and content-token subgroups."""

from __future__ import annotations

import numpy as np

from llm_memory_editability.hebbian_future import (
    ART,
    CONFIG,
    DATA,
    RESULTS,
    bootstrap_mean,
    read,
    ridge_predict,
    unit,
    write,
)


def main():
    cfg = read(CONFIG)
    facts = read(DATA / "real-facts.json")
    rows = read(RESULTS / "real/measurements.json")
    n = len(facts)
    refs = rows[::4]
    queries = [r for r in rows if r["view"]]
    dev = np.array([r["role"] == "dev" for r in facts])
    qdev = np.repeat(dev, 3)
    qt = ~qdev
    true = np.array([r["true_token"] for r in refs])
    ref = np.array([r["prediction_token"] for r in refs])
    actual = np.array([r["prediction_token"] for r in queries])
    y = np.array([1 - r["strict_correct"] for r in queries])
    relations = sorted({r["relation_id"] for r in facts})
    b = np.array(
        [
            [
                float(r["strict_correct"]),
                r["margin"],
                r["true_probability"],
                r["top_probabilities"][0],
                r["entropy"],
                len(r["prompt"]),
                *[float(r["relation"] == rel) for rel in relations],
            ]
            for r in refs
        ]
    )
    base = np.c_[np.repeat(b, 3, axis=0), [len(r["prompt"]) for r in queries]]
    baseline = ridge_predict(base[qdev], y[qdev], base, cfg["real"]["prediction_ridge"])
    tokens = sorted(set(ref))
    lookup = {t: i for i, t in enumerate(tokens)}
    onehot = np.zeros((n, len(tokens)))
    onehot[np.arange(n), [lookup[t] for t in ref]] = 1
    own = np.repeat(np.arange(n), 3)
    error_masks = {
        "all_wrong_first_tokens": qt & (actual != np.repeat(true, 3)),
        "wrong_tokens_in_answer_vocabulary": qt
        & (actual != np.repeat(true, 3))
        & np.isin(actual, true),
        "canonical_correct_wrong_first_tokens": qt
        & (actual != np.repeat(true, 3))
        & np.repeat([bool(r["strict_correct"]) for r in refs], 3),
    }
    features = np.load(RESULTS / "real/features.npz")
    outputs = []
    for layer in cfg["real"]["layers"]:
        for kind in ("x", "phi"):
            f = features[f"{kind}-{layer}"]
            reference, query = unit(f[:, 0]), unit(f[:, 1:].reshape(n * 3, -1))
            gram = reference @ reference.T
            values, vectors = np.linalg.eigh(gram)
            cross = query @ reference.T
            for ridge in cfg["real"]["kernel_ridges"]:
                kernel = (cross @ vectors / (np.maximum(values, 0) + ridge)) @ vectors.T
                self_score = kernel[np.arange(n * 3), own]
                other = kernel.copy()
                other[np.arange(n * 3), own] = -np.inf
                best_other = other.max(1)
                chosen = kernel.argmax(1)
                for shuffled in (False, True):
                    bank = onehot.copy()
                    if shuffled:
                        # Preserve relation-wise token frequencies while destroying identity.
                        rng = np.random.default_rng(cfg["seed"] + 99)
                        for relation in relations:
                            indices = np.array(
                                [i for i, r in enumerate(facts) if r["relation_id"] == relation]
                            )
                            bank[indices] = onehot[rng.permutation(indices)]
                    scores = kernel @ bank
                    prediction = np.array(tokens)[scores.argmax(1)]
                    stats = np.c_[
                        self_score,
                        best_other,
                        self_score - best_other,
                        (chosen == own).astype(float),
                        scores.max(1),
                        (prediction == np.repeat(true, 3)).astype(float),
                        (prediction == np.repeat(ref, 3)).astype(float),
                    ]
                    x = np.c_[base, stats]
                    p = ridge_predict(x[qdev], y[qdev], x, cfg["real"]["prediction_ridge"])
                    gain = ((baseline - y) ** 2 - (p - y) ** 2).reshape(n, 3).mean(1)
                    result = {
                        "method": f"{kind}-L{layer}-ridge{ridge}",
                        "shuffled": shuffled,
                        "eval_brier": float(np.mean((p[qt] - y[qt]) ** 2)),
                        "gain": float(gain[~dev].mean()),
                        "gain_ci": bootstrap_mean(
                            gain[~dev], cfg["seed"], cfg["real"]["bootstrap"]
                        ),
                    }
                    for label, mask in error_masks.items():
                        result[label] = {
                            "n": int(mask.sum()),
                            "accuracy": float(np.mean(prediction[mask] == actual[mask])),
                            "copy_accuracy": float(
                                np.mean(np.repeat(ref, 3)[mask] == actual[mask])
                            ),
                        }
                    outputs.append(result)
        print("control layer", layer, flush=True)
    write(
        ART / "real-controls.json",
        {
            "timing": "Added after initial real outcomes were inspected",
            "rows": outputs,
            "scope": (
                "Input/MLP comparisons use identical algorithms; shuffle is one descriptive "
                "relation-preserving permutation, not a permutation test. Answer-vocabulary "
                "tokens are a narrower proxy, not semantic generation scoring."
            ),
        },
    )


if __name__ == "__main__":
    main()
