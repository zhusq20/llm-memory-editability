"""Development-only predictor fitting and episode-cluster paired inference."""

from __future__ import annotations

import itertools

import numpy as np

from .hebbian_learning import (
    ARTIFACTS,
    DATA,
    RESULTS,
    config,
    now,
    q_auc,
    read_json,
    sha256,
    write_json,
)


def fit_ridge(x, y, alpha):
    mean, scale = x.mean(0), x.std(0)
    scale = np.where(scale > 1e-12, scale, 1)
    z = (x - mean) / scale
    intercept = float(y.mean())
    coef = np.linalg.solve(z.T @ z + alpha * np.eye(z.shape[1]), z.T @ (y - intercept))
    return {
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "coefficients": coef.tolist(),
        "intercept": intercept,
        "alpha": alpha,
        "n_features": x.shape[1],
    }


def predict(model, x):
    x = x[:, : model["n_features"]]
    return ((x - model["mean"]) / model["scale"]) @ np.array(model["coefficients"]) + model[
        "intercept"
    ]


def fact_trajectories(path):
    evaluations = [read_json(p) for p in sorted(path.glob("evaluation-*.json"))]
    nodes = [e["rows"][0]["step"] for e in evaluations]
    by_node = [{(r["case_id"], r["view_id"]): r for r in e["rows"]} for e in evaluations]
    cases = sorted({r["case_id"] for r in evaluations[0]["rows"]})
    result = []
    for case in cases:
        rewrite = [np.mean([rows[case, v]["answer_em"] for v in [1, 2]]) for rows in by_node]
        standard = [rows[case, 0]["answer_em"] for rows in by_node]
        all_correct = [all(rows[case, v]["answer_em"] for v in range(3)) for rows in by_node]
        sustained = next(
            (i for i in range(len(nodes) - 1) if all_correct[i] and all_correct[i + 1]), None
        )
        first = by_node[0][case, 0]
        result.append(
            {
                "case_id": case,
                "q_auc": float(q_auc(rewrite, nodes)),
                "q_gain_auc": float(q_auc(np.array(rewrite) - rewrite[0], nodes)),
                "standard_auc": float(q_auc(standard, nodes)),
                "nodes": nodes,
                "rewrite_accuracy": rewrite,
                "standard_accuracy": standard,
                "sustained_step": nodes[sustained] if sustained is not None else None,
                "sustained_exposures": by_node[sustained][case, 0]["exposures"]
                if sustained is not None
                else None,
                "endpoint_first_success": sustained is None and all_correct[-1],
                "right_censored": sustained is None,
                "relation_id": first["relation_id"],
            }
        )
    return result


def freeze_b():
    from .hebbian_train import frozen_training_sources, run_summary, select_config

    episodes = read_json(DATA / "episodes.json")["B_dev"]
    candidates = []
    for lr in config()["adapt"]["learning_rates"]:
        summaries = [run_summary(RESULTS / f"B/dev-lr{lr:g}-e{i}") for i in range(len(episodes))]
        candidates.append(
            {
                "learning_rate": lr,
                "q_auc": float(np.mean([s["q_auc"] for s in summaries])),
                "keep_damage": float(np.mean([s["keep_damage"] for s in summaries])),
                "episodes": summaries,
            }
        )
    chosen = select_config(candidates, "q_auc", True)
    lr = chosen["learning_rate"]
    x, y, groups = [], [], []
    for episode in range(len(episodes)):
        path = RESULTS / f"B/dev-lr{lr:g}-e{episode}"
        if not (path / "complete.json").exists():
            raise RuntimeError("All B_dev runs must have completion receipts")
        outcomes = {r["case_id"]: r["q_auc"] for r in fact_trajectories(path)}
        for row in read_json(path / "prospective-features.json")["rows"]:
            x.append(row["features"])
            y.append(outcomes[row["case_id"]])
            groups.append(episode)
    x, y, groups = np.array(x), np.array(y), np.array(groups)
    predictors = {}
    for name, width in [("P0", 4), ("P1", 8), ("P2", 10)]:
        scores = {}
        for alpha in config()["statistics"]["ridge_grid"]:
            errors = []
            for group in np.unique(groups):
                train = groups != group
                model = fit_ridge(x[train, :width], y[train], alpha)
                errors.append(np.mean(np.abs(predict(model, x[~train]) - y[~train])))
            scores[alpha] = float(np.mean(errors))
        best_alpha = min(scores, key=scores.get)
        predictors[name] = {**fit_ridge(x[:, :width], y, best_alpha), "cv_mae": scores}
    write_json(
        ARTIFACTS / "B-lock.json",
        {
            "time": now(),
            "status": "locked",
            "learning_rate": lr,
            "lr_candidates": candidates,
            "predictors": predictors,
            "source_hashes": frozen_training_sources(),
            "data_lock_sha256": sha256(ARTIFACTS / "data-lock.json"),
            "fitting_scope": "B_dev only; fold-specific training standardization in episode CV",
        },
    )


def paired_inference(differences):
    differences = np.asarray(differences, dtype=np.float64)
    rng = np.random.default_rng(config()["statistics"]["seed"])
    n = len(differences)
    samples = differences[rng.integers(0, n, (10000, n))].mean(1)
    observed = float(differences.mean())
    signs = np.array(list(itertools.product([-1, 1], repeat=n)))
    p = float(np.mean(np.abs((signs * differences).mean(1)) >= abs(observed) - 1e-14))
    return {
        "mean": observed,
        "ci95": np.quantile(samples, [0.025, 0.975]).tolist(),
        "exact_two_sided_sign_p": p,
        "episode_differences": differences.tolist(),
        "n_episodes": n,
    }


def rankdata(values):
    values = np.asarray(values)
    order = np.argsort(values, kind="stable")
    ranked = np.empty(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        ranked[order[i:j]] = (i + j - 1) / 2
        i = j
    return ranked


def spearman(x, y):
    a, b = rankdata(x), rankdata(y)
    return float(np.corrcoef(a, b)[0, 1]) if a.std() and b.std() else None
