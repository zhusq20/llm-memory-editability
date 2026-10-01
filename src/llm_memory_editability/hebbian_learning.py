"""Prospective Hebbian study: numerical calibration and auditable common primitives."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/hebbian-learning-v1"
RESULTS = ROOT / "results/hebbian-learning-v1"
ARTIFACTS = ROOT / "docs/development-artifacts/hebbian-learning-v1"
CONFIG_PATH = ROOT / "configs/hebbian-learning-v1.json"


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def config():
    return read_json(CONFIG_PATH)


def first_crossing(values, threshold):
    indices = np.flatnonzero(values <= threshold)
    return int(indices[0]) if len(indices) else None


def mode_prediction(rho, lr, steps, initial):
    factors = np.array([1 - lr * (1 + rho), 1 - lr * (1 - rho)])
    if not np.all((factors > 0) & (factors < 1)):
        raise ValueError("The declared monotone discrete regime is required")
    return factors[None, :, None] ** np.arange(steps + 1)[:, None, None] * initial


def calibration_trajectory(phi, steps, lr, output_dim, checkpoint_nodes):
    """Actual matrix SGD, compared against an independently computed recurrence."""
    phi = np.asarray(phi, dtype=np.float64)
    rho = float(phi[:, 0] @ phi[:, 1])
    targets = np.eye(output_dim, 2, dtype=np.float64)
    weight = np.zeros((output_dim, phi.shape[0]), dtype=np.float64)
    modes = np.empty((steps + 1, 2, output_dim), dtype=np.float64)
    checkpoints = {}
    for step in range(steps + 1):
        error = weight @ phi - targets
        modes[step, 0] = (error[:, 0] + error[:, 1]) / math.sqrt(2)
        modes[step, 1] = (error[:, 0] - error[:, 1]) / math.sqrt(2)
        if step in checkpoint_nodes:
            checkpoints[str(step)] = weight.copy()
        if step < steps:
            weight -= lr * (error @ phi.T)
    predicted = mode_prediction(rho, lr, steps, modes[0])
    norms = np.linalg.norm(modes, axis=2)
    predicted_norms = np.linalg.norm(predicted, axis=2)
    vector_errors = np.linalg.norm(modes - predicted, axis=2)
    eligible = predicted_norms > norms[0] * 1e-6
    max_relative = float(np.max(vector_errors[eligible] / predicted_norms[eligible]))
    small_absolute = float(np.max(vector_errors[~eligible])) if (~eligible).any() else 0.0
    crossings = []
    for i, name in enumerate(["common", "difference"]):
        for epsilon in [0.1, 0.01]:
            factor = 1 - lr * (1 + rho if i == 0 else 1 - rho)
            predicted_step = math.ceil(math.log(epsilon) / math.log(factor))
            observed = first_crossing(norms[:, i], epsilon * norms[0, i])
            crossings.append(
                {
                    "mode": name,
                    "epsilon": epsilon,
                    "predicted_step": predicted_step,
                    "observed_step": observed,
                    "right_censored": observed is None,
                    "pass": (observed is None and predicted_step > steps)
                    or (observed is not None and abs(observed - predicted_step) <= 1),
                }
            )
    summary = {
        "rho": rho,
        "max_relative_vector_error": max_relative,
        "small_error_max_absolute": small_absolute,
        "crossings": crossings,
        "pass": max_relative <= 1e-7 and all(x["pass"] for x in crossings),
    }
    return summary, norms, predicted_norms, checkpoints


def calibrate():
    import torch

    cfg = config()["A"]
    out = RESULTS / "A"
    out.mkdir(parents=True, exist_ok=True)
    source = DATA / "source/hebbian-mlps/src/hebbian/methods/hebbian/model.py"
    spec = importlib.util.spec_from_file_location("locked_hebbian_model", source)
    author = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(author)
    summaries = []
    for kind in ["exact", "bilinear"]:
        for seed in cfg["seeds"]:
            for rho in cfg["rho"]:
                width = cfg["exact_width"] if kind == "exact" else cfg["input_dim"]
                rng = np.random.default_rng(seed)
                rotation = np.linalg.qr(rng.normal(size=(width, width)))[0]
                inputs = rotation[:, :2] @ np.array([[1, rho], [0, np.sqrt(1 - rho**2)]])
                if kind == "bilinear":
                    gen = torch.Generator().manual_seed(seed)
                    a0 = torch.randn(
                        cfg["bilinear_width"], width, generator=gen, dtype=torch.float64
                    )
                    a1 = torch.randn(
                        cfg["bilinear_width"], width, generator=gen, dtype=torch.float64
                    )
                    feature_map = author.BilinearFeatureMap(a0, a1, normalize=True)
                    raw_phi = feature_map(torch.from_numpy(inputs.T)).numpy().T
                else:
                    raw_phi = inputs
                phi = raw_phi / np.linalg.norm(raw_phi, axis=0)
                summary, norms, prediction, weights = calibration_trajectory(
                    phi, cfg["steps"], cfg["lr"], cfg["output_dim"], cfg["checkpoints"]
                )
                run_id = f"{kind}-s{seed}-rho{rho}"
                summary.update(
                    run_id=run_id,
                    kind=kind,
                    seed=seed,
                    input_rho=rho,
                    raw_gram=(raw_phi.T @ raw_phi).tolist(),
                    unit_gram=(phi.T @ phi).tolist(),
                )
                np.savez_compressed(
                    out / f"{run_id}.npz",
                    norms=norms,
                    prediction=prediction,
                    phi=phi,
                    **{f"weight_{k}": v for k, v in weights.items()},
                )
                summaries.append(summary)
                print(run_id, summary["max_relative_vector_error"], summary["pass"], flush=True)
    audit = {
        "completed_at": now(),
        "config_sha256": sha256(CONFIG_PATH),
        "implementation_sha256": sha256(__file__),
        "author_file_sha256": sha256(source),
        "runs": summaries,
        "pass": all(x["pass"] for x in summaries),
    }
    write_json(ARTIFACTS / "A/audit.json", audit)
    if not audit["pass"]:
        raise RuntimeError("Stage A failed numerical acceptance")
    return audit


def q_auc(values, nodes):
    values, nodes = np.asarray(values), np.asarray(nodes)
    integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return integrate(values, nodes, axis=0) / nodes[-1]


def effective_rank(features):
    z = features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-8)
    gram = z @ z.T
    return float(np.trace(gram) ** 2 / np.square(gram).sum())
