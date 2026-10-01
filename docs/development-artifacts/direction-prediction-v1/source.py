"""Fit a small, frozen edit-failure predictor using development worlds only."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from llm_memory_editability.bios_data import write_json
from llm_memory_editability.bios_direction import ROOT, digest
from llm_memory_editability.bios_direction_cross import CROSS_ARMS
from llm_memory_editability.bios_direction_formation import TRAIN_ARMS

ART = ROOT / "docs/development-artifacts/direction-prediction-v1"
DEV = ROOT / "results/bios-direction-formation-v1"


def features(learning, person, kind, method, mechanism):
    bystep = {r["step"]: r for r in learning}
    current = bystep[1024]["factors"][person]
    early = bystep[128]["factors"][person]
    nll = current["new_target_nll"] - (0.5 * current["margin_ab"][0] if kind == "coherent" else 0.0)
    kernel = np.array(current["kernel"])
    x = [float(kind == "independent")] + [float(method == m) for m in CROSS_ARMS[1:]]
    x += [
        np.log(max(nll, 1e-8)),
        np.linalg.norm(current["margin_ab"]),
        np.log(max(np.trace(kernel), 1e-8)),
    ]
    if mechanism:
        for f in (early, current):
            x += [
                np.log(max(f["ratio"], 1e-10)),
                f["mixing"] / max(f["common"] + f["difference"], 1e-10),
                f["feature_cos"],
                f["downstream_cos"],
                f["gate_cos"],
                np.log(max(f["relation_input_difference"], 1e-8)),
                np.log(max(f["gate_difference"], 1e-8)),
            ]
    return np.array(x, dtype=np.float64)


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -40, 40)))


def fit(x, y):
    mean = x.mean(0)
    scale = x.std(0)
    scale[scale < 1e-8] = 1
    z = np.column_stack((np.ones(len(x)), (x - mean) / scale))
    w = np.zeros(z.shape[1])
    penalty = np.eye(z.shape[1]) * 0.01
    penalty[0, 0] = 0
    for _ in range(100):
        p = sigmoid(z @ w)
        h = (z.T * (p * (1 - p))) @ z / len(x) + penalty + 1e-8 * np.eye(len(w))
        g = z.T @ (p - y) / len(x) + penalty @ w
        step = np.linalg.solve(h, g)
        w -= step
        if np.linalg.norm(step) < 1e-9:
            break
    return dict(mean=mean.tolist(), scale=scale.tolist(), weight=w.tolist(), regularization=0.01)


def predict(model, x):
    z = (np.array(x) - np.array(model["mean"])) / np.array(model["scale"])
    return sigmoid(np.column_stack((np.ones(len(z)), z)) @ np.array(model["weight"]))


def load_development():
    records = []
    for directory in sorted(DEV.glob("world-*/*")):
        if not (directory / "learning.json").exists():
            continue
        learning = json.loads((directory / "learning.json").read_text())
        for method in CROSS_ARMS:
            rows = json.loads((directory / "edits" / method / "metrics.json").read_text())
            for r in rows:
                meta = r["task"]["metadata"]
                # Fixed attempt budget; keep solver failures as failures.
                point = max(
                    (t for t in r["timeline"] if t["attempt"] <= 256), key=lambda t: t["attempt"]
                )
                records.append(
                    dict(
                        world=meta["world"],
                        seed=meta["seed"],
                        person=meta["person"],
                        kind=meta["kind"],
                        train_arm=meta["train_arm"],
                        method=method,
                        failed=not point["joint"],
                        final_joint=r["timeline"][-1]["joint"],
                        baseline=features(learning, meta["person"], meta["kind"], method, False),
                        mechanism=features(learning, meta["person"], meta["kind"], method, True),
                    )
                )
    assert len(records) == 1344
    return records


def run():
    records = load_development()
    y = np.array([r["failed"] for r in records], float)
    models = {}
    cv = []
    for name in ("baseline", "mechanism"):
        x = np.stack([r[name] for r in records])
        models[name] = fit(x, y)
        for world in (4000, 4001, 4002, 4003):
            held = np.array([r["world"] == world for r in records])
            fitted = fit(x[~held], y[~held])
            prob = predict(fitted, x[held])
            cv.append(
                dict(
                    predictor=name,
                    world=world,
                    brier=float(np.mean((prob - y[held]) ** 2)),
                    n=int(held.sum()),
                )
            )
    arm_scores = {
        arm: np.mean(
            [
                r["final_joint"]
                for r in records
                if r["train_arm"] == arm
                and r["method"] == "func-soft-adam"
                and r["kind"] == "independent"
            ]
        )
        for arm in TRAIN_ARMS
    }
    candidates = [a for a in TRAIN_ARMS if a != "baseline"]
    chosen = max(
        candidates, key=lambda a: (arm_scores[a] - arm_scores["baseline"], -candidates.index(a))
    )
    created = datetime.now(timezone.utc).isoformat()
    payload = dict(
        created=created,
        models=models,
        leave_world_out=cv,
        arm_scores=arm_scores,
        selected_training_arm=chosen,
        independent_worlds=list(range(5000, 5008)),
        seeds=[0, 1],
        train_arms=["baseline", chosen],
        methods=list(CROSS_ARMS),
        target="E/local/U joint failure by attempt 256; solver failures retained.",
        method_policy="Choose lowest predicted failure probability; ties use method list order.",
        fixed_method=min(
            CROSS_ARMS, key=lambda m: np.mean([r["failed"] for r in records if r["method"] == m])
        ),
        selection=(
            "Intervention chosen by independent-edit final joint success "
            "under fixed AdamW on four development worlds."
        ),
        files={str(Path(__file__).relative_to(ROOT)): digest(__file__)},
    )
    write_json(ART / "predictor.json", payload)
    (ART / "source.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps(dict(cv=cv, arm_scores=arm_scores, selected=chosen)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["fit"])
    parser.parse_args()
    run()
