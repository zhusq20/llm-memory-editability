"""World-level tests and prospective prediction scoring after all worlds finish."""

import itertools
import json
from datetime import datetime

import numpy as np
import torch
from report_bios_direction import rows_for, summarize

from llm_memory_editability.bios_data import write_json
from llm_memory_editability.bios_direction import ROOT, digest

WORLD_IDS = list(range(5000, 5008))


def world_test(difference):
    x = np.asarray(difference, dtype=float)
    assert x.shape == (8,)
    observed = abs(x.mean())
    signs = np.array(list(itertools.product((-1.0, 1.0), repeat=8)))
    p = float(np.mean(np.abs((signs * x).mean(1)) >= observed - 1e-12))
    rng = np.random.default_rng(271526)
    boot = x[rng.integers(0, 8, size=(20000, 8))].mean(1)
    return dict(
        effect=float(x.mean()),
        world_differences=x.tolist(),
        p_two_sided=p,
        interval95=np.quantile(boot, [0.025, 0.975]).tolist(),
        unit="world",
    )


def holm_two(pvalues):
    order = np.argsort(pvalues)
    adjusted = np.zeros(2)
    running = 0.0
    for position, index in enumerate(order):
        running = max(running, (2 - position) * pvalues[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def key(meta, method):
    return (meta["world"], meta["seed"], meta["train_arm"], meta["person"], meta["kind"], method)


def confirm():
    root = ROOT / "results/bios-direction-confirm-v1"
    out = ROOT / "docs/development-artifacts/direction-confirm-v1"
    summarize(root, out, 768)
    predictor = json.loads(
        (ROOT / "docs/development-artifacts/direction-prediction-v1/predictor.json").read_text()
    )
    predictions = {}
    prediction_files = list(root.glob("world-*/*/predictions-before-edit.json"))
    assert len(prediction_files) == 32
    for p in prediction_files:
        data = json.loads(p.read_text())
        assert data["predictor_hash"] == digest(
            ROOT / "docs/development-artifacts/direction-prediction-v1/predictor.json"
        )
        assert data["parent_hash"] == digest(p.parent / "model-1024.pt")
        created = datetime.fromisoformat(data["created"]).timestamp()
        assert created < min(m.stat().st_mtime for m in (p.parent / "edits").glob("*/metrics.json"))
        for row in data["rows"]:
            predictions[key(row, row["method"])] = row
    rows = []
    for _path, _index, r in rows_for(root):
        meta = r["task"]["metadata"]
        point = max((t for t in r["timeline"] if t["attempt"] <= 256), key=lambda t: t["attempt"])
        rows.append(
            dict(
                **meta,
                method=r["arm"],
                final_joint=float(r["timeline"][-1]["joint"]),
                joint256=float(point["joint"]),
                predictions=predictions[key(meta, r["arm"])],
            )
        )
    assert len(predictions) == len(rows) == 768
    assert sorted({r["world"] for r in rows}) == WORLD_IDS

    def means(method, field="final_joint"):
        return np.array(
            [
                np.mean([r[field] for r in rows if r["world"] == w and r["method"] == method])
                for w in WORLD_IDS
            ]
        )

    reference = means("func-soft-adam")
    tests = []
    for method in ("func-hard-gn", "func-soft-gn"):
        tests.append(
            dict(method=method, reference="func-soft-adam", **world_test(means(method) - reference))
        )
    corrected = holm_two([x["p_two_sided"] for x in tests])
    for t, p in zip(tests, corrected, strict=True):
        t["p_holm"] = p
    brier = {
        name: np.array(
            [
                np.mean(
                    [
                        (r["predictions"][name] - (1 - r["joint256"])) ** 2
                        for r in rows
                        if r["world"] == w
                    ]
                )
                for w in WORLD_IDS
            ]
        )
        for name in ("baseline", "mechanism")
    }
    pred_test = world_test(brier["baseline"] - brier["mechanism"])
    pred_test.update(
        baseline_brier=float(brier["baseline"].mean()),
        mechanism_brier=float(brier["mechanism"].mean()),
        positive_means_mechanism_better=True,
    )
    choices = {name: [] for name in ("baseline", "mechanism", "fixed")}
    groups = {key(r, "")[:-1] for r in rows}
    for group in groups:
        candidates = [r for r in rows if key(r, "")[:-1] == group]
        assert len(candidates) == 4
        for name in ("baseline", "mechanism"):
            chosen = min(
                candidates,
                key=lambda r: (r["predictions"][name], predictor["methods"].index(r["method"])),
            )
            choices[name].append(
                dict(world=group[0], success=chosen["joint256"], method=chosen["method"])
            )
        fixed = next(r for r in candidates if r["method"] == predictor["fixed_method"])
        choices["fixed"].append(
            dict(world=group[0], success=fixed["joint256"], method=fixed["method"])
        )
    policy = {
        name: np.array(
            [np.mean([r["success"] for r in data if r["world"] == w]) for w in WORLD_IDS]
        )
        for name, data in choices.items()
    }
    policy_test = world_test(policy["mechanism"] - policy["fixed"])
    policy_test.update(
        rates={k: float(v.mean()) for k, v in policy.items()},
        fixed_method=predictor["fixed_method"],
    )
    training = {
        arm: np.array(
            [
                np.mean(
                    [
                        r["final_joint"]
                        for r in rows
                        if r["world"] == w
                        and r["train_arm"] == arm
                        and r["method"] == "func-soft-adam"
                        and r["kind"] == "independent"
                    ]
                )
                for w in WORLD_IDS
            ]
        )
        for arm in predictor["train_arms"]
    }
    training_test = world_test(training[predictor["selected_training_arm"]] - training["baseline"])
    training_test.update(
        arm=predictor["selected_training_arm"],
        rates={k: float(v.mean()) for k, v in training.items()},
    )
    write_json(
        out / "statistics.json",
        dict(
            primary=tests,
            prediction_secondary=pred_test,
            policy_secondary=policy_test,
            training_secondary=training_test,
            all_predictions_precede_edits=True,
            worlds=WORLD_IDS,
            statistical_unit="world; seeds/entities/conditions nested",
        ),
    )
    write_json(out / "selected-methods.json", choices)
    print(
        json.dumps(
            dict(primary=tests, prediction=pred_test, policy=policy_test, training=training_test)
        ),
        flush=True,
    )


def causal():
    root = ROOT / "results/bios-direction-causal-v1"
    out = ROOT / "docs/development-artifacts/direction-causal-v1"
    summarize(root, out, 288)
    parents = []
    for path in sorted(root.glob("world-*/*/model-1024.pt")):
        saved = torch.load(path, map_location="cpu", weights_only=False)
        initial = torch.load(path.with_name("model-0.pt"), map_location="cpu", weights_only=False)
        group = next(g for g in saved["optimizer"]["param_groups"] if g["name"] == "up")
        counts = [
            int(saved["optimizer"]["state"].get(i, {}).get("step", 0)) for i in group["params"]
        ]
        expected = 0 if saved["arm"] == "up-freezeall" else 768
        assert all(c == expected for c in counts), (path, counts, expected)
        if saved["arm"] in ("up-freezeearly", "up-freezeall"):
            middle = torch.load(
                path.with_name("model-256.pt"), map_location="cpu", weights_only=False
            )
            for name in ("blocks.0.mlp.up.weight", "blocks.0.mlp.up.bias"):
                assert torch.equal(initial["model"][name], middle["model"][name])
                if saved["arm"] == "up-freezeall":
                    assert torch.equal(initial["model"][name], saved["model"][name])
                else:
                    assert not torch.equal(middle["model"][name], saved["model"][name])
        learning = json.loads(path.with_name("learning.json").read_text())
        parents.append(
            dict(
                world=saved["world"],
                seed=saved["seed"],
                arm=saved["arm"],
                optimizer_up_steps=counts,
                accuracy=learning[-1]["accuracy"],
                loss=learning[-1]["loss"],
            )
        )
    assert len(parents) == 24
    write_json(
        out / "freeze-audit.json",
        dict(parents=parents, all_parameter_and_optimizer_freeze_checks_passed=True),
    )


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["confirm", "causal"])
    args = p.parse_args()
    (confirm if args.stage == "confirm" else causal)()
