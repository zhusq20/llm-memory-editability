"""World-paired summaries and independently checked intervention scores."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from llm_memory_editability.grok_depth import write_json
from llm_memory_editability.text_pretrain import sha

ROOT = Path("docs/development-artifacts/text-pretrain-v1")
CONFIRM = ROOT / "confirmation"
ARMS = ("P0", "P1", "P2")


def csv_write(path, rows):
    with path.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mechanisms():
    lock = json.loads((CONFIRM / "analysis-lock.json").read_text())
    for p, expected in lock["sources"].items():
        assert sha(p) == expected
    cfg = json.loads(Path("configs/text-pretrain-confirmation-v1.json").read_text())
    rows, audit_count = [], 0
    donor_reference, donor_comparisons = {}, 0
    for name in cfg["runs"]:
        root = Path(cfg["output_root"]) / name
        world = np.load(root / "world.npz")
        lookup = {(h, r): t for h, r, t in world["atomic"]}
        reserved = {tuple(r[[0, 4]]) for r in world["background_test"]}
        for node in lock["nodes"]:
            base = CONFIRM / "mechanism" / name / str(node)
            d = json.loads((base / "summary.json").read_text())
            p = np.load(base / "predictions.npz")
            assert d["script_sha256"] == lock["sources"]["scripts/analyze_text_pretrain.py"]
            assert d["checkpoint_sha256"] == sha(root / f"weights-{node:06d}.pt")
            assert d["patched_position"] == 3 and d["donor_valid_tokens"] == 4
            for result in d["results"]:
                kind = result["kind"]
                recipient = world["target_test"][p[kind + "_indices"]]
                donor = p[kind + "_donor"]
                key = (d["spec"]["world"], kind)
                arrays = (p[kind + "_indices"], donor)
                if key in donor_reference:
                    for left, right in zip(donor_reference[key], arrays, strict=True):
                        np.testing.assert_array_equal(left, right)
                    donor_comparisons += 1
                else:
                    donor_reference[key] = arrays
                assert len(recipient) == result["n"]
                assert result["coverage"] == len(recipient) / len(world["target_test"])
                assert max(result["self_patch_probability_error"]) < 1e-5
                np.testing.assert_array_equal(recipient[:, -1], p[kind + "_original_target"])
                for original, alternative in zip(recipient, donor, strict=True):
                    h, r1, b, r2, t = alternative
                    assert lookup[h, r1] == b and lookup[b, r2] == t
                    assert h != original[0] and r1 == original[1] and r2 == original[3]
                    if kind == "same_bridge":
                        assert b == original[2] and t == original[4]
                    else:
                        assert b != original[2] and t != original[4] and (h, t) in reserved
                conditions = [
                    ("baseline", result["baseline"]),
                    ("normal", result["normal_counterfactual"]),
                ] + [(f"L{r['layer']}_{r['component']}", r) for r in result["conditions"]]
                for suffix, scores in conditions:
                    prefix = kind + "_" + suffix
                    answer, stops = p[prefix + "_answer"], p[prefix + "_stops"]
                    correct = answer == donor[:, -1]
                    full = correct & (stops[:, 0] == 5) & (stops[:, 1] == 1)
                    probability = p[prefix + "_probability"]
                    assert abs(scores["accuracy"] - full.mean()) < 1e-12
                    assert abs(scores["answer_accuracy"] - correct.mean()) < 1e-12
                    assert abs(scores["answer_probability"] - probability.mean()) < 1e-12
                    if suffix not in ("baseline", "normal"):
                        retained = (
                            (answer == recipient[:, -1]) & (stops[:, 0] == 5) & (stops[:, 1] == 1)
                        )
                        assert scores["original_answer_retained"] == retained.mean()
                        logits = p[prefix + "_logits"]
                        np.testing.assert_array_equal(logits.argmax(-1), answer)
                        exp = np.exp(logits - logits.max(-1, keepdims=True))
                        prob = exp[np.arange(len(exp)), donor[:, -1]] / exp.sum(-1)
                        np.testing.assert_allclose(prob, probability, atol=1e-6, rtol=1e-5)
                    rows.append(
                        dict(
                            run=name,
                            world=d["spec"]["world"],
                            initialization=d["spec"]["initialization"],
                            arm=d["spec"]["arm"],
                            node=node,
                            kind=kind,
                            condition=suffix,
                            n=len(donor),
                            coverage=result["coverage"],
                            accuracy=float(full.mean()),
                            probability=float(probability.mean()),
                            original_answer_retained=scores.get("original_answer_retained"),
                        )
                    )
                    audit_count += len(donor)
    write_json(
        CONFIRM / "donor-pairing-audit.json",
        dict(
            status="passed",
            unique_world_and_donor_kind_tables=len(donor_reference),
            exact_paired_table_comparisons=donor_comparisons,
        ),
    )
    return rows, audit_count


def main():
    endpoints = json.loads((CONFIRM / "endpoint-scores.json").read_text())
    trajectory = json.loads((CONFIRM / "trajectory-scores.json").read_text())
    mechanism, checks = mechanisms()
    out = ROOT / "report"
    out.mkdir(exist_ok=True)
    worlds = sorted({r["world"] for r in endpoints})
    pairs = []
    for w in worlds:
        for seed in sorted({r["initialization"] for r in endpoints}):
            group = {
                r["arm"]: r for r in endpoints if r["world"] == w and r["initialization"] == seed
            }
            scores = {a: group[a]["metrics"]["target_test"]["accuracy"] for a in ARMS}
            pairs.append(
                dict(
                    world=w,
                    initialization=seed,
                    **scores,
                    P1_minus_P0=scores["P1"] - scores["P0"],
                    P2_minus_P1=scores["P2"] - scores["P1"],
                )
            )
    world_scores = [
        dict(
            world=w,
            **{
                key: float(np.mean([r[key] for r in pairs if r["world"] == w]))
                for key in ("P0", "P1", "P2", "P1_minus_P0", "P2_minus_P1")
            },
        )
        for w in worlds
    ]
    metrics = {
        a: {
            split: float(
                np.mean([r["metrics"][split]["accuracy"] for r in endpoints if r["arm"] == a])
            )
            for split in (
                "atomic",
                "background_train",
                "background_test",
                "target_train",
                "target_test",
            )
        }
        for a in ARMS
    }
    for a in ARMS:
        metrics[a]["target_probability"] = float(
            np.mean(
                [
                    r["metrics"]["target_test"]["answer_probability"]
                    for r in endpoints
                    if r["arm"] == a
                ]
            )
        )
        metrics[a]["atomic_correct_coverage"] = float(
            np.mean(
                [
                    r["metrics"]["target_atomic_correct"]["coverage"]
                    for r in endpoints
                    if r["arm"] == a
                ]
            )
        )
    mech_mean = []
    for node in (16000, 32000):
        for kind in ("same_bridge", "different_bridge"):
            for a in ARMS:
                for condition in dict.fromkeys(r["condition"] for r in mechanism):
                    selected = [
                        r
                        for r in mechanism
                        if r["arm"] == a
                        and r["node"] == node
                        and r["kind"] == kind
                        and r["condition"] == condition
                    ]
                    assert len(selected) == 6
                    mech_mean.append(
                        dict(
                            node=node,
                            kind=kind,
                            arm=a,
                            condition=condition,
                            **{
                                k: float(np.mean([r[k] for r in selected]))
                                for k in ("coverage", "accuracy", "probability")
                            },
                        )
                    )
    summary = dict(
        confirmation_worlds=worlds,
        initializations=2,
        runs=len(endpoints),
        primary_step=32000,
        metrics=metrics,
        paired_effects=pairs,
        world_effects=world_scores,
        mechanism=mech_mean,
        intervention_prediction_rows_checked=checks,
    )
    write_json(out / "summary.json", summary)
    csv_write(out / "paired-effects.csv", pairs)
    csv_write(out / "mechanism-cells.csv", mechanism)
    csv_write(out / "learning-cells.csv", trajectory)
    colors = {"P0": "#667085", "P1": "#2563eb", "P2": "#0f9d76"}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    for a in ARMS:
        nodes = sorted({r["step"] for r in trajectory})
        scores = [
            [
                r["accuracy"] * 100
                for r in trajectory
                if r["arm"] == a and r["step"] == s and r["split"] == "target_test"
            ]
            for s in nodes
        ]
        axes[0].plot(np.array(nodes) / 1000, np.mean(scores, axis=1), label=a, color=colors[a])
        axes[0].fill_between(
            np.array(nodes) / 1000,
            np.min(scores, axis=1),
            np.max(scores, axis=1),
            alpha=0.1,
            color=colors[a],
        )
    axes[0].set(
        title="Held-out target paths",
        xlabel="Training steps (thousands)",
        ylabel="Full answer accuracy (%)",
        ylim=(0, 103),
    )
    axes[0].legend()
    for r in world_scores:
        axes[1].plot(range(3), [100 * r[a] for a in ARMS], marker="o", label=str(r["world"]))
    axes[1].set(
        title="Three independent worlds",
        xticks=range(3),
        xticklabels=ARMS,
        ylabel="Accuracy, averaged over 2 inits (%)",
        ylim=(0, 103),
    )
    axes[1].legend(title="World")
    for a in ARMS:
        values = [
            next(
                r["accuracy"]
                for r in mech_mean
                if r["node"] == 32000
                and r["kind"] == "different_bridge"
                and r["arm"] == a
                and r["condition"] == f"L{layer}_full"
            )
            * 100
            for layer in range(4)
        ]
        axes[2].plot(range(1, 5), values, marker="o", color=colors[a], label=a)
    axes[2].set(
        title="Pure-prefix counterfactual transfer",
        xlabel="Executed block",
        ylabel="Donor-consistent full answer (%)",
        xticks=range(1, 5),
        ylim=(0, 103),
    )
    axes[2].legend()
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.savefig(out / "comparison.png", dpi=180)
    fig.savefig(out / "comparison.pdf")
    plt.close(fig)
    # Show all predetermined components; no best-layer selection.
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for row, kind in enumerate(("same_bridge", "different_bridge")):
        for col, component in enumerate(("attention", "mlp", "full")):
            ax = axes[row, col]
            for a in ARMS:
                values = [
                    next(
                        r["accuracy"]
                        for r in mech_mean
                        if r["node"] == 32000
                        and r["kind"] == kind
                        and r["arm"] == a
                        and r["condition"] == f"L{layer}_{component}"
                    )
                    * 100
                    for layer in range(4)
                ]
                base = (
                    next(
                        r["accuracy"]
                        for r in mech_mean
                        if r["node"] == 32000
                        and r["kind"] == kind
                        and r["arm"] == a
                        and r["condition"] == "baseline"
                    )
                    * 100
                )
                ax.plot(range(1, 5), values, color=colors[a], marker="o", label=a)
                ax.axhline(base, color=colors[a], alpha=0.4, linestyle=":")
            ax.set(
                title=f"{kind.replace('_', ' ')} / {component}",
                xlabel="Executed block",
                ylabel="Donor-consistent accuracy (%)",
                xticks=range(1, 5),
                ylim=(0, 103),
            )
            ax.grid(alpha=0.2)
            ax.legend()
    fig.savefig(out / "mechanism.png", dpi=180)
    fig.savefig(out / "mechanism.pdf")
    plt.close(fig)
    print(json.dumps(dict(metrics=metrics, world_effects=world_scores, checks=checks)), flush=True)


if __name__ == "__main__":
    main()
