"""Render audited P1/P4 development results; never read checkpoints or select cases."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

KEY = ["width", "world", "seed", "condition", "chain", "step"]
COLORS = {"exception": "#365d9d", "class-balanced-exception": "#c35a22"}
LABELS = {"exception": "Uniform", "class-balanced-exception": "Class balanced"}


def save(fig, folder, name):
    fig.savefig(folder / f"{name}.png", dpi=180, bbox_inches="tight")
    fig.savefig(folder / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def pair_status(left, right):
    """Pareto comparison: maximize E/D and minimize U; no weighted scalar."""
    if not (np.isfinite(left).all() and np.isfinite(right).all()):
        return "not_estimable"
    delta = (left - right) * np.array([1, 1, -1])
    if np.all(np.abs(delta) < 1e-12):
        return "equal"
    if np.all(delta >= -1e-12) and np.any(delta > 1e-12):
        return "balanced_dominates"
    if np.all(delta <= 1e-12) and np.any(delta < -1e-12):
        return "uniform_dominates"
    return "tradeoff_both_nondominated"


def p1(root):
    audit = json.loads((root / "audit.json").read_text())
    assert audit["state"] == "complete"
    assert audit["prediction_checkpoints_recomputed"] == 1152
    assert audit["weights_and_optimizer_states_verified"] == 192
    data = pd.read_csv(root / "editing.csv")
    arms = data[data.arm.isin(COLORS)].copy()
    assert len(arms) == 384
    uniform = arms[arms.arm == "exception"].set_index(KEY)
    balanced = arms[arms.arm == "class-balanced-exception"].set_index(KEY)
    assert uniform.index.equals(balanced.index)
    for field in ["E_changed_n", "D_conflict_heldout_n", "U_unseen_known", "U_unseen_n"]:
        np.testing.assert_array_equal(uniform[field], balanced[field])
    metrics = ["E_changed_accuracy", "D_conflict_heldout_accuracy", "U_unseen_rate"]
    paired = balanced[metrics].copy()
    for name in metrics:
        paired[f"balanced_{name}"] = balanced[name]
        paired[f"uniform_{name}"] = uniform[name]
        paired[f"delta_{name}"] = balanced[name] - uniform[name]
    paired = paired.drop(columns=metrics)
    paired["comparison"] = [
        pair_status(a, b)
        for a, b in zip(balanced[metrics].values, uniform[metrics].values, strict=True)
    ]
    assert not paired.comparison.eq("not_estimable").any()
    paired["balanced_nondominated"] = paired.comparison != "uniform_dominates"
    paired["uniform_nondominated"] = paired.comparison != "balanced_dominates"
    paired.reset_index().to_csv(root / "balanced-pareto-pairs.csv", index=False)
    counts = paired.reset_index().groupby(["width", "step", "comparison"]).size().rename("cases")
    counts.to_csv(root / "balanced-pareto-counts.csv")
    aggregate = []
    for width in [256, 768]:
        for step in [0, 32, 128, 512]:
            a = balanced.xs((width, step), level=("width", "step"))[metrics].mean().values
            b = uniform.xs((width, step), level=("width", "step"))[metrics].mean().values
            aggregate.append({"width": width, "step": step, "comparison": pair_status(a, b)})
    pd.DataFrame(aggregate).to_csv(root / "balanced-pareto-aggregate.csv", index=False)
    # Retention integers are reporting denominators, not independent replicates.
    pools = [
        "U_full",
        "U_unseen",
        "U_S_unchanged",
        "U_D_probe_unchanged",
        "U_unseen_strata_0",
        "U_unseen_strata_1",
        "U_unseen_strata_2",
        "U_unseen_strata_3",
        "U_other_chain",
        "U_independent",
    ]
    cover = []
    for (width, arm), frame in data[data.step == 512].groupby(["width", "arm"]):
        for pool in pools:
            n, known, broken = (frame[f"{pool}_{suffix}"] for suffix in ["n", "known", "broken"])
            cover.append(
                {
                    "width": width,
                    "arm": arm,
                    "pool": pool,
                    "cases": len(frame),
                    "n_sum": int(n.sum()),
                    "known_sum": int(known.sum()),
                    "broken_sum": int(broken.sum()),
                    "n_case_min": int(n.min()),
                    "n_case_max": int(n.max()),
                    "known_case_min": int(known.min()),
                    "known_case_max": int(known.max()),
                    "mean_case_rate": frame[f"{pool}_rate"].mean(),
                    "pooled_rate_descriptive": broken.sum() / known.sum()
                    if known.sum()
                    else np.nan,
                }
            )
    coverage = pd.DataFrame(cover)
    coverage.to_csv(root / "retention-denominators.csv", index=False)
    final = arms[arms.step == 512]
    assert np.allclose(final.E_changed_root_accuracy, 1)
    assert np.allclose(final.E_changed_actual_accuracy, 1)
    fig, axes = plt.subplots(3, 2, figsize=(13, 12), constrained_layout=True)
    category = [
        (org, chain)
        for org in ["company", "project", "neither"]
        for chain in ["company", "project"]
    ]
    for col, width in enumerate([256, 768]):
        subset = final[final.width == width]
        ax = axes[0, col]
        for _, frame in subset.groupby(["world", "seed", "condition", "chain"]):
            frame = frame.set_index("arm").loc[list(COLORS)]
            ax.plot(
                frame.U_unseen_rate * 100,
                frame.D_conflict_heldout_accuracy * 100,
                color="0.75",
                linewidth=0.7,
                zorder=1,
            )
        for arm in COLORS:
            frame = subset[subset.arm == arm]
            ax.scatter(
                frame.U_unseen_rate * 100,
                frame.D_conflict_heldout_accuracy * 100,
                color=COLORS[arm],
                marker="o" if arm == "exception" else "^",
                s=25,
                alpha=0.8,
                label=LABELS[arm],
            )
            for world, group in frame.groupby("world"):
                ax.scatter(
                    group.U_unseen_rate.mean() * 100,
                    group.D_conflict_heldout_accuracy.mean() * 100,
                    color=COLORS[arm],
                    marker="*" if world == 0 else "P",
                    s=180,
                    edgecolor="black",
                    linewidth=0.6,
                )
        ax.set(
            title=f"Width {width}: all 24 paired outcomes; E root / actual = 100%",
            xlabel="Unseen U damage (%) — lower is better",
            ylabel="Conflict heldout D (%)",
        )
        ax.grid(alpha=0.2)
        if col == 0:
            ax.legend(loc="upper right", fontsize=8)
        ax.text(
            0.02,
            0.97,
            "Stars: world 0 mean; crosses: world 1 mean",
            transform=ax.transAxes,
            va="top",
            fontsize=8,
        )
        ax = axes[1, col]
        p = paired.reset_index()
        p = p[(p.width == width) & (p.step == 512)]
        for (world, seed), group in p.groupby(["world", "seed"]):
            offset = -0.18 + 0.12 * (world * 2 + seed)
            xs = [category.index((r.condition, r.chain)) + offset for r in group.itertuples()]
            ax.scatter(
                xs,
                group.delta_D_conflict_heldout_accuracy * 100,
                marker="o" if seed == 0 else "^",
                color=["#457b9d", "#d08c32"][world],
                label=f"World {world}, seed {seed}",
                s=38,
            )
        for world, group in p.groupby("world"):
            ax.axhline(
                group.delta_D_conflict_heldout_accuracy.mean() * 100,
                color=["#457b9d", "#d08c32"][world],
                ls="--",
                lw=1,
            )
        ax.axhline(0, color="black", lw=0.7)
        ax.set(
            xticks=range(6),
            xticklabels=[f"{o[:3]} / {c[:3]}" for o, c in category],
            xlabel="Training organization / query chain",
            ylabel="Balanced − uniform D (pp)",
            title="Fixed 512 steps; no case selection (dashed: world means)",
        )
        if col == 0:
            ax.legend(fontsize=8, ncol=2)
        ax.grid(axis="y", alpha=0.2)
        ax = axes[2, col]
        for offset, arm in [(-0.17, "exception"), (0.17, "class-balanced-exception")]:
            sub = coverage[(coverage.width == width) & (coverage.arm == arm)].set_index("pool")
            names = ["U_unseen_strata_0", "U_unseen_strata_1"]
            heights = sub.loc[names, "mean_case_rate"] * 100
            ax.bar(np.arange(2) + offset, heights, width=0.3, color=COLORS[arm], label=LABELS[arm])
            for index, pool in enumerate(names):
                row = sub.loc[pool]
                ax.annotate(
                    f"{int(row.broken_sum)}/{int(row.known_sum)} known",
                    (index + offset, heights.iloc[index]),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                )
        n0 = coverage[(coverage.width == width) & (coverage.arm == "exception")].set_index("pool")
        ax.set(
            xticks=[0, 1],
            xticklabels=[
                "Selected groups: old-exception actual\n"
                f"Known {int(n0.loc['U_unseen_strata_0', 'known_sum'])} / "
                f"{int(n0.loc['U_unseen_strata_0', 'n_sum'])} eligible",
                "Selected people: other unchanged facts\n"
                f"Known {int(n0.loc['U_unseen_strata_1', 'known_sum'])} / "
                f"{int(n0.loc['U_unseen_strata_1', 'n_sum'])} eligible",
            ],
            ylabel="Local U damage (mean case %)",
            title="Local retention; labels retain broken / old-correct counts",
        )
        ax.margins(y=0.28)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle(
        "P1: identical exception task, paired supervision weights at 512 steps", fontsize=15
    )
    save(fig, root, "balanced-fixed-budget")
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), constrained_layout=True)
    columns = [
        ("E_changed_root_accuracy", "E: 3 default facts (%)"),
        ("E_changed_actual_accuracy", "E: 90 actual facts (%)"),
        ("D_conflict_heldout_accuracy", "Conflict heldout D (%)"),
        ("U_unseen_rate", "Unseen U damage (%)"),
    ]
    for row, width in enumerate([256, 768]):
        for col, (metric, title) in enumerate(columns):
            ax = axes[row, col]
            for arm in COLORS:
                frame = arms[(arms.width == width) & (arms.arm == arm)]
                for _, group in frame.groupby("world"):
                    means = group.groupby("step")[metric].mean()
                    ax.plot(range(4), means.values * 100, color=COLORS[arm], alpha=0.25, lw=1)
                means = frame.groupby("step")[metric].mean()
                ax.plot(
                    range(4), means.values * 100, color=COLORS[arm], marker="o", label=LABELS[arm]
                )
            ax.set(
                xticks=range(4),
                xticklabels=[0, 32, 128, 512],
                title=title,
                xlabel="Fixed edit step",
                ylabel=f"Width {width}" if col == 0 else "",
            )
            ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(
        "P1 trajectories: thick lines = all-case means; "
        "faint lines = each world (no best-step selection)"
    )
    save(fig, root, "balanced-trajectories")


def p4(source, destination):
    audit = json.loads((source / "audit.json").read_text())
    assert audit["status"] == "complete"
    if destination.is_symlink():
        raise ValueError("Use a durable summary directory, not a raw-result symlink")
    destination.mkdir(parents=True, exist_ok=True)
    for path in source.iterdir():
        if path.suffix in (".json", ".csv", ".md"):
            shutil.copy2(path, destination / path.name)
    data = pd.read_csv(source / "paired-necessity.csv")
    data = data[data.population == "all"]
    families = [
        "default",
        "actual",
        "membership",
        "group_root",
        "other_chain_default",
        "other_chain_actual",
        "attribute_3",
        "attribute_4",
        "attribute_5",
        "attribute_6",
    ]
    labels = [
        "Default\nderived",
        "Actual",
        "Membership",
        "Group root\n(64 queries)",
        "Other-chain\ndefault",
        "Other-chain\nactual",
        "Birth date",
        "Birth city",
        "University",
        "Major",
    ]
    contrasts = [
        ("target_remove", "clean"),
        ("target_remove", "random_remove"),
        ("target_remove", "complement_remove"),
        ("remove_rescue", "remove_wrong_source_rescue"),
        ("remove_rescue", "remove_random_rescue"),
    ]
    contrast_labels = [
        "Target − clean",
        "Target − random",
        "Target − complement",
        "Rescue − wrong-source",
        "Rescue − random",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(17, 8.5), constrained_layout=True)
    mats = []
    records = []
    for width in [256, 768]:
        for world in [0, 1]:
            block = data[(data.width == width) & (data.world == world)]
            matrix = np.zeros((len(contrasts), len(families)))
            for i, (left, right) in enumerate(contrasts):
                for j, family in enumerate(families):
                    sub = block[
                        (block.left == left) & (block.right == right) & (block.family == family)
                    ]
                    assert len(sub) == 12 and sub.common_available.gt(0).all()
                    matrix[i, j] = sub.accuracy_difference.mean() * 100
                    records.append(
                        {
                            "width": width,
                            "world": world,
                            "left": left,
                            "right": right,
                            "family": family,
                            "blocks": 12,
                            "common_available_sum": int(sub.common_available.sum()),
                            "mean_difference_pp": matrix[i, j],
                        }
                    )
            mats.append(matrix)
    limit = max(5, np.ceil(max(np.abs(m).max() for m in mats) / 5) * 5)
    for ax, matrix, (width, world) in zip(
        axes.ravel(), mats, [(256, 0), (256, 1), (768, 0), (768, 1)], strict=True
    ):
        im = ax.imshow(matrix, cmap="RdBu", vmin=-limit, vmax=limit, aspect="auto")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix[i, j]
                ax.text(
                    j,
                    i,
                    f"{value:+.2f}",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color="white" if abs(value) > limit * 0.55 else "black",
                )
        ax.set(
            xticks=range(len(labels)),
            xticklabels=labels,
            yticks=range(len(contrast_labels)),
            yticklabels=contrast_labels,
            title=f"Width {width}, development world {world} — 12 model/chain blocks",
        )
        ax.tick_params(axis="x", labelsize=8)
        ax.tick_params(axis="y", labelsize=9)
        plt.setp(ax.get_xticklabels(), rotation=25, ha="right", rotation_mode="anchor")
    fig.colorbar(im, ax=axes, shrink=0.8, label="Paired accuracy difference (percentage points)")
    fig.suptitle(
        "P4: frozen L0/position2 rank32 intervention and L1/position2 recovery\n"
        "All available queries; same people across 9 families (128 each); "
        "group roots are 64 independent queries",
        fontsize=14,
    )
    save(fig, destination, "necessity-family-controls")
    pd.DataFrame(records).to_csv(destination / "necessity-family-control-means.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--p1", type=Path, required=True)
    parser.add_argument("--p4-source", type=Path, required=True)
    parser.add_argument("--p4-output", type=Path, required=True)
    args = parser.parse_args()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42})
    p1(args.p1)
    p4(args.p4_source, args.p4_output)
    for folder in [args.p1, args.p4_output]:
        generated = [
            p
            for p in folder.iterdir()
            if (
                p.suffix in [".csv", ".png", ".pdf", ".md"]
                or p.name in ["audit.json", "interpretation-facts.json", "denominator-facts.json"]
            )
        ]
        manifest = {
            "producer": str(Path(__file__).resolve()),
            "producer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "note": "Descriptive rendering of audited data; no checkpoint selection or refitting.",
            "files": {
                p.name: {
                    "bytes": p.stat().st_size,
                    "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                }
                for p in sorted(generated)
            },
        }
        (folder / "small-artifact-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
