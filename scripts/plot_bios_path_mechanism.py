"""Display feature/error/parameter alignment; derive gauge-invariant logit contrasts."""

import csv
import json

import matplotlib
import numpy as np

from llm_memory_editability.bios_path_transfer import ROOT

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main():
    directory = ROOT / "docs/development-artifacts/path-minimal-v1"
    rows = []
    with np.load(directory / "jacobian-metrics.npz") as matrices:
        for seed in (0, 1):
            for person in (0, 1, 2):
                parent = ROOT / f"results/bios-path-minimal-v1/seed-{seed}"
                original = parent / f"person-{person}-conflict-full"
                case = json.loads((original / "trajectory.json").read_text())
                a, b, c = case["values_abc"]
                p = matrices[f"seed{seed}_person{person}_probabilities"]
                ei, ej = p[0].copy(), p[1].copy()
                ei[b] -= 1
                ej[a] -= 1
                m = matrices[f"seed{seed}_person{person}_metric"]
                with np.load(original / "factors.npz") as factors:
                    z, d = factors["z"][0].astype(float), factors["delta"][0].astype(float)
                    g = np.einsum("btd,btk->bdk", d, z)
                    feature_cosine = np.dot(z[0, -1], z[1, -1]) / (
                        np.linalg.norm(z[0, -1]) * np.linalg.norm(z[1, -1])
                    )
                with np.load(parent / f"person-{person}-conflict-no_cross/factors.npz") as factors:
                    cut = np.einsum(
                        "btd,btk->bdk",
                        factors["source_delta"][0].astype(float),
                        factors["source_z"][0].astype(float),
                    )[0]
                kernel = np.sum(g[0] * g[1])
                contrast = m[a, b] - m[a, c] - m[c, b] + m[c, c]
                rows.append(
                    {
                        "seed": seed,
                        "person": person,
                        "feature_cosine": feature_cosine,
                        "output_error_cosine": np.dot(ei, ej)
                        / (np.linalg.norm(ei) * np.linalg.norm(ej)),
                        "full_gradient_cosine": kernel
                        / (np.linalg.norm(g[0]) * np.linalg.norm(g[1])),
                        "no_cross_gradient_cosine": np.sum(cut * g[1])
                        / (np.linalg.norm(cut) * np.linalg.norm(g[1])),
                        "true_kernel": kernel,
                        "new_vs_old_contrast_kernel": contrast,
                        "contrast_relative_error": abs(contrast - kernel) / abs(kernel),
                    }
                )
    with (directory / "mechanism-geometry.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    labels = [f"{r['seed']}/{r['person']}" for r in rows]
    for column, label, color, offset in (
        ("feature_cosine", "MLP feature", "#718093", -0.24),
        ("output_error_cosine", "Output error", "#16a085", -0.08),
        ("full_gradient_cosine", "True parameter gradient", "#273c75", 0.08),
        ("no_cross_gradient_cosine", "Cut cross-position backward", "#d35400", 0.24),
    ):
        axes[0].scatter(
            np.arange(6) + offset, [r[column] for r in rows], label=label, color=color, s=40
        )
    axes[0].set_xticks(np.arange(6), labels)
    axes[0].set_xlabel("Initialization / person")
    axes[0].set_ylabel("Cosine similarity")
    axes[0].axhline(0, color="gray", lw=0.8)
    axes[0].set_title("Similar features do not determine transfer sign")
    axes[0].legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.23), ncol=2)
    with (directory / "mature-margin-summary.csv").open() as stream:
        margin = list(csv.DictReader(stream))
    arms = ("full", "no_cross", "no_mlp", "fixed_qk")
    for kind, color, offset in (("coherent", "#16a085", -0.18), ("conflict", "#d35400", 0.18)):
        values = [
            float(next(r for r in margin if r["kind"] == kind and r["arm"] == arm)["margin_change"])
            for arm in arms
        ]
        axes[1].bar(np.arange(4) + offset, values, width=0.34, label=kind, color=color)
    axes[1].set_xticks(np.arange(4), arms)
    axes[1].axhline(0, color="gray", lw=0.8)
    axes[1].set_ylabel("Desired-minus-competing answer logit change")
    axes[1].set_title("Mature 8-layer models: answer competition")
    axes[1].legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.23), ncol=2)
    for ext in ("png", "pdf"):
        fig.savefig(directory / f"mechanism-summary.{ext}", dpi=180)
    plt.close(fig)
    print(
        json.dumps(
            {
                "contrast_error_min": min(r["contrast_relative_error"] for r in rows),
                "contrast_error_max": max(r["contrast_relative_error"] for r in rows),
            }
        )
    )


if __name__ == "__main__":
    main()
