"""Audit two-layer trajectories and the secondary mature-model margin analysis."""

import csv
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from llm_memory_editability.bios_path_minimal import CONFIG, verify
from llm_memory_editability.bios_path_transfer import ROOT, sha, stamp


def csv_write(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def first_competition(timeline):
    for first, second in zip(timeline, timeline[1:], strict=False):
        if first["kernel_derived"] < -1e-8 and second["kernel_derived"] < -1e-8:
            return first["step"]
    return None


def factor_check(path, timeline):
    with np.load(path) as arrays:
        actual = np.einsum(
            "nbtd,nbtk->nbdk", arrays["delta"].astype(float), arrays["z"].astype(float)
        )
        source = np.einsum(
            "nbtd,nbtk->nbdk",
            arrays["source_delta"].astype(float),
            arrays["source_z"].astype(float),
        )[:, 0]
        base_norm = np.linalg.norm(actual[:, 0], axis=(1, 2))
        source_norm = np.linalg.norm(source, axis=(1, 2))
        direction = (
            source
            * np.divide(
                base_norm, source_norm, out=np.zeros_like(base_norm), where=source_norm != 0
            )[:, None, None]
        )
        kernel = np.sum(actual[:, 1] * direction, axis=(1, 2))
        expected = np.array([row["kernel_derived"] for row in timeline])
        scale = np.maximum(np.linalg.norm(actual[:, 1], axis=(1, 2)) * base_norm, 1e-30)
        error = np.max(abs(kernel - expected) / scale)
        if error > 5e-5:
            raise ValueError("Minimal archived factors disagree with transfer kernel")
        return float(error)


def collect(config):
    lock_hash = verify(config)
    summaries, points, errors, accuracy = [], [], [], []
    for seed in config["seeds"]:
        out = ROOT / config["output"] / f"seed-{seed}"
        complete = json.loads((out / "complete.json").read_text())
        if complete["lock_sha256"] != lock_hash:
            raise ValueError("Minimal worker has the wrong lock")
        for name, digest in complete["files"].items():
            if sha(out / name) != digest:
                raise ValueError("Minimal worker file hash mismatch")
        learned = json.loads((out / "learning.json").read_text())[-1]
        if learned["step"] != config["train_steps"]:
            raise ValueError("Wrong training budget")
        accuracy.append({"seed": seed, **learned})
        for person, kind, arm in itertools.product(
            config["people_to_edit"], config["kinds"], config["arms"]
        ):
            directory = out / f"person-{person}-{kind}-{arm}"
            receipt = json.loads((directory / "complete.json").read_text())
            if (
                not receipt["frozen_parameters_unchanged"]
                or receipt["steps"] != config["edit_steps"]
            ):
                raise ValueError("Minimal trajectory contract violation")
            for name, digest in receipt["files"].items():
                if sha(directory / name) != digest:
                    raise ValueError("Minimal trajectory hash mismatch")
            result = json.loads((directory / "trajectory.json").read_text())
            timeline = result["timeline"]
            if [row["step"] for row in timeline] != list(
                range(0, config["edit_steps"] + 1, config["measure_every"])
            ):
                raise ValueError("Minimal measurement grid incomplete")
            if len(result["steps"]) != config["edit_steps"]:
                raise ValueError("Minimal update budget incomplete")
            for step in result["steps"]:
                if abs(step["direction_norm"] - step["source_gradient_norm"]) > 1e-5 * max(
                    step["source_gradient_norm"], 1e-30
                ):
                    raise ValueError("Minimal update norm mismatch")
            errors.append(factor_check(directory / "factors.npz", timeline))
            first, final = timeline[0], timeline[-1]
            key = {"seed": seed, "person": person, "kind": kind, "arm": arm}
            summaries.append(
                {
                    **key,
                    "initial_derived_loss": first["loss"][1],
                    "final_derived_loss": final["loss"][1],
                    "minimum_derived_loss": min(r["loss"][1] for r in timeline),
                    "initial_kernel": first["kernel_derived"],
                    "final_kernel": final["kernel_derived"],
                    "first_competition_step": first_competition(timeline),
                    "source_target_probability": final["source_target_probability"],
                    "initial_margin": first["derived_margin_ab"],
                    "final_margin": final["derived_margin_ab"],
                    "first_step_margin_change": first["one_step_margin_change"],
                    "derived_probability_a": final["probabilities_abc"][1][0],
                    "derived_probability_b": final["probabilities_abc"][1][1],
                    "derived_probability_c": final["probabilities_abc"][1][2],
                    "retention_loss_change": final["loss"][2] - first["loss"][2],
                    "source_correct": final["prediction"][0] == result["labels"][0],
                    "derived_correct": final["prediction"][1] == result["labels"][1],
                    "retention_correct": final["prediction"][2] == result["labels"][2],
                }
            )
            for row in timeline:
                points.append(
                    {
                        **key,
                        "step": row["step"],
                        "source_loss": row["loss"][0],
                        "derived_loss": row["loss"][1],
                        "retention_loss": row["loss"][2],
                        "kernel": row["kernel_derived"],
                        "kernel_full": row["kernel_derived_full"],
                        "margin": row["derived_margin_ab"],
                        "source_probability": row["source_target_probability"],
                        "derived_probability_a": row["probabilities_abc"][1][0],
                        "derived_probability_b": row["probabilities_abc"][1][1],
                        "derived_probability_c": row["probabilities_abc"][1][2],
                        "isotropic_reference": row["isotropic_same_probability_reference"],
                        "one_step_observed": row.get("one_step_observed_derived"),
                        "one_step_predicted": row.get("one_step_predicted_derived"),
                    }
                )
    return (
        summaries,
        points,
        {
            "finished_at": stamp(),
            "lock_sha256": lock_hash,
            "parents": accuracy,
            "trajectories": len(summaries),
            "training_steps": len(accuracy) * config["train_steps"],
            "editing_steps": len(summaries) * config["edit_steps"],
            "measurement_points": len(points),
            "max_factor_kernel_error": max(errors),
            "frozen_parameters_unchanged": True,
            "all_update_norm_controls_passed": True,
            "analysis_source_sha256": sha(Path(__file__)),
        },
    )


def means(rows, keys, columns):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[k] for k in keys)].append(row)
    return [
        {
            **dict(zip(keys, key, strict=True)),
            "n": len(values),
            **{column: float(np.mean([float(r[column]) for r in values])) for column in columns},
        }
        for key, values in sorted(grouped.items())
    ]


def plots(directory, points):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"full": "#273c75", "no_cross": "#d35400", "no_mlp": "#16a085", "fixed_qk": "#8e44ad"}
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), constrained_layout=True)
    for row, kind in enumerate(("coherent", "conflict")):
        for arm, color in colors.items():
            aggregated = means(
                [r for r in points if r["kind"] == kind and r["arm"] == arm],
                ["step"],
                ["derived_loss", "margin", "source_probability"],
            )
            for column, field in enumerate(("derived_loss", "margin", "source_probability")):
                axes[row, column].plot(
                    [r["step"] for r in aggregated],
                    [r[field] for r in aggregated],
                    color=color,
                    label=arm,
                )
                axes[row, column].set_title(kind + ": " + field.replace("_", " "))
                axes[row, column].set_xlabel("Edit step")
        axes[row, 1].axhline(0, color="gray", ls="--", lw=0.7)
    axes[0, 0].legend(fontsize=8)
    for ext in ("png", "pdf"):
        fig.savefig(directory / f"minimal-trajectories.{ext}", dpi=180)
    plt.close(fig)


def main():
    config = json.loads(CONFIG.read_text())
    directory = ROOT / config["artifacts"]
    summaries, points, audit = collect(config)
    csv_write(directory / "endpoints.csv", summaries)
    csv_write(directory / "all-points.csv", points)
    mean_summary = means(
        summaries,
        ["kind", "arm"],
        [
            "initial_derived_loss",
            "final_derived_loss",
            "minimum_derived_loss",
            "source_target_probability",
            "final_margin",
            "derived_correct",
            "retention_correct",
        ],
    )
    csv_write(directory / "summary.csv", mean_summary)
    with (directory / "mature-margin-secondary.csv").open() as stream:
        margin = list(csv.DictReader(stream))
    margin_summary = means(
        margin, ["kind", "arm"], ["margin_change", "predicted_margin_change", "original_CE_change"]
    )
    csv_write(directory / "mature-margin-summary.csv", margin_summary)
    writeup = [
        "# Two-layer common-error / competition follow-up",
        "",
        json.dumps(audit, indent=2),
        "",
        "| Kind | Arm | Initial derived CE | Final derived CE | Min derived CE "
        "| Final source P | Final a-b margin | D correct | U correct |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in mean_summary:
        values = [
            row[field]
            for field in (
                "initial_derived_loss",
                "final_derived_loss",
                "minimum_derived_loss",
                "source_target_probability",
                "final_margin",
                "derived_correct",
                "retention_correct",
            )
        ]
        writeup.append(
            f"| {row['kind']} | {row['arm']} | " + " | ".join(f"{v:.6f}" for v in values) + " |"
        )
    writeup += [
        "",
        "## Sustained negative transfer onset",
        "",
        "| Kind | Arm | Trajectories with onset | First observed steps |",
        "|---|---|---:|---|",
    ]
    for kind, arm in itertools.product(config["kinds"], config["arms"]):
        selected = [r for r in summaries if r["kind"] == kind and r["arm"] == arm]
        onsets = [r["first_competition_step"] for r in selected]
        writeup.append(f"| {kind} | {arm} | {sum(x is not None for x in onsets)}/6 | {onsets} |")
    writeup += [
        "",
        "## Secondary mature-model replay",
        "",
        "| Kind | Arm | Desired CE change | a-b margin change |",
        "|---|---|---:|---:|",
    ]
    for row in margin_summary:
        writeup.append(
            f"| {row['kind']} | {row['arm']} | {row['original_CE_change']:+.8f} "
            f"| {row['margin_change']:+.8f} |"
        )
    (directory / "report.md").write_text("\n".join(writeup) + "\n")
    (directory / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    plots(directory, points)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
