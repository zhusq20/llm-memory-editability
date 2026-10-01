"""Audit archived factors/updates and summarize the fixed v2.13 development batch."""

import csv
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from llm_memory_editability.bios_path_transfer import CONFIG, ROOT, sha, stamp, verify_lock


def write_csv(path, rows):
    if not rows:
        raise ValueError("Empty summary")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def group(rows, keys):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    return groups


def aggregate(rows, keys, fields):
    return [
        {
            **dict(zip(keys, key, strict=True)),
            "n": len(values),
            **{field: float(np.mean([row[field] for row in values])) for field in fields},
        }
        for key, values in sorted(group(rows, keys).items())
    ]


def approximation(rows):
    prediction = np.array([row["predicted_change"] for row in rows])
    actual = np.array([row["observed_change"] for row in rows])
    # Numerical-resolution diagnostic only; all records remain in every mean.
    resolved = (abs(actual) > 1e-7) & (abs(prediction) > 1e-7)
    return {
        "n": len(rows),
        "resolved": int(resolved.sum()),
        "sign_agreement": float(np.mean((prediction[resolved] * actual[resolved]) > 0))
        if resolved.any()
        else None,
        "rmse": float(np.mean((prediction - actual) ** 2) ** 0.5),
        "relative_rmse": float(
            np.linalg.norm(prediction - actual) / max(np.linalg.norm(actual), 1e-30)
        ),
        "max_absolute_error": float(abs(prediction - actual).max()),
    }


def audit_factors(path, diagnostics):
    maximum = 0.0
    with np.load(path) as arrays:
        z, delta = arrays["evaluation_z"].astype(float), arrays["evaluation_delta"].astype(float)
        evaluations = np.einsum("btd,btk->bdk", delta, z)
        for row in diagnostics:
            arm, index = row["arm"], row["probe"]
            source = np.einsum(
                "btd,btk->bdk",
                arrays[arm + "_delta"].astype(float),
                arrays[arm + "_z"].astype(float),
            )[0]
            evaluation = evaluations[index]
            scale = max(np.linalg.norm(source) * np.linalg.norm(evaluation), 1e-30)
            reconstructed = np.sum(source * evaluation)
            error = abs(reconstructed - row["kernel"]) / scale
            maximum = max(maximum, error)
            parts = sum(
                row[f"kernel_{left}_{right}"]
                for left, right in itertools.product(("supervised", "unsupervised"), repeat=2)
            )
            if max(error, abs(parts - row["kernel"]) / scale) > 5e-5:
                raise ValueError("Saved factors do not reconstruct signed transfer")
    return maximum


def collect(config):
    lock_hash = verify_lock(config)
    out = ROOT / config["output"]
    artifacts = ROOT / config["artifacts"]
    expected_cases = json.loads((artifacts / "cases.json").read_text())
    rows, diagnostics, audits, shams = [], [], [], []
    for world, seed in itertools.product(config["worlds"], config["seeds"]):
        worker = out / f"world-{world}-seed-{seed}-neither"
        complete = json.loads((worker / "complete.json").read_text())
        if complete["lock_sha256"] != lock_hash:
            raise ValueError("Worker lock mismatch")
        for name, digest in complete["files"].items():
            if sha(worker / name) != digest:
                raise ValueError("Worker artifact hash mismatch")
        for step in config["states"]:
            state = worker / f"step-{step}"
            shams.append(json.loads((state / "sham.json").read_text())["relative_gradient_error"])
            for case in expected_cases[str(world)]["cases"]:
                directory = state / case["name"]
                record = json.loads((directory / "measurements.json").read_text())
                receipt = json.loads((directory / "receipt.json").read_text())
                if (
                    record["case"] != case
                    or not receipt["frozen_parameters_unchanged"]
                    or not receipt["parent_restored"]
                ):
                    raise ValueError("Case or parameter identity mismatch")
                for name, digest in receipt["files"].items():
                    if sha(directory / name) != digest:
                        raise ValueError("Case receipt mismatch")
                keys = {
                    "world": world,
                    "seed": seed,
                    "step": step,
                    "case": case["name"],
                    "kind": case["kind"],
                    "chain": case["chain"],
                    "group": case["group"],
                }
                expected = set(
                    itertools.product(
                        config["arms"], config["scales"], config["step_fractions"], range(6)
                    )
                )
                observed = {
                    (r["arm"], r["scale"], r["fraction"], r["probe"]) for r in record["updates"]
                }
                if expected != observed or len(observed) != len(record["updates"]):
                    raise ValueError("Incomplete or duplicate update matrix")
                duplicate = {}
                for row in record["updates"]:
                    if abs(row["after"] - row["before"] - row["observed_change"]) > 1e-12:
                        raise ValueError("Loss delta mismatch")
                    if (
                        row["id"] != case["ids"][row["probe"]]
                        or row["target"] != case["targets"][row["probe"]]
                    ):
                        raise ValueError("Probe label mismatch")
                    values = np.array(
                        [row[k] for k in ("before", "after", "predicted_change", "update_norm")]
                    )
                    if not np.isfinite(values).all():
                        raise ValueError("Non-finite update result")
                    if row["arm"] == "full":
                        key = row["fraction"], row["probe"]
                        check = (
                            row["weight_sha256"],
                            row["after"],
                            row["prediction"],
                            row["ended"],
                        )
                        if key in duplicate and duplicate[key] != check:
                            raise ValueError("Full shared-lr/norm control differs")
                        duplicate[key] = check
                    rows.append({**keys, **row})
                diagnostics += [{**keys, **row} for row in record["diagnostics"]]
                audits.append(audit_factors(directory / "factors.npz", record["diagnostics"]))
    return (
        rows,
        diagnostics,
        {
            "created_at": stamp(),
            "lock_sha256": lock_hash,
            "states": len(shams),
            "cases": len(audits),
            "single_step_conditions": len(rows) // 6,
            "probe_records": len(rows),
            "gradient_arms": len(diagnostics) // 6,
            "maximum_factor_kernel_error": max(audits),
            "maximum_sham_error": max(shams),
            "maximum_forward_error": max(row["forward_error"] for row in diagnostics),
            "maximum_gradient_reconstruction_error": max(
                row["reconstruction_error"] for row in diagnostics
            ),
            "maximum_no_cross_unsupervised_norm": max(
                row["unsupervised_norm"] for row in diagnostics if row["arm"] == "no_cross"
            ),
            "all_frozen_parameters_unchanged": True,
            "all_parents_restored": True,
            "all_full_duplicate_controls_identical": True,
        },
    )


def make_plots(directory, primary, rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"full": "#273c75", "no_cross": "#d35400", "no_mlp": "#16a085", "fixed_qk": "#8e44ad"}
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
    for axis, kind in zip(axes, ("root", "coherent", "conflict"), strict=True):
        means = []
        arms = list(colors)
        for arm in arms:
            subset = [
                r
                for r in primary
                if r["kind"] == kind and r["arm"] == arm and r["role"] == "derived_same"
            ]
            means.append(np.mean([r["observed_change"] for r in subset]))
        axis.bar(arms, means, color=list(colors.values()))
        for j, arm in enumerate(arms):
            values = aggregate(
                [
                    r
                    for r in primary
                    if r["kind"] == kind and r["arm"] == arm and r["role"] == "derived_same"
                ],
                ["world", "seed"],
                ["observed_change"],
            )
            axis.scatter(
                [j] * len(values), [r["observed_change"] for r in values], color="black", s=15
            )
        axis.axhline(0, color="gray", lw=0.8)
        axis.set_title(kind + " source update")
        axis.tick_params(axis="x", rotation=30)
    axes[0].set_ylabel("Derived-query loss change (negative = improvement)")
    fig.suptitle("Same update norm; mature models; dots = world/initialization blocks")
    for ext in ("png", "pdf"):
        fig.savefig(directory / f"transfer.{ext}", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), constrained_layout=True)
    for axis, step in zip(axes, (0, 5120, 15360), strict=True):
        for arm, color in colors.items():
            selected = [
                r
                for r in rows
                if r["step"] == step
                and r["arm"] == arm
                and r["scale"] == "matched_norm"
                and r["fraction"] == 0.0003
            ]
            axis.scatter(
                [r["predicted_change"] for r in selected],
                [r["observed_change"] for r in selected],
                s=6,
                alpha=0.5,
                color=color,
                label=arm,
            )
        bounds = axis.get_xlim()
        axis.plot(bounds, bounds, "k--", lw=0.7)
        axis.set_title(f"Training step {step}")
        axis.set_xlabel("Predicted loss change")
    axes[0].set_ylabel("Observed loss change")
    axes[-1].legend(fontsize=7)
    fig.savefig(directory / "prediction.png", dpi=180)
    plt.close(fig)


def main():
    config = json.loads(CONFIG.read_text())
    directory = ROOT / config["artifacts"]
    rows, diagnostics, audit = collect(config)
    write_csv(directory / "all-updates.csv", rows)
    write_csv(directory / "all-gradients.csv", diagnostics)
    primary = [
        r
        for r in rows
        if r["step"] == 15360
        and r["scale"] == "matched_norm"
        and r["fraction"] == config["primary_fraction"]
    ]
    summaries = aggregate(
        rows,
        ["step", "kind", "arm", "scale", "fraction", "role"],
        ["before", "after", "observed_change", "predicted_change", "update_norm"],
    )
    blocks = aggregate(
        primary,
        ["world", "seed", "kind", "arm", "role"],
        ["observed_change", "predicted_change", "update_norm"],
    )
    gradient_summary = aggregate(
        [r for r in diagnostics if r["role"] == "source"],
        ["step", "arm"],
        ["gradient_cosine_full", "gradient_norm", "supervised_norm", "unsupervised_norm"],
    )
    prediction = [
        {**dict(zip(("step", "scale", "fraction"), key, strict=True)), **approximation(value)}
        for key, value in sorted(group(rows, ["step", "scale", "fraction"]).items())
    ]
    write_csv(directory / "summary.csv", summaries)
    write_csv(directory / "paired-blocks.csv", blocks)
    write_csv(directory / "gradient-summary.csv", gradient_summary)
    write_csv(directory / "prediction-validation.csv", prediction)
    audit["source_updates_with_increased_loss"] = sum(
        r["observed_change"] > 0 for r in rows if r["role"] == "source"
    )
    audit["analysis_source_sha256"] = sha(Path(__file__))
    (directory / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    mean_primary = aggregate(primary, ["kind", "arm", "role"], ["observed_change"])
    lookup = {(r["kind"], r["arm"], r["role"]): r["observed_change"] for r in mean_primary}
    report = [
        "# v2.13 fixed-batch machine summary",
        "",
        json.dumps(audit, indent=2),
        "",
        "Mature checkpoints, matched norm, fraction 0.0003. Negative loss change is improvement.",
        "",
        "| Source kind | Arm | Source Δloss | Derived same Δloss "
        "| Derived other Δloss | Local retention Δloss |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for kind, arm in itertools.product(("root", "coherent", "conflict"), config["arms"]):
        values = [
            lookup[kind, arm, role]
            for role in ("source", "derived_same", "derived_other", "retention_local")
        ]
        report.append(f"| {kind} | {arm} | " + " | ".join(f"{v:+.8f}" for v in values) + " |")
    report += [
        "",
        "## Gradient geometry",
        "",
        "| Step | Arm | Cosine to full | Gradient norm | Supervised norm | Other-position norm |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in gradient_summary:
        report.append(
            f"| {row['step']} | {row['arm']} | {row['gradient_cosine_full']:.6f} "
            f"| {row['gradient_norm']:.6f} | {row['supervised_norm']:.6f} "
            f"| {row['unsupervised_norm']:.6f} |"
        )
    report += [
        "",
        "## Taylor validation",
        "",
        "| Step | Scale | Fraction | Relative RMSE | Resolved sign agreement |",
        "|---|---|---|---:|---:|",
    ]
    for row in prediction:
        report.append(
            f"| {row['step']} | {row['scale']} | {row['fraction']} "
            f"| {row['relative_rmse']:.6f} | {row['sign_agreement']:.6f} |"
        )
    (directory / "report.md").write_text("\n".join(report) + "\n")
    make_plots(directory, primary, rows)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
