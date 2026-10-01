"""Reproducible figures and compact evidence tables for the v2.15 campaign."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from report_bios_direction import rows_for, summarize, validate_batch

from llm_memory_editability.bios_data import write_json
from llm_memory_editability.bios_direction import ROOT, digest
from llm_memory_editability.bios_direction_cross import CROSS_ARMS

ART = ROOT / "docs/development-artifacts"
DEST = ART / "direction-report-v1"
METHODS = ("func-soft-adam", "func-soft-gn", "func-hard-gn", "repr-hard-adam")
LABELS = ("Soft / Adam", "Soft / GN", "Hard / GN", "Repr. / Adam")
COLORS = ("#3569a8", "#d48627", "#209882", "#9a65a7")


def read(path):
    return json.loads(Path(path).read_text())


def csv_rows(path):
    with Path(path).open() as f:
        return list(csv.DictReader(f))


def save(fig, name):
    fig.savefig(DEST / f"{name}.png", dpi=180, bbox_inches="tight")
    fig.savefig(DEST / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def common_style():
    DEST.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.2,
            "legend.frameon": False,
        }
    )


def core():
    common_style()
    confirmed = csv_rows(ART / "direction-confirm-v1/cases.csv")
    curves = csv_rows(ART / "direction-confirm-v1/curves.csv")
    mini = csv_rows(ART / "direction-v1/curves.csv")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    totals = []
    for i, (method, label, color) in enumerate(zip(METHODS, LABELS, COLORS, strict=True)):
        subset = [r for r in confirmed if r["arm"] == method]
        rates = [
            np.mean([int(r["joint"]) for r in subset if int(r["world"]) == w])
            for w in range(5000, 5008)
        ]
        axes[0, 0].scatter(np.arange(8) * 0.02 + i - 0.07, rates, color=color, alpha=0.8)
        axes[0, 0].plot([i - 0.2, i + 0.2], [np.mean(rates)] * 2, color=color, lw=3)
        totals.append(
            dict(
                method=method,
                n=len(subset),
                joint=sum(int(r["joint"]) for r in subset),
                E=sum(int(r["E_joint"]) for r in subset),
                world_rates=rates,
            )
        )
        points = []
        for step in (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024):
            batch = [r for r in mini if r["arm"] == method and int(r["attempt"]) == step]
            points.append(
                (
                    np.mean([int(r["U_broken"]) for r in batch]),
                    np.mean([int(r["E_joint"]) for r in batch]),
                )
            )
        axes[0, 1].plot(*np.array(points).T, color=color, marker=".", label=label)
        axes[0, 1].scatter(*points[-1], color=color, marker="s", s=50)
        for col, kind in enumerate(("coherent", "independent")):
            steps, values = [], []
            for step in (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024):
                batch = [
                    r
                    for r in curves
                    if r["arm"] == method and r["kind"] == kind and int(r["attempt"]) == step
                ]
                if batch:
                    steps.append(step)
                    values.append(np.mean([int(r["joint"]) for r in batch]))
            axes[1, col].plot(steps, values, color=color, marker=".", label=label)
    axes[0, 0].set(
        xticks=range(4),
        xticklabels=LABELS,
        ylabel="E + local + U joint success",
        ylim=(0, 0.65),
        title="Independent confirmation: 8 world means",
    )
    axes[0, 1].set(
        xlabel="Mean newly broken U facts / edit (out of 24)",
        ylabel="All E correct",
        title="Original miniature: complete trajectories; square = final",
    )
    axes[0, 1].legend(fontsize=9)
    for col, kind in enumerate(("coherent", "independent")):
        axes[1, col].set(
            xlabel="Attempted updates",
            ylabel="Joint success",
            ylim=(0, 1),
            title=f"Independent worlds: {kind} targets",
        )
        axes[1, col].set_xscale("symlog", linthresh=1)
        axes[1, col].set_xlim(0, 1024)
    axes[1, 0].legend(fontsize=9)
    save(fig, "editability")
    write_json(DEST / "confirmation-totals.json", totals)

    learning = []
    for path in sorted(
        (ROOT / "results/bios-direction-formation-v1").glob("world-*/*/learning.json")
    ):
        for checkpoint in read(path):
            for factor in checkpoint["factors"]:
                learning.append(dict(arm=path.parent.name, **checkpoint, factor=factor))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for arm in ("baseline", "qk-0.25", "qk-4", "up-0.25", "up-4", "post-0.25", "post-4"):
        steps = (0, 8, 32, 128, 256, 512, 1024)
        values = [
            np.median(
                [r["factor"]["ratio"] for r in learning if r["step"] == s and r["arm"] == arm]
            )
            for s in steps
        ]
        axes[0, 0].plot(steps, values, marker=".", label=arm)
    axes[0, 0].set(
        xlabel="Pretraining updates",
        ylabel="Median r_minus / r_plus",
        title="Unprotected local kernel; not a CE learning-rate law",
        yscale="log",
    )
    axes[0, 0].legend(ncol=2, fontsize=8)
    local_errors = defaultdict(list)
    for _path, _i, record in rows_for(ROOT / "results/bios-direction-v1/main"):
        if record["arm"] not in METHODS:
            continue
        for point in record["timeline"]:
            if "local_step" not in point:
                continue
            step = point["local_step"]
            pred, obs = (
                np.array(step["predicted_margin_change"]),
                np.array(step["observed_margin_change"]),
            )
            error = np.linalg.norm(obs - pred) / max(
                np.linalg.norm(pred), np.linalg.norm(obs), 1e-8
            )
            local_errors[(record["arm"], point["attempt"])].append(error)
    for method, label, color in zip(METHODS, LABELS, COLORS, strict=True):
        steps = sorted(k[1] for k in local_errors if k[0] == method)
        axes[0, 1].plot(
            steps,
            [np.median(local_errors[(method, s)]) for s in steps],
            color=color,
            marker=".",
            label=label,
        )
    axes[0, 1].set(
        xlabel="Edit attempt (actual finite increment)",
        ylabel="Median relative error",
        title="Jacobian prediction versus measured margin change",
    )
    axes[0, 1].set_xscale("symlog", linthresh=1)
    axes[0, 1].set_xlim(0, 512)
    axes[0, 1].legend(fontsize=8)
    stats = read(ART / "direction-confirm-v1/statistics.json")
    differences = stats["prediction_secondary"]["world_differences"]
    axes[1, 0].bar(
        range(8), differences, color=["#209882" if x > 0 else "#bf6254" for x in differences]
    )
    axes[1, 0].axhline(0, color="black", lw=0.7)
    axes[1, 0].set(
        xticks=range(8),
        xticklabels=range(5000, 5008),
        ylabel="Baseline Brier - mechanism Brier",
        title="Prospective prediction: positive favors mechanism",
    )
    dev = read(ART / "direction-formation-v1/summary.json") + read(
        ART / "direction-causal-v1/summary.json"
    )
    train_arms = ("baseline", "up-0.25", "up-freezeearly", "up-freezelate", "up-freezeall")
    for j, kind in enumerate(("coherent", "independent")):
        values = [
            next(
                r["joint"] / r["n"]
                for r in dev
                if r["train_arm"] == a and r["kind"] == kind and r["arm"] == "func-soft-adam"
            )
            for a in train_arms
        ]
        axes[1, 1].bar(np.arange(5) + (j - 0.5) * 0.35, values, 0.35, label=kind)
    axes[1, 1].set(
        xticks=range(5),
        xticklabels=("Baseline", "Early x.25", "Early freeze", "Late freeze", "All freeze"),
        ylabel="Joint success",
        ylim=(0, 0.55),
        title="Development interventions: fixed Adam editor",
    )
    axes[1, 1].tick_params(axis="x", labelsize=8)
    axes[1, 1].legend(fontsize=8)
    save(fig, "mechanism")

    cross = read(ART / "direction-cross-v1/summary.json")
    twostep = csv_rows(ART / "direction-cross-v1/two-step.csv")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    width = 0.19
    for i, kind in enumerate(("coherent", "independent")):
        for j, (field, label) in enumerate(
            (
                ("E_joint", "Direct new facts E"),
                ("D_focal", "One-pass composition D"),
                ("E_local_D_joint", "E + local + D"),
                ("two-step", "Autonomous two-step D"),
            )
        ):
            values = []
            for method in METHODS:
                row = next(r for r in cross if r["arm"] == method and r["kind"] == kind)
                value = (
                    sum(
                        int(r["autonomous"])
                        for r in twostep
                        if r["arm"] == method and r["kind"] == kind
                    )
                    if field == "two-step"
                    else row[field]
                )
                values.append(value / row["n"])
            axes[i].bar(np.arange(4) + (j - 1.5) * width, values, width, label=label)
        axes[i].set(
            xticks=range(4),
            xticklabels=LABELS,
            ylim=(0, 1.2),
            ylabel="Success rate",
            title=f"Eight-layer model: {kind} targets (24 / method)",
        )
        axes[i].tick_params(axis="x", labelsize=8, rotation=15)
    axes[0].legend(fontsize=8, loc="upper left")
    for i, kind in enumerate(("coherent", "independent")):
        values = [
            next(r["U_broken"] / r["U_known"] for r in cross if r["arm"] == m and r["kind"] == kind)
            for m in METHODS
        ]
        axes[2].bar(np.arange(4) + (i - 0.5) * 0.35, values, 0.35, label=kind)
    axes[2].set(
        xticks=range(4),
        xticklabels=LABELS,
        ylabel="Newly broken / formerly correct U",
        title="Broad retention damage remains",
    )
    axes[2].tick_params(axis="x", labelsize=8, rotation=15)
    axes[2].legend(fontsize=8)
    save(fig, "composition")

    calibration = []
    for epsilon in (0, 0.03, 0.1, 0.3, 1):
        phi = np.array([[1.0, epsilon], [1.0, -epsilon]])
        basis = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        rates = np.diag(basis @ (phi @ phi.T) @ basis.T)
        expected = np.array([2, 2 * epsilon**2])
        assert np.allclose(rates, expected, atol=1e-14)
        calibration.append(
            dict(
                epsilon=epsilon,
                measured_rates=rates.tolist(),
                theoretical_rates=expected.tolist(),
                contraction_gd_eta_01=(1 - 0.1 * rates).tolist(),
            )
        )
    write_json(DEST / "linear-instrument-calibration.json", calibration)

    parents = []
    for phase in ("formation", "confirm", "causal"):
        base = ROOT / f"results/bios-direction-{phase}-v1"
        for path in sorted(base.glob("world-*/*/learning.json")):
            receipt = read(path.parent / "complete.json")
            for name, sha in receipt["files"].items():
                assert digest(path.parent / name) == sha
            history = read(path)
            parents.append(
                dict(
                    phase=phase,
                    path=str(path.relative_to(ROOT)),
                    old_accuracy=history[-1]["accuracy"],
                    max_gradient_reconstruction_error=max(
                        f["reconstruction_error"] for r in history for f in r["factors"]
                    ),
                )
            )
    assert len(parents) == 112
    write_json(DEST / "training-audit.json", dict(parents=parents, all_receipts_valid=True))
    locks = []
    for phase in (
        "direction-v1",
        "direction-cross-v1",
        "direction-formation-v1",
        "direction-confirm-v1",
        "direction-causal-v1",
        "direction-routing-v1",
        "direction-supplement-v1",
    ):
        lock = read(ART / phase / "lock.json")
        for name, sha in lock["files"].items():
            assert digest(ROOT / name) == sha, (phase, name)
        locks.append(dict(phase=phase, input_files_verified=len(lock["files"])))
    write_json(DEST / "frozen-input-audit.json", locks)
    print(json.dumps(dict(confirmation_totals=totals, training_parents=len(parents))), flush=True)


def supplement():
    common_style()
    base = ROOT / "results/bios-direction-supplement-v1"
    summarize(base / "rank32", ART / "direction-supplement-v1/rank32", 384)
    from analyze_bios_direction_confirmation import world_test

    original = csv_rows(ART / "direction-confirm-v1/cases.csv")
    higher = csv_rows(ART / "direction-supplement-v1/rank32/cases.csv")
    rank_results = []
    for method in ("func-hard-gn", "func-soft-gn"):
        world_rates = {}
        for name, rows, arm in (
            ("rank8", original, method),
            ("rank32", higher, method),
            ("adam", original, "func-soft-adam"),
        ):
            world_rates[name] = np.array(
                [
                    np.mean(
                        [int(r["joint"]) for r in rows if r["arm"] == arm and int(r["world"]) == w]
                    )
                    for w in range(5000, 5008)
                ]
            )
        rank_results.append(
            dict(
                method=method,
                rates={k: float(v.mean()) for k, v in world_rates.items()},
                rank32_minus_rank8=world_test(world_rates["rank32"] - world_rates["rank8"]),
                rank32_minus_adam=world_test(world_rates["rank32"] - world_rates["adam"]),
            )
        )
    write_json(ART / "direction-supplement-v1/rank-sensitivity.json", rank_results)
    times, budgets = [], []
    for directory in sorted((base / "timing").glob("repeat-*-seed-*")):
        reference = read(directory / "func-soft-adam/timing.json")
        for method in CROSS_ARMS:
            folder = directory / method
            assert validate_batch(folder / "metrics.json") == 9
            timing = read(folder / "timing.json")
            timing["ratio_to_adam"] = timing["total_seconds"] / reference["total_seconds"]
            times.append(timing)
            records = read(folder / "metrics.json")
            for multiplier in (0.25, 0.5, 1, 2, 4):
                budget = reference["total_seconds"] * multiplier
                for record in records:
                    nodes = [
                        r
                        for r in record["timeline"]
                        if r["seconds"] + timing["setup_seconds"] <= budget
                    ]
                    # Pre-edit behavior is available if even the first measured node is too costly.
                    point = (
                        max(nodes, key=lambda r: r["attempt"]) if nodes else record["timeline"][0]
                    )
                    budgets.append(
                        dict(
                            repeat=timing["repeat"],
                            seed=timing["seed"],
                            arm=method,
                            person=record["task"]["metadata"]["person"],
                            kind=record["task"]["metadata"]["kind"],
                            multiplier=multiplier,
                            attempt=point["attempt"],
                            joint=point["joint"],
                            E=point["e_joint"],
                            U_broken=point["sets"]["U"]["broken"],
                            censored=budget > timing["total_seconds"],
                            measured_seconds=point["seconds"] + timing["setup_seconds"],
                        )
                    )
    assert len(times) == 16
    write_json(ART / "direction-supplement-v1/timing.json", times)
    write_json(ART / "direction-supplement-v1/budget-points.json", budgets)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    timing_summary = []
    for i, (method, label, color) in enumerate(zip(METHODS, LABELS, COLORS, strict=True)):
        row = [r for r in times if r["arm"] == method]
        values = [r["ratio_to_adam"] for r in row]
        axes[0].scatter([i] * len(values), values, color=color)
        axes[0].plot([i - 0.2, i + 0.2], [np.median(values)] * 2, color=color, lw=3)
        timing_summary.append(
            dict(
                method=method,
                median_ratio=float(np.median(values)),
                ratio_range=[min(values), max(values)],
                setup_seconds=[r["setup_seconds"] for r in row],
            )
        )
        x, y = [], []
        for multiplier in (0.25, 0.5, 1, 2, 4):
            rows = [r for r in budgets if r["arm"] == method and r["multiplier"] == multiplier]
            rate = np.mean([r["joint"] for r in rows])
            x.append(multiplier)
            y.append(rate)
            axes[1].scatter(
                multiplier,
                rate,
                edgecolor=color,
                facecolor="none" if any(r["censored"] for r in rows) else color,
            )
        axes[1].plot(x, y, color=color, label=label)
    axes[0].set(
        xticks=range(4),
        xticklabels=LABELS,
        ylabel="Synchronized batch time / Adam time",
        title="Original miniature: two serial replays / seed",
    )
    axes[0].tick_params(axis="x", labelsize=8, rotation=15)
    axes[1].set(
        xscale="log",
        xlabel="Budget / Adam 1024-step time",
        ylabel="Last measured joint success",
        title="Open markers: at least one trajectory censored",
    )
    axes[1].legend(fontsize=8)
    corrected_path = ART / "direction-refresh-v1/statistics.json"
    corrected_rows = read(corrected_path) if corrected_path.exists() else None
    for i, record in enumerate(rank_results):
        for j, name in enumerate(("adam", "rank8", "rank32")):
            value = record["rates"][name]
            if name == "rank32" and corrected_rows is not None:
                value = next(
                    r["rates"]["corrected_rank32"]
                    for r in corrected_rows
                    if r["method"] == record["method"]
                )
            axes[2].bar(
                i + (j - 1) * 0.22,
                value,
                0.22,
                label=name if i == 0 else None,
                color=("#3569a8", "#d48627", "#209882")[j],
            )
    axes[2].set(
        xticks=[0, 1],
        xticklabels=["Hard GN", "Soft GN"],
        ylim=(0, 0.55),
        ylabel="Joint success",
        title=(
            "Rank 32 with per-trajectory refresh"
            if corrected_rows is not None
            else "Rank sensitivity: v1 batched refresh"
        ),
    )
    axes[2].legend(fontsize=8)
    save(fig, "cost-and-rank")
    write_json(ART / "direction-supplement-v1/timing-summary.json", timing_summary)
    print(json.dumps(dict(rank=rank_results, timing=timing_summary)), flush=True)


def refresh():
    from analyze_bios_direction_confirmation import world_test

    root = ROOT / "results/bios-direction-refresh-v1/rank32"
    out = ART / "direction-refresh-v1"
    summarize(root, out, 384)
    original = csv_rows(ART / "direction-confirm-v1/cases.csv")
    higher = csv_rows(ART / "direction-supplement-v1/rank32/cases.csv")
    corrected = csv_rows(out / "cases.csv")
    results = []
    for method in ("func-hard-gn", "func-soft-gn"):
        rates = {}
        for name, rows, arm in (
            ("rank8", original, method),
            ("original_rank32", higher, method),
            ("corrected_rank32", corrected, method),
            ("adam", original, "func-soft-adam"),
        ):
            rates[name] = np.array(
                [
                    np.mean(
                        [int(r["joint"]) for r in rows if r["arm"] == arm and int(r["world"]) == w]
                    )
                    for w in range(5000, 5008)
                ]
            )
        results.append(
            dict(
                method=method,
                rates={k: float(v.mean()) for k, v in rates.items()},
                corrected_rank32_minus_adam=world_test(rates["corrected_rank32"] - rates["adam"]),
                corrected_rank32_minus_rank8=world_test(rates["corrected_rank32"] - rates["rank8"]),
            )
        )
    write_json(out / "statistics.json", results)
    print(json.dumps(results), flush=True)
    supplement()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("core", "supplement", "refresh"))
    args = parser.parse_args()
    {"core": core, "supplement": supplement, "refresh": refresh}[args.stage]()
