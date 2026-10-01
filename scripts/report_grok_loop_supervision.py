#!/usr/bin/env python3
"""Summarize all paired outcomes, worlds first; never select a best test loop."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from llm_memory_editability.grok_depth import write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    items, rows, pairs = {}, [], []
    for run_id, details in cfg["runs"].items():
        directory = Path(cfg["output_root"]) / run_id
        if not (directory / "complete.json").exists():
            raise ValueError(f"Missing endpoint: {run_id}")
        complete = json.loads((directory / "complete.json").read_text())
        learning = json.loads((directory / "learning.json").read_text())
        spec = {**cfg["base"], **details}
        if complete["spec"] != spec:
            raise ValueError("Changed specification")
        key = (details["world"], details["initialization"])
        items.setdefault(key, {})[details["arm"]] = (complete, learning, directory)
        for node in learning:
            for group, metric in node["evaluations"].items():
                split, repeat = group.rsplit("_r", 1)
                rows.append(
                    {
                        "run": run_id,
                        "world": key[0],
                        "initialization": key[1],
                        "arm": details["arm"],
                        "step": node["step"],
                        "split": split,
                        "repeat": int(repeat),
                        **metric,
                    }
                )
    for (world, init), arms in items.items():
        if set(arms) != {"single", "multi"}:
            raise ValueError("Incomplete pair")
        left, right = [arms[a][0] for a in ("single", "multi")]
        for field in ("initial_model_sha256", "exposure_sha256"):
            if left[field] != right[field]:
                raise ValueError(f"Pair mismatch: {field}")
        if (
            left["segment_start"] == right["segment_start"]
            and left["sampling_segment_sha256"] != right["sampling_segment_sha256"]
        ):
            raise ValueError("Pair batch sequence mismatch")
        steps = [[n["step"] for n in arms[a][1]] for a in ("single", "multi")]
        if steps[0] != steps[1]:
            raise ValueError("Pair evaluation nodes differ")
        for a, b in zip(arms["single"][1], arms["multi"][1], strict=True):
            e1, e2 = a["evaluations"], b["evaluations"]
            for split in ("atomic", "test_full_composite", "ood_composite"):
                av = [e1[f"{split}_r{r}"]["accuracy"] for r in (4, 8)]
                bv = [e2[f"{split}_r{r}"]["accuracy"] for r in (4, 8)]
                pairs.append(
                    {
                        "world": world,
                        "initialization": init,
                        "step": a["step"],
                        "split": split,
                        "single_r4": av[0],
                        "single_r8": av[1],
                        "multi_r4": bv[0],
                        "multi_r8": bv[1],
                        "r4_effect": bv[0] - av[0],
                        "r8_effect": bv[1] - av[1],
                        "single_change_8_minus_4": av[1] - av[0],
                        "multi_change_8_minus_4": bv[1] - bv[0],
                        "change_effect": (bv[1] - bv[0]) - (av[1] - av[0]),
                    }
                )
    transitions = []
    for (world, init), arms in items.items():
        baseline_masks = []
        for arm in ("single", "multi"):
            directory = arms[arm][2]
            with np.load(directory / "predictions-0000000.npz") as z:
                key = "ood_composite_r4_"
                baseline = (z[key + "answer"] == z[key + "target"]) & (z[key + "stop"] == 1)
            baseline_masks.append(baseline)
            with np.load(directory / f"predictions-{cfg['base']['steps']:07d}.npz") as z:
                correct = []
                for r in (4, 8):
                    key = f"ood_composite_r{r}_"
                    correct.append(
                        (z[key + "answer"] == z[key + "target"]) & (z[key + "stop"] == 1)
                    )
                c4, c8 = correct
                transitions.append(
                    {
                        "world": world,
                        "initialization": init,
                        "arm": arm,
                        "n": len(c4),
                        "correct_to_correct": int((c4 & c8).sum()),
                        "correct_to_wrong": int((c4 & ~c8).sum()),
                        "wrong_to_correct": int((~c4 & c8).sum()),
                        "wrong_to_wrong": int((~c4 & ~c8).sum()),
                        "source_r4_correct_n": int(baseline.sum()),
                        "source_r4_correct_coverage": float(baseline.mean()),
                        "r4_on_source_correct": float(c4[baseline].mean())
                        if baseline.any()
                        else None,
                        "r8_on_source_correct": float(c8[baseline].mean())
                        if baseline.any()
                        else None,
                    }
                )
        if not np.array_equal(*baseline_masks):
            raise ValueError("Different baseline selection in paired arms")
    worlds, summary = [], {}
    metrics = [k for k in pairs[0] if k not in ("world", "initialization", "step", "split")]
    for world in sorted({p["world"] for p in pairs}):
        for split in ("atomic", "test_full_composite", "ood_composite"):
            selected = [
                p
                for p in pairs
                if p["world"] == world and p["split"] == split and p["step"] == cfg["base"]["steps"]
            ]
            worlds.append(
                {
                    "world": world,
                    "split": split,
                    "initializations": len(selected),
                    **{m: float(np.mean([p[m] for p in selected])) for m in metrics},
                }
            )
    for split in ("atomic", "test_full_composite", "ood_composite"):
        selected = [p for p in worlds if p["split"] == split]
        summary[split] = {m: float(np.mean([p[m] for p in selected])) for m in metrics}
    for name, values in (
        ("nodes", rows),
        ("paired", pairs),
        ("worlds", worlds),
        ("transitions", transitions),
    ):
        with (output / f"{name}.csv").open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    write_json(
        output / "summary.json",
        {
            "experiment": cfg["experiment"],
            "runs": len(cfg["runs"]),
            "worlds": len({p["world"] for p in pairs}),
            "steps": cfg["base"]["steps"],
            "summary": summary,
            "world_results": worlds,
            "training_seconds": sum(
                item[0]["training_seconds"] for arms in items.values() for item in arms.values()
            ),
            "pair_initialization_and_exposure_audit": True,
            "interpretation": (
                "Paired continuation on existing worlds. Training-objective effect; "
                "not proof of fixed-point dynamics."
            ),
        },
    )
    print(json.dumps(summary, indent=2), flush=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    colors = {"single": "#b54a42", "multi": "#2673aa"}
    for split, ax in (("ood_composite", axes[1]), ("atomic", axes[2])):
        baseline = [
            100
            * np.mean(
                [
                    v["accuracy"]
                    for v in rows
                    if v["arm"] == "single"
                    and v["step"] == 0
                    and v["split"] == split
                    and v["repeat"] == r
                ]
            )
            for r in cfg["base"]["endpoint_repeats"]
        ]
        ax.plot(
            cfg["base"]["endpoint_repeats"],
            baseline,
            color="#777777",
            ls=":",
            label="before continuation",
        )
    for arm in ("single", "multi"):
        selected = [r for r in rows if r["arm"] == arm and r["split"] == "ood_composite"]
        for r, style in ((4, "--"), (8, "-")):
            points = []
            for step in cfg["base"]["nodes"]:
                perworld = [
                    np.mean(
                        [
                            v["accuracy"]
                            for v in selected
                            if v["world"] == w and v["repeat"] == r and v["step"] == step
                        ]
                    )
                    for w in sorted({x["world"] for x in selected})
                ]
                points.append(100 * np.mean(perworld))
            axes[0].plot(
                cfg["base"]["nodes"], points, style, color=colors[arm], label=f"{arm} R{r}"
            )
        for split, ax in (("ood_composite", axes[1]), ("atomic", axes[2])):
            curve = []
            for r in cfg["base"]["endpoint_repeats"]:
                perworld = [
                    np.mean(
                        [
                            v["accuracy"]
                            for v in rows
                            if v["world"] == w
                            and v["arm"] == arm
                            and v["repeat"] == r
                            and v["step"] == cfg["base"]["steps"]
                            and v["split"] == split
                        ]
                    )
                    for w in sorted({x["world"] for x in rows})
                ]
                curve.append(100 * np.mean(perworld))
            ax.plot(
                cfg["base"]["endpoint_repeats"], curve, marker=".", color=colors[arm], label=arm
            )
    for ax, title in zip(
        axes, ("OOD learning", "OOD by evaluation loops", "Atomic by evaluation loops"), strict=True
    ):
        ax.set_title(title)
        ax.set_ylabel("Answer + EOS accuracy (%)")
        ax.set_ylim(-2, 102)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    axes[0].set_xlabel("Continuation steps")
    for ax in axes[1:]:
        ax.set_xlabel("Evaluation loops")
        ax.axvline(4, color="gray", ls=":", lw=1)
    fig.tight_layout()
    fig.savefig(output / "comparison.png", dpi=180)
    fig.savefig(output / "comparison.pdf")


if __name__ == "__main__":
    main()
