"""Report the frozen shared-cache matrix, including negative and failed branches."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from llm_memory_editability.grok_depth import write_json

ROOT = Path("/ossfs/workspace/llm-memory-editability")
ARMS = ("local", "shared_full", "shared_window", "shared_detached")


def read(path):
    return json.loads(Path(path).read_text())


def report(config):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = ROOT / "results" / config["batch"]
    output = results / "report"
    output.mkdir(exist_ok=True)
    records, matched, interventions, editing, missing = [], [], [], [], []
    exposures, case_hashes = {}, {}
    for spec in config["specs"]:
        folder = results / "runs" / spec["name"]
        if not (folder / "worker-completion.json").exists():
            missing.append(spec["name"])
            continue
        complete = read(folder / "complete.json")
        assert read(folder / "audit.json")["passed"]
        assert read(folder / "intervention-audit.json")["passed"]
        assert complete["data_sha256"] == spec["data_sha256"]
        assert complete["initial_model_sha256"] == spec["initial_model_sha256"]
        assert complete["parameters"] == spec["parameters"]
        with np.load(folder / "exposures.npz") as raw:
            current = {k: raw[k].copy() for k in raw.files}
        if spec["world"] in exposures:
            for key in current:
                np.testing.assert_array_equal(current[key], exposures[spec["world"]][key])
        exposures[spec["world"]] = current
        metrics = complete["endpoint"]["metrics"]
        row = dict(
            name=spec["name"],
            phase=spec["phase"],
            world=spec["world"],
            arm=spec["memory_arm"],
            train_R=spec["repeats"],
            parameters=complete["parameters"],
            steps=spec["steps"],
            flops=complete["endpoint"]["estimated_training_flops"],
            training_seconds=complete["training_seconds"],
            atomic=metrics["common_atomic"]["accuracy"],
            train=metrics["train_composite"]["accuracy"],
            familiar=metrics["familiar_test"]["accuracy"],
            strict=metrics["strict_test"]["accuracy"],
            coverage=metrics["familiar_test"]["atomic_correct_coverage"],
            two_calls=metrics["familiar_test"]["autonomous_two_calls"],
        )
        records.append(row)
        node = next(
            r for r in read(folder / "learning.json") if r["step"] == spec["matched_compute_step"]
        )
        matched.append(
            dict(
                name=spec["name"],
                step=node["step"],
                budget=spec["reference_compute_budget"],
                flops=node["estimated_training_flops"],
                metrics=node["metrics"],
            )
        )
        interventions.append(dict(name=spec["name"], metrics=read(folder / "interventions.json")))
    for spec in config["editing_specs"]:
        folder = results / "runs" / spec["name"]
        if not (folder / "worker-completion.json").exists():
            missing.append(spec["name"])
            continue
        assert read(folder / "audit.json")["passed"]
        manifest = read(folder / "run.json")
        assert manifest["case_sha256"] == spec["case_sha256"]
        if spec["world"] in case_hashes:
            assert case_hashes[spec["world"]] == manifest["case_sha256"]
        case_hashes[spec["world"]] = manifest["case_sha256"]
        for branch in read(folder / "complete.json")["branches"]:
            editing.append(
                dict(
                    name=spec["name"],
                    phase=spec["phase"],
                    world=spec["world"],
                    memory_arm=spec["memory_arm"],
                    **branch,
                )
            )
    paired = []
    for world in sorted({r["world"] for r in records}):
        for repeat in sorted({r["train_R"] for r in records if r["world"] == world}):
            group = {r["arm"]: r for r in records if r["world"] == world and r["train_R"] == repeat}
            if not set(ARMS) <= set(group):
                continue
            for arm in ARMS[1:]:
                paired.append(
                    dict(
                        world=world,
                        train_R=repeat,
                        arm=arm,
                        phase=group[arm]["phase"],
                        **{
                            key + "_delta_pp": 100 * (group[arm][key] - group["local"][key])
                            for key in ["atomic", "train", "familiar", "strict"]
                        },
                    )
                )
    result = dict(
        passed=not missing,
        registered_training_runs=len(config["specs"]),
        registered_edit_parents=len(config["editing_specs"]),
        missing=missing,
        endpoints=records,
        paired_differences=paired,
        matched_compute=matched,
        inference_interventions=interventions,
        edit_branches=editing,
        independent_confirmation_worlds=3,
        exposures_identical_within_world=True,
        graph_edit_cases_identical_within_world=True,
        limitations=[
            "Artificial relation graphs and fixed sentence templates.",
            "Three confirmation worlds, one initialization per world.",
            "Strict facts have no composition practice; floor effects possible.",
            "Masked dense attention still executes doubled key dimensions.",
            "Matched FLOPs nodes inherit the full-run learning-rate schedule.",
            "Detached backward FLOPs use the same approximate convention.",
            "Inference ablation may leave the learned state distribution.",
            "Local MLP editing uses finite replay KL, not exact preservation.",
        ],
    )
    write_json(output / "summary.json", result)
    if records:
        with (output / "endpoints.csv").open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    confirmation = [r for r in records if r["phase"] == "confirmation"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, key in zip(axes, ["atomic", "familiar", "strict"], strict=True):
        for i, arm in enumerate(ARMS):
            values = [100 * r[key] for r in confirmation if r["arm"] == arm]
            if values:
                ax.bar(i, np.mean(values), alpha=0.65)
                ax.scatter([i] * len(values), values, color="black", s=22)
        ax.set(
            xticks=range(4),
            xticklabels=["Local", "Shared", "Window", "Detached"],
            ylabel="Answer + punctuation + EOS (%)",
            title=key,
        )
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Native R=4 | fixed 128k updates | dots are independent confirmation worlds")
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(output / f"confirmation.{ext}", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, repeat in zip(axes, [2, 4], strict=True):
        for arm in ARMS:
            spec = next(
                s
                for s in config["specs"]
                if s["phase"] == "development" and s["repeats"] == repeat and s["memory_arm"] == arm
            )
            path = results / "runs" / spec["name"] / "learning.json"
            if path.exists():
                rows = read(path)
                ax.plot(
                    [r["step"] for r in rows],
                    [100 * r["metrics"]["familiar_test"]["accuracy"] for r in rows],
                    label=arm,
                )
        ax.set(
            xlabel="Optimizer updates",
            ylabel="Familiar held-out composition (%)",
            title=f"Development R={repeat}",
            ylim=(-2, 103),
        )
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(output / f"development-curves.{ext}", dpi=180)
    plt.close(fig)
    lines = [
        "# Shared cache branch: frozen comparison",
        "",
        f"Complete: {not missing}; missing jobs: {missing}",
        "",
        "All scores use native training R at fixed endpoints. "
        "Development and confirmation are separate.",
        "",
        "| Phase | World | R | Arm | Atomic | Familiar | Strict |",
        "|---|---:|---:|---|---:|---:|---:|",
    ]
    for r in records:
        lines.append(
            f"| {r['phase']} | {r['world']} | {r['train_R']} | {r['arm']} | "
            f"{100 * r['atomic']:.2f}% | {100 * r['familiar']:.2f}% | "
            f"{100 * r['strict']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "Editing, prerequisites, old-answer retention and unrelated facts are retained "
            "in summary.json and independently reloaded per-case predictions.",
            "",
            *["- " + x for x in result["limitations"]],
        ]
    )
    (output / "report.md").write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "training": len(records),
                "edit_branches": len(editing),
                "report": str(output),
            }
        )
    )
    if missing:
        raise RuntimeError(f"Incomplete registered matrix: {missing}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    report(read(args.config))
