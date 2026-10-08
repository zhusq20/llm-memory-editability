"""Report every registered endpoint and recurrence setting, without best-R selection."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from llm_memory_editability.grok_depth import write_json

ROOT = Path("/ossfs/workspace/llm-memory-editability")


def read(path):
    return json.loads(Path(path).read_text())


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = read(args.config)
    output = ROOT / "results" / config["batch"] / "report"
    output.mkdir(parents=True, exist_ok=True)
    records, paths, matrices = [], {}, {}
    for spec in config["specs"]:
        paths[spec["name"]] = ROOT / "results" / config["batch"] / "runs" / spec["name"]
    for layers, repeat in [(1, r) for r in [1, 2, 3, 4, 8]] + [(2, 1)]:
        name = f"l{layers}-r{repeat}"
        paths[name] = (
            ROOT
            / "results/latent-scaling-v1"
            / (f"w730011-i731011-d128-l{layers}-r{repeat}-nall-s128000")
        )
    reference_exposures = None
    for name, folder in paths.items():
        result = read(folder / "complete.json")
        assert read(folder / "audit.json")["passed"]
        spec, metrics = result["spec"], result["endpoint"]["metrics"]
        assert result["data_sha256"] == config["specs"][0]["data_sha256"]
        with np.load(folder / "exposures.npz") as archive:
            exposures = {key: archive[key] for key in archive.files}
        if reference_exposures is None:
            reference_exposures = exposures
        for key in exposures:
            np.testing.assert_array_equal(exposures[key], reference_exposures[key])
        records.append(
            dict(
                name=name,
                layers=spec["layers"],
                train_R=spec["repeats"],
                historical=folder.parent.name == "latent-scaling-v1",
                parameters=result["parameters"],
                steps=spec["steps"],
                flops=result["endpoint"]["estimated_training_flops"],
                training_seconds=result["training_seconds"],
                atomic=metrics["common_atomic"]["accuracy"],
                train=metrics["train_composite"]["accuracy"],
                familiar=metrics["familiar_test"]["accuracy"],
                strict=metrics["strict_test"]["accuracy"],
                coverage=metrics["familiar_test"]["atomic_correct_coverage"],
                answer_only=metrics["familiar_test"]["answer_accuracy"],
            )
        )
        matrices[name] = result["repeat_metrics"].get("128000", {})
    records.sort(key=lambda r: (r["layers"], r["train_R"]))
    with (output / "endpoints.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    matched = []
    for spec in config["specs"]:
        rows = read(paths[spec["name"]] / "learning.json")
        row = next(r for r in rows if r["step"] == spec["matched_compute_step"])
        matched.append(
            dict(
                name=spec["name"],
                step=row["step"],
                flops=row["estimated_training_flops"],
                budget=config["specs"][0]["reference_compute_budget"],
                metrics=row["metrics"],
            )
        )
    four = [r for r in records if r["layers"] == 4]
    result = dict(
        passed=True,
        independent_worlds=1,
        initializations=1,
        new_runs=len(config["specs"]),
        historical_runs=6,
        endpoints=records,
        matched_compute=matched,
        endpoint_test_R=matrices,
        exposures_identical=True,
        limitations=[
            "Same historical development world, not independent confirmation.",
            "More unique blocks change parameters and residual initialization.",
            "Extra evaluation repeats exceed fixed training depth.",
            "Matched compute nodes retain the full-trajectory LR schedule.",
        ],
    )
    write_json(output / "summary.json", result)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for layers in [1, 2, 4]:
        selected = [r for r in records if r["layers"] == layers]
        for ax, key in zip(axes, ["atomic", "familiar", "strict"], strict=True):
            ax.plot(
                [r["train_R"] for r in selected],
                [100 * r[key] for r in selected],
                "o-",
                label=f"{layers} unique blocks",
            )
            ax.set(
                xlabel="Training repeats R (native evaluation)",
                ylabel="Answer + format (%)",
                title=key,
                ylim=(-2, 103),
                xticks=[1, 2, 3, 4, 6, 8],
            )
            ax.grid(alpha=0.2)
    axes[0].legend()
    fig.suptitle(
        "Fixed 128k updates and identical data exposure | one historical development world"
    )
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(output / f"training-depth.{ext}", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    for row, key in enumerate(["common_atomic", "familiar_test"]):
        for col, layers in enumerate([1, 2, 4]):
            ax = axes[row, col]
            for record in records:
                if record["layers"] != layers or not matrices[record["name"]]:
                    continue
                points = sorted(
                    (int(r), 100 * m[key]["accuracy"]) for r, m in matrices[record["name"]].items()
                )
                ax.plot(*zip(*points, strict=True), "o-", label=f"train R={record['train_R']}")
            ax.set(
                xlabel="Evaluation repeats at fixed endpoint weights",
                ylabel="Answer + format (%)",
                title=f"{layers} unique blocks | {key}",
                ylim=(-2, 103),
                xticks=[1, 2, 4, 8, 12, 16],
            )
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8)
    fig.suptitle("Keep native training depth and test-time extrapolation separate")
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(output / f"evaluation-depth.{ext}", dpi=180)
    plt.close(fig)
    print(json.dumps({"passed": True, "four_block_endpoints": four, "report": str(output)}))


if __name__ == "__main__":
    main()
