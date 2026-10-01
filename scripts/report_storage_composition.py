"""World-level statistics, exposure checks and artifact hashes for the frozen batch."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from llm_memory_editability.grok_depth import utc, write_json
from llm_memory_editability.storage_composition import ARCHITECTURES, file_hash

ROOT = Path("docs/development-artifacts/storage-composition-v1")


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    manifest, rows, endpoints = [], [], {}
    phases = ["development", "development-matched", "development-long", "confirmation"]
    for phase in phases:
        config_path = Path(f"configs/storage-composition-v1-{phase}.json")
        config = json.loads(config_path.read_text())
        assert config == json.loads((ROOT / phase / "frozen-config.json").read_text())
        for spec in config["specs"]:
            name = f"w{spec['world']}-{spec['architecture']}-{spec['load']}"
            out = Path("results/storage-composition-v1") / phase / name
            complete = json.loads((out / "complete.json").read_text())
            audit = json.loads((out / "audit.json").read_text())
            assert audit["passed"] and audit["step"] == spec["steps"]
            assert complete["spec"] == spec and complete["source"] == config["source"]
            assert complete["data_sha256"] == config["world_data"][str(spec["world"])]
            history = json.loads((out / "learning.json").read_text())
            assert [r["step"] for r in history] == spec["nodes"]
            metrics = complete["endpoint"]["metrics"]
            if phase == "confirmation":
                endpoints[spec["world"], spec["architecture"], spec["load"]] = complete
                rows.append(
                    {
                        "world": spec["world"],
                        "architecture": spec["architecture"],
                        "load": spec["load"],
                        "facts": complete["independent_facts"],
                        "parameters": complete["parameters"],
                        "atomic": metrics["common_atomic"]["accuracy"],
                        "train": metrics["train_composite"]["accuracy"],
                        "familiar": metrics["familiar_test"]["accuracy"],
                        "strict": metrics["strict_test"]["accuracy"],
                        "coverage": metrics["familiar_test"]["atomic_correct_coverage"],
                        "calls": metrics["familiar_test"]["autonomous_two_calls"],
                        "extra_atomic": metrics.get("extra_atomic", {}).get("accuracy"),
                        "supervised_tokens": complete["supervised_tokens"],
                        "training_flops": complete["endpoint"]["estimated_training_flops"],
                        "training_seconds": complete["training_seconds"],
                    }
                )
            hashes = {
                str(p): file_hash(p)
                for p in sorted(out.iterdir())
                if p.suffix in {".json", ".npz", ".pt"}
            }
            manifest.append(
                {
                    "phase": phase,
                    "run": name,
                    "config_sha256": file_hash(config_path),
                    "files": hashes,
                }
            )
    assert len(manifest) == 54 and len(rows) == 30
    worlds = [720001, 720002, 720003]
    for world in worlds:
        reference = None
        for architecture in ARCHITECTURES:
            for load in ("low", "high"):
                out = (
                    Path("results/storage-composition-v1/confirmation")
                    / f"w{world}-{architecture}-{load}"
                )
                exposures = np.load(out / "exposures.npz")
                common = [exposures[f"stratum{i}"] for i in (0, 1)]
                if reference is None:
                    reference = common
                for actual, expected in zip(common, reference, strict=True):
                    np.testing.assert_array_equal(actual, expected)
                assert endpoints[world, architecture, load]["independent_facts"] == (
                    1024 if load == "low" else 6144
                )
    aggregates = []
    for architecture in ARCHITECTURES:
        pairs = [
            [
                endpoints[w, architecture, load]["endpoint"]["metrics"]["familiar_test"]["accuracy"]
                for load in ("low", "high")
            ]
            for w in worlds
        ]
        aggregates.append(
            {
                "architecture": architecture,
                "parameters": endpoints[worlds[0], architecture, "low"]["parameters"],
                "worlds": worlds,
                "per_world_low_high": pairs,
                "low_mean": float(np.mean([p[0] for p in pairs])),
                "high_mean": float(np.mean([p[1] for p in pairs])),
                "per_world_high_minus_low": [p[1] - p[0] for p in pairs],
            }
        )
    write_json(
        ROOT / "confirmation" / "aggregate.json",
        {
            "unit": "three independent worlds, unweighted mean; one initialization",
            "architectures": aggregates,
            "rows": rows,
            "common_atomic_min": min(r["atomic"] for r in rows),
            "train_composite_min": min(r["train"] for r in rows),
            "coverage_min": min(r["coverage"] for r in rows),
            "calls_min": min(r["calls"] for r in rows),
            "extra_atomic_min": min(
                r["extra_atomic"] for r in rows if r["extra_atomic"] is not None
            ),
            "strict_range": [min(r["strict"] for r in rows), max(r["strict"] for r in rows)],
        },
    )
    with (ROOT / "confirmation" / "endpoints.csv").open("w") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fig, axes = plt.subplots(1, 5, figsize=(15, 4), sharey=True)
    for ax, group in zip(axes, aggregates, strict=True):
        for world, pair in zip(worlds, group["per_world_low_high"], strict=True):
            ax.plot([0, 1], np.asarray(pair) * 100, marker="o", label=str(world))
            for index, load in enumerate(("low", "high")):
                trained = endpoints[world, group["architecture"], load]["endpoint"]["metrics"][
                    "train_composite"
                ]["accuracy"]
                if trained < 0.95:
                    ax.annotate(
                        f"Train {trained:.1%}",
                        (index, pair[index] * 100),
                        xytext=(4, 12),
                        textcoords="offset points",
                        fontsize=8,
                        color="red",
                    )
        ax.set_xticks([0, 1], ["1024 facts", "6144 facts"])
        ax.set_title(group["architecture"])
        ax.set_ylim(0, 100)
    axes[0].set_ylabel("Held-out background composition accuracy (%)")
    axes[-1].legend(title="World")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(ROOT / "confirmation" / f"per-world.{suffix}", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for architecture in ("loop1x2", "loop1x3"):
        for load in ("low", "high"):
            counts = [1, 2, 3, 4, 6, 8]
            for ax, task in zip(axes, ("common_atomic", "familiar_test"), strict=True):
                values = [
                    np.mean(
                        [
                            endpoints[w, architecture, load]["repeat_metrics"][str(r)][task][
                                "accuracy"
                            ]
                            for w in worlds
                        ]
                    )
                    * 100
                    for r in counts
                ]
                ax.plot(counts, values, marker="o", label=f"{architecture}, {load}")
                ax.set_xlabel("Inference repeats (same weights)")
                ax.set_ylabel("Full accuracy (%)")
                ax.set_title(task)
                ax.set_ylim(0, 105)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(ROOT / "confirmation" / f"repeat-sweep.{suffix}", dpi=180)
    plt.close(fig)
    write_json(
        ROOT / "completion-manifest.json",
        {
            "finished_utc": utc(),
            "development_runs": 24,
            "confirmation_runs": 30,
            "audited_final_checkpoints": 54,
            "common_exposures_exactly_equal": True,
            "report_source_sha256": file_hash(__file__),
            "runs": manifest,
            "confirmation_supervised_tokens": sum(r["supervised_tokens"] for r in rows),
            "confirmation_training_flops": sum(r["training_flops"] for r in rows),
            "confirmation_training_seconds_sum": sum(r["training_seconds"] for r in rows),
        },
    )
    recovery = json.loads((ROOT / "recovery" / "summary.json").read_text())
    recovery_files = {}
    for load in ("low", "high"):
        out = Path("results/storage-composition-v1/recovery") / f"w720003-standard2-{load}"
        assert json.loads((out / "audit.json").read_text())["passed"]
        recovery_files[load] = {
            str(p): file_hash(p) for p in out.iterdir() if p.suffix in {".json", ".npz", ".pt"}
        }
    write_json(
        ROOT / "recovery" / "completion-manifest.json",
        {
            "exploratory": True,
            "audited_final_checkpoints": 2,
            "summary": recovery,
            "files": recovery_files,
            "training_execution_source_sha256": file_hash(ROOT / "recovery" / "source.py"),
            "revised_audit_source_sha256": file_hash("scripts/diagnose_storage_composition.py"),
        },
    )
    print(json.dumps(aggregates, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
