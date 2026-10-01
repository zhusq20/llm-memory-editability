"""World-level summaries and compact figures for the memory replacement study."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from llm_memory_editability.memory_reuse import sha, write_json


def known_mask(arrays, world):
    keys, pred = arrays["keys"], arrays["logits_B"].argmax(-1)
    mapping = world["mapping_B"]
    n = len(world["train_mask"])
    rows = np.arange(len(keys)) // n * n + mapping[keys]
    return (pred[:, 0] == mapping[keys]) & (pred[rows, 0] == mapping[mapping[keys]])


def mean_records(records, keys):
    return {key: float(np.mean([r[key] for r in records])) for key in keys}


def report(config_path):
    config = json.loads(Path(config_path).read_text())
    artifact = Path(config["lock"]).parent
    audit = json.loads((artifact / "audit.json").read_text())
    assert audit["status"] == "passed"
    assert audit["config_sha256"] == sha(config_path)
    output = artifact / "report"
    output.mkdir(exist_ok=True)
    records, results, curves, arrays, worlds = [], {}, {}, {}, {}
    for world, init, arm in itertools.product(
        config["worlds"], config["initializations"], config["arms"]
    ):
        root = Path(config["output_root"]) / f"w{world}-i{init}-{arm}"
        result = json.loads((root / "summary.json").read_text())
        audited = next(
            r for r in audit["runs"] if (r["world"], r["init"], r["arm"]) == (world, init, arm)
        )
        assert sha(root / "summary.json") == audited["summary_sha256"]
        key = world, init, arm
        results[key] = result
        arrays[key] = dict(np.load(root / "endpoints.npz"))
        worlds[world] = dict(np.load(root / "world.npz"))
        curves[key] = json.loads((root / "reader-curve.json").read_text())
        a, b = result["endpoints"]["A"], result["endpoints"]["B"]
        records.append(
            dict(
                world=world,
                init=init,
                arm=arm,
                A_one=a["one_hop"],
                A_two=a["two_hop"],
                A_train_two=a["two_hop_A_training_heads"],
                A_heldout_two=a["two_hop_A_heldout_heads"],
                B_one=b["one_hop"],
                B_two=b["two_hop"],
                B_changed_two=b["two_hop_changed"],
                B_changed_coverage=b["changed_n"] / b["n"],
                B_known_two=b["two_hop_both_atoms_correct"],
                B_known_coverage=b["both_atoms_correct_coverage"],
                wrong_memory_two=result["endpoints"]["wrong_A_on_B"]["two_hop"],
                clean_B_one=result["memories"]["B"]["metrics"]["atomic_accuracy"],
                B_output_target_cosine=result["memories"]["B"]["metrics"]["target_cosine"],
                seconds=result["seconds"],
            )
        )
    with (output / "endpoints.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    metrics = [k for k, v in records[0].items() if isinstance(v, float) and k != "seconds"]
    means, world_means = {}, []
    for world, arm in itertools.product(config["worlds"], config["arms"]):
        rows = [r for r in records if r["world"] == world and r["arm"] == arm]
        world_means.append(dict(world=world, arm=arm, **mean_records(rows, metrics)))
    for arm in config["arms"]:
        means[arm] = mean_records([r for r in world_means if r["arm"] == arm], metrics)
    paired = []
    for world, init in itertools.product(config["worlds"], config["initializations"]):
        ce, aligned = (
            next(r for r in records if (r["world"], r["init"], r["arm"]) == (world, init, arm))
            for arm in ("ce", "aligned")
        )
        c_arrays, a_arrays = arrays[world, init, "ce"], arrays[world, init, "aligned"]
        assert np.array_equal(c_arrays["inputs"], a_arrays["inputs"])
        common = known_mask(c_arrays, worlds[world]) & known_mask(a_arrays, worlds[world])
        targets = worlds[world]["mapping_B"]
        keys = c_arrays["keys"]
        labels = targets[targets[keys]]
        common_scores = {}
        for arm, values in (("ce", c_arrays), ("aligned", a_arrays)):
            correct = values["logits_B"][:, 1].argmax(-1) == labels
            common_scores[arm] = float(correct[common].mean()) if common.any() else None
        paired.append(
            dict(
                world=world,
                init=init,
                A_two_effect=aligned["A_two"] - ce["A_two"],
                B_one_effect=aligned["B_one"] - ce["B_one"],
                B_two_effect=aligned["B_two"] - ce["B_two"],
                transfer_drop_ce=ce["B_two"] - ce["A_two"],
                transfer_drop_aligned=aligned["B_two"] - aligned["A_two"],
                common_atomic_coverage=float(common.mean()),
                common_atomic_two=common_scores,
            )
        )
    world_effects = []
    for world in config["worlds"]:
        rows = [r for r in paired if r["world"] == world]
        world_effects.append(
            dict(
                world=world,
                **mean_records(
                    rows, ["A_two_effect", "B_one_effect", "B_two_effect", "common_atomic_coverage"]
                ),
            )
        )
    summary = dict(
        status="complete",
        phase=config["phase"],
        utc=datetime.now(timezone.utc).isoformat(),
        runs=len(records),
        independent_worlds=len(worlds),
        initializations=config["initializations"],
        means=means,
        world_means=world_means,
        paired=paired,
        world_effects=world_effects,
        audit_sha256=sha(artifact / "audit.json"),
        report_source_sha256=sha(__file__),
        source_lock_sha256=sha(config["lock"]),
        budget={
            k: sum(r["budget"].get(k, 0) for r in results.values())
            for k in (
                "memory_updates",
                "reused_memory_updates",
                "reader_updates",
                "atomic_exposures",
                "reader_input_tokens",
                "reader_supervised_answers",
                "approximate_matmul_flops",
            )
        },
        process_seconds=sum(r["seconds"] for r in results.values()),
        interpretation={
            "primary": "B two-hop after memory replacement; frozen A reader and vocabulary",
            "scope": "one relation applied twice; constructed interface; single-token decoding",
            "not_established": "other relations, natural pretraining, necessary alignment",
            "conditional": "post-treatment common-atomic subset; full pool remains primary",
        },
    )
    write_json(output / "summary.json", summary)

    colors = {"ce": "#4776AE", "aligned": "#D77B3D"}
    labels = {"ce": "Cross-entropy", "aligned": "Cross-entropy + vector alignment"}
    selected = ["A_two", "B_one", "B_two"]
    fig, ax = plt.subplots(figsize=(9, 4.7))
    x, width = np.arange(3), 0.34
    for i, arm in enumerate(config["arms"]):
        center = x + (i - 0.5) * width
        values = np.array([means[arm][m] for m in selected]) * 100
        ax.bar(center, values, width, color=colors[arm], label=labels[arm], alpha=0.85)
        for row in world_means:
            if row["arm"] == arm:
                ax.scatter(center, [row[m] * 100 for m in selected], color="black", s=20, zorder=3)
        tops = np.max(
            [[row[m] * 100 for m in selected] for row in world_means if row["arm"] == arm],
            axis=0,
        )
        for xpos, value, top in zip(center, values, tops, strict=True):
            ax.text(xpos, top + 2.2, f"{value:.1f}", ha="center", fontsize=10)
    ax.set(
        xticks=x,
        xticklabels=["Original A: two hops", "New B: one hop", "New B: two hops"],
        ylabel="Answer accuracy (%)",
        ylim=(0, 116),
        title="Freeze the reader, replace the facts",
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False)
    fig.text(
        0.5,
        0.01,
        "Bars: equal-weight world means. Dots: individual worlds. B has no composition training.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(output / "comparison.png", dpi=180)
    fig.savefig(output / "comparison.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, metric, title in zip(
        axes, ["one_hop", "two_hop"], ["Original A: one hop", "Original A: two hops"], strict=True
    ):
        for arm in config["arms"]:
            all_curves = []
            for world in config["worlds"]:
                row = np.mean(
                    [
                        [c[metric] for c in curves[world, init, arm]]
                        for init in config["initializations"]
                    ],
                    axis=0,
                )
                all_curves.append(row)
                ax.plot(config["spec"]["reader_nodes"], row * 100, color=colors[arm], alpha=0.25)
            ax.plot(
                config["spec"]["reader_nodes"],
                np.mean(all_curves, axis=0) * 100,
                color=colors[arm],
                label=labels[arm],
                linewidth=2.5,
            )
        ax.set(title=title, xlabel="Reader updates", ylim=(0, 103))
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Answer accuracy (%)")
    axes[1].legend(loc="lower right", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(output / "learning.png", dpi=180)
    plt.close(fig)
    print(json.dumps(dict(means=means, world_effects=world_effects), indent=2), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    report(args.config)
