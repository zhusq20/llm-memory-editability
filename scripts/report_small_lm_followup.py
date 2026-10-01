#!/usr/bin/env python3
"""Independently audit saved generations and plot the full-Qwen development."""

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from safetensors import safe_open


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def normalized(value):
    return re.sub(r"\s+", " ", value.strip().rstrip(".").strip()).casefold()


def grade(text, answer):
    return normalized(text.strip().split("\n")[0]) == normalized(answer)


def shapes(path):
    with safe_open(path, framework="pt", device="cpu") as f:
        return {key: f.get_slice(key).get_shape() for key in f.keys()}


def main(config_path):
    config = read(config_path)
    art, root = Path(config["artifact_root"]), Path(config["output_root"])
    summary = read(art / "summary.json")
    world = {r["id"]: r for r in read(art / "world.json")}
    edit = read(art / "edit-design.json")
    new_world = {r["id"]: r for r in edit["world"]}
    edited = set(edit["edited_ids"])
    template = shapes(Path(config["model_path"]) / "model.safetensors")
    original_config = read(Path(config["model_path"]) / "config.json")
    architecture_keys = (
        "model_type",
        "architectures",
        "num_hidden_layers",
        "hidden_size",
        "intermediate_size",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "rms_norm_eps",
        "rope_theta",
        "tie_word_embeddings",
        "hidden_act",
    )
    checked, autonomous, checkpoint_shapes = 0, 0, 0
    rows_out, endpoints, initial = [], {}, {}
    for arm in config["arms"]:
        for branch in (None, "update", "replay"):
            out = root / ("editing" if branch else "training") / arm
            if branch:
                out /= branch
            key = f"{arm}/{branch or 'training'}"
            steps = config["edit_steps"] if branch else config["steps"]
            gold_world = new_world if branch else world
            model_shapes = shapes(out / "model/model.safetensors")
            assert model_shapes == template, key
            saved_config = read(out / "model/config.json")
            assert all(saved_config[k] == original_config[k] for k in architecture_keys), key
            checkpoint_shapes += len(model_shapes)
            # Each runner already hashes the saved weights after exact GPU reload.
            # Preserve those hashes; avoid another full read of all nine large files.
            for path, expected in summary["runs"][key]["checkpoint_files"].items():
                assert Path(path).is_file() and re.fullmatch(r"[0-9a-f]{64}", expected)
            for node in read(out / "curve.json"):
                rows = read(out / f"predictions-{node['step']:05d}.json")
                if node["step"] == 0:
                    initial[key] = rows
                direct = {(r["id"], r["hop"], r["view"]): r for r in rows if r["mode"] == "direct"}
                for r in rows:
                    truth = gold_world[r["id"]]
                    gold = truth["bridge"] if r["hop"] == 0 else truth["city"]
                    assert r["answer"] == gold
                    assert r["correct"] == grade(r["text"], gold)
                    checked += 1
                    if r["mode"] == "autonomous":
                        first = direct[r["id"], 0, r["view"]]
                        bridge = first["text"].strip().split("\n")[0].strip().rstrip(".").strip()
                        assert r["generated_bridge"] == bridge
                        assert r["first_tokens"] == first["tokens"]
                        assert r["first_correct"] == first["correct"]
                        assert bridge in r["prompt"]
                        autonomous += 1
                if node["step"] == steps:
                    endpoints[key] = rows
            assert len(rows) == 560
            assert read(out / "reload-audit.json")["predictions"] == 560
            cells = {}
            for r in rows:
                pool = (
                    "background"
                    if r["split"] == "background"
                    else ("E" if r["id"] in edited else "U")
                )
                cell = (pool, r["view"], r["hop"], r["mode"])
                cells.setdefault(cell, []).append(r)
            for (pool, view, hop, mode), cell_rows in cells.items():
                row = dict(
                    arm=arm,
                    branch=branch or "training",
                    pool=pool,
                    view=view,
                    hop=hop,
                    mode=mode,
                    n=len(cell_rows),
                    correct=sum(r["correct"] for r in cell_rows),
                    old_correct=sum(
                        grade(r["text"], world[r["id"]]["bridge" if hop == 0 else "city"])
                        for r in cell_rows
                    ),
                )
                row["accuracy"] = row["correct"] / row["n"]
                rows_out.append(row)
    for arm in config["arms"]:
        assert [r["tokens"] for r in initial[f"{arm}/training"]] == [
            r["tokens"] for r in initial["separate/training"]
        ]
        assert [r["tokens"] for r in initial[f"{arm}/update"]] == [
            r["tokens"] for r in initial[f"{arm}/replay"]
        ]
    common = []
    for i in range(config["target_triples"]):
        if all(
            r["correct"]
            for arm in config["arms"]
            for r in endpoints[f"{arm}/training"]
            if r["id"] == i and r["view"] == 0 and r["mode"] == "direct" and r["hop"] < 2
        ):
            common.append(i)
    conditional = {}
    for arm in config["arms"]:
        selected = [
            r
            for r in endpoints[f"{arm}/training"]
            if r["id"] in common and r["mode"] == "direct" and r["view"] == 0 and r["hop"] == 2
        ]
        conditional[arm] = sum(r["correct"] for r in selected) / len(selected) if selected else None
    with (art / "endpoint-cells.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_out[0]))
        writer.writeheader()
        writer.writerows(rows_out)
    audit = dict(
        status="passed",
        node_predictions=checked,
        autonomous_actual_bridge_checks=autonomous,
        tensor_shape_checks=checkpoint_shapes,
        original_architecture_config_checks=9 * len(architecture_keys),
        checkpoint_hash_source="Per-run full SHA256 after exact GPU reload; not reread here",
        endpoint_reloaded_generations=9 * 560,
        paired_initial_generation_checks=6 * 240,
        common_atomic_n=len(common),
        common_atomic_coverage=len(common) / 64,
        common_atomic_two_hop=conditional,
        summary_sha256=digest(art / "summary.json"),
        analysis_source_sha256=digest(__file__),
    )
    (art / "independent-audit.json").write_text(json.dumps(audit, indent=2) + "\n")

    colors = dict(separate="#577590", shuffled="#f4a261", linked="#2a9d8f")
    labels = dict(
        separate="Separate + rehearsal",
        shuffled="Unrelated + rehearsal",
        linked="Related + rehearsal",
    )
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4), constrained_layout=True)
    for arm in config["arms"]:
        curve = read(root / "training" / arm / "curve.json")
        axes[0].plot(
            [r["step"] for r in curve],
            [r["second_hop"] * 100 for r in curve],
            marker="o",
            color=colors[arm],
            label=labels[arm],
        )
        axes[1].plot(
            [r["step"] for r in curve],
            [r["two_hop"] * 100 for r in curve],
            marker="o",
            color=colors[arm],
        )
    axes[0].set(
        title="Independent second-fact recall", xlabel="Training steps", ylabel="Accuracy (%)"
    )
    axes[1].set(title="Untrained target two-hop answers", xlabel="Training steps")
    axes[0].legend(fontsize=8, loc="lower right")
    x = list(range(3))
    for offset, branch, metric, label, color in (
        (-0.27, "update", "v0_E_second", "Updated single fact", "#264653"),
        (0, "update", "v0_E_two", "New two-hop answer", "#2a9d8f"),
        (0.27, "replay", "v0_E_two", "New answer, replay control", "#e9c46a"),
    ):
        values = [100 * summary["runs"][f"{a}/{branch}"]["metrics"][metric] for a in config["arms"]]
        axes[2].bar([v + offset for v in x], values, width=0.25, label=label, color=color)
    axes[2].set(
        title="Only the second fact receives new supervision",
        xticks=x,
        xticklabels=["Separate", "Unrelated", "Related"],
    )
    axes[2].legend(fontsize=8, loc="upper left")
    for ax in axes:
        ax.set_ylim(0, 105)
        ax.grid(axis="y", alpha=0.18)
        ax.set_axisbelow(True)
    fig.suptitle(
        "Qwen3-0.6B-Base | One development world, one seed | 64 targets; 32 updated", fontsize=12
    )
    fig.savefig(art / "comparison.png", dpi=170)
    fig.savefig(art / "comparison.pdf")
    plt.close(fig)
    print(json.dumps(audit), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/small-lm-composition-development-v2.json")
    main(parser.parse_args().config)
