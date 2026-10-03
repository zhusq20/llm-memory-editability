"""Audit the frozen confirmation matrix and report paired, world-weighted effects.

Seeds are averaged within each world before worlds receive equal weight. Query
counts are retained as denominators, never used as independent replication units.
This reporting source has its own hash and does not change the training contract.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np

from llm_memory_editability.grok_depth import utc, write_json
from llm_memory_editability.latent_scaling import build_world, run_name
from llm_memory_editability.storage_composition import data_digest, file_hash

WORLDS = (740101, 740102, 740103)
INITIALIZATIONS = (741101, 741102)
ARCHITECTURES = ((1, 1), (1, 2), (1, 3), (2, 1))
COUNTS = (256, "all")
TASKS = ("common_atomic", "anchor_atomic", "train_composite", "familiar_test", "strict_test")
DEFAULT_ARTIFACTS = Path("docs/development-artifacts/latent-confirmation-v1")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def paired_mean(values):
    """Keep undefined conditional scores undefined instead of selecting valid seeds."""
    return mean(values) if values and all(v is not None for v in values) else None


def metric_counts(metrics):
    """Retain all supplied scores, adding auditable counts and format differences."""
    flattened = {}
    for task in TASKS:
        values = metrics[task]
        n = values["n"]
        require(isinstance(n, int) and n > 0, f"Invalid denominator: {task}")
        for key, value in values.items():
            require(value is None or math.isfinite(value), f"Nonfinite {task}.{key}")
            flattened[f"{task}.{key}"] = value
        for field in (
            "accuracy",
            "answer_accuracy",
            "atomic_correct_coverage",
            "autonomous_two_calls",
        ):
            if field not in values:
                continue
            value = values[field]
            require(0 <= value <= 1, f"Invalid proportion: {task}.{field}")
            count = round(value * n)
            require(abs(count - value * n) < 1e-6, f"Nonintegral count: {task}.{field}")
            flattened[f"{task}.{field}_count"] = count
        gap = values["answer_accuracy"] - values["accuracy"]
        require(gap >= -1e-12, f"Full score exceeds answer score: {task}")
        flattened[f"{task}.format_gap"] = gap
        flattened[f"{task}.format_gap_count"] = round(gap * n)
        if "conditional_accuracy" in values:
            covered = flattened[f"{task}.atomic_correct_coverage_count"]
            score = values["conditional_accuracy"]
            require((covered == 0) == (score is None), f"Invalid conditional score: {task}")
            flattened[f"{task}.conditional_n"] = covered
            if score is not None:
                require(0 <= score <= 1, f"Invalid conditional proportion: {task}")
                correct = round(covered * score)
                require(abs(correct - covered * score) < 1e-6, f"Conditional count: {task}")
            else:
                correct = None
            flattened[f"{task}.conditional_accuracy_count"] = correct
    return flattened


def condition_means(rows):
    """Equal-weight seed means within worlds, then equal-weight world means."""
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        groups[(row["composition_count"], row["layers"], row["repeats"])][row["world"]].append(row)
    output = []
    for (count, layers, repeats), worlds in groups.items():
        keys = [k for k in next(iter(worlds.values()))[0] if "." in k]
        world_rows = []
        for world, seeds in sorted(worlds.items()):
            require(
                len(seeds) == len({r["initialization"] for r in seeds}),
                f"Duplicate initialization in world {world}",
            )
            world_rows.append(
                {
                    "world": world,
                    "initializations": [r["initialization"] for r in seeds],
                    "n_initializations": len(seeds),
                    "means": {k: paired_mean([r[k] for r in seeds]) for k in keys},
                    "denominators": {task: [r[f"{task}.n"] for r in seeds] for task in TASKS},
                }
            )
        output.append(
            {
                "composition_count": count,
                "layers": layers,
                "repeats": repeats,
                "n_worlds": len(world_rows),
                "worlds": world_rows,
                "means": {k: paired_mean([w["means"][k] for w in world_rows]) for k in keys},
            }
        )
    return output


def contrast_definitions():
    definitions = []
    for repeat in (2, 3):
        for count in COUNTS:
            definitions.append(
                (
                    f"R{repeat}-R1_n{count}",
                    "secondary",
                    [(count, 1, repeat, 1), (count, 1, 1, -1)],
                )
            )
        definitions.append(
            (
                f"interaction_R{repeat}",
                "primary" if repeat == 2 else "secondary",
                [("all", 1, repeat, 1), ("all", 1, 1, -1), (256, 1, repeat, -1), (256, 1, 1, 1)],
            )
        )
    for count in COUNTS:
        definitions.append(
            (
                f"standard2-Loop2_n{count}",
                "secondary",
                [(count, 2, 1, 1), (count, 1, 2, -1)],
            )
        )
    return definitions


def paired_contrasts(rows, definitions=None):
    """Report every seed effect, within-world effects, and the world-weighted mean."""
    lookup = {}
    for row in rows:
        key = (
            row["world"],
            row["initialization"],
            row["composition_count"],
            row["layers"],
            row["repeats"],
        )
        require(key not in lookup, f"Duplicate endpoint: {key}")
        lookup[key] = row
    pairs = sorted({(r["world"], r["initialization"]) for r in rows})
    fields = [
        k for k in rows[0] if k.split(".", 1)[0] in TASKS and not k.endswith((".n", "_count", "_n"))
    ]
    results = []
    for name, priority, definition in definitions or contrast_definitions():
        for field in fields:
            task, metric = field.split(".", 1)
            metric_priority = (
                "diagnostic"
                if priority == "primary" and (task, metric) != ("familiar_test", "accuracy")
                else priority
            )
            seed_results = []
            for world, initialization in pairs:
                terms = []
                for count, layers, repeats, coefficient in definition:
                    key = (world, initialization, count, layers, repeats)
                    require(key in lookup, f"Missing paired endpoint: {key}")
                    row = lookup[key]
                    terms.append(
                        {
                            "name": row["name"],
                            "coefficient": coefficient,
                            "value": row[field],
                            "n": row[f"{task}.n"],
                            "conditional_n": row.get(f"{task}.conditional_n"),
                        }
                    )
                effect = (
                    sum(t["coefficient"] * t["value"] for t in terms)
                    if all(t["value"] is not None for t in terms)
                    else None
                )
                seed_results.append(
                    {
                        "contrast": name,
                        "priority": metric_priority,
                        "task": task,
                        "metric": metric,
                        "level": "seed",
                        "world": world,
                        "initialization": initialization,
                        "n_worlds": 1,
                        "n_initializations": 1,
                        "effect": effect,
                        "effect_pp": 100 * effect
                        if effect is not None and metric != "answer_nll"
                        else None,
                        "terms": terms,
                    }
                )
            world_results = []
            for world in sorted({r["world"] for r in seed_results}):
                seeds = [r for r in seed_results if r["world"] == world]
                effect = paired_mean([r["effect"] for r in seeds])
                world_results.append(
                    {
                        "contrast": name,
                        "priority": metric_priority,
                        "task": task,
                        "metric": metric,
                        "level": "world",
                        "world": world,
                        "initialization": None,
                        "n_worlds": 1,
                        "n_initializations": len(seeds),
                        "effect": effect,
                        "effect_pp": 100 * effect
                        if effect is not None and metric != "answer_nll"
                        else None,
                        "terms": [
                            {"initialization": r["initialization"], "terms": r["terms"]}
                            for r in seeds
                        ],
                    }
                )
            effect = paired_mean([r["effect"] for r in world_results])
            results.extend(seed_results)
            results.extend(world_results)
            results.append(
                {
                    "contrast": name,
                    "priority": metric_priority,
                    "task": task,
                    "metric": metric,
                    "level": "all_worlds",
                    "world": None,
                    "initialization": None,
                    "n_worlds": len(world_results),
                    "n_initializations": len(seed_results),
                    "effect": effect,
                    "effect_pp": 100 * effect
                    if effect is not None and metric != "answer_nll"
                    else None,
                    "terms": [{"world": r["world"], "terms": r["terms"]} for r in world_results],
                }
            )
    return results


def validate_matrix(specs):
    expected = set(itertools.product(WORLDS, INITIALIZATIONS, COUNTS, ARCHITECTURES))
    actual = [
        (s["world"], s["initialization"], s["composition_count"], (s["layers"], s["repeats"]))
        for s in specs
    ]
    require(
        len(actual) == 48 and len(set(actual)) == 48 and set(actual) == expected,
        "Confirmation must contain exactly the 48 preregistered runs",
    )
    varying = {
        "world",
        "initialization",
        "stream_seed",
        "layers",
        "repeats",
        "composition_count",
        "test_repeats",
    }
    reference = {k: v for k, v in specs[0].items() if k not in varying}
    for spec in specs:
        require(spec["width"] == 128 and spec["steps"] == 128000, "Width or budget changed")
        require(spec["stream_seed"] == 742101 + WORLDS.index(spec["world"]), "Stream seed changed")
        require(
            {k: v for k, v in spec.items() if k not in varying} == reference,
            "A nonexperimental training setting differs between conditions",
        )
        require(
            spec["nodes"] == sorted(set(spec["nodes"]))
            and spec["nodes"][0] == 0
            and spec["nodes"][-1] == spec["steps"],
            "Invalid learning nodes",
        )
        require(
            set(spec["repeat_nodes"]) <= set(spec["checkpoint_nodes"]) <= set(spec["nodes"]),
            "Invalid saved checkpoint nodes",
        )


def expected_data_digest(config, spec):
    if "data" in config:
        return config["data"][f"{spec['world']}:{spec['composition_count']}"]["sha256"]
    hashes = config["data_sha256"]
    if run_name(spec) in hashes:
        return hashes[run_name(spec)]
    if str(spec["world"]) in hashes:
        return hashes[str(spec["world"])][str(spec["composition_count"])]
    return hashes[f"{spec['world']}:{spec['composition_count']}"]


def load_and_verify(config, results, artifacts):
    validate_matrix(config["specs"])
    require("confirmation" in config["phase"], "Configuration is not confirmation")
    require(config["source"], "Missing frozen training sources")
    for path, digest in config["source"].items():
        require(file_hash(path) == digest, f"Training source changed: {path}")
        snapshot = artifacts / "source" / path
        require(
            snapshot.is_file() and file_hash(snapshot) == digest, f"Frozen source missing: {path}"
        )
    rows, histories, repeats, audits, input_hashes = [], {}, {}, {}, {}
    worlds, exposure_references, initial_references = {}, {}, {}
    for spec in config["specs"]:
        name = run_name(spec)
        folder = results / name
        result = json.loads((folder / "complete.json").read_text())
        checked = json.loads((folder / "audit.json").read_text())
        require(
            result["spec"] == spec and result["source"] == config["source"], f"Run contract: {name}"
        )
        require(checked["passed"] and checked["endpoint_tokens_exact"], f"Failed audit: {name}")
        require(
            checked["repeat_checks"] == len(spec["repeat_nodes"]) * len(spec["test_repeats"]),
            f"Incomplete repeat audit: {name}",
        )
        require(
            file_hash(folder / "latest.pt") == result["checkpoint_sha256"],
            f"Checkpoint changed: {name}",
        )
        nodes = json.loads((folder / "learning.json").read_text())
        require(
            [n["step"] for n in nodes] == spec["nodes"] and nodes[-1] == result["endpoint"],
            f"Learning trajectory differs from frozen endpoint: {name}",
        )
        repeat = json.loads((folder / "repeat-metrics.json").read_text())
        require(
            repeat == result["repeat_metrics"]
            and set(repeat) == {str(n) for n in spec["repeat_nodes"]},
            f"Repeat metrics differ: {name}",
        )
        for values in repeat.values():
            require(
                set(values) == {str(r) for r in spec["test_repeats"]}, f"Incomplete sweep: {name}"
            )
        for node in nodes:
            metric_counts(node["metrics"])
            require(
                (folder / f"predictions-{node['step']:06d}.npz").is_file(),
                f"Missing predictions: {name}",
            )
        for node in spec["checkpoint_nodes"]:
            require((folder / f"model-{node:06d}.pt").is_file(), f"Missing checkpoint: {name}")
        world_key = (spec["world"], spec["composition_count"])
        if world_key not in worlds:
            worlds[world_key] = build_world(spec)
        expected = worlds[world_key]
        with np.load(folder / "world.npz", allow_pickle=False) as saved:
            require(set(saved.files) == set(expected), f"World keys changed: {name}")
            for key, value in expected.items():
                np.testing.assert_array_equal(
                    saved[key], value, err_msg=f"World data changed: {name}:{key}"
                )
        digest = data_digest(expected)
        require(
            digest == result["data_sha256"] == expected_data_digest(config, spec),
            f"World hash: {name}",
        )
        pair = (spec["world"], spec["initialization"])
        with np.load(folder / "exposures.npz", allow_pickle=False) as exposures:
            strata = ("common_atomic", "train_composite", "anchor_atomic")
            for index, task in enumerate(strata):
                counts = exposures[f"stratum{index}"]
                require(
                    counts.shape == (len(expected[task]),)
                    and np.issubdtype(counts.dtype, np.integer)
                    and int(counts.sum()) == spec["steps"] * spec["batch_size"] // 3
                    and int(np.ptp(counts)) <= 1,
                    f"Invalid exposure: {name}:{task}",
                )
                if index in (0, 2):
                    exposure_key = (spec["world"], index)
                    exposure_references.setdefault(exposure_key, counts.copy())
                    np.testing.assert_array_equal(
                        counts,
                        exposure_references[exposure_key],
                        err_msg=f"Unmatched atomic exposure: {name}",
                    )
        initial_key = (*pair, spec["width"], spec["layers"])
        initial_references.setdefault(initial_key, result["initial_model_sha256"])
        require(
            initial_references[initial_key] == result["initial_model_sha256"],
            f"Unpaired initialization: {name}",
        )
        row = {
            k: spec[k]
            for k in (
                "world",
                "initialization",
                "width",
                "layers",
                "repeats",
                "composition_count",
                "steps",
            )
        }
        row.update(
            name=name,
            parameters=result["parameters"],
            independent_facts=result["independent_facts"],
            composition_examples=result["composition_examples"],
            training_seconds=result["training_seconds"],
            process_seconds=result["process_seconds"],
            supervised_tokens=result["supervised_tokens"],
            padded_input_tokens=result["padded_input_tokens"],
            training_flops=result["endpoint"]["estimated_training_flops"],
            composition_epochs=result["endpoint"]["composition_epochs"],
            data_sha256=digest,
            initial_model_sha256=result["initial_model_sha256"],
        )
        row.update(metric_counts(result["endpoint"]["metrics"]))
        row.update({f"role_coverage.{k}": v for k, v in result["role_coverage"].items()})
        rows.append(row)
        histories[name], repeats[name], audits[name] = nodes, repeat, checked
        input_hashes[name] = {
            file: file_hash(folder / file)
            for file in (
                "complete.json",
                "audit.json",
                "learning.json",
                "repeat-metrics.json",
                "exposures.npz",
                "world.npz",
            )
        }
    for world in WORLDS:
        small, large = worlds[(world, 256)], worlds[(world, "all")]
        require(set(small) == set(large), "Different support world keys")
        for key in small:
            a, b = (
                (small[key], large[key][:256])
                if key == "train_composite"
                else (small[key], large[key])
            )
            np.testing.assert_array_equal(
                a, b, err_msg=f"Nonexperimental data change: {world}:{key}"
            )
    return rows, histories, repeats, audits, input_hashes


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {
                k: json.dumps(v, sort_keys=True) if isinstance(v, list | dict) else v
                for k, v in row.items()
            }
            for row in rows
        )


def repeat_secondary(rows, repeat_metrics):
    """Descriptive fixed-budget diagnosis added after launch; never choose best R."""
    common_r2, fixed_weights = [], []
    for row in rows:
        if row["layers"] != 1:
            continue
        values = repeat_metrics[row["name"]][str(row["steps"])]
        common_r2.append({**row, **metric_counts(values["2"])})
        if row["repeats"] == 2:
            for tested_repeat in (2, 3):
                fixed_weights.append(
                    {
                        **row,
                        **metric_counts(values[str(tested_repeat)]),
                        "name": f"{row['name']}:testR{tested_repeat}",
                        "repeats": tested_repeat,
                    }
                )
    matched_definitions = [
        (f"testR2_{name}", "descriptive secondary", definition)
        for name, _priority, definition in contrast_definitions()
        if "standard2" not in name
    ]
    fixed_definitions = [
        (
            f"trainedR2_testR3-testR2_n{count}",
            "descriptive secondary",
            [(count, 1, 3, 1), (count, 1, 2, -1)],
        )
        for count in COUNTS
    ]
    return {
        "scope": "Descriptive diagnosis added after training launch, using fixed 128000-update "
        "weights and the previously frozen complete test-R sweep. These summaries do not "
        "select an inference budget by test performance.",
        "uniform_test_R2_conditions": condition_means(common_r2),
        "uniform_test_R2_contrasts": paired_contrasts(common_r2, matched_definitions),
        "trained_R2_fixed_weight_R3_minus_R2": paired_contrasts(fixed_weights, fixed_definitions),
    }


def plots(rows, histories, artifacts):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {(1, 1): "#4c78a8", (1, 2): "#f58518", (1, 3): "#54a24b", (2, 1): "#b279a2"}
    labels = {(1, 1): "R1", (1, 2): "Loop R2", (1, 3): "Loop R3", (2, 1): "Standard 2"}
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharey="row")
    for column, world in enumerate(WORLDS):
        for line, task in enumerate(("familiar_test", "strict_test")):
            ax = axes[line, column]
            for offset, architecture in enumerate(ARCHITECTURES):
                values = []
                x = np.arange(2) + (offset - 1.5) * 0.18
                for index, count in enumerate(COUNTS):
                    seeds = [
                        r
                        for r in rows
                        if r["world"] == world
                        and r["composition_count"] == count
                        and (r["layers"], r["repeats"]) == architecture
                    ]
                    scores = [100 * r[f"{task}.accuracy"] for r in seeds]
                    values.append(mean(scores))
                    ax.scatter(
                        [x[index]] * len(scores),
                        scores,
                        color=colors[architecture],
                        alpha=0.45,
                        s=12,
                    )
                ax.plot(
                    x, values, marker="o", color=colors[architecture], label=labels[architecture]
                )
            ax.set_xticks([0, 1], ["256", "all"])
            ax.set_xlabel("Independent training compositions")
            ax.set_ylabel("Full generated accuracy (%)")
            ax.set_ylim(0, 105 if line == 0 else max(5, ax.get_ylim()[1]))
            ax.set_title(
                f"World {world}: {'familiar new chains' if line == 0 else 'strict role transfer'}"
            )
            ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(artifacts / f"comparison.{extension}", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharey=True)
    for column, world in enumerate(WORLDS):
        for line, count in enumerate(COUNTS):
            ax = axes[line, column]
            for architecture in ARCHITECTURES:
                selected = [
                    r
                    for r in rows
                    if r["world"] == world
                    and r["composition_count"] == count
                    and (r["layers"], r["repeats"]) == architecture
                ]
                nodes = [histories[r["name"]] for r in selected]
                steps = [n["step"] for n in nodes[0]]
                values = [
                    100 * mean(h[index]["metrics"]["familiar_test"]["accuracy"] for h in nodes)
                    for index in range(len(steps))
                ]
                ax.plot(
                    steps,
                    values,
                    marker=".",
                    color=colors[architecture],
                    label=labels[architecture],
                )
            ax.set_xscale("symlog", linthresh=256)
            ax.set_ylim(0, 105)
            ax.set_title(f"World {world}, support {count}")
            ax.set_xlabel("Training updates")
            ax.set_ylabel("Familiar new-chain accuracy (%)")
            ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(artifacts / f"learning.{extension}", dpi=180)
    plt.close(fig)


def report(config_path, results, artifacts):
    config = json.loads(config_path.read_text())
    require(file_hash(artifacts / "design.md") == config["design_sha256"], "Frozen design differs")
    frozen = artifacts / "frozen-config.json"
    require(frozen.is_file() and json.loads(frozen.read_text()) == config, "Frozen config differs")
    rows, histories, repeats, audits, input_hashes = load_and_verify(config, results, artifacts)
    contrasts = paired_contrasts(rows)
    aggregates = condition_means(rows)
    execution_path = artifacts / "execution.json"
    execution = json.loads(execution_path.read_text()) if execution_path.exists() else None
    reporting_hash = file_hash(__file__)
    summary = {
        "created_utc": utc(),
        "phase": "confirmation",
        "expected_runs": 48,
        "audited_runs": len(audits),
        "learning_nodes": sum(map(len, histories.values())),
        "repeat_checks": sum(a["repeat_checks"] for a in audits.values()),
        "unit": "3 independent worlds; 2 paired initializations averaged within each world",
        "aggregation": "Equal seed weights within world, then equal weights across worlds; "
        "no query pooling",
        "primary": "[(R2-R1)all-(R2-R1)256], familiar_test.accuracy at 128000 updates",
        "boundaries": "The interaction jointly varies unique-chain support and "
        "composition-role coverage. "
        "It does not isolate those explanations. Standard2 versus Loop2 matches execution depth, "
        "Independent parameters and initial tensors differ. Equal updates do not match FLOPs.",
        "runs": rows,
        "conditions": aggregates,
        "contrasts": contrasts,
        "histories": histories,
        "repeat_metrics": repeats,
        "repeat_secondary": repeat_secondary(rows, repeats),
        "budget": {
            "supervised_tokens": sum(r["supervised_tokens"] for r in rows),
            "padded_input_tokens": sum(r["padded_input_tokens"] for r in rows),
            "estimated_training_flops": sum(r["training_flops"] for r in rows),
            "summed_training_seconds": sum(r["training_seconds"] for r in rows),
            "summed_process_seconds": sum(r["process_seconds"] for r in rows),
            "execution_wall_seconds": execution.get("seconds") if execution else None,
        },
        "reporting_source_sha256": reporting_hash,
        "training_source": config["source"],
        "execution": execution,
    }
    artifacts.mkdir(parents=True, exist_ok=True)
    write_csv(artifacts / "endpoints.csv", rows)
    write_csv(artifacts / "contrasts.csv", contrasts)
    write_json(artifacts / "summary.json", summary)
    plots(rows, histories, artifacts)
    shutil.copy2(__file__, artifacts / "reporting-source.py")
    outputs = (
        "endpoints.csv",
        "contrasts.csv",
        "summary.json",
        "comparison.png",
        "comparison.pdf",
        "learning.png",
        "learning.pdf",
        "reporting-source.py",
    )
    write_json(
        artifacts / "completion-manifest.json",
        {
            "created_utc": utc(),
            "complete": True,
            "expected_runs": 48,
            "audited_runs": 48,
            "worlds": list(WORLDS),
            "initializations_per_world": list(INITIALIZATIONS),
            "learning_nodes": summary["learning_nodes"],
            "repeat_checks": summary["repeat_checks"],
            "matrix_exact": True,
            "source_exact": True,
            "endpoint_checkpoint_hashes_exact": True,
            "saved_worlds_exact": True,
            "architecture_data_exact": True,
            "support_changes_only_nested_train_composite": True,
            "common_and_anchor_exposures_exact": True,
            "paired_initial_states_exact_within_layers": True,
            "aggregation": summary["aggregation"],
            "budget": summary["budget"],
            "configuration_sha256": file_hash(config_path),
            "training_source": config["source"],
            "reporting_source_sha256": reporting_hash,
            "inputs": input_hashes,
            "outputs": {name: file_hash(artifacts / name) for name in outputs},
        },
    )
    primary = next(
        r
        for r in contrasts
        if r["contrast"] == "interaction_R2"
        and r["task"] == "familiar_test"
        and r["metric"] == "accuracy"
        and r["level"] == "all_worlds"
    )
    print(
        json.dumps(
            {
                "audited_runs": 48,
                "primary_interaction_pp": primary["effect_pp"],
                "artifacts": str(artifacts),
            }
        )
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/latent-confirmation-v1.json"))
    parser.add_argument("--results", type=Path, default=Path("results/latent-confirmation-v1"))
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    args = parser.parse_args()
    report(args.config, args.results, args.artifacts)


if __name__ == "__main__":
    main()
