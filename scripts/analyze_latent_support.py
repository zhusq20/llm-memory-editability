"""Report the frozen connected-versus-split support comparison without query pooling.

The primary score is full generated familiar-test accuracy at trained/test R2 and
128000 updates. Graph strata describe the fixed test pool; they are
neither new independent replications nor a replacement primary score.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from collections import Counter, defaultdict, deque
from pathlib import Path
from statistics import mean

import numpy as np

from llm_memory_editability.grok_depth import utc, write_json
from llm_memory_editability.storage_composition import data_digest, file_hash

WORLDS = (740101, 740102, 740103)
INITIALIZATIONS = (741101, 741102)
SUPPORTS = ("connected", "split")
TASKS = ("common_atomic", "anchor_atomic", "train_composite", "familiar_test", "strict_test")
SUBSETS = ("both_same_component", "connected_only_same_component", "neither_same_or_uncovered")
DEFAULT_ARTIFACTS = Path("docs/development-artifacts/latent-support-v1")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def paired_mean(values):
    """Undefined subset scores remain undefined rather than dropping seeds/worlds."""
    return mean(values) if values and all(v is not None for v in values) else None


def metric_counts(metrics, *, allow_empty=False):
    flattened = {}
    for task, values in metrics.items():
        require(task in TASKS, f"Unknown task: {task}")
        n = values["n"]
        require(isinstance(n, int) and n >= (0 if allow_empty else 1), f"Invalid n: {task}")
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
            require((value is None) == (n == 0), f"Undefined proportion: {task}.{field}")
            if value is None:
                count = 0
            else:
                require(0 <= value <= 1, f"Invalid proportion: {task}.{field}")
                count = round(value * n)
                require(abs(count - value * n) < 1e-6, f"Nonintegral count: {task}.{field}")
            flattened[f"{task}.{field}_count"] = count
        a, full = values["answer_accuracy"], values["accuracy"]
        gap = None if n == 0 else a - full
        require(gap is None or gap >= -1e-12, f"Full score exceeds answer score: {task}")
        flattened[f"{task}.format_gap"] = gap
        flattened[f"{task}.format_gap_count"] = round(gap * n) if gap is not None else 0
        if "conditional_accuracy" in values:
            covered = flattened[f"{task}.atomic_correct_coverage_count"]
            score = values["conditional_accuracy"]
            require((score is None) == (covered == 0), f"Invalid conditional score: {task}")
            flattened[f"{task}.conditional_n"] = covered
            if score is not None:
                require(0 <= score <= 1, f"Invalid conditional proportion: {task}")
                count = round(score * covered)
                require(abs(count - score * covered) < 1e-6, f"Nonintegral conditional: {task}")
            else:
                count = None
            flattened[f"{task}.conditional_accuracy_count"] = count
    return flattened


def validate_matrix(specs):
    expected = set(itertools.product(WORLDS, INITIALIZATIONS, SUPPORTS))
    actual = [(s["world"], s["initialization"], s["support"]) for s in specs]
    require(
        len(actual) == 12 and len(set(actual)) == 12 and set(actual) == expected,
        "Support comparison must contain exactly the 12 frozen runs",
    )
    varying = {
        "world",
        "initialization",
        "stream_seed",
        "support",
        "composition_indices",
        "frozen_data_sha256",
    }
    reference = {k: v for k, v in specs[0].items() if k not in varying}
    selections = {}
    for spec in specs:
        require(
            (
                spec["width"],
                spec["layers"],
                spec["repeats"],
                spec["steps"],
                spec["composition_count"],
            )
            == (128, 1, 2, 128000, 512),
            "Width, architecture, support count or fixed budget changed",
        )
        require(spec["batch_size"] == 192, "Exposure budget changed")
        require(spec["stream_seed"] == 742101 + WORLDS.index(spec["world"]), "Stream seed changed")
        require(
            {k: v for k, v in spec.items() if k not in varying} == reference,
            "A nonexperimental training setting differs between supports",
        )
        indices = spec["composition_indices"]
        require(
            len(indices) == 512
            and len(set(indices)) == 512
            and all(type(i) is int and i >= 0 for i in indices),
            "Selection must contain 512 unique available-pool indices",
        )
        key = (spec["world"], spec["support"])
        selections.setdefault(key, indices)
        require(selections[key] == indices, "Support selection differs between initializations")
        require(
            spec["nodes"] == sorted(set(spec["nodes"]))
            and len(spec["nodes"]) == 12
            and all(n % 8 == 0 for n in spec["nodes"])
            and spec["nodes"][0] == 0
            and spec["nodes"][-1] == 128000,
            "Invalid 12-node learning trajectory",
        )
        require(
            spec["repeat_nodes"] == [32000, 128000] and spec["test_repeats"] == [1, 2, 3, 4],
            "Fixed repeat scan changed",
        )
        require(
            set(spec["repeat_nodes"]) <= set(spec["checkpoint_nodes"]) <= set(spec["nodes"]),
            "Invalid saved checkpoint nodes",
        )


def role_exposures(rows, counts):
    """Count actual first/second atomic fact use, weighting every sampled chain."""
    require(len(rows) == len(counts), "Exposure vector length differs from support")
    first, second = Counter(), Counter()
    for (h, r1, b, r2, t), count in zip(rows, counts, strict=True):
        first[int(h), int(r1), int(b)] += int(count)
        second[int(b), int(r2), int(t)] += int(count)
    return first, second


def graph_components(rows):
    """Independent BFS on role vertices; labels never use test scores or predictions."""
    adjacency = defaultdict(set)
    for h, r1, b, r2, _t in rows:
        left, right = ("first", int(h), int(r1)), ("second", int(b), int(r2))
        adjacency[left].add(right)
        adjacency[right].add(left)
    components = {}
    for root in sorted(adjacency):
        if root in components:
            continue
        label = root
        queue = deque([root])
        components[root] = label
        while queue:
            for neighbour in adjacency[queue.popleft()]:
                if neighbour not in components:
                    components[neighbour] = label
                    queue.append(neighbour)
    return components


def graph_summary(world):
    all_roles = graph_components(world["available_composite"])
    trained_roles = graph_components(world["train_composite"])
    isolated = len(set(all_roles) - set(trained_roles))
    return {
        "roles": len(all_roles),
        "components": len(set(trained_roles.values())) + isolated,
        "isolated_roles": isolated,
    }


def structural_subsets(connected_rows, split_rows, test_rows):
    """Partition the entire fixed test pool, retaining missing-role/different cases."""
    connected, split = graph_components(connected_rows), graph_components(split_rows)

    def same(components, row):
        h, r1, b, r2, _t = row
        left, right = ("first", int(h), int(r1)), ("second", int(b), int(r2))
        return left in components and right in components and components[left] == components[right]

    a = np.array([same(connected, row) for row in test_rows], dtype=bool)
    b = np.array([same(split, row) for row in test_rows], dtype=bool)
    require(not np.any(b & ~a), "Split connects test roles outside the connected support partition")
    masks = dict(zip(SUBSETS, (a & b, a & ~b, ~a & ~b), strict=True))
    require(np.all(sum(mask.astype(int) for mask in masks.values()) == 1), "Incomplete partition")
    return masks


def scores_from_predictions(predictions, task, rows, mask=None):
    """Score the complete saved generation, deriving counts from predictions, not averages."""
    generated = predictions[f"{task}_generated"]
    correct = predictions[f"{task}_correct"]
    nll = predictions[f"{task}_answer_nll"]
    require(generated.shape == (len(rows), 3), f"Incomplete generation: {task}")
    require(np.issubdtype(generated.dtype, np.integer), f"Invalid generated tokens: {task}")
    require(correct.shape == nll.shape == (len(rows),), f"Incomplete prediction vector: {task}")
    require(correct.dtype == bool, f"Invalid correctness vector: {task}")
    require(np.isfinite(nll).all(), f"Nonfinite prediction NLL: {task}")
    answer = generated[:, 0] == rows[:, -1]
    full = answer & (generated[:, 1] == 5) & (generated[:, 2] == 1)
    np.testing.assert_array_equal(correct, full, err_msg=f"Correctness differs from tokens: {task}")
    mask = np.ones(len(rows), dtype=bool) if mask is None else np.asarray(mask)
    require(mask.dtype == bool and mask.shape == (len(rows),), f"Invalid subset mask: {task}")
    n = int(mask.sum())
    result = {
        "n": n,
        "accuracy": float(full[mask].mean()) if n else None,
        "answer_accuracy": float(answer[mask].mean()) if n else None,
        "answer_nll": float(nll[mask].mean()) if n else None,
    }
    if task in ("familiar_test", "strict_test"):
        coverage, calls = (predictions[f"{task}_{k}"] for k in ("coverage", "two_calls"))
        require(
            coverage.shape == calls.shape == (len(rows),) and coverage.dtype == calls.dtype == bool,
            f"Incomplete conditional predictions: {task}",
        )
        covered = coverage & mask
        result.update(
            atomic_correct_coverage=float(coverage[mask].mean()) if n else None,
            conditional_accuracy=float(full[covered].mean()) if covered.any() else None,
            autonomous_two_calls=float(calls[mask].mean()) if n else None,
        )
    return result


def verify_prediction_metrics(predictions, world, metrics):
    require(set(metrics) == set(TASKS), "Endpoint task set changed")
    for task in TASKS:
        actual = scores_from_predictions(predictions, task, world[task])
        require(set(actual) == set(metrics[task]), f"Metric fields changed: {task}")
        for key, value in actual.items():
            saved = metrics[task][key]
            if key == "answer_nll":
                require(math.isclose(value, saved, rel_tol=1e-5, abs_tol=1e-5), f"NLL: {task}")
            else:
                require(value == saved, f"Metric differs from complete predictions: {task}.{key}")


def condition_means(rows):
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        groups[row["support"]][row["world"]].append(row)
    output = []
    for support in SUPPORTS:
        if support not in groups:
            continue
        worlds = []
        for world, seeds in sorted(groups[support].items()):
            require(
                len(seeds) == len({r["initialization"] for r in seeds}), "Duplicate initialization"
            )
            keys = [k for k in seeds[0] if k.split(".")[0] in TASKS]
            worlds.append(
                {
                    "world": world,
                    "initializations": [r["initialization"] for r in seeds],
                    "n_initializations": len(seeds),
                    "means": {k: paired_mean([r[k] for r in seeds]) for k in keys},
                    "denominators": {
                        task: [r[f"{task}.n"] for r in seeds]
                        for task in TASKS
                        if f"{task}.n" in seeds[0]
                    },
                }
            )
        output.append(
            {
                "support": support,
                "n_worlds": len(worlds),
                "worlds": worlds,
                "means": {k: paired_mean([w["means"][k] for w in worlds]) for k in keys},
            }
        )
    return output


def paired_contrasts(rows, *, descriptive=False):
    lookup = {}
    for row in rows:
        key = (row["world"], row["initialization"], row["support"])
        require(key not in lookup, f"Duplicate endpoint: {key}")
        lookup[key] = row
    pairs = sorted({(r["world"], r["initialization"]) for r in rows})
    fields = [
        k for k in rows[0] if k.split(".")[0] in TASKS and not k.endswith((".n", "_count", "_n"))
    ]
    result = []
    for field in fields:
        task, metric = field.split(".", 1)
        priority = (
            "descriptive secondary"
            if descriptive
            else "primary"
            if (task, metric) == ("familiar_test", "accuracy")
            else "secondary"
        )
        seeds = []
        for world, initialization in pairs:
            terms = []
            for support, coefficient in (("connected", 1), ("split", -1)):
                key = (world, initialization, support)
                require(key in lookup, f"Missing paired endpoint: {key}")
                row = lookup[key]
                terms.append(
                    {
                        "name": row["name"],
                        "support": support,
                        "coefficient": coefficient,
                        "value": row[field],
                        "n": row[f"{task}.n"],
                        "conditional_n": row.get(f"{task}.conditional_n"),
                    }
                )
            effect = (
                terms[0]["value"] - terms[1]["value"]
                if all(t["value"] is not None for t in terms)
                else None
            )
            seeds.append(
                {
                    "contrast": "connected-split",
                    "priority": priority,
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
        worlds = []
        for world in sorted({r["world"] for r in seeds}):
            selected = [r for r in seeds if r["world"] == world]
            effect = paired_mean([r["effect"] for r in selected])
            worlds.append(
                {
                    **{k: v for k, v in selected[0].items() if k not in ("terms",)},
                    "level": "world",
                    "initialization": None,
                    "n_initializations": len(selected),
                    "effect": effect,
                    "effect_pp": 100 * effect
                    if effect is not None and metric != "answer_nll"
                    else None,
                    "terms": [
                        {"initialization": r["initialization"], "terms": r["terms"]}
                        for r in selected
                    ],
                }
            )
        effect = paired_mean([r["effect"] for r in worlds])
        result.extend(seeds)
        result.extend(worlds)
        result.append(
            {
                **{k: v for k, v in worlds[0].items() if k not in ("terms",)},
                "level": "all_worlds",
                "world": None,
                "n_worlds": len(worlds),
                "n_initializations": len(seeds),
                "effect": effect,
                "effect_pp": 100 * effect
                if effect is not None and metric != "answer_nll"
                else None,
                "terms": [{"world": r["world"], "terms": r["terms"]} for r in worlds],
            }
        )
    return result


def load_and_verify(config, results, artifacts):
    from llm_memory_editability.latent_support import build_world, run_name

    validate_matrix(config["specs"])
    graph_path = Path(config["support_graphs"]["path"])
    require(
        file_hash(graph_path) == config["support_graphs"]["sha256"], "Frozen support graph changed"
    )
    graphs = {g["world"]: g for g in json.loads(graph_path.read_text())["worlds"]}
    graph_fields = {
        "connected": "forest_matched512_available_composite_indices",
        "split": "degree_matched_disconnected512_available_composite_indices",
    }
    require(set(graphs) == set(WORLDS), "Frozen support graph worlds changed")
    for spec in config["specs"]:
        require(
            spec["composition_indices"] == graphs[spec["world"]][graph_fields[spec["support"]]],
            "Selection differs from the frozen training-only graph selection",
        )
    require(config["source"], "Missing frozen training sources")
    for path, digest in config["source"].items():
        require(file_hash(path) == digest, f"Training source changed: {path}")
        snapshot = artifacts / "source" / path
        require(
            snapshot.is_file() and file_hash(snapshot) == digest, f"Frozen source missing: {path}"
        )
    rows, histories, repeats, audits, input_hashes = [], {}, {}, {}, {}
    worlds, exposures, predictions, initial_hashes = {}, {}, {}, {}
    for spec in config["specs"]:
        name = run_name(spec)
        folder = results / name
        result = json.loads((folder / "complete.json").read_text())
        checked = json.loads((folder / "audit.json").read_text())
        require(
            result["spec"] == spec and result["source"] == config["source"], f"Run contract: {name}"
        )
        require(
            checked["passed"] and checked["endpoint_tokens_exact"], f"Failed endpoint audit: {name}"
        )
        require(checked["repeat_checks"] == 8, f"Incomplete repeat audit: {name}")
        require(
            all(
                checked.get(k) is True
                for k in (
                    "frozen_data_exact",
                    "archived_world_exact",
                    "composition_complete_epochs",
                    "role_exposures_exact",
                )
            ),
            f"Incomplete data/exposure audit: {name}",
        )
        require(
            file_hash(folder / "latest.pt") == result["checkpoint_sha256"],
            f"Checkpoint changed: {name}",
        )
        history = json.loads((folder / "learning.json").read_text())
        require(
            [node["step"] for node in history] == spec["nodes"]
            and history[-1] == result["endpoint"],
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
            for metrics in values.values():
                metric_counts(metrics)
        files = [
            "complete.json",
            "audit.json",
            "learning.json",
            "repeat-metrics.json",
            "exposures.npz",
            "world.npz",
            "latest.pt",
        ]
        for node in history:
            metric_counts(node["metrics"])
            files.append(f"predictions-{node['step']:06d}.npz")
        files.extend(f"model-{n:06d}.pt" for n in spec["checkpoint_nodes"])
        files.extend(
            f"repeat-{n:06d}-r{r:02d}.npz"
            for n in spec["repeat_nodes"]
            for r in spec["test_repeats"]
        )
        for filename in files:
            require((folder / filename).is_file(), f"Missing saved artifact: {name}:{filename}")
        key = (spec["world"], spec["support"])
        if key not in worlds:
            worlds[key] = build_world(spec)
        world = worlds[key]
        with np.load(folder / "world.npz", allow_pickle=False) as saved:
            require(set(saved.files) == set(world), f"World keys changed: {name}")
            for task, value in world.items():
                np.testing.assert_array_equal(
                    saved[task], value, err_msg=f"World data changed: {name}:{task}"
                )
        digest = data_digest(world)
        require(
            digest
            == result["data_sha256"]
            == spec["frozen_data_sha256"]
            == config["data"][f"{spec['world']}:{spec['support']}"]["sha256"],
            f"World hash: {name}",
        )
        with np.load(folder / "exposures.npz", allow_pickle=False) as saved:
            require(
                set(saved.files) == {f"stratum{i}" for i in range(3)}, f"Exposure strata: {name}"
            )
            counts = []
            for index, task in enumerate(("common_atomic", "train_composite", "anchor_atomic")):
                count = saved[f"stratum{index}"]
                require(
                    count.shape == (len(world[task]),) and np.issubdtype(count.dtype, np.integer),
                    f"Exposure shape/type: {name}:{task}",
                )
                require(
                    int(count.min()) >= 0
                    and int(count.sum()) == 8192000
                    and int(np.ptp(count)) <= 1,
                    f"Exposure budget: {name}:{task}",
                )
                if index == 1:
                    require(
                        np.all(count == 16000),
                        f"Composition exposure must be 16000 complete epochs: {name}",
                    )
                counts.append(count.copy())
            exposures[name] = counts
        first, second = role_exposures(world["train_composite"], counts[1])
        actual_exposures = {
            "first_roles": {f"{h}:{r}": count for (h, r, _b), count in first.items()},
            "second_roles": {f"{b}:{r}": count for (b, r, _t), count in second.items()},
            "columns": [{} for _ in range(5)],
        }
        for column in range(5):
            marginal = Counter()
            for chain, count in zip(world["train_composite"], counts[1], strict=True):
                marginal[str(int(chain[column]))] += int(count)
            actual_exposures["columns"][column] = dict(marginal)
        require(
            actual_exposures == result["support_exposures"],
            f"Saved actual exposure summary: {name}",
        )
        components = graph_summary(world)
        require(components == result["support_components"], f"Saved graph summary: {name}")
        familiar = world["familiar_test"]
        coverage = {
            "first_atoms_used": len(first),
            "second_atoms_used": len(second),
            "familiar_both_composition_roles": float(
                np.mean(
                    [
                        (int(h), int(r1), int(b)) in first and (int(b), int(r2), int(t)) in second
                        for h, r1, b, r2, t in familiar
                    ]
                )
            ),
        }
        require(coverage == result["role_coverage"], f"Saved role coverage: {name}")
        pair = (spec["world"], spec["initialization"])
        initial_hashes.setdefault(pair, result["initial_model_sha256"])
        require(
            initial_hashes[pair] == result["initial_model_sha256"],
            f"Unpaired initialization: {name}",
        )
        for node in history:
            with np.load(
                folder / f"predictions-{node['step']:06d}.npz", allow_pickle=False
            ) as saved:
                pred = {k: saved[k].copy() for k in saved.files}
            verify_prediction_metrics(pred, world, node["metrics"])
            if node["step"] == spec["steps"]:
                predictions[name] = pred
        for node in spec["repeat_nodes"]:
            for repeat_count in spec["test_repeats"]:
                with np.load(
                    folder / f"repeat-{node:06d}-r{repeat_count:02d}.npz", allow_pickle=False
                ) as saved:
                    pred = {k: saved[k].copy() for k in saved.files}
                verify_prediction_metrics(pred, world, repeat[str(node)][str(repeat_count)])
            require(
                repeat[str(node)]["2"] == next(n["metrics"] for n in history if n["step"] == node),
                f"Fixed test R2 differs from the training-budget score: {name}:{node}",
            )
        row = {
            k: spec[k]
            for k in (
                "world",
                "initialization",
                "support",
                "width",
                "layers",
                "repeats",
                "composition_count",
                "steps",
            )
        }
        row.update(
            name=name,
            data_sha256=digest,
            initial_model_sha256=result["initial_model_sha256"],
            parameters=result["parameters"],
            independent_facts=result["independent_facts"],
            composition_examples=result["composition_examples"],
            training_seconds=result["training_seconds"],
            process_seconds=result["process_seconds"],
            supervised_tokens=result["supervised_tokens"],
            padded_input_tokens=result["padded_input_tokens"],
            training_flops=result["endpoint"]["estimated_training_flops"],
            composition_epochs=result["endpoint"]["composition_epochs"],
        )
        require(
            row["composition_examples"] == 512 and row["composition_epochs"] == 16000,
            f"Support count/epochs: {name}",
        )
        require(
            row["parameters"] == 416256
            and row["independent_facts"]
            == len(world["common_atomic"]) + len(world["anchor_atomic"])
            and row["supervised_tokens"] == spec["steps"] * spec["batch_size"] * 23 // 3
            and row["padded_input_tokens"] == spec["steps"] * spec["batch_size"] * 9,
            f"Fixed parameter/fact/token budget differs: {name}",
        )
        row.update(metric_counts(result["endpoint"]["metrics"]))
        row.update({f"support_components.{k}": v for k, v in components.items()})
        row.update({f"role_coverage.{k}": v for k, v in result["role_coverage"].items()})
        rows.append(row)
        histories[name], repeats[name], audits[name] = history, repeat, checked
        input_hashes[name] = {filename: file_hash(folder / filename) for filename in files}
    pair_audits = []
    for world_id in WORLDS:
        a, b = (worlds[world_id, support] for support in SUPPORTS)
        require(set(a) == set(b), "Different support world keys")
        for key in a:
            if key != "train_composite":
                np.testing.assert_array_equal(
                    a[key], b[key], err_msg=f"Nonexperimental data: {world_id}:{key}"
                )
        full_partition = graph_components(a["available_composite"])
        connected_partition = graph_components(a["train_composite"])
        require(
            set(full_partition) == set(connected_partition),
            f"Connected arm has uncovered roles: {world_id}",
        )
        require(
            len({(full_partition[v], connected_partition[v]) for v in full_partition})
            == len(set(full_partition.values()))
            == len(set(connected_partition.values())),
            f"Connected arm does not preserve the full available-pool partition: {world_id}",
        )
        for column in range(5):
            require(
                Counter(map(int, a["train_composite"][:, column]))
                == Counter(map(int, b["train_composite"][:, column])),
                f"Unmatched chain-token marginal: {world_id}:{column}",
            )
        for initialization in INITIALIZATIONS:
            paired = [
                next(
                    row
                    for row in rows
                    if (row["world"], row["initialization"], row["support"])
                    == (world_id, initialization, support)
                )
                for support in SUPPORTS
            ]
            ac, bc = (exposures[row["name"]] for row in paired)
            for index in (0, 2):
                np.testing.assert_array_equal(
                    ac[index],
                    bc[index],
                    err_msg=f"Unmatched atomic/anchor exposure: {world_id}:{initialization}",
                )
            require(
                role_exposures(a["train_composite"], ac[1])
                == role_exposures(b["train_composite"], bc[1]),
                f"Unmatched actual first/second role exposure: {world_id}:{initialization}",
            )
            pair_audits.append(
                {
                    "world": world_id,
                    "initialization": initialization,
                    "composition_exposures_per_arm": int(ac[1].sum()),
                    "composition_epochs_per_arm": 16000,
                    "weighted_first_fact_exposures_exact": True,
                    "weighted_second_fact_exposures_exact": True,
                    "atomic_and_anchor_exposures_exact": True,
                }
            )
    structural = []
    for world_id in WORLDS:
        a, b = (worlds[world_id, support] for support in SUPPORTS)
        for task in ("familiar_test", "strict_test"):
            masks = structural_subsets(a["train_composite"], b["train_composite"], a[task])
            for subset, mask in masks.items():
                for row in rows:
                    if row["world"] != world_id:
                        continue
                    structural.append(
                        {
                            "name": row["name"],
                            "world": world_id,
                            "initialization": row["initialization"],
                            "support": row["support"],
                            "task": task,
                            "subset": subset,
                            "fixed_pool_n": len(a[task]),
                            "subset_coverage": float(mask.mean()),
                            **metric_counts(
                                {
                                    task: scores_from_predictions(
                                        predictions[row["name"]], task, a[task], mask
                                    )
                                },
                                allow_empty=True,
                            ),
                        }
                    )
    return rows, histories, repeats, audits, input_hashes, pair_audits, structural


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


def structural_report(rows):
    output = []
    for task, subset in itertools.product(("familiar_test", "strict_test"), SUBSETS):
        selected = [r for r in rows if r["task"] == task and r["subset"] == subset]
        output.append(
            {
                "task": task,
                "subset": subset,
                "runs": selected,
                "conditions": condition_means(selected),
                "contrasts": paired_contrasts(selected, descriptive=True),
            }
        )
    return output


def repeat_report(rows, repeat_metrics):
    """Publish every frozen repeat budget; never select the best test recurrence."""
    output = []
    for node, repeat in itertools.product((32000, 128000), (1, 2, 3, 4)):
        selected = [
            {**row, **metric_counts(repeat_metrics[row["name"]][str(node)][str(repeat)])}
            for row in rows
        ]
        output.append(
            {
                "step": node,
                "test_repeats": repeat,
                "conditions": condition_means(selected),
                "contrasts": paired_contrasts(selected, descriptive=True),
            }
        )
    return output


def plots(rows, histories, artifacts):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"connected": "#4c78a8", "split": "#f58518"}
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharey="row")
    for column, world in enumerate(WORLDS):
        for line, task in enumerate(("familiar_test", "strict_test")):
            ax = axes[line, column]
            for index, support in enumerate(SUPPORTS):
                selected = [r for r in rows if r["world"] == world and r["support"] == support]
                values = [100 * r[f"{task}.accuracy"] for r in selected]
                ax.scatter([index] * len(values), values, color=colors[support], alpha=0.55, s=20)
                ax.scatter([index], [mean(values)], color=colors[support], marker="_", s=250)
            paired = paired_contrasts([r for r in rows if r["world"] == world])
            effect = next(
                r["effect_pp"]
                for r in paired
                if r["task"] == task and r["metric"] == "accuracy" and r["level"] == "world"
            )
            ax.set_title(f"World {world}; connected-split {effect:+.2f} pp")
            ax.set_xticks([0, 1], SUPPORTS)
            ax.set_xlim(-0.4, 1.4)
            ax.set_ylabel(f"{'Familiar' if line == 0 else 'Strict'} full accuracy (%)")
            ax.set_ylim(0, 105 if line == 0 else max(5, ax.get_ylim()[1]))
            ax.grid(alpha=0.2)
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(artifacts / f"comparison.{extension}", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharey=True)
    for column, world in enumerate(WORLDS):
        for line, task in enumerate(("train_composite", "familiar_test")):
            ax = axes[line, column]
            for support in SUPPORTS:
                selected = [r for r in rows if r["world"] == world and r["support"] == support]
                nodes = [histories[r["name"]] for r in selected]
                steps = [n["step"] for n in nodes[0]]
                values = [
                    100 * mean(h[i]["metrics"][task]["accuracy"] for h in nodes)
                    for i in range(len(steps))
                ]
                ax.plot(steps, values, marker=".", color=colors[support], label=support)
            ax.set_xscale("symlog", linthresh=256)
            ax.set_ylim(0, 105)
            ax.set_title(f"World {world}; {'train chains' if line == 0 else 'fixed familiar test'}")
            ax.set_xlabel("Training updates")
            ax.set_ylabel("Full generated accuracy (%)")
            ax.grid(alpha=0.2)
    axes[0, 0].legend()
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(artifacts / f"learning.{extension}", dpi=180)
    plt.close(fig)


def report(config_path, results, artifacts):
    config = json.loads(config_path.read_text())
    require(file_hash(artifacts / "design.md") == config["design_sha256"], "Frozen design differs")
    frozen = artifacts / "frozen-config.json"
    require(frozen.is_file() and json.loads(frozen.read_text()) == config, "Frozen config differs")
    rows, histories, repeats, audits, inputs, pair_audits, structural = load_and_verify(
        config, results, artifacts
    )
    require(
        len(audits) == 12
        and sum(map(len, histories.values())) == 144
        and sum(a["repeat_checks"] for a in audits.values()) == 96,
        "Incomplete endpoint/node/repeat audit matrix",
    )
    execution_path = artifacts / "execution.json"
    execution = json.loads(execution_path.read_text()) if execution_path.exists() else None
    contrasts = paired_contrasts(rows)
    primary = next(
        r for r in contrasts if r["priority"] == "primary" and r["level"] == "all_worlds"
    )
    reporting_hash = file_hash(__file__)
    summary = {
        "created_utc": utc(),
        "phase": config.get("phase"),
        "expected_runs": 12,
        "audited_runs": len(audits),
        "learning_nodes": sum(map(len, histories.values())),
        "repeat_checks": sum(a["repeat_checks"] for a in audits.values()),
        "unit": "3 existing independent worlds; "
        "2 paired initializations averaged within each world",
        "aggregation": "Equal initialization weights within world, then equal world weights; "
        "no query pooling",
        "primary": "connected-split familiar_test.accuracy at trained/test R2, width128 "
        "and 128000 updates",
        "primary_effect_pp": primary["effect_pp"],
        "boundaries": "A controlled structural training treatment in existing worlds. "
        "Matched chain count, role degrees, token marginals, actual role exposures, "
        "initialization and updates do not establish a unique LM mechanism or turn "
        "incidence rank into a GPT capacity bound. Treatment also changes higher-order "
        "pairings, batch order and optimization trajectories. Graph strata are descriptive subsets "
        "of a fixed full pool, not independent replications or replacement primary scores.",
        "runs": rows,
        "conditions": condition_means(rows),
        "contrasts": contrasts,
        "histories": histories,
        "repeat_metrics": repeats,
        "repeat_secondary": repeat_report(rows, repeats),
        "exposure_pair_audits": pair_audits,
        "structural_secondary": structural_report(structural),
        "structural_secondary_scope": "Defined before this batch's model training; "
        "descriptive analysis in previously observed worlds. Masks use training-graph "
        "BFS and fixed query roles without consulting correctness. All three strata "
        "partition the full fixed pool; report undefined empty strata without dropping worlds.",
        "budget": {
            "supervised_tokens": sum(r["supervised_tokens"] for r in rows),
            "padded_input_tokens": sum(r["padded_input_tokens"] for r in rows),
            "estimated_training_flops": sum(r["training_flops"] for r in rows),
            "summed_training_seconds": sum(r["training_seconds"] for r in rows),
            "summed_process_seconds": sum(r["process_seconds"] for r in rows),
            "execution_wall_seconds": execution.get("seconds") if execution else None,
        },
        "configuration_sha256": file_hash(config_path),
        "training_source": config["source"],
        "reporting_source_sha256": reporting_hash,
        "execution": execution,
    }
    artifacts.mkdir(parents=True, exist_ok=True)
    write_csv(artifacts / "endpoints.csv", rows)
    write_csv(artifacts / "contrasts.csv", contrasts)
    write_csv(artifacts / "structural-subsets.csv", structural)
    plots(rows, histories, artifacts)
    write_json(artifacts / "summary.json", summary)
    outputs = (
        "endpoints.csv",
        "contrasts.csv",
        "structural-subsets.csv",
        "comparison.png",
        "comparison.pdf",
        "learning.png",
        "learning.pdf",
        "summary.json",
    )
    write_json(
        artifacts / "completion-manifest.json",
        {
            "created_utc": utc(),
            "complete": True,
            "expected_runs": 12,
            "audited_runs": 12,
            "worlds": list(WORLDS),
            "initializations_per_world": list(INITIALIZATIONS),
            "learning_nodes": 144,
            "repeat_checks": 96,
            "matrix_exact": True,
            "source_exact": True,
            "endpoint_checkpoint_hashes_exact": True,
            "saved_worlds_exact": True,
            "support_changes_only_train_composite": True,
            "weighted_first_and_second_fact_exposures_exact": True,
            "common_and_anchor_exposures_exact": True,
            "paired_initial_states_exact": True,
            "full_predictions_reproduce_endpoint_metrics": True,
            "learning_predictions_reproduce_metrics": True,
            "repeat_predictions_reproduce_metrics": True,
            "structural_subsets_partition_fixed_test_pools": True,
            "aggregation": summary["aggregation"],
            "budget": summary["budget"],
            "configuration_sha256": summary["configuration_sha256"],
            "training_source": config["source"],
            "reporting_source_sha256": reporting_hash,
            "inputs": inputs,
            "outputs": {name: file_hash(artifacts / name) for name in outputs},
        },
    )
    print(
        json.dumps(
            {
                "audited_runs": 12,
                "primary_connected_minus_split_pp": primary["effect_pp"],
                "artifacts": str(artifacts),
            }
        )
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/latent-support-v1.json"))
    parser.add_argument("--results", type=Path, default=Path("results/latent-support-v1"))
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    args = parser.parse_args()
    report(args.config, args.results, args.artifacts)


if __name__ == "__main__":
    main()
