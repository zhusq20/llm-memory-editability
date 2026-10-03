"""Ordinary end-to-end GPT depth and optimization-time development experiment.

The frozen random graph is shared across two-, three- and four-hop training.
Only terminal entities enter composition text; intermediate truth is used for
evaluation and audit. Existing experimental implementations remain unchanged.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .bios_model import ModelConfig
from .grok_depth import EpochStream, make_optimizer, utc, write_json
from .grok_loop_data import build_world as single_length_world
from .grok_loop_model import LoopGPT, flops
from .latent_scaling import model_digest
from .storage_composition import FullTokenStep, file_hash
from .storage_frontier import learning_rate

BOS, EOS, PAD, ENTITY_OFFSET = 2, 1, 0, 3
HOPS = (2, 3, 4)
TRAIN_SPLITS = ("atomic", "train_2", "train_3", "train_4")
EVALUATION_SPLITS = ("atomic",) + tuple(
    f"{split}_{hop}" for hop in HOPS for split in ("train", "familiar", "strict")
)
DATA_DEFAULTS = {
    "entities": 64,
    "relations": 4,
    "degree": 4,
    "phi": 4.0,
    "id_fraction": 0.75,
    "id_test_fraction": 0.2,
    "evaluation_size": 512,
}
SOURCE_FILES = [
    "src/llm_memory_editability/bios_model.py",
    "src/llm_memory_editability/grok_depth.py",
    "src/llm_memory_editability/grok_depth_data.py",
    "src/llm_memory_editability/grok_loop_data.py",
    "src/llm_memory_editability/grok_multihop_data.py",
    "src/llm_memory_editability/grok_loop_model.py",
    "src/llm_memory_editability/latent_scaling.py",
    "src/llm_memory_editability/storage_composition.py",
    "src/llm_memory_editability/storage_frontier.py",
    "src/llm_memory_editability/text_pretrain.py",
    "src/llm_memory_editability/depth_step.py",
    "scripts/run_depth_step.py",
    "tests/test_depth_step.py",
]


def data_digest(world):
    digest = hashlib.sha256()
    for name in EVALUATION_SPLITS:
        rows = world[name]
        digest.update(name.encode())
        digest.update(np.asarray(rows.shape, dtype="<i8").tobytes())
        digest.update(rows.astype("<i8", copy=False).tobytes())
    meta = world["metadata"]
    digest.update(
        json.dumps(
            {key: meta[key] for key in ("world_seed", "entities", "relations", "id_mask")},
            sort_keys=True,
        ).encode()
    )
    return digest.hexdigest()


def truth_path_details(world, rows):
    """Return independent truth trajectories and constituent atomic row indices."""
    rows = np.asarray(rows)
    atoms = world["atomic"]
    lookup = {(int(h), int(r)): (index, int(t)) for index, (h, r, t) in enumerate(atoms)}
    if len(lookup) != len(atoms):
        raise ValueError("Duplicate atomic keys")
    if rows.ndim != 2 or rows.shape[1] < 3:
        raise ValueError("Invalid compact truth rows")
    nodes = np.empty((len(rows), rows.shape[1] - 1), dtype=np.int64)
    indices = np.empty((len(rows), rows.shape[1] - 2), dtype=np.int64)
    nodes[:, 0] = rows[:, 0]
    for i, row in enumerate(rows):
        current = int(row[0])
        for j, relation in enumerate(row[1:-1]):
            edge = lookup.get((current, int(relation)))
            if edge is None:
                raise ValueError("Composition references a missing atomic edge")
            index, current = edge
            indices[i, j], nodes[i, j + 1] = index, current
        if current != int(row[-1]):
            raise ValueError("Composition terminal entity disagrees with graph truth")
    return nodes, indices


def audit_world(world):
    meta = world["metadata"]
    entities, relations = meta["entities"], meta["relations"]
    relation_offset = ENTITY_OFFSET + entities
    id_mask = np.asarray(meta["id_mask"], dtype=bool)
    if id_mask.shape != (len(world["atomic"]),):
        raise ValueError("Invalid atomic ID mask")
    counts = {}
    for name in EVALUATION_SPLITS:
        rows = world[name]
        expected_columns = 3 if name == "atomic" else int(name.rsplit("_", 1)[1]) + 2
        if rows.dtype != np.int64 or rows.ndim != 2 or rows.shape[1] != expected_columns:
            raise ValueError("Invalid compact array shape/dtype: " + name)
        if np.any((rows[:, [0, -1]] < ENTITY_OFFSET) | (rows[:, [0, -1]] >= relation_offset)):
            raise ValueError("Invalid entity token: " + name)
        if np.any(
            (rows[:, 1:-1] < relation_offset) | (rows[:, 1:-1] >= relation_offset + relations)
        ):
            raise ValueError("Invalid relation token: " + name)
        if len(set(map(tuple, rows[:, :-1]))) != len(rows):
            raise ValueError("Duplicate complete query: " + name)
        _, edges = truth_path_details(world, rows)
        if name.startswith(("train_", "familiar_")) and not id_mask[edges].all():
            raise ValueError("Non-ID fact in familiar training/test: " + name)
        if name.startswith("strict_") and id_mask[edges].any():
            raise ValueError("ID fact in strict composition: " + name)
        counts[name] = len(rows)
    if len(world["atomic"]) != entities * meta["degree"]:
        raise ValueError("Atomic graph degree does not match the contract")
    for hop in HOPS:
        query_sets = [
            set(map(tuple, world[f"{split}_{hop}"][:, :-1]))
            for split in ("train", "familiar", "strict")
        ]
        if any(query_sets[i] & query_sets[j] for i in range(3) for j in range(i)):
            raise ValueError("Complete query leakage between length-specific splits")
    trained_edges = set()
    for hop in HOPS:
        trained_edges.update(truth_path_details(world, world[f"train_{hop}"])[1].ravel())
    for hop in HOPS:
        strict_edges = set(truth_path_details(world, world[f"strict_{hop}"])[1].ravel())
        if strict_edges & trained_edges:
            raise ValueError("Strict facts participated in another length's composition training")
    digest = data_digest(world)
    if "dataset_sha256" in meta and meta["dataset_sha256"] != digest:
        raise ValueError("Frozen dataset digest differs from actual arrays")
    return {
        "passed": True,
        "counts": counts,
        "dataset_sha256": digest,
        "truth_edges_independently_traversed": True,
        "complete_query_overlap": 0,
        "all_strict_facts_absent_from_composition_training": True,
        "cross_length_subpath_holdout": False,
    }


def build_world(spec):
    seed = spec.get("world_seed", spec.get("world"))
    if seed is None:
        raise ValueError("A fixed world_seed is required")
    settings = {key: spec.get(key, value) for key, value in DATA_DEFAULTS.items()}
    world, shared_atoms, shared_ids = {}, None, None
    source_metadata = {}
    for hop in HOPS:
        original = single_length_world({"world_seed": seed, "hops": hop, **settings})
        # The historical graph reserves only PAD/EOS and uses entity offset 2.
        # Shift every entity/relation token once to reserve the explicit BOS=2.
        atoms = original["atomic"] + 1
        id_set = set(map(tuple, original["id_atomic"]))
        id_mask = np.asarray([tuple(row) in id_set for row in original["atomic"]], dtype=bool)
        if shared_atoms is None:
            shared_atoms, shared_ids = atoms.copy(), id_mask.copy()
        elif not np.array_equal(shared_atoms, atoms) or not np.array_equal(shared_ids, id_mask):
            raise ValueError("Path lengths do not share identical graph and ID mask")
        world[f"train_{hop}"] = original["train_composite"] + 1
        world[f"familiar_{hop}"] = original["test_full_composite"] + 1
        world[f"strict_{hop}"] = original["ood_composite"] + 1
        source_metadata[str(hop)] = original["metadata"]
    world["atomic"] = shared_atoms
    world["metadata"] = {
        "world_seed": int(seed),
        **settings,
        "id_mask": shared_ids.tolist(),
        "source_worlds": source_metadata,
        "same_atomic_and_id_mask_all_lengths": True,
        "source_entity_offset": 2,
        "entity_offset": ENTITY_OFFSET,
        "source_token_shift": 1,
        "separator_token": ENTITY_OFFSET + settings["entities"] + settings["relations"],
        "vocab_size": ENTITY_OFFSET + settings["entities"] + settings["relations"] + 1,
        "evaluation_pools": "Complete test_full_composite and ood_composite",
        "composition_supervision": "Terminal entity only; no intermediate entity",
        "training_objective": "All nonpadding next tokens including EOS",
        "holdout": "Complete queries held out within each length; not subpaths across lengths",
    }
    report = audit_world(world)
    world["metadata"]["dataset_sha256"] = report["dataset_sha256"]
    world["metadata"]["independent_audit"] = report
    expected = spec.get("frozen_data_sha256")
    if expected is not None and expected != report["dataset_sha256"]:
        raise ValueError("Rebuilt data differ from the frozen specification")
    return world


def construct(spec, device):
    if spec.get("context", 9) < 9:
        raise ValueError("Context must support four-hop answer and EOS")
    config = ModelConfig(
        vocab_size=ENTITY_OFFSET + spec.get("entities", 64) + spec.get("relations", 4) + 1,
        width=spec["width"],
        layers=spec["layers"],
        heads=spec["heads"],
        context=spec.get("context", 9),
    )
    if config.layers > 6:
        raise ValueError("Canonical paired initialization is limited to six unique blocks")
    torch.manual_seed(spec["initialization"])
    canonical = LoopGPT(
        replace(config, layers=6),
        repeats=1,
        dropout=spec.get("dropout", 0.0),
        initialization="scaled_effective",
    )
    model = LoopGPT(
        config,
        repeats=spec["repeats"],
        dropout=spec.get("dropout", 0.0),
        initialization="scaled_effective",
    )
    initial = canonical.state_dict()
    state = {}
    factor = math.sqrt(6 / (config.layers * spec["repeats"]))
    for key in model.state_dict():
        value = initial[key].clone()
        if key.endswith(("attention.proj.weight", "mlp.down.weight")):
            value.mul_(factor)
        state[key] = value
    model.load_state_dict(state)
    return model.to(device)


def prompt_rows(rows, separator=71):
    rows = np.asarray(rows, dtype=np.int64)
    return np.c_[
        np.full(len(rows), BOS, dtype=np.int64),
        rows[:, :-1],
        np.full(len(rows), separator, dtype=np.int64),
    ]


def pack_rows(rows, sequence=8, separator=71):
    """All nonpadding next-token labels; compact rows contain no intermediate nodes."""
    rows = np.asarray(rows, dtype=np.int64)
    sentence = np.c_[
        prompt_rows(rows, separator), rows[:, -1], np.full(len(rows), EOS, dtype=np.int64)
    ]
    length = sentence.shape[1] - 1
    if length > sequence:
        raise ValueError("Packed sequence is too short")
    tokens = np.zeros((len(rows), sequence), dtype=np.int64)
    labels = np.full_like(tokens, -100)
    tokens[:, :length], labels[:, :length] = sentence[:, :-1], sentence[:, 1:]
    return tokens, labels


def run_name(spec):
    seed = spec.get("world_seed", spec.get("world"))
    return (
        f"w{seed}-i{spec['initialization']}-d{spec['width']}"
        f"-l{spec['layers']}-r{spec['repeats']}-s{spec['steps']}"
    )


def _mean(values):
    return float(np.asarray(values).mean()) if len(values) else None


@torch.no_grad()
def generate_rows(model, rows, device, separator=71, batch_size=512):
    """Ordinary full forward, then free tail/EOS generation with answer feedback."""
    rows = np.asarray(rows, dtype=np.int64)
    generated, nll = [], []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        tokens = torch.as_tensor(prompt_rows(batch, separator), device=device)
        first = model(tokens)[:, -1]
        target = torch.as_tensor(batch[:, -1], device=device)
        nll.append(F.cross_entropy(first, target, reduction="none").cpu().numpy())
        answer = first.argmax(-1)
        tokens = torch.cat([tokens, answer[:, None]], dim=1)
        stop = model(tokens)[:, -1].argmax(-1)
        generated.append(torch.stack([answer, stop], dim=1).cpu().numpy())
    pred = np.concatenate(generated) if generated else np.empty((0, 2), dtype=np.int64)
    nll = np.concatenate(nll) if nll else np.empty(0, dtype=np.float32)
    answer_correct = pred[:, 0] == rows[:, -1]
    correct = answer_correct & (pred[:, 1] == EOS)
    return {
        "n": len(rows),
        "accuracy": _mean(correct),
        "answer_accuracy": _mean(answer_correct),
        "answer_nll": _mean(nll),
    }, {"generated": pred, "correct": correct, "answer_nll": nll}


@torch.no_grad()
def evaluate(model, world, device):
    before = model.training
    model.eval()
    metrics, predictions = {}, {}
    separator = world["metadata"]["separator_token"]
    try:
        atomic_metrics, atomic_pred = generate_rows(model, world["atomic"], device, separator)
        metrics["atomic"] = atomic_metrics
        predictions.update({"atomic_" + key: value for key, value in atomic_pred.items()})
        for name in EVALUATION_SPLITS[1:]:
            rows = world[name]
            task_metrics, direct = generate_rows(model, rows, device, separator)
            nodes, edges = truth_path_details(world, rows)
            coverage = atomic_pred["correct"][edges].all(axis=1)
            current = rows[:, 0].copy()
            hops = rows.shape[1] - 2
            calls = np.empty((len(rows), hops, 2), dtype=np.int64)
            for hop in range(hops):
                query = np.c_[current, rows[:, hop + 1], nodes[:, hop + 1]]
                _, result = generate_rows(model, query, device, separator)
                calls[:, hop] = result["generated"]
                current = result["generated"][:, 0]
            formats = (calls[:, :, 1] == EOS).all(axis=1)
            autonomous_correct = formats & (current == rows[:, -1])
            path_correct = formats & (calls[:, :, 0] == nodes[:, 1:]).all(axis=1)
            task_metrics.update(
                {
                    "atomic_correct_coverage": _mean(coverage),
                    "conditional_accuracy": _mean(direct["correct"][coverage]),
                    "autonomous_two_calls": _mean(autonomous_correct),
                    "autonomous_path_accuracy": _mean(path_correct),
                    "hop_count": hops,
                }
            )
            metrics[name] = task_metrics
            direct.update(
                {
                    "coverage": coverage,
                    "autonomous_generated": calls,
                    "autonomous_correct": autonomous_correct,
                    "autonomous_path_correct": path_correct,
                }
            )
            predictions.update({name + "_" + key: value for key, value in direct.items()})
    finally:
        model.train(before)
    return metrics, predictions


def _validate_spec(spec):
    nodes = spec["nodes"]
    if nodes != sorted(set(nodes)) or nodes[0] != 0 or nodes[-1] != spec["steps"]:
        raise ValueError("Nodes must increase from initialization to the fixed endpoint")
    if not set(spec["checkpoint_nodes"]) <= set(nodes) or not {0, spec["steps"]} <= set(
        spec["checkpoint_nodes"]
    ):
        raise ValueError("Checkpoint nodes must include initialization and endpoint")
    if spec["batch_size"] != 128 or spec.get("dropout", 0.0) != 0.0:
        raise ValueError("The fixed design requires four 32-example streams and dropout zero")
    if spec.get("model_initialization", "scaled_effective") != "scaled_effective":
        raise ValueError("The batch uses scaled_effective residual initialization")


def _save_world(out, world):
    np.savez_compressed(out / "world.npz", **{key: world[key] for key in EVALUATION_SPLITS})
    write_json(out / "world-metadata.json", world["metadata"])


def _check_source(source):
    if not source or any(file_hash(path) != digest for path, digest in source.items()):
        raise RuntimeError("Actual source differs from the required frozen source hashes")


def _validate_output(out):
    """Permit scheduler control files while preserving every existing experiment artifact."""
    out = Path(out)
    if out.exists():
        for path in out.iterdir():
            allowed = path.is_file() and (
                path.name == "input-spec.json" or path.name.endswith("-process.log")
            )
            if not allowed:
                raise FileExistsError(
                    "Do not overwrite an existing experiment artifact: " + str(path)
                )


def train(spec, out, source, device="cuda:0"):
    _validate_spec(spec)
    _check_source(source)
    if "frozen_data_sha256" not in spec:
        raise ValueError("Training requires frozen_data_sha256")
    out = Path(out)
    _validate_output(out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.cuda.set_device(torch.device(device))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    started_process = time.perf_counter()
    world = build_world(spec)
    digest = data_digest(world)
    _save_world(out, world)
    write_json(out / "spec.json", {"spec": spec, "source": source, "data_sha256": digest})
    model = construct(spec, device)
    initial_digest = model_digest(model)
    lr = torch.tensor(spec["lr"], device=device)
    optimizer = make_optimizer(model, lr, spec["weight_decay"])
    strata = [world[name] for name in TRAIN_SPLITS]
    sizes = [len(rows) for rows in strata]
    if any(size == 0 for size in sizes):
        raise ValueError("Do not reroll a world with an empty sampled training stratum")
    packed = [pack_rows(rows, separator=world["metadata"]["separator_token"]) for rows in strata]
    table = tuple(
        torch.as_tensor(np.concatenate(parts), device=device) for parts in zip(*packed, strict=True)
    )
    offsets = np.cumsum([0, *sizes[:-1]])
    streams = [EpochStream(size, spec["stream_seed"] + i) for i, size in enumerate(sizes)]
    counts = [np.zeros(size, dtype=np.int64) for size in sizes]
    graph = FullTokenStep(model, optimizer, table, spec["batch_size"], clip=spec.get("clip", 1.0))
    flop_step = flops(model.config, spec["repeats"], spec["batch_size"], 8, output_positions=8)
    token_step = 32 * sum(int((labels[0] >= 0).sum()) for _, labels in packed)
    history, step, training_seconds = [], 0, 0.0

    def measure(loss=None):
        metrics, predictions = evaluate(model, world, device)
        row = {
            "step": step,
            "lr": learning_rate(spec, step) if step else 0.0,
            "metrics": metrics,
            "loss": loss,
            "training_seconds": training_seconds,
            "estimated_training_flops": step * flop_step,
            "supervised_tokens": step * token_step,
            "padded_input_tokens": step * spec["batch_size"] * 8,
        }
        history.append(row)
        write_json(out / "learning.json", history)
        np.savez_compressed(out / f"predictions-{step:06d}.npz", **predictions)
        np.savez_compressed(
            out / f"exposures-{step:06d}.npz",
            **{name: count for name, count in zip(TRAIN_SPLITS, counts, strict=True)},
        )
        state = {
            "spec": spec,
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "streams": [s.state_dict() for s in streams],
            "counts": counts,
            "cpu_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(device),
            "source": source,
            "data_sha256": digest,
        }
        torch.save(state, out / "latest.tmp.pt")
        (out / "latest.tmp.pt").replace(out / "latest.pt")
        if step in spec["checkpoint_nodes"]:
            torch.save(
                {"spec": spec, "step": step, "model": model.state_dict()},
                out / f"model-{step:06d}.pt",
            )
        write_json(out / "status.json", {"step": step, "budget": spec["steps"]})
        print(
            json.dumps(
                {
                    "run": run_name(spec),
                    "step": step,
                    "atomic": metrics["atomic"]["accuracy"],
                    "familiar": {str(hop): metrics[f"familiar_{hop}"]["accuracy"] for hop in HOPS},
                }
            ),
            flush=True,
        )
        return row

    measure()
    for end in spec["nodes"][1:]:
        torch.cuda.synchronize()
        started = time.perf_counter()
        while step < end:
            updates = min(128, end - step)
            indices = []
            for i, stream in enumerate(streams):
                draws = stream.take(updates * 32).reshape(updates, 32)
                counts[i] += np.bincount(draws.ravel(), minlength=sizes[i])
                indices.append(draws + offsets[i])
            indices = torch.as_tensor(np.concatenate(indices, axis=1), device=device)
            for j in range(updates):
                lr.fill_(learning_rate(spec, step + j + 1))
                loss = graph(indices[j])
            step += updates
        torch.cuda.synchronize()
        training_seconds += time.perf_counter() - started
        loss_value = float(loss)
        if not np.isfinite(loss_value):
            raise FloatingPointError("Nonfinite loss at step " + str(step))
        endpoint = measure(loss_value)
    np.savez_compressed(
        out / "exposures.npz",
        **{name: count for name, count in zip(TRAIN_SPLITS, counts, strict=True)},
    )
    result = {
        "spec": spec,
        "source": source,
        "data_sha256": digest,
        "initial_model_sha256": initial_digest,
        "checkpoint_sha256": file_hash(out / "latest.pt"),
        "finished_utc": utc(),
        "parameters": sum(p.numel() for p in model.parameters()),
        "strata_counts": dict(zip(TRAIN_SPLITS, sizes, strict=True)),
        "endpoint": endpoint,
        "supervised_tokens": step * token_step,
        "padded_input_tokens": step * spec["batch_size"] * 8,
        "estimated_training_flops": step * flop_step,
        "training_seconds": training_seconds,
        "process_seconds": time.perf_counter() - started_process,
        "environment": {
            "torch": torch.__version__,
            "numpy": np.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
            "tf32": True,
        },
    }
    write_json(out / "complete.json", result)
    return result


def compare_metrics(actual, expected):
    if set(actual) != set(expected):
        raise AssertionError("Metric task keys differ")
    for task, values in actual.items():
        if set(values) != set(expected[task]):
            raise AssertionError("Metric field keys differ")
        for key, value in values.items():
            if key == "answer_nll" and value is not None:
                np.testing.assert_allclose(value, expected[task][key], rtol=1e-5, atol=1e-5)
            elif value != expected[task][key]:
                raise AssertionError(f"Saved metric differs: {task}.{key}")


def compare_predictions(actual, path):
    maximum = 0.0
    with np.load(path) as saved:
        if set(actual) != set(saved.files):
            raise AssertionError("Prediction keys differ")
        for key, value in actual.items():
            if key.endswith("answer_nll"):
                np.testing.assert_allclose(value, saved[key], rtol=1e-5, atol=1e-5)
                if value.size:
                    maximum = max(maximum, float(np.abs(value - saved[key]).max()))
            else:
                np.testing.assert_array_equal(value, saved[key])
    return maximum


def audit(out, device="cuda:0"):
    out = Path(out)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.cuda.set_device(torch.device(device))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    result = json.loads((out / "complete.json").read_text())
    recorded = json.loads((out / "spec.json").read_text())
    source, spec = result["source"], result["spec"]
    _check_source(source)
    _validate_spec(spec)
    if recorded["spec"] != spec or recorded["source"] != source:
        raise AssertionError("Recorded specification/source differ")
    if file_hash(out / "latest.pt") != result["checkpoint_sha256"]:
        raise AssertionError("Latest checkpoint hash differs")
    world = build_world(spec)
    digest = data_digest(world)
    if digest != result["data_sha256"] or digest != recorded["data_sha256"]:
        raise AssertionError("Rebuilt data differ from records")
    if json.loads((out / "world-metadata.json").read_text()) != world["metadata"]:
        raise AssertionError("Archived world metadata differ")
    with np.load(out / "world.npz") as archived:
        if set(archived.files) != set(EVALUATION_SPLITS):
            raise AssertionError("Archived world keys differ")
        for name in EVALUATION_SPLITS:
            np.testing.assert_array_equal(archived[name], world[name])
    model = construct(spec, device)
    if model_digest(model) != result["initial_model_sha256"]:
        raise AssertionError("Paired initialization differs")
    history = json.loads((out / "learning.json").read_text())
    if [row["step"] for row in history] != spec["nodes"]:
        raise AssertionError("Recorded learning nodes differ")
    by_node = {row["step"]: row for row in history}
    if by_node[spec["steps"]] != result["endpoint"]:
        raise AssertionError("Endpoint differs from the learning curve")
    maximum = 0.0
    for node in spec["checkpoint_nodes"]:
        checkpoint = torch.load(
            out / f"model-{node:06d}.pt", map_location=device, weights_only=False
        )
        if checkpoint["spec"] != spec or checkpoint["step"] != node:
            raise AssertionError("Checkpoint specification/node differ")
        model.load_state_dict(checkpoint["model"])
        metrics, predictions = evaluate(model, world, device)
        compare_metrics(metrics, by_node[node]["metrics"])
        maximum = max(
            maximum, compare_predictions(predictions, out / f"predictions-{node:06d}.npz")
        )
    saved = torch.load(out / "latest.pt", map_location=device, weights_only=False)
    if (
        saved["spec"] != spec
        or saved["source"] != source
        or saved["step"] != spec["steps"]
        or saved["data_sha256"] != digest
    ):
        raise AssertionError("Latest state identity differs")
    sizes = [len(world[name]) for name in TRAIN_SPLITS]
    streams = [EpochStream(size, spec["stream_seed"] + i) for i, size in enumerate(sizes)]
    counts = [np.zeros(size, dtype=np.int64) for size in sizes]
    previous = 0
    for node in spec["nodes"]:
        for i, stream in enumerate(streams):
            draws = (
                stream.take((node - previous) * 32)
                if node > previous
                else np.empty(0, dtype=np.int64)
            )
            counts[i] += np.bincount(draws, minlength=sizes[i])
        with np.load(out / f"exposures-{node:06d}.npz") as actual:
            for name, count in zip(TRAIN_SPLITS, counts, strict=True):
                np.testing.assert_array_equal(actual[name], count)
        previous = node
    with np.load(out / "exposures.npz") as actual:
        for i, (name, count) in enumerate(zip(TRAIN_SPLITS, counts, strict=True)):
            np.testing.assert_array_equal(actual[name], count)
            np.testing.assert_array_equal(saved["counts"][i], count)
            if count.sum() != spec["steps"] * 32 or np.ptp(count) > 1:
                raise AssertionError("EpochStream actual exposures are unbalanced")
            stream = EpochStream(sizes[i], spec["stream_seed"] + i)
            stream.load_state_dict(saved["streams"][i])
            np.testing.assert_array_equal(stream.take(32), streams[i].take(32))
    model.load_state_dict(saved["model"])
    metrics, predictions = evaluate(model, world, device)
    compare_metrics(metrics, result["endpoint"]["metrics"])
    maximum = max(
        maximum, compare_predictions(predictions, out / f"predictions-{spec['steps']:06d}.npz")
    )
    checked = {
        "passed": True,
        "dataset_rebuilt_and_archived_exact": True,
        "initial_model_exact": True,
        "checkpoint_nodes": spec["checkpoint_nodes"],
        "endpoint_tokens_exact": True,
        "all_nodes_exposure_counters_exact": True,
        "stream_state_continuation_exact": True,
        "max_nll_difference": maximum,
        "nll_rtol": 1e-5,
        "nll_atol": 1e-5,
        "utc": utc(),
    }
    write_json(out / "audit.json", checked)
    return checked
