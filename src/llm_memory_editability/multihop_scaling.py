"""Width, composition experience and compute depth in complete causal GPT models.

The data, full-token objective and generation scoring extend ``depth_step``.
Support arms use nested training paths and identical held-out evaluation pools;
four independent streams keep atomics and each hop length at 32 draws per step.
"""

from __future__ import annotations

import copy
import hashlib
import json
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from . import depth_step
from .depth_step import (  # noqa: F401 -- public interface reused by checkpoint analysis
    BOS,
    DATA_DEFAULTS,
    ENTITY_OFFSET,
    EOS,
    EVALUATION_SPLITS,
    HOPS,
    PAD,
    TRAIN_SPLITS,
    compare_metrics,
    compare_predictions,
    construct,
    data_digest,
    evaluate,
    generate_rows,
    pack_rows,
    prompt_rows,
    truth_path_details,
)
from .grok_depth import EpochStream, make_optimizer, utc, write_json
from .grok_loop_model import flops
from .latent_scaling import model_digest
from .storage_composition import FullTokenStep, file_hash
from .storage_frontier import learning_rate

SUPPORT_LEVELS = (1.0, 4.0)
ARCHITECTURES = ((2, 1), (4, 1), (1, 2), (1, 4))
COMMON_SPLITS = ("atomic",) + tuple(
    f"{split}_{hop}" for hop in HOPS for split in ("familiar", "strict")
)
SOURCE_FILES = [
    *depth_step.SOURCE_FILES,
    "src/llm_memory_editability/multihop_scaling.py",
    "scripts/run_multihop_scaling.py",
    "scripts/execute_multihop_scaling.py",
    "tests/test_multihop_scaling.py",
]


@lru_cache(maxsize=16)
def _reference_world(seed, settings):
    return depth_step.build_world({"world_seed": seed, **dict(settings), "phi": 4.0})


def _reference(spec):
    seed = spec.get("world_seed", spec.get("world"))
    if seed is None:
        raise ValueError("A fixed world_seed is required")
    settings = tuple(
        (key, spec.get(key, value)) for key, value in DATA_DEFAULTS.items() if key != "phi"
    )
    return _reference_world(int(seed), settings)


def common_evaluation_digest(world):
    digest = hashlib.sha256()
    for name in COMMON_SPLITS:
        rows = world[name]
        digest.update(name.encode())
        digest.update(np.asarray(rows.shape, dtype="<i8").tobytes())
        digest.update(rows.astype("<i8", copy=False).tobytes())
    digest.update(json.dumps(world["metadata"]["id_mask"]).encode())
    return digest.hexdigest()


def build_world(spec):
    phi = float(spec.get("phi", 4.0))
    if phi not in SUPPORT_LEVELS:
        raise ValueError("Support phi must be 1 or 4")
    reference = _reference(spec)
    world = {name: reference[name].copy() for name in EVALUATION_SPLITS}
    metadata = copy.deepcopy(reference["metadata"])
    metadata.pop("dataset_sha256", None)
    metadata.pop("independent_audit", None)
    ids = sum(metadata["id_mask"])
    selections = {}
    for hop in HOPS:
        name = f"train_{hop}"
        count = min(round(phi * ids), len(reference[name]))
        world[name] = reference[name][:count].copy()
        selections[str(hop)] = {
            "requested": round(phi * ids),
            "actual": count,
            "effective_phi": count / ids if ids else None,
            "high_support_count": len(reference[name]),
            "capped": round(phi * ids) > len(reference[name]),
            "prefix_indices": list(range(count)),
        }
    # Preserve source lineage without duplicating descriptive source statistics.
    metadata["source_worlds"] = {
        hop: {
            key: values[key]
            for key in (
                "seed",
                "hops",
                "dataset_sha256",
                "counts",
                "phi_requested",
                "phi_actual",
                "train_composite_capped",
                "id_paths_total",
                "all_paths_total",
                "source_url",
                "source_commit",
                "source_file",
                "source_license",
            )
        }
        for hop, values in metadata["source_worlds"].items()
    }
    metadata.update(
        phi=phi,
        reference_phi=4.0,
        reference_dataset_sha256=reference["metadata"]["dataset_sha256"],
        composition_support=selections,
        support_selection="Prefix of each phi=4 training array; no resampling of facts or tests",
        exposure_contract="32 atomic + 32 each of 2/3/4-hop examples per step",
        training_objective="All nonpadding next tokens including EOS; no intermediate targets",
    )
    world["metadata"] = metadata
    metadata["common_evaluation_sha256"] = common_evaluation_digest(world)
    checked = audit_world(world)
    metadata["dataset_sha256"] = checked["dataset_sha256"]
    metadata["independent_audit"] = checked
    expected = spec.get("frozen_data_sha256")
    if expected is not None and expected != checked["dataset_sha256"]:
        raise ValueError("Rebuilt data differ from the frozen specification")
    expected_common = spec.get("common_evaluation_sha256")
    if expected_common is not None and expected_common != metadata["common_evaluation_sha256"]:
        raise ValueError("Common evaluation pools differ from the frozen specification")
    return world


def audit_world(world):
    checked = depth_step.audit_world(world)
    metadata = world["metadata"]
    phi = metadata["phi"]
    if phi not in SUPPORT_LEVELS or metadata.get("reference_phi") != 4.0:
        raise ValueError("Invalid nested support contract")
    reference = _reference(metadata)
    ids = sum(metadata["id_mask"])
    for name in EVALUATION_SPLITS:
        expected = reference[name]
        if name.startswith("train_"):
            expected = expected[: min(round(phi * ids), len(expected))]
        if not np.array_equal(world[name], expected):
            raise ValueError("Common pools/nested support differ from reference: " + name)
    common = common_evaluation_digest(world)
    if common != metadata["common_evaluation_sha256"]:
        raise ValueError("Common evaluation digest differs")
    if reference["metadata"]["dataset_sha256"] != metadata["reference_dataset_sha256"]:
        raise ValueError("High-support reference differs")
    return {
        **checked,
        "low_support_is_high_support_prefix": True,
        "same_atomic_id_ood_and_heldout_pools_across_support": True,
        "common_evaluation_sha256": common,
    }


def run_name(spec):
    seed = spec.get("world_seed", spec.get("world"))
    return (
        f"w{seed}-i{spec['initialization']}-d{spec['width']}-phi{spec['phi']:g}"
        f"-l{spec['layers']}-r{spec['repeats']}-s{spec['steps']}"
    )


def _validate_spec(spec):
    depth_step._validate_spec(spec)
    if float(spec["phi"]) not in SUPPORT_LEVELS:
        raise ValueError("Unsupported composition support")
    if (spec["layers"], spec["repeats"]) not in ARCHITECTURES:
        raise ValueError("Unsupported architecture")


def _check_source(source):
    depth_step._check_source(source)
    required = {"src/llm_memory_editability/multihop_scaling.py", *depth_step.SOURCE_FILES}
    if not required <= set(source):
        raise RuntimeError("Source lock omits experiment dependencies")


def _validate_output(out):
    depth_step._validate_output(out)


def _device_setup(device):
    device = torch.device(device)
    torch.set_num_threads(1)
    # Standalone workers call this once; CPU integration checks may call twice.
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    return device


class _CpuStep:
    """Same objective and optimizer groups for inexpensive engineering checks."""

    def __init__(self, model, optimizer, table, clip):
        self.model, self.optimizer, self.table, self.clip = model, optimizer, table, clip

    def __call__(self, indices):
        self.optimizer.zero_grad(set_to_none=True)
        tokens, labels = (part[indices] for part in self.table)
        logits = self.model(tokens)
        loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), ignore_index=-100)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip, foreach=True)
        self.optimizer.step()
        return loss.detach()


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def train(spec, out, source, device="cuda:0"):
    _validate_spec(spec)
    _check_source(source)
    if "frozen_data_sha256" not in spec or "common_evaluation_sha256" not in spec:
        raise ValueError("Training requires frozen data and common evaluation hashes")
    out = Path(out)
    _validate_output(out)
    out.mkdir(parents=True, exist_ok=True)
    device = _device_setup(device)
    started_process = time.perf_counter()
    world = build_world(spec)
    digest = data_digest(world)
    depth_step._save_world(out, world)
    write_json(out / "spec.json", {"spec": spec, "source": source, "data_sha256": digest})
    model = construct(spec, device)
    initial_digest = model_digest(model)
    if spec.get("initial_model_sha256", initial_digest) != initial_digest:
        raise AssertionError("Frozen initialization differs")
    lr = torch.tensor(spec["lr"], device=device)
    if device.type == "cuda":
        optimizer = make_optimizer(model, lr, spec["weight_decay"])
    else:
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": [p for p in model.parameters() if p.ndim >= 2],
                    "weight_decay": spec["weight_decay"],
                },
                {"params": [p for p in model.parameters() if p.ndim < 2], "weight_decay": 0.0},
            ],
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
    sizes = [len(world[name]) for name in TRAIN_SPLITS]
    if any(size == 0 for size in sizes):
        raise ValueError("Do not reroll a world with an empty sampled training stratum")
    packed = [
        pack_rows(world[name], separator=world["metadata"]["separator_token"])
        for name in TRAIN_SPLITS
    ]
    table = tuple(
        torch.as_tensor(np.concatenate(parts), device=device) for parts in zip(*packed, strict=True)
    )
    offsets = np.cumsum([0, *sizes[:-1]])
    streams = [EpochStream(size, spec["stream_seed"] + i) for i, size in enumerate(sizes)]
    counts = [np.zeros(size, dtype=np.int64) for size in sizes]
    if device.type == "cuda":
        graph = FullTokenStep(
            model, optimizer, table, spec["batch_size"], clip=spec.get("clip", 1.0)
        )
    else:
        graph = _CpuStep(model, optimizer, table, spec.get("clip", 1.0))
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
            **dict(zip(TRAIN_SPLITS, counts, strict=True)),
        )
        state = {
            "spec": spec,
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "streams": [s.state_dict() for s in streams],
            "counts": counts,
            "cpu_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
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
    endpoint = history[0]
    for end in spec["nodes"][1:]:
        _synchronize(device)
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
        _synchronize(device)
        training_seconds += time.perf_counter() - started
        loss_value = float(loss)
        if not np.isfinite(loss_value):
            raise FloatingPointError("Nonfinite loss at step " + str(step))
        endpoint = measure(loss_value)
    np.savez_compressed(out / "exposures.npz", **dict(zip(TRAIN_SPLITS, counts, strict=True)))
    result = {
        "spec": spec,
        "source": source,
        "data_sha256": digest,
        "common_evaluation_sha256": world["metadata"]["common_evaluation_sha256"],
        "initial_model_sha256": initial_digest,
        "checkpoint_sha256": file_hash(out / "latest.pt"),
        "finished_utc": utc(),
        "parameters": sum(p.numel() for p in model.parameters()),
        "architecture": "loop" if spec["repeats"] > 1 else "standard",
        "effective_depth": spec["layers"] * spec["repeats"],
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
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "tf32": device.type == "cuda",
        },
    }
    write_json(out / "complete.json", result)
    return result


def audit(out, device="cuda:0"):
    out = Path(out)
    if (out / "audit.json").exists():
        raise FileExistsError(out / "audit.json")
    device = _device_setup(device)
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
    if result["common_evaluation_sha256"] != common_evaluation_digest(world):
        raise AssertionError("Common evaluation identity differs")
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
            if set(actual.files) != set(TRAIN_SPLITS):
                raise AssertionError("Exposure strata differ")
            for name, count in zip(TRAIN_SPLITS, counts, strict=True):
                np.testing.assert_array_equal(actual[name], count)
        row = by_node[node]
        if row["supervised_tokens"] != node * 832 or row["padded_input_tokens"] != node * 1024:
            raise AssertionError("Token accounting differs")
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
        "nested_support_and_common_pools_checked": True,
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
