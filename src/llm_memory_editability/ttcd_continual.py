"""Paired single-fact versus recurrent multi-fact TTCD outer training.

Only atomic correction labels train the fast-state interface. Real GPT blocks
and the audited local writer are reused; no edited composite is queried here.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .grok_depth import write_json
from .grokking_reproduction import model_digest
from .path_overlap import sha256
from .ttcd_editing import TTCDReader, correction_packet
from .ttcd_scratch import (
    CODE_FILES,
    data_contract,
    new_scratch_reader,
    row_loss,
    session_write,
    training_readout,
    validate_scratch_inputs,
)

EXTRA_CODE = (
    "src/llm_memory_editability/ttcd_continual.py",
    "src/llm_memory_editability/ttcd_continual_eval.py",
    "scripts/run_ttcd_continual.py",
    "scripts/launch_ttcd_continual.py",
)


def validate_config(config):
    world = validate_scratch_inputs(config)
    c, t, parent = config["continual"], config["training"], config["parent"]
    if set(config["models"]) != {"single", "stream"}:
        raise ValueError("The paired single/stream matrix is required")
    for key in ("train_length", "eval_length", "rotations", "affected", "controls"):
        if isinstance(c[key], bool) or not isinstance(c[key], int) or c[key] < 1:
            raise ValueError(f"Invalid continual {key}")
    if t["batch_size"] % c["train_length"]:
        raise ValueError("The fact batch must be divisible by stream length")
    if parent["checkpoint_step"] != t["warmup_steps"] or parent["steps"] != t["adaptation_steps"]:
        raise ValueError("Warmup and additional training budget differ from the pinned parent")
    if parent["parent_model"] != "discoloop" or any(
        not spec["discoloop"] for spec in config["models"].values()
    ):
        raise ValueError("This comparison holds the DiscoLoop architecture fixed")
    if any(v != 1.0 for v in t["loss_weights"].values()):
        raise ValueError("The five matched objectives must retain unit coefficients")
    split = json.loads((Path(config["data_dir"]) / "training-split.json").read_text())
    if len(split["train_edit_atoms"]) < c["train_length"]:
        raise ValueError("Not enough distinct training keys")
    return world


def identity(config):
    root = Path(__file__).resolve().parents[2]
    inputs = [Path(config["model_config"])] + [
        Path(config["data_dir"]) / n
        for n in ("world.npz", "panels.npz", "complete.json", "training-split.json")
    ]
    return {
        "config": config,
        "inputs": {str(p): sha256(p) for p in inputs},
        "code": {p: sha256(root / p) for p in (*CODE_FILES, *EXTRA_CODE)},
    }


def validate_parent(config, name):
    """Check immutable parent bytes, complete warmup state and scientific recipe."""
    world = validate_config(config)
    spec = config["parent"]
    manifest_path, source = Path(spec["parent_manifest_path"]), Path(spec["parent_source_dir"])
    if sha256(manifest_path) != spec["parent_manifest_sha256"]:
        raise ValueError("Pinned parent source manifest differs")
    manifest = json.loads(manifest_path.read_text())
    for relative, expected in manifest["files"].items():
        path = source / relative
        if not path.resolve().is_relative_to(source.resolve()) or sha256(path) != expected:
            raise ValueError(f"Frozen parent file differs: {relative}")
    original_path = source / "configs/experiment.json"
    if sha256(original_path) != spec["parent_config_sha256"]:
        raise ValueError("Pinned parent configuration differs")
    original = json.loads(original_path.read_text())
    if original["batch"] != spec["parent_batch"] or data_contract(original) != data_contract(
        config
    ):
        raise ValueError("Parent graph/model/reader contract differs")
    for key in (
        "warmup_steps",
        "adaptation_steps",
        "batch_size",
        "learning_rate",
        "minimum_learning_rate",
        "lr_warmup_steps",
        "weight_decay",
        "gradient_clip",
    ):
        if original["training"][key] != config["training"][key]:
            raise ValueError(f"Parent training recipe changed: {key}")
    path = Path(spec["checkpoints"][name])
    if sha256(path) != spec["checkpoint_sha256"]:
        raise ValueError("Pinned warmup checkpoint differs")
    parent = torch.load(path, map_location="cpu", weights_only=True)
    expected_step = spec["checkpoint_step"]
    if (
        parent["schema"] != "ttcd-scratch-parent-v1"
        or parent["step"] != expected_step
        or parent.get("branch") is not None
        or parent["training_identity"]["config"] != original
        or parent["architecture"] != {"discoloop": True, "first_round_kv": False}
        or parent["spec"]["name"] != spec["parent_model"]
        or parent["counts"]["fast_state_updates"] != 0
        or parent["counts"]["corrected_two_hop_training_examples"] != 0
        or parent["counts"]["base_parameter_updates"] != expected_step
        or parent["counts"]["supervised_target_tokens"]
        != expected_step * config["training"]["batch_size"] * 6
    ):
        raise ValueError("Expected this architecture's complete pre-write warmup endpoint")
    for relative, expected in parent["training_identity"]["code"].items():
        if manifest["files"].get(relative) != expected:
            raise ValueError("Checkpoint does not bind the frozen parent source")
    current_inputs = identity(config)["inputs"]
    if current_inputs != parent["training_identity"]["inputs"]:
        raise ValueError("Current input bytes differ from the inherited checkpoint")
    if not parent["optimizer"]["state"]:
        raise ValueError("Complete inherited AdamW state is required")
    provenance = {
        "checkpoint_sha256": sha256(path),
        "model_sha256": parent["model_sha256"],
        "parent_manifest_sha256": spec["parent_manifest_sha256"],
        "step": expected_step,
        "inherited_counts": parent["counts"],
    }
    return parent, provenance, world


def restore_data_rng(state):
    rng = np.random.RandomState()
    rng.set_state(
        (
            state["algorithm"],
            np.asarray(state["keys"], dtype=np.uint32),
            state["position"],
            state["has_gauss"],
            state["cached_gaussian"],
        )
    )
    return rng


def pack_rng(rng):
    state = rng.get_state()
    return dict(
        algorithm=state[0],
        keys=state[1].tolist(),
        position=state[2],
        has_gauss=state[3],
        cached_gaussian=state[4],
    )


def sample_batch(config, world, rng):
    """Identical labels for both arms; retention excludes every key in a stream."""
    n, length = config["training"]["batch_size"], config["continual"]["train_length"]
    split = json.loads((Path(config["data_dir"]) / "training-split.json").read_text())
    edit_ids = np.stack(
        [rng.choice(split["train_edit_atoms"], length, replace=False) for _ in range(n // length)]
    )
    flat = edit_ids.reshape(-1)
    atoms = world.atoms[flat]
    tails = rng.randint(world.entities - 1, size=n)
    tails += tails >= atoms[:, 2]
    controls = np.concatenate(
        [
            rng.choice(np.setdiff1d(np.arange(len(world.atoms)), ids), length, replace=True)
            for ids in edit_ids
        ]
    )
    packets = [
        correction_packet(
            atom,
            int(tail),
            atom[:2],
            world.metadata,
            **{k: config["writer"][k] for k in ("gap", "teacher_window", "student_window")},
        )
        for atom, tail in zip(atoms, tails, strict=True)
    ]
    corrected = atoms.copy()
    corrected[:, 2] = tails
    return {
        "edit_ids": flat,
        "retention_ids": controls,
        "old_atoms": world.atoms[rng.randint(len(world.atoms), size=n)],
        "old_chains": world.chains[rng.choice(np.flatnonzero(world.support(config["phi"])), n)],
        "corrected_atoms": corrected,
        "retention_atoms": world.atoms[controls],
        "packets": np.asarray(packets, dtype=np.int64),
    }


def write_states(reader, packets, writer, length, name, *, observe=None):
    """Return one differentiable final fast state per query row in original order."""
    if name == "single":
        fast = None
        for _ in range(writer["steps"]):
            fast = session_write(reader, packets, fast, writer)
        return fast
    if name != "stream" or len(packets) % length:
        raise ValueError("Unknown arm or incomplete fact streams")
    grouped = packets.reshape(-1, length, packets.shape[-1])
    fast = None
    for position in range(length):
        for _ in range(writer["steps"]):
            fast = session_write(reader, grouped[:, position], fast, writer)
        if observe is not None:
            observe(position, fast)
    return {layer: state.repeat_interleave(length, dim=0) for layer, state in fast.items()}


def outer_objective(reader, config, world, batch, name):
    device = next(reader.parameters()).device
    losses = {
        "atoms": row_loss(reader, batch["old_atoms"], world.metadata),
        "chains": row_loss(reader, batch["old_chains"], world.metadata),
    }
    packets = torch.as_tensor(batch["packets"], device=device)
    new = torch.as_tensor(batch["corrected_atoms"][:, 2] + world.eo, device=device)
    tokens = torch.cat((packets, new[:, None]), dim=1)
    positions = packets.new_tensor([packets.shape[1] - 1, packets.shape[1]])[None].expand(
        len(packets), -1
    )
    labels = torch.stack((new, torch.full_like(new, world.metadata["end_marker"])), dim=1)
    logits = reader(tokens, positions, window=config["writer"]["teacher_window"])
    losses["reader"] = F.cross_entropy(logits.flatten(0, 1), labels.flatten())
    fast = write_states(
        reader, packets, config["writer"], config["continual"]["train_length"], name
    )
    losses["fast_atomic"] = row_loss(reader, batch["corrected_atoms"], world.metadata, fast=fast)
    losses["fast_retention"] = row_loss(reader, batch["retention_atoms"], world.metadata, fast=fast)
    loss = sum(config["training"]["loss_weights"][k] * v for k, v in losses.items())
    return loss, {k: float(v.detach()) for k, v in losses.items()}


def compare_replay(expected, observed):
    if isinstance(expected, dict):
        if set(expected) != set(observed):
            raise ValueError("Replay keys differ")
        for key in expected:
            compare_replay(expected[key], observed[key])
    elif isinstance(expected, list):
        if len(expected) != len(observed):
            raise ValueError("Replay coverage differs")
        for a, b in zip(expected, observed, strict=True):
            compare_replay(a, b)
    elif isinstance(expected, float):
        if not math.isclose(expected, observed, rel_tol=1e-5, abs_tol=1e-5):
            raise ValueError("Replay numeric value differs")
    elif expected != observed:
        raise ValueError("Replay token/rank/value differs")


def new_counts():
    return dict(
        base_parameter_updates=0,
        old_atomic_examples=0,
        old_chain_examples=0,
        correction_examples=0,
        fast_state_updates=0,
        fast_atomic_examples=0,
        fast_retention_examples=0,
        supervised_target_tokens=0,
        corrected_two_hop_training_examples=0,
    )


def train(config, name, out, device, log=None):
    parent, provenance, world = validate_parent(config, name)
    directory = Path(out) / "training"
    directory.mkdir(parents=True, exist_ok=False)
    backup = directory / "parent-input.pt"
    shutil.copy2(config["parent"]["checkpoints"][name], backup)
    if sha256(backup) != provenance["checkpoint_sha256"]:
        raise ValueError("Warmup checkpoint changed while copying")
    reader = new_scratch_reader(config, name, world, device)
    reader.backbone.load_state_dict(parent["model"], strict=True)
    if model_digest(reader.backbone) != parent["model_sha256"]:
        raise ValueError("Warmup parameter digest differs")
    t = config["training"]
    optimizer = torch.optim.AdamW(
        reader.parameters(), lr=t["learning_rate"], weight_decay=t["weight_decay"]
    )
    optimizer.load_state_dict(parent["optimizer"])
    rng = restore_data_rng(parent["data_rng"])
    torch.set_rng_state(parent["torch_rng"].cpu())
    if str(device).startswith("cuda"):
        if len(parent["cuda_rng"]) != torch.cuda.device_count():
            raise ValueError("Inherited isolated CUDA RNG layout differs")
        torch.cuda.set_rng_state_all([s.cpu() for s in parent["cuda_rng"]])
        torch.cuda.reset_peak_memory_stats(device)
    frozen_identity = identity(config)
    counts, compute, seconds = new_counts(), dict(reader.compute), 0.0
    trace = hashlib.sha256()
    start = time.monotonic()
    initial_step, total = parent["step"], parent["step"] + config["parent"]["steps"]

    def checkpoint(step, readout):
        terminal = step == total
        path = directory / ("parent.pt" if terminal else f"checkpoint-{step:07d}.pt")
        payload = {
            "schema": "ttcd-continual-parent-v1",
            "step": step,
            "arm": name,
            "model": {k: v.detach().cpu().clone() for k, v in reader.backbone.state_dict().items()},
            "model_sha256": model_digest(reader.backbone),
            "optimizer": optimizer.state_dict(),
            "identity": frozen_identity,
            "parent": provenance,
            "additional_counts": dict(counts),
            "training_compute": dict(compute),
            "compute_including_validation": dict(reader.compute),
            "training_seconds": seconds,
            "wall_seconds": time.monotonic() - start,
            "data_rng": pack_rng(rng),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if str(device).startswith("cuda") else [],
            "trace_sha256": trace.hexdigest(),
            "readout": readout,
        }
        temporary = path.with_suffix(".partial")
        torch.save(payload, temporary)
        temporary.replace(path)
        write_json(
            directory / "checkpoint-index.json",
            dict(step=step, path=str(path), sha256=sha256(path), terminal=terminal),
        )

    readout = training_readout(reader, config, world, device)
    compare_replay(parent["readout"], readout)
    checkpoint(initial_step, readout)
    with (directory / "curves.jsonl").open("x") as curves:
        for step in range(initial_step + 1, total + 1):
            before, compute_before = time.monotonic(), dict(reader.compute)
            fraction = (step - 1) / max(1, total - 1)
            rate = t["minimum_learning_rate"] + 0.5 * (
                t["learning_rate"] - t["minimum_learning_rate"]
            ) * (1 + math.cos(math.pi * fraction))
            rate *= min(1.0, step / max(1, t["lr_warmup_steps"]))
            for group in optimizer.param_groups:
                group["lr"] = rate
            optimizer.zero_grad(set_to_none=True)
            batch = sample_batch(config, world, rng)
            for key, value in sorted(batch.items()):
                trace.update(key.encode())
                trace.update(value.astype("<i8", copy=False).tobytes())
            loss, losses = outer_objective(reader, config, world, batch, name)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite continual outer loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                reader.parameters(), t["gradient_clip"], error_if_nonfinite=True
            )
            optimizer.step()
            if str(device).startswith("cuda"):
                torch.cuda.synchronize(device)
            seconds += time.monotonic() - before
            for key in compute:
                compute[key] += reader.compute[key] - compute_before[key]
            n = t["batch_size"]
            counts["base_parameter_updates"] += 1
            for key in (
                "old_atomic_examples",
                "old_chain_examples",
                "correction_examples",
                "fast_atomic_examples",
                "fast_retention_examples",
            ):
                counts[key] += n
            counts["fast_state_updates"] += n * config["writer"]["steps"]
            counts["supervised_target_tokens"] += n * 10
            record = dict(
                step=step,
                additional_step=step - initial_step,
                phase=name,
                lr=rate,
                loss=float(loss.detach()),
                losses=losses,
                gradient_norm=float(norm),
                training_seconds=seconds,
                wall_seconds=time.monotonic() - start,
                counts=dict(counts),
                compute=dict(compute),
                trace_sha256=trace.hexdigest(),
            )
            if step % t["validation_every"] == 0 or step == total:
                readout = training_readout(reader, config, world, device)
                write_json(directory / f"validation-{step:07d}.json", readout)
                record["validation"] = {k: v["metrics"] for k, v in readout["panels"].items()}
                record["calibration"] = {
                    group: {
                        "open_new_accuracy": float(
                            np.mean(
                                [
                                    x["open"]["new_correct"]
                                    for x in readout["calibration_atoms"]
                                    if x["group"] == group
                                ]
                            )
                        ),
                        "closed_new_accuracy": float(
                            np.mean(
                                [
                                    x["metrics"]["new_accuracy"]
                                    for x in readout["calibration_atoms"]
                                    if x["group"] == group
                                ]
                            )
                        ),
                    }
                    for group in ("id", "ood")
                }
                checkpoint(step, readout)
            curves.write(json.dumps(record) + "\n")
            curves.flush()
            write_json(directory / "progress.json", record)
            if log:
                log(step, record)
    if frozen_identity != identity(config):
        raise ValueError("Continual code/config/inputs changed during execution")
    terminal = directory / "parent.pt"
    result = dict(
        status="training_complete_pending_independent_audit",
        step=total,
        arm=name,
        parent_sha256=sha256(terminal),
        model_sha256=model_digest(reader.backbone),
        inherited=provenance,
        additional_counts=counts,
        training_compute=compute,
        compute_including_validation=reader.compute,
        training_seconds=seconds,
        wall_seconds=time.monotonic() - start,
        trace_sha256=trace.hexdigest(),
        identity=frozen_identity,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device)
        if str(device).startswith("cuda")
        else None,
    )
    write_json(directory / "summary.json", result)
    return result


def load_reader(config, name, device, require_audit=True):
    world = validate_config(config)
    path = Path(config["models"][name]["checkpoint"])
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    current_identity = identity(config)
    if (
        checkpoint["schema"] != "ttcd-continual-parent-v1"
        or checkpoint["arm"] != name
        or checkpoint["identity"] != current_identity
        or checkpoint["step"] != config["models"][name]["step"]
    ):
        raise ValueError("Continual endpoint identity/step differs")
    reader = new_scratch_reader(config, name, world, device)
    reader.backbone.load_state_dict(checkpoint["model"], strict=True)
    if model_digest(reader.backbone) != checkpoint["model_sha256"]:
        raise ValueError("Continual model digest differs")
    reader = TTCDReader(reader.backbone).eval()
    if require_audit:
        receipt = json.loads((path.parent / "audit.json").read_text())
        if (
            not receipt["passed"]
            or receipt["parent_sha256"] != sha256(path)
            or receipt["identity"] != current_identity
        ):
            raise ValueError("Independent training reload audit must bind this endpoint")
    return reader, world


def audit_training(config, name, out, device):
    reader, world = load_reader(config, name, device, require_audit=False)
    directory = Path(out) / "training"
    summary = json.loads((directory / "summary.json").read_text())
    terminal = directory / "parent.pt"
    if summary["parent_sha256"] != sha256(terminal) or summary["identity"] != identity(config):
        raise ValueError("Training completion summary differs")
    coverage = []
    checkpoints = sorted(directory.glob("checkpoint-*.pt")) + [terminal]
    for path in checkpoints:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint["identity"] != identity(config):
            raise ValueError("Saved checkpoint identity differs")
        reader.backbone.load_state_dict(checkpoint["model"], strict=True)
        if model_digest(reader.backbone) != checkpoint["model_sha256"]:
            raise ValueError("Saved checkpoint tensor digest differs")
        actual = training_readout(reader, config, world, device)
        compare_replay(checkpoint["readout"], actual)
        coverage.append(dict(step=checkpoint["step"], sha256=sha256(path)))
    result = dict(
        passed=True,
        status="training_independently_reloaded",
        arm=name,
        parent_sha256=sha256(terminal),
        identity=identity(config),
        checkpoints=coverage,
        step=config["models"][name]["step"],
        trace_sha256=summary["trace_sha256"],
    )
    write_json(directory / "audit.json", result)
    return result
