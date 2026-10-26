"""TTCD adaptation of an independently verified successful Wang DiscoLoop.

The foundation retains its original recipe. The common record reader and the
paired write learners are explicitly new adaptation stages. Updated two-hop
answers never enter a training objective or a training checkpoint decision.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import shutil
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from transformers import GPT2Config

from .edit_scope_development import evaluate_queries, prediction_metrics
from .grok_depth import utc, write_json
from .grokking_reproduction import MergedBatchStream, encode_rows, model_digest
from .path_overlap import World, sha256
from .supervised_alignment import SupervisedAlignmentGPT
from .ttcd_continual import compare_replay, pack_rng, restore_data_rng, sample_batch, write_states
from .ttcd_continual_eval import build_panel, case_bank
from .ttcd_editing import TTCDReader, build_case_bank, correction_packet
from .ttcd_experiment import open_query, packet_for
from .ttcd_scratch import CODE_FILES as SHARED_CODE

PANELS = ("atomic_id", "atomic_ood", "train_ii", "test_ii", "test_io", "test_oi", "test_oo")
CODE_FILES = tuple(
    dict.fromkeys(
        (
            *SHARED_CODE,
            "src/llm_memory_editability/ttcd_continual.py",
            "src/llm_memory_editability/ttcd_continual_eval.py",
            "src/llm_memory_editability/ttcd_verified_base.py",
            "scripts/run_ttcd_verified_base.py",
        )
    )
)


def prepare_data(config, original_data_dir=None, original_model_config=None):
    """Copy original bytes and add correction-only exclusions in a new directory."""
    original = Path(original_data_dir or config["original_data_dir"])
    model_source = Path(original_model_config or config["original_model_config"])
    data, model_path = Path(config["data_dir"]), Path(config["model_config"])
    if data.resolve() == original.resolve():
        raise ValueError("Adaptation inputs must not overwrite original Wang data")
    if (data / "ttcd-input-manifest.json").exists():
        world = validate_inputs(config)
        if config["continual"].get("panel_source_path"):
            panel = build_panel(config, world)
            panel_path = Path(config["continual"]["panel_source_path"])
            if panel_path.exists() and json.loads(panel_path.read_text()) != panel:
                raise ValueError("Existing prepared panel differs from the fixed graph panel")
            if not panel_path.exists():
                write_json(panel_path, panel)
        return world.metadata
    if data.exists() and any(data.iterdir()):
        raise FileExistsError("Preserve incomplete adaptation inputs; use an amended path")
    data.mkdir(parents=True, exist_ok=True)
    for name in ("world.npz", "panels.npz", "complete.json"):
        shutil.copyfile(original / name, data / name)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    if model_path.resolve() == model_source.resolve():
        raise ValueError("Use a distinct copied model configuration")
    shutil.copyfile(model_source, model_path)
    world = World(data)
    bank = build_case_bank(world, phi=config["phi"], **config["cases"])
    reserved = {case["atom"] for case in bank}
    split = {
        "schema": "ttcd-verified-base-correction-split-v1",
        "case_bank": bank,
        "train_edit_atoms": sorted(set(range(len(world.atoms))) - reserved),
        "rule": "Only correction episodes exclude every development/evaluation target key; "
        "the inherited old-world native CE still includes original facts and original II.",
    }
    write_json(data / "training-split.json", split)
    write_json(
        data / "ttcd-input-manifest.json",
        {
            "schema": "ttcd-verified-base-input-v1",
            "original_data_dir": str(original),
            "original_model_config": str(model_source),
            "files": {
                name: sha256(data / name)
                for name in ("world.npz", "panels.npz", "complete.json", "training-split.json")
            },
            "model_config_sha256": sha256(model_path),
            "phi": config["phi"],
            "cases": config["cases"],
        },
    )
    panel = build_panel(config, world)
    panel_path = config["continual"].get("panel_source_path")
    if panel_path:
        write_json(Path(panel_path), panel)
    validate_inputs(config)
    return world.metadata


def validate_inputs(config):
    data = Path(config["data_dir"])
    manifest = json.loads((data / "ttcd-input-manifest.json").read_text())
    if manifest["phi"] != config["phi"] or manifest["cases"] != config["cases"]:
        raise ValueError("Adaptation support/case contract changed")
    for name, expected in manifest["files"].items():
        if sha256(data / name) != expected:
            raise ValueError(f"Prepared adaptation input differs: {name}")
    if sha256(config["model_config"]) != manifest["model_config_sha256"]:
        raise ValueError("Copied original model configuration differs")
    world = World(data)
    for name in ("world", "panels"):
        if sha256(data / f"{name}.npz") != world.metadata[f"{name}_sha256"]:
            raise ValueError("Original world/panel hash mismatch")
    bank = build_case_bank(world, phi=config["phi"], **config["cases"])
    split = json.loads((data / "training-split.json").read_text())
    reserved = {case["atom"] for case in bank}
    if split["case_bank"] != bank or split["train_edit_atoms"] != sorted(
        set(range(len(world.atoms))) - reserved
    ):
        raise ValueError("Training correction keys overlap reserved facts or bank changed")
    if set(config["models"]) != {"single", "stream"}:
        raise ValueError("Expected precisely the single/stream comparison")
    training = config["training"]
    for field in (
        "steps",
        "native_batch_size",
        "correction_batch_size",
        "micro_native",
        "micro_facts",
        "validation_every",
        "evaluation_batch_size",
    ):
        if (
            isinstance(training[field], bool)
            or int(training[field]) != training[field]
            or training[field] < 1
        ):
            raise ValueError(f"Invalid positive training integer: {field}")
    length = config["continual"]["train_length"]
    if training["correction_batch_size"] % length or training["micro_facts"] % length:
        raise ValueError("Microbatches must contain complete shared-state streams")
    if any(spec["step"] != training["steps"] for spec in config["models"].values()):
        raise ValueError("Both arms must use the same fixed terminal update budget")
    if any(not spec["discoloop"] for spec in config["models"].values()):
        raise ValueError("The successful DiscoLoop architecture is held fixed")
    if (
        not all(
            math.isfinite(training[key])
            for key in ("learning_rate", "gradient_clip", "weight_decay")
        )
        or min(training["learning_rate"], training["gradient_clip"]) <= 0
        or training["weight_decay"] < 0
    ):
        raise ValueError("Invalid adaptation optimization settings")
    for case in bank:
        correction_packet(
            world.atoms[case["atom"]],
            case["new_tail"],
            world.atoms[case["atom"], :2],
            world.metadata,
            **{key: config["writer"][key] for key in ("gap", "teacher_window", "student_window")},
        )
    return world


def identity(config):
    root = Path(__file__).resolve().parents[2]
    paths = [Path(config["model_config"])] + [
        Path(config["data_dir"]) / name
        for name in (
            "world.npz",
            "panels.npz",
            "complete.json",
            "training-split.json",
            "ttcd-input-manifest.json",
        )
    ]
    paths += [Path(config["foundation"][key]) for key in ("checkpoint", "gate_receipt")]
    if config["foundation"].get("config_path"):
        paths.append(Path(config["foundation"]["config_path"]))
    return {
        "config": config,
        "inputs": {str(path): sha256(path) for path in paths},
        "code": {name: sha256(root / name) for name in CODE_FILES},
    }


def _load(path):
    # Foundation latest.pt contains the original NumPy epoch-stream state.
    # These are trusted locally frozen project checkpoints, hash-bound below.
    return torch.load(path, map_location="cpu", weights_only=False)


def _backbone(config, saved, world, device):
    hf = copy.deepcopy(GPT2Config.from_json_file(config["model_config"]))
    hf.vocab_size, hf.n_layer = world.metadata["vocab_size"], saved["spec"]["unique_layers"]
    hf.use_cache = False
    # Explicit new deterministic adaptation stage; original JSON is preserved.
    hf.embd_pdrop = hf.resid_pdrop = hf.attn_pdrop = 0.0
    flags = saved["architecture"]
    if not flags.get("discoloop") or flags.get("first_round_kv"):
        raise ValueError("Expected standalone successful DiscoLoop foundation")
    model = SupervisedAlignmentGPT(hf, saved["spec"]["repeats"], **flags)
    model.load_state_dict(saved["model"], strict=True)
    if saved.get("model_sha256") and model_digest(model) != saved["model_sha256"]:
        raise ValueError("Checkpoint internal model digest differs")
    return model.to(device).eval()


def foundation_inputs(config, device, *, trainable=False):
    world = validate_inputs(config)
    parent = config["foundation"]
    checkpoint_path = Path(parent["checkpoint"])
    digest = sha256(checkpoint_path)
    if parent.get("checkpoint_sha256") and digest != parent["checkpoint_sha256"]:
        raise ValueError("Pinned foundation checkpoint differs")
    receipt = json.loads(Path(parent["gate_receipt"]).read_text())
    field = parent.get("gate_checkpoint_field", "checkpoint_sha256")
    if not receipt["passed"] or receipt[field] != digest:
        raise ValueError("Successful foundation gate must bind the exact checkpoint")
    saved = _load(checkpoint_path)
    if saved["step"] != parent["step"] or saved["spec"]["phi"] != config["phi"]:
        raise ValueError("Wrong foundation terminal step or combination support")
    for key, expected in parent.get("required_spec", {}).items():
        if saved["spec"].get(key) != expected:
            raise ValueError(f"Foundation successful recipe differs: {key}")
    if parent.get("config_path"):
        original_config = json.loads(Path(parent["config_path"]).read_text())
        if saved["spec"] != original_config["spec"]:
            raise ValueError("Foundation checkpoint differs from the frozen restoration recipe")
    model = _backbone(config, saved, world, device)
    if receipt.get("model_sha256") and model_digest(model) != receipt["model_sha256"]:
        raise ValueError("Foundation gate binds another model")
    reader = TTCDReader(model, trainable_backbone=trainable, native_causal_attention=True).eval()
    provenance = {
        "checkpoint_sha256": digest,
        "model_sha256": model_digest(model),
        "gate_receipt_sha256": sha256(parent["gate_receipt"]),
        "step": saved["step"],
        "spec": saved["spec"],
        "architecture": saved["architecture"],
    }
    return reader, world, saved, provenance


@torch.no_grad()
def native_readout(reader, config, world, device):
    """Use original CUDA BF16 greedy scoring; CPU fixtures use float32.

    Common-stage retention compares both models through this same precision
    and wrapper. The original foundation's separate native BF16 audit remains
    the prerequisite rather than being silently replaced by FP32 measurements.
    """
    panels = dict(np.load(Path(config["data_dir"]) / "panels.npz"))
    if set(panels) != set(PANELS):
        raise ValueError("The seven original Wang panels are required")
    records = {}
    precision = "bf16" if str(device).startswith("cuda") else "fp32"
    for name in PANELS:
        prediction = evaluate_queries(
            reader,
            panels[name],
            world.metadata,
            torch.device(device),
            batch_size=config["training"]["evaluation_batch_size"],
            precision=precision,
        )
        records[name] = {
            "precision": precision,
            "metrics": prediction_metrics(prediction, world.metadata),
            "predictions": {key: value.tolist() for key, value in prediction.items()},
        }
    return records


@torch.no_grad()
def wrapper_agreement(reader, config, world, device):
    panels = dict(np.load(Path(config["data_dir"]) / "panels.npz"))
    results = []
    for name in PANELS:
        arrays = encode_rows(panels[name][:8], world.metadata)
        tokens, positions = (torch.as_tensor(array, device=device) for array in arrays[:2])
        for precision in ("fp32", "bf16") if str(device).startswith("cuda") else ("fp32",):
            with (
                torch.autocast("cuda", dtype=torch.bfloat16)
                if precision == "bf16"
                else nullcontext()
            ):
                reference = reader.backbone(tokens, positions)
                observed = reader(tokens, positions)
            exact_tokens = torch.equal(reference.argmax(-1), observed.argmax(-1))
            torch.testing.assert_close(observed, reference, rtol=1e-5, atol=1e-5)
            if not exact_tokens:
                raise AssertionError("Native foundation/wrapper greedy tokens differ")
            results.append(
                {
                    "panel": name,
                    "precision": precision,
                    "tokens_exact": True,
                    "max_logit_difference": float((observed - reference).abs().max()),
                }
            )
    return results


def preflight(config, out, device):
    reader, world, _, parent = foundation_inputs(config, device)
    result = {
        "passed": True,
        "identity": identity(config),
        "foundation": parent,
        "wrapper_agreement": wrapper_agreement(reader, config, world, device),
        "native_readout": native_readout(reader, config, world, device),
        "formal_D_used": False,
    }
    write_json(Path(out) / "preflight.json", result)
    return result


def _amp(device):
    return (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if str(device).startswith("cuda")
        else nullcontext()
    )


def native_table(world, phi):
    # Preserve the original table's row order: an epoch-stream index refers
    # to the same fact/composition as in run_supervised_alignment.make_table.
    support = round(phi * world.metadata["atomic_id"])
    selected = world.train_order[:support]
    if len(selected) != support:
        raise ValueError("The requested original composition support is unavailable")
    atoms, chains = (
        encode_rows(world.atoms, world.metadata),
        encode_rows(world.chains[selected], world.metadata),
    )
    return tuple(np.concatenate((a, b)) for a, b in zip(atoms, chains, strict=True)), len(
        world.atoms
    )


def sample_step(config, world, rng, stream):
    proxy = {
        **config,
        "training": {
            **config["training"],
            "batch_size": config["training"]["correction_batch_size"],
        },
    }
    batch = sample_batch(proxy, world, rng)
    indices, count = stream.batch(config["training"]["native_batch_size"])
    batch["native_indices"] = indices[:count]
    return batch


def _reader_loss(reader, config, world, batch):
    device = next(reader.parameters()).device
    packets = torch.as_tensor(batch["packets"], device=device)
    new = torch.as_tensor(batch["corrected_atoms"][:, 2] + world.eo, device=device)
    tokens = torch.cat((packets, new[:, None]), dim=1)
    positions = packets.new_tensor([packets.shape[1] - 1, packets.shape[1]])[None].expand(
        len(packets), -1
    )
    labels = torch.stack((new, torch.full_like(new, world.metadata["end_marker"])), dim=1)
    logits = reader(tokens, positions, window=config["writer"]["teacher_window"])
    return F.cross_entropy(logits.flatten(0, 1), labels.flatten())


def _row_loss(reader, rows, metadata, fast=None):
    arrays = encode_rows(rows, metadata)
    tokens, positions, labels = (
        torch.as_tensor(x, device=next(reader.parameters()).device) for x in arrays
    )
    return F.cross_entropy(reader(tokens, positions, fast=fast).flatten(0, 1), labels.flatten())


def backward_objective(reader, config, world, table, batch, phase, *, micro=True):
    """Same full mean objective, one optimizer update after all microbatches.

    Entire streams are microbatch units. Teachers are stopped by the original
    audited writer; no fast-state detach occurs between successive facts.
    """
    if phase not in ("common", "single", "stream"):
        raise ValueError("Unknown verified-base phase")
    device = next(reader.parameters()).device
    settings = config["training"]
    losses = {
        key: 0.0
        for key in (
            ("native", "reader")
            if phase == "common"
            else ("native", "reader", "fast_atomic", "fast_retention")
        )
    }
    indices = batch["native_indices"]
    size = settings["micro_native"] if micro else len(indices)
    for start in range(0, len(indices), size):
        part = indices[start : start + size]
        tokens, positions, labels = (torch.as_tensor(x[part], device=device) for x in table)
        with _amp(device):
            loss = F.cross_entropy(reader(tokens, positions).flatten(0, 1), labels.flatten())
        weight = len(part) / len(indices)
        (loss * weight).backward()
        losses["native"] += float(loss.detach()) * weight
    count = len(batch["packets"])
    size = settings["micro_facts"] if micro else count
    for start in range(0, count, size):
        part = {
            key: value[start : start + size]
            for key, value in batch.items()
            if key != "native_indices"
        }
        weight = len(part["packets"]) / count
        with _amp(device):
            loss = _reader_loss(reader, config, world, part)
        # This graph is independent of the write/read graph and can be freed now.
        (loss * weight).backward()
        losses["reader"] += float(loss.detach()) * weight
        if phase != "common":
            with _amp(device):
                packets = torch.as_tensor(part["packets"], device=device)
                fast = write_states(
                    reader, packets, config["writer"], config["continual"]["train_length"], phase
                )
                atomic = _row_loss(reader, part["corrected_atoms"], world.metadata, fast)
                retention = _row_loss(reader, part["retention_atoms"], world.metadata, fast)
            ((atomic + retention) * weight).backward()
            losses["fast_atomic"] += float(atomic.detach()) * weight
            losses["fast_retention"] += float(retention.detach()) * weight
    if not all(np.isfinite(value) for value in losses.values()):
        raise FloatingPointError("Nonfinite verified-base adaptation loss")
    return sum(losses.values()), losses


def optimizer_for(reader, config):
    settings = config["training"]
    named = list(reader.named_parameters())
    return torch.optim.AdamW(
        [
            {
                "params": [
                    p for name, p in named if not any(key in name for key in ("bias", "ln"))
                ],
                "weight_decay": settings["weight_decay"],
            },
            {
                "params": [p for name, p in named if any(key in name for key in ("bias", "ln"))],
                "weight_decay": 0.0,
            },
        ],
        lr=settings["learning_rate"],
        betas=(0.9, 0.999),
        eps=1e-8,
    )


def _counts():
    return {
        key: 0
        for key in (
            "base_parameter_updates",
            "native_atomic_examples",
            "native_chain_examples",
            "correction_examples",
            "fast_atomic_examples",
            "fast_retention_examples",
            "fast_state_updates",
            "supervised_target_tokens",
            "corrected_two_hop_training_examples",
        )
    }


def _parent_for_phase(config, phase, device):
    reader, world, base, foundation = foundation_inputs(config, device, trainable=True)
    if phase == "common":
        return reader, world, base, foundation, None
    checkpoint = Path(config["common"]["checkpoint"])
    receipt = json.loads(Path(config["common"]["gate_receipt"]).read_text())
    if (
        not receipt["passed"]
        or receipt["checkpoint_sha256"] != sha256(checkpoint)
        or receipt["identity"] != identity(config)
    ):
        raise ValueError("A passing common reader/capability gate must bind this branch parent")
    parent = _load(checkpoint)
    if (
        parent["phase"] != "common"
        or parent["identity"] != identity(config)
        or parent["step"] != config["common"]["steps"]
    ):
        raise ValueError("Wrong complete common-stage parent")
    reader.backbone.load_state_dict(parent["model"], strict=True)
    if model_digest(reader.backbone) != parent["model_sha256"]:
        raise ValueError("Common parent internal model hash differs")
    return reader, world, base, foundation, parent


def train(config, phase, out, device, log=None):
    if phase not in ("common", "single", "stream"):
        raise ValueError("Invalid adaptation phase")
    reader, world, base, foundation, parent = _parent_for_phase(config, phase, device)
    frozen_identity = identity(config)
    directory = Path(out) / "training"
    directory.mkdir(parents=True, exist_ok=False)
    destination = Path(
        config["common"]["checkpoint"]
        if phase == "common"
        else config["models"][phase]["checkpoint"]
    )
    if destination.resolve() != (directory / "parent.pt").resolve():
        raise ValueError("Terminal checkpoint must use this phase's isolated output")
    table, atomic_size = native_table(world, config["phi"])
    stream = MergedBatchStream(len(table[0]), config["seed"] + 3)
    rng = np.random.RandomState(config["seed"] + 2)
    optimizer = optimizer_for(reader, config)
    if parent is not None:
        optimizer.load_state_dict(parent["optimizer"])
        stream.load_state_dict(parent["native_stream"])
        rng = restore_data_rng(parent["data_rng"])
        torch.set_rng_state(parent["torch_rng"])
        if str(device).startswith("cuda"):
            torch.cuda.set_rng_state(parent["cuda_rng"], device)
    steps = config["common"]["steps"] if phase == "common" else config["training"]["steps"]
    counts, trace = _counts(), hashlib.sha256()
    start, seconds = time.monotonic(), 0.0
    compute_start = dict(reader.compute)
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)

    def save(step, readout, terminal=False):
        path = directory / ("parent.pt" if terminal else f"checkpoint-{step:07d}.pt")
        payload = {
            "schema": "ttcd-verified-base-v1",
            "phase": phase,
            "step": step,
            "identity": frozen_identity,
            "foundation": foundation,
            "common_parent_sha256": sha256(config["common"]["checkpoint"])
            if parent is not None
            else None,
            "model": {
                key: value.detach().cpu().clone()
                for key, value in reader.backbone.state_dict().items()
            },
            "model_sha256": model_digest(reader.backbone),
            "optimizer": optimizer.state_dict(),
            "spec": base["spec"],
            "architecture": base["architecture"],
            "native_stream": stream.state_dict(),
            "data_rng": pack_rng(rng),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(device)
            if str(device).startswith("cuda")
            else None,
            "counts": dict(counts),
            "trace_sha256": trace.hexdigest(),
            "readout": readout,
            "training_seconds": seconds,
        }
        temporary = path.with_suffix(".partial")
        torch.save(payload, temporary)
        temporary.replace(path)
        write_json(
            directory / "checkpoint-index.json",
            {"step": step, "path": str(path), "sha256": sha256(path), "terminal": terminal},
        )

    initial_readout = native_readout(reader, config, world, device)
    save(0, initial_readout)
    with (directory / "curves.jsonl").open("x") as curves:
        for step in range(1, steps + 1):
            batch = sample_step(config, world, rng, stream)
            for key, value in sorted(batch.items()):
                trace.update(key.encode())
                trace.update(value.astype("<i8", copy=False).tobytes())
            before = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            total, losses = backward_objective(reader, config, world, table, batch, phase)
            norm = torch.nn.utils.clip_grad_norm_(
                reader.parameters(), config["training"]["gradient_clip"], error_if_nonfinite=True
            )
            optimizer.step()
            if str(device).startswith("cuda"):
                torch.cuda.synchronize(device)
            seconds += time.monotonic() - before
            native = batch["native_indices"]
            n = len(batch["packets"])
            counts["base_parameter_updates"] += 1
            counts["native_atomic_examples"] += int((native < atomic_size).sum())
            counts["native_chain_examples"] += int((native >= atomic_size).sum())
            counts["correction_examples"] += n
            counts["supervised_target_tokens"] += 2 * (len(native) + n)
            if phase != "common":
                counts["fast_atomic_examples"] += n
                counts["fast_retention_examples"] += n
                counts["fast_state_updates"] += n * config["writer"]["steps"]
                counts["supervised_target_tokens"] += 4 * n
            record = {
                "step": step,
                "phase": phase,
                "loss": total,
                "losses": losses,
                "gradient_norm": float(norm),
                "lr": config["training"]["learning_rate"],
                "counts": dict(counts),
                "trace_sha256": trace.hexdigest(),
                "training_seconds": seconds,
                "wall_seconds": time.monotonic() - start,
                "compute": {
                    key: value - compute_start[key] for key, value in reader.compute.items()
                },
            }
            if step % config["training"]["validation_every"] == 0 or step == steps:
                readout = native_readout(reader, config, world, device)
                record["native_metrics"] = {key: value["metrics"] for key, value in readout.items()}
                save(step, readout, terminal=step == steps)
            curves.write(json.dumps(record) + "\n")
            curves.flush()
            write_json(directory / "progress.json", record)
            if log:
                log(step, record)
    if identity(config) != frozen_identity:
        raise ValueError("Adaptation code/config/inputs changed during training")
    result = {
        "status": "training_complete_pending_independent_reload",
        "phase": phase,
        "step": steps,
        "identity": frozen_identity,
        "foundation": foundation,
        "checkpoint_sha256": sha256(destination),
        "model_sha256": model_digest(reader.backbone),
        "counts": counts,
        "trace_sha256": trace.hexdigest(),
        "training_seconds": seconds,
        "wall_seconds": time.monotonic() - start,
        "compute": dict(reader.compute),
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device)
        if str(device).startswith("cuda")
        else None,
    }
    write_json(directory / "summary.json", result)
    return result


def load_reader(config, phase, device, *, require_audit=True):
    world = validate_inputs(config)
    path = Path(
        config["common"]["checkpoint"]
        if phase == "common"
        else config["models"][phase]["checkpoint"]
    )
    saved = _load(path)
    expected_step = config["common"]["steps"] if phase == "common" else config["training"]["steps"]
    if (
        saved["schema"] != "ttcd-verified-base-v1"
        or saved["phase"] != phase
        or saved["identity"] != identity(config)
        or saved["step"] != expected_step
    ):
        raise ValueError("Adaptation endpoint identity/phase/step differs")
    reader = TTCDReader(
        _backbone(config, saved, world, device), native_causal_attention=True
    ).eval()
    if require_audit:
        receipt = json.loads((path.parent / "audit.json").read_text())
        if (
            not receipt["passed"]
            or receipt["checkpoint_sha256"] != sha256(path)
            or receipt["identity"] != identity(config)
        ):
            raise ValueError("Fresh training reload audit must bind this endpoint")
    return reader, world


def audit_training(config, phase, out, device):
    reader, world = load_reader(config, phase, device, require_audit=False)
    directory = Path(out) / "training"
    summary = json.loads((directory / "summary.json").read_text())
    if summary["checkpoint_sha256"] != sha256(directory / "parent.pt") or summary[
        "identity"
    ] != identity(config):
        raise ValueError("Training completion identity differs")
    checks = []
    for path in sorted(directory.glob("checkpoint-*.pt")) + [directory / "parent.pt"]:
        saved = _load(path)
        if saved["identity"] != identity(config) or saved["phase"] != phase:
            raise ValueError("Saved checkpoint has another adaptation identity")
        reader.backbone.load_state_dict(saved["model"], strict=True)
        if model_digest(reader.backbone) != saved["model_sha256"]:
            raise ValueError("Saved checkpoint internal model hash differs")
        observed = native_readout(reader, config, world, device)
        compare_replay(saved["readout"], observed)
        checks.append(
            {"step": saved["step"], "sha256": sha256(path), "all_native_tokens_exact": True}
        )
    result = {
        "passed": True,
        "identity": identity(config),
        "phase": phase,
        "checkpoint_sha256": sha256(directory / "parent.pt"),
        "checkpoints": checks,
        "formal_D_used": False,
        "utc": utc(),
    }
    write_json(directory / "audit.json", result)
    return result


def common_gate(config, out, device):
    reader, world = load_reader(config, "common", device)
    native = native_readout(reader, config, world, device)
    parent, _, _, _ = foundation_inputs(config, device)
    before = native_readout(parent, config, world, device)
    settings = config["capability_gate"]
    checks = {}
    for name in PANELS:
        score, original = (
            native[name]["metrics"]["old_accuracy"],
            before[name]["metrics"]["old_accuracy"],
        )
        if "minimum_accuracy" in settings:
            floor = settings["minimum_accuracy"][name]
        else:
            floor = (
                settings["minimum_atomic_and_ii"]
                if name in ("atomic_id", "atomic_ood", "train_ii", "test_ii")
                else settings["minimum_io_oi_oo"]
            )
        maximum_drop = settings.get("maximum_reference_drop", settings.get("maximum_drop"))
        checks[name] = {
            "accuracy": score,
            "foundation_accuracy": original,
            "minimum": floor,
            "drop": original - score,
            "passed": score >= floor and original - score <= maximum_drop + 1e-12,
        }
    records = []
    for case in case_bank(config):
        if case["split"] != "calibration":
            continue
        atom = world.atoms[case["atom"]]
        record = open_query(
            reader,
            None,
            packet_for(config, world, case),
            config["writer"]["teacher_window"],
            int(atom[-1]),
            case["new_tail"],
            world.metadata,
            device,
        )
        records.append({"case": case, "readout": record})
    groups = {}
    for group in ("id", "ood"):
        values = [
            record["readout"]["new_answer_correct"]
            for record in records
            if record["case"]["group"] == group
        ]
        groups[group] = {
            "n": len(values),
            "accuracy": float(np.mean(values)) if values else None,
            "passed": len(values) >= config["reading_gate"]["minimum_cases_per_group"]
            and bool(values)
            and float(np.mean(values)) >= config["reading_gate"]["minimum_accuracy"],
        }
    passed = all(value["passed"] for value in checks.values()) and all(
        value["passed"] for value in groups.values()
    )
    result = {
        "passed": passed,
        "identity": identity(config),
        "checkpoint_sha256": sha256(config["common"]["checkpoint"]),
        "native_checks": checks,
        "reading_groups": groups,
        "reading_records": records,
        "native_readout": native,
        "foundation_readout": before,
        "formal_D_used": False,
        "status": "ready_to_branch" if passed else "common_gate_failed_no_branch",
        "utc": utc(),
    }
    write_json(Path(out) / "gate.json", result)
    return result


def evaluation_identity(config, name, panel):
    checkpoint = Path(config["models"][name]["checkpoint"])
    return {
        "identity": identity(config),
        "name": name,
        "panel_sha256": hashlib.sha256(json.dumps(panel, sort_keys=True).encode()).hexdigest(),
        "checkpoint_sha256": sha256(checkpoint),
        "training_audit_sha256": sha256(checkpoint.parent / "audit.json"),
        "common_gate_sha256": sha256(config["common"]["gate_receipt"]),
    }
