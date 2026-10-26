"""From-scratch symbolic Loop/DiscoLoop training with atomic TTCD episodes.

This is an equation-(6) GPT adaptation, not the original IP-TTCD-v9 model.
Outer CE trains the backbone; the local write stops only the teacher target.
Each batch item has its own fast matrices, discarded after the outer step.
Updated composite answers are never inputs or labels of this training module.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from transformers import GPT2Config

from .edit_scope_development import evaluate_queries, prediction_metrics
from .grok_depth import utc, write_json
from .grokking_reproduction import audit_graph, build_graph, encode_rows, model_digest
from .path_overlap import World, sha256
from .supervised_alignment import SupervisedAlignmentGPT
from .ttcd_editing import TTCDReader, build_case_bank, correction_packet, tensor_digest

CODE_FILES = (
    "src/llm_memory_editability/ttcd_scratch.py",
    "src/llm_memory_editability/ttcd_editing.py",
    "src/llm_memory_editability/ttcd_experiment.py",
    "src/llm_memory_editability/supervised_alignment.py",
    "src/llm_memory_editability/self_decode_alignment.py",
    "src/llm_memory_editability/grokking_reproduction.py",
    "src/llm_memory_editability/path_overlap.py",
    "src/llm_memory_editability/edit_scope_development.py",
    "src/llm_memory_editability/grok_depth.py",
    "scripts/run_ttcd_second_hop.py",
    "scripts/prepare_ttcd_scratch.py",
)


def data_contract(config):
    return {
        **{k: config[k] for k in ("seed", "phi", "world", "model", "cases", "writer")},
        "validation_panel_size": config["training"]["validation_panel_size"],
    }


def prepare_scratch_data(config):
    """Generate a new, explicitly smaller world using the existing Wang generator."""
    data = Path(config["data_dir"])
    complete = data / "complete.json"
    if complete.exists():
        return validate_scratch_inputs(config).metadata
    if data.exists() and any(data.iterdir()):
        raise FileExistsError("Incomplete data directory; preserve it and use an amended path")
    data.mkdir(parents=True, exist_ok=True)
    spec = config["world"]
    if not 2 <= spec["degree"] <= spec["relations"] or spec["entities"] < 4:
        raise ValueError("Invalid scratch graph dimensions")
    graph = build_graph(**spec)
    audit_graph(graph, spec["degree"])
    n_support = round(config["phi"] * (~graph["ood"]).sum())
    if not 1 <= n_support < len(graph["train_order"]):
        raise ValueError("phi must leave both nonempty training and held-out combinations")
    np.savez_compressed(data / "world.npz", **graph)
    rng = np.random.RandomState(config["seed"] + 1)

    def sample(ids):
        ids = np.asarray(ids)
        return rng.permutation(ids)[: config["training"]["validation_panel_size"]]

    panels = {
        "atomic_id": graph["atoms"][sample(np.flatnonzero(~graph["ood"]))],
        "atomic_ood": graph["atoms"][sample(np.flatnonzero(graph["ood"]))],
        "train_ii": graph["chains"][sample(graph["train_order"][:n_support])],
        "test_ii": graph["chains"][sample(graph["test_ii"])],
        **{
            f"test_{name}": graph["chains"][sample(np.flatnonzero(graph["kind"] == kind))]
            for name, kind in (("io", 1), ("oi", 2), ("oo", 3))
        },
    }
    np.savez_compressed(data / "panels.npz", **panels)
    # PAD=0, EOS=1; entities and relations each occupy disjoint contiguous ranges.
    metadata = {
        "schema": "ttcd-scratch-world-v1",
        "data_contract": data_contract(config),
        "world": spec["seed"],
        "entities": spec["entities"],
        "relations": spec["relations"],
        "degree": spec["degree"],
        "atomic_id": int((~graph["ood"]).sum()),
        "atomic_ood": int(graph["ood"].sum()),
        "entity_offset": 2,
        "relation_offset": 2 + spec["entities"],
        "vocab_size": 2 + spec["entities"] + spec["relations"],
        "end_marker": 1,
        "pad_token": 0,
        "world_sha256": sha256(data / "world.npz"),
        "panels_sha256": sha256(data / "panels.npz"),
        "created_utc": utc(),
        "interpretation": "New smaller development world; not the original paper reproduction",
    }
    write_json(complete, metadata)
    world = World(data)
    bank = build_case_bank(world, phi=config["phi"], **config["cases"])
    reserved = {c["atom"] for c in bank}
    train_atoms = sorted(set(range(len(world.atoms))) - reserved)
    if not train_atoms:
        raise ValueError("No correction-training facts remain")
    write_json(
        data / "training-split.json",
        {
            "train_edit_atoms": train_atoms,
            "case_bank": bank,
            "rule": "Old atoms may be learned for all facts; correction episodes exclude "
            "every calibration/evaluation target fact. Only ordinary old II chains are trained.",
        },
    )
    m = config["model"]
    hf = GPT2Config(
        vocab_size=metadata["vocab_size"],
        n_positions=m["positions"],
        n_embd=m["width"],
        n_head=m["heads"],
        n_layer=m["unique_layers"],
        n_inner=m["intermediate"],
        embd_pdrop=0,
        resid_pdrop=0,
        attn_pdrop=0,
        use_cache=False,
        bos_token_id=metadata["end_marker"],
        eos_token_id=metadata["end_marker"],
        pad_token_id=metadata["pad_token"],
    )
    Path(config["model_config"]).parent.mkdir(parents=True, exist_ok=True)
    hf.to_json_file(config["model_config"])
    metadata.update(
        split_sha256=sha256(data / "training-split.json"),
        model_config_sha256=sha256(config["model_config"]),
    )
    write_json(complete, metadata)
    validate_scratch_inputs(config)
    return metadata


def validate_scratch_inputs(config):
    if "scratch" not in config or config["scratch"]["version"] != 1:
        raise ValueError("Expected an explicit from-scratch contract")
    t, m = config["training"], config["model"]
    integers = (
        "warmup_steps",
        "adaptation_steps",
        "batch_size",
        "validation_every",
        "validation_panel_size",
    )
    if any(isinstance(t[k], bool) or not isinstance(t[k], int) or t[k] < 1 for k in integers):
        raise ValueError("Training budgets must be positive integers")
    if (
        not isinstance(t["lr_warmup_steps"], int)
        or isinstance(t["lr_warmup_steps"], bool)
        or t["lr_warmup_steps"] < 0
    ):
        raise ValueError("LR warmup must be a nonnegative integer")
    w = config["writer"]
    if (
        not isinstance(w["steps"], int)
        or isinstance(w["steps"], bool)
        or w["steps"] < 1
        or not math.isfinite(w["learning_rate"])
        or w["learning_rate"] <= 0
    ):
        raise ValueError("Scratch fast-write budget and rate must be positive")
    if m["width"] % m["heads"] or min(m.values()) < 1:
        raise ValueError("Invalid model dimensions")
    for key in ("learning_rate", "minimum_learning_rate", "gradient_clip"):
        if not math.isfinite(t[key]) or t[key] <= 0:
            raise ValueError("Invalid optimization settings")
    if t["minimum_learning_rate"] > t["learning_rate"]:
        raise ValueError("Minimum LR exceeds peak LR")
    if not math.isfinite(t["weight_decay"]) or t["weight_decay"] < 0:
        raise ValueError("Invalid weight decay")
    if any(not math.isfinite(v) or v <= 0 for v in t["loss_weights"].values()):
        raise ValueError("All declared objective weights must be positive")
    if set(t["loss_weights"]) != {"atoms", "chains", "reader", "fast_atomic", "fast_retention"}:
        raise ValueError("Unknown or missing objective; edited composite supervision is prohibited")
    world = World(config["data_dir"])
    if world.metadata["data_contract"] != data_contract(config):
        raise ValueError("Prepared data contract changed; create a separate amended batch")
    for stem in ("world", "panels"):
        if sha256(Path(config["data_dir"]) / f"{stem}.npz") != world.metadata[f"{stem}_sha256"]:
            raise ValueError("Prepared graph/panel hash mismatch")
    if sha256(config["model_config"]) != world.metadata["model_config_sha256"]:
        raise ValueError("Prepared model config hash mismatch")
    split_path = Path(config["data_dir"]) / "training-split.json"
    if sha256(split_path) != world.metadata["split_sha256"]:
        raise ValueError("Prepared split hash mismatch")
    split = json.loads(split_path.read_text())
    bank = build_case_bank(world, phi=config["phi"], **config["cases"])
    if split["case_bank"] != bank:
        raise ValueError("Frozen case bank changed")
    reserved = {c["atom"] for c in bank}
    expected = sorted(set(range(len(world.atoms))) - reserved)
    if split["train_edit_atoms"] != expected or not expected:
        raise ValueError("Correction training/test facts overlap or training pool is empty")
    for case in bank:
        if np.any(world.support(config["phi"])[case["D"]]):
            raise ValueError("An edited test composition occurs in ordinary training")
        packet = correction_packet(
            world.atoms[case["atom"]],
            case["new_tail"],
            world.atoms[case["atom"], :2],
            world.metadata,
            **{k: config["writer"][k] for k in ("gap", "teacher_window", "student_window")},
        )
        if len(packet) + 1 > m["positions"]:
            raise ValueError("Model positions cannot hold reader answer/EOS supervision")
    total = t["warmup_steps"] + t["adaptation_steps"]
    for parent in config["models"].values():
        if parent["step"] != total:
            raise ValueError("Evaluation parent step must equal the fixed scratch training budget")
    return world


def training_identity(config):
    root = Path(__file__).resolve().parents[2]
    paths = [Path(config["model_config"])]
    paths += [
        Path(config["data_dir"]) / n
        for n in ("world.npz", "panels.npz", "complete.json", "training-split.json")
    ]
    code_files = CODE_FILES
    if "branch" in config:
        code_files = (*code_files, "src/llm_memory_editability/ttcd_branch.py")
    return {
        "config": config,
        "inputs": {str(p): sha256(p) for p in paths},
        "code": {p: sha256(root / p) for p in code_files},
    }


def new_scratch_reader(config, name, world, device):
    # Identical tensors between architectures, with independent random physical layers.
    torch.manual_seed(config["seed"])
    hf = GPT2Config.from_json_file(config["model_config"])
    parent = SupervisedAlignmentGPT(
        hf, repeats=config["model"]["repeats"], discoloop=config["models"][name]["discoloop"]
    ).to(device)
    return TTCDReader(parent, trainable_backbone=True).eval()


def session_write(reader, packets, fast, writer):
    """Differentiable outer computation; one equation-(6) update per example/layer.

    One write position means N=1 for each session. No cross-example average is
    allowed. All layers commit together, and only round zero supplies the write.
    The teacher target is stopped; student features, student outputs, prior fast
    states and subsequent closed reads remain connected to the outer CE graph.
    """
    if not reader.trainable_backbone:
        raise ValueError("Outer-training writes require a trainable scratch reader")
    if not math.isfinite(writer["learning_rate"]) or writer["learning_rate"] < 0:
        raise ValueError("Invalid fast learning rate")
    if fast is None:
        fast = {
            k: v.unsqueeze(0).expand(len(packets), -1, -1).clone()
            for k, v in reader.empty_fast().items()
        }
    if any(v.ndim != 3 or v.shape[0] != len(packets) for v in fast.values()):
        raise ValueError("Each correction-training example needs its own fast matrices")
    capture = {
        "student_window": writer["student_window"],
        "positions": packets.new_full((len(packets),), packets.shape[1] - 1),
        "outer_grad": True,
    }
    reader(
        packets,
        capture["positions"][:, None],
        fast=fast,
        window=writer["teacher_window"],
        capture=capture,
    )
    updated = {}
    for layer, (z, teacher, student) in capture["features"].items():
        residual = teacher.detach().float() - student.float()
        updated[layer] = fast[layer] + writer["learning_rate"] * (
            z.float().unsqueeze(-1) * residual.unsqueeze(-2)
        )
        cost = 2 * len(z) * z.shape[-1] * residual.shape[-1]
        reader.compute["estimated_dense_forward_matmul_flops"] += cost
        if torch.is_grad_enabled():
            reader.compute["estimated_backward_matmul_flops_proxy"] += 2 * cost
    reader.validate_fast(updated)
    return updated


def sample_batch(config, world, rng):
    """Only old atomic/II labels and corrected ATOMIC labels are constructed."""
    n = config["training"]["batch_size"]
    split = json.loads((Path(config["data_dir"]) / "training-split.json").read_text())
    edit_ids = rng.choice(split["train_edit_atoms"], n)
    atoms = world.atoms[edit_ids]
    tails = rng.randint(world.entities - 1, size=n)
    tails += tails >= atoms[:, 2]
    support = np.flatnonzero(world.support(config["phi"]))
    old_atoms = world.atoms[rng.randint(len(world.atoms), size=n)]
    old_chains = world.chains[rng.choice(support, n)]
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
    # Retention labels under edited fast weights exclude precisely that session's target key.
    control_ids = rng.randint(len(world.atoms) - 1, size=n)
    control_ids += control_ids >= edit_ids
    return {
        "edit_ids": edit_ids,
        "old_atoms": old_atoms,
        "old_chains": old_chains,
        "corrected_atoms": corrected,
        "retention_atoms": world.atoms[control_ids],
        "packets": np.asarray(packets, dtype=np.int64),
    }


def row_loss(reader, rows, metadata, *, fast=None):
    x, positions, labels = encode_rows(rows, metadata)
    device = next(reader.parameters()).device
    x, positions, labels = (torch.as_tensor(v, device=device) for v in (x, positions, labels))
    logits = reader(x, positions, fast=fast)
    return F.cross_entropy(logits.flatten(0, 1), labels.flatten())


def outer_objective(reader, config, world, batch, *, adaptation):
    losses = {
        "atoms": row_loss(reader, batch["old_atoms"], world.metadata),
        "chains": row_loss(reader, batch["old_chains"], world.metadata),
    }
    device = next(reader.parameters()).device
    packets = torch.as_tensor(batch["packets"], device=device)
    new = torch.as_tensor(batch["corrected_atoms"][:, 2] + world.eo, device=device)
    open_tokens = torch.cat((packets, new[:, None]), dim=1)
    positions = packets.new_tensor([packets.shape[1] - 1, packets.shape[1]])
    positions = positions[None].expand(len(packets), -1)
    labels = torch.stack((new, torch.full_like(new, world.metadata["end_marker"])), dim=1)
    logits = reader(open_tokens, positions, window=config["writer"]["teacher_window"])
    losses["reader"] = F.cross_entropy(logits.flatten(0, 1), labels.flatten())
    if adaptation:
        fast = None
        for _ in range(config["writer"]["steps"]):
            fast = session_write(reader, packets, fast, config["writer"])
        losses["fast_atomic"] = row_loss(
            reader, batch["corrected_atoms"], world.metadata, fast=fast
        )
        losses["fast_retention"] = row_loss(
            reader, batch["retention_atoms"], world.metadata, fast=fast
        )
    loss = sum(config["training"]["loss_weights"][k] * v for k, v in losses.items())
    return loss, {k: float(v.detach()) for k, v in losses.items()}


@torch.no_grad()
def training_readout(reader, config, world, device):
    """Old-world panels + calibration ATOMS; no updated two-hop evaluation here."""
    from .ttcd_experiment import Session, open_query

    result = {"panels": {}, "calibration_atoms": []}
    for key, rows in world.panels.items():
        if not len(rows):
            continue
        pred = evaluate_queries(
            reader, rows, world.metadata, torch.device(device), precision="fp32"
        )
        result["panels"][key] = {
            "metrics": prediction_metrics(pred, world.metadata),
            "predictions": {k: v.tolist() for k, v in pred.items()},
        }
    split = json.loads((Path(config["data_dir"]) / "training-split.json").read_text())
    for case in split["case_bank"]:
        if case["split"] != "calibration":
            continue
        atom = world.atoms[case["atom"]]
        packet = correction_packet(
            atom,
            case["new_tail"],
            atom[:2],
            world.metadata,
            **{k: config["writer"][k] for k in ("gap", "teacher_window", "student_window")},
        )
        fast = reader.empty_fast()
        open_pred = open_query(
            reader,
            fast,
            packet,
            config["writer"]["teacher_window"],
            int(atom[2]),
            case["new_tail"],
            world.metadata,
            device,
        )
        tokens = torch.tensor([packet], device=device)
        # Inference write deliberately uses the existing detached, frozen-style rule.
        for _ in range(config["writer"]["steps"]):
            fast, _ = reader.write(
                tokens,
                fast,
                **{
                    k: config["writer"][k]
                    for k in ("teacher_window", "student_window", "learning_rate")
                },
            )
        pred = evaluate_queries(
            Session(reader, fast).eval(),
            atom[None],
            world.metadata,
            torch.device(device),
            new_answers=[case["new_tail"]],
            precision="fp32",
        )
        result["calibration_atoms"].append(
            {
                "case_id": case["case_id"],
                "group": case["group"],
                "open": open_pred,
                "closed": {k: v.tolist() for k, v in pred.items()},
                "metrics": prediction_metrics(pred, world.metadata),
            }
        )
    return result


def train_scratch(
    config, name, out, device, log=None, *, _branch_payload=None, _branch_provenance=None
):
    """Fixed-budget run; no outcome-dependent extension, retry, or checkpoint selection."""
    if ("branch" in config) != (_branch_payload is not None):
        raise ValueError("Branch training requires the validated train_branch entrypoint")
    if (_branch_payload is None) != (_branch_provenance is None):
        raise ValueError("Branch checkpoint and provenance must be supplied together")
    world = validate_scratch_inputs(config)
    identity = training_identity(config)
    directory = Path(out) / "training"
    if directory.exists():
        raise FileExistsError(
            "Training output already exists; preserve it, do not restart this batch"
        )
    directory.mkdir(parents=True)
    parent_path = Path(config["models"][name]["checkpoint"])
    if parent_path.resolve() != (directory / "parent.pt").resolve():
        raise ValueError("Scratch parent checkpoint must live in this model's training output")
    reader = new_scratch_reader(config, name, world, device)
    initial = tensor_digest(reader.backbone.state_dict())
    optimizer = torch.optim.AdamW(
        reader.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    rng = np.random.RandomState(config["seed"] + 2)
    t = config["training"]
    total = t["warmup_steps"] + t["adaptation_steps"]
    counts = {
        "base_parameter_updates": 0,
        "old_atomic_examples": 0,
        "old_chain_examples": 0,
        "correction_examples": 0,
        "fast_state_updates": 0,
        "adaptation_episodes": 0,
        "supervised_target_tokens": 0,
        "corrected_two_hop_training_examples": 0,
    }
    started = time.monotonic()
    training_seconds = 0.0
    training_compute = {k: 0 for k in reader.compute}
    start_step, inherited_wall_seconds = 0, 0.0
    inherited_cost = None
    if _branch_payload is not None:
        parent = _branch_payload
        if initial != parent["initial_tensor_digest"]:
            raise ValueError("Branch architecture/initialization differs from the saved parent")
        reader.backbone.load_state_dict(parent["model"], strict=True)
        optimizer.load_state_dict(parent["optimizer"])
        state = parent["data_rng"]
        rng.set_state(
            (
                state["algorithm"],
                np.asarray(state["keys"], dtype=np.uint32),
                state["position"],
                state["has_gauss"],
                state["cached_gaussian"],
            )
        )
        torch.set_rng_state(parent["torch_rng"].cpu())
        if str(device).startswith("cuda"):
            if len(parent["cuda_rng"]) != torch.cuda.device_count():
                raise ValueError("Saved CUDA RNG coverage differs from the isolated device layout")
            torch.cuda.set_rng_state_all([value.cpu() for value in parent["cuda_rng"]])
        start_step = parent["step"]
        counts = dict(parent["counts"])
        training_seconds = parent["training_seconds"]
        inherited_wall_seconds = parent["wall_seconds"]
        reader.compute.update(parent["compute"])
        training_compute = dict(parent["training_compute"])
        inherited_cost = {
            "counts": dict(counts),
            "training_seconds": training_seconds,
            "wall_seconds": inherited_wall_seconds,
            "training_compute": dict(training_compute),
            "compute_including_validation": dict(reader.compute),
        }
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)

    def checkpoint(step, readout=None, terminal=False):
        state = rng.get_state()
        payload = {
            "schema": "ttcd-scratch-parent-v1",
            "step": step,
            "model": {k: v.detach().cpu().clone() for k, v in reader.backbone.state_dict().items()},
            "optimizer": optimizer.state_dict(),
            "model_sha256": model_digest(reader.backbone),
            "initial_tensor_digest": initial,
            "architecture": {
                "discoloop": config["models"][name]["discoloop"],
                "first_round_kv": False,
            },
            "spec": {
                "unique_layers": config["model"]["unique_layers"],
                "repeats": config["model"]["repeats"],
                "phi": config["phi"],
                "name": name,
            },
            "training_identity": identity,
            "counts": dict(counts),
            "training_seconds": training_seconds,
            "wall_seconds": inherited_wall_seconds + time.monotonic() - started,
            "compute": dict(reader.compute),
            "training_compute": dict(training_compute),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if str(device).startswith("cuda") else [],
            "data_rng": {
                "algorithm": state[0],
                "keys": state[1].tolist(),
                "position": state[2],
                "has_gauss": state[3],
                "cached_gaussian": state[4],
            },
            "readout": readout,
        }
        if _branch_provenance is not None:
            payload["branch"] = _branch_provenance
        path = directory / ("parent.pt" if terminal else f"checkpoint-{step:07d}.pt")
        temporary = path.with_suffix(".partial")
        torch.save(payload, temporary)
        temporary.replace(path)
        write_json(
            directory / "checkpoint-index.json",
            {
                "step": step,
                "path": str(path),
                "sha256": sha256(path),
                "terminal": terminal,
            },
        )

    checkpoint(start_step, _branch_payload["readout"] if _branch_payload is not None else None)
    with (directory / "curves.jsonl").open("x") as curves:
        for step in range(start_step + 1, total + 1):
            before = time.monotonic()
            compute_before = dict(reader.compute)
            adaptation = _branch_payload is None and step > t["warmup_steps"]
            fraction = (step - 1) / max(1, total - 1)
            rate = t["minimum_learning_rate"] + 0.5 * (
                t["learning_rate"] - t["minimum_learning_rate"]
            ) * (1 + math.cos(math.pi * fraction))
            rate *= min(1.0, step / max(1, t["lr_warmup_steps"]))
            for group in optimizer.param_groups:
                group["lr"] = rate
            optimizer.zero_grad(set_to_none=True)
            batch = sample_batch(config, world, rng)
            loss, losses = outer_objective(reader, config, world, batch, adaptation=adaptation)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite outer loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                reader.parameters(), t["gradient_clip"], error_if_nonfinite=True
            )
            optimizer.step()
            if str(device).startswith("cuda"):
                torch.cuda.synchronize(device)
            training_seconds += time.monotonic() - before
            for key in training_compute:
                training_compute[key] += reader.compute[key] - compute_before[key]
            n = t["batch_size"]
            counts["base_parameter_updates"] += 1
            counts["old_atomic_examples"] += n
            counts["old_chain_examples"] += n
            counts["correction_examples"] += n
            counts["fast_state_updates"] += n * config["writer"]["steps"] if adaptation else 0
            counts["adaptation_episodes"] += n if adaptation else 0
            counts["supervised_target_tokens"] += n * (10 if adaptation else 6)
            record = {
                "step": step,
                "phase": (
                    "reader_control"
                    if _branch_payload is not None
                    else ("adaptation" if adaptation else "warmup")
                ),
                "lr": rate,
                "loss": float(loss.detach()),
                "losses": losses,
                "gradient_norm": float(norm),
                "training_seconds": training_seconds,
                "wall_seconds": inherited_wall_seconds + time.monotonic() - started,
                "counts": dict(counts),
                "compute": dict(training_compute),
            }
            if step % t["validation_every"] == 0 or step in (t["warmup_steps"], total):
                readout = training_readout(reader, config, world, device)
                write_json(directory / f"validation-{step:07d}.json", readout)
                record["validation"] = {k: v["metrics"] for k, v in readout["panels"].items()}
                record["calibration"] = {
                    group: {
                        "open_new_accuracy": float(
                            np.mean(
                                [
                                    c["open"]["new_correct"]
                                    for c in readout["calibration_atoms"]
                                    if c["group"] == group
                                ]
                            )
                        ),
                        "closed_new_accuracy": float(
                            np.mean(
                                [
                                    c["metrics"]["new_accuracy"]
                                    for c in readout["calibration_atoms"]
                                    if c["group"] == group
                                ]
                            )
                        ),
                    }
                    for group in ("id", "ood")
                }
                checkpoint(step, readout, terminal=step == total)
            curves.write(json.dumps(record) + "\n")
            curves.flush()
            write_json(directory / "progress.json", record)
            if log:
                log(step, record)
    if identity != training_identity(config):
        raise ValueError("Training inputs/code/config changed during execution")
    result = {
        "status": "training_complete_pending_independent_audit",
        "step": total,
        "parent": str(parent_path),
        "parent_sha256": sha256(parent_path),
        "initial_tensor_digest": initial,
        "model_sha256": model_digest(reader.backbone),
        "parameters": sum(p.numel() for p in reader.parameters()),
        "counts": counts,
        "training_seconds": training_seconds,
        "wall_seconds": inherited_wall_seconds + time.monotonic() - started,
        "compute_including_validation": dict(reader.compute),
        "training_compute": training_compute,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device)
        if str(device).startswith("cuda")
        else None,
        "training_identity": identity,
        "claim": "Execution only; no claim that reading, editing or two-hop propagation succeeded",
    }
    if inherited_cost is not None:
        result.update(
            branch=_branch_provenance,
            inherited_cost=inherited_cost,
            additional_counts={k: v - inherited_cost["counts"][k] for k, v in counts.items()},
            branch_training_seconds=training_seconds - inherited_cost["training_seconds"],
            branch_wall_seconds=time.monotonic() - started,
            branch_training_compute={
                k: v - inherited_cost["training_compute"][k] for k, v in training_compute.items()
            },
            branch_compute_including_validation={
                k: v - inherited_cost["compute_including_validation"][k]
                for k, v in reader.compute.items()
            },
            peak_cuda_scope="This new branch process only; inherited parent peak is not added",
        )
    write_json(directory / "summary.json", result)
    return result


def audit_scratch_training(config, name, out, device):
    """Fresh-process parent reload and full validation replay, before any editing."""
    from .ttcd_experiment import load_inputs

    world = validate_scratch_inputs(config)
    reader, _ = load_inputs(config, name, device, require_training_audit=False)
    checkpoint = torch.load(
        config["models"][name]["checkpoint"], map_location="cpu", weights_only=True
    )
    summary = json.loads((Path(out) / "training/summary.json").read_text())
    if (
        summary["parent_sha256"] != sha256(config["models"][name]["checkpoint"])
        or summary["model_sha256"] != checkpoint["model_sha256"]
        or summary["step"] != checkpoint["step"]
    ):
        raise ValueError("Terminal checkpoint differs from the training completion record")
    if checkpoint["training_identity"] != training_identity(config):
        raise ValueError("Scratch training provenance differs from current inputs/code")
    actual = training_readout(reader, config, world, device)

    # Token/rank exactness, with a small fp32 tolerance for saved margins across devices.
    def compare(expected, observed):
        if isinstance(expected, dict):
            if set(expected) != set(observed):
                raise ValueError("Training replay keys differ")
            for k in expected:
                compare(expected[k], observed[k])
        elif isinstance(expected, list):
            if len(expected) != len(observed):
                raise ValueError("Training replay coverage differs")
            for a, b in zip(expected, observed, strict=True):
                compare(a, b)
        elif isinstance(expected, float):
            if not math.isclose(expected, observed, rel_tol=1e-5, abs_tol=1e-5):
                raise ValueError("Training replay numeric value differs")
        elif expected != observed:
            raise ValueError("Training replay token/rank/value differs")

    compare(checkpoint["readout"], actual)
    result = {
        "passed": True,
        "status": "training_independently_reloaded",
        "step": checkpoint["step"],
        "parent_sha256": sha256(config["models"][name]["checkpoint"]),
        "training_identity": training_identity(config),
        "readout": actual,
    }
    write_json(Path(out) / "training/audit.json", result)
    return result
