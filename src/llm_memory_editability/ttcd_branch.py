"""Pinned warmup-state branches that omit both fast-weight training losses.

The original frozen batch is read-only. A branch retains its own copy of the
published checkpoint and continues the original AdamW/data/global-LR state.
It adds no new graph, initialization, checkpoint selection, or edited two-hop
training. The same ordinary scratch loop supplies its three retained losses.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from pathlib import Path

import torch

from .grok_depth import write_json
from .path_overlap import sha256
from .ttcd_scratch import CODE_FILES, train_scratch, training_identity, validate_scratch_inputs

OBJECTIVE = "old_atoms_old_chains_context_reader_only"


def validate_branch_parent(config, name, checkpoint_path):
    """Bind a complete warmup state to the original frozen code/data/recipe."""
    branch = config["branch"]
    if branch["objective"] != OBJECTIVE:
        raise ValueError("Unknown branch objective")
    source = Path(branch["parent_source_dir"])
    manifest_path = Path(branch["parent_manifest_path"])
    if sha256(manifest_path) != branch["parent_manifest_sha256"]:
        raise ValueError("Original source manifest hash differs from the pinned parent")
    manifest = json.loads(manifest_path.read_text())
    for relative, expected in manifest["files"].items():
        path = source / relative
        if not path.resolve().is_relative_to(source.resolve()) or sha256(path) != expected:
            raise ValueError(f"Original frozen source changed: {relative}")
    original_config = source / "configs/experiment.json"
    if (
        sha256(original_config) != branch["parent_config_sha256"]
        or manifest["config_sha256"] != branch["parent_config_sha256"]
    ):
        raise ValueError("Original configuration differs from the pinned parent")
    parent_config = json.loads(original_config.read_text())
    if parent_config["batch"] != branch["parent_batch"] or "branch" in parent_config:
        raise ValueError("Expected the original scratch batch, not another branch")
    # Only output/provenance/resource fields may differ. All science settings,
    # including the original 30k LR horizon, stay identical to the parent.
    allowed = {"batch", "authorization", "phase", "scope", "runtime", "models", "branch"}
    if {k: v for k, v in parent_config.items() if k not in allowed} != {
        k: v for k, v in config.items() if k not in allowed
    }:
        raise ValueError(
            "Branch changed original data, model, objective weights or training recipe"
        )
    if set(config["models"]) != set(parent_config["models"]):
        raise ValueError("Branch model matrix differs from the parent")
    for model, spec in parent_config["models"].items():
        if {k: v for k, v in spec.items() if k != "checkpoint"} != {
            k: v for k, v in config["models"][model].items() if k != "checkpoint"
        }:
            raise ValueError("Branch architecture or terminal step differs from the parent")
    step, steps = branch["checkpoint_step"], branch["steps"]
    if (
        isinstance(step, bool)
        or isinstance(steps, bool)
        or not isinstance(step, int)
        or not isinstance(steps, int)
        or step != parent_config["training"]["warmup_steps"]
        or steps != parent_config["training"]["adaptation_steps"]
        or steps < 1
    ):
        raise ValueError("Branch must continue the complete warmup through the fixed final step")
    path = Path(checkpoint_path)
    expected = Path(parent_config["models"][name]["checkpoint"]).with_name(
        f"checkpoint-{step:07d}.pt"
    )
    if (
        path.resolve() != expected.resolve()
        or path.resolve() != Path(branch["checkpoints"][name]).resolve()
    ):
        raise ValueError("Branch checkpoint is not this model's original warmup endpoint")
    validate_scratch_inputs(parent_config)
    validate_scratch_inputs(config)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    identity = payload["training_identity"]
    if (
        payload["schema"] != "ttcd-scratch-parent-v1"
        or payload["step"] != step
        or payload.get("branch") is not None
        or identity["config"] != parent_config
        or identity["inputs"] != training_identity(parent_config)["inputs"]
        or set(identity["code"]) != set(CODE_FILES)
        or any(manifest["files"].get(k) != v for k, v in identity["code"].items())
    ):
        raise ValueError("Parent checkpoint step or training provenance does not match")
    if payload["architecture"] != {
        "discoloop": config["models"][name]["discoloop"],
        "first_round_kv": False,
    } or payload["spec"] != {
        "unique_layers": config["model"]["unique_layers"],
        "repeats": config["model"]["repeats"],
        "phi": config["phi"],
        "name": name,
    }:
        raise ValueError("Parent checkpoint architecture is wrong for this branch")
    digest = hashlib.sha256()
    for key, tensor in payload["model"].items():
        if not torch.isfinite(tensor).all():
            raise ValueError("Parent checkpoint contains nonfinite model weights")
        digest.update(key.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    if digest.hexdigest() != payload["model_sha256"]:
        raise ValueError("Parent checkpoint model hash does not match its tensors")
    n = parent_config["training"]["batch_size"]
    expected_counts = {
        "base_parameter_updates": step,
        "old_atomic_examples": step * n,
        "old_chain_examples": step * n,
        "correction_examples": step * n,
        "fast_state_updates": 0,
        "adaptation_episodes": 0,
        "supervised_target_tokens": step * n * 6,
        "corrected_two_hop_training_examples": 0,
    }
    if payload["counts"] != expected_counts:
        raise ValueError(
            "Parent checkpoint warmup exposure is incomplete or includes write training"
        )
    optimizer = payload["optimizer"]
    data_rng = payload.get("data_rng", {})
    if (
        set(data_rng) != {"algorithm", "keys", "position", "has_gauss", "cached_gaussian"}
        or data_rng["algorithm"] != "MT19937"
        or len(data_rng["keys"]) != 624
        or not 0 <= data_rng["position"] <= 624
        or data_rng["has_gauss"] not in (0, 1)
        or not math.isfinite(data_rng["cached_gaussian"])
    ):
        raise ValueError("Parent data RNG state is incomplete or invalid")
    total = step + steps
    t = parent_config["training"]
    rate = t["minimum_learning_rate"] + 0.5 * (t["learning_rate"] - t["minimum_learning_rate"]) * (
        1 + math.cos(math.pi * (step - 1) / max(1, total - 1))
    )
    rate *= min(1.0, step / max(1, t["lr_warmup_steps"]))
    if (
        not optimizer["state"]
        or any(int(state["step"]) != step for state in optimizer["state"].values())
        or any(not math.isclose(g["lr"], rate, rel_tol=1e-12) for g in optimizer["param_groups"])
        or not payload["readout"]
        or "torch_rng" not in payload
        or "cuda_rng" not in payload
    ):
        raise ValueError("Parent optimizer, RNG or validation state is incomplete")
    provenance = {
        "parent_batch": branch["parent_batch"],
        "parent_checkpoint": str(path),
        "checkpoint_step": step,
        "checkpoint_sha256": sha256(path),
        "parent_model_sha256": payload["model_sha256"],
        "parent_manifest_sha256": branch["parent_manifest_sha256"],
        "parent_config_sha256": branch["parent_config_sha256"],
        "new_steps": steps,
        "objective": OBJECTIVE,
        "restored": ["model", "optimizer", "data_rng", "torch_rng", "cuda_rng", "global_lr"],
        "cost_accounting": "Inherited 20k state is reused; only the new continuation is added cost",
    }
    return payload, provenance


def train_branch(config, name, out, device, log=None):
    """Create a new reader-only continuation without overwriting the parent."""
    out = Path(out)
    if (out / "training").exists() or (out / "branch-input").exists():
        raise FileExistsError("Branch output already exists; preserve it, do not retry")
    parent_path = Path(config["branch"]["checkpoints"][name])
    payload, provenance = validate_branch_parent(config, name, parent_path)
    backup = out / "branch-input"
    backup.mkdir(parents=True)
    copied = backup / parent_path.name
    shutil.copy2(parent_path, copied)
    if sha256(copied) != provenance["checkpoint_sha256"]:
        raise ValueError("Parent checkpoint changed while the branch copied it")
    provenance["preserved_checkpoint"] = str(copied)
    write_json(backup / "provenance.json", provenance)
    return train_scratch(
        config,
        name,
        out,
        device,
        log,
        _branch_payload=payload,
        _branch_provenance=provenance,
    )
