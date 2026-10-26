"""One-step stream adaptation from an unchanged, audited common reader.

Only the per-fact writer budget, output identity and execution resources change.
The inherited engine retains the complete differentiable stream objective,
optimizer state, data stream and RNG. Single/continuous/merge are deployment
conditions on the resulting endpoint, not additional training objectives.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
import uuid
from pathlib import Path

import numpy as np
import torch

from . import ttcd_verified_base as verified
from .grok_depth import write_json
from .grokking_reproduction import MergedBatchStream, model_digest
from .path_overlap import sha256
from .ttcd_editing import tensor_digest

BATCH = "ttcd-one-step-training-v1"
ADAPTER_FILE = "src/llm_memory_editability/ttcd_one_step_training.py"


def _private_verified():
    name = f"llm_memory_editability._ttcd_one_step_training_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(name, verified.__file__)
    engine = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(engine)
    finally:
        sys.dont_write_bytecode = previous
    return engine


def _base_engine(config):
    if "architecture" in config:
        from .ttcd_architecture import build_engine

        return build_engine(config)
    return _private_verified()


def validate_task(task):
    """Check the authorized delta without rewriting the parent's identity."""
    if task["batch"] != BATCH or task["training_method"] != "stream":
        raise ValueError("Only the authorized one-step stream training batch is supported")
    name = task["architecture_name"]
    if name not in ("transformer", "loop", "discoloop"):
        raise ValueError("Unknown one-step training architecture")
    source = Path(task["parent_config"])
    if sha256(source) != task["parent_config_sha256"]:
        raise ValueError("Original parent configuration bytes differ")
    parent = json.loads(source.read_text())
    config = copy.deepcopy(task["training_config"])
    if parent.get("architecture", {}).get("name", "discoloop") != name:
        raise ValueError("Original parent architecture differs")
    if parent["writer"]["steps"] != 4 or config["writer"]["steps"] != 1:
        raise ValueError("Preserve the four-step control and train exactly one step per fact")
    if set(config["models"]) != {"stream"} or config["batch"] != BATCH:
        raise ValueError("Exactly one isolated stream endpoint is required")
    expected = copy.deepcopy(parent)
    expected["batch"] = BATCH
    expected["writer"]["steps"] = 1
    expected["runtime"] = copy.deepcopy(config["runtime"])
    expected["models"] = {"stream": copy.deepcopy(parent["models"]["stream"])}
    destination = config["models"]["stream"]["checkpoint"]
    expected["models"]["stream"]["checkpoint"] = destination
    if config != expected:
        raise ValueError("One-step training may only change writer budget, output and resources")
    fixture = parent.get("engineering_fixture_only") is True
    if not fixture and (
        parent["common"]["steps"] != 10000
        or config["training"]["steps"] != 10000
        or config["continual"]["train_length"] != 8
    ):
        raise ValueError("Reuse common10000 and retain stream10000 with length8")
    if type(config["writer"]["steps"]) is not int:
        raise ValueError("Writer steps must be the integer one")
    inputs = {
        "parent_config_sha256": source,
        "parent_common_checkpoint_sha256": Path(parent["common"]["checkpoint"]),
        "parent_common_gate_sha256": Path(parent["common"]["gate_receipt"]),
    }
    for field, path in inputs.items():
        if sha256(path) != task[field]:
            raise ValueError(f"Pinned original common-reader input differs: {field}")
    for field, source_key in (
        ("parent_common_checkpoint", "checkpoint"),
        ("parent_common_gate", "gate_receipt"),
    ):
        if (
            field in task
            and Path(task[field]).resolve() != Path(parent["common"][source_key]).resolve()
        ):
            raise ValueError("Task common-reader path differs from the original configuration")
    protected = {
        Path(parent["foundation"]["checkpoint"]).resolve(),
        Path(parent["common"]["checkpoint"]).resolve(),
        *(Path(spec["checkpoint"]).resolve() for spec in parent["models"].values()),
    }
    if Path(destination).resolve() in protected:
        raise ValueError("New adaptation must not overwrite any original checkpoint")
    return parent, config


def build_engine(task_config):
    """Return a private engine with unchanged train/load/audit function signatures.

    Use ``engine.config`` (or the equal task ``training_config``) and phase
    ``stream``. Every identity call rechecks pinned common-reader provenance.
    The parent's gate and complete checkpoint are interpreted only through its
    original engine/config, so their historical identity is never relabelled.
    """
    task = copy.deepcopy(task_config)
    parent, config = validate_task(task)
    original = _base_engine(parent)
    engine = _base_engine(config)
    root = Path(__file__).resolve().parents[2]
    code_files = tuple(dict.fromkeys((*original.CODE_FILES, ADAPTER_FILE)))

    def check_config(candidate):
        if candidate != config:
            raise ValueError("Use this task's exact one-step training configuration")
        observed, _ = validate_task(task)
        if observed != parent:
            raise ValueError("Original common-reader configuration changed")
        receipt = json.loads(Path(parent["common"]["gate_receipt"]).read_text())
        original_identity = original.identity(parent)
        if (
            receipt.get("passed") is not True
            or receipt.get("checkpoint_sha256") != task["parent_common_checkpoint_sha256"]
            or receipt.get("identity") != original_identity
        ):
            raise ValueError("Original common gate must bind the original audited source identity")
        return original_identity

    def identity(candidate):
        source_identity = check_config(candidate)
        return {
            "schema": BATCH,
            "config": copy.deepcopy(candidate),
            "architecture_name": task["architecture_name"],
            "training_method": "stream",
            "parent": {
                "config_path": task["parent_config"],
                "config_sha256": task["parent_config_sha256"],
                "common_checkpoint_sha256": task["parent_common_checkpoint_sha256"],
                "common_gate_sha256": task["parent_common_gate_sha256"],
                "identity": source_identity,
            },
            "code": {name: sha256(root / name) for name in code_files},
        }

    def validate_inputs(candidate):
        check_config(candidate)
        proxy = copy.deepcopy(candidate)
        if task["architecture_name"] == "discoloop":
            proxy["models"] = {
                "single": dict(candidate["models"]["stream"]),
                "stream": dict(candidate["models"]["stream"]),
            }
        return original.validate_inputs(proxy)

    def parent_for_phase(candidate, phase, device):
        if phase != "stream":
            raise ValueError("Only stream adaptation may run; never retrain foundation/common")
        check_config(candidate)
        # Includes original identity, phase, step, backbone and gate checks.
        # train() restores optimizer/native-stream/data/torch/CUDA RNG from
        # this exact complete parent using the unmodified audited code.
        return original._parent_for_phase(parent, "stream", device)

    base_load_reader = engine.load_reader

    def load_reader(candidate, phase, device, *, require_audit=True):
        if phase != "stream":
            raise ValueError("Only the new one-step stream endpoint may be loaded")
        return base_load_reader(candidate, phase, device, require_audit=require_audit)

    engine.identity = identity
    engine.validate_inputs = validate_inputs
    engine._parent_for_phase = parent_for_phase
    engine.load_reader = load_reader
    engine.CODE_FILES = code_files
    engine.config = copy.deepcopy(config)
    engine.one_step_parent_provenance = identity(config)["parent"]
    return engine


def state_digest(value):
    """Hash nested checkpoint/RNG values, including scalar optimizer tensors."""
    result = hashlib.sha256()

    def visit(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            result.update(str((tensor.dtype, tuple(tensor.shape))).encode())
            result.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, np.ndarray):
            result.update(str((item.dtype, item.shape)).encode())
            result.update(item.tobytes())
        elif isinstance(item, dict):
            result.update(b"dict")
            for key in sorted(item, key=str):
                visit(key)
                visit(item[key])
        elif isinstance(item, (tuple, list)):
            result.update(type(item).__name__.encode())
            for child in item:
                visit(child)
        else:
            result.update(repr(item).encode())

    visit(value)
    return result.hexdigest()


def preflight(task_config, out, device):
    """Zero optimizer updates: real common parent, full macro, exact speedup replay.

    Reads native/atomic training examples only. Neither formal edited two-hop
    queries nor checkpoint selection enter this numerical verification.
    """
    from .ttcd_train_speedup import enabled

    engine = build_engine(task_config)
    config = engine.config
    reader, world, _, _, parent = engine._parent_for_phase(config, "stream", device)
    before = model_digest(reader.backbone)
    if before != parent["model_sha256"]:
        raise ValueError("One-step training must begin at the exact common model")
    optimizer = engine.optimizer_for(reader, config)
    optimizer.load_state_dict(parent["optimizer"])
    inherited_optimizer = state_digest(parent["optimizer"])
    if state_digest(optimizer.state_dict()) != inherited_optimizer:
        raise AssertionError("Inherited AdamW state differs")
    table, _ = engine.native_table(world, config["phi"])
    stream = MergedBatchStream(len(table[0]), config["seed"] + 3)
    stream.load_state_dict(parent["native_stream"])
    if state_digest(stream.state_dict()) != state_digest(parent["native_stream"]):
        raise AssertionError("Inherited native data stream differs")
    rng = engine.restore_data_rng(parent["data_rng"])
    if engine.pack_rng(rng) != parent["data_rng"]:
        raise AssertionError("Inherited correction-data RNG differs")
    batch = engine.sample_step(config, world, rng, stream)
    first_packets = torch.as_tensor(
        batch["packets"][: config["training"]["micro_facts"]], device=device
    )
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)

    def restore_rng():
        torch.set_rng_state(parent["torch_rng"])
        torch.testing.assert_close(torch.get_rng_state(), parent["torch_rng"], rtol=0, atol=0)
        if str(device).startswith("cuda"):
            torch.cuda.set_rng_state(parent["cuda_rng"], device)
            torch.testing.assert_close(
                torch.cuda.get_rng_state(device), parent["cuda_rng"], rtol=0, atol=0
            )

    def fast_trace():
        trace = []
        with engine._amp(device):
            state = engine.write_states(
                reader,
                first_packets,
                config["writer"],
                config["continual"]["train_length"],
                "stream",
                observe=lambda position, fast: trace.append(
                    {"position": position, "sha256": tensor_digest(fast)}
                ),
            )
        if len(trace) != config["continual"]["train_length"]:
            raise AssertionError("Each stream fact must cause exactly one update")
        final = tensor_digest(state)
        del state
        return trace, final

    try:
        restore_rng()
        trace, final = fast_trace()
        reader.zero_grad(set_to_none=True)
        total, losses = engine.backward_objective(reader, config, world, table, batch, "stream")
        gradients = {
            name: None if parameter.grad is None else parameter.grad.detach().cpu().clone()
            for name, parameter in reader.named_parameters()
        }
        reader.zero_grad(set_to_none=True)
        restore_rng()
        with enabled() as speedup:
            replay_trace, replay_final = fast_trace()
            replay_total, replay_losses = engine.backward_objective(
                reader, config, world, table, batch, "stream"
            )
        if replay_trace != trace or replay_final != final:
            raise AssertionError("Capture speedup changed a one-step fast state")
        if replay_total != total or replay_losses != losses:
            raise AssertionError("Capture speedup changed the full-macro four-loss objective")
        for name, parameter in reader.named_parameters():
            expected = gradients[name]
            if expected is None:
                if parameter.grad is not None:
                    raise AssertionError(f"Capture speedup changed gradient presence: {name}")
            else:
                if parameter.grad is None:
                    raise AssertionError(f"Capture speedup removed a gradient: {name}")
                torch.testing.assert_close(parameter.grad.cpu(), expected, rtol=0, atol=0)
        if model_digest(reader.backbone) != before:
            raise AssertionError("Zero-update preflight changed backbone parameters")
        if state_digest(optimizer.state_dict()) != inherited_optimizer:
            raise AssertionError("Zero-update preflight changed inherited AdamW state")
        result = {
            "passed": True,
            "identity": engine.identity(config),
            "architecture_name": task_config["architecture_name"],
            "common_model_sha256": before,
            "common_optimizer_sha256": inherited_optimizer,
            "common_native_stream_sha256": state_digest(parent["native_stream"]),
            "common_data_rng_sha256": state_digest(parent["data_rng"]),
            "common_torch_rng_sha256": state_digest(parent["torch_rng"]),
            "common_cuda_rng_sha256": state_digest(parent["cuda_rng"]),
            "writer_steps": 1,
            "stream_length": config["continual"]["train_length"],
            "native_macro": len(batch["native_indices"]),
            "correction_macro": len(batch["packets"]),
            "micro_native": config["training"]["micro_native"],
            "micro_facts": config["training"]["micro_facts"],
            "explicit_fast_state_fact_rows": len(first_packets),
            "fast_trace": trace,
            "final_fast_sha256": final,
            "losses": losses,
            "all_fast_states_exact": True,
            "all_four_losses_exact": True,
            "all_parameter_gradients_exact": True,
            "gradient_sha256": state_digest(gradients),
            "gradient_parameters": sum(value is not None for value in gradients.values()),
            "speedup": dict(speedup),
            "base_parameter_updates": 0,
            "optimizer_updates": 0,
            "formal_D_queries": 0,
            "backbone_unchanged": True,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device)
            if str(device).startswith("cuda")
            else None,
        }
        write_json(Path(out) / "preflight.json", result)
        return result
    finally:
        reader.zero_grad(set_to_none=True)
        restore_rng()
