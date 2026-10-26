"""Architecture comparison through isolated copies of the audited TTCD engine.

Only backbone construction, stream-only input validation, provenance and
advisory prerequisite reporting differ. Losses, sampling, optimization, fast
updates and replay remain the existing verified-base implementation. Each
engine has its own module namespace; no imported experiment is monkeypatched.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
import uuid
from pathlib import Path

import numpy as np
from transformers import GPT2Config

from . import ttcd_continual_eval as evaluation
from . import ttcd_verified_base as verified
from .grok_depth import utc, write_json
from .grokking_reproduction import model_digest
from .path_overlap import sha256
from .supervised_alignment import SupervisedAlignmentGPT
from .ttcd_editing import TTCDReader
from .ttcd_experiment import open_query, packet_for

CODE_FILES = tuple(
    dict.fromkeys(
        (
            *verified.CODE_FILES,
            "src/llm_memory_editability/ttcd_architecture.py",
            "scripts/run_ttcd_architecture.py",
        )
    )
)


def architecture_spec(config):
    spec = config["architecture"]
    expected = {"transformer": (8, 1), "loop": (4, 2)}
    if spec["name"] not in expected:
        raise ValueError("Only the new ordinary D8 and Loop4x2 arms use this adapter")
    layers, repeats = expected[spec["name"]]
    if (
        spec["unique_layers"] != layers
        or spec["repeats"] != repeats
        or spec.get("discoloop") is not False
    ):
        raise ValueError("Preserve D8/Loop4x2 depth and disable the Disco channel")
    return {"name": spec["name"], "unique_layers": layers, "repeats": repeats, "discoloop": False}


def validate_inputs(config):
    architecture_spec(config)
    if set(config["models"]) != {"stream"}:
        raise ValueError("This comparison trains only the continual stream arm")
    if config["models"]["stream"].get("discoloop", False):
        raise ValueError("The new ordinary/Loop arms cannot enable Disco")
    # Reuse all existing graph, exclusion, packet and optimizer checks. This
    # compatibility view is never saved, trained or used as an identity.
    proxy = copy.deepcopy(config)
    stream = {**proxy["models"]["stream"], "discoloop": True}
    proxy["models"] = {"single": dict(stream), "stream": dict(stream)}
    return verified.validate_inputs(proxy)


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
        "adapter": "ttcd-architecture-stream-v1",
    }


def load_backbone(config, saved, world, device):
    architecture = architecture_spec(config)
    spec = saved["spec"]
    if any(spec.get(key) != architecture[key] for key in ("unique_layers", "repeats")):
        raise ValueError("Checkpoint physical layers/repeats differ from the declared architecture")
    flags = saved.get("architecture", {})
    if flags.get("discoloop", False) or flags.get("first_round_kv", False):
        raise ValueError("Ordinary/Loop foundations cannot contain Disco or KV fusion")
    hf = copy.deepcopy(GPT2Config.from_json_file(config["model_config"]))
    hf.vocab_size, hf.n_layer = world.metadata["vocab_size"], architecture["unique_layers"]
    hf.use_cache = False
    hf.embd_pdrop = hf.resid_pdrop = hf.attn_pdrop = 0.0
    model = SupervisedAlignmentGPT(
        hf, architecture["repeats"], discoloop=False, first_round_kv=False
    )
    model.load_state_dict(saved["model"], strict=True)
    if saved.get("model_sha256") and model_digest(model) != saved["model_sha256"]:
        raise ValueError("Checkpoint internal model digest differs")
    return model.to(device).eval()


def foundation_model(config, device, *, trainable=False):
    world = validate_inputs(config)
    parent = config["foundation"]
    checkpoint = Path(parent["checkpoint"])
    digest = sha256(checkpoint)
    if parent.get("checkpoint_sha256") and digest != parent["checkpoint_sha256"]:
        raise ValueError("Pinned native foundation checkpoint differs")
    receipt = json.loads(Path(parent["gate_receipt"]).read_text())
    if (
        receipt.get("passed") is not True
        or receipt.get(parent.get("gate_checkpoint_field", "checkpoint_sha256")) != digest
    ):
        raise ValueError("Passing native numerical/identity audit must bind this exact checkpoint")
    saved = copy.copy(verified._load(checkpoint))
    if saved["step"] != parent["step"] or saved["spec"]["phi"] != config["phi"]:
        raise ValueError("Wrong native terminal step or composition support")
    for key, expected in parent.get("required_spec", {}).items():
        if saved["spec"].get(key) != expected:
            raise ValueError(f"Native foundation recipe differs: {key}")
    if parent.get("config_path"):
        original = json.loads(Path(parent["config_path"]).read_text())
        if original["spec"] != saved["spec"]:
            raise ValueError("Native foundation differs from the frozen training recipe")
    model = load_backbone(config, saved, world, device)
    if receipt.get("model_sha256") and model_digest(model) != receipt["model_sha256"]:
        raise ValueError("Native numerical audit binds another model")
    saved["architecture"] = {
        "first_round_kv": False,
        "discoloop": False,
        "discoloop_alpha": 1.0,
        "discoloop_temperature": 1.0,
        "discoloop_eps": 1e-6,
    }
    reader = TTCDReader(model, trainable_backbone=trainable, native_causal_attention=True).eval()
    provenance = {
        "checkpoint_sha256": digest,
        "model_sha256": model_digest(model),
        "gate_receipt_sha256": sha256(parent["gate_receipt"]),
        "step": saved["step"],
        "spec": saved["spec"],
        "architecture": saved["architecture"],
        "comparison_architecture": architecture_spec(config),
        "quality_selection": False,
    }
    return reader, world, saved, provenance


def architecture_accounting(reader):
    model = reader.backbone
    fast = reader.empty_fast()
    return {
        "physical_layers": len(model.transformer.h),
        "repeats": model.repeats,
        "effective_block_executions": len(model.transformer.h) * model.repeats,
        "base_parameters": sum(p.numel() for p in model.parameters()),
        "fast_matrices": len(fast),
        "fast_elements": sum(a.numel() for a in fast.values()),
        "fast_bytes": sum(a.numel() * a.element_size() for a in fast.values()),
        "fast_sharing": "One matrix per physical MLP, reused on every round",
        "equal_compute_or_capacity_claim": False,
    }


def common_gate(engine, config, out, device):
    # load_reader requires the independent terminal numerical replay audit.
    reader, world = engine.load_reader(config, "common", device)
    native = engine.native_readout(reader, config, world, device)
    parent, _, _, _ = engine.foundation_inputs(config, device)
    before = engine.native_readout(parent, config, world, device)
    settings, checks = config["capability_gate"], {}
    for name in engine.PANELS:
        score = native[name]["metrics"]["old_accuracy"]
        original = before[name]["metrics"]["old_accuracy"]
        floor = settings.get("minimum_accuracy", {}).get(name)
        if floor is None:
            floor = (
                settings.get("minimum_atomic_and_ii", 0.98)
                if name in ("atomic_id", "atomic_ood", "train_ii", "test_ii")
                else settings.get("minimum_io_oi_oo", 0.9)
            )
        drop = settings.get("maximum_reference_drop", settings.get("maximum_drop", 0.02))
        checks[name] = {
            "accuracy": score,
            "foundation_accuracy": original,
            "drop": original - score,
            "minimum": floor,
            "passed": score >= floor and original - score <= drop + 1e-12,
            "report_only": True,
        }
    records = []
    for case in evaluation.case_bank(config):
        if case["split"] == "calibration":
            atom = world.atoms[case["atom"]]
            records.append(
                {
                    "case": case,
                    "readout": open_query(
                        reader,
                        None,
                        packet_for(config, world, case),
                        config["writer"]["teacher_window"],
                        int(atom[-1]),
                        case["new_tail"],
                        world.metadata,
                        device,
                    ),
                }
            )
    groups = {}
    for group in ("id", "ood"):
        values = [
            r["readout"]["new_answer_correct"] for r in records if r["case"]["group"] == group
        ]
        groups[group] = {
            "n": len(values),
            "accuracy": float(np.mean(values)) if values else None,
            "passed": len(values) >= config["reading_gate"]["minimum_cases_per_group"]
            and bool(values)
            and float(np.mean(values)) >= config["reading_gate"]["minimum_accuracy"],
            "report_only": True,
        }
    result = {
        "passed": True,
        "numerical_audit_passed": True,
        "quality_passed": all(c["passed"] for c in checks.values())
        and all(g["passed"] for g in groups.values()),
        "quality_is_advisory": True,
        "identity": engine.identity(config),
        "checkpoint_sha256": sha256(config["common"]["checkpoint"]),
        "native_checks": checks,
        "reading_groups": groups,
        "reading_records": records,
        "native_readout": native,
        "foundation_readout": before,
        "architecture_accounting": architecture_accounting(reader),
        "formal_D_used": False,
        "status": "numerical_audit_passed_fixed_budget_stream",
        "utc": utc(),
    }
    write_json(Path(out) / "gate.json", result)
    return result


def build_engine(config):
    """Return a private verified-base namespace without mutating its import."""
    architecture_spec(config)
    name = f"llm_memory_editability._ttcd_architecture_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(name, verified.__file__)
    engine = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(engine)
    finally:
        sys.dont_write_bytecode = previous
    engine.validate_inputs, engine.identity = validate_inputs, identity
    engine._backbone, engine.foundation_inputs = load_backbone, foundation_model
    engine.common_gate = lambda cfg, out, device: common_gate(engine, cfg, out, device)
    engine.CODE_FILES = CODE_FILES
    return engine


def calibrate(engine, config, name, out, device):
    result = evaluation.calibrate(
        config,
        name,
        out,
        device,
        reader_loader=engine.load_reader,
        identity_factory=engine.evaluation_identity,
    )
    result["quality_passed"] = result["passed"]
    result.update(
        passed=True, quality_is_advisory=True, status="numerically_valid_prerequisites_reported"
    )
    write_json(Path(out) / "calibration.json", result)
    return result


def supplementary_identity(engine, config, settings, name, panel):
    from .ttcd_double_new_eval import identity as double_identity

    return double_identity(
        config, settings, name, panel, parent_identity=engine.evaluation_identity
    )
