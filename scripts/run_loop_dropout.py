#!/usr/bin/env python3
"""Zero-dropout paired arm using the unchanged, frozen native training loop.

The scientific run stops at the last evaluation node (500k); its cosine learning
rate still uses the original 1.5m-step horizon. Only the three model dropout
probabilities change. Environment, smoke, grid, 33-file replay and cloud checks
reuse the state-MLP runner, with a separate endpoint contract.
"""

from __future__ import annotations

import copy
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import run_loop_state_handoff as original  # noqa: E402

native = original.native
torch = original.torch
BATCH = "loop-dropout-development-v1"
ARM = original.ARM
BACKBONE = original.BACKBONE
INITIAL_MODEL = "70289bd6aa675caded73065118b76b917282dcb81b040c2ef5ef7d1d69cc3640"
STOP_STEP = 500000
TRAINING = copy.deepcopy(original.TRAINING)
MODEL = {**original.MODEL, "attn_pdrop": 0.0, "embd_pdrop": 0.0, "resid_pdrop": 0.0}
NODES = [step for step in original.NODES if step <= STOP_STEP]
DROPOUT_KEYS = ("attn_pdrop", "embd_pdrop", "resid_pdrop")
FROZEN_SOURCES = {
    "scripts/run_loop_state_handoff.py": (
        "9530bc32f8cef40b28eaff1baf50ef97b3b0dee9d3be2acc7ea56047f19d5a72"
    ),
    "scripts/run_loop_native_handoff.py": (
        "7b100f2e99888381c5483776b5331c2d1aee562103060a4f775f8e15bb434a1b"
    ),
    "scripts/run_loop_hop.py": ("89ace5fde790a000c3b9b9e8aaad0c43a2cadc5f1daf22e05a41f8dab5913946"),
    "src/llm_memory_editability/loop_native_handoff.py": (
        "2c49b84595707bac69ef38375486c697237b98d6cde844cd8181037114ea53d7"
    ),
    "src/llm_memory_editability/self_decode_alignment.py": (
        "1bfd5c15da162ea032caa0579eae6b06b7316e497b3e7ccd747c3252212138a3"
    ),
    "src/llm_memory_editability/supervised_alignment.py": (
        "9c84f60e536d5bd39919a2b3037b7ecc416dc5fc2a9f2da16ae3c5855524ec74"
    ),
    "src/llm_memory_editability/loop_hop_data.py": (
        "940efb287f125d1f4b14bc71e72a967d9fd2685723bf41f6e93b632532d64dd6"
    ),
    "src/llm_memory_editability/grokking_reproduction.py": (
        "7e3a89d96f6fef7bd4665ac46dd1acb6e7cb461c6878647ecc7c6f425ac31117"
    ),
    "src/llm_memory_editability/round_evaluation.py": (
        "3956ca9347099013230eb3170d00f92e10cd0e6ccba5335a979f3b1c97b2610a"
    ),
    "src/llm_memory_editability/grok_depth.py": (
        "7d6d5af38a5dc4438b1e7ff44df8e833169d500f0d60d8933932bdb8822aca26"
    ),
}
_NATIVE_TRAIN = native.train
_INITIALIZE = native.initialize
_ENVIRONMENT = original.environment
_ADMISSION = original.check_training_admission
_AUDIT = original.audit
_CLOUD = original.cloud
_PARSER = original.parser
_READ = original._read


def validate_config(config):
    """Lock the paired science, distinct stop/horizon and one-arm authorization."""
    exact = {
        "batch": BATCH,
        "arms": {ARM: {"handoff": ARM}},
        "data": {
            "world_seed": 2026101601,
            "entities_per_graph": 500,
            "relations": 10,
            "train_compositions_per_hop": 10000,
            "evaluation_templates": 10,
            "training_panel_size": 1000,
        },
        "model": MODEL,
        "training": TRAINING,
        "handoff": original.HANDOFF,
        "score_design": original.SCORE_DESIGN,
        "dropout_seed": 2026101604,
        "evaluation_nodes": NODES,
        "execution_stop_step": STOP_STEP,
        "evaluation_rounds": [4, 8],
        "evaluation_batch_size": 256,
        "grid_rounds": 32,
        "grid_joint_rounds": [4, 8, 16, 32],
        "heartbeat_steps": 1000,
        "smoke_steps": 32,
        "tracking": {
            "provider": "wandb",
            "entity": "zsqzz",
            "project": "llm-memory-editability",
            "mode": "online",
        },
    }
    for key, expected in exact.items():
        if config.get(key) != expected:
            raise ValueError(f"Unexpected {key}: only the three dropout probabilities may change")
    paired = config.get("paired_recipe", {})
    if paired.get("initial_backbone_sha256") != BACKBONE:
        raise ValueError("The original backbone initialization must be reused")
    if paired.get("initial_model_sha256") != INITIAL_MODEL:
        raise ValueError("The complete original state-MLP initialization must be reused")
    if any(config.get("input_origin", {}).get(k) != v for k, v in original.DATA_HASHES.items()):
        raise ValueError("The original world and panels must be reused")
    if not config.get("input_root") or not config.get("source_references"):
        raise ValueError("Missing input root or source references")
    comparator = config.get("reused_comparators", {}).get(ARM, {})
    if not comparator.get("run") or comparator.get("new_training_updates") != 0:
        raise ValueError("Reuse the original state-MLP control without additional updates")
    for key, expected in {
        "new_scientific_arms": 1,
        "new_scientific_updates": STOP_STEP,
        "final_steps_per_arm": STOP_STEP,
        "learning_rate_horizon_steps": TRAINING["steps"],
        "engineering_smoke_updates_per_arm": 32,
        "new_world_updates": 0,
    }.items():
        if config.get("budget", {}).get(key) != expected:
            raise ValueError(f"Unexpected budget: {key}")
    runtime = config.get("runtime", {})
    if runtime.get("image") != original.IMAGE or runtime.get("docker_context") != "lm-memory":
        raise ValueError("Use the original verified LM image and Docker context")
    original._no_empty_dicts(config)


def verify_frozen_sources():
    """Refuse model/training/data changes beneath the thin adapter."""
    hashes = {path: native.digest(ROOT / path) for path in FROZEN_SOURCES}
    if hashes != FROZEN_SOURCES:
        changed = [path for path in hashes if hashes[path] != FROZEN_SOURCES[path]]
        raise ValueError(f"Original frozen source changed: {changed}")
    if (
        Path(native.train.__code__.co_filename).resolve()
        != ROOT / "scripts/run_loop_native_handoff.py"
    ):
        raise ValueError("The imported native training function comes from another source tree")
    if native.train is not _NATIVE_TRAIN:
        raise ValueError("The native training function must remain unchanged")
    return hashes


def assert_zero_dropout(model):
    """Check unused modules too, plus the config used by functional dropout/SDPA."""
    modules = {
        name: float(module.p)
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.modules.dropout._DropoutNd)
    }
    functional = {key: float(getattr(model.config, key)) for key in DROPOUT_KEYS}
    if not modules or any(value != 0 for value in (*modules.values(), *functional.values())):
        raise AssertionError("All module and functional dropout probabilities must be zero")
    return {"module_probabilities": modules, "functional_probabilities": functional}


def assert_full_initialization(model, reference_state, expected=INITIAL_MODEL):
    """Include scorer, beta, tied aliases and buffers, not just the backbone."""
    state = model.state_dict()
    if state.keys() != reference_state.keys():
        raise AssertionError("The full paired model state keys differ")
    for name, value in state.items():
        reference = reference_state[name]
        if value.dtype != reference.dtype or value.shape != reference.shape:
            raise AssertionError(f"The full paired model state shape/dtype differs: {name}")
        if not torch.equal(value.detach().cpu(), reference.detach().cpu()):
            raise AssertionError(f"The full paired model initialization differs: {name}")
    if native.model_digest(model) != expected:
        raise AssertionError("The complete initial model hash differs from the original state-MLP")
    return len(state)


def checked_initialize(config, name, metadata, device):
    model = _INITIALIZE(config, name, metadata, device)
    assert_zero_dropout(model)
    # The reused environment check also creates its historical zero-score EMA reference.
    if config["arms"][name]["handoff"] == ARM and native.model_digest(model) != INITIAL_MODEL:
        raise AssertionError("The complete initial state-MLP hash changed")
    return model


def environment(config, base, config_path, device):
    """Extend the original CUDA/gradient checks with the exact paired full state."""
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("The environment stage requires a real CUDA device")
    result = _ENVIRONMENT(config, base, config_path, device)
    source_hashes = verify_frozen_sources()
    reference_root = Path(config["reused_comparators"][ARM]["run"])
    reference_checkpoint = reference_root / "checkpoint-0000000.pt"
    reference_run = _READ(reference_root / "run.json")
    saved = torch.load(reference_checkpoint, map_location="cpu", weights_only=False)
    if saved.get("step") != 0 or saved.get("arm") != ARM or saved.get("handoff") != ARM:
        raise AssertionError("The paired reference is not the original state-MLP step zero")
    if saved.get("world_sha256") != original.DATA_HASHES["world_sha256"]:
        raise AssertionError("The paired reference uses another world")
    if reference_run.get("initial_model_sha256") != INITIAL_MODEL:
        raise AssertionError("The original run does not identify the expected full initialization")
    if saved.get("model_sha256") != INITIAL_MODEL:
        raise AssertionError("The original step-zero checkpoint identifies another full model")
    _, panels, metadata = original._load_data(config)
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        model = checked_initialize(config, ARM, metadata, device).float()
        state_tensors = assert_full_initialization(model, saved["model"])
        dropout = assert_zero_dropout(model)
        packed = native.pack_rows(panels["atomic_id"][:2], metadata)
        tokens, positions, _ = [torch.as_tensor(part, device=device) for part in packed]
        with torch.no_grad():
            model.train()
            train_logits = model.run(tokens, positions, rounds=8, carry=False)[0]
            model.eval()
            eval_logits = model.run(tokens, positions, rounds=8, carry=False)[0]
            torch.testing.assert_close(train_logits, eval_logits, rtol=0, atol=0)
            model.train()
            _, carry = model.run(tokens, positions, rounds=4)
            segmented = model.run(tokens, positions, rounds=4, state=carry, carry=False)[0]
            torch.testing.assert_close(train_logits, segmented, rtol=0, atol=0)
        torch.cuda.synchronize(device)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
    result.update(
        initial_full_model_exact=True,
        reference_initial_model_sha256=INITIAL_MODEL,
        initial_state_tensors_compared=state_tensors,
        reference_step_zero_checkpoint=str(reference_checkpoint),
        reference_step_zero_checkpoint_sha256=native.digest(reference_checkpoint),
        dropout_contract=dropout,
        train_eval_logits_exact=True,
        train_segmented_logits_exact=True,
        frozen_source_sha256=source_hashes,
        execution_stop_step=STOP_STEP,
        lr_horizon_steps=TRAINING["steps"],
    )
    result["source_sha256"][str(Path(__file__).relative_to(ROOT))] = native.digest(__file__)
    native.write_json(base / "environment.json", result)
    return result


def check_training_admission(config, base, config_path):
    """Do not admit training with only the older backbone-level environment proof."""
    evidence = _READ(base / "environment.json")
    for key in (
        "initial_full_model_exact",
        "train_eval_logits_exact",
        "train_segmented_logits_exact",
    ):
        if evidence.get(key) is not True:
            raise ValueError(f"Missing zero-dropout/full-initialization CUDA evidence: {key}")
    if evidence.get("initial_model_sha256") != INITIAL_MODEL:
        raise ValueError("Environment has a different full model initialization")
    if evidence.get("reference_initial_model_sha256") != INITIAL_MODEL:
        raise ValueError("Environment has a different paired full model initialization")
    dropout = evidence.get("dropout_contract", {})
    modules, functional = (
        dropout.get("module_probabilities"),
        dropout.get("functional_probabilities"),
    )
    if not modules or functional != dict.fromkeys(DROPOUT_KEYS, 0.0):
        raise ValueError("Incomplete dropout audit")
    if any(value != 0 for value in modules.values()):
        raise ValueError("Nonzero module dropout")
    if evidence.get("frozen_source_sha256") != verify_frozen_sources():
        raise ValueError("Environment used another frozen source")
    checkpoint = Path(config["reused_comparators"][ARM]["run"]) / "checkpoint-0000000.pt"
    if evidence.get("reference_step_zero_checkpoint_sha256") != native.digest(checkpoint):
        raise ValueError("The paired step-zero reference checkpoint changed")
    _ADMISSION(config, base, config_path)


@contextmanager
def endpoint_contract():
    """Change only the legacy audit/cloud endpoint constant, never config.training."""
    previous = original.TRAINING
    original.TRAINING = {**TRAINING, "steps": STOP_STEP}
    try:
        yield
    finally:
        original.TRAINING = previous


def audit(config, base, config_path, device, *, smoke):
    with endpoint_contract():
        return _AUDIT(config, base, config_path, device, smoke=smoke)


def cloud(config, base, config_path):
    with endpoint_contract():
        return _CLOUD(config, base, config_path)


def independent_audit(args, *, smoke):
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "smoke-audit" if smoke else "audit",
            "--config",
            str(args.config),
            "--base",
            str(args.base),
            "--arm",
            args.arm,
            "--device",
            args.device,
        ],
        check=True,
    )


def report_budget(value):
    """Relabel output metadata in place before the original progress writer prints it."""
    if isinstance(value, dict) and value.get("budget") == TRAINING["steps"]:
        value.update(
            budget=STOP_STEP, execution_stop_step=STOP_STEP, lr_horizon_steps=TRAINING["steps"]
        )
    return value


@contextmanager
def native_contract():
    """Patch configuration/admission/output adapters; leave native.train untouched."""
    verify_frozen_sources()
    previous = native.validate_config, native.step_flops, native.initialize, native.write_json
    write_json = native.write_json

    def write(path, value):
        if Path(path).name in ("status.json", "complete.json"):
            report_budget(value)
        return write_json(path, value)

    native.validate_config, native.step_flops = validate_config, original.step_flops
    native.initialize, native.write_json = checked_initialize, write
    try:
        yield
    finally:
        native.validate_config, native.step_flops, native.initialize, native.write_json = previous


def parser():
    result = _PARSER()
    result.description = __doc__
    return result


validate_arguments = original.validate_arguments


def main(argv=None):
    replacements = {
        "parser": parser,
        "validate_config": validate_config,
        "environment": environment,
        "check_training_admission": check_training_admission,
        "native_contract": native_contract,
        "independent_audit": independent_audit,
        "audit": audit,
        "cloud": cloud,
    }
    previous = {name: getattr(original, name) for name in replacements}
    for name, value in replacements.items():
        setattr(original, name, value)
    try:
        return original.main(argv)
    finally:
        for name, value in previous.items():
            setattr(original, name, value)


if __name__ == "__main__":
    os.umask(0o022)
    main()
