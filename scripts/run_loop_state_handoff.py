#!/usr/bin/env python3
"""One raw-state MLP handoff arm, reusing the frozen native training recipe.

The environment and smoke stages are engineering checks. Scientific training is
admitted only after a CUDA initialization check and an independent smoke replay.
The original native runner and its historical handoff matrix remain unchanged.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import traceback
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import run_loop_native_handoff as native  # noqa: E402

BATCH = "loop-state-handoff-development-v1"
ARM = "state-mlp"
IMAGE = "sha256:c77af324439d8a2f838b57c3a386fc8477135015a361e4fbf6a8389facbf9f14"
BACKBONE = "39a487823c05311a223db99c3d0e2df3e7a1752498e261337f4298abe0cfa7b0"
DATA_HASHES = {
    "world_sha256": "e132eaf6bba5786bbf05f6f36c1b7383f698e93ee0dee313f568f6d50a77026f",
    "panels_sha256": "c991cebeeeec05639be87d981cd20031016f47965c43fed8bf5cc55579c3b468",
}
MODEL = {
    "n_embd": 768,
    "n_head": 12,
    "n_positions": 1024,
    "n_ctx": 1024,
    "activation_function": "gelu_new",
    "attn_pdrop": 0.1,
    "embd_pdrop": 0.1,
    "resid_pdrop": 0.1,
    "initializer_range": 0.02,
    "layer_norm_epsilon": 1e-5,
}
TRAINING = {
    "initialization": 2026101602,
    "stream_seed": 2026101603,
    "schedule_seed": 2026102001,
    "unique_layers": 1,
    "evaluation_repeats": 4,
    "weight_decay": 0.1,
    "lr": 0.0001,
    "warmup": 2000,
    "final_lr_fraction": 0.1,
    "betas": [0.9, 0.95],
    "eps": 1e-8,
    "grad_clip": 1.0,
    "ema_decay": 0.999,
    "batch_size": 512,
    "steps": 1500000,
    "segments": 2,
    "first_segment_rounds": [4, 6],
    "later_segment_rounds": [1, 4],
}
HANDOFF = {
    "initial_beta": 0.55,
    "score_hidden": 16,
    "score_clip": 8.0,
    "score_seed": 2026102002,
}
SCORE_DESIGN = {
    "inputs": ["previous_state", "candidate"],
    "hidden": 16,
    "activation": "gelu",
    "output": "scalar score; existing normalized history aggregation",
    "additional_normalization": False,
}
NODES = [
    0,
    1000,
    8000,
    16000,
    32000,
    64000,
    128000,
    256000,
    300000,
    400000,
    500000,
    600000,
    700000,
    800000,
    900000,
    1000000,
    1100000,
    1200000,
    1300000,
    1400000,
    1500000,
]
_NATIVE_FLOPS = native.step_flops


def _no_empty_dicts(value, path="experiment"):
    if isinstance(value, dict):
        if not value:
            raise ValueError(f"Empty dictionary cannot survive W&B config roundtrip: {path}")
        for key, item in value.items():
            _no_empty_dicts(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _no_empty_dicts(item, f"{path}[{index}]")


def validate_config(config):
    """Reject changes to the reused scientific recipe or the authorized one-arm budget."""
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
        "handoff": HANDOFF,
        "score_design": SCORE_DESIGN,
        "dropout_seed": 2026101604,
        "evaluation_nodes": NODES,
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
            raise ValueError(f"Unexpected {key}: the state-MLP recipe is fixed")
    if config.get("paired_recipe", {}).get("initial_backbone_sha256") != BACKBONE:
        raise ValueError("The original backbone initialization must be reused")
    if any(config.get("input_origin", {}).get(k) != v for k, v in DATA_HASHES.items()):
        raise ValueError("The original world and panels must be reused")
    if not config.get("input_root"):
        raise ValueError("Missing input_root")
    if not isinstance(config.get("reused_comparators"), dict) or not config["reused_comparators"]:
        raise ValueError("Reused comparators must identify the existing scientific runs")
    if not config.get("source_references"):
        raise ValueError("Missing source references")
    budget = config.get("budget", {})
    for key, expected in {
        "new_scientific_arms": 1,
        "new_scientific_updates": 1500000,
        "final_steps_per_arm": 1500000,
        "engineering_smoke_updates_per_arm": 32,
        "new_world_updates": 0,
    }.items():
        if budget.get(key) != expected:
            raise ValueError(f"Unexpected budget: {key}")
    runtime = config.get("runtime", {})
    if runtime.get("image") != IMAGE or runtime.get("docker_context") != "lm-memory":
        raise ValueError("Use the verified LM image and Docker context")
    _no_empty_dicts(config)


def step_flops(model, metadata, batch_size, schedule):
    """Add the state scorer's two matrices to native training FLOPs (factor three)."""
    result = _NATIVE_FLOPS(model, metadata, batch_size, schedule)
    if model.handoff == ARM:
        width, hidden = model.config.n_embd, model.handoff_score_in.shape[0]
        result += (
            3
            * 2
            * batch_size
            * metadata["sequence_length"]
            * (2 * width * hidden + hidden)
            * sum(schedule)
        )
    return result


@contextmanager
def native_contract():
    """Install this batch's contract locally without changing historical runner behavior."""
    old_validate, old_flops = native.validate_config, native.step_flops
    native.validate_config, native.step_flops = validate_config, step_flops
    try:
        yield
    finally:
        native.validate_config, native.step_flops = old_validate, old_flops


def _read(path):
    return json.loads(Path(path).read_text())


def _load_data(config):
    world, panels, metadata = native.load(config["input_root"])
    if any(metadata.get(k) != v for k, v in DATA_HASHES.items()):
        raise ValueError("Loaded data do not match the paired world and panel hashes")
    return world, panels, metadata


def environment(config, base, config_path, device):
    """Real CUDA computation and full-width initial equality; no W&B initialization."""
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("The environment stage requires a real CUDA device")
    _, panels, metadata = _load_data(config)
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        torch.manual_seed(2026102201)
        left = torch.randn(32, 32, device=device, requires_grad=True)
        right = torch.randn(32, 32, device=device, requires_grad=True)
        product = left @ right
        product.square().mean().backward()
        torch.cuda.synchronize(device)
        if not all(torch.isfinite(x).all() for x in (product, left.grad, right.grad)):
            raise FloatingPointError("CUDA matrix/backward check is non-finite")
        current = native.initialize(config, ARM, metadata, device).float().eval()
        comparison = copy.deepcopy(config)
        comparison["arms"][ARM]["handoff"] = "ema"
        reference = native.initialize(comparison, ARM, metadata, device).float().eval()
        # Eval disables all configured dropout without changing the paired initial weights.
        backbones = [native.backbone_digest(m) for m in (current, reference)]
        if backbones != [BACKBONE, BACKBONE]:
            raise AssertionError("CUDA models do not match the reused initial backbone")
        initial_sha = native.model_digest(current)
        reference_sha = native.model_digest(reference)
        packed = native.pack_rows(panels["atomic_id"][:2], metadata)
        tokens, positions, labels = [torch.as_tensor(p, device=device) for p in packed]
        a = current.run(tokens, positions, rounds=3, carry=False)[0]
        b = reference.run(tokens, positions, rounds=3, carry=False)[0]
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        F.cross_entropy(a.flatten(0, 1), labels.flatten()).backward()
        F.cross_entropy(b.flatten(0, 1), labels.flatten()).backward()
        checked, score_gradients = [], {}
        for name, parameter in current.named_parameters():
            if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                raise AssertionError(f"Missing or non-finite gradient: {name}")
            if name.startswith("handoff_score_"):
                # A zero output matrix deliberately gives the first layer zero initial gradient.
                score_gradients[name] = {
                    "finite": True,
                    "nonzero": bool(torch.count_nonzero(parameter.grad)),
                }
                continue
            expected = reference.get_parameter(name).grad
            if expected is None or not torch.isfinite(expected).all():
                raise AssertionError(f"Missing or non-finite EMA gradient: {name}")
            torch.testing.assert_close(parameter.grad, expected, rtol=0, atol=0, msg=name)
            checked.append(name)
        if "handoff_beta_logit" not in checked:
            raise AssertionError("The beta gradient was not compared")
        torch.cuda.synchronize(device)
        properties = torch.cuda.get_device_properties(device)
        result = {
            "passed": True,
            "engineering_only": True,
            "stage": "environment",
            "config_sha256": native.digest(config_path),
            "image": config["runtime"]["image"],
            **DATA_HASHES,
            "initial_model_sha256": initial_sha,
            "initial_ema_model_sha256": reference_sha,
            "initial_backbone_sha256": backbones[0],
            "reference_initial_backbone_sha256": backbones[1],
            "cuda_matrix_backward_synchronized": True,
            "initial_logits_exact": True,
            "initial_backbone_and_beta_gradients_exact": True,
            "compared_gradient_names": checked,
            "score_gradients": score_gradients,
            "width": current.config.n_embd,
            "batch_size": len(tokens),
            "rounds": 3,
            "dtype": "float32; TF32 and dropout disabled",
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "gpu_name": properties.name,
            "gpu_uuid": str(getattr(properties, "uuid", "not exposed by this PyTorch build")),
            "gpu_index": device.index or 0,
            "gpu_memory_bytes": properties.total_memory,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "hostname": platform.node(),
            "pid": os.getpid(),
            "utc": native.utc(),
            "source_sha256": {
                str(path.relative_to(ROOT)): native.digest(path)
                for path in (
                    Path(__file__).resolve(),
                    ROOT / "scripts/run_loop_native_handoff.py",
                    ROOT / "src/llm_memory_editability/loop_native_handoff.py",
                )
            },
        }
        native.write_json(base / "environment.json", result)
        return result
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32


def compare_prediction(saved, replay):
    """Compare every saved field, including exact answer/EOS tokens and row order."""
    if set(saved) != set(replay) or set(saved) != {"rows", "answer", "stop", "nll"}:
        raise AssertionError("Unexpected or missing prediction fields")
    evidence = {}
    for key in saved:
        original, actual = saved[key], replay[key]
        if original.shape != actual.shape or original.dtype != actual.dtype:
            raise AssertionError(f"Prediction shape/dtype changed: {key}")
        if key == "nll":
            if not np.isfinite(original).all() or not np.isfinite(actual).all():
                raise AssertionError("Non-finite saved/replayed NLL")
            np.testing.assert_allclose(original, actual, rtol=1e-6, atol=1e-6)
        else:
            np.testing.assert_array_equal(original, actual)
        evidence[key] = {
            "sha256": hashlib.sha256(original.tobytes()).hexdigest(),
            "replay_sha256": hashlib.sha256(actual.tobytes()).hexdigest(),
            "exact": bool(np.array_equal(original, actual)),
        }
    return evidence


def audit(config, base, config_path, device, *, smoke):
    """Fresh-process reload of all 33 endpoint prediction files, not just accuracies."""
    out = base / ("smoke" if smoke else "runs") / ARM
    complete, run = _read(out / "complete.json"), _read(out / "run.json")
    config_sha = native.digest(config_path)
    if any(item.get("config_sha256") != config_sha for item in (complete, run)):
        raise AssertionError("Run configuration identity changed")
    expected_step = config["smoke_steps"] if smoke else TRAINING["steps"]
    if complete["step"] != expected_step or complete["engineering_only"] != smoke:
        raise AssertionError("Wrong endpoint or run kind")
    if run["pid"] == os.getpid():
        raise AssertionError("Replay must execute in a process independent of training")
    checkpoint = out / f"checkpoint-{complete['step']:07d}.pt"
    checkpoint_sha = native.digest(checkpoint)
    if checkpoint_sha != complete["final_checkpoint_sha256"]:
        raise AssertionError("Endpoint checkpoint changed")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if saved["config_sha256"] != config_sha or saved["world_sha256"] != DATA_HASHES["world_sha256"]:
        raise AssertionError("Checkpoint configuration or data identity changed")
    if saved["step"] != expected_step or saved["handoff"] != ARM or saved["arm"] != ARM:
        raise AssertionError("Wrong checkpoint identity")
    world, panels, metadata = _load_data(config)
    limit_rows = run.get("limited_composition_rows") if smoke else None
    panels = native.limited(panels, limit_rows)
    model = native.initialize(config, ARM, metadata, device)
    if native.backbone_digest(model) != BACKBONE:
        raise AssertionError("Replay initialization does not match the paired backbone")
    initial_model_sha = native.model_digest(model)
    if initial_model_sha != run.get("initial_model_sha256"):
        raise AssertionError("Replay full-model initialization differs from the training run")
    model.load_state_dict(saved["model"], strict=True)
    if native.model_digest(model) != saved["model_sha256"]:
        raise AssertionError("Reloaded model hash changed")
    if complete.get("model_sha256") != saved["model_sha256"]:
        raise AssertionError("Completion record identifies another endpoint model")
    average = native.ParameterAverage(model, config["training"]["ema_decay"])
    average.load_state_dict(saved["ema"])
    model.eval()
    comparisons, queries = {}, 0
    for kind, rounds in (("ema", [4, 8]), ("raw", [4])):
        context = average.applied(model) if kind == "ema" else native._nothing()
        with context:
            replay = native.evaluate_rounds(
                model,
                world,
                panels,
                metadata,
                device,
                config["evaluation_batch_size"],
                rounds,
            )
            for repeats, (_, predictions) in replay.items():
                for panel in native.panel_names():
                    key = f"{kind}-r{repeats}/{panel}"
                    path = out / "predictions" / f"{saved['step']:07d}" / f"{key}.npz"
                    with np.load(path, allow_pickle=False) as original:
                        comparisons[key] = compare_prediction(dict(original), predictions[panel])
                    queries += len(predictions[panel]["answer"])
    if len(comparisons) != 33:
        raise AssertionError("The endpoint replay must cover three weights/rounds x eleven panels")
    result = {
        "passed": True,
        "engineering_only": smoke,
        "independent_process": True,
        "training_pid": run["pid"],
        "audit_pid": os.getpid(),
        "step": expected_step,
        "config_sha256": config_sha,
        "checkpoint_sha256": checkpoint_sha,
        "initial_model_sha256": initial_model_sha,
        "model_sha256": saved["model_sha256"],
        **DATA_HASHES,
        "prediction_files": len(comparisons),
        "queries": queries,
        "exact_fields": ["rows", "answer", "stop"],
        "nll_atol": 1e-6,
        "nll_rtol": 1e-6,
        "comparisons": comparisons,
        "utc": native.utc(),
    }
    native.write_json(out / "reload-audit.json", result)
    return result


def check_training_admission(config, base, config_path):
    """Require engineering evidence from this exact config and immutable smoke endpoint."""
    config_sha = native.digest(config_path)
    evidence = _read(base / "environment.json")
    if evidence.get("passed") is not True or evidence.get("config_sha256") != config_sha:
        raise ValueError("Missing matching CUDA environment check")
    if evidence.get("initial_backbone_sha256") != BACKBONE:
        raise ValueError("CUDA environment check has the wrong paired backbone")
    out = base / "smoke" / ARM
    complete, grid, replay = (
        _read(out / name)
        for name in (
            "complete.json",
            "grid.json",
            "reload-audit.json",
        )
    )
    checkpoint_sha = native.digest(out / f"checkpoint-{config['smoke_steps']:07d}.pt")
    if (
        complete.get("step") != config["smoke_steps"]
        or complete.get("engineering_only") is not True
    ):
        raise ValueError("Full 32-update smoke is required")
    if complete.get("final_checkpoint_sha256") != checkpoint_sha:
        raise ValueError("Smoke checkpoint changed")
    for item in (complete, grid, replay):
        if item.get("config_sha256") != config_sha:
            raise ValueError("Engineering checks used another configuration")
    if any(item.get("checkpoint_sha256") != checkpoint_sha for item in (grid, replay)):
        raise ValueError("Grid/replay did not check this smoke endpoint")
    if replay.get("passed") is not True or replay.get("independent_process") is not True:
        raise ValueError("Independent smoke prediction replay must pass before training")
    if replay.get("prediction_files") != 33 or replay.get("step") != config["smoke_steps"]:
        raise ValueError("Smoke replay coverage is incomplete")


def independent_audit(args, *, smoke):
    command = [
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
    ]
    subprocess.run(command, check=True)


def cloud(config, base, config_path):
    """Check the remote terminal state, exact config, and all local endpoint metrics."""
    import wandb

    out = base / "runs" / ARM
    complete, tracking = _read(out / "complete.json"), _read(out / "tracking.json")
    replay = _read(out / "reload-audit.json")
    config_sha = native.digest(config_path)
    if complete.get("step") != TRAINING["steps"] or complete.get("engineering_only") is not False:
        raise ValueError("Cloud verification requires a scientific endpoint")
    if any(item.get("config_sha256") != config_sha for item in (complete, replay)):
        raise ValueError("Local completion/replay configuration changed")
    if replay.get("passed") is not True or replay.get("prediction_files") != 33:
        raise ValueError("Independent endpoint replay must pass before cloud verification")
    settings = config["tracking"]
    remote = wandb.Api(timeout=60).run(
        f"{settings['entity']}/{settings['project']}/{tracking['id']}"
    )
    if remote.state != "finished" or remote.summary.get("_step") != TRAINING["steps"]:
        raise AssertionError("W&B terminal state or optimizer step differs")
    if remote.config.get("experiment") != config:
        raise AssertionError("W&B experiment configuration differs from the frozen local config")
    expected = native.learning_metrics(complete["endpoint"])
    for key, value in expected.items():
        actual = remote.summary.get(key)
        if not isinstance(actual, (int, float)) or not math.isfinite(actual):
            raise AssertionError(f"Missing or non-finite cloud endpoint metric: {key}")
        if ("accuracy" in key or "coverage" in key) and not 0 <= actual <= 1:
            raise AssertionError(f"Invalid cloud endpoint proportion: {key}")
        if key.endswith(("/nll", "_nll", "_n")) and actual < 0:
            raise AssertionError(f"Invalid negative cloud endpoint statistic: {key}")
        if not math.isclose(actual, value, rel_tol=1e-9, abs_tol=1e-9):
            raise AssertionError(f"Cloud endpoint metric differs: {key}")
    result = {
        "passed": True,
        "state": remote.state,
        "step": remote.summary["_step"],
        "config_exact": True,
        "config_sha256": config_sha,
        "summary_metrics_checked": len(expected),
        "url": tracking["url"],
        "utc": native.utc(),
    }
    native.write_json(out / "cloud-audit.json", result)
    return result


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "stage",
        choices=(
            "environment",
            "smoke",
            "smoke-grid",
            "smoke-audit",
            "train",
            "grid",
            "audit",
            "cloud",
        ),
    )
    result.add_argument("--config", type=Path, required=True)
    result.add_argument("--base", type=Path, required=True)
    result.add_argument("--arm", required=True, choices=(ARM,))
    result.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    result.add_argument("--limit-rows", type=int)
    result.add_argument("--smoke-steps", type=int)
    result.add_argument("--smoke-grid-rounds", type=int)
    result.add_argument("--threads", type=int)
    return result


def validate_arguments(args):
    engineering = (args.limit_rows, args.smoke_steps, args.smoke_grid_rounds, args.threads)
    if any(value is not None for value in engineering) and args.stage not in (
        "smoke",
        "smoke-grid",
    ):
        raise ValueError("Engineering overrides apply to smoke and smoke-grid only")
    if args.device == "cpu" and args.stage in ("environment", "train", "grid", "audit"):
        raise ValueError("The environment and scientific model stages require CUDA")
    if args.smoke_steps is not None and (args.stage != "smoke" or args.smoke_steps != 32):
        raise ValueError("The engineering smoke budget is exactly 32 updates")
    if args.smoke_grid_rounds is not None and args.stage != "smoke-grid":
        raise ValueError("--smoke-grid-rounds applies to smoke-grid only")
    if any(value is not None and value <= 0 for value in engineering):
        raise ValueError("Engineering overrides must be positive")


def main(argv=None):
    args = parser().parse_args(argv)
    validate_arguments(args)
    config = _read(args.config)
    validate_config(config)
    torch.set_num_threads(args.threads or config["runtime"]["cpu_threads"])
    torch.set_num_interop_threads(1)
    if args.stage == "cloud":
        cloud(config, args.base, args.config)
        return
    device = torch.device("cuda:0" if args.device == "cuda" else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(0)
        torch.cuda.set_per_process_memory_fraction(config["runtime"]["gpu_memory_fraction"], 0)
        torch.backends.cuda.matmul.allow_tf32 = True
    try:
        with native_contract():
            if args.stage == "environment":
                environment(config, args.base, args.config, device)
            elif args.stage == "smoke":
                _load_data(config)
                native.train(
                    config,
                    ARM,
                    args.base / "smoke" / ARM,
                    True,
                    args.config,
                    None,
                    device,
                    args.limit_rows or 8,
                )
            elif args.stage == "train":
                check_training_admission(config, args.base, args.config)
                _load_data(config)
                out, holder = args.base / "runs" / ARM, {}

                def track(metadata):
                    holder["log"] = native.tracker_for(config, ARM, out, args.config)
                    return holder["log"]

                try:
                    native.train(config, ARM, out, False, args.config, track, device)
                except BaseException:
                    if "log" in holder:
                        holder["log"].run.finish(exit_code=1)
                    raise
                holder["log"].run.finish()
            elif args.stage in ("smoke-grid", "grid"):
                smoke = args.stage == "smoke-grid"
                run = _read(args.base / ("smoke" if smoke else "runs") / ARM / "run.json")
                if smoke:
                    config = copy.deepcopy(config)
                    config["grid_rounds"] = args.smoke_grid_rounds or 8
                    config["grid_joint_rounds"] = [
                        r for r in config["grid_joint_rounds"] if r <= config["grid_rounds"]
                    ]
                    limit = run.get("limited_composition_rows")
                    if args.limit_rows is not None and args.limit_rows != limit:
                        raise ValueError("Smoke grid must replay the smoke training panels")
                else:
                    limit = None
                _load_data(config)
                native.grid(
                    config,
                    ARM,
                    args.base,
                    args.config,
                    device,
                    "smoke" if smoke else "runs",
                    limit,
                )
                independent_audit(args, smoke=smoke)
            else:
                audit(config, args.base, args.config, device, smoke=args.stage == "smoke-audit")
    except BaseException as error:
        native.write_json(
            args.base / "failures" / f"{ARM}-{args.stage}.json",
            {
                "utc": native.utc(),
                "error": repr(error),
                "traceback": traceback.format_exc(),
            },
        )
        raise


if __name__ == "__main__":
    os.umask(0o022)
    main()
