#!/usr/bin/env python3
"""Causal answer+EOS evaluation of cross-round gates on public dependency tasks.

The native scientific training loop, ordinary CE, sampler, AdamW policy and
CUDA-graph step are reused. Only data, architecture construction, accounting,
task metrics and run naming are adapted. Test panels are evaluated solely at
the registered terminal checkpoint by the separate grid stage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from transformers import GPT2Config

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import run_dual_gate as original  # noqa: E402

from llm_memory_editability import dependency_data as data  # noqa: E402
from llm_memory_editability.experiment_tracking import learning_metrics  # noqa: E402
from llm_memory_editability.grok_depth import utc, write_json  # noqa: E402
from llm_memory_editability.grokking_reproduction import digest, model_digest  # noqa: E402
from llm_memory_editability.residual_gate_injection import ResidualGateInjectionGPT  # noqa: E402
from llm_memory_editability.supervised_alignment import SupervisedAlignmentGPT  # noqa: E402

native = original.native
optimizer_for = original.optimizer_for
TRAIN_PANELS = ("train_sample", "valid_iid", "valid_depth")
TEST_PANELS = ("test_iid", "test_depth")
ALL_PANELS = TRAIN_PANELS + TEST_PANELS
ARMS = {
    "loop": ("none", "full"),
    "only-u": ("none", "token"),
    "disco": ("fixed1", "full"),
    "fixed-q-plus-u": ("fixed1", "token"),
}
TASKS = {
    "listops": {
        "repeats": 20,
        "lr": 2e-4,
        "weight_decay": 0.05,
        "inference_repeats": [1, 4, 8, 12, 20, 24],
    },
    "arithmetic": {
        "repeats": 15,
        "lr": 1.5e-4,
        "weight_decay": 0.01,
        "inference_repeats": [1, 4, 8, 15, 20],
    },
}
NODES = [0, 1000, 5000, 10000, 25000, 50000, 75000, 100000]


def resolve_config(root_config, task):
    """Materialize a native-compatible task configuration without mutating input."""
    if task not in TASKS or task not in root_config.get("tasks", {}):
        raise ValueError(f"Unregistered dependency task: {task}")
    selected = root_config["tasks"][task]
    expected_declarations = {
        name: {
            "architecture": "loop" if injection == "none" else "discoloop",
            "unique_layers": 1,
            "injection": injection,
            "update_mode": mode,
            "initial_update": 0.95,
        }
        for name, (injection, mode) in ARMS.items()
    }
    if root_config.get("arms", expected_declarations) != expected_declarations:
        raise ValueError("Root configuration changes the registered four architecture arms")
    result = {key: value for key, value in root_config.items() if key != "tasks"}
    result.update(selected)
    result["task"] = task
    result["runtime"] = {**root_config.get("runtime", {}), **selected.get("runtime", {})}
    result["arms"] = {
        name: {
            "architecture": "loop" if injection == "none" else "discoloop",
            "unique_layers": 1,
            "repeats": result["repeats"],
            "injection": injection,
            "update_mode": mode,
            "initial_update": 0.95,
        }
        for name, (injection, mode) in ARMS.items()
    }
    return result


def validate_config(config):
    task = config["task"]
    if task not in TASKS:
        raise ValueError("Only the registered ListOps/arithmetic tasks are supported")
    expected = TASKS[task]
    if config["repeats"] != expected["repeats"]:
        raise ValueError("Training recurrence differs from the registered task")
    if config["inference_repeats"] != expected["inference_repeats"]:
        raise ValueError("The complete registered endpoint inference grid is required")
    if config["evaluation_nodes"] != NODES or config["heartbeat_steps"] != 1000:
        raise ValueError("Training evaluation nodes/heartbeat differ from the registered recipe")
    model = config["model"]
    for key, value in {
        "n_embd": 256,
        "n_head": 4,
        "attn_pdrop": 0.1,
        "embd_pdrop": 0.1,
        "resid_pdrop": 0.1,
    }.items():
        if model.get(key) != value:
            raise ValueError(f"Registered model setting changed: {key}")
    if model.get("n_inner", 4 * model["n_embd"]) != 1024:
        raise ValueError("MLP hidden width must remain 1024")
    if model.get("n_layer", 1) != 1:
        raise ValueError("Each round must have exactly one shared block")
    for key, value in {
        "steps": 100000,
        "batch_size": 512,
        "warmup": 2000,
        "lr": expected["lr"],
        "weight_decay": expected["weight_decay"],
    }.items():
        if config["training"].get(key) != value:
            raise ValueError(f"Registered ordinary training setting changed: {key}")
    for key in ("initialization", "stream_seed"):
        if not isinstance(config["training"].get(key), int):
            raise ValueError(f"Explicit training seed required: {key}")
    if not isinstance(config.get("dropout_seed"), int):
        raise ValueError("An explicit paired dropout seed is required")
    if set(config["arms"]) != set(ARMS):
        raise ValueError("Exactly the four registered architecture arms are required")
    for name, (injection, mode) in ARMS.items():
        arm = config["arms"][name]
        if arm != {
            "architecture": "loop" if injection == "none" else "discoloop",
            "unique_layers": 1,
            "repeats": expected["repeats"],
            "injection": injection,
            "update_mode": mode,
            "initial_update": 0.95,
        }:
            raise ValueError(f"Registered architecture changed: {name}")
    if config.get("training_panels", list(TRAIN_PANELS)) != list(TRAIN_PANELS):
        raise ValueError("Test panels may not enter training-time evaluation")
    if config.get("endpoint_panels", list(ALL_PANELS)) != list(ALL_PANELS):
        raise ValueError("Endpoint evaluation must retain every registered panel")
    policy = config.get("evaluation_policy", {})
    if policy.get("during_training", list(TRAIN_PANELS)) != list(TRAIN_PANELS):
        raise ValueError("Root evaluation policy cannot add test panels during training")
    if policy.get("endpoint_only", list(TEST_PANELS)) != list(TEST_PANELS):
        raise ValueError("Root policy must reserve both test panels for the endpoint")


def verify_input(config):
    world, panels, metadata = data.load(config["input_root"])
    if metadata["task"] != config["task"]:
        raise ValueError("Mounted input task differs from configuration")
    if set(panels) != set(ALL_PANELS):
        raise ValueError("Mounted panels differ from the registered train/validation/test split")
    if len(world["tokens"]) != metadata["training_examples"]:
        raise ValueError("Training table count differs from metadata")
    if metadata["sequence_length"] > config["model"]["n_positions"]:
        raise ValueError("Data exceed the model context; truncation is not allowed")
    for key in ("world_sha256", "panels_sha256"):
        expected = config.get("input_origin", {}).get(key)
        if expected is not None and metadata[key] != expected:
            raise ValueError(f"Pinned data changed: {key}")
    return metadata


def initialize(config, name, metadata, device):
    spec, arm = native.arm_spec(config, name)
    options = {
        key: config["arms"][name][key] for key in ("injection", "update_mode", "initial_update")
    }
    arm = {
        **arm,
        **options,
        "task": config["task"],
        "gate_scope": "continuous_update_only_nonfinal_all_tokens",
    }
    torch.manual_seed(spec["initialization"])
    hf = GPT2Config(
        **{
            **config["model"],
            "n_layer": spec["unique_layers"],
            "vocab_size": metadata["vocab_size"],
            "bos_token_id": metadata.get("bos_token", 0),
            "eos_token_id": metadata["end_marker"],
            "pad_token_id": metadata["pad_token"],
            "use_cache": False,
        }
    )
    model = ResidualGateInjectionGPT(
        hf, spec["repeats"], **options, **native.architecture_flags(arm)
    ).to(device)
    initial = model_digest(model)
    torch.manual_seed(config["dropout_seed"])
    return model, spec, arm, initial


def accounting(model, spec, arm, metadata):
    return {
        **original.original_accounting(model, spec, arm, metadata),
        "task": arm["task"],
        "injection": model.injection,
        "update_mode": model.update_mode,
        "initial_update": model.initial_update,
        "paper_gate_parameters": 0,
        "update_gate_parameters": model.config.n_embd + 1 if model.update_mode == "token" else 0,
        "gate_scope": arm["gate_scope"],
        "boundary": "X + u(H) * (H-X)" + (" + q(H)" if model.discoloop else ""),
        "final_readout": "raw H; no final boundary",
        "supervised_positions_per_example": 2,
        "padded_sequence_length": metadata["sequence_length"],
        "primary_score": "full-vocabulary answer_accuracy",
        "flops_note": f"Dense padded T={metadata['sequence_length']}, two supervised "
        "output positions, actual vocabulary, training factor3; includes both "
        "differentiable vocabulary projections when enabled. Excludes softmax, "
        "normalization, dropout, pointwise gates and memory traffic; no early exit.",
    }


def _scores(answer, stop, nll, labels, end_marker):
    correct = answer == labels[:, 0]
    return {
        "n": len(answer),
        "correct": int(correct.sum()),
        "answer_accuracy": float(correct.mean()) if len(answer) else None,
        "accuracy": float((correct & (stop == end_marker)).mean()) if len(answer) else None,
        "answer_nll": float(nll[:, 0].mean()) if len(answer) else None,
        "nll": float(nll.mean()) if len(answer) else None,
    }


@torch.no_grad()
def evaluate(model, panel, metadata, device, batch_size):
    """Full-vocabulary digit prediction, followed by EOS on the generated digit."""
    was_training = model.training
    model.eval()
    count = len(panel["tokens"])
    if count < 1:
        raise ValueError("Registered evaluation panels must be nonempty")
    answers, stops, losses = [], [], []
    try:
        for start in range(0, count, batch_size):
            section = slice(start, start + batch_size)
            x, positions, labels = [
                torch.as_tensor(panel[key][section], device=device, dtype=torch.long)
                for key in ("tokens", "positions", "labels")
            ]
            x = x.clone()  # A CPU evaluation must not mutate the saved data view.
            if labels.shape != positions.shape or labels.shape[1] != 2:
                raise ValueError("Dependency tasks require exactly answer and EOS labels")
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(x, positions)
                nll = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), reduction="none")
                answer = logits[:, 0].argmax(-1)
                x[torch.arange(len(x), device=device), positions[:, 1]] = answer
                stop = model(x, positions)[:, 1].argmax(-1)
            answers.append(answer.cpu().numpy())
            stops.append(stop.cpu().numpy())
            losses.append(nll.view(-1, 2).float().cpu().numpy())
    finally:
        model.train(was_training)
    answer, stop, nll = np.concatenate(answers), np.concatenate(stops), np.concatenate(losses)
    labels = np.asarray(panel["labels"])
    metrics = _scores(answer, stop, nll, labels, metadata["end_marker"])
    for feature in (
        "depth",
        "tree_depth",
        "length",
        "branch_imbalance",
        "computed_branch_imbalance",
    ):
        values = np.asarray(panel[feature])
        if values.shape != (count,) or not np.issubdtype(values.dtype, np.integer):
            raise ValueError(f"Registered grouping must be an integer per example: {feature}")
        metrics[f"by_{feature}"] = {
            str(value): _scores(
                answer[mask], stop[mask], nll[mask], labels[mask], metadata["end_marker"]
            )
            for value in np.unique(values)
            for mask in [values == value]
        }
    joins = np.asarray(panel["join_count"])
    if joins.shape != (count,) or not np.issubdtype(joins.dtype, np.integer) or np.any(joins < 0):
        raise ValueError("join_count must be a nonnegative integer per example")
    metrics["by_join_presence"] = {
        key: _scores(answer[mask], stop[mask], nll[mask], labels[mask], metadata["end_marker"])
        for key, mask in (("0", joins == 0), ("ge1", joins >= 1))
    }
    predictions = {
        key: np.asarray(value).copy()
        for key, value in panel.items()
        if key not in {"tokens", "positions"}
    }
    predictions.update(answer=answer, stop=stop, nll=nll)
    return metrics, predictions


def measure_panels(model, world, panels, metadata, device, batch_size, names=TRAIN_PANELS):
    del world
    metrics, predictions = {}, {}
    for name in names:
        metrics[name], predictions[name] = evaluate(
            model, panels[name], metadata, device, batch_size
        )
    return metrics, predictions


def _forward_gradients(model, table, indices, seed):
    model.train()
    model.zero_grad(set_to_none=True)
    torch.manual_seed(seed)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(table[0][indices], table[1][indices])
        loss = F.cross_entropy(logits.flatten(0, 1), table[2][indices].flatten())
    loss.backward()
    return logits.detach(), {
        name: p.grad.detach().clone() for name, p in model.named_parameters() if p.grad is not None
    }


def preflight(config, name, base, config_path):
    """Actual full-size CUDA comparison and one discarded restored graph replay."""
    destination = base / "preflight" / f"{name}.json"
    if destination.exists():
        raise FileExistsError(destination)
    world, _, metadata = data.load(config["input_root"])
    device = torch.device("cuda:0")
    model, spec, arm, initial = initialize(config, name, metadata, device)
    reference = SupervisedAlignmentGPT(
        model.config, spec["repeats"], **native.architecture_flags(arm)
    ).to(device)
    reference.load_state_dict(
        {key: value for key, value in model.state_dict().items() if not key.startswith("gate_")},
        strict=True,
    )
    backbone_hash = model_digest(reference)
    if model.update_mode == "token":
        del reference
        reference = ResidualGateInjectionGPT(
            model.config,
            spec["repeats"],
            injection=model.injection,
            update_mode="fixed",
            initial_update=0.95,
        ).to(device)
        reference.load_state_dict(model.state_dict(), strict=True)
    table = data.make_table(world, metadata, device)
    stream = native.MergedBatchStream(len(table[0]) - 1, spec["stream_seed"])
    indices = torch.as_tensor(stream.batch(spec["batch_size"])[0], device=device)
    first, first_grad = _forward_gradients(reference, table, indices, config["dropout_seed"])
    second, second_grad = _forward_gradients(model, table, indices, config["dropout_seed"])
    gates = {n: value for n, value in second_grad.items() if n.startswith("gate_update_")}
    findings = {
        "reference_forward_exact": torch.equal(first, second),
        "all_common_gradients_exact": set(first_grad) <= set(second_grad)
        and all(torch.equal(value, second_grad[n]) for n, value in first_grad.items()),
        "gate_gradients_finite_and_nonzero": (
            len(gates) == 2
            and all(
                torch.isfinite(g).all().item() and torch.count_nonzero(g).item() > 0
                for g in gates.values()
            )
        )
        if model.update_mode == "token"
        else not gates,
        "no_learned_injection_gate": not any(
            n.startswith("gate_paper_") for n in model.state_dict()
        ),
    }
    del reference, first, second, first_grad, second_grad, gates
    model.zero_grad(set_to_none=True)
    lr = torch.tensor(0.0, device=device)
    optimizer = optimizer_for(model, lr, spec["weight_decay"])
    gate_ids = {id(p) for n, p in model.named_parameters() if n.startswith("gate_")}
    findings["all_gate_parameters_have_zero_decay"] = all(
        group["weight_decay"] == 0
        for group in optimizer.param_groups
        if any(id(p) in gate_ids for p in group["params"])
    )
    cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
    graph = native.AlignmentStep(model, optimizer, table, spec["batch_size"], arm)
    findings.update(
        capture_restores_weights=model_digest(model) == initial,
        capture_restores_cpu_rng=torch.equal(cpu_rng, torch.get_rng_state()),
        capture_restores_cuda_rng=torch.equal(cuda_rng, torch.cuda.get_rng_state()),
        capture_restores_adam=all(
            not torch.count_nonzero(v)
            for state in optimizer.state.values()
            for v in state.values()
            if torch.is_tensor(v)
        ),
    )
    saved = {n: p.detach().clone() for n, p in model.state_dict().items()}
    lr.fill_(spec["lr"])
    graph(indices)
    torch.cuda.synchronize()
    findings["captured_update_finite"] = all(
        np.isfinite(value) for value in native.graph_values(graph, 0).values()
    )
    findings["captured_update_nonzero"] = model_digest(model) != initial
    model.load_state_dict(saved, strict=True)
    for state in optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value):
                value.zero_()
    lr.zero_()
    model.zero_grad(set_to_none=True)
    torch.set_rng_state(cpu_rng)
    torch.cuda.set_rng_state(cuda_rng)
    findings["engineering_update_restored"] = model_digest(model) == initial
    write_json(
        destination,
        {
            "passed": all(findings.values()),
            "findings": findings,
            "task": config["task"],
            "arm": name,
            "engineering_only": True,
            "engineering_updates_executed": 1,
            "engineering_updates_retained": 0,
            "capture_preparation_restored_optimizer_calls": 4,
            "config_sha256": digest(config_path),
            "world_sha256": metadata["world_sha256"],
            "panels_sha256": metadata["panels_sha256"],
            "initial_model_sha256": initial,
            "initial_backbone_sha256": backbone_hash,
            "accounting": accounting(model, spec, arm, metadata),
            "utc": utc(),
        },
    )
    if not all(findings.values()):
        raise AssertionError(findings)


@contextmanager
def installed():
    """Change no global training function and restore every adapted callback."""
    replacements = {
        "initialize": initialize,
        "accounting": accounting,
        "optimizer_for": optimizer_for,
        "load": data.load,
        "make_table": data.make_table,
        "exposure_summary": data.exposure_summary,
        "measure_panels": measure_panels,
    }
    saved = {key: getattr(native, key) for key in replacements}
    try:
        for key, value in replacements.items():
            setattr(native, key, value)
        yield
    finally:
        for key, value in saved.items():
            setattr(native, key, value)


def tracked_train(config, name, base, config_path):
    """Native W&B callback pattern with a task-qualified name and unique run ID."""
    import wandb

    original_write, run, previous = native.write_json, None, None

    def publish(destination, value):
        nonlocal run, previous
        filename = Path(destination).name
        if filename == "complete.json" or (
            filename == "status.json" and value.get("state") == "complete"
        ):
            value = {
                **value,
                "independent_reload": "not_requested",
                "test_evaluation": "pending_fixed_endpoint_grid",
            }
        original_write(destination, value)
        if filename == "run.json":
            identity = f"{digest(config_path)}:{config['task']}:{name}"
            run = wandb.init(
                entity="zhusq20",
                project="llm-memory-editability",
                group=config["batch"],
                name=f"{config['task']}-{name}",
                id="dg" + hashlib.sha256(identity.encode()).hexdigest()[:14],
                config={"experiment": config, "run": value},
                dir=str(Path(destination).parent),
                mode=config["runtime"].get("tracking", "online"),
                save_code=False,
                settings=wandb.Settings(disable_git=True),
            )
            original_write(
                Path(destination).parent / "tracking.json", {"id": run.id, "url": run.url}
            )
        elif filename == "learning.json" and run is not None:
            record = value[-1]
            if set(record["metrics"]) != set(TRAIN_PANELS):
                raise AssertionError("Training-time metrics contain an unregistered panel")
            run.log(learning_metrics(record, previous), step=record["step"])
            previous = record
        elif filename == "status.json" and run is not None:
            run.summary["training_state"] = value["state"]
            run.summary["scientific_final_step"] = value["step"]

    native.write_json = publish
    try:
        native.train(config, name, base / "runs" / name, False, config_path, base)
    except BaseException:
        if run is not None:
            run.finish(exit_code=1)
        raise
    else:
        if run is not None:
            run.finish()
    finally:
        native.write_json = original_write


@torch.no_grad()
def sweep(model, world, panels, metadata, device, batch_size, repeats):
    original_repeats, was_training = model.repeats, model.training
    before = model_digest(model)
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    metrics, predictions = {}, {}
    try:
        for count in repeats:
            model.repeats = count
            metrics[str(count)], predictions[str(count)] = measure_panels(
                model, world, panels, metadata, device, batch_size, names=ALL_PANELS
            )
    finally:
        model.repeats = original_repeats
        model.train(was_training)
    if model_digest(model) != before or not torch.equal(cpu_rng, torch.get_rng_state()):
        raise AssertionError("Endpoint grid changed weights or CPU RNG")
    if cuda_rng is not None and not torch.equal(cuda_rng, torch.cuda.get_rng_state(device)):
        raise AssertionError("Endpoint grid consumed CUDA RNG")
    return metrics, predictions


def endpoint(config, name, base, config_path):
    out = base / "runs" / name
    destination = out / "grid.json"
    if destination.exists():
        raise FileExistsError(destination)
    complete = json.loads((out / "complete.json").read_text())
    if (
        complete["state"] != "complete"
        or complete["step"] != config["training"]["steps"]
        or complete.get("budget") != config["training"]["steps"]
        or complete.get("arm") != name
        or complete.get("engineering_only") is not False
        or complete.get("config_sha256") != digest(config_path)
    ):
        raise ValueError("Test evaluation requires the fixed registered terminal checkpoint")
    saved_path = out / f"checkpoint-{complete['step']:07d}.pt"
    saved = torch.load(saved_path, map_location="cpu", weights_only=False)
    world, panels, metadata = data.load(config["input_root"])
    device = torch.device("cuda:0")
    model, spec, arm, initial = initialize(config, name, metadata, device)
    if (
        saved["config_sha256"] != digest(config_path)
        or saved["world_sha256"] != metadata["world_sha256"]
        or saved["spec"] != spec
        or saved["arm"] != arm
        or saved["step"] != complete["step"]
        or digest(saved_path) != complete["final_checkpoint_sha256"]
    ):
        raise ValueError("Terminal checkpoint identity mismatch")
    model.load_state_dict(saved["model"], strict=True)
    if model_digest(model) != complete["model_sha256"]:
        raise ValueError("Terminal parameter hash mismatch")
    metrics, predictions = sweep(
        model,
        world,
        panels,
        metadata,
        device,
        config["evaluation_batch_size"],
        config["inference_repeats"],
    )
    for count, per_panel in predictions.items():
        folder = out / "grid" / f"r{count}"
        folder.mkdir(parents=True, exist_ok=False)
        for panel, values in per_panel.items():
            np.savez_compressed(folder / f"{panel}.npz", **values)
    write_json(
        destination,
        {
            "task": config["task"],
            "arm": name,
            "step": saved["step"],
            "training_repeats": spec["repeats"],
            "unique_layers": spec["unique_layers"],
            "inference_repeats": config["inference_repeats"],
            "world_sha256": metadata["world_sha256"],
            "panels_sha256": metadata["panels_sha256"],
            "config_sha256": digest(config_path),
            "model_sha256": model_digest(model),
            "initial_model_sha256": initial,
            "metrics": metrics,
            "test_policy": "Only fixed endpoint; no test-based checkpoint or horizon selection",
            "engineering_only": False,
            "independent_reload_audit": False,
            "utc": utc(),
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("preflight", "train", "grid"), required=True)
    parser.add_argument("--task", choices=tuple(TASKS), required=True)
    parser.add_argument("--arm", choices=tuple(ARMS), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    args = parser.parse_args()
    config = resolve_config(json.loads(args.config.read_text()), args.task)
    validate_config(config)
    verify_input(config)
    torch.set_num_threads(config["runtime"]["cpu_threads"])
    torch.set_num_interop_threads(1)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(config["runtime"]["gpu_memory_fraction"], 0)
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        with installed():
            if args.stage == "preflight":
                preflight(config, args.arm, args.base, args.config)
            elif args.stage == "train":
                tracked_train(config, args.arm, args.base, args.config)
            else:
                endpoint(config, args.arm, args.base, args.config)
    except BaseException as error:
        import traceback

        write_json(
            args.base / "failures" / f"{args.arm}-{args.stage}.json",
            {
                "task": args.task,
                "arm": args.arm,
                "stage": args.stage,
                "error": repr(error),
                "traceback": traceback.format_exc(),
                "utc": utc(),
            },
        )
        raise


if __name__ == "__main__":
    os.umask(0o022)
    main()
