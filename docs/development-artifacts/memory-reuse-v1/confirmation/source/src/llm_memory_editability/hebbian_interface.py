"""Paired CE/MSE memory interfaces on the released synthetic recall task.

The official implementation supplies memory fitting, the reader, and batches.
This adapter fixes budgets and seeds, removes the future answer from inputs,
and records a complete crossing without selecting on the replacement mapping.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPEC = {
    "num_facts": 128,
    "d_model": 64,
    "hidden_dim": 256,
    "junk_len": 9,
    "junk_vocab_size": 9,
    "batch_size": 256,
    "mlp_epochs": 10000,
    "mlp_lr": 0.001,
    "mlp_min_lr": 1e-6,
    "mlp_cutoff": -1.0,
    "mlp_activation": "swish",
    "mlp_bias": True,
    "reader_steps": 4000,
    "reader_lr": 2e-4,
    "reader_weight_decay": 0.1,
    "eval_every": 1000,
    "eval_repeats": 8,
    "cpu_threads": 1,
    "source_path": "data/hebbian-interface-v1/source",
}


def _source(spec):
    source = ROOT / spec["source_path"] / "src"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))


def tensor_hash(tensors):
    """Hash names, shapes, dtypes and raw values, independently of serialization."""
    digest = hashlib.sha256()
    for name, value in sorted(tensors.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(f"{name}:{tuple(tensor.shape)}:{tensor.dtype}".encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _state(model, *, non_mlp=False, frozen=False):
    return {
        name: value
        for name, value in model.named_parameters()
        if (not non_mlp or ".mlp." not in name) and (not frozen or not value.requires_grad)
    }


def _memory_config(spec, loss, device):
    from hebbian.mlp_core.mlp_gd import GDMLPConfig
    from hebbian.mlp_core.task import SharedConstructionConfig

    shared = SharedConstructionConfig(
        build_dtype=torch.float32,
        final_dtype=torch.float32,
        device=str(device),
        verbose=False,
    )
    shared.mlp_config.activation.activation = spec["mlp_activation"]
    return GDMLPConfig(
        shared=shared,
        m=spec["hidden_dim"],
        bias=spec["mlp_bias"],
        num_epochs=spec["mlp_epochs"],
        lr=spec["mlp_lr"],
        min_lr=spec["mlp_min_lr"],
        cutoff=spec["mlp_cutoff"],
        loss_fn=loss,
        batch_size=None,
        eval_every=spec["eval_every"],
    )


def _new_memory(spec, device):
    from hebbian.mlp_core.mlp_gd import _create_gd_mlp

    return _create_gd_mlp(spec["d_model"], spec["hidden_dim"], _memory_config(spec, "ce", device))


def make_config(spec, seed, device):
    """The fixed SSFR architecture; memory and optimizer numbers come from spec."""
    _source(spec)
    from hebbian.transformer.config import AssociativeRecallConfig

    config = AssociativeRecallConfig()
    dc, tc = config.dataset_config, config.train_config
    dc.num_facts = spec["num_facts"]
    dc.junk_vocab_size = spec["junk_vocab_size"]
    dc.min_seq_length = dc.max_seq_length = spec["junk_len"]
    dc.custom_finalize()
    tc.device, tc.dtype, tc.seed = str(device), torch.float32, seed
    tc.embeddings_config.d_model = spec["d_model"]
    tc.embeddings_config.tie_embeddings = True
    tc.mlp_method, tc.mlp_hidden_dim = "gd", spec["hidden_dim"]
    tc.batch_size, tc.steps_per_dataset = spec["batch_size"], spec["reader_steps"]
    arch = tc.transformer_config
    arch.n_layers = arch.n_head = 1
    arch.use_identity_mlp = True
    arch.mlp_residual = arch.attn_residual = False
    arch.use_rope, arch.no_positional_encoding = False, True
    arch.freeze_value_dense_identity = True
    arch.mlp_norm_type = arch.lm_head_norm_type = "unit_rmsnorm"
    arch.attn_norm_type, arch.bias = "rmsnorm", False
    return config


def make_batches(spec, mapping, *, batch_size=None, num_batches=None):
    """Author batches on CPU avoid a GPU synchronization per junk position."""
    from hebbian.transformer.data import AssociativeRecallBatchGenerator

    return AssociativeRecallBatchGenerator(
        mapping,
        spec["num_facts"],
        spec["junk_vocab_size"],
        spec["junk_len"],
        spec["junk_len"],
        batch_size or spec["batch_size"],
        num_batches or spec["reader_steps"],
        device=torch.device("cpu"),
    )


def fixed_inputs(spec, mapping, seed):
    """Balanced keys and independently seeded junk, with the answer removed."""
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        keys = torch.arange(spec["num_facts"]).repeat_interleave(spec["eval_repeats"])
        inputs, _ = next(iter(make_batches(spec, mapping, batch_size=len(keys), num_batches=1)))
        inputs[:, spec["junk_len"]] = keys
    return inputs[:, :-1].contiguous(), keys


def _errors(actual, expected):
    relative = (actual - expected).norm(dim=-1) / expected.norm(dim=-1).clamp_min(1e-12)
    cosine = F.cosine_similarity(actual, expected, dim=-1).clamp(-1, 1)
    return relative, cosine.acos()


def _accuracy(scores, targets, changed):
    correct = scores.argmax(-1).eq(targets)
    return {
        "accuracy_all": correct.float().mean().item(),
        "accuracy_changed": correct[changed].float().mean().item() if changed.any() else None,
        "n_all": targets.numel(),
        "n_changed": int(changed.sum()),
    }


def _save_arrays(path, **arrays):
    np.savez_compressed(
        path,
        **{
            name: value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value
            for name, value in arrays.items()
        },
    )


@torch.no_grad()
def standalone(memory, factset, embeddings, changed, path):
    outputs = memory(factset.input_embeddings)
    targets = torch.tensor(factset.mapping.outputs, device=outputs.device)
    expected = factset.output_embeddings[targets]
    relative, angle = _errors(outputs, expected)
    normalized = F.normalize(outputs, dim=-1)
    fact_scores = normalized @ factset.output_embeddings.T
    full_scores = normalized @ embeddings.weight.T
    norm_error = (outputs.norm(dim=-1) - expected.norm(dim=-1)).abs()
    result = {
        "fact_accuracy": fact_scores.argmax(-1).eq(targets).float().mean().item(),
        "full_accuracy": full_scores.argmax(-1).eq(targets).float().mean().item(),
        "relative_error_mean": relative.mean().item(),
        "angle_radians_mean": angle.mean().item(),
        "norm_error_mean": norm_error.mean().item(),
        "output_norm_mean": outputs.norm(dim=-1).mean().item(),
        "full_vocab": _accuracy(full_scores, targets, changed.to(outputs.device)),
        "artifact": path.name,
    }
    _save_arrays(
        path,
        key=torch.arange(len(targets)),
        target=targets,
        changed_mapping=changed,
        pred=full_scores.argmax(-1),
        pred_fact=fact_scores.argmax(-1),
        scores=full_scores,
        scores_fact=fact_scores,
        scores_fact_raw=outputs @ factset.output_embeddings.T,
        relative_error=relative,
        angle_radians=angle,
        norm_error=norm_error,
        output_norm=outputs.norm(dim=-1),
    )
    return result


@torch.no_grad()
def evaluate_reader(model, inputs, keys, targets, changed, embeddings, batch_size, path=None):
    """Use actual query-position inputs to the MLP; never fit a predictor."""
    device = next(model.parameters()).device
    cuda_devices = [device.index or 0] if device.type == "cuda" else []
    scores, queries = [], []
    was_training = model.training
    model.eval()
    hook = model.transformer.h[0].mlp.register_forward_pre_hook(
        lambda module, args: queries.append(args[0][:, -1].detach().cpu())
    )
    try:
        with torch.random.fork_rng(devices=cuda_devices):
            for batch in inputs.split(batch_size):
                logits, _ = model(batch.to(device))
                scores.append(logits[:, 0].cpu())
    finally:
        hook.remove()
        model.train(was_training)
    scores, queries = torch.cat(scores), torch.cat(queries)
    relative, angle = _errors(queries, embeddings.weight.detach().cpu()[keys])
    result = _accuracy(scores, targets, changed)
    result.update(
        {
            "query_relative_error_mean": relative.mean().item(),
            "query_angle_radians_mean": angle.mean().item(),
        }
    )
    if path is not None:
        _save_arrays(
            path,
            key=keys,
            target=targets,
            changed_mapping=changed,
            pred=scores.argmax(-1),
            scores=scores,
            query_input=queries,
            query_relative_error=relative,
            query_angle_radians=angle,
        )
        result["artifact"] = path.name
    return result


def load_checkpoint(path, device="cpu"):
    """Load our tensor-only checkpoint; no refitting or custom-object unpickling."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    spec = payload["spec"]
    _source(spec)
    if payload["kind"] == "memory":
        model = _new_memory(spec, device)
    else:
        from hebbian.transformer.model import GPT, GPTConfig

        model = GPT(GPTConfig(**payload["gpt_config"])).to(device)
        model.transformer.h[0].mlp = _new_memory(spec, device)
    model.load_state_dict(payload["state_dict"], strict=True)
    names = set(payload.get("trainable_names", []))
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in names)
    return model.eval(), payload


def run_world(spec, seed, output_dir, device):
    """Fit four memories and two A-readers; evaluate all fixed final crossings."""
    spec = {**DEFAULT_SPEC, **spec}
    if (
        min(
            spec[k]
            for k in (
                "num_facts",
                "d_model",
                "hidden_dim",
                "mlp_epochs",
                "reader_steps",
                "eval_every",
                "eval_repeats",
            )
        )
        <= 0
    ):
        raise ValueError("Dimensions and fixed training/evaluation budgets must be positive")
    if spec["mlp_cutoff"] >= 0:
        raise ValueError("This comparison requires the MLP loss cutoff to be disabled")
    _source(spec)
    from hebbian.transformer.fact_store import build_fact_mlp, build_factset, build_token_embeddings
    from hebbian.transformer.model import GPT
    from hebbian.transformer.train import _make_eval_factset
    from hebbian.transformer.utils import (
        copy_embeddings_to_gpt,
        create_gpt_config,
        insert_mlp_into_gpt,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device)
    torch.set_num_threads(spec["cpu_threads"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    config = make_config(spec, seed, device)
    factsets = {"A": build_factset(config, seed=seed)}
    factsets["B"] = _make_eval_factset(factsets["A"], seed=seed + 7777)
    embeddings = build_token_embeddings(
        factsets["A"],
        spec["junk_vocab_size"],
        embedding_init="spherical",
        dtype=torch.float32,
        seed=seed,
    ).to(device)
    mappings = {name: torch.tensor(fs.mapping.outputs) for name, fs in factsets.items()}
    changed = mappings["A"] != mappings["B"]
    inputs, keys = fixed_inputs(spec, factsets["A"].mapping, seed + 20000)
    _save_arrays(
        output_dir / "world.npz",
        inputs=inputs,
        key=keys,
        mapping_A=mappings["A"],
        mapping_B=mappings["B"],
        changed_mapping=changed,
        embeddings=embeddings.weight,
    )
    summary = {
        "seed": seed,
        "spec": spec,
        "state": "running",
        "memories": {},
        "readers": {},
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "transformers": importlib.metadata.version("transformers"),
            "cuda": torch.version.cuda,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
        "world": {
            "changed_facts": int(changed.sum()),
            "facts": spec["num_facts"],
            "test_inputs_hash": tensor_hash({"inputs": inputs}),
            "mapping_A": mappings["A"].tolist(),
            "mapping_B": mappings["B"].tolist(),
        },
    }

    def persist():
        temporary = output_dir / "summary.tmp"
        temporary.write_text(json.dumps(summary, indent=2) + "\n")
        temporary.replace(output_dir / "summary.json")

    memories = {}
    for loss in ("ce", "mse"):
        memories[loss], summary["memories"][loss] = {}, {}
        for name, factset in factsets.items():
            stage = time.perf_counter()
            torch.manual_seed(seed)
            initial = _new_memory(spec, device)
            initial_hash = tensor_hash(initial.state_dict())
            del initial
            memory, metrics = build_fact_mlp(
                config, factset, method_config=_memory_config(spec, loss, device)
            )
            memory.eval().requires_grad_(False)
            memories[loss][name] = memory
            checkpoint = f"memory-{loss}-{name}.pt"
            torch.save(
                {
                    "kind": "memory",
                    "spec": spec,
                    "seed": seed,
                    "loss": loss,
                    "mapping": name,
                    "state_dict": {k: v.detach().cpu() for k, v in memory.state_dict().items()},
                },
                output_dir / checkpoint,
            )
            _save_arrays(
                output_dir / f"memory-{loss}-{name}-curve.npz",
                train_loss=np.asarray(metrics["train_losses"]),
            )
            assert len(metrics["train_losses"]) == spec["mlp_epochs"]
            summary["memories"][loss][name] = {
                "initial_hash": initial_hash,
                "final_hash": tensor_hash(memory.state_dict()),
                "seconds": time.perf_counter() - stage,
                "checkpoint": checkpoint,
                "training_final_loss": metrics["train_losses"][-1],
                "training_final_fact_accuracy": metrics["final_accuracy"],
            }
            print(f"world {seed}: memory {loss}/{name} complete", flush=True)
            persist()

    models = {}
    for loss in ("ce", "mse"):
        stage = time.perf_counter()
        torch.manual_seed(seed)
        gpt_config = create_gpt_config(config.train_config, config.dataset_config)
        model = GPT(gpt_config).to(device=device, dtype=torch.float32).eval()
        copy_embeddings_to_gpt(model, embeddings)
        insert_mlp_into_gpt(model, memories[loss]["A"], embeddings)
        initial_hash = tensor_hash(_state(model, non_mlp=True))
        frozen_before = tensor_hash(_state(model, frozen=True))
        optimizer = model.configure_optimizers(
            weight_decay=spec["reader_weight_decay"],
            learning_rate=spec["reader_lr"],
            betas=(0.9, 0.999),
            device_type=device.type,
        )
        curve, stream_hash, train_loss_sum = [], hashlib.sha256(), 0.0
        torch.manual_seed(seed + 10000)
        for step, (batch, labels) in enumerate(make_batches(spec, factsets["A"].mapping), 1):
            batch, labels = batch[:, :-1].contiguous(), labels[:, :-1].contiguous()
            stream_hash.update(batch.numpy().tobytes())
            stream_hash.update(labels[:, -1].numpy().tobytes())
            model.train()
            logits, _ = model(batch.to(device))
            objective = F.cross_entropy(logits[:, 0], labels[:, -1].to(device))
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            optimizer.step()
            train_loss_sum += objective.item()
            if step % spec["eval_every"] == 0 or step == spec["reader_steps"]:
                evaluation = evaluate_reader(
                    model,
                    inputs,
                    keys,
                    mappings["A"][keys],
                    changed[keys],
                    embeddings,
                    spec["batch_size"],
                )
                curve.append(
                    {"step": step, "mean_training_loss": train_loss_sum / step, "A": evaluation}
                )
                print(
                    f"world {seed}: reader {loss} step {step}, A={evaluation['accuracy_all']:.4f}",
                    flush=True,
                )
        model.eval()
        models[loss] = model
        frozen_after = tensor_hash(_state(model, frozen=True))
        assert frozen_after == frozen_before
        checkpoint = f"reader-{loss}.pt"
        torch.save(
            {
                "kind": "reader",
                "spec": spec,
                "seed": seed,
                "loss": loss,
                "gpt_config": asdict(gpt_config),
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "trainable_names": [k for k, v in model.named_parameters() if v.requires_grad],
            },
            output_dir / checkpoint,
        )
        summary["readers"][loss] = {
            "initial_non_mlp_hash": initial_hash,
            "final_non_mlp_hash": tensor_hash(_state(model, non_mlp=True)),
            "frozen_hash_before": frozen_before,
            "frozen_hash_after": frozen_after,
            "training_stream_hash": stream_hash.hexdigest(),
            "curve": curve,
            "seconds": time.perf_counter() - stage,
            "checkpoint": checkpoint,
            "endpoints": {},
        }
        persist()

    stage = time.perf_counter()
    for loss, model in models.items():
        for name in ("A", "B"):
            summary["memories"][loss][name]["standalone"] = standalone(
                memories[loss][name],
                factsets[name],
                embeddings,
                changed,
                output_dir / f"standalone-{loss}-{name}.npz",
            )
        conditions = {
            "A": (memories[loss]["A"], "A"),
            "wrong_A_on_B": (memories[loss]["A"], "B"),
            "B_ce": (memories["ce"]["B"], "B"),
            "B_mse": (memories["mse"]["B"], "B"),
        }
        for condition, (memory, target_map) in conditions.items():
            insert_mlp_into_gpt(model.eval(), memory, embeddings)
            result = evaluate_reader(
                model,
                inputs,
                keys,
                mappings[target_map][keys],
                changed[keys],
                embeddings,
                spec["batch_size"],
                output_dir / f"reader-{loss}-{condition}.npz",
            )
            summary["readers"][loss]["endpoints"][condition] = result
        insert_mlp_into_gpt(model.eval(), memories[loss]["A"], embeddings)
        assert (
            tensor_hash(_state(model, non_mlp=True))
            == summary["readers"][loss]["final_non_mlp_hash"]
        )
    initial_hashes = {
        v["initial_hash"] for losses in summary["memories"].values() for v in losses.values()
    }
    summary["audits"] = {
        "paired_memory_initializations": len(initial_hashes) == 1,
        "paired_reader_initializations": len(
            {r["initial_non_mlp_hash"] for r in summary["readers"].values()}
        )
        == 1,
        "paired_training_streams": len(
            {r["training_stream_hash"] for r in summary["readers"].values()}
        )
        == 1,
        "frozen_parameters_unchanged": all(
            r["frozen_hash_before"] == r["frozen_hash_after"] for r in summary["readers"].values()
        ),
        "future_answer_removed": inputs.shape[1] == 2 * spec["junk_len"] + 2,
        "B_used_for_reader_optimization_or_selection": False,
    }
    assert all(
        v
        for k, v in summary["audits"].items()
        if k != "B_used_for_reader_optimization_or_selection"
    )
    summary["cost"] = {
        "memory_epochs_per_mapping_objective": spec["mlp_epochs"],
        "memory_fact_exposures_per_mapping_objective": spec["mlp_epochs"] * spec["num_facts"],
        "reader_steps_per_objective": spec["reader_steps"],
        "reader_tokens_per_objective": spec["reader_steps"]
        * spec["batch_size"]
        * (2 * spec["junk_len"] + 2),
        "reader_supervised_positions_per_objective": spec["reader_steps"] * spec["batch_size"],
        "peak_gpu_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
    }
    summary["timing_seconds"] = {
        "final_evaluation": time.perf_counter() - stage,
        "whole_world": time.perf_counter() - started,
    }
    summary["state"] = "complete"
    persist()
    return summary
