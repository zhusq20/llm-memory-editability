"""Protocol v2.8 P2: audited continuation of frozen crossover checkpoints."""

import argparse
import copy
import fcntl
import hashlib
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .bios_cross import (
    CHAINS,
    audit,
    documents,
    edit_pair,
    epoch_documents,
    make_cross_world,
    qa_schedule,
    render,
)
from .bios_cross_train import edit_metrics as original_edit_metrics
from .bios_cross_train import evaluate, learning_metrics, source_hashes, tensor_queries
from .bios_data import array_hash, rng_for, write_json
from .bios_model import CausalLM, ModelConfig, matmul_flops, select_parameters
from .bios_organization_train import atomic_numpy_save, atomic_torch_save, state_hash
from .bios_train import precision

PARENT_STEP = 15360
FINAL_STEP = 30720
CHECKPOINTS = (15360, 20480, 25600, 30720)
EDIT_CHECKPOINTS = (0, 32, 128, 512)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def continuation_sources():
    return {**source_hashes(), Path(__file__).name: file_hash(__file__)}


def tree_hash(value):
    """Stable audit hash for nested optimizer/RNG states, without pickle metadata."""
    digest = hashlib.sha256()

    def visit(item):
        if isinstance(item, torch.Tensor):
            item = item.detach().cpu().contiguous()
            digest.update(f"tensor:{item.dtype}:{tuple(item.shape)}:".encode())
            digest.update(item.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, np.ndarray):
            digest.update(f"numpy:{item.dtype}:{item.shape}:".encode())
            digest.update(np.ascontiguousarray(item).tobytes())
        elif isinstance(item, dict):
            digest.update(b"dict[")
            for key in sorted(item, key=lambda key: (type(key).__name__, str(key))):
                visit(key)
                visit(item[key])
            digest.update(b"]")
        elif isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode() + b"[")
            for child in item:
                visit(child)
            digest.update(b"]")
        else:
            digest.update(f"{type(item).__name__}:{item!r};".encode())

    visit(value)
    return digest.hexdigest()


def continuation_lr(base_lr, absolute_step):
    """Use the original absolute epoch; never restart warmup at a branch point."""
    return base_lr * min((absolute_step // 128 + 1) / 16, 1)


def validate_optimizer(optimizer_state, step, learning_rate):
    states = optimizer_state["state"]
    groups = optimizer_state["param_groups"]
    if not states or sum(len(group["params"]) for group in groups) != len(states):
        raise ValueError("Missing parent optimizer moments")
    for state in states.values():
        if not {"step", "exp_avg", "exp_avg_sq"} <= state.keys():
            raise ValueError("Incomplete AdamW state")
        if int(state["step"]) != step:
            raise ValueError("Optimizer update count differs from checkpoint step")
    for group in groups:
        if group["lr"] != learning_rate or group["weight_decay"] != 0.1:
            raise ValueError("Parent optimizer learning rate or decay changed")


def expected_exposure(world, docs, schedule, step, base_lr):
    """Reconstruct original full-epoch exposure, positions, and LR-weighted exposure."""
    epochs, remainder = divmod(step, 128)
    if remainder:
        raise ValueError("Parent must end on a complete document epoch")
    occurrence = np.bincount(docs.ravel(), minlength=world.n_base)
    counts = np.bincount(schedule[:step].ravel(), minlength=len(world.answers))
    counts[: world.n_base] += epochs * occurrence
    slots = np.repeat((occurrence * (epochs // 10))[:, None], 10, axis=1)
    for epoch in range(epochs % 10):
        arranged = np.roll(docs, epoch, axis=1)
        np.add.at(slots, (arranged, np.arange(10)[None, :]), 1)
    weighted = np.zeros(len(world.answers), dtype=np.float64)
    weighted[: world.n_base] = occurrence * sum(
        continuation_lr(base_lr, epoch * 128) for epoch in range(epochs)
    )
    rates = np.repeat([continuation_lr(base_lr, i) for i in range(step)], 40)
    np.add.at(weighted, schedule[:step].ravel(), rates)
    return counts, slots, weighted


def validate_parent(parent, world, new_lr):
    """Verify immutable parent provenance and the longer stream before any updates."""
    parent = Path(parent).resolve()
    config = json.loads((parent / "config.json").read_text())
    study = config["study"]
    if study["width"] not in (128, 256) or config["world"] not in (0, 1):
        raise ValueError("P2 includes only the two declared sizes and development worlds")
    if config["seed"] not in (0, 1) or config["condition"] not in ("company", "project", "neither"):
        raise ValueError("Unexpected P2 initialization or organization")
    if new_lr not in (0.0001, 0.0003) or (new_lr == 0.0003 and config["seed"] != 0):
        raise ValueError("The sensitivity branch contains only initialization zero")
    if study["steps"] != PARENT_STEP or study["lr"] != 0.0001:
        raise ValueError("P2 must fork the original 15360-step, 1e-4 trajectory")
    if config["sources"] != source_hashes():
        raise ValueError("Frozen parent training sources changed")
    if config["torch"] != torch.__version__ or config["numpy"] != np.__version__:
        raise ValueError("Parent Torch/NumPy versions differ from continuation")
    if config["world"] != world.seed or config["truth_sha256"] != array_hash(world.answers):
        raise ValueError("Parent world/truth mismatch")
    if config["prompts_sha256"] != array_hash(world.prompts):
        raise ValueError("Parent query prompts changed")
    for name, expected in {
        "layers": 8,
        "heads": study["width"] // 64,
        "documents_per_step": 16,
        "facts_per_document": 10,
        "QA_per_chain_per_step": 20,
        "document_QA_weights": [0.8, 0.2],
        "edit_steps": 512,
        "edit_lr": 0.00003,
        "retention_kl": 1.0,
    }.items():
        if study[name] != expected:
            raise ValueError(f"Unexpected frozen parent setting: {name}")
    docs, schedule = documents(world, config["condition"]), qa_schedule(world, FINAL_STEP)
    with np.load(parent / "schedule.npz") as old:
        np.testing.assert_array_equal(docs, old["documents"])
        np.testing.assert_array_equal(schedule[:PARENT_STEP], old["qa"])
    if array_hash(docs) != config["documents_sha256"]:
        raise ValueError("Parent document hash mismatch")
    if array_hash(schedule[:PARENT_STEP]) != config["qa_sha256"]:
        raise ValueError("Extended QA stream does not preserve the original prefix")
    resume = torch.load(parent / "resume.pt", map_location="cpu", weights_only=False)
    if resume["step"] != PARENT_STEP or resume["config_sha256"] != file_hash(
        parent / "config.json"
    ):
        raise ValueError("Parent resume step/config mismatch")
    validate_optimizer(resume["optimizer"], PARENT_STEP, study["lr"])
    saved_model = torch.load(
        parent / f"model-{PARENT_STEP}.pt", map_location="cpu", weights_only=False
    )
    completed = json.loads((parent / "learning-complete.json").read_text())
    model_hash = state_hash(resume["model"])
    if (
        model_hash != state_hash(saved_model["model"])
        or model_hash != completed["model_sha256"]
        or saved_model["step"] != PARENT_STEP
        or saved_model["config"] != config["model"]
    ):
        raise ValueError("Parent final model and resume state disagree")
    expected = expected_exposure(world, docs, schedule, PARENT_STEP, study["lr"])
    with np.load(parent / f"predictions-{PARENT_STEP}.npz") as predictions:
        for key, wanted in zip(("counts", "slots", "weighted"), expected, strict=True):
            if key == "weighted":
                np.testing.assert_allclose(resume[key], wanted, rtol=0, atol=1e-10)
            else:
                np.testing.assert_array_equal(resume[key], wanted)
            stored_key = "exposure" if key == "counts" else key
            np.testing.assert_array_equal(resume[key], predictions[stored_key])
    if resume["timeline"][-1]["step"] != PARENT_STEP:
        raise ValueError("Parent learning history lacks the final checkpoint")
    if not isinstance(resume["torch_rng"], torch.Tensor):
        raise ValueError("Parent torch RNG is missing")
    files = (
        "config.json",
        "resume.pt",
        f"model-{PARENT_STEP}.pt",
        "schedule.npz",
        f"predictions-{PARENT_STEP}.npz",
        "learning-complete.json",
    )
    provenance = {
        "directory": str(parent),
        "files_sha256": {name: file_hash(parent / name) for name in files},
        "model_sha256": model_hash,
        "optimizer_sha256": tree_hash(resume["optimizer"]),
        "torch_rng_sha256": tree_hash(resume["torch_rng"]),
        "cuda_rng_sha256": tree_hash(resume["cuda_rng"]),
        "qa_prefix_verified": True,
        "exposure_reconstruction_verified": True,
    }
    return config, resume, docs, schedule, provenance


def restore_state(model, optimizer, saved, device):
    model.load_state_dict(saved["model"])
    # load_state_dict may alias CPU tensors; branch updates must not mutate the parent.
    optimizer.load_state_dict(copy.deepcopy(saved["optimizer"]))
    torch.set_rng_state(saved["torch_rng"].cpu())
    if device.type == "cuda":
        if saved["cuda_rng"] is None:
            raise ValueError("CUDA parent/resume RNG state is missing")
        torch.cuda.set_rng_state(saved["cuda_rng"].cpu(), device)


def capture_state(model, optimizer, device, **metadata):
    return {
        **metadata,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
    }


def learning_update(model, optimizer, cache, data, qa, start, end, lr, device):
    """The original arithmetic/gradient order, with an absolute-step learning rate."""
    for group in optimizer.param_groups:
        group["lr"] = lr
    model.train()
    optimizer.zero_grad(set_to_none=True)
    with precision(device):
        logits = model(cache["tokens"][start:end], cache["positions"][start:end])
        doc_loss = F.cross_entropy(
            logits.float().flatten(0, 1), cache["labels"][start:end].flatten()
        )
    (0.8 * doc_loss).backward()
    with precision(device):
        logits = model(data["tokens"][qa], data["positions"][qa])
        qa_loss = F.cross_entropy(logits.float().flatten(0, 1), data["labels"][qa].flatten())
    (0.2 * qa_loss).backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
    if not torch.isfinite(doc_loss + qa_loss + norm):
        raise FloatingPointError("Nonfinite continuation update")
    optimizer.step()


def learning(parent, out, world, base_lr, device):
    old_config, parent_resume, docs, schedule, provenance = validate_parent(parent, world, base_lr)
    study = {
        **old_config["study"],
        "protocol": "v2.8-development-p2-continuation",
        "steps": FINAL_STEP,
        "checkpoints": list(CHECKPOINTS),
        "lr": base_lr,
        "edit_scopes": ["mlp"],
        "seeds": [0] if base_lr == 0.0003 else [0, 1],
        "learning_runs": 6 if base_lr == 0.0003 else 12,
        "edit_cases": 24 if base_lr == 0.0003 else 48,
        "edit_cases_per_run": 4,
        "parent_step": PARENT_STEP,
        "warmup": "original absolute epoch; no restart",
        "primary_learning": (
            "heldout full-answer accuracy at 30720 absolute steps; "
            "preserve all prespecified earlier checkpoints"
        ),
        "aggregation": (
            "pair organizations within world/initialization; "
            "development worlds are not confirmatory replications"
        ),
    }
    identity = {
        "world": old_config["world"],
        "seed": old_config["seed"],
        "condition": old_config["condition"],
        "study": study,
        "sources": continuation_sources(),
        "parent": provenance,
        "model": old_config["model"],
        "initial_sha256": old_config["initial_sha256"],
        "truth_sha256": array_hash(world.answers),
        "prompts_sha256": array_hash(world.prompts),
        "documents_sha256": array_hash(docs),
        "qa_sha256": array_hash(schedule),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device_type": device.type,
        "cuda": torch.version.cuda,
        "gpu_type": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }
    config_path = out / "config.json"
    if config_path.exists():
        existing = json.loads(config_path.read_text())
        if any(existing.get(key) != value for key, value in identity.items()):
            raise ValueError("Continuation identity, parent, software or sources changed")
    else:
        write_json(
            config_path,
            {
                **identity,
                "python": platform.python_version(),
                "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                "visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
                "precision": "BF16 autocast; FP32 weights and optimizer",
                "parameters": old_config["parameters"],
            },
        )
        atomic_numpy_save(out / "schedule.npz", documents=docs, qa=schedule)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    model_config = ModelConfig(**old_config["model"])
    model = CausalLM(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=base_lr, weight_decay=0.1, fused=device.type == "cuda"
    )
    if (out / "resume.pt").exists():
        saved = torch.load(out / "resume.pt", map_location="cpu", weights_only=False)
        if saved["config_sha256"] != file_hash(config_path):
            raise ValueError("Continuation resume configuration mismatch")
    else:
        saved = {**parent_resume, "timeline": [], "config_sha256": file_hash(config_path)}
    restore_state(model, optimizer, saved, device)
    first, seconds = saved["step"], saved["seconds"]
    timeline = saved["timeline"]
    counts, slots, weighted = saved["counts"], saved["slots"], saved["weighted"]
    if not PARENT_STEP <= first <= FINAL_STEP:
        raise ValueError("Continuation resume step is outside its frozen interval")
    data = tensor_queries(world, device)

    def save_resume(step):
        atomic_torch_save(
            capture_state(
                model,
                optimizer,
                device,
                step=step,
                timeline=timeline,
                seconds=seconds,
                counts=counts,
                slots=slots,
                weighted=weighted,
                config_sha256=file_hash(config_path),
            ),
            out / "resume.pt",
        )

    epoch_cached = -1
    for step in range(first, FINAL_STEP + 1):
        if step in CHECKPOINTS and (not timeline or timeline[-1]["step"] != step):
            if step == PARENT_STEP:
                with np.load(Path(parent) / f"predictions-{PARENT_STEP}.npz") as old:
                    arrays = {
                        key: old[key] for key in ("prediction", "ended", "correct", "value_nll")
                    }
            else:
                arrays = evaluate(model, data, world.answers)
            point = {
                "step": step,
                **learning_metrics(world, arrays),
                "train_seconds": seconds,
                "exposure_sha256": array_hash(counts),
                "slots_sha256": array_hash(slots),
                "weighted_sha256": array_hash(weighted),
                "train_matmul_flops_estimate": step
                * (matmul_flops(model_config, 16, 60, 20) + matmul_flops(model_config, 40, 6, 2)),
            }
            timeline.append(point)
            atomic_numpy_save(
                out / f"predictions-{step}.npz",
                **arrays,
                exposure=counts,
                slots=slots,
                weighted=weighted,
            )
            atomic_torch_save(
                {
                    "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "config": model.config_dict(),
                    "step": step,
                },
                out / f"model-{step}.pt",
            )
            write_json(out / "learning.json", timeline)
            save_resume(step)
            print(json.dumps({"event": "checkpoint", **point}), flush=True)
        if step == FINAL_STEP:
            break
        epoch, offset = divmod(step, 128)
        if epoch != epoch_cached:
            arranged = epoch_documents(docs, world.seed, epoch)
            cache = {
                key: torch.as_tensor(value, device=device)
                for key, value in render(world, arranged).items()
            }
            epoch_cached = epoch
        start, end = offset * 16, (offset + 1) * 16
        qa = torch.as_tensor(schedule[step], device=device)
        lr = continuation_lr(base_lr, step)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        learning_update(model, optimizer, cache, data, qa, start, end, lr, device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        seconds += time.perf_counter() - started
        fact_ids = arranged[start:end]
        np.add.at(counts, fact_ids.ravel(), 1)
        np.add.at(counts, schedule[step], 1)
        np.add.at(slots, (fact_ids, np.arange(10)[None, :]), 1)
        np.add.at(weighted, fact_ids.ravel(), lr)
        np.add.at(weighted, schedule[step], lr)
        if (step + 1) % 256 == 0 and step + 1 not in CHECKPOINTS:
            save_resume(step + 1)
            print(
                json.dumps({"event": "progress", "step": step + 1, "seconds": seconds}), flush=True
            )
    write_json(
        out / "learning-complete.json",
        {
            "status": "complete",
            "final": timeline[-1],
            "model_sha256": state_hash(model.state_dict()),
        },
    )
    return model, data, study


def edit_metrics(world, pair, arrays, old_correct):
    metrics = original_edit_metrics(world, pair, arrays, old_correct)
    heldout = np.intersect1d(pair["conflict_D"], world.heldout_ids.ravel())
    metrics["D_conflict_heldout_n"] = len(heldout)
    metrics["D_conflict_heldout"] = (
        float(arrays["correct"][heldout].mean()) if len(heldout) else None
    )
    for label, ids in (("default", world.root_ids.ravel()), ("actual", world.actual_ids.ravel())):
        subset = np.intersect1d(pair["E"], ids)
        metrics[f"E_{label}_n"] = len(subset)
        metrics[f"E_{label}"] = float(arrays["correct"][subset].mean())
    return metrics


def edit_update(model, optimizer, selected, data, new_data, references, ei, ri, choices, device):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    with precision(device):
        logits = model(new_data["tokens"][ei], new_data["positions"][ei]).float()
        ce = F.cross_entropy(logits.flatten(0, 1), new_data["labels"][ei].flatten())
        replay_logits = model(data["tokens"][ri], data["positions"][ri]).float()
        kl = (
            F.kl_div(
                F.log_softmax(replay_logits, -1),
                F.softmax(references[choices].float(), -1),
                reduction="none",
            )
            .sum(-1)
            .mean()
        )
        loss = ce + kl
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(selected, 1)
    if not torch.isfinite(loss + norm):
        raise FloatingPointError("Nonfinite continuation edit")
    optimizer.step()


def edits(world, model, data, out, study, device):
    baseline = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    old_arrays = evaluate(model, data, world.answers)
    config_sha = file_hash(out / "config.json")
    for chain, name in enumerate(CHAINS):
        pair = edit_pair(world, chain)
        rng = rng_for(world.seed, 908, chain)
        e_choices = rng.integers(len(pair["E"]), size=(512, 128))
        r_choices = rng.integers(len(pair["replay"]), size=(512, 128))
        for kind in ("coherent", "exception"):
            dest = out / "edits" / f"{name}-{kind}-mlp"
            if (dest / "complete.json").exists():
                continue
            dest.mkdir(parents=True, exist_ok=True)
            model.load_state_dict(baseline)
            selected = select_parameters(model, "mlp", 3)
            model.eval()
            references = []
            with torch.no_grad(), precision(device):
                for begin in range(0, len(pair["replay"]), 256):
                    ids = torch.as_tensor(pair["replay"][begin : begin + 256], device=device)
                    references.append(model(data["tokens"][ids], data["positions"][ids]).detach())
            references = torch.cat(references)
            target = pair[kind]
            new_data = tensor_queries(world, device, target)
            atomic_numpy_save(
                dest / "sets.npz",
                **pair,
                old_correct=old_arrays["correct"],
                edit_sampling=e_choices,
                replay_sampling=r_choices,
            )
            optimizer = torch.optim.AdamW(
                selected, lr=study["edit_lr"], weight_decay=0.1, fused=device.type == "cuda"
            )
            timeline, seconds, first = [], 0.0, 0
            if (dest / "resume.pt").exists():
                saved = torch.load(dest / "resume.pt", map_location="cpu", weights_only=False)
                if saved["config_sha256"] != config_sha or saved["case"] != dest.name:
                    raise ValueError("Editing resume identity mismatch")
                restore_state(model, optimizer, saved, device)
                timeline, seconds, first = saved["timeline"], saved["seconds"], saved["step"]
            for step in range(first, 513):
                if step in EDIT_CHECKPOINTS and (not timeline or timeline[-1]["step"] != step):
                    arrays = evaluate(model, data, target)
                    point = {
                        "step": step,
                        **edit_metrics(world, pair, arrays, old_arrays["correct"]),
                        "edit_seconds": seconds,
                    }
                    timeline.append(point)
                    atomic_numpy_save(dest / f"predictions-{step}.npz", **arrays)
                    write_json(dest / "trajectory.json", timeline)
                if step % 32 == 0:
                    atomic_torch_save(
                        capture_state(
                            model,
                            optimizer,
                            device,
                            step=step,
                            timeline=timeline,
                            seconds=seconds,
                            config_sha256=config_sha,
                            case=dest.name,
                        ),
                        dest / "resume.pt",
                    )
                if step == 512:
                    break
                ei = torch.as_tensor(pair["E"][e_choices[step]], device=device)
                ri = torch.as_tensor(pair["replay"][r_choices[step]], device=device)
                choices = torch.as_tensor(r_choices[step], device=device)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                started = time.perf_counter()
                edit_update(
                    model, optimizer, selected, data, new_data, references, ei, ri, choices, device
                )
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                seconds += time.perf_counter() - started
            atomic_torch_save(
                {"model": model.state_dict(), "config": model.config_dict()},
                dest / "model-final.pt",
            )
            write_json(
                dest / "complete.json",
                {
                    "status": "complete",
                    "chain": name,
                    "kind": kind,
                    "scope": "mlp",
                    "final": timeline[-1],
                },
            )
            print(
                json.dumps(
                    {
                        "event": "edit_complete",
                        "case": dest.name,
                        "E": timeline[-1]["E"],
                        "D": timeline[-1]["D"],
                    }
                ),
                flush=True,
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--lr", type=float, choices=(0.0001, 0.0003), default=0.0001)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--learning-only", action="store_true")
    args = parser.parse_args()
    parent, out = Path(args.parent).resolve(), Path(args.output).resolve()
    if parent == out or parent in out.parents or out in parent.parents:
        raise ValueError("Continuation output must be separate from its immutable parent")
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        torch.set_num_threads(args.threads)
        world = make_cross_world(json.loads((parent / "config.json").read_text())["world"])
        audit(world)
        device = torch.device(args.device)
        try:
            model, data, study = learning(parent, out, world, args.lr, device)
            if not args.learning_only:
                edits(world, model, data, out, study, device)
                write_json(
                    out / "complete.json",
                    {
                        "status": "complete",
                        "learning_steps": FINAL_STEP,
                        "edit_cases": 4,
                        "finished": time.time(),
                    },
                )
        except Exception as error:
            write_json(
                out / "failure.json",
                {"type": type(error).__name__, "message": str(error), "time": time.time()},
            )
            raise


if __name__ == "__main__":
    main()
