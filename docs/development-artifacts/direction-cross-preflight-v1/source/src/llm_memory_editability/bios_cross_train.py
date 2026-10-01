"""Frozen learning and editing runs for the two-relation crossover."""

import argparse
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
    CONDITIONS,
    audit,
    documents,
    edit_pair,
    epoch_documents,
    make_cross_world,
    qa_schedule,
    render,
)
from .bios_data import EOS, array_hash, rng_for, write_json
from .bios_model import CausalLM, ModelConfig, matmul_flops, select_parameters
from .bios_organization_train import atomic_numpy_save, atomic_torch_save, state_hash
from .bios_train import precision


def tensor_queries(world, device, answers=None):
    answers = world.answers if answers is None else answers
    prompts = torch.as_tensor(world.prompts, device=device)
    lengths = torch.as_tensor(world.lengths, device=device)
    targets = torch.as_tensor(answers, device=device)
    tokens = torch.zeros((len(answers), 6), device=device, dtype=torch.long)
    tokens[:, :5] = prompts
    tokens[torch.arange(len(answers), device=device), lengths] = targets
    tokens[lengths == 4, 5] = EOS
    return {
        "prompts": prompts,
        "lengths": lengths,
        "tokens": tokens,
        "positions": torch.stack([lengths - 1, lengths], -1),
        "labels": torch.stack([targets, torch.full_like(targets, EOS)], -1),
    }


@torch.no_grad()
def evaluate(model, data, answers, batch_size=512):
    model.eval()
    device = data["tokens"].device
    predictions, ended, nll = [], [], []
    for begin in range(0, len(answers), batch_size):
        end = min(begin + batch_size, len(answers))
        lengths = data["lengths"][begin:end]
        with precision(device):
            logits = model(data["prompts"][begin:end], (lengths - 1)[:, None])[:, 0].float()
            first = logits.argmax(-1)
            continuation = torch.zeros((end - begin, 6), device=device, dtype=torch.long)
            continuation[:, :5] = data["prompts"][begin:end]
            continuation[torch.arange(end - begin, device=device), lengths] = first
            second = model(continuation, lengths[:, None])[:, 0].argmax(-1)
        predictions.append(first.cpu().numpy())
        ended.append(second.eq(EOS).cpu().numpy())
        targets = torch.as_tensor(answers[begin:end], device=device)
        nll.append(F.cross_entropy(logits, targets, reduction="none").cpu().numpy())
    prediction, ended = np.concatenate(predictions), np.concatenate(ended)
    return {
        "prediction": prediction,
        "ended": ended,
        "correct": (prediction == answers) & ended,
        "value_nll": np.concatenate(nll),
    }


def learning_metrics(world, arrays):
    correct = arrays["correct"]
    result = {
        "base_accuracy": float(correct[: world.n_base].mean()),
        "base_nll": float(arrays["value_nll"][: world.n_base].mean()),
    }
    for chain, name in enumerate(CHAINS):
        for split, ids in (
            ("trained", world.train_ids[chain]),
            ("heldout", world.heldout_ids[chain]),
        ):
            result[f"{name}_{split}"] = float(correct[ids].mean())
            result[f"{name}_{split}_nll"] = float(arrays["value_nll"][ids].mean())
        for label, ids in (
            ("membership", world.membership_ids[chain]),
            ("default", world.root_ids[chain]),
            ("actual", world.actual_ids[chain]),
        ):
            result[f"{name}_{label}"] = float(correct[ids].mean())
        # A conditional diagnostic, never a replacement for the whole test pool.
        prerequisites = (
            correct[world.membership_ids[chain]]
            & correct[world.root_ids[chain, world.memberships[chain]]]
        )
        known_ids = np.intersect1d(
            world.heldout_ids[chain], world.derived_ids[chain, prerequisites]
        )
        result[f"{name}_known_heldout_n"] = len(known_ids)
        result[f"{name}_known_heldout"] = (
            float(correct[known_ids].mean()) if len(known_ids) else None
        )
    result["mean_heldout"] = (result["company_heldout"] + result["project_heldout"]) / 2
    return result


def source_hashes():
    root = Path(__file__).parent
    return {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in (
            "bios_cross.py",
            "bios_cross_train.py",
            "bios_model.py",
            "bios_data.py",
            "bios_train.py",
            "bios_organization_train.py",
        )
    }


def learning(args, world, out, study, device):
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
    config = ModelConfig(world.vocab_size, study["width"], study["layers"], study["heads"])
    model = CausalLM(config).to(device)
    initial_hash = state_hash(model.state_dict())
    docs, schedule = documents(world, args.condition), qa_schedule(world, study["steps"])
    identity = {
        "world": args.world,
        "seed": args.seed,
        "condition": args.condition,
        "study": study,
        "sources": source_hashes(),
        "model": model.config_dict(),
        "initial_sha256": initial_hash,
        "truth_sha256": array_hash(world.answers),
        "prompts_sha256": array_hash(world.prompts),
        "documents_sha256": array_hash(docs),
        "qa_sha256": array_hash(schedule),
    }
    config_path = out / "config.json"
    if config_path.exists():
        saved = json.loads(config_path.read_text())
        if any(saved[k] != v for k, v in identity.items()):
            raise ValueError("Frozen run identity or sources changed")
    else:
        write_json(
            config_path,
            {
                **identity,
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "python": platform.python_version(),
                "numpy": np.__version__,
                "device": str(device),
                "visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
                "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                "precision": "BF16 autocast; FP32 weights and optimizer",
                "parameters": sum(p.numel() for p in model.parameters()),
            },
        )
        atomic_numpy_save(out / "schedule.npz", documents=docs, qa=schedule)
    data = tensor_queries(world, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=study["lr"], weight_decay=0.1, fused=device.type == "cuda"
    )
    timeline, first, seconds = [], 0, 0.0
    counts = np.zeros(len(world.answers), dtype=np.int64)
    slots = np.zeros((world.n_base, 10), dtype=np.int64)
    weighted = np.zeros(len(world.answers), dtype=np.float64)
    if (out / "resume.pt").exists():
        resume = torch.load(out / "resume.pt", map_location="cpu", weights_only=False)
        if resume["config_sha256"] != hashlib.sha256(config_path.read_bytes()).hexdigest():
            raise ValueError("Resume configuration mismatch")
        model.load_state_dict(resume["model"])
        optimizer.load_state_dict(resume["optimizer"])
        first, timeline, seconds = resume["step"], resume["timeline"], resume["seconds"]
        counts, slots, weighted = resume["counts"], resume["slots"], resume["weighted"]
        torch.set_rng_state(resume["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(resume["cuda_rng"], device)
    epoch_cached, cache = -1, None
    for step in range(first, study["steps"] + 1):
        if step in study["checkpoints"] and (not timeline or step != first):
            arrays = evaluate(model, data, world.answers)
            point = {
                "step": step,
                **learning_metrics(world, arrays),
                "train_seconds": seconds,
                "exposure_sha256": array_hash(counts),
                "slots_sha256": array_hash(slots),
                "weighted_sha256": array_hash(weighted),
                "train_matmul_flops_estimate": step
                * (matmul_flops(config, 16, 60, 20) + matmul_flops(config, 40, 6, 2)),
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
            atomic_torch_save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "timeline": timeline,
                    "seconds": seconds,
                    "counts": counts,
                    "slots": slots,
                    "weighted": weighted,
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
                    "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
                },
                out / "resume.pt",
            )
            print(json.dumps({"event": "checkpoint", **point}), flush=True)
        if step == study["steps"]:
            break
        epoch, offset = divmod(step, 128)
        if epoch != epoch_cached:
            arranged = epoch_documents(docs, world.seed, epoch)
            cache = {
                k: torch.as_tensor(v, device=device) for k, v in render(world, arranged).items()
            }
            epoch_cached = epoch
        start, end = offset * 16, (offset + 1) * 16
        fact_ids = arranged[start:end]
        qa = torch.as_tensor(schedule[step], device=device)
        lr = study["lr"] * min((epoch + 1) / 16, 1)
        for group in optimizer.param_groups:
            group["lr"] = lr
        model.train()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
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
            raise FloatingPointError("Nonfinite learning update")
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        seconds += time.perf_counter() - started
        np.add.at(counts, fact_ids.ravel(), 1)
        np.add.at(counts, schedule[step], 1)
        np.add.at(slots, (fact_ids, np.arange(10)[None, :]), 1)
        np.add.at(weighted, fact_ids.ravel(), lr)
        np.add.at(weighted, schedule[step], lr)
        if (step + 1) % 256 == 0:
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
    return model, data


def edit_metrics(world, pair, arrays, old_correct):
    correct = arrays["correct"]
    metrics = {
        "E": float(correct[pair["E"]].mean()),
        "D": float(correct[pair["D"]].mean()),
        "D_conflict": float(correct[pair["conflict_D"]].mean()),
    }
    for split, ids in (
        ("trained", world.train_ids.ravel()),
        ("heldout", world.heldout_ids.ravel()),
    ):
        subset = np.intersect1d(pair["D"], ids)
        metrics[f"D_{split}_n"] = len(subset)
        metrics[f"D_{split}"] = float(correct[subset].mean()) if len(subset) else None
    for name, pool in (("full", np.arange(len(correct))), ("heldout", pair["heldout"])):
        strata = {}
        for group in range(4):
            ids = np.intersect1d(pool, np.flatnonzero((pair["strata"] == group) & old_correct))
            strata[str(group)] = {
                "known": len(ids),
                "broken": int((~correct[ids]).sum()),
                "rate": float((~correct[ids]).mean()) if len(ids) else None,
            }
        metrics[f"U_{name}"] = strata
    return metrics


def edits(args, world, model, data, out, study, device):
    baseline = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    old_arrays = evaluate(model, data, world.answers)
    for chain, name in enumerate(CHAINS):
        pair = edit_pair(world, chain)
        rng = rng_for(world.seed, 908, chain)
        e_choices = rng.integers(len(pair["E"]), size=(study["edit_steps"], 128))
        r_choices = rng.integers(len(pair["replay"]), size=(study["edit_steps"], 128))
        for kind in ("coherent", "exception"):
            target = pair[kind]
            new_data = tensor_queries(world, device, target)
            for scope in ("mlp", "all"):
                dest = out / "edits" / f"{name}-{kind}-{scope}"
                if (dest / "complete.json").exists():
                    continue
                dest.mkdir(parents=True, exist_ok=True)
                model.load_state_dict(baseline)
                selected = select_parameters(model, scope, 3)
                model.eval()
                references = []
                with torch.no_grad(), precision(device):
                    for begin in range(0, len(pair["replay"]), 256):
                        ids = torch.as_tensor(pair["replay"][begin : begin + 256], device=device)
                        references.append(
                            model(data["tokens"][ids], data["positions"][ids]).detach()
                        )
                references = torch.cat(references)
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
                timeline = []
                seconds = 0.0
                for step in range(study["edit_steps"] + 1):
                    if step in study["edit_checkpoints"]:
                        arrays = evaluate(model, data, target)
                        point = {
                            "step": step,
                            **edit_metrics(world, pair, arrays, old_arrays["correct"]),
                            "edit_seconds": seconds,
                        }
                        timeline.append(point)
                        atomic_numpy_save(dest / f"predictions-{step}.npz", **arrays)
                        write_json(dest / "trajectory.json", timeline)
                    if step == study["edit_steps"]:
                        break
                    model.train()
                    ei = torch.as_tensor(pair["E"][e_choices[step]], device=device)
                    choices = torch.as_tensor(r_choices[step], device=device)
                    ri = torch.as_tensor(pair["replay"][r_choices[step]], device=device)
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    started = time.perf_counter()
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
                        loss = ce + study["retention_kl"] * kl
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(selected, 1)
                    if not torch.isfinite(loss + norm):
                        raise FloatingPointError("Nonfinite edit")
                    optimizer.step()
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
                        "scope": scope,
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
                del references, optimizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--config", default="configs/bios-cross-development-v1.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        study = json.loads(Path(args.config).read_text())
        torch.set_num_threads(args.threads)
        device = torch.device(args.device)
        world = make_cross_world(args.world)
        audit(world)
        try:
            model, data = learning(args, world, out, study, device)
            edits(args, world, model, data, out, study, device)
            write_json(
                out / "complete.json",
                {
                    "status": "complete",
                    "learning_steps": study["steps"],
                    "edit_cases": 8,
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
