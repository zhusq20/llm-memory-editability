"""P1 counterfactual edits with frozen parent models and exact restart state.

The supervised pool S is fixed at 93 facts. E_changed, propagation D, and
unchanged U are recomputed from each arm's truth, independently of that pool.
"""

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

from .bios_cross import CHAINS, CONDITIONS, edit_pair, make_cross_world
from .bios_cross_train import evaluate, tensor_queries
from .bios_cross_train import source_hashes as parent_source_hashes
from .bios_data import array_hash, rng_for, write_json
from .bios_model import CausalLM, ModelConfig, select_parameters
from .bios_organization_train import atomic_numpy_save, atomic_torch_save, state_hash
from .bios_train import precision

ARMS = ("old-fact-rehearsal", "root-only", "actual-only", "class-balanced-exception")
OPTIONAL_ARM = "all-freeze-embedding"
REFERENCE_ARMS = ("coherent", "exception")
DEFAULT_STUDY = {
    "protocol": "v2.8-development-mechanism-P1",
    "widths": [256, 768],
    "worlds": [0, 1],
    "seeds": [0, 1],
    "conditions": list(CONDITIONS),
    "arms": list(ARMS),
    "edit_steps": 512,
    "edit_checkpoints": [0, 32, 128, 512],
    "edit_lr": 3e-5,
    "retention_kl": 1.0,
    "batch_size": 128,
    "save_every": 32,
    "gradient_steps": [0, 31, 127, 511],
}


def load_study(path=None, arms=None):
    study = json.loads(json.dumps(DEFAULT_STUDY))
    if path:
        supplied = json.loads(Path(path).read_text())
        unknown = set(supplied) - set(study)
        if unknown:
            raise ValueError(f"Unknown P1 configuration keys: {sorted(unknown)}")
        study.update(supplied)
    if arms is not None:
        study["arms"] = list(arms)
    if not study["arms"] or len(set(study["arms"])) != len(study["arms"]):
        raise ValueError("At least one unique arm is required")
    if set(study["arms"]) - set((*ARMS, OPTIONAL_ARM)):
        raise ValueError("Unknown P1 arm")
    if study["edit_steps"] < 1 or study["batch_size"] != 128 or study["save_every"] < 1:
        raise ValueError("Positive budgets and the original batch size 128 are required")
    checkpoints = study["edit_checkpoints"]
    if not checkpoints or checkpoints != sorted(set(checkpoints)) or checkpoints[0] != 0:
        raise ValueError("Checkpoints must be unique, ordered, and include zero")
    if checkpoints[-1] != study["edit_steps"]:
        raise ValueError("The final checkpoint must equal the fixed edit budget")
    if study["edit_lr"] <= 0 or study["retention_kl"] < 0:
        raise ValueError("Invalid optimization settings")
    for key in ("widths", "worlds", "seeds", "conditions"):
        if not study[key] or len(set(study[key])) != len(study[key]):
            raise ValueError(f"Nonempty, unique matrix entries required: {key}")
    if set(study["conditions"]) - set(CONDITIONS):
        raise ValueError("Unknown organization condition")
    return study


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_hashes():
    result = parent_source_hashes()
    result[Path(__file__).name] = file_hash(__file__)
    runner = Path(__file__).resolve().parents[2] / "scripts/run_bios_mechanism_edit.py"
    result[runner.name] = file_hash(runner)
    return result


def sampling_streams(world, chain, steps=512, batch_size=128):
    """Match the original generator, including the order of its two draws."""
    pair = edit_pair(world, chain)
    rng = rng_for(world.seed, 908, chain)
    supervised = rng.integers(len(pair["E"]), size=(steps, batch_size))
    replay = rng.integers(len(pair["replay"]), size=(steps, batch_size))
    return supervised, replay


def make_arm(world, chain, arm, pair=None):
    if arm not in (*ARMS, OPTIONAL_ARM, *REFERENCE_ARMS):
        raise ValueError(f"Unknown arm: {arm}")
    pair = edit_pair(world, chain) if pair is None else pair
    supervised = pair["E"].copy()
    root = np.intersect1d(supervised, world.root_ids[chain])
    actual = np.intersect1d(supervised, world.actual_ids[chain])
    assert (len(supervised), len(root), len(actual)) == (93, 3, 90)
    target = world.answers.copy()
    if arm == "root-only":
        target[root] = pair["exception"][root]
    elif arm == "actual-only":
        target[actual] = pair["exception"][actual]
    elif arm in ("exception", "class-balanced-exception", OPTIONAL_ARM):
        target[supervised] = pair["exception"][supervised]
    elif arm == "coherent":
        target[supervised] = pair["coherent"][supervised]
    # Recompute derived truth, rather than copying the original D declaration.
    for task in range(2):
        target[world.derived_ids[task]] = target[world.root_ids[task, world.memberships[task]]]
    changed = target != world.answers
    e = np.flatnonzero(changed & (world.relation < 10))
    d = np.flatnonzero(changed & (world.relation >= 10))
    u = np.flatnonzero(~changed)
    if not np.isin(e, supervised).all() or not np.array_equal(
        target[pair["replay"]], world.answers[pair["replay"]]
    ):
        raise ValueError("Supervision or common retention pool violates the truth contract")
    if np.intersect1d(pair["replay"], supervised).size:
        raise ValueError("The replay pool overlaps S")
    selected_people = np.isin(world.memberships[chain], pair["groups"])
    selected_facts = (world.person >= 0) & selected_people[np.maximum(world.person, 0)]
    old_exceptions = np.zeros(len(world.answers), dtype=bool)
    old_exceptions[world.actual_ids[chain, selected_people & world.exceptions[chain]]] = True
    relevant = np.isin(world.relation, [0, 1, 2, 10] if chain == 0 else [7, 8, 9, 11])
    strata = np.full(len(world.answers), -1, dtype=np.int64)
    strata[~changed & old_exceptions] = 0
    strata[~changed & selected_facts & ~old_exceptions] = 1
    strata[~changed & ~selected_facts & relevant] = 2
    strata[~changed & ~selected_facts & ~relevant] = 3
    assert np.array_equal(strata >= 0, ~changed)
    supervised_unchanged = np.intersect1d(supervised, u)
    probe_unchanged = np.intersect1d(pair["D"], u)
    supplied = np.union1d(supervised, pair["replay"])
    # Keep the archived heldout pool; add newly unchanged D, which is never trained.
    heldout = np.setdiff1d(
        np.intersect1d(np.union1d(pair["heldout"], probe_unchanged), u), supplied
    )
    weights = np.ones(len(supervised), dtype=np.float64)
    root_mask = np.isin(supervised, root)
    if arm == "class-balanced-exception":
        weights[root_mask] = len(supervised) / (2 * len(root))
        weights[~root_mask] = len(supervised) / (2 * len(actual))
    return {
        "target": target,
        "S": supervised,
        "S_root": root,
        "S_actual": actual,
        "S_root_mask": root_mask,
        "S_weights": weights,
        "E_changed": e,
        "E_changed_root": np.intersect1d(e, root),
        "E_changed_actual": np.intersect1d(e, actual),
        "D": d,
        "D_conflict": np.intersect1d(d, pair["conflict_D"]),
        "D_probe": pair["D"],
        "D_probe_conflict": pair["conflict_D"],
        "D_probe_unchanged": probe_unchanged,
        "U": u,
        "strata": strata,
        "heldout": heldout,
        "unseen": np.setdiff1d(u, supplied),
        "S_unchanged": supervised_unchanged,
        "replay": pair["replay"],
        "other_chain": np.flatnonzero(
            ~changed & np.isin(world.relation, [7, 8, 9, 11] if chain == 0 else [0, 1, 2, 10])
        ),
        "independent": np.flatnonzero(~changed & np.isin(world.relation, [3, 4, 5, 6])),
        "groups": pair["groups"],
    }


def accuracy(correct, ids):
    return {
        "n": len(ids),
        "correct": int(correct[ids].sum()),
        "accuracy": float(correct[ids].mean()) if len(ids) else None,
    }


def retention(correct, old_correct, ids):
    known = ids[old_correct[ids]]
    broken = int((~correct[known]).sum())
    return {
        "n": len(ids),
        "known": len(known),
        "unknown": len(ids) - len(known),
        "broken": broken,
        "rate": broken / len(known) if len(known) else None,
    }


def arm_metrics(world, spec, arrays, old_correct):
    correct = arrays["correct"]
    result = {
        name: accuracy(correct, spec[name])
        for name in (
            "S",
            "S_root",
            "S_actual",
            "E_changed",
            "E_changed_root",
            "E_changed_actual",
            "D",
            "D_conflict",
            "D_probe",
        )
    }
    for name in ("D", "D_conflict", "D_probe", "D_probe_conflict"):
        for split, pool in (("trained", world.train_ids), ("heldout", world.heldout_ids)):
            result[f"{name}_{split}"] = accuracy(correct, np.intersect1d(spec[name], pool))
    for name, pool in (
        ("full", spec["U"]),
        ("heldout", spec["heldout"]),
        ("unseen", spec["unseen"]),
    ):
        result[f"U_{name}"] = retention(correct, old_correct, pool)
        result[f"U_{name}_strata"] = {
            str(group): retention(
                correct, old_correct, np.intersect1d(pool, np.flatnonzero(spec["strata"] == group))
            )
            for group in range(4)
        }
    for name in ("S_unchanged", "D_probe_unchanged", "replay", "other_chain", "independent"):
        result[f"U_{name}"] = retention(correct, old_correct, spec[name])
    return result


def weighted_supervision(logits, labels, weights):
    """Average over sampled examples, never renormalize observed class mass."""
    losses = F.cross_entropy(logits.float().flatten(0, 1), labels.flatten(), reduction="none")
    per_example = losses.reshape(len(labels), -1).mean(-1)
    return (per_example * weights).mean(), per_example


def select_edit_parameters(model, arm):
    if arm != OPTIONAL_ARM:
        return select_parameters(model, "mlp", 3)
    selected = select_parameters(model, "all")
    model.token.weight.requires_grad_(False)
    return [parameter for parameter in selected if parameter.requires_grad]


def _gradient_norm(loss, selected):
    gradients = torch.autograd.grad(loss, selected, retain_graph=True, allow_unused=True)
    total = sum(g.detach().float().square().sum() for g in gradients if g is not None)
    return float(torch.sqrt(total)) if isinstance(total, torch.Tensor) else 0.0


def _freeze_json(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"Frozen contract mismatch: {path}")
    else:
        write_json(path, value)


def _contract_hash(contract):
    return hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()


def _checkpoint(model, optimizer, step, contract, timeline, stats, seconds, device):
    return {
        "model": model.state_dict(),
        "config": model.config_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "contract_sha256": _contract_hash(contract),
        "timeline": timeline,
        "update_stats": stats,
        "seconds": seconds,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
    }


def run_case(model, baseline_state, world, data, old_arrays, chain, arm, dest, study, parent):
    """Run or resume one fixed-budget case; no evaluation result changes training."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    device = data["tokens"].device
    spec = make_arm(world, chain, arm)
    e_choices, r_choices = sampling_streams(world, chain, study["edit_steps"], study["batch_size"])
    contract = {
        "parent": parent,
        "study": study,
        "chain": CHAINS[chain],
        "arm": arm,
        "sources": source_hashes(),
        "sets_sha256": {key: array_hash(value) for key, value in spec.items()},
        "edit_sampling_sha256": array_hash(e_choices),
        "replay_sampling_sha256": array_hash(r_choices),
        "old_correct_sha256": array_hash(old_arrays["correct"]),
    }
    _freeze_json(dest / "contract.json", contract)
    if (dest / "complete.json").exists():
        completed = json.loads((dest / "complete.json").read_text())
        if completed["contract_sha256"] != _contract_hash(contract):
            raise ValueError("Completion contract mismatch")
        return completed
    model.load_state_dict(baseline_state)
    selected = select_edit_parameters(model, arm)
    model.eval()
    references = []
    with torch.no_grad(), precision(device):
        for begin in range(0, len(spec["replay"]), 256):
            ids = torch.as_tensor(spec["replay"][begin : begin + 256], device=device)
            references.append(model(data["tokens"][ids], data["positions"][ids]).detach())
    references = torch.cat(references)
    new_data = tensor_queries(world, device, spec["target"])
    atomic_numpy_save(
        dest / "sets.npz",
        **spec,
        old_correct=old_arrays["correct"],
        edit_sampling=e_choices,
        replay_sampling=r_choices,
    )
    optimizer = torch.optim.AdamW(
        selected, lr=study["edit_lr"], weight_decay=0.1, fused=device.type == "cuda"
    )
    first, timeline, stats, seconds = 0, [], [], 0.0
    resume_path = dest / "resume.pt"
    if resume_path.exists():
        state = torch.load(resume_path, map_location="cpu", weights_only=False)
        if state["contract_sha256"] != _contract_hash(contract):
            raise ValueError("Resume contract mismatch")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        first, timeline, stats, seconds = (
            state["step"],
            state["timeline"],
            state["update_stats"],
            state["seconds"],
        )
        if len(stats) != first or first > study["edit_steps"]:
            raise ValueError("Invalid restart progress")
        torch.set_rng_state(state["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(state["cuda_rng"], device)
    weights = torch.as_tensor(spec["S_weights"], device=device, dtype=torch.float32)
    root_mask = torch.as_tensor(spec["S_root_mask"], device=device)
    for step in range(first, study["edit_steps"] + 1):
        if step in study["edit_checkpoints"] and not any(p["step"] == step for p in timeline):
            arrays = evaluate(model, data, spec["target"])
            atomic_numpy_save(dest / f"predictions-{step}.npz", **arrays)
            timeline.append(
                {
                    "step": step,
                    **arm_metrics(world, spec, arrays, old_arrays["correct"]),
                    "edit_seconds": seconds,
                }
            )
            write_json(dest / "trajectory.json", timeline)
        if step % study["save_every"] == 0 or step in study["edit_checkpoints"]:
            atomic_torch_save(
                _checkpoint(model, optimizer, step, contract, timeline, stats, seconds, device),
                resume_path,
            )
            write_json(dest / "update-stats.json", stats)
        if step == study["edit_steps"]:
            break
        model.train()
        optimizer.zero_grad(set_to_none=True)
        choices = torch.as_tensor(e_choices[step], device=device)
        ei = torch.as_tensor(spec["S"][e_choices[step]], device=device)
        replay_choices = torch.as_tensor(r_choices[step], device=device)
        ri = torch.as_tensor(spec["replay"][r_choices[step]], device=device)
        roots = root_mask[choices]
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        with precision(device):
            logits = model(new_data["tokens"][ei], new_data["positions"][ei]).float()
            ce, per_example = weighted_supervision(logits, new_data["labels"][ei], weights[choices])
            replay_logits = model(data["tokens"][ri], data["positions"][ri]).float()
            kl = (
                F.kl_div(
                    F.log_softmax(replay_logits, -1),
                    F.softmax(references[replay_choices].float(), -1),
                    reduction="none",
                )
                .sum(-1)
                .mean()
            )
            loss = ce + study["retention_kl"] * kl
        point = {
            "step": step + 1,
            "root_samples": int(roots.sum()),
            "actual_samples": int((~roots).sum()),
            "ce": float(ce.detach()),
            "kl": float(kl.detach()),
            "loss": float(loss.detach()),
        }
        if step in study["gradient_steps"]:
            for name, mask in (("root", roots), ("actual", ~roots)):
                component = (per_example * weights[choices] * mask).mean()
                point[f"{name}_ce_gradient_norm"] = _gradient_norm(component, selected)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(selected, 1)
        if not torch.isfinite(loss + norm):
            raise FloatingPointError("Nonfinite P1 edit")
        point["total_gradient_norm"] = float(norm)
        point["gradient_clipped"] = bool(norm > 1)
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        seconds += time.perf_counter() - started
        stats.append(point)
        if (step + 1) % 128 == 0:
            print(
                json.dumps({"event": "progress", "case": dest.name, "step": step + 1}), flush=True
            )
    atomic_torch_save(
        {"model": model.state_dict(), "config": model.config_dict(), "step": study["edit_steps"]},
        dest / "model-final.pt",
    )
    completed = {
        "status": "complete",
        "contract_sha256": _contract_hash(contract),
        "final_model_sha256": state_hash(model.state_dict()),
        "parameters_updated": sum(p.numel() for p in selected),
        "final": timeline[-1],
        "root_samples": sum(p["root_samples"] for p in stats),
        "actual_samples": sum(p["actual_samples"] for p in stats),
        "clipped_updates": sum(p["gradient_clipped"] for p in stats),
    }
    if source_hashes() != contract["sources"]:
        raise ValueError("P1 source files changed during this edit")
    write_json(dest / "complete.json", completed)
    return completed


def rescore_references(world, baseline, out, study, old_correct, parent):
    """Archive provenance and recompute metrics without mutating historical files."""
    for chain, name in enumerate(CHAINS):
        e_choices, r_choices = sampling_streams(world, chain, study["edit_steps"])
        pair = edit_pair(world, chain)
        for arm in REFERENCE_ARMS:
            source = baseline / "edits" / f"{name}-{arm}-mlp"
            spec = make_arm(world, chain, arm, pair)
            with np.load(source / "sets.npz") as archived:
                for key, expected in (
                    ("E", spec["S"]),
                    ("replay", spec["replay"]),
                    ("edit_sampling", e_choices),
                    ("replay_sampling", r_choices),
                    ("old_correct", old_correct),
                    (arm, spec["target"]),
                ):
                    if not np.array_equal(archived[key], expected):
                        raise ValueError(f"Reference reuse contract failed for {source}/{key}")
            if not (source / "complete.json").exists():
                raise ValueError(f"Historical reference is incomplete: {source}")
            timeline, hashes = [], {}
            for step in study["edit_checkpoints"]:
                path = source / f"predictions-{step}.npz"
                with np.load(path) as stored:
                    arrays = {key: stored[key] for key in ("prediction", "ended", "correct")}
                recomputed = (arrays["prediction"] == spec["target"]) & arrays["ended"]
                if not np.array_equal(arrays["correct"], recomputed):
                    raise ValueError("Historical prediction correctness mismatch")
                timeline.append({"step": step, **arm_metrics(world, spec, arrays, old_correct)})
                hashes[path.name] = file_hash(path)
            dest = out / "references" / f"{name}-{arm}-mlp"
            dest.mkdir(parents=True, exist_ok=True)
            _freeze_json(
                dest / "contract.json", {"parent": parent, "source": str(source), "files": hashes}
            )
            atomic_numpy_save(dest / "sets.npz", **spec, old_correct=old_correct)
            write_json(dest / "trajectory.json", timeline)


def run_model(baseline, output, study, device):
    baseline, output = Path(baseline).resolve(), Path(output).resolve()
    if output == baseline or baseline in output.parents:
        raise ValueError("P1 results must not be written inside a frozen parent directory")
    output.mkdir(parents=True, exist_ok=True)
    config_path = baseline / "config.json"
    config = json.loads(config_path.read_text())
    if config["sources"] != parent_source_hashes():
        raise ValueError("Frozen parent training sources have changed")
    for key, current in (("torch", torch.__version__), ("cuda", torch.version.cuda)):
        if config[key] != current:
            raise ValueError(f"Parent and P1 runtime differ: {key}")
    parent_study = config["study"]
    for key in ("edit_steps", "edit_checkpoints", "edit_lr", "retention_kl"):
        if study[key] != parent_study[key]:
            raise ValueError(f"Historical reference configuration differs: {key}")
    parent_step = parent_study["steps"]
    weights_path = baseline / f"model-{parent_step}.pt"
    saved = torch.load(weights_path, map_location="cpu", weights_only=False)
    if saved["config"] != config["model"] or saved["step"] != parent_step:
        raise ValueError("Parent weight configuration mismatch")
    final = json.loads((baseline / "learning-complete.json").read_text())
    if state_hash(saved["model"]) != final["model_sha256"]:
        raise ValueError("Parent model state hash mismatch")
    world = make_cross_world(config["world"])
    if (
        array_hash(world.answers) != config["truth_sha256"]
        or array_hash(world.prompts) != config["prompts_sha256"]
    ):
        raise ValueError("Parent world does not match the current generated truth")
    parent = {
        "directory": str(baseline),
        "config_sha256": file_hash(config_path),
        "weights_sha256": file_hash(weights_path),
        "model_sha256": final["model_sha256"],
        "width": config["model"]["width"],
        "world": config["world"],
        "seed": config["seed"],
        "condition": config["condition"],
        "step": parent_step,
    }
    runtime = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "device_type": device.type,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "precision": "BF16 autocast; FP32 weights and optimizer"
        if device.type == "cuda"
        else "FP32",
    }
    contract = {"parent": parent, "study": study, "sources": source_hashes(), "runtime": runtime}
    _freeze_json(output / "contract.json", contract)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.cuda.manual_seed_all(config["seed"])
    torch.manual_seed(config["seed"])
    model = CausalLM(ModelConfig(**saved["config"])).to(device)
    model.load_state_dict(saved["model"])
    data = tensor_queries(world, device)
    old_arrays = evaluate(model, data, world.answers)
    with np.load(baseline / f"predictions-{parent_step}.npz") as archived:
        for key in ("correct", "prediction", "ended"):
            if not np.array_equal(old_arrays[key], archived[key]):
                raise ValueError(f"Parent predictions changed under this runtime: {key}")
    atomic_numpy_save(output / "baseline-predictions.npz", **old_arrays)
    write_json(
        output / "runtime.json",
        {**runtime, "device": str(device), "visible_devices": os.getenv("CUDA_VISIBLE_DEVICES")},
    )
    rescore_references(world, baseline, output, study, old_arrays["correct"], parent)
    completed = []
    for chain, name in enumerate(CHAINS):
        for arm in study["arms"]:
            dest = output / "edits" / f"{name}-{arm}"
            result = run_case(
                model, saved["model"], world, data, old_arrays, chain, arm, dest, study, parent
            )
            completed.append(f"{name}-{arm}")
            write_json(
                output / "status.json",
                {"state": "running", "complete": completed, "updated": time.time()},
            )
            print(
                json.dumps(
                    {"event": "edit_complete", "case": dest.name, "D": result["final"]["D"]}
                ),
                flush=True,
            )
    write_json(
        output / "complete.json",
        {
            "status": "complete",
            "contract_sha256": _contract_hash(contract),
            "edit_cases": len(completed),
            "cases": completed,
            "references": 4,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config")
    parser.add_argument("--arms", nargs="+", choices=(*ARMS, OPTIONAL_ARM))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    study = load_study(args.config, args.arms)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        torch.set_num_threads(args.threads)
        try:
            run_model(args.baseline, out, study, torch.device(args.device))
        except Exception as exc:
            write_json(out / "failure.json", {"type": type(exc).__name__, "message": str(exc)})
            raise


if __name__ == "__main__":
    main()
