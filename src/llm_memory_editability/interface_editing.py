"""Local factual updates and downstream composition in existing aligned GPTs.

Cases are selected from the graph alone, before loading any model predictions.
Every edit has a same-budget old-fact sham, fresh optimizer, and frozen parent
replay reference. No composition labels or intermediate-state targets enter
optimization. Finite replay KL is a regularizer, not exact preservation.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .grok_depth import utc, write_json
from .latent_scaling import build_world, model_digest
from .representation_alignment import new_model
from .storage_composition import data_digest, file_hash, generate_rows, pack_sentences

ROLES = ("first", "second")
STRATA = ("familiar", "strict")
DEFAULT_NODES = (0, 32, 128, 512)


def _rows(values, width):
    return np.asarray(values, dtype=np.int64).reshape(-1, width)


def _unique(values, width):
    return _rows(sorted(set(map(tuple, _rows(values, width)))), width)


def _hash_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _batch_hash(batch):
    digest = hashlib.sha256()
    for tensor in batch:
        array = tensor.detach().cpu().numpy()
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def learned_atoms(world):
    """Only atomics actually supervised in the representation parent."""
    return _unique(np.concatenate([world["common_atomic"], world["anchor_atomic"]]), 3)


def composition_pools(world):
    """All legal untrained queries, plus the original holdout and training sets."""
    trained = set(map(tuple, world["train_composite"]))
    full = _unique(np.concatenate([world["available_composite"], world["familiar_test"]]), 5)
    familiar = _rows([r for r in full if tuple(r) not in trained], 5)
    return {
        "familiar": familiar,
        "strict": world["strict_test"].copy(),
        "registered_familiar": world["familiar_test"].copy(),
        "train": world["train_composite"].copy(),
    }


def affected(rows, fact, role):
    """Whether a query uses this factual address, independently of its answer."""
    columns = (0, 1) if role == "first" else (2, 3)
    return (rows[:, columns[0]] == fact[0]) & (rows[:, columns[1]] == fact[1])


def rewrite(rows, old_fact, new_fact, role, lookup):
    result = rows.copy()
    mask = affected(rows, old_fact, role)
    if role == "first":
        result[mask, 2] = new_fact[2]
        for index in np.flatnonzero(mask):
            result[index, 4] = lookup[int(new_fact[2]), int(rows[index, 3])]
    else:
        result[mask, 4] = new_fact[2]
    return result


def graph_cases(parent_spec, *, seed=780011, per_cell=2, replay_n=32):
    """Return JSON-compatible selected cases and graph-only eligibility counts.

    Each distinct atomic address receives at most one counterfactual per role.
    First-hop replacements stay in the original familiar/strict bridge group and
        change at least one affected untrained composition answer. Strict cases never
    borrow a bridge with composition experience from the familiar group.
    """
    if per_cell < 1 or replay_n < 1:
        raise ValueError("Positive case and replay counts are required")
    world = build_world(parent_spec)
    atoms = learned_atoms(world)
    lookup = {(int(h), int(r)): int(t) for h, r, t in atoms}
    pools = composition_pools(world)
    rng = np.random.default_rng(seed + int(parent_spec["world"]))
    candidates, qualification = [], {}
    for role in ROLES:
        for stratum in STRATA:
            rows = pools[stratum]
            addresses = _unique(rows[:, [0, 1, 2] if role == "first" else [2, 3, 4]], 3)
            bridges = np.unique(rows[:, 2])
            # Include all trained bridges in that group, even if the holdout
            # happens not to visit them (strict rows enumerate all chains).
            group = world["background_atomic"] if stratum == "familiar" else world["common_atomic"]
            group_bridges = np.unique(group[group[:, 1] >= 17, 0])
            if stratum == "strict":
                familiar_bridges = world["background_atomic"]
                familiar_bridges = familiar_bridges[familiar_bridges[:, 1] >= 17, 0]
                group_bridges = np.setdiff1d(group_bridges, familiar_bridges)
            bridges = np.union1d(bridges, group_bridges)
            tails = np.unique(atoms[atoms[:, 1] >= 17, 2])
            eligible, replacement_count = [], 0
            for old in addresses:
                impacted = rows[affected(rows, old, role)]
                choices = []
                for value in bridges if role == "first" else tails:
                    if value == old[2]:
                        continue
                    new = [int(old[0]), int(old[1]), int(value)]
                    updated = rewrite(impacted, old, new, role, lookup)
                    if np.any(updated[:, -1] != impacted[:, -1]):
                        choices.append(new)
                replacement_count += len(choices)
                if choices:
                    eligible.append((old.tolist(), choices))
            key = f"{role}_{stratum}"
            qualification[key] = {
                "candidate_addresses": len(addresses),
                "eligible_addresses": len(eligible),
                "eligible_replacements": replacement_count,
                "untrained_queries": len(rows),
            }
            if len(eligible) < per_cell:
                raise ValueError(f"Insufficient graph-qualified {key} cases: {qualification[key]}")
            order = rng.permutation(len(eligible))[:per_cell]
            for serial, index in enumerate(order):
                old, choices = eligible[index]
                new = choices[int(rng.integers(len(choices)))]
                candidates.append(
                    {
                        "case_id": f"{key}-{serial:02d}",
                        "role": role,
                        "stratum": stratum,
                        "old_fact": old,
                        "new_fact": new,
                    }
                )
    for case in candidates:
        tasks, originals = case_tasks(world, case, replay_indices=None)
        necessary = set(map(tuple, tasks["necessary_atomic"]))
        allowed = [
            i
            for i, row in enumerate(atoms)
            if tuple(row) != tuple(case["old_fact"]) and tuple(row) not in necessary
        ]
        if len(allowed) < replay_n:
            raise ValueError("Insufficient independent replay atomics")
        case["replay_indices"] = sorted(map(int, rng.choice(allowed, replay_n, replace=False)))
        case["affected_untrained_n"] = {name: len(rows) for name, rows in originals.items()}
    return candidates, qualification


def case_tasks(world, case, replay_indices=None):
    """Construct E/R/U/D with address-based exclusions and recomputed truth."""
    atoms = learned_atoms(world)
    lookup = {(int(h), int(r)): int(t) for h, r, t in atoms}
    old, new, role = case["old_fact"], case["new_fact"], case["role"]
    if lookup[tuple(old[:2])] != old[2] or old[:2] != new[:2] or old[2] == new[2]:
        raise ValueError("An edit must replace one existing factual value")
    tasks = {"E_old": _rows([old], 3), "E_new": _rows([new], 3)}
    originals, necessary = {}, []
    for stratum, rows in composition_pools(world).items():
        mask = affected(rows, old, role)
        tasks[f"U_{stratum}"] = rows[~mask].copy()
        old_rows = rows[mask].copy()
        new_rows = rewrite(old_rows, old, new, role, lookup)
        changed = old_rows[:, -1] != new_rows[:, -1]
        if stratum == "train":
            # Training-query transfer is reported separately from held-out D.
            name = f"T_{role}_train"
            originals[name] = old_rows
            tasks[name] = new_rows
        else:
            name = f"D_{role}_{stratum}"
            originals[name] = old_rows[changed]
            tasks[name] = new_rows[changed]
            same = f"S_same_answer_{role}_{stratum}"
            originals[same] = old_rows[~changed]
            tasks[same] = new_rows[~changed]
        if stratum in STRATA:
            for h, r1, b, r2, t in tasks[name]:
                prerequisite = (b, r2, t) if role == "first" else (h, r1, b)
                necessary.append(prerequisite)
    tasks["necessary_atomic"] = _unique(necessary, 3)
    replay = case.get("replay_indices", []) if replay_indices is None else replay_indices
    if len(set(replay)) != len(replay):
        raise ValueError("Replay indices must be unique")
    target = np.flatnonzero(np.all(atoms == np.asarray(old), axis=1))
    if len(target) != 1 or int(target[0]) in replay:
        raise ValueError("Replay cannot contain the edited address")
    tasks["R_atomic"] = atoms[replay].copy()
    tasks["U_atomic"] = np.delete(atoms, [int(target[0]), *replay], axis=0)
    return tasks, originals


def answer_batch(rows, device):
    """Only answer, punctuation and EOS labels; no prefix or bridge labels."""
    tokens, labels = pack_sentences(_rows(rows, 3))
    positions = np.tile([4, 5, 6], (len(rows), 1))
    targets = labels[np.arange(len(rows))[:, None], positions]
    expected = np.column_stack([np.asarray(rows)[:, 2], np.full(len(rows), 5), np.ones(len(rows))])
    np.testing.assert_array_equal(targets, expected)
    return tuple(torch.as_tensor(x, device=device) for x in (tokens, positions, targets))


def edit_objective(model, target_batch, replay_batch, parent_log, replay_weight=1.0):
    target = model(target_batch[0], positions=target_batch[1])
    ce = F.cross_entropy(target.flatten(0, 1), target_batch[2].flatten())
    replay = model(replay_batch[0], positions=replay_batch[1]).log_softmax(-1)
    if parent_log.requires_grad:
        raise ValueError("Replay reference must be detached")
    kl = F.kl_div(replay, parent_log, log_target=True, reduction="none").sum(-1).mean()
    return ce + replay_weight * kl, ce, kl


def _generate(model, rows, device, batch_size):
    if len(rows):
        return generate_rows(model, rows, device, batch_size=batch_size)
    return (
        {"n": 0, "accuracy": None, "answer_accuracy": None, "answer_nll": None},
        {
            "generated": np.empty((0, 3), dtype=np.int64),
            "correct": np.empty(0, dtype=bool),
            "answer_nll": np.empty(0),
        },
    )


def evaluate_case(model, tasks, originals, parent_correct, role, device, batch_size=512):
    """Full-pool scores and separately labelled prerequisite/parent subsets."""
    metrics, raw = {}, {}
    for name, rows in tasks.items():
        values, predictions = _generate(model, rows, device, batch_size)
        raw.update({f"{name}__{k}": v for k, v in predictions.items()})
        if name in originals:
            old = originals[name]
            generated = predictions["generated"]
            changed = old[:, -1] != rows[:, -1]
            old_correct = (generated[:, 0] == old[:, -1]) & np.all(
                generated[:, 1:] == [5, 1], axis=1
            )
            prerequisite = rows[:, [2, 3, 4] if role == "first" else [0, 1, 2]]
            _m, needed = _generate(model, prerequisite, device, batch_size)
            edited_ok = metrics["E_new"]["accuracy"] == 1.0
            covered = needed["correct"] & edited_ok
            baseline = parent_correct[name]
            valid_parent = baseline & changed
            correct = predictions["correct"]
            values.update(
                {
                    "changed_answer_n": int(changed.sum()),
                    "old_answer_accuracy": float(old_correct.mean()) if len(rows) else None,
                    "parent_correct_n": int(baseline.sum()),
                    "parent_correct_coverage": float(baseline.mean()) if len(rows) else None,
                    "new_following_parent_correct": float(correct[valid_parent].mean())
                    if valid_parent.any()
                    else None,
                    "old_retained_parent_correct": float(old_correct[valid_parent].mean())
                    if valid_parent.any()
                    else None,
                    "direct_edit_success": edited_ok,
                    "necessary_atomic_correct_n": int(needed["correct"].sum()),
                    "operation_correct_n": int(covered.sum()),
                    "operation_correct_coverage": float(covered.mean()) if len(rows) else None,
                    "new_following_operation_correct": float(correct[covered].mean())
                    if covered.any()
                    else None,
                }
            )
            raw[f"{name}__parent_correct"] = baseline
            raw[f"{name}__operation_correct"] = covered
            raw[f"{name}__old_correct"] = old_correct
            raw[f"{name}__necessary_generated"] = needed["generated"]
            raw[f"{name}__new_rows"] = rows
            raw[f"{name}__old_rows"] = old
        metrics[name] = values
    return metrics, raw


def editable_parameters(model, scope="mlp", block_index=0):
    prefix = f"blocks.{block_index}.mlp."
    selected = [
        name
        for name, _ in model.named_parameters()
        if name.startswith(prefix) and (scope == "mlp" or name == prefix + "down.weight")
    ]
    if scope not in {"mlp", "down"} or not selected:
        raise ValueError("Unknown or empty local parameter scope")
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in selected)
        parameter.grad = None
    return selected


def _configure(device):
    device = torch.device(device)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if device.type == "cuda":
        torch.cuda.set_device(device)
    return device


def _load_parent(parent_dir, device):
    path = Path(parent_dir) / "model.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = new_model(payload["spec"], device).eval()
    model.load_state_dict(payload["model"])
    world = build_world(payload["spec"])
    if (Path(parent_dir) / "world.npz").exists():
        with np.load(Path(parent_dir) / "world.npz") as saved:
            for name, rows in world.items():
                np.testing.assert_array_equal(rows, saved[name])
    return model, world, payload["spec"]


def _parent_masks(parent, originals, device, batch_size):
    return {
        name: _generate(parent, rows, device, batch_size)[1]["correct"]
        for name, rows in originals.items()
    }


def run(spec, out, device="cuda:0"):
    """Execute one parent, a graph-selected case set and optional development lr grid.

    The caller owns source freezing, resource allocation and W&B startup. Each
    branch has run/learning/complete artifacts. Root ``step`` is a strictly
    increasing evaluation index, including every branch's unchanged node zero;
    ``optimizer_updates`` separately records cumulative optimization work.
    Target labels and replay KL positions have separate exposure counters.
    ``status.json`` retains completed-branch progress, explicitly labelled in
    the manifest; it does not define the tracking series' step axis.
    """
    device = _configure(device)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "run.json").exists() or (out / "complete.json").exists():
        raise FileExistsError(f"Do not overwrite an existing attempt: {out}")
    nodes = spec.get("nodes", list(DEFAULT_NODES))
    if not nodes or nodes[0] != 0 or nodes != sorted(set(nodes)) or nodes[-1] <= 0:
        raise ValueError("Nodes must increase from zero to a positive fixed endpoint")
    rates = spec.get("learning_rates", [spec.get("lr", 1e-4)])
    if not rates or any(lr <= 0 for lr in rates) or len(set(rates)) != len(rates):
        raise ValueError("Learning rates must be positive and unique")
    if spec.get("phase") not in {"development", "engineering"} and len(rates) > 1:
        raise ValueError("Learning-rate grids are development-only")
    arms = spec.get("arms", ["edit", "sham"])
    if sorted(arms) != ["edit", "sham"]:
        raise ValueError("Every case requires paired edit and sham branches")
    parent, world, parent_spec = _load_parent(spec["parent_dir"], device)
    parent_hash = model_digest(parent)
    cases, qualification = graph_cases(
        parent_spec,
        seed=spec.get("candidate_seed", 780011),
        per_cell=spec.get("per_cell", 2),
        replay_n=spec.get("replay_n", 32),
    )
    if "case_ids" in spec:
        requested = set(spec["case_ids"])
        if not requested or not requested <= {c["case_id"] for c in cases}:
            raise ValueError("Unknown/empty fixed graph-selected case subset")
        cases = [c for c in cases if c["case_id"] in requested]
    manifest = {
        "spec": spec,
        "parent_spec": parent_spec,
        "cases": cases,
        "qualification": qualification,
        "case_sha256": _hash_json(cases),
        "parent_file_sha256": file_hash(Path(spec["parent_dir"]) / "model.pt"),
        "parent_model_sha256": parent_hash,
        "world_sha256": data_digest(world),
        "created_utc": utc(),
        "dtype": "float32",
        "allow_tf32": True,
        "pid": os.getpid(),
        "gpu": device.index if device.type == "cuda" else None,
        "branches": len(cases) * len(rates) * len(arms),
        "planned_updates": len(cases) * len(rates) * len(arms) * nodes[-1],
        "planned_evaluations": len(cases) * len(rates) * len(arms) * len(nodes),
        "learning_step_unit": "evaluation_index",
        "status_step_unit": "completed_branches",
        "scope_warning": "The selected MLP is shared across all repeated executions",
    }
    write_json(out / "run.json", manifest)
    write_json(out / "cases.json", cases)
    np.savez_compressed(out / "world.npz", **world)
    root_learning, branch_summaries, completed_steps = [], [], 0
    started = time.perf_counter()
    training_seconds = 0.0
    batch_size = spec.get("eval_batch_size", 512)
    for case in cases:
        tasks, originals = case_tasks(world, case)
        parent_correct = _parent_masks(parent, originals, device, batch_size)
        parent_metrics, parent_predictions = evaluate_case(
            parent,
            tasks,
            originals,
            parent_correct,
            case["role"],
            device,
            batch_size,
        )
        case_out = out / case["case_id"]
        case_out.mkdir()
        np.savez_compressed(case_out / "parent-predictions.npz", **parent_predictions)
        write_json(case_out / "parent-metrics.json", parent_metrics)
        np.savez_compressed(case_out / "tasks.npz", **tasks)
        replay_batch = answer_batch(tasks["R_atomic"], device)
        with torch.no_grad():
            parent_log = parent(replay_batch[0], positions=replay_batch[1]).log_softmax(-1).detach()
        for lr in rates:
            for arm in arms:
                branch_name = f"lr{lr:.8g}-{arm}"
                branch_out = case_out / branch_name
                branch_out.mkdir()
                model = copy.deepcopy(parent).eval()
                selected = editable_parameters(
                    model,
                    spec.get("parameter_scope", "mlp"),
                    spec.get("block_index", 0),
                )
                parameters = dict(model.named_parameters())
                optimizer = torch.optim.Adam([parameters[n] for n in selected], lr=lr)
                target = tasks["E_new" if arm == "edit" else "E_old"]
                target_batch = answer_batch(target, device)
                write_json(
                    branch_out / "run.json",
                    {
                        "case_id": case["case_id"],
                        "arm": arm,
                        "lr": lr,
                        "parent_model_sha256": parent_hash,
                        "editable_parameters": selected,
                        "optimizer": "Adam, fresh state, zero weight decay",
                        "sampling": "same fixed full replay batch on every update; no dropout",
                        "old_fact_sha256": _hash_json(case["old_fact"]),
                        "new_fact_sha256": _hash_json(case["new_fact"]),
                        "target_batch_sha256": _batch_hash(target_batch),
                        "replay_batch_sha256": _batch_hash(replay_batch),
                        "parent_reference_sha256": _batch_hash((parent_log,)),
                        "nodes": nodes,
                    },
                )
                np.savez_compressed(
                    branch_out / "replay-reference.npz", log_probabilities=parent_log.cpu().numpy()
                )
                step, losses, history = 0, [], []
                for node in nodes:
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    began = time.perf_counter()
                    while step < node:
                        optimizer.zero_grad(set_to_none=True)
                        loss, ce, kl = edit_objective(
                            model,
                            target_batch,
                            replay_batch,
                            parent_log,
                            spec.get("replay_weight", 1.0),
                        )
                        if not torch.isfinite(loss):
                            raise FloatingPointError(f"Nonfinite edit loss at {step}")
                        loss.backward()
                        gradient_norm = torch.nn.utils.clip_grad_norm_(
                            [parameters[n] for n in selected],
                            spec.get("clip_norm", 1.0),
                        )
                        optimizer.step()
                        step += 1
                        losses.append(
                            {
                                "step": step,
                                "loss": float(loss.detach()),
                                "target_ce": float(ce.detach()),
                                "replay_kl": float(kl.detach()),
                                "gradient_norm": float(gradient_norm),
                            }
                        )
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    training_seconds += time.perf_counter() - began
                    metrics, predictions = evaluate_case(
                        model,
                        tasks,
                        originals,
                        parent_correct,
                        case["role"],
                        device,
                        batch_size,
                    )
                    current_hash = model_digest(model)
                    record = {
                        "step": step,
                        "metrics": metrics,
                        "model_sha256": current_hash,
                        "target_success": metrics["E_new" if arm == "edit" else "E_old"]["accuracy"]
                        == 1,
                        "U_atomic_at_least_95": metrics["U_atomic"]["accuracy"] >= 0.95,
                        "loss": losses[-1] if losses else None,
                    }
                    history.append(record)
                    write_json(branch_out / "learning.json", history)
                    write_json(branch_out / "losses.json", losses)
                    np.savez_compressed(branch_out / f"predictions-{step:06d}.npz", **predictions)
                    torch.save(
                        {
                            "edited_tensors": {
                                n: parameters[n].detach().cpu().clone() for n in selected
                            },
                            "model_sha256": current_hash,
                            "step": step,
                        },
                        branch_out / f"tensors-{step:06d}.pt",
                    )
                    root_learning.append(
                        {
                            "step": len(root_learning),
                            "branch_step": step,
                            "completed_branches": len(branch_summaries) + int(step == nodes[-1]),
                            "optimizer_updates": completed_steps + step,
                            "case_id": case["case_id"],
                            "arm": arm,
                            "lr": lr,
                            "metrics": metrics,
                            "training_seconds": training_seconds,
                            "wall_seconds": time.perf_counter() - started,
                            "examples": (completed_steps + step) * (1 + len(tasks["R_atomic"])),
                            "target_supervised_tokens": (completed_steps + step) * 3,
                            "replay_distillation_positions": (completed_steps + step)
                            * len(tasks["R_atomic"])
                            * 3,
                        }
                    )
                    write_json(out / "learning.json", root_learning)
                    write_json(
                        out / "status.json",
                        {
                            "state": "running",
                            "step": len(branch_summaries),
                            "optimizer_updates": completed_steps + step,
                            "budget": manifest["branches"],
                            "case_id": case["case_id"],
                            "arm": arm,
                            "lr": lr,
                        },
                    )
                changed = [
                    n
                    for n, t in model.state_dict().items()
                    if not torch.equal(t, parent.state_dict()[n])
                ]
                if set(changed) - set(selected) or model_digest(parent) != parent_hash:
                    raise AssertionError("A frozen parameter or the parent changed")
                complete = {
                    "case_id": case["case_id"],
                    "role": case["role"],
                    "stratum": case["stratum"],
                    "lr": lr,
                    "arm": arm,
                    "path": str(branch_out.relative_to(out)),
                    "metrics": metrics,
                    "steps": step,
                    "changed_tensors": changed,
                    "parent_model_sha256": parent_hash,
                    "final_model_sha256": current_hash,
                    "delta_l2": float(
                        torch.sqrt(
                            sum(
                                (parameters[n] - dict(parent.named_parameters())[n]).square().sum()
                                for n in selected
                            )
                        ).detach()
                    ),
                    "tensor_file_sha256": file_hash(branch_out / f"tensors-{step:06d}.pt"),
                }
                write_json(branch_out / "complete.json", complete)
                branch_summaries.append(complete)
                completed_steps += step
    result = {
        "status": "complete",
        "branches": branch_summaries,
        "updates": completed_steps,
        "training_seconds": training_seconds,
        "wall_seconds": time.perf_counter() - started,
        "finished_utc": utc(),
        "case_sha256": manifest["case_sha256"],
    }
    write_json(out / "complete.json", result)
    write_json(
        out / "status.json",
        {
            "state": "complete",
            "step": len(branch_summaries),
            "optimizer_updates": completed_steps,
            "budget": manifest["branches"],
        },
    )
    return result


def audit(out, device="cuda:0"):
    """Reload every saved node from parent plus exact tensors and re-score it."""
    device, out = _configure(device), Path(out)
    manifest = json.loads((out / "run.json").read_text())
    spec = manifest["spec"]
    if file_hash(Path(spec["parent_dir"]) / "model.pt") != manifest["parent_file_sha256"]:
        raise AssertionError("Parent checkpoint hash changed")
    parent, world, parent_spec = _load_parent(spec["parent_dir"], device)
    if model_digest(parent) != manifest["parent_model_sha256"]:
        raise AssertionError("Parent model digest changed")
    cases, qualification = graph_cases(
        parent_spec,
        seed=spec.get("candidate_seed", 780011),
        per_cell=spec.get("per_cell", 2),
        replay_n=spec.get("replay_n", 32),
    )
    if "case_ids" in spec:
        cases = [c for c in cases if c["case_id"] in spec["case_ids"]]
    if cases != manifest["cases"] or _hash_json(cases) != manifest["case_sha256"]:
        raise AssertionError("Graph-selected cases changed")
    if qualification != manifest["qualification"] or data_digest(world) != manifest["world_sha256"]:
        raise AssertionError("World qualification changed")
    summary = json.loads((out / "complete.json").read_text())
    rates = spec.get("learning_rates", [spec.get("lr", 1e-4)])
    expected_branches = {
        (c["case_id"], lr, arm) for c in cases for lr in rates for arm in ("edit", "sham")
    }
    actual_branches = [(b["case_id"], b["lr"], b["arm"]) for b in summary["branches"]]
    if len(actual_branches) != len(expected_branches) or set(actual_branches) != expected_branches:
        raise AssertionError("The fixed paired branch matrix is incomplete")
    root_learning = json.loads((out / "learning.json").read_text())
    nodes = spec.get("nodes", list(DEFAULT_NODES))
    expected_evaluations = len(actual_branches) * len(nodes)
    if len(root_learning) != expected_evaluations or [row["step"] for row in root_learning] != list(
        range(expected_evaluations)
    ):
        raise AssertionError("Root learning evaluation indices are incomplete or repeated")
    for index, row in enumerate(root_learning):
        branch_index, node_index = divmod(index, len(nodes))
        updates = branch_index * nodes[-1] + nodes[node_index]
        if (
            (row["case_id"], row["lr"], row["arm"]) != actual_branches[branch_index]
            or row["branch_step"] != nodes[node_index]
            or row["optimizer_updates"] != updates
            or row["target_supervised_tokens"] != updates * 3
            or row["replay_distillation_positions"] != updates * spec.get("replay_n", 32) * 3
            or "supervised_tokens" in row
        ):
            raise AssertionError("Root learning optimizer or exposure accounting changed")
    batch_size = spec.get("eval_batch_size", 512)
    checked_nodes, checked_predictions, max_error = 0, 0, 0.0
    for case in cases:
        tasks, originals = case_tasks(world, case)
        parent_correct = _parent_masks(parent, originals, device, batch_size)
        for branch in summary["branches"]:
            if branch["case_id"] != case["case_id"]:
                continue
            directory = out / branch["path"]
            branch_spec = json.loads((directory / "run.json").read_text())
            target = tasks["E_new" if branch["arm"] == "edit" else "E_old"]
            target_batch = answer_batch(target, device)
            replay_batch = answer_batch(tasks["R_atomic"], device)
            with torch.no_grad():
                reference = parent(replay_batch[0], positions=replay_batch[1]).log_softmax(-1)
            expected_hashes = {
                "old_fact_sha256": _hash_json(case["old_fact"]),
                "new_fact_sha256": _hash_json(case["new_fact"]),
                "target_batch_sha256": _batch_hash(target_batch),
                "replay_batch_sha256": _batch_hash(replay_batch),
                "parent_reference_sha256": _batch_hash((reference,)),
            }
            if any(branch_spec[key] != value for key, value in expected_hashes.items()):
                raise AssertionError("Target or paired replay provenance changed")
            history = json.loads((directory / "learning.json").read_text())
            if [record["step"] for record in history] != spec.get("nodes", list(DEFAULT_NODES)):
                raise AssertionError("Saved edit node matrix is incomplete")
            if (
                file_hash(directory / f"tensors-{history[-1]['step']:06d}.pt")
                != branch["tensor_file_sha256"]
            ):
                raise AssertionError("Final exact tensor file hash changed")
            for record in history:
                state = torch.load(
                    directory / f"tensors-{record['step']:06d}.pt",
                    map_location="cpu",
                    weights_only=True,
                )
                model = copy.deepcopy(parent).eval()
                selected = editable_parameters(
                    model, spec.get("parameter_scope", "mlp"), spec.get("block_index", 0)
                )
                if set(state["edited_tensors"]) != set(selected):
                    raise AssertionError("Unexpected edited tensor scope")
                merged = model.state_dict()
                merged.update(state["edited_tensors"])
                model.load_state_dict(merged)
                if model_digest(model) != record["model_sha256"]:
                    raise AssertionError("Exact saved tensors did not reconstruct the model")
                metrics, predictions = evaluate_case(
                    model,
                    tasks,
                    originals,
                    parent_correct,
                    case["role"],
                    device,
                    batch_size,
                )
                with np.load(directory / f"predictions-{record['step']:06d}.npz") as saved:
                    if set(saved.files) != set(predictions):
                        raise AssertionError("Prediction keys changed")
                    for name, actual in predictions.items():
                        if name.endswith("answer_nll"):
                            np.testing.assert_allclose(actual, saved[name], rtol=1e-5, atol=1e-5)
                            if len(actual):
                                max_error = max(
                                    max_error, float(np.max(np.abs(actual - saved[name])))
                                )
                        else:
                            np.testing.assert_array_equal(actual, saved[name])
                        checked_predictions += actual.size
                for name, values in metrics.items():
                    for key, value in values.items():
                        expected = record["metrics"][name][key]
                        if key == "answer_nll" and value is not None:
                            np.testing.assert_allclose(value, expected, rtol=1e-5, atol=1e-5)
                        elif value != expected:
                            raise AssertionError(f"Re-scoring mismatch: {name}/{key}")
                checked_nodes += 1
    result = {
        "passed": True,
        "nodes": checked_nodes,
        "array_elements": checked_predictions,
        "max_nll_error": max_error,
        "case_sha256": manifest["case_sha256"],
        "finished_utc": utc(),
    }
    write_json(out / "audit.json", result)
    return result
