"""Does writing the same new entity into the early state restore its later use?

The three objectives use identical atomic labels, replay, shared-MLP parameters,
and update counts. Only the objective weights change. Graph-only strict first-hop
cases and the full-pool evaluator are inherited from the established edit study.
No downstream relation or composed answer is passed to the optimizer.
"""

from __future__ import annotations

import copy
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from . import interface_editing as ie
from .grok_depth import utc, write_json
from .latent_scaling import model_digest
from .storage_composition import data_digest, file_hash

OBJECTIVES = ("final_ce", "early_ce", "early_ce_align")


def objective(model, target_batch, replay_batch, parent_log, kind, spec):
    """All early supervision is the exact entity label of the atomic answer."""
    if kind not in OBJECTIVES:
        raise ValueError(f"Unknown writing objective: {kind}")
    if parent_log.requires_grad:
        raise ValueError("Replay reference must be detached")
    logits, state = model(target_batch[0], positions=target_batch[1], return_bridge=True)
    final_ce = F.cross_entropy(logits.flatten(0, 1), target_batch[2].flatten())
    entity = target_batch[2][:, 0]
    early_logits = F.linear(model.ln_final(state), model.token.weight)
    early_ce = F.cross_entropy(early_logits, entity)
    canonical = model.token(entity).detach()
    alignment = (
        (F.normalize(state, dim=-1) - F.normalize(canonical, dim=-1)).square().sum(-1).mean()
    )
    replay = model(replay_batch[0], positions=replay_batch[1]).log_softmax(-1)
    kl = F.kl_div(replay, parent_log, log_target=True, reduction="none").sum(-1).mean()
    # Compute the same graph in all arms, including the zero-weight components.
    early_weight = spec.get("early_weight", 0.3) if kind != "final_ce" else 0.0
    alignment_weight = spec.get("alignment_weight", 0.3) if kind == "early_ce_align" else 0.0
    loss = (
        final_ce
        + spec.get("replay_weight", 1.0) * kl
        + early_weight * early_ce
        + alignment_weight * alignment
    )
    return loss, {
        "final_ce": final_ce,
        "replay_kl": kl,
        "early_ce": early_ce,
        "alignment": alignment,
    }


@torch.no_grad()
def early_readout(model, case, device):
    """Read the causal first-block state without fitting a probe or using D."""
    rows = np.asarray([case["new_fact"]], dtype=np.int64)
    tokens, positions, _ = ie.answer_batch(rows, device)
    _, state = model(tokens, positions=positions, return_bridge=True)
    logits = F.linear(model.ln_final(state), model.token.weight)
    old, new = int(case["old_fact"][2]), int(case["new_fact"][2])
    probabilities = logits.softmax(-1)
    old_embedding = model.token.weight[old : old + 1]
    new_embedding = model.token.weight[new : new + 1]
    prediction = int(logits.argmax(-1).item())
    metrics = {
        "n": 1,
        "new_entity_accuracy": float(prediction == new),
        "old_entity_accuracy": float(prediction == old),
        "new_entity_probability": float(probabilities[0, new]),
        "old_entity_probability": float(probabilities[0, old]),
        "new_entity_nll": float(-logits.log_softmax(-1)[0, new]),
        "new_direction_cosine": float(F.cosine_similarity(state, new_embedding).item()),
        "old_direction_cosine": float(F.cosine_similarity(state, old_embedding).item()),
        "state_norm": float(state.norm()),
    }
    raw = {
        "early__state": state.cpu().numpy(),
        "early__entity_prediction": np.asarray([prediction], dtype=np.int64),
        "early__entity_probabilities": probabilities[:, [old, new]].cpu().numpy(),
    }
    return metrics, raw


def selected_cases(parent_spec, spec):
    cases, qualification = ie.graph_cases(
        parent_spec,
        seed=spec.get("candidate_seed", 780101),
        per_cell=spec.get("per_cell", 4),
        replay_n=spec.get("replay_n", 32),
    )
    cases = [case for case in cases if case["role"] == "first" and case["stratum"] == "strict"]
    if "case_ids" in spec:
        wanted = set(spec["case_ids"])
        if not wanted or not wanted <= {case["case_id"] for case in cases}:
            raise ValueError("Only graph-selected strict first-hop cases are permitted")
        cases = [case for case in cases if case["case_id"] in wanted]
    return cases, qualification


def _evaluate(model, case, tasks, originals, parent_correct, device, batch_size):
    metrics, raw = ie.evaluate_case(
        model, tasks, originals, parent_correct, "first", device, batch_size
    )
    metrics["early"], early_raw = early_readout(model, case, device)
    raw.update(early_raw)
    return metrics, raw


def _restore(parent, tensor_path, scope="mlp", block_index=0):
    state = torch.load(tensor_path, map_location="cpu", weights_only=True)
    model = copy.deepcopy(parent).eval()
    selected = ie.editable_parameters(model, scope, block_index)
    if set(state["edited_tensors"]) != set(selected):
        raise AssertionError("Saved edit changes an unexpected parameter scope")
    merged = model.state_dict()
    merged.update(state["edited_tensors"])
    model.load_state_dict(merged)
    if model_digest(model) != state["model_sha256"]:
        raise AssertionError("Saved exact tensors do not reconstruct their model digest")
    return model


def diagnostics(existing_dir, out, device="cuda:0"):
    """Reload existing final-answer edits before any new training; never mutate them."""
    device, existing_dir, out = ie._configure(device), Path(existing_dir), Path(out)
    if out.exists():
        raise FileExistsError(out)
    manifest = json.loads((existing_dir / "run.json").read_text())
    spec = manifest["spec"]
    parent, _world, parent_spec = ie._load_parent(spec["parent_dir"], device)
    if model_digest(parent) != manifest["parent_model_sha256"]:
        raise AssertionError("Historical parent digest changed")
    cases, _ = ie.graph_cases(
        parent_spec,
        seed=spec.get("candidate_seed", 780011),
        per_cell=spec.get("per_cell", 2),
        replay_n=spec.get("replay_n", 32),
    )
    cases = {c["case_id"]: c for c in cases if c["role"] == "first" and c["stratum"] == "strict"}
    records = []
    summary = json.loads((existing_dir / "complete.json").read_text())
    for branch in summary["branches"]:
        if branch["case_id"] not in cases:
            continue
        case = cases[branch["case_id"]]
        directory = existing_dir / branch["path"]
        parent_metrics, parent_raw = early_readout(parent, case, device)
        for record in json.loads((directory / "learning.json").read_text()):
            tensor_path = directory / f"tensors-{record['step']:06d}.pt"
            model = _restore(parent, tensor_path, spec.get("parameter_scope", "mlp"))
            if model_digest(model) != record["model_sha256"]:
                raise AssertionError("Historical trajectory digest changed")
            values, raw = early_readout(model, case, device)
            values["state_delta_l2"] = float(
                np.linalg.norm(raw["early__state"] - parent_raw["early__state"])
            )
            records.append(
                {
                    "case_id": case["case_id"],
                    "arm": branch["arm"],
                    "step": record["step"],
                    "early": values,
                    "parent_early": parent_metrics,
                    "E_new": record["metrics"]["E_new"],
                    "E_old": record["metrics"]["E_old"],
                    "tensor_file_sha256": file_hash(tensor_path),
                    "model_sha256": model_digest(model),
                }
            )
    if not records:
        raise ValueError("Historical run contains no strict first-hop edit nodes")
    result = {
        "status": "complete",
        "training_updates": 0,
        "existing_dir": str(existing_dir),
        "parent_model_sha256": manifest["parent_model_sha256"],
        "source_run_sha256": file_hash(existing_dir / "run.json"),
        "records": records,
        "finished_utc": utc(),
    }
    write_json(out, result)
    return result


def _validate(spec):
    nodes = spec.get("nodes", [0, 32, 128, 512])
    if not nodes or nodes[0] != 0 or nodes != sorted(set(nodes)) or nodes[-1] <= 0:
        raise ValueError("Nodes must increase from zero to a positive fixed endpoint")
    kinds = spec.get("objectives", list(OBJECTIVES))
    if not kinds or len(set(kinds)) != len(kinds) or set(kinds) - set(OBJECTIVES):
        raise ValueError("Unknown or repeated objective")
    if spec.get("lr", 1e-5) <= 0 or "learning_rates" in spec:
        raise ValueError("Use one fixed positive learning rate")
    if spec.get("parameter_scope", "mlp") != "mlp" or spec.get("block_index", 0) != 0:
        raise ValueError("This comparison edits the same first shared MLP only")
    return nodes, kinds


def run(spec, out, device="cuda:0"):
    """One parent and one or more fixed objectives, each with all edit/sham pairs."""
    device, out = ie._configure(device), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "run.json").exists() or (out / "complete.json").exists():
        raise FileExistsError(f"Do not overwrite an existing attempt: {out}")
    nodes, kinds = _validate(spec)
    parent, world, parent_spec = ie._load_parent(spec["parent_dir"], device)
    parent_file_hash = file_hash(Path(spec["parent_dir"]) / "model.pt")
    if spec.get("parent_checkpoint_sha256", parent_file_hash) != parent_file_hash:
        raise AssertionError("Preselected parent checkpoint hash changed")
    if parent_spec["layers"] != 1 or parent_spec["repeats"] != 2:
        raise ValueError("Use the established two-execution shared GPT parent")
    parent_hash = model_digest(parent)
    cases, qualification = selected_cases(parent_spec, spec)
    manifest = {
        "spec": spec,
        "parent_spec": parent_spec,
        "cases": cases,
        "qualification": qualification,
        "case_sha256": ie._hash_json(cases),
        "parent_file_sha256": parent_file_hash,
        "parent_model_sha256": parent_hash,
        "initial_model_sha256": parent_hash,
        "world_sha256": data_digest(world),
        "created_utc": utc(),
        "dtype": "float32",
        "allow_tf32": True,
        "pid": os.getpid(),
        "gpu": device.index if device.type == "cuda" else None,
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "branches": len(cases) * len(kinds) * 2,
        "planned_updates": len(cases) * len(kinds) * 2 * nodes[-1],
        "learning_step_unit": "evaluation_index",
        "status_step_unit": "completed_branches",
        "early_state": "first executed block at first relation token, position 3",
        "scope_warning": "Selected MLP parameters are shared across both executions",
        "supervision": "same atomic entity label; no second relation or composed answer",
    }
    write_json(out / "run.json", manifest)
    write_json(out / "learning.json", [])
    write_json(out / "cases.json", cases)
    np.savez_compressed(out / "world.npz", **world)
    if spec.get("diagnostic_existing_dir"):
        diagnostic = diagnostics(
            spec["diagnostic_existing_dir"], out / "existing-edit-diagnostics.json", device
        )
        if diagnostic["parent_model_sha256"] != parent_hash:
            raise AssertionError("Diagnostic and training parent must be identical")
    root_learning, summaries = [], []
    completed_updates, early_presentations, alignment_presentations = 0, 0, 0
    started, training_seconds = time.perf_counter(), 0.0
    batch_size = spec.get("eval_batch_size", 512)
    for case in cases:
        tasks, originals = ie.case_tasks(world, case)
        masks = ie._parent_masks(parent, originals, device, batch_size)
        parent_metrics, parent_raw = _evaluate(
            parent, case, tasks, originals, masks, device, batch_size
        )
        case_out = out / case["case_id"]
        case_out.mkdir()
        write_json(case_out / "parent-metrics.json", parent_metrics)
        np.savez_compressed(case_out / "parent-predictions.npz", **parent_raw)
        np.savez_compressed(case_out / "tasks.npz", **tasks)
        replay = ie.answer_batch(tasks["R_atomic"], device)
        with torch.no_grad():
            reference = parent(replay[0], positions=replay[1]).log_softmax(-1).detach()
        for kind in kinds:
            for arm in ("edit", "sham"):
                directory = case_out / f"{kind}-{arm}"
                directory.mkdir()
                model = copy.deepcopy(parent).eval()
                selected = ie.editable_parameters(model)
                parameters = dict(model.named_parameters())
                optimizer = torch.optim.Adam(
                    [parameters[n] for n in selected], lr=spec.get("lr", 1e-5)
                )
                target_name = "E_new" if arm == "edit" else "E_old"
                target = ie.answer_batch(tasks[target_name], device)
                write_json(
                    directory / "run.json",
                    {
                        "case_id": case["case_id"],
                        "objective": kind,
                        "arm": arm,
                        "editable_parameters": selected,
                        "parent_model_sha256": parent_hash,
                        "target_batch_sha256": ie._batch_hash(target),
                        "replay_batch_sha256": ie._batch_hash(replay),
                        "parent_reference_sha256": ie._batch_hash((reference,)),
                        "optimizer": "Adam, fresh state, zero weight decay",
                        "sampling": "same fixed full replay batch every update; no dropout",
                        "nodes": nodes,
                    },
                )
                history, losses, step = [], [], 0
                for node in nodes:
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    began = time.perf_counter()
                    while step < node:
                        optimizer.zero_grad(set_to_none=True)
                        loss, components = objective(model, target, replay, reference, kind, spec)
                        if not torch.isfinite(loss):
                            raise FloatingPointError(f"Nonfinite loss at {case['case_id']}/{step}")
                        loss.backward()
                        norm = torch.nn.utils.clip_grad_norm_(
                            [parameters[n] for n in selected], spec.get("clip_norm", 1.0)
                        )
                        optimizer.step()
                        step += 1
                        losses.append(
                            {
                                "step": step,
                                "loss": float(loss.detach()),
                                "gradient_norm": float(norm),
                                **{key: float(value.detach()) for key, value in components.items()},
                            }
                        )
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    training_seconds += time.perf_counter() - began
                    metrics, predictions = _evaluate(
                        model, case, tasks, originals, masks, device, batch_size
                    )
                    digest = model_digest(model)
                    record = {
                        "step": step,
                        "model_sha256": digest,
                        "metrics": metrics,
                        "loss": losses[-1] if losses else None,
                    }
                    history.append(record)
                    write_json(directory / "learning.json", history)
                    write_json(directory / "losses.json", losses)
                    np.savez_compressed(directory / f"predictions-{step:06d}.npz", **predictions)
                    torch.save(
                        {
                            "edited_tensors": {
                                n: parameters[n].detach().cpu().clone() for n in selected
                            },
                            "model_sha256": digest,
                            "step": step,
                        },
                        directory / f"tensors-{step:06d}.pt",
                    )
                    updates = completed_updates + step
                    root_learning.append(
                        {
                            "step": len(root_learning),
                            "branch_step": step,
                            "optimizer_updates": updates,
                            "case_id": case["case_id"],
                            "condition": kind,
                            "arm": arm,
                            "metrics": metrics,
                            "training_seconds": training_seconds,
                            "wall_seconds": time.perf_counter() - started,
                            "examples": updates * (1 + len(tasks["R_atomic"])),
                            "target_supervised_tokens": updates * 3,
                            "replay_distillation_positions": updates * len(tasks["R_atomic"]) * 3,
                            "early_target_presentations": early_presentations
                            + step * (kind != "final_ce"),
                            "alignment_target_presentations": alignment_presentations
                            + step * (kind == "early_ce_align"),
                        }
                    )
                    write_json(out / "learning.json", root_learning)
                    write_json(
                        out / "status.json",
                        {
                            "state": "running",
                            "step": len(summaries),
                            "optimizer_updates": updates,
                            "budget": manifest["branches"],
                            "case_id": case["case_id"],
                            "objective": kind,
                            "arm": arm,
                        },
                    )
                    print(
                        json.dumps(
                            {
                                "case": case["case_id"],
                                "objective": kind,
                                "arm": arm,
                                "step": step,
                                "E": metrics[target_name]["accuracy"],
                                "early_new": metrics["early"]["new_entity_accuracy"],
                                "D": metrics["D_first_strict"]["accuracy"],
                            }
                        ),
                        flush=True,
                    )
                changed = [
                    n
                    for n, value in model.state_dict().items()
                    if not torch.equal(value, parent.state_dict()[n])
                ]
                if set(changed) - set(selected) or model_digest(parent) != parent_hash:
                    raise AssertionError("A frozen parameter or the parent changed")
                complete = {
                    "case_id": case["case_id"],
                    "role": "first",
                    "stratum": "strict",
                    "objective": kind,
                    "arm": arm,
                    "path": str(directory.relative_to(out)),
                    "metrics": metrics,
                    "steps": step,
                    "changed_tensors": changed,
                    "parent_model_sha256": parent_hash,
                    "final_model_sha256": digest,
                    "delta_l2": float(
                        torch.sqrt(
                            sum(
                                (parameters[n] - dict(parent.named_parameters())[n]).square().sum()
                                for n in selected
                            )
                        ).detach()
                    ),
                    "tensor_file_sha256": file_hash(directory / f"tensors-{step:06d}.pt"),
                }
                write_json(directory / "complete.json", complete)
                summaries.append(complete)
                completed_updates += step
                early_presentations += step * (kind != "final_ce")
                alignment_presentations += step * (kind == "early_ce_align")
    result = {
        "status": "complete",
        "branches": summaries,
        "updates": completed_updates,
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
            "step": len(summaries),
            "optimizer_updates": completed_updates,
            "budget": manifest["branches"],
        },
    )
    return result


def audit(out, device="cuda:0"):
    """Independently reconstruct every branch/node and compare stored predictions."""
    device, out = ie._configure(device), Path(out)
    manifest = json.loads((out / "run.json").read_text())
    spec = manifest["spec"]
    nodes, kinds = _validate(spec)
    parent, world, parent_spec = ie._load_parent(spec["parent_dir"], device)
    if (
        model_digest(parent) != manifest["parent_model_sha256"]
        or file_hash(Path(spec["parent_dir"]) / "model.pt") != manifest["parent_file_sha256"]
    ):
        raise AssertionError("Parent changed")
    cases, qualification = selected_cases(parent_spec, spec)
    if (
        cases != manifest["cases"]
        or qualification != manifest["qualification"]
        or data_digest(world) != manifest["world_sha256"]
    ):
        raise AssertionError("Graph cases or world changed")
    result = json.loads((out / "complete.json").read_text())
    branches = result["branches"]
    expected = [
        (case["case_id"], kind, arm) for case in cases for kind in kinds for arm in ("edit", "sham")
    ]
    if [(b["case_id"], b["objective"], b["arm"]) for b in branches] != expected:
        raise AssertionError("Incomplete paired objective matrix")
    learning = json.loads((out / "learning.json").read_text())
    if len(learning) != len(expected) * len(nodes) or [r["step"] for r in learning] != list(
        range(len(learning))
    ):
        raise AssertionError("Root evaluation indices changed")
    early_count, alignment_count = 0, 0
    for index, row in enumerate(learning):
        branch_index, node_index = divmod(index, len(nodes))
        case_id, kind, arm = expected[branch_index]
        node = nodes[node_index]
        updates = branch_index * nodes[-1] + node
        if (row["case_id"], row["condition"], row["arm"], row["branch_step"]) != (
            case_id,
            kind,
            arm,
            node,
        ):
            raise AssertionError("Root branch ordering changed")
        wanted = {
            "optimizer_updates": updates,
            "target_supervised_tokens": updates * 3,
            "replay_distillation_positions": updates * spec.get("replay_n", 32) * 3,
            "early_target_presentations": early_count + node * (kind != "final_ce"),
            "alignment_target_presentations": alignment_count + node * (kind == "early_ce_align"),
        }
        if any(row[key] != value for key, value in wanted.items()):
            raise AssertionError("Optimizer or exposure accounting changed")
        if node_index == len(nodes) - 1:
            early_count += node * (kind != "final_ce")
            alignment_count += node * (kind == "early_ce_align")
    checked, max_error = 0, 0.0
    batch_size = spec.get("eval_batch_size", 512)
    for case in cases:
        tasks, originals = ie.case_tasks(world, case)
        masks = ie._parent_masks(parent, originals, device, batch_size)
        replay = ie.answer_batch(tasks["R_atomic"], device)
        with torch.no_grad():
            reference = parent(replay[0], positions=replay[1]).log_softmax(-1)
        for branch in branches:
            if branch["case_id"] != case["case_id"]:
                continue
            directory = out / branch["path"]
            branch_spec = json.loads((directory / "run.json").read_text())
            target = ie.answer_batch(tasks["E_new" if branch["arm"] == "edit" else "E_old"], device)
            hashes = {
                "target_batch_sha256": ie._batch_hash(target),
                "replay_batch_sha256": ie._batch_hash(replay),
                "parent_reference_sha256": ie._batch_hash((reference,)),
            }
            if any(branch_spec[key] != value for key, value in hashes.items()):
                raise AssertionError("Supervision or replay provenance changed")
            history = json.loads((directory / "learning.json").read_text())
            if [r["step"] for r in history] != nodes:
                raise AssertionError("Saved node matrix incomplete")
            if (
                branch["metrics"] != history[-1]["metrics"]
                or branch["final_model_sha256"] != history[-1]["model_sha256"]
            ):
                raise AssertionError("Final summary differs from audited trajectory")
            if file_hash(directory / f"tensors-{nodes[-1]:06d}.pt") != branch["tensor_file_sha256"]:
                raise AssertionError("Final tensor hash changed")
            for record in history:
                model = _restore(parent, directory / f"tensors-{record['step']:06d}.pt")
                if model_digest(model) != record["model_sha256"]:
                    raise AssertionError("Recorded model digest changed")
                metrics, raw = _evaluate(model, case, tasks, originals, masks, device, batch_size)
                with np.load(directory / f"predictions-{record['step']:06d}.npz") as saved:
                    if set(saved.files) != set(raw):
                        raise AssertionError("Prediction keys changed")
                    for name, values in raw.items():
                        if values.dtype.kind == "f":
                            np.testing.assert_allclose(values, saved[name], rtol=1e-5, atol=1e-5)
                            if values.size:
                                max_error = max(
                                    max_error, float(np.max(np.abs(values - saved[name])))
                                )
                        else:
                            np.testing.assert_array_equal(values, saved[name])
                for name, values in metrics.items():
                    for key, value in values.items():
                        expected_value = record["metrics"][name][key]
                        if isinstance(value, float):
                            np.testing.assert_allclose(value, expected_value, rtol=1e-5, atol=1e-5)
                        elif value != expected_value:
                            raise AssertionError(f"Metric changed: {name}/{key}")
                checked += 1
    result = {
        "passed": True,
        "nodes": checked,
        "max_float_error": max_error,
        "case_sha256": manifest["case_sha256"],
        "finished_utc": utc(),
    }
    write_json(out / "audit.json", result)
    return result


def development_decision(run_dirs):
    """Operational calibration reads only E and independent atomic retention.

    This is an admissibility check for the fixed recipe, not selection using
    downstream propagation. Early readout success is recorded, not an exclusion.
    """
    records, observed = [], set()
    for directory in map(Path, run_dirs):
        manifest = json.loads((directory / "run.json").read_text())
        if manifest["spec"].get("phase") not in {"development", "engineering"}:
            raise ValueError("Calibration may read development runs only")
        if json.loads((directory / "audit.json").read_text()).get("passed") is not True:
            raise ValueError("Calibration requires independent reload audit")
        result = json.loads((directory / "complete.json").read_text())
        for branch in result["branches"]:
            observed.add(branch["objective"])
            metrics = branch["metrics"]
            target = "E_new" if branch["arm"] == "edit" else "E_old"
            records.append(
                {
                    "run": str(directory),
                    "case_id": branch["case_id"],
                    "objective": branch["objective"],
                    "arm": branch["arm"],
                    "target_accuracy": metrics[target]["accuracy"],
                    "U_atomic_accuracy": metrics["U_atomic"]["accuracy"],
                    "necessary_atomic_accuracy": metrics["necessary_atomic"]["accuracy"],
                    "early_target_accuracy": metrics["early"][
                        "new_entity_accuracy" if branch["arm"] == "edit" else "old_entity_accuracy"
                    ],
                }
            )
    if observed != set(OBJECTIVES):
        raise ValueError("Calibration requires all three objectives")
    passed = bool(records) and all(
        r["target_accuracy"] == 1
        and r["U_atomic_accuracy"] >= 0.95
        and r["necessary_atomic_accuracy"] >= 0.95
        for r in records
    )
    return {
        "operational_gate_passed": passed,
        "decision_rule": "Every edit/sham E=1; U_atomic and necessary_atomic >=.95; no D access",
        "configuration_search": False,
        "records": records,
        "created_utc": utc(),
    }
