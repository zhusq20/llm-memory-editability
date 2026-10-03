"""Answer-only MLP editing calibration with frozen parent-logit regularization.

This development grid changes the editor objective and learning rate, preserving
the completed v1 experiment. KL on a finite replay set is a regularizer, not an
exact preservation constraint. Kdev checks the operation; U never selects it.
"""

from __future__ import annotations

import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .depth_step import EOS, pack_rows
from .depth_step import SOURCE_FILES as TRAIN_SOURCES
from .depth_step_mechanism import edit_cases, generate
from .grok_depth import utc, write_json
from .latent_scaling import model_digest
from .storage_composition import file_hash

SOURCE_FILES = list(
    dict.fromkeys(
        [
            *TRAIN_SOURCES,
            "src/llm_memory_editability/depth_step_mechanism.py",
            "src/llm_memory_editability/depth_step_edit_calibration.py",
            "scripts/calibrate_depth_step_edit.py",
            "tests/test_depth_step_edit_calibration.py",
        ]
    )
)
PARAMETER = "blocks.0.mlp.down.weight"
NODES = (0, 20, 100, 200)
LEARNING_RATES = (0.0001, 0.001, 0.003)


def _array_hash(array):
    array = np.asarray(array)
    digest = hashlib.sha256(str(array.dtype).encode())
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def calibration_cases(world, n_cases=2, replay_n=32, keep_n=32, total_cases=8):
    """Partition fixed cases without consulting a checkpoint or any predictions."""
    all_cases = edit_cases(world, n_facts=total_cases, replay_n=replay_n)
    if not 0 < n_cases <= total_cases:
        raise ValueError("Invalid calibration-case prefix")
    atoms = world["atomic"]
    order = sorted(range(len(atoms)), key=lambda index: tuple(atoms[index]))
    for case in all_cases[:n_cases]:
        successor_rows = set(map(tuple, case["tasks"]["necessary_successor_atomic"]))
        successor_indices = {i for i, row in enumerate(atoms) if tuple(row) in successor_rows}
        excluded = {case["atomic_index"], *case["replay_indices"], *successor_indices}
        keep_indices = [i for i in order if i not in excluded][:keep_n]
        if len(keep_indices) != keep_n:
            raise ValueError("Insufficient development keep atoms")
        case["keep_dev_indices"] = keep_indices
        case["tasks"]["Kdev_atomic"] = atoms[keep_indices].copy()
        excluded_u = {case["atomic_index"], *case["replay_indices"], *keep_indices}
        unused = [i for i in range(len(atoms)) if i not in excluded_u]
        case["unused_atomic_indices"] = unused
        case["tasks"]["U_atomic"] = atoms[unused].copy()
    return all_cases[:n_cases], [case["atomic_index"] for case in all_cases[n_cases:]]


def answer_eos_batch(rows, world, device):
    """Select only SEP->answer and teacher-forced answer->EOS supervision."""
    rows = np.asarray(rows, dtype=np.int64)
    tokens, labels = pack_rows(rows, separator=world["metadata"]["separator_token"])
    positions = np.tile([rows.shape[1], rows.shape[1] + 1], (len(rows), 1))
    batch_indices = np.arange(len(rows))[:, None]
    selected = labels[batch_indices, positions]
    if not np.array_equal(selected, np.c_[rows[:, -1], np.full(len(rows), EOS)]):
        raise ValueError("Answer/EOS positions disagree with the packed sequence")
    return tuple(torch.as_tensor(value, device=device) for value in (tokens, positions, selected))


def answer_eos_logits(model, batch):
    tokens, positions, _labels = batch
    return model(tokens, positions=positions)


def target_and_replay_loss(model, target_batch, replay_batch, parent_log_probabilities):
    """CE only on two factual-output tokens; KL(parent || edited), equally weighted."""
    if parent_log_probabilities.requires_grad:
        raise ValueError("Parent logits must be detached")
    target_logits = answer_eos_logits(model, target_batch)
    ce = F.cross_entropy(target_logits.flatten(0, 1), target_batch[2].flatten())
    replay_log = answer_eos_logits(model, replay_batch).log_softmax(-1)
    kl = F.kl_div(
        replay_log.flatten(0, 1),
        parent_log_probabilities.flatten(0, 1),
        reduction="batchmean",
        log_target=True,
    )
    return ce + kl, ce, kl


def _measure(model, case, world, device, step):
    metrics, raw = {}, {}
    for name, rows in case["tasks"].items():
        values, predictions = generate(model, rows, world, device)
        if name in case["original_d_rows"]:
            original = case["original_d_rows"][name]
            changed = rows[:, -1] != original[:, -1]
            predicted = predictions["predictions"]
            correct = (predicted[:, 0] == rows[:, -1]) & (predicted[:, 1] == EOS)
            old_correct = (predicted[:, 0] == original[:, -1]) & (predicted[:, 1] == EOS)
            values.update(
                {
                    "changed_answer_n": int(changed.sum()),
                    "changed_answer_coverage": float(changed.mean()) if len(rows) else None,
                    "changed_answer_accuracy": float(correct[changed].mean())
                    if changed.any()
                    else None,
                    "old_answer_accuracy": float(old_correct.mean()) if len(rows) else None,
                }
            )
        metrics[name] = values
        raw.update({f"step{step}_{name}_{key}": value for key, value in predictions.items()})
    return metrics, raw


def calibrated_edit_one(model, case, world, device, arm, lr, nodes=NODES):
    """One independently initialized branch, with fixed parent KL reference."""
    if arm not in {"edit", "review"}:
        raise ValueError("Arm must be edit or review")
    if tuple(nodes) != tuple(sorted(set(nodes))) or not nodes or nodes[0] != 0:
        raise ValueError("Nodes must increase from zero")
    parent_hash = model_digest(model)
    edited = copy.deepcopy(model).eval()
    for name, parameter in edited.named_parameters():
        parameter.requires_grad_(name == PARAMETER)
        parameter.grad = None
    before = {name: value.detach().clone() for name, value in edited.state_dict().items()}
    parameter = dict(edited.named_parameters())[PARAMETER]
    optimizer = torch.optim.Adam([parameter], lr=lr)
    target_rows = np.asarray([case["new_fact"] if arm == "edit" else case["old_fact"]])
    target_batch = answer_eos_batch(target_rows, world, device)
    replay_batch = answer_eos_batch(case["tasks"]["R_atomic"], world, device)
    with torch.no_grad():
        parent_log = answer_eos_logits(model, replay_batch).log_softmax(-1).detach()
    raw = {
        "parent_replay_log_probabilities": parent_log.cpu().numpy(),
        "replay_teacherforced_labels": replay_batch[2].cpu().numpy(),
    }
    for name, old_rows in case["original_d_rows"].items():
        raw[name + "_original_rows"] = old_rows
        raw[name + "_changed_answer_mask"] = case["tasks"][name][:, -1] != old_rows[:, -1]
    step, history, loss_values = 0, [], None
    for node in nodes:
        while step < node:
            optimizer.zero_grad(set_to_none=True)
            loss, ce, kl = target_and_replay_loss(edited, target_batch, replay_batch, parent_log)
            loss.backward()
            optimizer.step()
            step += 1
            loss_values = {
                "total": float(loss.detach()),
                "target_ce": float(ce.detach()),
                "replay_kl": float(kl.detach()),
            }
        metrics, predictions = _measure(edited, case, world, device, step)
        raw.update(predictions)
        target_name = "E_new" if arm == "edit" else "E_old"
        keep_pass = metrics["Kdev_atomic"]["accuracy"] >= 0.95
        target_pass = metrics[target_name]["accuracy"] == 1.0
        history.append(
            {
                "step": step,
                "loss": loss_values,
                "metrics": metrics,
                "Kdev_at_least_95_percent": keep_pass,
                "target_atomic_success": target_pass,
                "calibration_operation_pass": keep_pass and target_pass,
            }
        )
    changed = [
        name for name, value in edited.state_dict().items() if not torch.equal(value, before[name])
    ]
    if set(changed) - {PARAMETER}:
        raise ValueError("A frozen state tensor changed")
    if model_digest(model) != parent_hash:
        raise ValueError("Original checkpoint model changed")
    delta = parameter.detach() - before[PARAMETER]
    raw["mlp_down_weight_delta"] = delta.cpu().numpy()
    record = {
        "arm": arm,
        "lr": lr,
        "atomic_index": case["atomic_index"],
        "old_fact": case["old_fact"],
        "new_fact": case["new_fact"],
        "replay_indices": case["replay_indices"],
        "keep_dev_indices": case["keep_dev_indices"],
        "unused_atomic_indices": case["unused_atomic_indices"],
        "parent_model_sha256": parent_hash,
        "final_model_sha256": model_digest(edited),
        "initial_down_weight_sha256": _array_hash(before[PARAMETER].cpu().numpy()),
        "final_down_weight_sha256": _array_hash(parameter.detach().cpu().numpy()),
        "updated_parameter": PARAMETER,
        "changed_state_tensors": changed,
        "weight_delta_l2": float(delta.norm()),
        "history": history,
    }
    return record, raw


def prepare_calibration(model, world, source, out, learning_rates=LEARNING_RATES, n_cases=2):
    """Freeze data, all arms, sources, and initial weights before any model inference."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    if any(lr <= 0 or not np.isfinite(lr) for lr in learning_rates):
        raise ValueError("Invalid learning rate")
    cases, remaining = calibration_cases(world, n_cases=n_cases)
    serialized = []
    for case in cases:
        entry = {
            key: value for key, value in case.items() if key not in {"tasks", "original_d_rows"}
        }
        entry["task_rows"] = {key: value.tolist() for key, value in case["tasks"].items()}
        entry["task_sha256"] = {key: _array_hash(value) for key, value in case["tasks"].items()}
        entry["original_d_rows"] = {
            key: value.tolist() for key, value in case["original_d_rows"].items()
        }
        serialized.append(entry)
    config = {
        "phase": "development_calibration",
        "created_utc": utc(),
        "source_checkpoint": source,
        "source": {path: file_hash(path) for path in SOURCE_FILES},
        "parent_model_sha256": model_digest(model),
        "initial_down_weight_sha256": _array_hash(
            model.blocks[0].mlp.down.weight.detach().cpu().numpy()
        ),
        "learning_rates": list(learning_rates),
        "nodes": list(NODES),
        "cases": serialized,
        "n_cases": n_cases,
        "matrix": [
            {"case": i, "lr": lr, "arm": arm}
            for i in range(n_cases)
            for lr in learning_rates
            for arm in ("edit", "review")
        ],
        "reserved_original_case_indices": remaining,
        "reserved_cases_are_independent_world_confirmation": False,
        "optimizer": "Adam, no weight decay",
        "parameter": PARAMETER,
        "objective": "target answer/EOS CE + KL(parent || edited) on 32 replay answer/EOS logits",
        "replay_kl_weight": 1.0,
        "supervised_prefix_tokens": 0,
        "trained_compositions": 0,
        "Kdev_selection": "First 32 lexicographic atoms excluding E, R, necessary successors",
        "Kdev_threshold": 0.95,
        "U_selection": "All remaining atomic facts, including unused necessary successors",
        "U_or_D_select_editor": False,
        "all_grid_failures_retained": True,
        "executions_affected_by_shared_edit": model.repeats,
        "limits": [
            "Finite replay KL is not strict output preservation",
            "A successful Kdev check does not establish unobserved U preservation",
            "Cases share one development world and original checkpoints",
        ],
    }
    write_json(out / "calibration-config.json", config)
    return config, cases


def load_prepared_calibration(model, world, source, out, learning_rates=LEARNING_RATES, n_cases=2):
    """Validate an existing prepared configuration, without rewriting it."""
    out = Path(out)
    config_file = out / "calibration-config.json"
    config = json.loads(config_file.read_text())
    if any(path.name != config_file.name for path in out.iterdir()):
        raise FileExistsError("Existing branch artifacts prevent a repeated execution")
    if config["learning_rates"] != list(learning_rates) or config["n_cases"] != n_cases:
        raise ValueError("Requested grid differs from prepared calibration")
    expected_source = config["source_checkpoint"]
    for key in ("checkpoint_sha256", "dataset_sha256", "step", "spec"):
        if source[key] != expected_source[key]:
            raise ValueError("Checkpoint or data differ from prepared calibration: " + key)
    if model_digest(model) != config["parent_model_sha256"]:
        raise ValueError("Parent parameter hash differs from prepared calibration")
    expected_matrix = [
        {"case": i, "lr": lr, "arm": arm}
        for i in range(n_cases)
        for lr in learning_rates
        for arm in ("edit", "review")
    ]
    if config["matrix"] != expected_matrix or config["nodes"] != list(NODES):
        raise ValueError("Prepared update matrix or nodes changed")
    if (
        config["parameter"] != PARAMETER
        or config["replay_kl_weight"] != 1.0
        or config["Kdev_threshold"] != 0.95
        or config["U_or_D_select_editor"]
    ):
        raise ValueError("Prepared operation contract changed")
    cases, remaining = calibration_cases(world, n_cases=n_cases)
    if remaining != config["reserved_original_case_indices"]:
        raise ValueError("Reserved cases changed")
    for case, frozen in zip(cases, config["cases"], strict=True):
        for key in (
            "atomic_index",
            "old_fact",
            "new_fact",
            "replay_indices",
            "keep_dev_indices",
            "unused_atomic_indices",
            "propagation_scope",
        ):
            if case[key] != frozen[key]:
                raise ValueError("Case selection changed: " + key)
        if set(case["tasks"]) != set(frozen["task_rows"]):
            raise ValueError("Prepared task set changed")
        for name, rows in case["tasks"].items():
            if (
                rows.tolist() != frozen["task_rows"][name]
                or _array_hash(rows) != frozen["task_sha256"][name]
            ):
                raise ValueError("Prepared task rows changed: " + name)
        for name, rows in case["original_d_rows"].items():
            if rows.tolist() != frozen["original_d_rows"][name]:
                raise ValueError("Prepared original D labels changed")
    for path, expected in config["source"].items():
        if file_hash(path) != expected:
            raise ValueError("Source changed after calibration freeze: " + path)
    return config, cases


def execute_calibration(model, world, config, cases, out, device):
    out = Path(out)
    if model_digest(model) != config["parent_model_sha256"]:
        raise ValueError("Parent parameter hash differs from frozen configuration")
    for path, expected in config["source"].items():
        if file_hash(path) != expected:
            raise ValueError("Source changed after calibration freeze: " + path)
    for case, expected in zip(cases, config["cases"], strict=True):
        for name, rows in case["tasks"].items():
            if _array_hash(rows) != expected["task_sha256"][name]:
                raise ValueError("Task rows changed after freeze: " + name)
    started = time.perf_counter()
    records = []
    for item in config["matrix"]:
        number, lr, arm = item["case"], item["lr"], item["arm"]
        record, raw = calibrated_edit_one(
            model, cases[number], world, device, arm, lr, nodes=tuple(config["nodes"])
        )
        label = f"case{number:02d}-lr{lr:.4f}-{arm}"
        np.savez_compressed(out / (label + "-raw.npz"), **raw)
        write_json(out / (label + ".json"), record)
        records.append(record)
        print(f"Calibration branch completed: {label}", flush=True)
    report = {
        "phase": "development_calibration",
        "records": records,
        "config_sha256": file_hash(out / "calibration-config.json"),
        "completed_branches": len(records),
        "wall_seconds": time.perf_counter() - started,
        "model_selection_by_U_or_D": False,
    }
    write_json(out / "calibration-summary.json", report)
    return report
