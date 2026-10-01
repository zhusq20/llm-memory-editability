"""Prefix-only causal interventions on frozen grok-depth-v1 checkpoints.

Donors are selected from world facts before any model is loaded. The only donor
input is its two-token atomic query; second relations and answers never enter the
forward pass that supplies the post-first-block residual state.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import platform
import sys
import time
import types
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

CONDITIONS = (
    "baseline",
    "identity_r1",
    "different_bridge_r1",
    "same_bridge_r1",
    "different_bridge_h",
    "different_bridge_h_r1",
    "counterfactual_input",
)
DATA_FIELDS = ("entities", "relations", "degree", "phi", "id_fraction", "id_test_fraction")


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def _frozen_modules(source_dir, metadata):
    """Verify all historical copies, then import exclusively from that snapshot."""
    snapshot = source_dir / "source"
    copied = {p.relative_to(snapshot).as_posix(): p for p in snapshot.rglob("*") if p.is_file()}
    verified = {}
    for original, expected in metadata["files"].items():
        matches = [
            p for rel, p in copied.items() if original == rel or original.endswith("/" + rel)
        ]
        require(len(matches) == 1, f"missing or ambiguous historical copy: {original}")
        path = matches[0]
        actual = digest(path)
        require(actual == expected, f"historical source hash mismatch: {path}")
        verified[path.relative_to(snapshot).as_posix()] = actual
    configs = [snapshot / p for p in verified if p.startswith("configs/")]
    require(len(configs) == 1, "expected exactly one frozen configuration")
    frozen = read_json(configs[0])
    require(
        {**frozen["base"], **frozen["runs"][source_dir.name]} == metadata["spec"],
        "historical metadata differs from its frozen configuration",
    )
    package = "_grok_bridge_source_" + hashlib.sha256(str(source_dir).encode()).hexdigest()[:16]
    namespace = types.ModuleType(package)
    namespace.__path__ = [str(snapshot / "src/llm_memory_editability")]
    sys.modules[package] = namespace
    training = importlib.import_module(package + ".grok_depth")
    data = importlib.import_module(package + ".grok_depth_data")
    return training, data, verified


def load_source_run(source_dir, step, device="cuda:0"):
    """Return a frozen model/world after source, world and checkpoint consistency checks."""
    source_dir = Path(source_dir).resolve()
    metadata = read_json(source_dir / "metadata.json")
    spec = metadata["spec"]
    require(spec["phase"] == source_dir.parent.name, "historical phase/path mismatch")
    training, data, verified = _frozen_modules(source_dir, metadata)
    require(step in spec["weight_nodes"], f"unregistered historical weight node {step}")
    checkpoint_path = source_dir / f"weights-{step:07d}.pt"
    checkpoint_hash = digest(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    require(
        checkpoint["spec"] == spec and checkpoint["step"] == step, "checkpoint identity mismatch"
    )
    require(digest(checkpoint_path) == checkpoint_hash, "checkpoint changed while loading")
    if step == spec["steps"]:
        latest = torch.load(source_dir / "latest.pt", map_location="cpu", weights_only=False)
        require(
            latest["step"] == step and latest["spec"] == spec, "latest checkpoint identity mismatch"
        )
        require(
            checkpoint["model"].keys() == latest["model"].keys()
            and all(torch.equal(v, latest["model"][k]) for k, v in checkpoint["model"].items()),
            "endpoint weights differ from latest checkpoint",
        )
        del latest
    world_path = source_dir / "world.npz"
    world_hash = digest(world_path)
    with np.load(world_path, allow_pickle=False) as stored:
        world = {key: stored[key].copy() for key in stored.files}
    world["metadata"] = read_json(source_dir / "world-metadata.json")
    require(digest(world_path) == world_hash, "world changed while loading")
    audit = data.audit_world(world)
    require(audit == read_json(source_dir / "data-audit.json"), "historical world audit mismatch")
    require(
        audit["dataset_sha256"] == world["metadata"]["dataset_sha256"],
        "world content hash mismatch",
    )
    rebuilt = data.build_world(spec["world_seed"], **{key: spec[key] for key in DATA_FIELDS})
    require(rebuilt["metadata"] == world["metadata"], "regenerated world metadata mismatch")
    for key, array in world.items():
        if key != "metadata":
            require(np.array_equal(array, rebuilt[key]), f"regenerated world mismatch: {key}")
    cfg = training.ModelConfig(
        vocab_size=2 + spec["entities"] + spec["relations"],
        width=spec["width"],
        layers=spec["layers"],
        heads=spec["heads"],
        context=8,
    )
    model = training.SmallGPT(cfg, spec["dropout"]).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    provenance = {
        "source_dir": str(source_dir),
        "source_hashes": verified,
        "checkpoint_sha256": checkpoint_hash,
        "world_file_sha256": world_hash,
        "dataset_sha256": audit["dataset_sha256"],
        "world_audit": audit,
        "source_metadata_sha256": digest(source_dir / "metadata.json"),
        "source_predictions_sha256": digest(source_dir / f"predictions-{step:07d}.npz"),
        "world_regenerated_identically": True,
    }
    return model, world, spec, provenance


def _choose(candidates, world_hash, seed, row, kind):
    """Content-based uniform deterministic selection; no model input is accepted."""
    if not candidates:
        return None
    material = json.dumps([world_hash, int(seed), list(map(int, row)), kind], separators=(",", ":"))
    rng = np.random.default_rng(
        int.from_bytes(hashlib.sha256(material.encode()).digest()[:16], "big")
    )
    return sorted(candidates)[int(rng.integers(len(candidates)))]


def select_donors(world, seed):
    """Select once per test row; preserve all rows, masks and rejection counts.

    Different-bridge candidates share r1, change h/b/t, have ID second facts,
    and yield a reserved-test or unused counterfactual. Same-bridge candidates
    may change r1; only their bridge, query distinctness and h != original t
    matter. No candidate is selected using model behavior.
    """
    rows = world["test_composite"]
    n = len(rows)
    atoms = {tuple(map(int, row)) for row in world["id_atomic"]}
    lookup = {(h, r): t for h, r, t in atoms}
    held_out = {
        tuple(map(int, row))
        for name in ("test_composite", "unused_composite")
        for row in world[name]
    }
    training = {tuple(map(int, row)) for row in world["train_composite"]}
    world_hash = world["metadata"]["dataset_sha256"]
    out = {
        "original_rows": rows.copy(),
        "original_bridge": np.full(n, -1, dtype=np.int64),
        "different_donor": np.full((n, 3), -1, dtype=np.int64),
        "same_donor": np.full((n, 3), -1, dtype=np.int64),
        "counterfactual_rows": np.full((n, 4), -1, dtype=np.int64),
        "different_valid": np.zeros(n, dtype=bool),
        "same_valid": np.zeros(n, dtype=bool),
        "different_reason": np.full(n, "", dtype="U64"),
        "same_reason": np.full(n, "", dtype="U64"),
        "different_candidate_count": np.zeros(n, dtype=np.int64),
        "same_candidate_count": np.zeros(n, dtype=np.int64),
        "same_relation_control": np.full(n, "missing", dtype="U24"),
        "different_nondegenerate": np.zeros(n, dtype=bool),
    }
    rejection_names = (
        "same_head",
        "same_bridge",
        "missing_id_second_fact",
        "same_tail",
        "head_is_answer",
        "counterfactual_in_train",
        "counterfactual_not_held_out",
    )
    for reason in rejection_names:
        out["different_rejected_" + reason] = np.zeros(n, dtype=np.int64)
    for index, row in enumerate(rows):
        h, r1, r2, tail = map(int, row)
        bridge = lookup[h, r1]
        require(lookup.get((bridge, r2)) == tail, "recipient is not an ID-ID chain")
        out["original_bridge"][index] = bridge
        candidates = []
        for dh, dr, db in sorted(atoms):
            if dr != r1:
                continue
            dt = lookup.get((db, r2))
            cf = (dh, r1, r2, dt)
            reason = (
                "same_head"
                if dh == h
                else "same_bridge"
                if db == bridge
                else "missing_id_second_fact"
                if dt is None
                else "same_tail"
                if dt == tail
                else "head_is_answer"
                if dh in (tail, dt)
                else "counterfactual_in_train"
                if cf in training
                else "counterfactual_not_held_out"
                if cf not in held_out
                else None
            )
            if reason:
                out["different_rejected_" + reason][index] += 1
            else:
                candidates.append((dh, dr, db, dt))
        donor = _choose(candidates, world_hash, seed, row, "different")
        out["different_candidate_count"][index] = len(candidates)
        if donor is None:
            out["different_reason"][index] = "no_candidate_after_registered_exclusions"
        else:
            dh, dr, db, dt = donor
            out["different_donor"][index] = dh, dr, db
            out["counterfactual_rows"][index] = dh, r1, r2, dt
            out["different_valid"][index] = True
            out["different_reason"][index] = "eligible"
            out["different_nondegenerate"][index] = db != dt and db != tail
        candidates = [
            (dh, dr, db)
            for dh, dr, db in atoms
            if db == bridge and (dh, dr) != (h, r1) and dh != tail
        ]
        same_relation = [candidate for candidate in candidates if candidate[1] == r1]
        donor = _choose(same_relation or candidates, world_hash, seed, row[:2], "same")
        out["same_candidate_count"][index] = len(candidates)
        if donor is None:
            out["same_reason"][index] = "no_distinct_id_prefix_with_same_bridge_and_head_not_answer"
        else:
            out["same_donor"][index] = donor
            out["same_valid"][index] = True
            out["same_reason"][index] = "eligible"
            out["same_relation_control"][index] = "same_r1" if donor[1] == r1 else "different_r1"
    return out


@torch.no_grad()
def traced_forward(
    model,
    tokens,
    positions=None,
    donor_state=None,
    patch_positions=(),
    identity=False,
    return_state=False,
):
    """Frozen SmallGPT eval forward, replacing residuals after complete block 0.

    Both attention and MLP in the first block run before replacement. There is
    no patch in another block, and final LN/output projection remain untouched.
    """
    require(not model.training, "interventions require model.eval()")
    require(
        donor_state is None or not identity, "identity and donor patches are mutually exclusive"
    )
    x = model.token(tokens) + model.position(torch.arange(tokens.shape[1], device=tokens.device))
    first_state = None
    for index, block in enumerate(model.blocks):
        z = block.ln1(x)
        batch, length, width = z.shape
        attention = block.attention
        q, k, v = (
            attention.qkv(z)
            .view(batch, length, 3, attention.heads, width // attention.heads)
            .unbind(2)
        )
        y = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True, dropout_p=0.0
        )
        y = attention.proj(y.transpose(1, 2).reshape(batch, length, width))
        x = x + y
        x = x + block.mlp(block.ln2(x))
        if index == 0:
            if return_state:
                first_state = x.clone()
            if donor_state is not None or identity:
                x = x.clone()
                replacement = x.clone() if identity else donor_state
                for position in patch_positions:
                    x[:, position, :] = replacement[:, position, :]
    x = model.ln_final(x)
    if positions is not None:
        x = x[torch.arange(len(tokens), device=tokens.device)[:, None], positions]
    logits = F.linear(x, model.token.weight)
    return (logits, first_state) if return_state else logits


@torch.no_grad()
def evaluate_condition(
    model,
    rows,
    device,
    batch_size=1024,
    donor_prefixes=None,
    patch_positions=(),
    identity=False,
    verify_trace=False,
):
    """Answer generation followed by EOS with the generated answer fed back.

    The exact same prefix state is reapplied on the second forward pass. Arrays
    are local to the supplied rows; the caller scatters them to the full test set.
    """
    n = len(rows)
    answers, stops, original_logits = [], [], []
    trace_max_delta, trace_disagreements = 0.0, 0
    prefix_max_delta = 0.0
    for start in range(0, n, batch_size):
        selected = rows[start : start + batch_size]
        arity = rows.shape[1]
        require(arity in (3, 4), "expected atomic or composite rows")
        packed = np.zeros((len(selected), 4), dtype=np.int64)
        packed[:, :arity] = selected
        x = torch.as_tensor(packed, device=device)
        pos = torch.as_tensor(np.tile([arity - 2, arity - 1], (len(x), 1)), device=device)
        donor_state = None
        if donor_prefixes is not None:
            prefixes = torch.as_tensor(donor_prefixes[start : start + batch_size], device=device)
            require(prefixes.shape == (len(x), 2), "donor model may receive only h and r1")
            _, donor_state = traced_forward(model, prefixes, return_state=True)
        logits = traced_forward(model, x, pos, donor_state, patch_positions, identity)
        if verify_trace:
            reference = model(x, pos)
            trace_max_delta = max(trace_max_delta, float((reference - logits).abs().max()))
            trace_disagreements += int((reference.argmax(-1) != logits.argmax(-1)).sum())
            _, prefix = traced_forward(model, x[:, :2], return_state=True)
            _, whole = traced_forward(model, x, return_state=True)
            prefix_max_delta = max(prefix_max_delta, float((prefix - whole[:, :2]).abs().max()))
            require(
                torch.allclose(prefix, whole[:, :2], atol=1e-4, rtol=1e-4),
                "pure-prefix state does not match causal full-input prefix",
            )
        answer = logits[:, 0].argmax(-1)
        original_logits.append(logits[:, 0].cpu().numpy())
        x[:, arity - 1] = answer
        generated_logits = traced_forward(model, x, pos, donor_state, patch_positions, identity)
        if verify_trace:
            reference = model(x, pos)
            trace_max_delta = max(
                trace_max_delta, float((reference - generated_logits).abs().max())
            )
            trace_disagreements += int((reference.argmax(-1) != generated_logits.argmax(-1)).sum())
        answers.append(answer.cpu().numpy())
        stops.append(generated_logits[:, 1].argmax(-1).cpu().numpy())
    require(trace_disagreements == 0, "traced forward changes standard-forward discrete outputs")
    require(trace_max_delta <= 1e-5, f"traced forward mismatch: {trace_max_delta}")
    return {
        "answer": np.concatenate(answers) if n else np.empty(0, dtype=np.int64),
        "stop": np.concatenate(stops) if n else np.empty(0, dtype=np.int64),
        "answer_logits": np.concatenate(original_logits)
        if n
        else np.empty((0, model.token.num_embeddings), dtype=np.float32),
    }, {
        "standard_forward_max_logit_delta": trace_max_delta,
        "standard_forward_argmax_disagreements": trace_disagreements,
        "prefix_vs_full_first_block_max_delta": prefix_max_delta,
        "trace_comparison_performed": verify_trace,
    }


def _metrics(answer, stop, original, cf, selected, cf_valid, total, baseline_answer):
    count = int(selected.sum())
    counterfactual = selected & cf_valid
    ncf = int(counterfactual.sum())
    correct = answer == original
    cf_correct = answer == cf
    return {
        "n": count,
        "total_test_n": total,
        "coverage": count / total if total else None,
        "original_answer_accuracy": float(correct[selected].mean()) if count else None,
        "original_complete_accuracy": float((correct & (stop == 1))[selected].mean())
        if count
        else None,
        "eos_accuracy": float((stop[selected] == 1).mean()) if count else None,
        "answer_change_rate": float((answer[selected] != baseline_answer[selected]).mean())
        if count
        else None,
        "cf_target_n": ncf,
        "cf_answer_accuracy": float(cf_correct[counterfactual].mean()) if ncf else None,
        "cf_complete_accuracy": float((cf_correct & (stop == 1))[counterfactual].mean())
        if ncf
        else None,
    }


def evaluate_run(model, world, donors, device="cuda:0", batch_size=1024):
    """Evaluate all fixed conditions, then expose full and prerequisite subgroups."""
    started = time.perf_counter()
    rows = world["test_composite"]
    n = len(rows)
    different, same = donors["different_valid"], donors["same_valid"]
    arrays = {key: value.copy() for key, value in donors.items()}
    audits = {}
    for condition in CONDITIONS:
        mask = (
            same
            if condition == "same_bridge_r1"
            else different
            if condition.startswith("different_") or condition == "counterfactual_input"
            else np.ones(n, dtype=bool)
        )
        eval_rows = (
            donors["counterfactual_rows"][mask]
            if condition == "counterfactual_input"
            else rows[mask]
        )
        donor_key = "same_donor" if condition == "same_bridge_r1" else "different_donor"
        donor_prefix = donors[donor_key][mask, :2] if "bridge_" in condition else None
        positions = (
            (0, 1)
            if condition == "different_bridge_h_r1"
            else (0,)
            if condition == "different_bridge_h"
            else (1,)
        )
        result, audit = evaluate_condition(
            model,
            eval_rows,
            device,
            batch_size,
            donor_prefix,
            positions,
            condition == "identity_r1",
            condition == "baseline",
        )
        audits[condition] = audit
        answer = np.full(n, -1, dtype=np.int64)
        stop = np.full(n, -1, dtype=np.int64)
        answer[mask], stop[mask] = result["answer"], result["stop"]
        arrays[condition + "_answer"], arrays[condition + "_stop"] = answer, stop
        arrays[condition + "_valid"] = mask
        original_logit = result["answer_logits"][np.arange(mask.sum()), rows[mask, -1]]
        cf_logit = np.full(n, np.nan, dtype=np.float32)
        cf_local = different[mask]
        cf_logit[mask.nonzero()[0][cf_local]] = (
            result["answer_logits"][
                np.flatnonzero(cf_local), donors["counterfactual_rows"][mask][cf_local, -1]
            ]
            - original_logit[cf_local]
        )
        arrays[condition + "_cf_minus_original_logit"] = cf_logit
    for column in ("answer", "stop"):
        require(
            np.array_equal(arrays["baseline_" + column], arrays["identity_r1_" + column]),
            f"identity patch changed {column}",
        )
        if len(model.blocks) == 1:
            for condition in (
                "different_bridge_r1",
                "same_bridge_r1",
                "different_bridge_h",
                "different_bridge_h_r1",
            ):
                valid = arrays[condition + "_valid"]
                require(
                    np.array_equal(
                        arrays["baseline_" + column][valid], arrays[condition + "_" + column][valid]
                    ),
                    f"one-layer structural negative control changed {condition}.{column}",
                )
    baseline_correct = (arrays["baseline_answer"] == rows[:, -1]) & (arrays["baseline_stop"] == 1)
    cf_correct = (
        different
        & (arrays["counterfactual_input_answer"] == donors["counterfactual_rows"][:, -1])
        & (arrays["counterfactual_input_stop"] == 1)
    )
    atomic_rows = {
        "different_first": (donors["different_donor"], different),
        "counterfactual_second": (
            np.c_[
                donors["different_donor"][:, 2], rows[:, 2], donors["counterfactual_rows"][:, -1]
            ],
            different,
        ),
        "same_first": (donors["same_donor"], same),
    }
    atomic_scores = {}
    atomic_correct = {}
    for name, (facts, mask) in atomic_rows.items():
        prediction, _ = evaluate_condition(model, facts[mask], device, batch_size)
        answer = np.full(n, -1, dtype=np.int64)
        stop = np.full(n, -1, dtype=np.int64)
        answer[mask], stop[mask] = prediction["answer"], prediction["stop"]
        arrays["atomic_" + name + "_rows"] = facts.copy()
        arrays["atomic_" + name + "_answer"] = answer
        arrays["atomic_" + name + "_stop"] = stop
        arrays["atomic_" + name + "_valid"] = mask.copy()
        correct = mask & (answer == facts[:, -1])
        atomic_correct[name] = correct & (stop == 1)
        count = int(mask.sum())
        atomic_scores[name] = {
            "n": count,
            "total_test_n": n,
            "coverage": count / n if n else None,
            "answer_accuracy": float(correct[mask].mean()) if count else None,
            "complete_accuracy": float(atomic_correct[name][mask].mean()) if count else None,
            "eos_accuracy": float((stop[mask] == 1).mean()) if count else None,
        }
    groups = {
        "all_test": np.ones(n, dtype=bool),
        "different_donor_available": different,
        "same_donor_available": same,
        "baseline_and_counterfactual_complete_correct": baseline_correct & cf_correct,
        "different_bridge_nondegenerate": donors["different_nondegenerate"],
        "same_bridge_same_r1": same & (donors["same_relation_control"] == "same_r1"),
        "same_bridge_different_r1": same & (donors["same_relation_control"] == "different_r1"),
        "both_counterfactual_atomic_facts_complete_correct": atomic_correct["different_first"]
        & atomic_correct["counterfactual_second"],
    }
    for group, mask in groups.items():
        arrays["subset_" + group] = mask
    scores = {
        group: {
            "subset_n": int(mask.sum()),
            "total_test_n": n,
            "coverage": float(mask.mean()) if n else None,
            "conditions": {
                condition: _metrics(
                    arrays[condition + "_answer"],
                    arrays[condition + "_stop"],
                    rows[:, -1],
                    donors["counterfactual_rows"][:, -1],
                    mask & arrays[condition + "_valid"],
                    different,
                    n,
                    arrays["baseline_answer"],
                )
                for condition in CONDITIONS
            },
        }
        for group, mask in groups.items()
    }
    summary = {
        "conditions": list(CONDITIONS),
        "scores": scores,
        "atomic_preconditions": atomic_scores,
        "donor_coverage": {
            "total_test_n": n,
            "different_n": int(different.sum()),
            "same_n": int(same.sum()),
            "different_reasons": dict(Counter(donors["different_reason"].tolist())),
            "same_reasons": dict(Counter(donors["same_reason"].tolist())),
            "same_relation_control": dict(Counter(donors["same_relation_control"].tolist())),
        },
        "engineering_checks": {
            "identity_predictions_equal": True,
            "one_layer_all_early_position_patch_predictions_equal": True
            if len(model.blocks) == 1
            else None,
            "trace": audits["baseline"],
            "eos_uses_generated_answer_and_reapplies_patch": True,
            "donor_forward_input_tokens": 2,
        },
        "evaluation_seconds": time.perf_counter() - started,
    }
    return summary, arrays


def environment(device):
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda": torch.version.cuda,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(torch.device(device))
        if str(device).startswith("cuda")
        else None,
        "cuda_matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "precision": "FP32 parameters and evaluation; historical TF32 flags",
    }
