"""Frozen cross-sample TTCD panels, sequential readouts and independent replay.

Formal edited compositions are read only after training/audit/calibration. They
never choose a checkpoint, fact, order, writer hyperparameter or stopping rule.
Rotations repeatedly measure the same facts; they are not independent worlds.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from .edit_scope_development import evaluate_queries, prediction_metrics
from .grok_depth import utc, write_json
from .path_overlap import sha256
from .ttcd_editing import load_fast, save_fast, tensor_digest
from .ttcd_experiment import Session, mean, open_query, packet_for, retention


def _json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _optional_mean(values):
    return mean([v for v in values if v is not None])


def case_bank(config):
    """Use the original frozen split, including its fixed replacement entities."""
    path = Path(config["data_dir"]) / "training-split.json"
    return json.loads(path.read_text())["case_bank"]


def build_panel(config, world):
    """Pure graph selection; no model, logits, correctness or formal outcomes."""
    settings = config["continual"]
    length, rotations = settings["eval_length"], settings["rotations"]
    affected, controls = settings["affected"], settings["controls"]
    if min(length, rotations, affected, controls) < 1 or rotations > length:
        raise ValueError("Invalid continual panel sizes")
    bank = case_bank(config)
    selected = [c for c in bank if c["split"] == "evaluation"]
    if len(selected) != length or len({c["atom"] for c in selected}) != length:
        raise ValueError("Evaluation must use every distinct original evaluation fact")
    if world.entities < 3:
        raise ValueError("Repeated correction requires a third entity")
    rng = np.random.RandomState(settings["order_seed"])
    edited = np.array([c["atom"] for c in selected], dtype=np.int64)
    support = world.support(config["phi"])
    legal_first = ~np.isin(world.first, edited)
    facts = []
    for case in selected:
        atom = int(case["atom"])
        if not 0 <= case["new_tail"] < world.entities:
            raise ValueError("Original replacement entity is outside the world")
        old = int(world.atoms[atom, -1])
        if case["new_tail"] == old:
            raise ValueError("Original replacement is identical to old truth")
        valid = np.flatnonzero((world.second == atom) & legal_first & ~support).tolist()
        # Preserve legal original D entries, then supplement from the same graph.
        legal = set(valid)
        dependent = list(dict.fromkeys(int(i) for i in case["D"] if int(i) in legal))
        remaining = [int(i) for i in rng.permutation(valid) if int(i) not in dependent]
        dependent = (dependent + remaining)[:affected]
        if not dependent:
            raise ValueError(
                f"No eligible D for fixed fact {case['case_id']}; revise before freeze"
            )
        replacements = np.setdiff1d(np.arange(world.entities), [old, case["new_tail"]])
        facts.append(
            {
                "case_id": case["case_id"],
                "split": "evaluation",
                "group": case["group"],
                "atom": atom,
                "old_tail": old,
                "new_tail": int(case["new_tail"]),
                "revision_tail": int(rng.choice(replacements)),
                "D": dependent,
            }
        )
    atomic_controls = np.flatnonzero(~np.isin(np.arange(len(world.atoms)), edited))
    chain_controls = np.flatnonzero(legal_first & ~np.isin(world.second, edited) & ~support)
    if min(len(atomic_controls), len(chain_controls)) < controls:
        raise ValueError("Insufficient U coverage; do not silently shrink the formal panel")
    order = rng.permutation(length).tolist()
    orders = [order[i:] + order[:i] for i in range(rotations)]
    return {
        "schema": "ttcd-continual-panel-v1",
        "facts": facts,
        "edited_atoms": edited.tolist(),
        "orders": orders,
        "U_atoms": rng.permutation(atomic_controls)[:controls].tolist(),
        "U_chains": rng.permutation(chain_controls)[:controls].tolist(),
        "selection": "Original evaluation facts and tails; graph-only D/U; cyclic orders",
        "order_seed": settings["order_seed"],
    }


def _panel(config, world):
    generated = build_panel(config, world)
    path = config["continual"].get("panel_path")
    if path:
        frozen = json.loads(Path(path).read_text())
        if generated != frozen:
            raise ValueError("Continual panel differs from preflight frozen panel")
        return frozen
    return generated


def _load_reader(config, name, device):
    # Keep the trainer independent of this evaluation module during imports.
    from .ttcd_continual import load_reader

    return load_reader(config, name, device, require_audit=True)


def fingerprint(config, name, panel):
    paths = [
        Path(config["models"][name]["checkpoint"]),
        Path(config["model_config"]),
        *[
            Path(config["data_dir"]) / p
            for p in ("world.npz", "panels.npz", "complete.json", "training-split.json")
        ],
    ]
    paths += [
        Path(__file__).with_name(p)
        for p in (
            "ttcd_continual.py",
            "ttcd_continual_eval.py",
            "ttcd_editing.py",
            "ttcd_experiment.py",
            "edit_scope_development.py",
            "path_overlap.py",
        )
    ]
    return {
        "config": json.loads(json.dumps(config)),
        "name": name,
        "panel_sha256": _json_digest(panel),
        "files": {str(p): sha256(p) for p in paths},
    }


def write_fact(reader, fast, config, world, case, device):
    """Only the atomic correction packet may mutate a persistent fast state."""
    packet = packet_for(config, world, case)
    long_read = open_query(
        reader,
        fast,
        packet,
        config["writer"]["teacher_window"],
        int(world.atoms[case["atom"], -1]),
        case["new_tail"],
        world.metadata,
        device,
    )
    history = []
    x = torch.tensor([packet], dtype=torch.long, device=device)
    for _ in range(config["writer"]["steps"]):
        fast, info = reader.write(
            x,
            fast,
            teacher_window=config["writer"]["teacher_window"],
            student_window=config["writer"]["student_window"],
            learning_rate=config["writer"]["learning_rate"],
        )
        history.append(info)
    return fast, {"long_window_before_write": long_read, "updates": history}


@torch.no_grad()
def read_state(reader, fast, world, panel, indices, targets, device, batch_size=128):
    """Batch same-length queries and verify that answering cannot change A."""
    before = tensor_digest(fast)
    facts = [panel["facts"][i] for i in indices]
    if len(facts) != len(targets):
        raise ValueError("Each written fact needs its current target")
    pools = {
        "E": (world.atoms[[f["atom"] for f in facts]], targets),
        "D": (
            world.chains[[j for f in facts for j in f["D"]]],
            [target for f, target in zip(facts, targets, strict=True) for _ in f["D"]],
        ),
        "U_atoms": (world.atoms[panel["U_atoms"]], None),
        "U_chains": (world.chains[panel["U_chains"]], None),
        "prerequisite_atoms": (world.atoms[world.first[[j for f in facts for j in f["D"]]]], None),
    }
    predictions = {}
    session = Session(reader, fast).eval()
    for width in (3, 4):
        keys = [k for k, (rows, _) in pools.items() if rows.shape[1] == width]
        rows = np.concatenate([pools[k][0] for k in keys])
        labels = np.concatenate(
            [pools[k][0][:, -1] if pools[k][1] is None else np.asarray(pools[k][1]) for k in keys]
        )
        result = evaluate_queries(
            session,
            rows,
            world.metadata,
            torch.device(device),
            new_answers=labels,
            batch_size=batch_size,
            precision="fp32",
        )
        offset = 0
        for key in keys:
            count = len(pools[key][0])
            predictions[key] = {k: v[offset : offset + count] for k, v in result.items()}
            offset += count
    metrics = {k: prediction_metrics(v, world.metadata) for k, v in predictions.items()}
    per_fact, offset = [], 0
    for i, fact in enumerate(facts):
        count = len(fact["D"])
        per_fact.append(
            {
                "atom": fact["atom"],
                "case_id": fact["case_id"],
                "group": fact["group"],
                "target": int(targets[i]),
                "E": prediction_metrics(
                    {k: v[i : i + 1] for k, v in predictions["E"].items()}, world.metadata
                ),
                "D": prediction_metrics(
                    {k: v[offset : offset + count] for k, v in predictions["D"].items()},
                    world.metadata,
                ),
                "prerequisite": prediction_metrics(
                    {
                        k: v[offset : offset + count]
                        for k, v in predictions["prerequisite_atoms"].items()
                    },
                    world.metadata,
                ),
            }
        )
        offset += count
    metrics["D_fact_equal"] = mean([p["D"]["new_accuracy"] for p in per_fact])
    metrics["E_fact_equal"] = mean([p["E"]["new_accuracy"] for p in per_fact])
    if tensor_digest(fast) != before:
        raise AssertionError("Read-only queries mutated the fast state")
    return {
        "fast_digest_before": before,
        "fast_digest_after": before,
        "metrics": metrics,
        "per_fact": per_fact,
        "predictions": {
            pool: {key: value.tolist() for key, value in p.items()}
            for pool, p in predictions.items()
        },
    }


def calibrate(config, name, out, device, *, reader_loader=None, identity_factory=None):
    """Original development facts only; no formal D readouts or selection."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    reader, world = (reader_loader or _load_reader)(config, name, device)
    panel = _panel(config, world)
    identity = (identity_factory or fingerprint)(config, name, panel)
    records = []
    development = [c for c in case_bank(config) if c["split"] == "calibration"]
    fast = reader.empty_fast()
    for case in development:
        atom = world.atoms[case["atom"]]
        packet = packet_for(config, world, case)
        record = {"case": case}
        for view in ("teacher", "student"):
            record[view] = open_query(
                reader,
                fast,
                packet,
                config["writer"][f"{view}_window"],
                int(atom[-1]),
                case["new_tail"],
                world.metadata,
                device,
            )
        single, history = write_fact(reader, fast, config, world, case, device)
        result = evaluate_queries(
            Session(reader, single).eval(),
            atom[None],
            world.metadata,
            torch.device(device),
            new_answers=[case["new_tail"]],
            precision="fp32",
        )
        record["single_atomic"] = {k: v.tolist() for k, v in result.items()}
        record["write_history"] = history
        records.append(record)
    # Numerical short-stream calibration uses ONLY original development atoms.
    stream = reader.empty_fast()
    short = development[: config["continual"]["train_length"]]
    stream_history = []
    for case in short:
        stream, history = write_fact(reader, stream, config, world, case, device)
        stream_history.append(history)
    reader.validate_fast(stream)
    short_result = evaluate_queries(
        Session(reader, stream).eval(),
        world.atoms[[c["atom"] for c in short]],
        world.metadata,
        torch.device(device),
        new_answers=[c["new_tail"] for c in short],
        precision="fp32",
    )
    groups = {
        group: {
            "n": sum(r["case"]["group"] == group for r in records),
            "teacher_new_answer_accuracy": mean(
                [r["teacher"]["new_answer_correct"] for r in records if r["case"]["group"] == group]
            ),
            "teacher_new_joint_accuracy": mean(
                [r["teacher"]["new_correct"] for r in records if r["case"]["group"] == group]
            ),
        }
        for group in ("id", "ood")
    }
    gate = config["reading_gate"]
    passed = all(
        g["n"] >= gate["minimum_cases_per_group"]
        and g["teacher_new_answer_accuracy"] is not None
        and g["teacher_new_answer_accuracy"] >= gate["minimum_accuracy"]
        for g in groups.values()
    )
    result = {
        "identity": identity,
        "passed": passed,
        "groups": groups,
        "records": records,
        "development_short_stream": {
            "cases": [c["case_id"] for c in short],
            "history": stream_history,
            "predictions": {k: v.tolist() for k, v in short_result.items()},
            "fast_digest": tensor_digest(stream),
            "finite": True,
        },
        "formal_D_used": False,
        "status": "reading_passed" if passed else "reading_needs_attention",
    }
    write_json(out / "panel.json", panel)
    write_json(out / "calibration.json", result)
    return result


def _spec(kind, indices, panel, *, rotation=None, t=None, revision=False, read=True):
    targets = [panel["facts"][i]["new_tail"] for i in indices]
    if revision:
        targets[0] = panel["facts"][indices[0]]["revision_tail"]
    return {
        "kind": kind,
        "rotation": rotation,
        "t": t,
        "indices": list(indices),
        "targets": targets,
        "revision": revision,
        "read": read,
    }


def state_plan(config, panel):
    """Complete expected matrix, independently reconstructed by the audit."""
    length = len(panel["facts"])
    plans = {"baseline": _spec("baseline", range(length), panel)}
    for i in range(length):
        plans[f"single/f{i:03d}"] = _spec("single", [i], panel, t=1)
    k = config["continual"]["train_length"]
    if k < 1 or 2 * k - 1 > length:
        raise ValueError("Evaluation stream too short for the repeated-correction branch")
    for rotation, order in enumerate(panel["orders"]):
        for t in range(1, length + 1):
            plans[f"main/r{rotation:03d}/t{t:03d}"] = _spec(
                "main", order[:t], panel, rotation=rotation, t=t
            )
        for j in range(k):
            t = k + 1 + j
            indices = order[:k] + order[k : k + j]
            plans[f"revision/r{rotation:03d}/t{t:03d}"] = _spec(
                "revision",
                indices,
                panel,
                rotation=rotation,
                t=t,
                revision=True,
                read=j in (0, k - 1),
            )
    return plans


def _save_state(out, key, spec, reader, fast, world, panel, device, write_history=None):
    directory = Path(out) / "states" / key
    directory.mkdir(parents=True, exist_ok=False)
    digest = save_fast(directory / "fast.pt", reader, fast, spec)
    readout = (
        read_state(reader, fast, world, panel, spec["indices"], spec["targets"], device)
        if spec["read"]
        else None
    )
    record = {
        "key": key,
        "spec": spec,
        "fast_digest": digest,
        "write_history": write_history,
        "readout": readout,
    }
    write_json(directory / "record.json", record)
    return record


def _retention_metrics(before, after, metadata):
    return {
        pool: retention(before["predictions"][pool], after["predictions"][pool], metadata)
        for pool in ("U_atoms", "U_chains")
    }


def evaluate(config, name, out, device, log=None, *, reader_loader=None, identity_factory=None):
    out = Path(out)
    receipt = json.loads((out / "calibration.json").read_text())
    reader, world = (reader_loader or _load_reader)(config, name, device)
    panel = _panel(config, world)
    identity = (identity_factory or fingerprint)(config, name, panel)
    if not receipt["passed"] or receipt["identity"] != identity:
        raise ValueError("Passing development calibration with identical provenance is required")
    if panel != json.loads((out / "panel.json").read_text()):
        raise ValueError("Panel changed since calibration")
    original = tensor_digest(reader.backbone.state_dict())
    plans = state_plan(config, panel)
    summaries, complete = [], []
    started = time.monotonic()

    def save(key, fast, history=None):
        record = _save_state(out, key, plans[key], reader, fast, world, panel, device, history)
        complete.append(key)
        if record["readout"] is not None:
            summaries.append(record)
        write_json(
            out / "evaluation-progress.json",
            {"states": len(complete), "expected": len(plans), "last": key, "utc": utc()},
        )
        if log:
            log(len(complete), record)
        return record

    baseline = save("baseline", reader.empty_fast())["readout"]
    for i, fact in enumerate(panel["facts"]):
        fast, history = write_fact(reader, reader.empty_fast(), config, world, fact, device)
        save(f"single/f{i:03d}", fast, history)
    k = config["continual"]["train_length"]
    for rotation, order in enumerate(panel["orders"]):
        fast = reader.empty_fast()
        branch = None
        for t, i in enumerate(order, 1):
            fast, history = write_fact(reader, fast, config, world, panel["facts"][i], device)
            save(f"main/r{rotation:03d}/t{t:03d}", fast, history)
            if t == k:
                branch = {layer: value.detach().clone() for layer, value in fast.items()}
        revision = dict(panel["facts"][order[0]])
        revision["new_tail"] = revision["revision_tail"]
        for j in range(k):
            fact = revision if j == 0 else panel["facts"][order[k + j - 1]]
            branch, history = write_fact(reader, branch, config, world, fact, device)
            save(f"revision/r{rotation:03d}/t{k + 1 + j:03d}", branch, history)
    if set(complete) != set(plans) or len(complete) != len(plans):
        raise AssertionError("Incomplete evaluation state matrix")
    if tensor_digest(reader.backbone.state_dict()) != original:
        raise AssertionError("Evaluation mutated frozen base parameters")
    if (identity_factory or fingerprint)(config, name, panel) != identity:
        raise ValueError("Evaluation inputs or code changed during execution")
    result = summarize(config, panel, summaries, baseline, world.metadata)
    result.update(
        {
            "identity": identity,
            "states": len(complete),
            "readout_states": len(summaries),
            "state_keys": complete,
            "wall_seconds": time.monotonic() - started,
            "compute": dict(reader.compute),
            "audit_state": "pending_fresh_process_reload",
            "parameters": {
                "base_total": sum(p.numel() for p in reader.parameters()),
                "base_trainable": sum(p.numel() for p in reader.parameters() if p.requires_grad),
                "persistent_fast_elements": sum(v.numel() for v in reader.empty_fast().values()),
            },
        }
    )
    write_json(out / "summary.json", result)
    return result


def summarize(config, panel, records, baseline, metadata):
    """Fact-equal primary; every rotation and every t remain in raw records."""
    curves = {}
    for t in range(1, len(panel["facts"]) + 1):
        at_t = [r for r in records if r["spec"]["kind"] == "main" and r["spec"]["t"] == t]
        by_fact = {}
        for record in at_t:
            for fact in record["readout"]["per_fact"]:
                by_fact.setdefault(fact["atom"], []).append(fact)
        fact_means = {
            str(atom): {
                pool: mean([f[pool]["new_accuracy"] for f in values]) for pool in ("E", "D")
            }
            for atom, values in by_fact.items()
        }
        curves[str(t)] = {
            "E": mean([f["E"] for f in fact_means.values()]),
            "D": mean([f["D"] for f in fact_means.values()]),
            "fact_means": fact_means,
            "groups": {
                group: {
                    pool: _optional_mean(
                        [
                            fact_means[str(atom)][pool]
                            for atom, values in by_fact.items()
                            if values[0].get("group") == group
                        ]
                    )
                    for pool in ("E", "D")
                }
                for group in ("id", "ood")
            },
            "by_age": {
                str(age): {
                    pool: mean(
                        [r["readout"]["per_fact"][t - age - 1][pool]["new_accuracy"] for r in at_t]
                    )
                    for pool in ("E", "D")
                }
                for age in range(t)
            },
            "U_atoms": mean([r["readout"]["metrics"]["U_atoms"]["old_accuracy"] for r in at_t]),
            "U_chains": mean([r["readout"]["metrics"]["U_chains"]["old_accuracy"] for r in at_t]),
            "U_retention": {
                pool: {
                    "original_correct": _retention_metrics(baseline, at_t[0]["readout"], metadata)[
                        pool
                    ]["original_correct"],
                    "retained": _optional_mean(
                        [
                            _retention_metrics(baseline, r["readout"], metadata)[pool]["retained"]
                            for r in at_t
                        ]
                    ),
                }
                for pool in ("U_atoms", "U_chains")
            },
        }
    revision = []
    for record in records:
        if record["spec"]["kind"] != "revision":
            continue
        readout = record["readout"]
        first = panel["facts"][record["spec"]["indices"][0]]
        old_new = first["new_tail"] + metadata["entity_offset"]
        d_n = len(first["D"])
        revision.append(
            {
                "rotation": record["spec"]["rotation"],
                "t": record["spec"]["t"],
                "anchor": readout["per_fact"][0],
                "first_new_residual": {
                    "E": mean(
                        [
                            a == old_new and s == metadata["end_marker"]
                            for a, s in zip(
                                readout["predictions"]["E"]["answer"][:1],
                                readout["predictions"]["E"]["stop"][:1],
                                strict=True,
                            )
                        ]
                    ),
                    "D": mean(
                        [
                            a == old_new and s == metadata["end_marker"]
                            for a, s in zip(
                                readout["predictions"]["D"]["answer"][:d_n],
                                readout["predictions"]["D"]["stop"][:d_n],
                                strict=True,
                            )
                        ]
                    ),
                },
                "other_facts_E": mean([f["E"]["new_accuracy"] for f in readout["per_fact"][1:]]),
                "other_facts_D": mean([f["D"]["new_accuracy"] for f in readout["per_fact"][1:]]),
                "U_retention": _retention_metrics(baseline, readout, metadata),
            }
        )
    return {
        "primary_t": config["continual"]["train_length"],
        "primary": curves[str(config["continual"]["train_length"])],
        "curves": curves,
        "revision": revision,
        "analysis_unit": "Same fixed facts in one observed world; rotations are repeated measures",
        "primary_selection": "All facts, no model-correctness or immediate-write filter",
    }


def _compare_nested(replay, recorded, location=""):
    if isinstance(replay, dict):
        if not isinstance(recorded, dict) or replay.keys() != recorded.keys():
            raise AssertionError(f"Replay fields changed: {location}")
        for key, value in replay.items():
            _compare_nested(value, recorded[key], f"{location}/{key}")
    elif isinstance(replay, list):
        if not isinstance(recorded, list) or len(replay) != len(recorded):
            raise AssertionError(f"Replay lengths changed: {location}")
        for i, (value, wanted) in enumerate(zip(replay, recorded, strict=True)):
            _compare_nested(value, wanted, f"{location}/{i}")
    elif isinstance(replay, float):
        np.testing.assert_allclose(replay, recorded, rtol=1e-5, atol=1e-5)
    elif replay != recorded:
        raise AssertionError(f"Replay value changed: {location}")


def _compare_readouts(replay, recorded):
    if (
        replay["fast_digest_before"] != recorded["fast_digest_before"]
        or replay["fast_digest_after"] != recorded["fast_digest_after"]
    ):
        raise AssertionError("Read state hash mismatch")
    for pool, pred in replay["predictions"].items():
        for field, value in pred.items():
            expected = recorded["predictions"][pool][field]
            if field in (
                "answer",
                "stop",
                "rows",
                "old_answers",
                "new_answers",
                "old_rank",
                "new_rank",
            ):
                if value != expected:
                    raise AssertionError(f"Independent token/label replay mismatch: {pool}/{field}")
            else:
                np.testing.assert_allclose(value, expected, rtol=1e-5, atol=1e-5)
    _compare_nested(replay["metrics"], recorded["metrics"], "metrics")
    _compare_nested(replay["per_fact"], recorded["per_fact"], "per_fact")


def _replay_write(reader, config, out, world, panel, spec, device):
    if spec["kind"] == "baseline":
        return reader.empty_fast(), None
    if spec["kind"] == "single" or (spec["kind"] == "main" and spec["t"] == 1):
        prior = reader.empty_fast()
    else:
        kind, t = spec["kind"], spec["t"] - 1
        if kind == "revision" and t == config["continual"]["train_length"]:
            kind = "main"
        key = f"{kind}/r{spec['rotation']:03d}/t{t:03d}"
        prior, _ = load_fast(out / "states" / key / "fast.pt", reader)
    if spec["kind"] == "revision" and spec["t"] == config["continual"]["train_length"] + 1:
        fact = dict(panel["facts"][spec["indices"][0]])
        fact["new_tail"] = fact["revision_tail"]
    else:
        fact = panel["facts"][spec["indices"][-1]]
    return write_fact(reader, prior, config, world, fact, device)


def audit(config, name, out, device, *, reader_loader=None, identity_factory=None):
    """Load every saved A in a separate process and replay every answer and EOS."""
    out = Path(out)
    reader, world = (reader_loader or _load_reader)(config, name, device)
    panel = _panel(config, world)
    identity = (identity_factory or fingerprint)(config, name, panel)
    receipt = json.loads((out / "calibration.json").read_text())
    summary = json.loads((out / "summary.json").read_text())
    if not receipt["passed"] or receipt["identity"] != identity or summary["identity"] != identity:
        raise ValueError("Audit code/config/input identity differs from evaluation")
    if panel != json.loads((out / "panel.json").read_text()):
        raise ValueError("Audit panel differs from evaluation")
    plans = state_plan(config, panel)
    paths = list((out / "states").rglob("record.json"))
    keys = {str(p.parent.relative_to(out / "states")) for p in paths}
    if keys != set(plans) or len(paths) != len(plans):
        raise ValueError("Incomplete or unexpected evaluation state matrix")
    if set(summary["state_keys"]) != set(plans) or summary["states"] != len(plans):
        raise ValueError("Summary omits an expected state")
    checked_queries, transitions = 0, 0
    replay_records = []
    for key, spec in plans.items():
        directory = out / "states" / key
        record = json.loads((directory / "record.json").read_text())
        if record["key"] != key or record["spec"] != spec:
            raise ValueError("Altered state/query/target provenance")
        fast, provenance = load_fast(directory / "fast.pt", reader)
        if provenance != spec or tensor_digest(fast) != record["fast_digest"]:
            raise ValueError("Saved state provenance/hash mismatch")
        rebuilt, history = _replay_write(reader, config, out, world, panel, spec, device)
        if tensor_digest(rebuilt) != record["fast_digest"]:
            raise AssertionError("Independent write transition differs from saved state")
        _compare_nested(history, record["write_history"], "write_history")
        transitions += spec["kind"] != "baseline"
        if spec["read"]:
            replay = read_state(
                reader, fast, world, panel, spec["indices"], spec["targets"], device
            )
            _compare_readouts(replay, record["readout"])
            checked_queries += sum(len(p["answer"]) for p in replay["predictions"].values())
            replay_records.append({"spec": spec, "readout": replay})
        elif record["readout"] is not None:
            raise ValueError("Unexpected intermediate branch readout")
    baseline = next(r["readout"] for r in replay_records if r["spec"]["kind"] == "baseline")
    recomputed = summarize(config, panel, replay_records, baseline, world.metadata)
    _compare_nested(recomputed, {k: summary[k] for k in recomputed}, "summary")
    result = {
        "passed": True,
        "identity": identity,
        "states": len(plans),
        "queries": checked_queries,
        "write_transitions": transitions,
        "answer_and_EOS_exact": True,
        "write_states_exact": True,
        "summary_recomputed": True,
        "all_saved_states_bound_to_backbone": True,
        "utc": utc(),
    }
    write_json(out / "audit.json", result)
    return result
