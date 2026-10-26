"""Supplementary TTCD evaluation: two revised facts, one persistent state.

The original trainer, writer and evaluation are imported without alteration.
This panel changes only evaluation replacement tails and queries. Every target
key was already excluded from correction training. Matching is graph-only.
"""

from __future__ import annotations

import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from . import ttcd_continual_eval as original
from .edit_scope_development import evaluate_queries
from .grok_depth import utc, write_json
from .path_overlap import sha256
from .ttcd_editing import load_fast, save_fast, tensor_digest
from .ttcd_experiment import Session, mean


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _mean(values):
    return mean([value for value in values if value is not None])


def build_panel(config, settings, world):
    parent = original.build_panel(config, world)
    facts = copy.deepcopy(parent["facts"])
    split = json.loads((Path(config["data_dir"]) / "training-split.json").read_text())
    if set(parent["edited_atoms"]) & set(split["train_edit_atoms"]):
        raise ValueError("Supplementary keys must remain excluded from correction training")
    quota = int(settings["pairs_per_type"])
    if len(facts) != quota * 8 or settings["rotations"] != len(parent["orders"]):
        raise ValueError("Use all reserved facts, balanced types and original rotation coverage")
    support = world.support(config["phi"])
    old_queries = {
        tuple(world.chains[j, :3]): int(j)
        for j in np.flatnonzero(np.isin(world.first, parent["edited_atoms"]))
    }
    candidates = []
    for a, first in enumerate(facts):
        h, r, old_bridge = map(int, world.atoms[first["atom"]])
        for b, second in enumerate(facts):
            bridge, r2, old_second = map(int, world.atoms[second["atom"]])
            key = (h, r, r2)
            j = old_queries.get(key)
            if (
                a == b
                or bridge == old_bridge
                or j is None
                or support[j]
                or int(world.second[j]) in parent["edited_atoms"]
                or second["new_tail"] in (h, bridge)
                or old_second == int(world.chains[j, -1])
                or second["new_tail"] in (old_second, int(world.chains[j, -1]))
            ):
                continue
            candidates.append(
                {
                    "first": a,
                    "second": b,
                    "type": ("I" if first["group"] == "id" else "O")
                    + ("I" if second["group"] == "id" else "O"),
                    "row": world.chains[j].tolist(),
                    "old_chain_index": j,
                    "bridge": bridge,
                    "target": second["new_tail"],
                    "first_only_target": old_second,
                }
            )
    rng = np.random.RandomState(settings["pair_seed"])
    candidates = [candidates[int(i)] for i in rng.permutation(len(candidates))]
    failed = set()

    def match(unused, counts):
        state = (tuple(sorted(unused)), tuple(counts[k] for k in ("II", "IO", "OI", "OO")))
        if state in failed:
            return None
        if not unused:
            return [] if all(value == quota for value in counts.values()) else None
        available = [
            p
            for p in candidates
            if p["first"] in unused and p["second"] in unused and counts[p["type"]] < quota
        ]
        incident = {i: [p for p in available if i in (p["first"], p["second"])] for i in unused}
        anchor = min(sorted(unused), key=lambda i: len(incident[i]))
        for p in incident[anchor]:
            next_counts = {**counts, p["type"]: counts[p["type"]] + 1}
            rest = match(unused - {p["first"], p["second"]}, next_counts)
            if rest is not None:
                return [p, *rest]
        failed.add(state)
        return None

    pairs = match(set(range(len(facts))), {kind: 0 for kind in ("II", "IO", "OI", "OO")})
    if pairs is None:
        raise ValueError("No full balanced graph matching; do not shrink or filter by model")
    for number, pair in enumerate(pairs):
        pair["pair_id"] = f"pair-{number:02d}"
        facts[pair["first"]]["new_tail"] = pair["bridge"]
    for fact in facts:
        fact.pop("revision_tail", None)
    panel = {
        **parent,
        "schema": "ttcd-double-new-panel-v1",
        "facts": facts,
        "pairs": pairs,
        "parent_panel_sha256": digest(parent),
        "selection": "All reserved keys; legal unseen old queries; balanced graph matching",
        "pair_seed": settings["pair_seed"],
    }
    # Recompute truth by key after applying every edit, never by a cached old edge.
    edited_truth = {tuple(map(int, row[:2])): int(row[-1]) for row in world.atoms}
    for fact in facts:
        edited_truth[tuple(map(int, world.atoms[fact["atom"], :2]))] = fact["new_tail"]
    for pair in pairs:
        h, r, r2, old = pair["row"]
        bridge = edited_truth[(h, r)]
        target = edited_truth[(bridge, r2)]
        if bridge != pair["bridge"] or target != pair["target"] or target == old:
            raise AssertionError("Two-new-fact truth is not the intended changed chain")
    return panel


def state_plan(panel):
    plans = {"baseline": {"kind": "baseline", "indices": [], "previous": None, "write": None}}
    for i in range(len(panel["facts"])):
        plans[f"single/f{i:03d}"] = {
            "kind": "single",
            "indices": [i],
            "previous": "baseline",
            "write": i,
        }
    for p, pair in enumerate(panel["pairs"]):
        for direction, (a, b) in enumerate(
            ((pair["first"], pair["second"]), (pair["second"], pair["first"]))
        ):
            plans[f"pair/p{p:03d}/d{direction}"] = {
                "kind": "pair",
                "pair": p,
                "direction": direction,
                "indices": [a, b],
                "previous": f"single/f{a:03d}",
                "write": b,
            }
    for rotation, order in enumerate(panel["orders"]):
        for t, fact in enumerate(order, 1):
            plans[f"main/r{rotation:03d}/t{t:03d}"] = {
                "kind": "main",
                "rotation": rotation,
                "t": t,
                "indices": order[:t],
                "previous": "baseline" if t == 1 else f"main/r{rotation:03d}/t{t - 1:03d}",
                "write": fact,
            }
    return plans


def _joint(prediction, metadata):
    return (
        (
            np.asarray(prediction["answer"])
            == np.asarray(prediction["new_answers"]) + metadata["entity_offset"]
        )
        & (np.asarray(prediction["stop"]) == metadata["end_marker"])
    ).tolist()


@torch.no_grad()
def read_state(reader, fast, world, panel, spec, device):
    before = tensor_digest(fast)
    indices = spec["indices"] or list(range(len(panel["facts"])))
    targets = [panel["facts"][i]["new_tail"] for i in indices]
    core = original.read_state(reader, fast, world, panel, indices, targets, device)
    pools = {
        "atoms": (
            world.atoms[[f["atom"] for f in panel["facts"]]],
            [f["new_tail"] for f in panel["facts"]],
        ),
        "chains": (
            np.array([p["row"] for p in panel["pairs"]]),
            [p["target"] for p in panel["pairs"]],
        ),
    }
    predictions = {}
    for name, (rows, labels) in pools.items():
        pred = evaluate_queries(
            Session(reader, fast).eval(),
            rows,
            world.metadata,
            torch.device(device),
            new_answers=labels,
            batch_size=128,
            precision="fp32",
        )
        predictions[name] = {key: value.tolist() for key, value in pred.items()}
    atom_correct, chain_correct = [
        _joint(predictions[k], world.metadata) for k in ("atoms", "chains")
    ]
    written = set(spec["indices"])
    atom_current_targets = [
        f["new_tail"] if i in written else f["old_tail"] for i, f in enumerate(panel["facts"])
    ]
    atom_current_correct = [
        answer == target + world.metadata["entity_offset"] and stop == world.metadata["end_marker"]
        for answer, stop, target in zip(
            predictions["atoms"]["answer"],
            predictions["atoms"]["stop"],
            atom_current_targets,
            strict=True,
        )
    ]
    pairs = []
    for i, pair in enumerate(panel["pairs"]):
        a, b = pair["first"], pair["second"]
        eligible = a in written and b in written
        positions = {fact: position for position, fact in enumerate(spec["indices"], 1)}
        current_target = (
            pair["row"][-1]
            if a not in written
            else pair["first_only_target"]
            if b not in written
            else pair["target"]
        )
        current_correct = (
            predictions["chains"]["answer"][i] == current_target + world.metadata["entity_offset"]
            and predictions["chains"]["stop"][i] == world.metadata["end_marker"]
        )
        pairs.append(
            {
                "pair_id": pair["pair_id"],
                "type": pair["type"],
                "eligible": eligible,
                "correct": chain_correct[i],
                "current_target": current_target,
                "current_correct": current_correct,
                "first_atomic_correct": atom_correct[a],
                "second_atomic_correct": atom_correct[b],
                "both_atoms_correct": atom_correct[a] and atom_correct[b],
                "first_current_atomic_correct": atom_current_correct[a],
                "second_current_atomic_correct": atom_current_correct[b],
                "completion_t": max(positions[a], positions[b]) if eligible else None,
                "completion_age": len(written) - max(positions[a], positions[b])
                if eligible
                else None,
                "first_before_second": positions[a] < positions[b] if eligible else None,
                "first_age": len(written) - positions[a] if a in written else None,
                "second_age": len(written) - positions[b] if b in written else None,
                "write_gap": abs(positions[a] - positions[b]) if eligible else None,
            }
        )
    if tensor_digest(fast) != before:
        raise AssertionError("Supplementary queries mutated persistent A")
    return {
        "core": core,
        "predictions": predictions,
        "atom_correct": atom_correct,
        "atom_current_targets": atom_current_targets,
        "atom_current_correct": atom_current_correct,
        "pairs": pairs,
    }


def identity(config, settings, name, panel, *, parent_identity=None):
    if parent_identity is None:
        from .ttcd_verified_base import evaluation_identity

        parent_identity = evaluation_identity
    root = Path(__file__).resolve().parents[2]
    return {
        "parent": parent_identity(config, name, panel),
        "settings": settings,
        "panel_sha256": digest(panel),
        "supplementary_code": {
            relative: sha256(root / relative)
            for relative in (
                "src/llm_memory_editability/ttcd_double_new_eval.py",
                "scripts/run_ttcd_double_new_eval.py",
            )
        },
    }


def _load_panel(config, settings, world, panel_path):
    panel = json.loads(Path(panel_path).read_text())
    if panel != build_panel(config, settings, world):
        raise ValueError("Supplementary panel differs from the frozen graph selection")
    return panel


def evaluate(
    config,
    settings,
    name,
    out,
    panel_path,
    device,
    log=None,
    *,
    reader_loader=None,
    identity_factory=None,
):
    if reader_loader is None:
        from .ttcd_verified_base import load_reader

        reader_loader = load_reader
    out = Path(out)
    reader, world = reader_loader(config, name, device)
    panel = _load_panel(config, settings, world, panel_path)
    bound = (identity_factory or identity)(config, settings, name, panel)
    original_base = tensor_digest(reader.backbone.state_dict())
    plans, records = state_plan(panel), []
    started = time.monotonic()
    for key, spec in plans.items():
        if spec["previous"] is None:
            fast, history = reader.empty_fast(), None
        else:
            fast, _ = load_fast(out / "states" / spec["previous"] / "fast.pt", reader)
            fast, history = original.write_fact(
                reader, fast, config, world, panel["facts"][spec["write"]], device
            )
        directory = out / "states" / key
        directory.mkdir(parents=True, exist_ok=False)
        fast_digest = save_fast(directory / "fast.pt", reader, fast, spec)
        readout = read_state(reader, fast, world, panel, spec, device)
        record = {
            "key": key,
            "spec": spec,
            "fast_digest": fast_digest,
            "write_history": history,
            "readout": readout,
        }
        write_json(directory / "record.json", record)
        records.append(record)
        write_json(
            out / "progress.json",
            {"states": len(records), "expected": len(plans), "key": key, "utc": utc()},
        )
        if log:
            log(len(records), record)
    if tensor_digest(reader.backbone.state_dict()) != original_base:
        raise AssertionError("Supplementary evaluation changed base parameters")
    if (identity_factory or identity)(config, settings, name, panel) != bound:
        raise ValueError("Inputs or code changed during supplementary evaluation")
    summary = summarize(settings, panel, records, world.metadata)
    summary.update(
        identity=bound,
        states=len(records),
        state_keys=list(plans),
        wall_seconds=time.monotonic() - started,
        compute=dict(reader.compute),
        scientific_training_updates=0,
        audit_state="pending_independent_reload",
    )
    write_json(out / "summary.json", summary)
    return summary


def _conditional(rows, eligible, value):
    chosen = [row for row in rows if row[eligible]]
    return {
        "total": len(rows),
        "eligible": len(chosen),
        "coverage": mean([bool(r[eligible]) for r in rows]),
        "accuracy": mean([r[value] for r in chosen]),
    }


def summarize(settings, panel, records, metadata):
    by_key = {record["key"]: record for record in records}
    baseline = by_key["baseline"]["readout"]["core"]
    main = [r for r in records if r["spec"]["kind"] == "main"]
    curves, atom_age, pair_age = {}, {}, {}
    for t in range(1, len(panel["facts"]) + 1):
        at_t = [r for r in main if r["spec"]["t"] == t]
        pair_rows, atom_rows = [], []
        u = {pool: [] for pool in ("U_atoms", "U_chains")}
        for record in at_t:
            spec, readout = record["spec"], record["readout"]
            rotation = spec["rotation"]
            for position, i in enumerate(spec["indices"], 1):
                anchor = by_key[f"main/r{rotation:03d}/t{position:03d}"]["readout"]
                item = {
                    "fact": i,
                    "correct": readout["atom_correct"][i],
                    "immediate_correct": anchor["atom_correct"][i],
                    "age": t - position,
                }
                atom_rows.append(item)
                atom_age.setdefault(str(item["age"]), []).append(item)
            for p, pair in enumerate(readout["pairs"]):
                if not pair["eligible"]:
                    continue
                anchor = by_key[f"main/r{rotation:03d}/t{pair['completion_t']:03d}"]["readout"][
                    "pairs"
                ][p]
                item = {**pair, "immediate_correct": anchor["correct"]}
                pair_rows.append(item)
                pair_age.setdefault(str(item["completion_age"]), []).append(item)
            for pool in u:
                u[pool].append(
                    original._retention_metrics(baseline, readout["core"], metadata)[pool]
                )
        pair_means = {
            p["pair_id"]: mean([r["correct"] for r in pair_rows if r["pair_id"] == p["pair_id"]])
            for p in panel["pairs"]
        }
        curves[str(t)] = {
            "two_new_accuracy": _mean(pair_means.values()),
            "pair_means": pair_means,
            "eligible_pair_readouts": len(pair_rows),
            "total_pair_readouts": len(at_t) * len(panel["pairs"]),
            "by_type": {
                kind: _mean(
                    [
                        value
                        for key, value in pair_means.items()
                        if next(p["type"] for p in panel["pairs"] if p["pair_id"] == key) == kind
                    ]
                )
                for kind in ("II", "IO", "OI", "OO")
            },
            "both_atoms_correct": _conditional(pair_rows, "both_atoms_correct", "correct"),
            "first_atomic_accuracy": mean([r["first_atomic_correct"] for r in pair_rows]),
            "second_atomic_accuracy": mean([r["second_atomic_correct"] for r in pair_rows]),
            "by_write_order": {
                direction: {
                    "n": sum(r["first_before_second"] == flag for r in pair_rows),
                    "accuracy": mean(
                        [r["correct"] for r in pair_rows if r["first_before_second"] == flag]
                    ),
                }
                for direction, flag in (("first_then_second", True), ("second_then_first", False))
            },
            "two_new_retention_from_completion": _conditional(
                pair_rows, "immediate_correct", "correct"
            ),
            "earlier_two_new_retention_from_completion": _conditional(
                [r for r in pair_rows if r["completion_age"] > 0],
                "immediate_correct",
                "correct",
            ),
            "all_written_atoms_accuracy": mean([r["correct"] for r in atom_rows]),
            "earlier_written_atoms_accuracy": mean(
                [r["correct"] for r in atom_rows if r["age"] > 0]
            ),
            "earlier_atoms_retention_from_immediate": _conditional(
                [r for r in atom_rows if r["age"] > 0], "immediate_correct", "correct"
            ),
            "U": {
                pool: {
                    "total": values[0]["total"],
                    "original_correct": values[0]["original_correct"],
                    "coverage": values[0]["coverage"],
                    "retained": _mean([v["retained"] for v in values]),
                    "accuracy": mean(
                        [r["readout"]["core"]["metrics"][pool]["old_accuracy"] for r in at_t]
                    ),
                }
                for pool, values in u.items()
            },
        }
    controls = []
    for p, pair in enumerate(panel["pairs"]):
        labels = {
            "baseline": "baseline",
            "first_only": f"single/f{pair['first']:03d}",
            "second_only": f"single/f{pair['second']:03d}",
            "both_forward": f"pair/p{p:03d}/d0",
            "both_reverse": f"pair/p{p:03d}/d1",
        }
        controls.append(
            {
                "pair_id": pair["pair_id"],
                "type": pair["type"],
                "conditions": {
                    label: by_key[key]["readout"]["pairs"][p] for label, key in labels.items()
                },
            }
        )
    return {
        "primary_t": settings["primary_t"],
        "primary": curves[str(settings["primary_t"])],
        "curves": curves,
        "by_fact_age": {
            age: {
                "accuracy": mean([r["correct"] for r in rows]),
                "retention_from_immediate": _conditional(rows, "immediate_correct", "correct"),
            }
            for age, rows in atom_age.items()
        },
        "by_pair_completion_age": {
            age: {
                "accuracy": mean([r["correct"] for r in rows]),
                "retention_from_completion": _conditional(rows, "immediate_correct", "correct"),
                "both_atoms_correct": _conditional(rows, "both_atoms_correct", "correct"),
            }
            for age, rows in pair_age.items()
        },
        "by_constituent_age_and_gap": {
            axis: {
                str(value): {
                    "n": len(rows),
                    "accuracy": mean([r["correct"] for r in rows]),
                    "both_atoms_correct": _conditional(rows, "both_atoms_correct", "correct"),
                }
                for value in sorted({r[axis] for values in pair_age.values() for r in values})
                if (rows := [r for values in pair_age.values() for r in values if r[axis] == value])
            }
            for axis in ("first_age", "second_age", "write_gap")
        },
        "omission_and_isolated_pair_controls": controls,
        "selection": "All frozen pairs and facts; conditional subsets reported with denominators",
        "analysis_unit": (
            "Fixed pairs/keys in one observed graph; rotations are repeated measures"
        ),
    }


def audit(
    config, settings, name, out, panel_path, device, *, reader_loader=None, identity_factory=None
):
    if reader_loader is None:
        from .ttcd_verified_base import load_reader

        reader_loader = load_reader
    out = Path(out)
    reader, world = reader_loader(config, name, device)
    panel = _load_panel(config, settings, world, panel_path)
    bound = (identity_factory or identity)(config, settings, name, panel)
    summary = json.loads((out / "summary.json").read_text())
    if summary["identity"] != bound:
        raise ValueError("Supplementary audit identity changed")
    plans = state_plan(panel)
    paths = list((out / "states").rglob("record.json"))
    if {str(p.parent.relative_to(out / "states")) for p in paths} != set(plans) or len(
        paths
    ) != len(plans):
        raise ValueError("Incomplete or unexpected supplementary states")
    if summary["states"] != len(plans) or summary["state_keys"] != list(plans):
        raise ValueError("Supplementary summary state matrix differs")
    records, queries = [], 0
    original_base = tensor_digest(reader.backbone.state_dict())
    for key, spec in plans.items():
        directory = out / "states" / key
        record = json.loads((directory / "record.json").read_text())
        fast, provenance = load_fast(directory / "fast.pt", reader)
        if record["key"] != key or record["spec"] != spec or provenance != spec:
            raise ValueError("Supplementary state provenance differs")
        if spec["previous"] is None:
            rebuilt, history = reader.empty_fast(), None
        else:
            prior, _ = load_fast(out / "states" / spec["previous"] / "fast.pt", reader)
            rebuilt, history = original.write_fact(
                reader, prior, config, world, panel["facts"][spec["write"]], device
            )
        if (
            tensor_digest(rebuilt) != record["fast_digest"]
            or tensor_digest(fast) != record["fast_digest"]
        ):
            raise AssertionError("Supplementary write transition mismatch")
        original._compare_nested(history, record["write_history"], "write_history")
        replay = read_state(reader, fast, world, panel, spec, device)
        original._compare_readouts(replay["core"], record["readout"]["core"])
        original._compare_nested(
            {k: v for k, v in replay.items() if k != "core"},
            {k: v for k, v in record["readout"].items() if k != "core"},
            "supplementary",
        )
        queries += sum(len(p["answer"]) for p in replay["core"]["predictions"].values())
        queries += sum(len(p["answer"]) for p in replay["predictions"].values())
        records.append({**record, "readout": replay})
    recomputed = summarize(settings, panel, records, world.metadata)
    original._compare_nested(recomputed, {key: summary[key] for key in recomputed}, "summary")
    if tensor_digest(reader.backbone.state_dict()) != original_base:
        raise AssertionError("Supplementary audit mutated frozen base")
    result = {
        "passed": True,
        "identity": bound,
        "states": len(plans),
        "queries": queries,
        "write_transitions": len(plans) - 1,
        "answer_and_EOS_exact": True,
        "write_states_exact": True,
        "summary_recomputed": True,
        "scientific_training_updates": 0,
        "utc": utc(),
    }
    write_json(out / "audit.json", result)
    return result
