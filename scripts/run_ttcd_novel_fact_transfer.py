"""Evaluate absent-key additions using the audited, frozen DiscoLoop writer.

Run with PYTHONPATH pointing at the parent's frozen source. Only the explicit
atomic packet is passed to write(); evaluation and audit never update theta.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from check_discoloop_edit_rounds import inspect_rows, save_json, sha256
from llm_memory_editability.grokking_reproduction import encode_rows
from llm_memory_editability.ttcd_editing import load_fast, save_fast, tensor_digest
from llm_memory_editability.ttcd_verified_base import load_reader


def addition_packet(fact, metadata, writer):
    """No invented old tail: a missing key has no original ground truth."""
    h, r, t = map(int, fact)
    if not (0 <= h < metadata["entities"] and 0 <= t < metadata["entities"]):
        raise ValueError("Addition entities are outside the inherited vocabulary")
    if not 0 <= r < metadata["relations"]:
        raise ValueError("Addition relation is outside the inherited vocabulary")
    eo, ro = metadata["entity_offset"], metadata["relation_offset"]
    packet = [h + eo, r + ro, t + eo, metadata["end_marker"]]
    packet += [metadata["pad_token"]] * writer["gap"] + [h + eo, r + ro]
    end = len(packet) - 1
    if end >= writer["teacher_window"] or end - 3 < writer["student_window"]:
        raise ValueError("Record must be entirely visible to long and absent from short view")
    return packet


def validate_fact(fact, world):
    h, r, t = fact["novel_fact"]
    if world.lookup[h, r] != -1 or fact["preexisting_truth_for_novel_key"] is not None:
        raise ValueError("This batch only adds absent keys, never overwrites existing truth")
    if not 0 <= t < world.entities:
        raise ValueError("Invalid new entity")
    for role in ("first_hop", "second_hop"):
        for query in fact[role]:
            a, r1, r2 = query["query"]
            known = world.atoms[query["known_atom"]].tolist()
            if known != query["known_fact"]:
                raise ValueError("Known companion truth changed")
            if role == "first_hop":
                valid = (a, r1, query["bridge"]) == (h, r, t) and known == [t, r2, query["answer"]]
            else:
                valid = known == [a, r1, h] and (r2, query["answer"], query["bridge"]) == (r, t, h)
            if not valid:
                raise ValueError("A query must use exactly this new edge at its declared hop")


@torch.no_grad()
def greedy_rows(reader, fast, rows, metadata, batch_size=64):
    rows = np.asarray(rows, dtype=np.int64)
    output = {key: [] for key in ("rows", "answer", "stop", "expected", "final_correct")}
    for offset in range(0, len(rows), batch_size):
        part = rows[offset : offset + batch_size]
        device = next(reader.parameters()).device
        tokens, positions, labels = (
            torch.as_tensor(a, device=device) for a in encode_rows(part, metadata)
        )
        logits = reader(tokens, positions, fast=fast)
        answer = logits[:, 0].argmax(-1)
        tokens[torch.arange(len(part), device=device), positions[:, 1]] = answer
        stop = reader(tokens, positions, fast=fast)[:, 1].argmax(-1)
        output["rows"].extend(part.tolist())
        output["answer"].extend(answer.cpu().tolist())
        output["stop"].extend(stop.cpu().tolist())
        output["expected"].extend(labels[:, 0].cpu().tolist())
        output["final_correct"].extend(
            ((answer == labels[:, 0]) & (stop == labels[:, 1])).cpu().tolist()
        )
    return output


def trace_rows(reader, fast, rows, metadata, *, kind, bridges=None):
    """Reuse the verified observation; do not assign an 'old answer' to new keys."""
    result = inspect_rows(
        reader, fast, rows, metadata, np.asarray(rows)[:, -1], kind=kind, bridges=bridges
    )
    result["expected"] = result.pop("expected_new")
    result["final_correct"] = result.pop("final_new_joint")
    del result["old"], result["final_old_joint"]
    for stage in result["stages"].values():
        for key in ("old_correct", "new_correct", "new_minus_old"):
            stage["answer"].pop(key, None)
    return result


def two_step(reader, fast, queries, first_readout, metadata):
    """Ask the second atom with the model's head, never an oracle bridge."""
    eo = metadata["entity_offset"]
    valid = [eo <= token < eo + metadata["entities"] for token in first_readout["answer"]]
    selected = [i for i, ok in enumerate(valid) if ok]
    second_rows = [
        [first_readout["answer"][i] - eo, queries[i]["query"][2], queries[i]["answer"]]
        for i in selected
    ]
    second = greedy_rows(reader, fast, second_rows, metadata) if selected else None
    answer, stop, correct = [None] * len(queries), [None] * len(queries), [False] * len(queries)
    for j, i in enumerate(selected):
        answer[i], stop[i] = second["answer"][j], second["stop"][j]
        correct[i] = bool(
            second["final_correct"][j] and first_readout["stop"][i] == metadata["end_marker"]
        )
    return {
        "first_answer": first_readout["answer"],
        "first_stop": first_readout["stop"],
        "first_bridge_correct": [
            token == q["bridge"] + eo
            for token, q in zip(first_readout["answer"], queries, strict=True)
        ],
        "valid_entity_head": valid,
        "second_rows": second_rows,
        "second_row_indices": selected,
        "answer": answer,
        "stop": stop,
        "final_correct": correct,
    }


def query_rows(queries):
    return [q["query"] + [q["answer"]] for q in queries]


@torch.no_grad()
def evaluate_state(reader, fast, fact, world, panel, baseline_U=None):
    meta = world.metadata
    atomic = trace_rows(reader, fast, [fact["novel_fact"]], meta, kind="novel_atom")
    result = {"novel_atom": atomic}
    for role in ("first_hop", "second_hop"):
        queries = fact[role]
        companion = greedy_rows(reader, fast, [q["known_fact"] for q in queries], meta)
        result[role] = trace_rows(
            reader,
            fast,
            query_rows(queries),
            meta,
            kind=role,
            bridges=[q["bridge"] for q in queries],
        )
        result[role]["companion"] = companion
        first = (
            {key: atomic[key] * len(queries) for key in ("answer", "stop")}
            if role == "first_hop"
            else companion
        )
        result[role]["explicit_two_step"] = two_step(reader, fast, queries, first, meta)
    result["old_matched_atom"] = greedy_rows(
        reader, fast, [fact["historical_reference"]["old_fact"]], meta
    )
    result["old_matched_chains"] = greedy_rows(
        reader,
        fast,
        [q["historical_query"] + [q["historical_answer"]] for q in fact["second_hop"]],
        meta,
    )
    if baseline_U is None:
        result["U_atoms"] = greedy_rows(reader, fast, world.atoms[panel["U_atoms"]], meta)
        result["U_chains"] = greedy_rows(reader, fast, world.chains[panel["U_chains"]], meta)
    else:
        result.update(baseline_U)
    return result


def rate(values):
    values = list(values)
    return {
        "correct": sum(values),
        "n": len(values),
        "accuracy": sum(values) / len(values) if values else None,
    }


def bootstrap(values, settings):
    values = np.asarray(values, dtype=float)
    draws = np.random.RandomState(settings["bootstrap_seed"]).randint(
        len(values), size=(settings["bootstrap_draws"], len(values))
    )
    limits = np.quantile(values[draws].mean(-1), [0.025, 0.975])
    return {
        "fact_mean": float(values.mean()),
        "ci95_fact_bootstrap": limits.tolist(),
        "facts": len(values),
    }


def summarize(records, settings):
    result = {"facts": len(records), "conditions": {}, "retention": {}, "conditional": {}}
    kinds = ("novel_atom", "first_hop", "second_hop")
    for condition in ("before", "after"):
        pools = {}
        for kind in kinds:
            items = [r[condition][kind] for r in records]
            pool = rate(v for item in items for v in item["final_correct"])
            pool.update(bootstrap([np.mean(item["final_correct"]) for item in items], settings))
            pool["stages"] = {}
            for stage in items[0]["stages"]:
                stage_result = rate(
                    v for item in items for v in item["stages"][stage]["answer"]["expected_correct"]
                )
                if kind != "novel_atom":
                    stage_result["bridge"] = rate(
                        v
                        for item in items
                        for v in item["stages"][stage]["bridge"]["expected_correct"]
                    )
                pool["stages"][stage] = stage_result
            if kind != "novel_atom":
                pool["companion"] = rate(
                    v for item in items for v in item["companion"]["final_correct"]
                )
                pool["explicit_two_step"] = rate(
                    v for item in items for v in item["explicit_two_step"]["final_correct"]
                )
            pools[kind] = pool
        result["conditions"][condition] = pools
    result["paired_changes"] = {}
    for kind in kinds:
        result["paired_changes"][kind] = bootstrap(
            [
                np.mean(r["after"][kind]["final_correct"])
                - np.mean(r["before"][kind]["final_correct"])
                for r in records
            ],
            settings,
        )
    result["paired_changes"]["second_minus_first_hop_after"] = bootstrap(
        [
            np.mean(r["after"]["second_hop"]["final_correct"])
            - np.mean(r["after"]["first_hop"]["final_correct"])
            for r in records
        ],
        settings,
    )
    for kind in ("U_atoms", "U_chains", "old_matched_atom", "old_matched_chains"):
        before = [v for r in records for v in r["before"][kind]["final_correct"]]
        after = [v for r in records for v in r["after"][kind]["final_correct"]]
        old_tokens = [
            v
            for r in records
            for v in zip(r["before"][kind]["answer"], r["before"][kind]["stop"], strict=True)
        ]
        new_tokens = [
            v
            for r in records
            for v in zip(r["after"][kind]["answer"], r["after"][kind]["stop"], strict=True)
        ]
        result["retention"][kind] = {
            "before": rate(before),
            "after": rate(after),
            "initially_correct_retained": rate(a for a, b in zip(after, before, strict=True) if b),
            "initially_correct_lost": sum(b and not a for a, b in zip(after, before, strict=True)),
            "answer_or_EOS_flipped": sum(
                a != b for a, b in zip(old_tokens, new_tokens, strict=True)
            ),
        }
    for role in ("first_hop", "second_hop"):
        masks, outcomes = {}, []
        for record in records:
            before, after = record["before"][role], record["after"][role]
            n = len(after["final_correct"])
            local = {
                "companion_A0_correct": before["companion"]["final_correct"],
                "companion_A4_correct": after["companion"]["final_correct"],
                "novel_atom_final_correct": record["after"]["novel_atom"]["final_correct"] * n,
                "novel_atom_round1_correct": record["after"]["novel_atom"]["stages"]["round1_end"][
                    "answer"
                ]["expected_correct"]
                * n,
                "actual_bridge_round1_correct": after["stages"]["round1_end"]["bridge"][
                    "expected_correct"
                ],
            }
            joint = [
                all(v)
                for v in zip(
                    *(
                        local[k]
                        for k in (
                            "companion_A0_correct",
                            "companion_A4_correct",
                            "novel_atom_final_correct",
                            "actual_bridge_round1_correct",
                        )
                    ),
                    strict=True,
                )
            ]
            local["all_final_prerequisites"] = joint
            local["all_final_prerequisites_and_atom_round1"] = [
                a and b for a, b in zip(joint, local["novel_atom_round1_correct"], strict=True)
            ]
            for key, mask in local.items():
                masks.setdefault(key, []).extend(mask)
            outcomes.extend(after["final_correct"])
        result["conditional"][role] = {
            key: {
                **rate(o for o, keep in zip(outcomes, mask, strict=True) if keep),
                "coverage_n": sum(mask),
                "full_n": len(mask),
            }
            for key, mask in masks.items()
        }
    result["baseline_spontaneous_atomic_correct"] = rate(
        r["before"]["novel_atom"]["final_correct"][0] for r in records
    )
    result["teacher_record_visible"] = rate(r["teacher"]["correct"] for r in records)
    return result


def initialize(args):
    design = json.loads(args.design.read_text())
    if sha256(args.panel) != design["panel"]["sha256"]:
        raise ValueError("Frozen addition panel changed")
    panel = json.loads(args.panel.read_text())
    if sha256(design["parent"]["checkpoint"]) != design["parent"]["checkpoint_sha256"]:
        raise ValueError("Pinned existing checkpoint changed")
    if (
        sha256(design["parent"]["baseline_fast_state"])
        != design["parent"]["baseline_fast_state_sha256"]
    ):
        raise ValueError("Pinned A0 changed")
    config = json.loads(args.parent_config.read_text())
    reader, world = load_reader(config, "single", "cpu")
    reader.requires_grad_(False)
    if any(p.requires_grad for p in reader.parameters()):
        raise AssertionError("All slow parameters must be frozen")
    baseline, _ = load_fast(design["parent"]["baseline_fast_state"], reader)
    if any(torch.count_nonzero(v) for v in baseline.values()):
        raise AssertionError("Expected the inherited zero A0")
    for fact in panel["facts"]:
        validate_fact(fact, world)
    bound = {str(p): sha256(p) for p in (args.design, args.panel, args.parent_config)}
    for entry in panel["inputs"].values():
        if sha256(entry["path"]) != entry["sha256"]:
            raise ValueError(f"Input changed: {entry['path']}")
        bound[entry["path"]] = entry["sha256"]
    for key in ("checkpoint", "baseline_fast_state"):
        bound[design["parent"][key]] = design["parent"][f"{key}_sha256"]
    return design, panel, reader, world, baseline, bound


def start_wandb(args, design):
    if args.tracking == "disabled":
        return None
    import wandb

    run = wandb.init(
        entity="zhusq20",
        project="llm-memory-editability",
        group=design["batch"],
        name=f"absent-key/{args.phase}",
        job_type="test-time-addition-evaluation",
        dir=str(args.out),
        mode=args.tracking,
        config={"design": design, "phase": args.phase, "slow_parameter_updates": 0},
        settings=wandb.Settings(disable_git=True),
    )
    save_json(args.out / "wandb.json", {"id": run.id, "url": run.url})
    return run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--parent-config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--phase", choices=("development", "evaluation", "audit"), required=True)
    parser.add_argument("--input-root", type=Path)
    parser.add_argument("--tracking", choices=("online", "offline", "disabled"), default="online")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    design, panel, reader, world, baseline, bound = initialize(args)
    model_digest = tensor_digest(reader.backbone.state_dict())
    baseline_digest = tensor_digest(baseline)
    run = start_wandb(args, design)
    baseline_U = {
        "U_atoms": greedy_rows(reader, baseline, world.atoms[panel["U_atoms"]], world.metadata),
        "U_chains": greedy_rows(reader, baseline, world.chains[panel["U_chains"]], world.metadata),
    }
    records = []
    fast_updates = 0
    facts = (
        panel["facts"]
        if args.phase == "audit"
        else [f for f in panel["facts"] if f["split"] == args.phase]
    )
    for index, fact in enumerate(facts):
        fact_started = time.monotonic()
        state_path = args.out / "states" / f"{fact['case_id']}.pt"
        if args.phase == "audit":
            directory = args.input_root / fact["split"]
            saved_record = json.loads((directory / f"{fact['case_id']}.json").read_text())
            fast, provenance = load_fast(directory / "states" / f"{fact['case_id']}.pt", reader)
            if provenance["fact"] != fact or provenance["fast_updates"] != 4:
                raise AssertionError("Saved fast state identity differs")
            write_info = saved_record["write_info"]
        else:
            packet = torch.tensor(
                [addition_packet(fact["novel_fact"], world.metadata, design["writer"])]
            )
            fast = {k: v.clone() for k, v in baseline.items()}
            write_info = []
            for _ in range(design["writer"]["steps_per_fact"]):
                fast, info = reader.write(
                    packet,
                    fast,
                    teacher_window=design["writer"]["teacher_window"],
                    student_window=design["writer"]["student_window"],
                    learning_rate=design["writer"]["learning_rate"],
                )
                write_info.append(info)
                fast_updates += 1
            save_fast(state_path, reader, fast, {"fact": fact, "fast_updates": 4})
        packet = torch.tensor(
            [addition_packet(fact["novel_fact"], world.metadata, design["writer"])]
        )
        with torch.no_grad():
            position = torch.tensor([[packet.shape[1] - 1]])
            teacher_logits = reader(
                packet, position, fast=baseline, window=design["writer"]["teacher_window"]
            )
            teacher_token = int(teacher_logits[0, 0].argmax())
        record = {
            "fact": fact,
            "write_info": write_info,
            "teacher": {
                "answer": teacher_token,
                "correct": teacher_token == fact["novel_fact"][2] + world.eo,
            },
            "before": evaluate_state(reader, baseline, fact, world, panel, baseline_U),
            "after": evaluate_state(reader, fast, fact, world, panel),
            "fast_digest": tensor_digest(fast),
        }
        if args.phase == "audit":
            if record != saved_record:
                raise AssertionError(f"Fresh reload/readout differs: {fact['case_id']}")
        save_json(args.out / f"{fact['case_id']}.json", record)
        records.append(record)
        progress = {
            "phase": args.phase,
            "completed_facts": index + 1,
            "facts": len(facts),
            "fast_updates": fast_updates,
            "slow_parameter_updates": 0,
            "seconds": time.monotonic() - started,
            "last_fact_seconds": time.monotonic() - fact_started,
        }
        save_json(args.out / "progress.json", progress)
        print(json.dumps(progress), flush=True)
        if run:
            run.log(progress, step=index + 1)
    if (
        tensor_digest(reader.backbone.state_dict()) != model_digest
        or tensor_digest(baseline) != baseline_digest
    ):
        raise AssertionError("Frozen theta or inherited A0 changed")
    if any(sha256(path) != digest for path, digest in bound.items()):
        raise AssertionError("Read-only input changed")
    summary = {
        "schema": "ttcd-novel-fact-transfer-result-v1",
        "phase": args.phase,
        "passed": True,
        "slow_parameter_updates": 0,
        "fast_updates": fast_updates,
        "backbone_unchanged": True,
        "baseline_unchanged": True,
        "all_inputs_unchanged": True,
        "new_keys_only": True,
        "hook_logits_exact": True,
        "precision": "fp32",
        "device": "cpu",
        "results": {
            phase: summarize(
                [r for r in records if r["fact"]["split"] == phase],
                design["evaluation"]["statistics"],
            )
            for phase in sorted({f["split"] for f in facts})
        },
        "input_sha256": bound,
        "model_digest": model_digest,
        "compute": reader.compute,
        "wall_seconds": time.monotonic() - started,
        "fresh_reload_all_record_fields_exact": args.phase == "audit",
    }
    save_json(args.out / "summary.json", summary)
    if run:
        run.summary.update(summary)
        run.save(str(args.out / "summary.json"), base_path=str(args.out), policy="now")
        run.finish()
    print(
        json.dumps({"complete": True, "phase": args.phase, "fast_updates": fast_updates}),
        flush=True,
    )


if __name__ == "__main__":
    main()
