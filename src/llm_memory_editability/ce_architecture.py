"""Endpoint comparisons for identical atomic-CE edits across native architectures."""

from __future__ import annotations

import numpy as np
import torch

from .alphaedit_summary import bootstrap, rate
from .grokking_reproduction import encode_rows
from .ttcd_editing import tensor_digest


def old_retention_rates(totals):
    rates = {
        name: value["retained"] / value["originally_correct"]
        if value["originally_correct"]
        else None
        for name, value in totals.items()
    }
    defined = [value for value in rates.values() if value is not None]
    if not defined:
        raise ValueError("No originally correct old questions; retention is undefined")
    return rates, min(defined)


@torch.no_grad()
def inspect_endpoints(reader, fast, rows, metadata, expected, *, kind, bridges=None):
    """Keep native greedy answer/EOS scoring without inventing recurrent stages."""
    device = next(reader.parameters()).device
    tokens, positions, labels = (
        torch.as_tensor(v, device=device)
        for v in encode_rows(np.asarray(rows, dtype=np.int64), metadata)
    )
    before = tensor_digest(fast)
    answer = reader(tokens, positions, fast=fast)[:, 0].argmax(-1)
    tokens[torch.arange(len(tokens), device=device), positions[:, 1]] = answer
    stop = reader(tokens, positions, fast=fast)[:, 1].argmax(-1)
    new = torch.as_tensor(expected, device=device) + metadata["entity_offset"]
    if tensor_digest(fast) != before:
        raise AssertionError("Readout changed the edit state")
    return {
        "kind": kind,
        "rows": np.asarray(rows).tolist(),
        "answer_position": positions[:, 0].cpu().tolist(),
        "answer": answer.cpu().tolist(),
        "stop": stop.cpu().tolist(),
        "expected_new": new.cpu().tolist(),
        "old": labels[:, 0].cpu().tolist(),
        "final_new_joint": ((answer == new) & (stop == metadata["end_marker"])).cpu().tolist(),
        "final_old_joint": ((answer == labels[:, 0]) & (stop == metadata["end_marker"]))
        .cpu()
        .tolist(),
        "stages": {},
        "observation": "native endpoint only; no architectural boundary assumed",
    }


def summarize_endpoints(records, settings):
    """Apply the same endpoint rules to new arms and saved DiscoLoop records."""
    result = {
        "facts": len(records),
        "conditions": {},
        "retention": {},
        "conditional": {},
    }
    kinds = ("novel_atom", "first_hop", "second_hop")
    for condition in ("before", "after"):
        pools = {}
        for kind in kinds:
            items = [r[condition][kind] for r in records]
            pool = rate(v for item in items for v in item["final_correct"])
            pool.update(bootstrap([np.mean(item["final_correct"]) for item in items], settings))
            if kind != "novel_atom":
                for key in ("companion", "explicit_two_step"):
                    pool[key] = rate(v for item in items for v in item[key]["final_correct"])
            pools[kind] = pool
        result["conditions"][condition] = pools
    result["paired_changes"] = {
        kind: bootstrap(
            [
                np.mean(r["after"][kind]["final_correct"])
                - np.mean(r["before"][kind]["final_correct"])
                for r in records
            ],
            settings,
        )
        for kind in kinds
    }
    for kind in ("U_atoms", "U_chains", "old_matched_atom", "old_matched_chains"):
        before = [v for r in records for v in r["before"][kind]["final_correct"]]
        after = [v for r in records for v in r["after"][kind]["final_correct"]]
        result["retention"][kind] = {
            "before": rate(before),
            "after": rate(after),
            "initially_correct_retained": rate(a for a, b in zip(after, before, strict=True) if b),
            "initially_correct_lost": sum(b and not a for a, b in zip(after, before, strict=True)),
        }
    for role in ("first_hop", "second_hop"):
        masks, outcomes = {}, []
        for record in records:
            before, after = record["before"][role], record["after"][role]
            local = {
                "companion_A0_correct": before["companion"]["final_correct"],
                "companion_edited_correct": after["companion"]["final_correct"],
                "novel_atom_final_correct": record["after"]["novel_atom"]["final_correct"]
                * len(after["final_correct"]),
            }
            local["both_atoms_current_correct"] = [
                a and b
                for a, b in zip(
                    local["companion_edited_correct"],
                    local["novel_atom_final_correct"],
                    strict=True,
                )
            ]
            local["old_companion_kept_and_new_atom_correct"] = [
                a and b
                for a, b in zip(
                    local["companion_A0_correct"],
                    local["both_atoms_current_correct"],
                    strict=True,
                )
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
    return result


def paired_report(records_by_arm, settings):
    arms = list(records_by_arm)
    reference = [r["fact"] for r in records_by_arm[arms[0]]]
    if len(reference) != 32 or any(
        [r["fact"] for r in records_by_arm[a]] != reference for a in arms[1:]
    ):
        raise AssertionError("Architecture comparison must retain all identical32 facts")
    result = {
        "architectures": {},
        "paired_fact_differences": {},
        "common_prerequisites": {},
    }
    for arm, records in records_by_arm.items():
        summary = summarize_endpoints(records, settings)
        old = {"atoms": [], "compositions": []}
        for record in records:
            for name, before in record["before"]["extended_old"].items():
                after = record["after"]["extended_old"][name]
                group = "atoms" if name.startswith("atomic") else "compositions"
                old[group].extend(
                    a
                    for a, b in zip(after["final_correct"], before["final_correct"], strict=True)
                    if b
                )
        summary["extended_old_retained"] = {k: rate(v) for k, v in old.items()}
        result["architectures"][arm] = summary
    for i, left in enumerate(arms):
        for right in arms[i + 1 :]:
            result["paired_fact_differences"][f"{left}_minus_{right}"] = {
                kind: bootstrap(
                    [
                        np.mean(a["after"][kind]["final_correct"])
                        - np.mean(b["after"][kind]["final_correct"])
                        for a, b in zip(records_by_arm[left], records_by_arm[right], strict=True)
                    ],
                    settings,
                )
                for kind in ("novel_atom", "first_hop", "second_hop")
            }
    for role in ("first_hop", "second_hop"):
        masks = []
        for index in range(32):
            local = [
                [
                    b and a and record["after"]["novel_atom"]["final_correct"][0]
                    for b, a in zip(
                        record["before"][role]["companion"]["final_correct"],
                        record["after"][role]["companion"]["final_correct"],
                        strict=True,
                    )
                ]
                for arm in arms
                for record in [records_by_arm[arm][index]]
            ]
            masks.extend(all(v) for v in zip(*local, strict=True))
        result["common_prerequisites"][role] = {
            "coverage_n": sum(masks),
            "full_n": len(masks),
            "architectures": {
                arm: rate(
                    value
                    for value, keep in zip(
                        (v for r in records for v in r["after"][role]["final_correct"]),
                        masks,
                        strict=True,
                    )
                    if keep
                )
                for arm, records in records_by_arm.items()
            },
        }
    return result
