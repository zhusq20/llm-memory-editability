"""World-paired E39 organization interactions and exception answer signatures."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np

from llm_memory_editability.bios_cross import make_cross_world


def write_csv(path, rows):
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(source):
    source = Path(source).resolve()
    output = source / "summary"
    audit = json.loads((output / "audit.json").read_text())
    if not audit["complete"] or audit["cases"] != 96:
        raise ValueError("Full audited E39 matrix required")
    rows = list(csv.DictReader((output / "editing.csv").open()))
    rows = [row for row in rows if row["step"] == "512"]
    interactions, organizations = [], []
    for world in (0, 1):
        for seed in ("0", "1", "both"):
            for kind in ("coherent", "exception"):
                common = [
                    r
                    for r in rows
                    if int(r["world"]) == world
                    and r["kind"] == kind
                    and (seed == "both" or r["seed"] == seed)
                ]
                metrics = {}
                for phase in ("low", "high"):
                    cells = {}
                    for condition in ("company", "project", "neither"):
                        for chain in ("company", "project"):
                            group = [
                                r
                                for r in common
                                if (r["phase"], r["condition"], r["chain"])
                                == (phase, condition, chain)
                            ]
                            assert len(group) == (2 if seed == "both" else 1)
                            cells[condition, chain] = mean(
                                float(r["paired_reference_D_heldout_accuracy"]) for r in group
                            )
                        organizations.append(
                            dict(
                                world=world,
                                seed=seed,
                                kind=kind,
                                phase=phase,
                                condition=condition,
                                accuracy=mean(
                                    cells[condition, chain] for chain in ("company", "project")
                                ),
                            )
                        )
                    cc, cp = cells["company", "company"], cells["company", "project"]
                    pc, pp = cells["project", "company"], cells["project", "project"]
                    nc, np_ = cells["neither", "company"], cells["neither", "project"]
                    metrics[phase] = dict(
                        matching=((cc - pc) + (pp - cp)) / 2,
                        matched_vs_neither=((cc - nc) + (pp - np_)) / 2,
                        mismatched_vs_neither=((cp - np_) + (pc - nc)) / 2,
                        company_vs_neither=(cc + cp - nc - np_) / 2,
                        project_vs_neither=(pc + pp - nc - np_) / 2,
                    )
                for metric in metrics["low"]:
                    interactions.append(
                        dict(
                            world=world,
                            seed=seed,
                            kind=kind,
                            metric=metric,
                            low=metrics["low"][metric],
                            high=metrics["high"][metric],
                            high_minus_low=metrics["high"][metric] - metrics["low"][metric],
                        )
                    )
    signatures = []
    for world_seed in (0, 1):
        world = make_cross_world(world_seed)
        for phase in ("low", "high"):
            for seed in (0, 1):
                for condition in ("company", "project", "neither"):
                    for chain, chain_name in enumerate(("company", "project")):
                        dest = (
                            source
                            / phase
                            / "width-256"
                            / f"world-{world_seed}-seed-{seed}-{condition}"
                        )
                        dest /= f"{chain_name}-exception-mlp"
                        with np.load(dest / "sets.npz") as saved:
                            sets = dict(saved)
                        target = sets[f"{phase}_exception"]
                        for step in (0, 32, 128, 512):
                            with np.load(dest / f"predictions-{step}.npz") as saved:
                                predictions, ended = saved["prediction"], saved["ended"]
                            for cohort, key in (
                                ("all18", "exception_conflict_D"),
                                ("heldout9", "exception_conflict_D_heldout"),
                            ):
                                ids = sets[key]
                                actual = target[world.actual_ids[chain, world.person[ids]]]
                                old = world.answers[ids]
                                assert np.all(actual != target[ids]) and np.all(actual != old)
                                assert np.all(old != target[ids])
                                masks = dict(
                                    correct=(predictions[ids] == target[ids]) & ended[ids],
                                    new_actual=(predictions[ids] == actual) & ended[ids],
                                    old_default=(predictions[ids] == old) & ended[ids],
                                    no_EOS=~ended[ids],
                                )
                                masks["other"] = ~np.logical_or.reduce(list(masks.values()))
                                assert sum(int(mask.sum()) for mask in masks.values()) == len(ids)
                                signatures.append(
                                    dict(
                                        world=world_seed,
                                        seed=seed,
                                        condition=condition,
                                        chain=chain_name,
                                        phase=phase,
                                        step=step,
                                        cohort=cohort,
                                        n=len(ids),
                                        **{key: int(mask.sum()) for key, mask in masks.items()},
                                    )
                                )
    grouped = defaultdict(list)
    for row in signatures:
        if row["step"] == 512:
            grouped[row["world"], row["phase"], row["cohort"]].append(row)
    totals = [
        dict(
            world=key[0],
            phase=key[1],
            cohort=key[2],
            **{
                name: sum(r[name] for r in group)
                for name in ("n", "correct", "new_actual", "old_default", "no_EOS", "other")
            },
        )
        for key, group in sorted(grouped.items())
    ]
    for filename, data in (
        ("organization-interactions.csv", interactions),
        ("organization-reference-accuracy.csv", organizations),
        ("exception-answer-signatures.csv", signatures),
        ("exception-answer-signatures-world.csv", totals),
    ):
        write_csv(output / filename, data)
    print(
        json.dumps(
            {
                "interactions": len(interactions),
                "signature_cells": len(signatures),
                "fixed_endpoint_world_counts": totals,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="results/bios-mechanism-dev-v1/p3-shortcut-edit-v2")
    analyze(parser.parse_args().source)
