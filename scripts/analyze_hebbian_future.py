#!/usr/bin/env python3
"""Descriptive source margins, perturbations, and paired rehearsal effects."""

from __future__ import annotations

import csv
import itertools

import numpy as np

from llm_memory_editability.hebbian_future import ART, CONFIG, RESULTS, read, write


def main():
    cfg = read(CONFIG)["chains"]
    rows, world_effects, groups = [], [], []
    for directory in sorted((RESULTS / "chains").glob("w*-r*-d*")):
        if not (directory / "complete.json").exists():
            continue
        meta = read(directory / "complete.json")
        key = {k: meta[k] for k in ("world", "rho", "depth")}
        parent = read(directory / "learning.json")[-1]
        stages = [("parent", parent)]
        stages += [
            (arm["arm"], arm["trajectory"][-1]) for arm in read(directory / "rehearsals.json")
        ]
        for arm, row in stages:
            record = {**key, "arm": arm, **row["summary"]}
            for kind in ("member", "root", "home", "composite"):
                margins = np.asarray(row["rows"][kind]["margin"])
                record[kind + "_margin_median"] = float(np.median(margins))
                record[kind + "_margin_q10"] = float(np.quantile(margins, 0.1))
            rows.append(record)
        for arm in cfg["rehearsal_arms"]:
            final = next(
                r for r in rows if all(r[k] == v for k, v in key.items()) and r["arm"] == arm
            )
            for metric in (
                "held_composite",
                "held_common_conflict_composite",
                "held_two_step",
                "member_accuracy",
                "root_accuracy",
                "home_accuracy",
            ):
                world_effects.append(
                    {
                        **key,
                        "arm": arm,
                        "metric": metric,
                        "change_from_parent": final[metric] - parent["summary"][metric],
                    }
                )
    metrics = [
        "held_composite",
        "held_common_conflict_composite",
        "held_two_step",
        "member_accuracy",
        "root_accuracy",
        "home_accuracy",
        "member_margin_median",
        "root_margin_median",
        "home_margin_median",
    ]
    for depth, rho, arm in itertools.product(
        cfg["layers"], cfg["correlations"], ["parent", *cfg["rehearsal_arms"]]
    ):
        part = [r for r in rows if r["depth"] == depth and r["rho"] == rho and r["arm"] == arm]
        if part:
            groups.append(
                {
                    "depth": depth,
                    "rho": rho,
                    "arm": arm,
                    "worlds": len(part),
                    **{k: float(np.mean([r[k] for r in part])) for k in metrics},
                }
            )
    for name, values in [
        ("rehearsal-detail.csv", rows),
        ("rehearsal-effects.csv", world_effects),
        ("rehearsal-aggregates.csv", groups),
    ]:
        with (ART / name).open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    write(
        ART / "rehearsal-summary.json",
        {
            "aggregates": groups,
            "effects": world_effects,
            "interpretation": (
                "Raw margins are descriptive within model/arm, not scale invariant robustness. "
                "Use perturbation curves separately. Rehearsal changes parameters broadly, "
                "not margin alone."
            ),
        },
    )


if __name__ == "__main__":
    main()
