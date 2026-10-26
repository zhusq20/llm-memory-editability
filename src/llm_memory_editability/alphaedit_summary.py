"""Fact-clustered composition/retention summary without TTCD teacher fields.

Scoring, prerequisites and bootstrap are the inherited novel-fact evaluation
rules. Only the inapplicable teacher field and A4 label are removed/renamed.
"""

import numpy as np


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
                "companion_edited_correct": after["companion"]["final_correct"],
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
                            "companion_edited_correct",
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
    return result
