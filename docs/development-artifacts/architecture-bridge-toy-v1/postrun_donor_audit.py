"""Post hoc donor-purity audit using existing predictions only; no new generation."""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from llm_memory_editability.twohop_depth import ENTITY, EOS, digest, write

ROOT = Path(__file__).resolve().parents[3]
ART = Path(__file__).resolve().parent
CFG = json.loads((ROOT / "configs/architecture-bridge-toy-v1.json").read_text())


def main():
    case_records, run_records = [], []
    aggregated = defaultdict(list)
    for seed in CFG["worlds"]:
        world = dict(np.load(ROOT / f"data/architecture-bridge-toy-v1/world-{seed}.npz"))
        cases = world["cases"]
        relations = CFG["relations"]
        correct_ids = (cases[:, 9] * relations + cases[:, 10]) * relations + cases[:, 3]
        wrong_ids = (cases[:, 6] * relations + cases[:, 2]) * relations + cases[:, 3]
        correct_target_visible = cases[:, 9] == cases[:, 5]
        wrong_target_visible = cases[:, 6] == cases[:, 5]
        correct_competitor_visible = cases[:, 9] == cases[:, 8]
        wrong_competitor_visible = cases[:, 6] == cases[:, 8]
        any_target_visible = correct_target_visible | wrong_target_visible
        any_candidate_visible = (
            any_target_visible | correct_competitor_visible | wrong_competitor_visible
        )
        masks = {
            "frozen_full": np.ones(len(cases), dtype=bool),
            "exclude_explicit_correct_answer": ~any_target_visible,
            "exclude_either_explicit_scored_candidate": ~any_candidate_visible,
        }
        for i, case in enumerate(cases):
            assert world["composite_y"][correct_ids[i]] == ENTITY + case[5]
            assert world["composite_y"][wrong_ids[i]] == ENTITY + case[8]
            case_records.append(
                {
                    "world": seed,
                    "case_id": int(case[0]),
                    "correct_donor_composition_id": int(correct_ids[i]),
                    "wrong_donor_composition_id": int(wrong_ids[i]),
                    "correct_donor_split": "train"
                    if world["train_mask"][correct_ids[i]]
                    else "heldout",
                    "wrong_donor_split": "train"
                    if world["train_mask"][wrong_ids[i]]
                    else "heldout",
                    "correct_donor_exposes_correct_answer": bool(correct_target_visible[i]),
                    "wrong_donor_exposes_correct_answer": bool(wrong_target_visible[i]),
                    "correct_donor_exposes_wrong_candidate": bool(correct_competitor_visible[i]),
                    "wrong_donor_exposes_wrong_candidate": bool(wrong_competitor_visible[i]),
                }
            )
        for init in CFG["initializations"]:
            for architecture in CFG["architectures"]:
                name = f"w{seed}-s{init}-{architecture['name']}"
                folder = ROOT / "results/architecture-bridge-toy-v1" / name
                predictions = dict(np.load(folder / f"predictions-{CFG['steps']}.npz"))
                diag = dict(np.load(folder / "diagnostics.npz"))
                full_pred = np.empty(len(world["comps"]), dtype=np.int64)
                full_eos = np.empty_like(full_pred)
                for label, mask in (("train", world["train_mask"]), ("test", ~world["train_mask"])):
                    full_pred[mask] = predictions[f"{label}_pred"]
                    full_eos[mask] = predictions[f"{label}_eos"]
                donor_success = {}
                for label, ids in (("correct", correct_ids), ("wrong", wrong_ids)):
                    exact = (full_pred[ids] == world["composite_y"][ids]) & (full_eos[ids] == EOS)
                    donor_success[label] = {
                        "n": len(ids),
                        "exact_answer_eos_n": int(exact.sum()),
                        "case_exact": exact.tolist(),
                    }
                conditions = json.loads((ART / name / "diagnostics.json").read_text())
                baseline_ok = (diag["baseline_pred"] == diag["target"]) & (
                    diag["baseline_eos"] == EOS
                )
                sensitivities = {}
                for label, mask in masks.items():
                    rows = {}
                    for condition in conditions:
                        exact = (diag[f"{condition}_pred"] == diag["target"]) & (
                            diag[f"{condition}_eos"] == EOS
                        )
                        wrong_exact = (diag[f"{condition}_pred"] == diag["wrong_target"]) & (
                            diag[f"{condition}_eos"] == EOS
                        )
                        item = {
                            "n": int(mask.sum()),
                            "correct_n": int(exact[mask].sum()),
                            "wrong_n": int(wrong_exact[mask].sum()),
                            "rescued": int((exact & ~baseline_ok & mask).sum()),
                            "damaged": int((~exact & baseline_ok & mask).sum()),
                            "baseline_correct_n": int((baseline_ok & mask).sum()),
                            "margin_sum": float(
                                diag[f"{condition}_margin"][mask].astype(float).sum()
                            ),
                        }
                        rows[condition] = item
                        aggregated[(architecture["name"], label, condition)].append(item)
                    sensitivities[label] = rows
                run_records.append(
                    {
                        "name": name,
                        "world": seed,
                        "initialization": init,
                        "architecture": architecture["name"],
                        "donor_success": donor_success,
                        "sensitivity": sensitivities,
                    }
                )
    aggregate_rows = []
    for (architecture, subset, condition), values in aggregated.items():
        totals = {key: sum(row[key] for row in values) for key in values[0]}
        totals["correct_exact"] = totals["correct_n"] / totals["n"]
        totals["mean_margin"] = totals["margin_sum"] / totals["n"]
        aggregate_rows.append(
            {
                "architecture": architecture,
                "subset": subset,
                "condition": condition,
                "runs": len(values),
                "independent_worlds": 2,
                **totals,
            }
        )
    report = {
        "scope": (
            "Post hoc transparency and sensitivity analysis; frozen main metrics remain unchanged"
        ),
        "additional_training_steps": 0,
        "additional_generations": 0,
        "scoring_source": (
            "Existing endpoint evaluation of all train/heldout compositions "
            "and frozen patch predictions"
        ),
        "distinct_cases": len(case_records),
        "correct_target_exposed_cases": sum(
            row["correct_donor_exposes_correct_answer"] or row["wrong_donor_exposes_correct_answer"]
            for row in case_records
        ),
        "either_candidate_exposed_cases": sum(
            any(row[key] for key in row if "exposes" in key) for row in case_records
        ),
        "donor_split_counts": {
            f"{donor}_{split}": sum(row[f"{donor}_donor_split"] == split for row in case_records)
            for donor in ("correct", "wrong")
            for split in ("train", "heldout")
        },
        "limitations": [
            (
                "The reused world audit's answer_free_donors flag only checks the wrong donor's "
                "counterfactual final answer; it is not a guarantee that every donor lacks "
                "either scored candidate."
            ),
            (
                "Correct donor is a semantic relation-world label, not a guarantee that the "
                "model retrieves the correct bridge or answer."
            ),
            (
                "The L1 position3 mixer donor was computed on a complete two-hop question. "
                "It may carry a final-answer representation or a memorized training-composition "
                "representation; it is not a pure first-hop donor "
                "or bridge-specific mechanism proof."
            ),
            (
                "The L0 position2 MLP output is causal and does not see relation2, but its donor "
                "prefix can explicitly contain one scored entity. Such cases are flagged, "
                "not removed from the frozen analysis."
            ),
            (
                "Sensitivity subsets are post hoc, descriptive and correlated across "
                "initialization; they are not new independent experiments."
            ),
        ],
        "case_flags": case_records,
        "runs": run_records,
        "aggregate_descriptive_only": aggregate_rows,
        "audit_script_sha256": digest(Path(__file__)),
    }
    write(ART / "donor-purity-sensitivity.json", report)
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "distinct_cases",
                    "correct_target_exposed_cases",
                    "either_candidate_exposed_cases",
                    "donor_split_counts",
                    "additional_training_steps",
                    "additional_generations",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
