"""Audit locked P4 diagnostics from sealed raw queries, never refit or select Q.

Completed models alone enter tables. Missing models are explicit partial coverage;
rank-unavailable completed cases stay in the calibration/coverage tables. Accuracy
contrasts always share a query-level availability mask. Conditional knowledge and
donor-answer strata are descriptive, not selection rules for the primary effect.
"""

import argparse
import csv
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from llm_memory_editability import bios_mechanism_causal_v2 as producer_v2
from llm_memory_editability.bios_cross import CHAINS, CONDITIONS, edit_pair, make_cross_world
from llm_memory_editability.bios_data import array_hash, write_json
from llm_memory_editability.bios_mechanism_causal import (
    ARMS,
    FAMILIES,
    TRANSFER_ARMS,
    TRANSFER_TYPES,
    _sources,
    make_probe_plan,
    make_transfer_plan,
    scientific_outputs,
    validate_lock,
    wrong_source_indices,
)
from llm_memory_editability.bios_mechanism_interventions import calibration_plan
from llm_memory_editability.bios_path_diagnostics import file_sha256

IDENTITY = ("width", "world", "seed", "condition", "chain")
COMPARISONS = (
    ("target_remove", "clean"),
    ("random_remove", "clean"),
    ("complement_remove", "clean"),
    ("target_remove", "random_remove"),
    ("target_remove", "complement_remove"),
    ("remove_rescue", "target_remove"),
    ("remove_wrong_source_rescue", "target_remove"),
    ("remove_random_rescue", "target_remove"),
    ("remove_rescue", "remove_wrong_source_rescue"),
    ("remove_rescue", "remove_random_rescue"),
    ("sham", "clean"),
    ("rescue_only", "clean"),
)


def read_npz(path):
    with np.load(path) as stored:
        return dict(stored)


def assert_arrays(actual, expected, context):
    if set(actual) != set(expected):
        raise ValueError(f"Plan fields changed: {context}")
    for key, value in expected.items():
        if not np.array_equal(actual[key], value):
            raise ValueError(f"Plan changed: {context}/{key}")


def read_predictions(path, ids, vocab_size, width, batch_size, lengths):
    arrays = read_npz(path)
    if not np.array_equal(arrays["query_id"], ids):
        raise ValueError(f"Raw query ordering differs: {path}")
    n = len(ids)
    prediction, ended = arrays["prediction"], arrays["ended"]
    if (
        prediction.shape != (n,)
        or prediction.dtype.kind not in "iu"
        or ended.shape != (n,)
        or ended.dtype != bool
    ):
        raise ValueError(f"Malformed free-generation arrays: {path}")
    if ((prediction < 0) | (prediction >= vocab_size)).any():
        raise ValueError(f"Out-of-vocabulary prediction: {path}")
    for key, value in arrays.items():
        if key.endswith("_valid"):
            if value.shape != (n, 2) or value.dtype != bool:
                raise ValueError(f"Malformed operation availability: {path}/{key}")
        elif key.endswith("_norm"):
            if value.shape != (n, 2) or not np.isfinite(value).all() or (value < 0).any():
                raise ValueError(f"Malformed perturbation norm: {path}/{key}")
        elif key.startswith("state_") and not key.endswith("_pass_drift"):
            if value.shape != (n, width) or not np.isfinite(value).all():
                raise ValueError(f"Malformed clean state: {path}/{key}")
    expected_cost = {
        "forward_calls": 2 * ((n + batch_size - 1) // batch_size),
        "forward_examples": 2 * n,
        "padded_token_positions": 11 * n,
        "logical_token_positions": int((2 * np.asarray(lengths) + 1).sum()),
    }
    if any(int(arrays[key]) != value for key, value in expected_cost.items()):
        raise ValueError(f"Generation cost changed: {path}")
    return arrays


def available(arrays):
    mask = np.ones(len(arrays["prediction"]), dtype=bool)
    for key, value in arrays.items():
        if key.endswith("_valid"):
            mask &= value.all(axis=1)
    return mask


def correct(arrays, answers):
    return (arrays["prediction"] == answers) & arrays["ended"]


def population_mask(plan, label, code, population):
    chosen = plan[label] == code
    if population != "all":
        chosen &= plan["recipient_old_exception"] == (population == "old_exception")
    return chosen


def condition_rows(arrays, baseline, plan, arm, families=FAMILIES, label_key="family"):
    before, after, valid = (
        correct(baseline, plan["answer"]),
        correct(arrays, plan["answer"]),
        available(arrays),
    )
    rows = []
    for code, family in enumerate(families):
        for population in ("all", "ordinary", "old_exception"):
            requested = population_mask(plan, label_key, code, population)
            chosen = requested & valid
            n, nb, na = int(chosen.sum()), int((chosen & before).sum()), int((chosen & after).sum())
            row = {
                "arm": arm,
                "family": family,
                "population": population,
                "requested": int(requested.sum()),
                "available": n,
                "unavailable": int((requested & ~valid).sum()),
                "baseline_correct": nb,
                "correct": na,
                "accuracy": na / n if n else None,
                "baseline_accuracy": nb / n if n else None,
                "delta_accuracy": (na - nb) / n if n else None,
                "known_became_wrong": int((chosen & before & ~after).sum()),
                "wrong_became_correct": int((chosen & ~before & after).sum()),
                "known_denominator": nb,
                "known_retention": int((chosen & before & after).sum()) / nb if nb else None,
                "termination_error": int((chosen & ~arrays["ended"]).sum()),
                "prediction_changed": int(
                    (
                        chosen
                        & (
                            (arrays["prediction"] != baseline["prediction"])
                            | (arrays["ended"] != baseline["ended"])
                        )
                    ).sum()
                ),
            }
            for site in ("source", "downstream"):
                for kind in ("delta", "realized_delta"):
                    key = f"{site}_{kind}_norm"
                    row[f"mean_{key}"] = (
                        float(arrays[key][chosen, 0].mean()) if n and key in arrays else None
                    )
            rows.append(row)
    return rows


def donor_outcome_rows(arms, baseline, donor_baseline, plan, full_plan):
    rows = []
    scoring_plan = {**plan, "answer": plan["recipient_default"]}
    for arm, arrays in arms.items():
        generated_rows = condition_rows(
            arrays, baseline, scoring_plan, arm, families=TRANSFER_TYPES, label_key="pair_type"
        )
        for row in generated_rows:
            code = TRANSFER_TYPES.index(row["family"])
            requested = population_mask(full_plan, "pair_type", code, row["population"])
            chosen = population_mask(plan, "pair_type", code, row["population"]) & available(arrays)
            row["plan_requested"] = int(requested.sum())
            row["missing_donors"] = int((requested & (full_plan["donor"] < 0)).sum())
            for role in ("recipient_actual", "donor_default", "donor_actual"):
                row[f"matches_{role}"] = int((chosen & correct(arrays, plan[role])).sum())
                row[f"baseline_matches_{role}"] = int(
                    (chosen & correct(baseline, plan[role])).sum()
                )
            row["donor_answered_default"] = int(
                (chosen & correct(donor_baseline, plan["donor_default"])).sum()
            )
        rows.extend(generated_rows)
    return rows


def contrast_rows(left, right, baseline, plan, left_name, right_name):
    lc, rc, bc = (correct(value, plan["answer"]) for value in (left, right, baseline))
    lv, rv = available(left), available(right)
    rows = []
    for code, family in enumerate(FAMILIES):
        for population in ("all", "ordinary", "old_exception"):
            requested = population_mask(plan, "family", code, population)
            chosen = requested & lv & rv
            n, nl, nr = int(chosen.sum()), int((chosen & lc).sum()), int((chosen & rc).sum())
            known_right_wrong = chosen & bc & ~rc
            recovery_n = int(known_right_wrong.sum())
            rows.append(
                {
                    "left": left_name,
                    "right": right_name,
                    "family": family,
                    "population": population,
                    "requested": int(requested.sum()),
                    "common_available": n,
                    "left_unavailable": int((requested & ~lv).sum()),
                    "right_unavailable": int((requested & ~rv).sum()),
                    "left_correct": nl,
                    "right_correct": nr,
                    "accuracy_difference": (nl - nr) / n if n else None,
                    "right_wrong_left_correct": int((chosen & ~rc & lc).sum()),
                    "right_correct_left_wrong": int((chosen & rc & ~lc).sum()),
                    "original_known_right_wrong": recovery_n,
                    "original_known_right_wrong_left_recovered": int(
                        (known_right_wrong & lc).sum()
                    ),
                    "conditional_recovery_auxiliary": int((known_right_wrong & lc).sum())
                    / recovery_n
                    if recovery_n
                    else None,
                }
            )
    return rows


def selectivity_rows(target, control, plan, control_name):
    """Person-level difference of contrasts; roots are explicitly a separate cohort."""
    tc, cc = correct(target, plan["answer"]), correct(control, plan["answer"])
    valid = available(target) & available(control)
    delta = tc.astype(float) - cc
    defaults = np.flatnonzero(plan["family"] == 0)
    by_person = {
        person: index for person, index in zip(plan["person"][defaults], defaults, strict=True)
    }
    rows = []
    for code, family in enumerate(FAMILIES[1:], 1):
        for population in ("all", "ordinary", "old_exception"):
            requested = population_mask(plan, "family", code, population)
            control_ids = np.flatnonzero(requested)
            if family == "group_root":
                if population != "all":
                    continue
                di, ci = defaults[valid[defaults]], control_ids[valid[control_ids]]
                paired_people = False
            else:
                default_ids = np.asarray(
                    [by_person[p] for p in plan["person"][control_ids]], dtype=int
                )
                common = valid[default_ids] & valid[control_ids]
                di, ci, paired_people = default_ids[common], control_ids[common], True
            default_effect = float(delta[di].mean()) if len(di) else None
            control_effect = float(delta[ci].mean()) if len(ci) else None
            rows.append(
                {
                    "control_arm": control_name,
                    "control_family": family,
                    "population": population,
                    "paired_people": paired_people,
                    "requested_control_queries": int(requested.sum()),
                    "default_queries": len(di),
                    "control_queries": len(ci),
                    "default_target_minus_control": default_effect,
                    "other_target_minus_control": control_effect,
                    "default_minus_other_contrast": default_effect - control_effect
                    if default_effect is not None and control_effect is not None
                    else None,
                }
            )
    return rows


def check_csv(path, expected, keys):
    with path.open() as stream:
        saved = list(csv.DictReader(stream))
    by_key = {tuple(row[key] for key in keys): row for row in saved}
    if len(by_key) != len(saved) or len(saved) != len(expected):
        raise ValueError(f"Stored score rows missing or duplicated: {path}")
    for row in expected:
        observed = by_key[tuple(str(row[key]) for key in keys)]
        for key, value in row.items():
            if key not in observed:
                continue  # Independent summary adds fields beyond the producer table.
            if value is None:
                same = observed[key] == ""
            elif isinstance(value, (int, float, np.number)):
                same = np.isclose(float(observed[key]), value, rtol=1e-10, atol=1e-10)
            else:
                same = observed[key] == str(value)
            if not same:
                raise ValueError(f"Stored score disagrees with raw prediction: {path}/{key}")


def check_baseline(arrays, archived, ids):
    for key in ("prediction", "ended"):
        if not np.array_equal(arrays[key], archived[key][ids]):
            raise ValueError(f"Clean state differs from frozen parent {key}")


def audit_archive_difference(path, arrays, archived, ids):
    expected = {
        "query_id": ids,
        "local_prediction": arrays["prediction"],
        "local_ended": arrays["ended"],
        "archive_prediction": archived["prediction"][ids],
        "archive_ended": archived["ended"][ids],
        "prediction_changed": arrays["prediction"] != archived["prediction"][ids],
        "ended_changed": arrays["ended"] != archived["ended"][ids],
    }
    assert_arrays(read_npz(path), expected, "archive discrepancy evidence")
    changed = expected["prediction_changed"] | expected["ended_changed"]
    details = [
        {key: np.asarray(value[index]).item() for key, value in expected.items()}
        for index in np.flatnonzero(changed)
    ]
    coverage = {
        "queries": len(ids),
        "prediction_differences": int(expected["prediction_changed"].sum()),
        "ended_differences": int(expected["ended_changed"].sum()),
        "case_exclusions": 0,
    }
    return coverage, details


def check_norms(arms):
    pairs = [
        ("random_remove", "target_remove", "source"),
        ("complement_remove", "target_remove", "source"),
        ("remove_wrong_source_rescue", "remove_rescue", "downstream"),
        ("remove_random_rescue", "remove_rescue", "downstream"),
    ]
    for left, right, site in pairs:
        if left not in arms or right not in arms:
            continue
        common = available(arms[left]) & available(arms[right])
        key = f"{site}_delta_norm"
        if not np.allclose(arms[left][key][common], arms[right][key][common], rtol=1e-4, atol=1e-7):
            raise ValueError(f"Norm-matched control differs: {left}/{right}")


def donor_strata(arrays, plan):
    default = arrays["prediction"] == plan["donor_default"]
    actual = arrays["prediction"] == plan["donor_actual"]
    labels = np.full(len(default), "other", dtype="U16")
    labels[default & ~actual] = "default_only"
    labels[actual & ~default] = "actual_only"
    labels[actual & default] = "both"
    labels[~arrays["ended"]] = "unterminated"
    return labels


def transfer_contrasts(arms, baseline, donor_baseline, plan, full_plan):
    """Primary locked eligibility and broad nuisance controls both remain visible."""
    pairs = (
        ("full_donor", "full_norm_random"),
        ("full_donor", "sham"),
        ("projected_target", "projected_random"),
        ("projected_target", "projected_complement"),
        ("projected_target", "sham"),
        ("projected_target", "full_donor"),
    )
    strata = donor_strata(donor_baseline, plan)
    distinct = (plan["donor_default"] != plan["recipient_default"]) & (
        plan["donor_default"] != plan["recipient_actual"]
    )
    rows = []
    for left, right in pairs:
        if left not in arms or right not in arms:
            continue
        lv, rv = available(arms[left]), available(arms[right])
        lhit, rhit = (
            correct(arms[left], plan["donor_default"]),
            correct(arms[right], plan["donor_default"]),
        )
        for code, pair_type in enumerate(TRANSFER_TYPES):
            for population in ("all", "ordinary", "old_exception"):
                for scope in ("all_matched", "locked_distinct"):
                    if scope == "locked_distinct" and code != 0:
                        continue
                    for donor_class in (
                        "all",
                        "default_only",
                        "actual_only",
                        "both",
                        "other",
                        "unterminated",
                    ):
                        matched = population_mask(plan, "pair_type", code, population)
                        if scope == "locked_distinct":
                            matched &= distinct
                        if donor_class != "all":
                            matched &= strata == donor_class
                        chosen = matched & lv & rv
                        n = int(chosen.sum())
                        request = population_mask(full_plan, "pair_type", code, population)
                        # Missing donor semantics cannot have a distinct-answer/class assignment.
                        missing = int((request & (full_plan["donor"] < 0)).sum())
                        rows.append(
                            {
                                "left": left,
                                "right": right,
                                "pair_type": pair_type,
                                "population": population,
                                "scope": scope,
                                "donor_class": donor_class,
                                "matching_plan_requested": int(request.sum()),
                                "missing_donors": missing,
                                "eligible_matched": int(matched.sum()),
                                "common_available": n,
                                "left_donor_default_hits": int((chosen & lhit).sum()),
                                "right_donor_default_hits": int((chosen & rhit).sum()),
                                "donor_default_hit_difference": float(
                                    (lhit[chosen].astype(float) - rhit[chosen]).mean()
                                )
                                if n
                                else None,
                                "baseline_recipient_correct": int(
                                    (chosen & correct(baseline, plan["recipient_default"])).sum()
                                ),
                            }
                        )
    return rows


def capture_rows(arms, plan):
    """Compare two causal contrasts on exactly the same queries; avoid unstable ratios."""
    required = ("full_donor", "full_norm_random", "projected_target", "projected_random")
    if not all(arm in arms for arm in required):
        return []
    valid = np.logical_and.reduce([available(arms[arm]) for arm in required])
    hits = {arm: correct(arms[arm], plan["donor_default"]) for arm in required}
    full = hits["full_donor"].astype(float) - hits["full_norm_random"]
    projected = hits["projected_target"].astype(float) - hits["projected_random"]
    distinct = (plan["donor_default"] != plan["recipient_default"]) & (
        plan["donor_default"] != plan["recipient_actual"]
    )
    rows = []
    for population in ("all", "ordinary", "old_exception"):
        requested = population_mask(plan, "pair_type", 0, population) & distinct
        chosen = requested & valid
        n = int(chosen.sum())
        denominator = arms["full_donor"]["source_delta_norm"][:, 0]
        nonzero = chosen & (denominator > 1e-12)
        fraction = (
            arms["projected_target"]["source_delta_norm"][:, 0][nonzero] / denominator[nonzero]
        )
        rows.append(
            {
                "population": population,
                "eligible_matched": int(requested.sum()),
                "common_available": n,
                "full_minus_full_random": float(full[chosen].mean()) if n else None,
                "projected_minus_projected_random": float(projected[chosen].mean()) if n else None,
                "projected_contrast_minus_full_contrast": float(
                    (projected[chosen] - full[chosen]).mean()
                )
                if n
                else None,
                "nonzero_full_delta_queries": int(nonzero.sum()),
                "mean_projected_to_full_delta_norm": float(fraction.mean())
                if len(fraction)
                else None,
            }
        )
    return rows


def audit_calibration(directory, world, chain, contract, archive, reader):
    pair = edit_pair(world, chain)
    plan = calibration_plan(
        world.actual_ids[chain],
        world.person,
        world.membership_ids[chain],
        pair["E"],
        pair["replay"],
    )
    assert_arrays(
        read_npz(directory / "calibration-plan.npz"),
        {key: np.asarray(value) for key, value in plan.items()},
        "calibration",
    )
    actual = reader("calibration-actual.npz", plan["actual_query_ids"])
    membership = reader("calibration-membership.npz", plan["membership_query_ids"])
    allowed = np.array(
        [i for i, label in enumerate(world.token_labels) if label.startswith(f"{CHAINS[chain]}:")]
    )
    valid = membership["ended"] & np.isin(membership["prediction"], allowed)
    predicted = np.where(valid, membership["prediction"], -1)
    assert_arrays(
        read_npz(directory / "self-predicted-groups.npz"),
        {"prediction": predicted, "valid": valid},
        "self predicted groups",
    )
    info = json.loads((directory / "calibration.json").read_text())
    if (
        info["derived_calibration_queries"] != 0
        or info["true_memberships_supplied"] is not False
        or info["actual_queries"] != len(predicted)
        or info["additional_membership_queries"] != len(predicted)
        or info["basis_sha256"] != file_sha256(directory / "bases.npz")
    ):
        raise ValueError("Calibration information boundary or basis seal changed")
    bases = read_npz(directory / "bases.npz")
    row = {
        "actual_queries": len(predicted),
        "membership_queries": len(predicted),
        "valid_group_predictions": int(valid.sum()),
        "group_prediction_valid_rate": float(valid.mean()),
        "membership_correct": int(
            correct(membership, world.answers[plan["membership_query_ids"]]).sum()
        ),
        "membership_accuracy": float(
            correct(membership, world.answers[plan["membership_query_ids"]]).mean()
        ),
        "actual_accuracy": float(correct(actual, world.answers[plan["actual_query_ids"]]).mean()),
        "additional_membership_forward_examples": int(membership["forward_examples"]),
    }
    recorded = json.loads((directory / "calibration-diagnostics.json").read_text())
    if (
        recorded["queries"] != len(predicted)
        or recorded["correct_predictions"] != row["membership_correct"]
        or recorded["valid_predictions"] != row["valid_group_predictions"]
    ):
        raise ValueError("Offline membership score differs from raw predictions")
    for site in ("source", "downstream"):
        report = info["basis_reports"][site]
        row[f"{site}_available"] = report["available"]
        row[f"{site}_reason"] = report["reason"]
        row[f"{site}_target_numerical_rank"] = report.get("target_numerical_rank")
        row[f"{site}_complement_numerical_rank"] = report.get("complement_numerical_rank")
        if report["rank_requested"] != 32 or report["used_rows"] != int(valid.sum()):
            raise ValueError("Calibration rank or row selection changed")
        if not report["available"]:
            if any(key.startswith(site + "_") for key in bases):
                raise ValueError("Unavailable basis has unexpected stored directions")
            continue
        for kind in ("target", "random", "complement"):
            q, mean, scale = (bases[f"{site}_{kind}_{field}"] for field in ("q", "mean", "scale"))
            if q.shape != (contract["width"], 32) or mean.shape != (32,) or scale.shape != (32,):
                raise ValueError("Fixed-rank basis shape changed")
            if (
                not np.isfinite(q).all()
                or not np.isfinite(mean).all()
                or not np.isfinite(scale).all()
                or (scale <= 0).any()
            ):
                raise ValueError("Invalid basis values")
            if not np.allclose(q.T @ q, np.eye(32), atol=1e-5):
                raise ValueError("Basis is not orthonormal")
        if not np.allclose(
            bases[f"{site}_target_q"].T @ bases[f"{site}_complement_q"], 0, atol=1e-5
        ):
            raise ValueError("Within-group complement overlaps target")
    return row, info


def audit_model(directory, contract, world, archive):
    tables = defaultdict(list)
    for chain, name in enumerate(CHAINS):
        dest = directory / name
        identity = {key: contract[key] for key in ("width", "world", "seed", "condition")}
        identity.update(
            chain=name,
            organization_role="neither"
            if contract["condition"] == "neither"
            else "aligned"
            if contract["condition"] == name
            else "opposite",
        )

        numerical_v2 = (
            contract.get("numerical_policy", {}).get("version") == "p4-numerical-policy-v2"
        )

        def reader(filename, ids, dest=dest, identity=identity, numerical_v2=numerical_v2):
            arrays = read_predictions(
                dest / filename,
                ids,
                world.vocab_size,
                contract["width"],
                contract["batch_size"],
                world.lengths[ids],
            )
            evidence = {
                "calibration-actual.npz": "archive-calibration-actual.npz",
                "calibration-membership.npz": "archive-calibration-membership.npz",
                "clean.npz": "archive-probe-clean.npz",
                "transfer-clean.npz": "archive-transfer-cache.npz",
                "transfer-local-clean.npz": "archive-transfer-local-clean.npz",
            }
            if filename in evidence:
                if numerical_v2:
                    coverage, details = audit_archive_difference(
                        dest / evidence[filename], arrays, archive, ids
                    )
                    tables["archive-difference-coverage"].append(
                        {**identity, "component": filename, **coverage}
                    )
                    tables["archive-discrepant-queries"].extend(
                        {**identity, "component": filename, **row} for row in details
                    )
                else:
                    check_baseline(arrays, archive, ids)
                for key, value in arrays.items():
                    if key.endswith("_pass_drift"):
                        tables["prefix-drift"].append(
                            {
                                **identity,
                                "component": filename,
                                "site": key,
                                "queries": len(ids),
                                "nonzero": int(np.count_nonzero(value)),
                                "maximum": float(value.max()),
                                "mean": float(value.mean()),
                            }
                        )
            return arrays

        calibration, info = audit_calibration(dest, world, chain, contract, archive, reader)
        tables["calibration"].append({**identity, **calibration})
        transfer_plan = make_transfer_plan(world, chain, 64)
        assert_arrays(read_npz(dest / "transfer-plan.npz"), transfer_plan, "donor plan")
        recipients = transfer_plan["recipient"][transfer_plan["pair_type"] == 0]
        plan = make_probe_plan(world, chain, recipients)
        plan["wrong_source_index"] = wrong_source_indices(plan["family"])
        plan["wrong_source_query_id"] = plan["query_id"][plan["wrong_source_index"]]
        plan["wrong_source_same_answer"] = (
            plan["answer"] == plan["answer"][plan["wrong_source_index"]]
        )
        assert_arrays(read_npz(dest / "probe-plan.npz"), plan, "probe plan")
        if len(plan["query_id"]) != 1216:
            raise ValueError("The ten-family query budget changed")
        status = json.loads((dest / "status.json").read_text())
        arms = {}
        for arm in ARMS:
            expected = "complete"
            if arm not in ("clean", "sham") and not calibration["source_available"]:
                expected = "unavailable_source_rank"
            elif "rescue" in arm and not calibration["downstream_available"]:
                expected = "unavailable_downstream_rank"
            if status["arms"][arm] != expected:
                raise ValueError("Arm availability differs from fixed-rank contract")
            if expected == "complete":
                arms[arm] = reader(f"{arm}.npz", plan["query_id"])
            tables["coverage"].append(
                {**identity, "stage": "necessity", "arm": arm, "status": expected}
            )
        clean = arms["clean"]
        for arm in ("sham", "rescue_only"):
            if arm in arms and any(
                not np.array_equal(arms[arm][key], clean[key]) for key in ("prediction", "ended")
            ):
                raise ValueError("A no-lesion sham changed outputs")
        check_norms(arms)
        rows, paired = [], []
        for arm, arrays in arms.items():
            rows.extend(condition_rows(arrays, clean, plan, arm))
        for left, right in COMPARISONS:
            if left in arms and right in arms:
                paired.extend(contrast_rows(arms[left], arms[right], clean, plan, left, right))
        check_csv(dest / "necessity.csv", rows, ("arm", "family", "population"))
        check_csv(dest / "paired-necessity.csv", paired, ("left", "right", "family", "population"))
        tables["necessity"].extend({**identity, **row} for row in rows)
        tables["paired-necessity"].extend({**identity, **row} for row in paired)
        for control in ("random_remove", "complement_remove"):
            if "target_remove" in arms and control in arms:
                tables["selectivity"].extend(
                    {**identity, **row}
                    for row in selectivity_rows(arms["target_remove"], arms[control], plan, control)
                )
        valid = transfer_plan["donor"] >= 0
        selected = {key: value[valid] for key, value in transfer_plan.items()}
        people = np.unique(np.r_[selected["recipient"], selected["donor"]])
        donor_ids = world.derived_ids[chain, people]
        donor_clean = reader("transfer-clean.npz", donor_ids)
        ri, di = (
            np.searchsorted(people, selected["recipient"]),
            np.searchsorted(people, selected["donor"]),
        )
        if numerical_v2:
            local_ids = world.derived_ids[chain, selected["recipient"]]
            local_clean = reader("transfer-local-clean.npz", local_ids)
            baseline = {key: local_clean[key] for key in ("prediction", "ended")}
            drift = {
                "query_id": local_ids,
                "cache_vs_local_state_norm": np.linalg.norm(
                    donor_clean["state_0_2"][ri] - local_clean["state_0_2"], axis=1
                ),
                "cache_prediction": donor_clean["prediction"][ri],
                "cache_ended": donor_clean["ended"][ri],
                "local_prediction": local_clean["prediction"],
                "local_ended": local_clean["ended"],
            }
            assert_arrays(
                read_npz(dest / "transfer-reference-drift.npz"),
                drift,
                "transfer numerical reference drift",
            )
            recorded = json.loads((dest / "archive-discrepancies.json").read_text())
            for component, metadata in recorded.items():
                data = read_npz(dest / f"{component}.npz")
                if (
                    metadata["queries"] != len(data["query_id"])
                    or metadata["case_exclusion"] is not False
                    or metadata["prediction_differences"] != int(data["prediction_changed"].sum())
                    or metadata["ended_differences"] != int(data["ended_changed"].sum())
                ):
                    raise ValueError(
                        "Archive numerical discrepancy summary differs from raw evidence"
                    )
        else:
            baseline = {key: donor_clean[key][ri] for key in ("prediction", "ended")}
        donor_baseline = {key: donor_clean[key][di] for key in ("prediction", "ended")}
        donor_arms = {}
        for arm in TRANSFER_ARMS:
            expected = (
                "unavailable_source_rank"
                if arm.startswith("projected_") and not calibration["source_available"]
                else "complete"
            )
            if status["transfer_arms"][arm] != expected:
                raise ValueError("Donor availability differs from basis availability")
            if expected == "complete":
                donor_arms[arm] = reader(
                    f"transfer-{arm}.npz", world.derived_ids[chain, selected["recipient"]]
                )
            tables["coverage"].append(
                {**identity, "stage": "donor", "arm": arm, "status": expected}
            )
        for key in ("prediction", "ended"):
            if not np.array_equal(donor_arms["sham"][key], baseline[key]):
                raise ValueError("Donor sham changed output")
        for left, right in (
            ("full_norm_random", "full_donor"),
            ("projected_random", "projected_target"),
            ("projected_complement", "projected_target"),
        ):
            if left in donor_arms and right in donor_arms:
                common = available(donor_arms[left]) & available(donor_arms[right])
                if not np.allclose(
                    donor_arms[left]["source_delta_norm"][common],
                    donor_arms[right]["source_delta_norm"][common],
                    rtol=1e-4,
                    atol=1e-7,
                ):
                    raise ValueError("Donor norm-matched control differs")
        donor_outcomes = donor_outcome_rows(
            donor_arms, baseline, donor_baseline, selected, transfer_plan
        )
        check_csv(dest / "transfer.csv", donor_outcomes, ("arm", "family", "population"))
        tables["donor-outcomes"].extend({**identity, **row} for row in donor_outcomes)
        tables["donor-contrasts"].extend(
            {**identity, **row}
            for row in transfer_contrasts(
                donor_arms, baseline, donor_baseline, selected, transfer_plan
            )
        )
        tables["projection-capture"].extend(
            {**identity, **row} for row in capture_rows(donor_arms, selected)
        )
        for code, pair_type in enumerate(TRANSFER_TYPES):
            for population in ("all", "ordinary", "old_exception"):
                chosen = population_mask(transfer_plan, "pair_type", code, population)
                tables["donor-coverage"].append(
                    {
                        **identity,
                        "pair_type": pair_type,
                        "population": population,
                        "requested": int(chosen.sum()),
                        "matched": int((chosen & valid).sum()),
                        "missing": int((chosen & ~valid).sum()),
                    }
                )
        for stage, costs in (
            ("calibration", info["calibration_cost"]),
            ("evaluation", status["evaluation_cost"]),
        ):
            for label, cost in costs.items():
                tables["costs"].append({**identity, "stage": stage, "component": label, **cost})
    return tables


def write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows, keys, metric):
    groups = defaultdict(list)
    for row in rows:
        if row.get(metric) is not None:
            groups[tuple(row[key] for key in keys)].append(row[metric])
    return [
        {
            **dict(zip(keys, key, strict=True)),
            "model_chain_blocks": len(values),
            "mean": float(np.mean(values)),
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
        }
        for key, values in sorted(groups.items())
    ]


def report(tables, audit):
    lines = [
        f"P4 necessity/restoration: {audit['status']} ({audit['complete_models']}/24 models).",
        "",
        "All reported contrasts are recomputed from sealed value-plus-EOS predictions. "
        "Worlds 0/1 and both model seeds remain separate in the CSV tables. "
        "Group-root facts have 64 independent queries; each other family has 128 people.",
        "",
        "Negative target-minus-random/complement accuracy means a larger target lesion. "
        "Positive same-query-rescue minus wrong-source/random-rescue accuracy "
        "means stronger recovery. "
        "Recovery is after an artificial lesion, not repair of a learned edit.",
        "",
    ]
    calibration = tables["calibration"]
    source_n = sum(row["source_available"] for row in calibration)
    downstream_n = sum(row["downstream_available"] for row in calibration)
    lines.append(
        f"Basis availability: source {source_n}/{len(calibration)}, "
        f"downstream {downstream_n}/{len(calibration)} model-chain blocks."
    )
    lines.extend(
        [
            "",
            "Mean membership self-prediction and actual-query accuracy (all calibration rows):",
            "",
            "| Width | Organization | Chain | Blocks | Membership | Actual |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    grouped = defaultdict(list)
    for row in calibration:
        grouped[(row["width"], row["condition"], row["chain"])].append(row)
    for key, values in sorted(grouped.items()):
        lines.append(
            f"| {key[0]} | {key[1]} | {key[2]} | {len(values)}/4 | "
            f"{np.mean([row['membership_accuracy'] for row in values]):.2%} | "
            f"{np.mean([row['actual_accuracy'] for row in values]):.2%} |"
        )
    lines.extend(
        [
            "",
            "Paired mean accuracy contrasts over all eligible queries; pp = percentage points.",
            "",
            "| Width | Organization | Chain | Query family | Contrast | Blocks | Effect (pp) |",
            "|---|---|---|---|---|---:|---:|",
        ]
    )
    chosen = [
        row
        for row in tables["paired-necessity"]
        if row["population"] == "all"
        and (row["left"], row["right"])
        in (
            ("target_remove", "random_remove"),
            ("target_remove", "complement_remove"),
            ("remove_rescue", "remove_wrong_source_rescue"),
            ("remove_rescue", "remove_random_rescue"),
        )
    ]
    for row in aggregate(
        chosen, ("width", "condition", "chain", "family", "left", "right"), "accuracy_difference"
    ):
        lines.append(
            f"| {row['width']} | {row['condition']} | {row['chain']} | {row['family']} | "
            f"{row['left']} − {row['right']} | {row['model_chain_blocks']}/4 | "
            f"{row['mean'] * 100:+.3f} |"
        )
    lines.extend(
        [
            "",
            "Projection capture uses the original distinct-donor-answer eligibility "
            "and a common mask for all four interventions. Full and projected effects, "
            "their difference, and perturbation-norm "
            "fractions are retained; no unstable behavioral capture ratio is used.",
            "",
            "Interpretation limits: actual-query Q may fail to capture the derived-query "
            "localization signal; that null cannot by itself disprove the localized node. "
            "Generic identity, relation-offset or answer signals remain alternatives unless "
            "the ten-family and matched-donor controls distinguish them. Rank availability "
            "and initial knowledge are outcomes of training, not causal matching criteria. "
            "Known-only recovery and donor response strata are auxiliary diagnostics.",
            "",
        ]
    )
    return "\n".join(lines)


def summarize(manifest, output, require_complete=False):
    manifest, output = Path(manifest).resolve(), Path(output).resolve()
    queue = json.loads(manifest.read_text())
    expected_v2 = "src/llm_memory_editability/bios_mechanism_causal_v2.py" in queue["sources"]
    expected = {
        f"width-{width}/world-{world}-seed-{seed}-{condition}"
        for width, world, seed, condition in itertools.product(
            (256, 768), (0, 1), (0, 1), CONDITIONS
        )
    }
    if {job["key"] for job in queue["jobs"]} != expected or len(queue["jobs"]) != 24:
        raise ValueError("Queue must contain the complete predeclared 24-model matrix")
    repository = Path(__file__).resolve().parents[1]
    for relative, digest in queue["sources"].items():
        if file_sha256(repository / relative) != digest:
            raise ValueError("Frozen producer changed")
    tables, ledger, missing, worlds = defaultdict(list), [], [], {}
    for job in queue["jobs"]:
        directory = Path(job["output"])
        if not (directory / "complete.json").exists():
            missing.append(job["key"])
            continue
        complete = json.loads((directory / "complete.json").read_text())
        contract = json.loads((directory / "contract.json").read_text())
        numerical_v2 = (
            contract.get("numerical_policy", {}).get("version") == "p4-numerical-policy-v2"
        )
        if numerical_v2 != expected_v2:
            raise ValueError("Do not mix v1 and v2 numerical policies in one analysis matrix")
        current_sources = producer_v2._sources() if numerical_v2 else _sources()
        expected_seal = (
            producer_v2.scientific_outputs(directory)
            if numerical_v2
            else scientific_outputs(directory)
        )
        if complete["status"] != "complete" or complete["outputs"] != expected_seal:
            raise ValueError(f"Producer scientific seal failed: {job['key']}")
        if numerical_v2 and contract["numerical_policy"] != producer_v2.NUMERICAL_POLICY:
            raise ValueError("V2 numerical policy changed")
        validate_lock(contract["lock"])
        identity_key = (
            f"width-{contract['width']}/world-{contract['world']}-"
            f"seed-{contract['seed']}-{contract['condition']}"
        )
        if (
            identity_key != job["key"]
            or contract["sources"] != current_sources
            or contract["lock_sha256"] != queue["candidate_lock_sha256"]
        ):
            raise ValueError("Model identity, producer source, or candidate lock changed")
        parent = Path(contract["run"])
        if file_sha256(parent / "config.json") != contract["parent_config_sha256"]:
            raise ValueError("Frozen parent configuration changed")
        config = json.loads((parent / "config.json").read_text())
        world_id = contract["world"]
        if world_id not in worlds:
            worlds[world_id] = make_cross_world(world_id, repository / "data/bios-organization-v1")
        world = worlds[world_id]
        if (
            array_hash(world.answers) != config["truth_sha256"]
            or array_hash(world.prompts) != config["prompts_sha256"]
        ):
            raise ValueError("Frozen truth or prompts changed")
        with np.load(parent / f"predictions-{contract['step']}.npz") as stored:
            archived = {key: stored[key] for key in ("prediction", "ended")}
        if numerical_v2:
            replay = read_predictions(
                directory / "original-replay.npz",
                np.arange(len(world.answers)),
                world.vocab_size,
                contract["width"],
                512,
                world.lengths,
            )
            check_baseline(replay, archived, np.arange(len(world.answers)))
            replay_report = json.loads((directory / "original-replay.json").read_text())
            if (
                replay_report["queries"] != 20608
                or replay_report["batch_size"] != 512
                or replay_report["prediction_differences"] != 0
                or replay_report["ended_differences"] != 0
                or replay_report["raw_sha256"] != file_sha256(directory / "original-replay.npz")
            ):
                raise ValueError("Original full-order replay identity gate changed")
            tables["costs"].append(
                {
                    **{key: contract[key] for key in IDENTITY if key != "chain"},
                    "chain": "both",
                    "stage": "identity",
                    "component": "full_original_replay",
                    **replay_report["extra_identity_gate_cost"],
                }
            )
        model_tables = audit_model(directory, contract, world, archived)
        for label, rows in model_tables.items():
            tables[label].extend(rows)
        ledger.append(
            {
                "key": job["key"],
                "complete_sha256": file_sha256(directory / "complete.json"),
                "scientific_artifacts": len(complete["outputs"]),
                "status": "seal_and_query_recompute_passed",
            }
        )
        print(json.dumps({"event": "p4_model_audited", "key": job["key"]}), flush=True)
    output.mkdir(parents=True, exist_ok=True)
    for label, rows in tables.items():
        write_csv(output / f"{label}.csv", rows)
    audit = {
        "status": "complete" if not missing else "partial",
        "expected_models": 24,
        "complete_models": len(ledger),
        "model_chain_blocks": len(tables["calibration"]),
        "missing_models": missing,
        "manifest": str(manifest),
        "manifest_sha256": file_sha256(manifest),
        "summary_source_sha256": file_sha256(__file__),
        "models": ledger,
        "parent_weights_rehashed": False,
        "producer_seals_verified": True,
        "raw_value_and_EOS_recomputed": True,
        "Q_refitted_or_selected": False,
        "numerical_policy": "v2" if expected_v2 else "v1",
    }
    write_json(output / "audit.json", audit)
    (output / "report.md").write_text(report(tables, audit))
    if require_complete and missing:
        raise RuntimeError(f"P4 incomplete: {len(ledger)}/24 models; partial audit saved")
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    summarize(args.manifest, args.output, args.require_complete)


if __name__ == "__main__":
    main()
