"""P4 v2: unchanged interventions with explicit local numerical baselines.

Q is fitted only on actual queries in an archived exception case's S union R,
grouped by the model's own membership predictions. Truth-matched donor queries
are a subsequent diagnostic, never calibration. Restoring a same-query clean
downstream signal tests recovery from an artificial lesion, not edit repair.
"""

import csv
import json
from pathlib import Path

import numpy as np
import torch

from .bios_causal_paths import make_donor_plan
from .bios_cross import CHAINS, edit_pair, make_cross_world
from .bios_data import EOS, array_hash, rng_for, write_json
from .bios_mechanism_interventions import (
    Site,
    attach_state_operation,
    calibration_plan,
    downstream_routing_site,
    fit_predicted_group_bases,
    match_delta_norm,
    projection_component,
    query_site_roles,
    remove_projection,
    restore_projection,
    valid_predicted_groups,
)
from .bios_model import CausalLM, ModelConfig
from .bios_organization_train import atomic_numpy_save, state_hash
from .bios_path_diagnostics import file_sha256
from .bios_train import precision

SOURCE = Site(0, 2)
DOWNSTREAM = Site(1, 2)
RANK = 32
NUMERICAL_POLICY = {
    "version": "p4-numerical-policy-v2",
    "original_identity_gate": "20608 queries; original order/batch512; exact archive values/EOS",
    "subset_baseline": "Same query/order/batch local clean for every causal contrast",
    "archive_subset_difference": "All queries and both outputs retained; no exclusions",
    "donor_baseline": "Extra clean replay in duplicated recipient order and intervention batch",
    "local_sham_and_rescue_only": "Strict exact value/EOS equality; never relaxed",
    "restore_reference": "Unchanged v1 first-pass pre-answer source; pass drift remains recorded",
    "scientific_design": "Candidate, rank, calibration, intervention, pairing, strata unchanged",
    "precision": "Unchanged BF16 autocast; FP32 weights; TF32 allowed as in original training",
}
FAMILIES = (
    "default",
    "actual",
    "other_chain_default",
    "attribute_3",
    "attribute_4",
    "attribute_5",
    "attribute_6",
    "membership",
    "other_chain_actual",
    "group_root",
)
ARMS = (
    "clean",
    "sham",
    "target_remove",
    "random_remove",
    "complement_remove",
    "rescue_only",
    "remove_rescue",
    "remove_wrong_source_rescue",
    "remove_random_rescue",
)
TRANSFER_TYPES = (
    "different_group_same_actual_different_default",
    "different_group_same_default",
    "different_group_same_default_same_actual",
)
TRANSFER_ARMS = (
    "sham",
    "full_donor",
    "full_norm_random",
    "projected_target",
    "projected_random",
    "projected_complement",
)


def _freeze_json(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"Frozen diagnostic contract changed: {path}")
    else:
        write_json(path, value)


def _same_arrays(path, arrays):
    if path.exists():
        with np.load(path) as existing:
            if set(existing.files) != set(arrays) or any(
                not np.array_equal(existing[key], value) for key, value in arrays.items()
            ):
                raise ValueError(f"Diagnostic plan changed: {path}")
    else:
        atomic_numpy_save(path, **arrays)


def _sources():
    directory = Path(__file__).parent
    names = (
        "bios_mechanism_causal_v2.py",
        "bios_mechanism_causal.py",
        "bios_mechanism_interventions.py",
        "bios_causal_paths.py",
        "bios_cross.py",
        "bios_data.py",
        "bios_model.py",
        "bios_train.py",
        "bios_organization_train.py",
        "bios_path_diagnostics.py",
    )
    sources = {name: file_sha256(directory / name) for name in names}
    runner = directory.parents[1] / "scripts/analyze_bios_mechanism_causal_v2.py"
    sources["runner"] = file_sha256(runner)
    return sources


def validate_lock(lock):
    if (lock.get("mechanism"), lock.get("layer"), lock.get("position")) != ("default_path", 0, 2):
        raise ValueError("This runner only instantiates the locked default_path L0/pos2 candidate")
    if lock.get("known_only") is not False or not lock.get("locked_at_utc"):
        raise ValueError("An all-eligible, timestamped candidate lock is required")


def make_probe_plan(world, chain, recipients):
    """Same people and fixed queries in every intervention, without knowledge filtering."""
    recipients = np.asarray(recipients, dtype=np.int64)
    ids = [
        world.derived_ids[chain, recipients],
        world.actual_ids[chain, recipients],
        world.derived_ids[1 - chain, recipients],
    ]
    for relation in (3, 4, 5, 6):
        candidates = np.flatnonzero(world.relation == relation)
        by_person = np.full(len(world.actual_ids[chain]), -1, dtype=np.int64)
        by_person[world.person[candidates]] = candidates
        if (by_person[recipients] < 0).any():
            raise ValueError("An independent control query is missing")
        ids.append(by_person[recipients])
    ids.extend([world.membership_ids[chain, recipients], world.actual_ids[1 - chain, recipients]])
    person_families = len(ids)
    roots = world.root_ids[chain]
    query_ids = np.concatenate([*ids, roots])
    return {
        "query_id": query_ids,
        "person": np.r_[np.tile(recipients, person_families), np.full(len(roots), -1)],
        "family": np.r_[
            np.repeat(np.arange(person_families), len(recipients)),
            np.full(len(roots), person_families),
        ],
        # -1 means not applicable; root facts are not ordinary people.
        "recipient_old_exception": np.r_[
            np.tile(world.exceptions[chain, recipients].astype(np.int8), person_families),
            np.full(len(roots), -1, dtype=np.int8),
        ],
        "answer": world.answers[query_ids],
    }


def wrong_source_indices(families):
    """Fixed one-person cyclic shift within each family, never chosen by answers."""
    families = np.asarray(families)
    index = np.arange(len(families))
    for family in np.unique(families):
        selected = np.flatnonzero(families == family)
        if len(selected) < 2:
            raise ValueError("Wrong-source controls require at least two queries per family")
        index[selected] = np.roll(selected, 1)
    return index


def make_transfer_plan(world, chain, per_population=64):
    """Truth matching is restricted to this explicitly labeled offline diagnostic."""
    old = make_donor_plan(world, chain, per_population)
    original = old["pair_type"] == 1
    recipients = old["recipient"][original]
    train = world.person[world.train_ids[chain]]
    group = world.memberships[chain]
    default = world.answers[world.root_ids[chain, group]]
    actual = world.answers[world.actual_ids[chain]]
    rng = rng_for(world.seed, 978, chain)
    donor_rows = [old["donor"][original]]
    count_rows = [old["candidate_count"][original]]
    for same_actual in (False, True):
        donors, counts = [], []
        for recipient in recipients:
            mask = (group[train] != group[recipient]) & (default[train] == default[recipient])
            if same_actual:
                mask &= actual[train] == actual[recipient]
            candidates = train[mask]
            donors.append(rng.choice(candidates) if len(candidates) else -1)
            counts.append(len(candidates))
        donor_rows.append(np.asarray(donors))
        count_rows.append(np.asarray(counts))
    donor = np.concatenate(donor_rows).astype(np.int64)
    recipient = np.tile(recipients, len(TRANSFER_TYPES))
    safe = np.maximum(donor, 0)
    return {
        "case_id": np.arange(len(donor)),
        "recipient": recipient,
        "donor": donor,
        "pair_type": np.repeat(np.arange(len(TRANSFER_TYPES)), len(recipients)),
        "candidate_count": np.concatenate(count_rows),
        "recipient_old_exception": world.exceptions[chain, recipient],
        "same_actual": (actual[safe] == actual[recipient]) & (donor >= 0),
        "same_default": (default[safe] == default[recipient]) & (donor >= 0),
        "recipient_default": default[recipient],
        "recipient_actual": actual[recipient],
        "donor_default": np.where(donor >= 0, default[safe], -1),
        "donor_actual": np.where(donor >= 0, actual[safe], -1),
    }


def generate_queries(
    model, prompts, lengths, device, batch_size=256, *, operations=None, capture_sites=()
):
    """Two free generation passes with identical pre-answer operations, no answer input.

    operations(begin,end) returns (label, site, operation) triples. Every operation
    returns its intended delta and availability; realized post-cast norms are
    measured separately. Both passes are retained so norm matching is auditable.
    """
    if len(prompts) == 0 or batch_size < 1:
        raise ValueError("Generation needs nonempty queries and a positive batch size")
    lengths = np.asarray(lengths)
    if prompts.shape != (len(lengths), 5) or not np.isin(lengths, [4, 5]).all():
        raise ValueError("Unexpected symbolic query shape")
    arrays = {"prediction": [], "ended": []}
    for site in capture_sites:
        query_site_roles(lengths, site.position)
        arrays[f"state_{site.layer}_{site.position}"] = []
    calls = 0
    model.eval()
    with torch.no_grad():
        for begin in range(0, len(prompts), batch_size):
            end = min(begin + batch_size, len(prompts))
            p = torch.as_tensor(prompts[begin:end], device=device)
            n = torch.as_tensor(lengths[begin:end], device=device)
            hooks, traces, captures = [], {}, {}
            try:
                for label, site, operation in () if operations is None else operations(begin, end):
                    query_site_roles(lengths[begin:end], site.position)

                    def measured(value, operation=operation):
                        changed, trace = operation(value)
                        return changed, {**trace, "realized_delta": changed.float() - value.float()}

                    hook, trace = attach_state_operation(model, site, measured)
                    hooks.append(hook)
                    traces[label] = trace
                for site in capture_sites:
                    key = f"state_{site.layer}_{site.position}"
                    captures[key] = []

                    def capture(
                        _module, _inputs, output, key=key, position=site.position, captures=captures
                    ):
                        captures[key].append(output[:, position].detach().float())

                    hooks.append(model.blocks[site.layer].register_forward_hook(capture))
                with precision(device):
                    first = model(p, (n - 1)[:, None])[:, 0].float().argmax(-1)
                    continuation = torch.zeros((len(p), 6), dtype=torch.long, device=device)
                    continuation[:, :5] = p
                    continuation[torch.arange(len(p), device=device), n] = first
                    second = model(continuation, n[:, None])[:, 0].argmax(-1)
                calls += 2
                arrays["prediction"].append(first.cpu().numpy())
                arrays["ended"].append(second.eq(EOS).cpu().numpy())
                for key, states in captures.items():
                    arrays[key].append(states[0].cpu().numpy())
                    drift = torch.linalg.vector_norm(states[1] - states[0], dim=-1)
                    arrays.setdefault(f"{key}_pass_drift", []).append(drift.cpu().numpy())
                for label, trace in traces.items():
                    if len(trace) != 2:
                        raise ValueError(
                            "Every intervention must occur in both free-generation passes"
                        )
                    for kind in ("delta", "realized_delta"):
                        norms = torch.stack(
                            [torch.linalg.vector_norm(t[kind].float(), dim=-1) for t in trace],
                            dim=1,
                        )
                        arrays.setdefault(f"{label}_{kind}_norm", []).append(norms.cpu().numpy())
                    valid = torch.stack([t["valid"] for t in trace], dim=1)
                    arrays.setdefault(f"{label}_valid", []).append(valid.cpu().numpy())
            finally:
                for hook in hooks:
                    hook.remove()
    result = {key: np.concatenate(value) for key, value in arrays.items()}
    result.update(
        forward_calls=np.asarray(calls),
        forward_examples=np.asarray(2 * len(prompts)),
        padded_token_positions=np.asarray(11 * len(prompts)),
        logical_token_positions=np.asarray(int((2 * lengths + 1).sum())),
    )
    return result


def _identity_trace(value):
    return value, {
        "delta": torch.zeros_like(value, dtype=torch.float32),
        "valid": torch.ones(len(value), dtype=torch.bool, device=value.device),
    }


def lesion_operations(
    arm, source_bases, downstream_bases, clean_downstream, wrong_index, seed, device
):
    """Return a deterministic batch factory. Wrong-source choice never sees truth."""
    if arm not in ARMS:
        raise ValueError("Unknown necessity/restoration arm")
    if arm == "clean":
        return lambda _begin, _end: []
    if arm == "sham":
        return lambda _begin, _end: [("source", SOURCE, _identity_trace)]
    source = {key: basis.tensors(device) for key, basis in source_bases.items()}
    downstream = (
        None
        if downstream_bases is None
        else {key: basis.tensors(device) for key, basis in downstream_bases.items()}
    )
    needs_downstream = "rescue" in arm
    if needs_downstream and downstream is None:
        raise ValueError("Downstream basis is unavailable; recovery cannot be estimated")
    wrong_index = np.asarray(wrong_index)
    if len(wrong_index) != len(clean_downstream):
        raise ValueError("Wrong-source plan length mismatch")
    coefficients = (
        np.random.default_rng(seed)
        .standard_normal((len(clean_downstream), source_bases["random"].rank))
        .astype(np.float32)
    )

    def factory(begin, end):
        ops = []
        if arm == "clean":
            return ops
        if arm == "sham":
            return [("source", SOURCE, _identity_trace)]
        if arm != "rescue_only":
            kind = (
                arm.removesuffix("_remove")
                if arm in ("random_remove", "complement_remove")
                else "target"
            )
            q, mean, _ = source[kind]
            tq, tm, _ = source["target"]

            def remove(value):
                reference = (
                    None
                    if kind == "target"
                    else torch.linalg.vector_norm(projection_component(value, tq, tm), dim=-1)
                )
                return remove_projection(value, q, mean, reference)

            ops.append(("source", SOURCE, remove))
        if needs_downstream:
            clean = torch.as_tensor(clean_downstream[begin:end], device=device)
            wrong = torch.as_tensor(clean_downstream[wrong_index[begin:end]], device=device)
            tq = downstream["target"][0]
            rq = downstream["random"][0]
            noise = torch.as_tensor(coefficients[begin:end], device=device) @ rq.T

            def recover(value):
                _, same_query_trace = restore_projection(value, clean, tq)
                correct_delta = same_query_trace["delta"]
                reference = torch.linalg.vector_norm(correct_delta, dim=-1)
                if arm == "remove_wrong_source_rescue":
                    return restore_projection(value, wrong, tq, reference)
                if arm == "remove_random_rescue":
                    delta, valid = match_delta_norm(noise, reference)
                    return value + delta.to(value.dtype), {"delta": delta, "valid": valid}
                return restore_projection(value, clean, tq)

            ops.append(("downstream", DOWNSTREAM, recover))
        return ops

    return factory


def transfer_operations(arm, bases, recipient, donor, seed, device):
    """A locked-basis manipulation check, never a basis-fitting input."""
    if arm not in TRANSFER_ARMS or recipient.shape != donor.shape:
        raise ValueError("Invalid donor manipulation")
    if arm.startswith("projected_") and bases is None:
        raise ValueError("Projected donor diagnostic requires an available source basis")
    full_delta = donor.astype(np.float32) - recipient.astype(np.float32)
    if arm == "sham":
        delta, valid = np.zeros_like(full_delta), np.ones(len(donor), dtype=bool)
    elif arm == "full_donor":
        delta, valid = full_delta, np.ones(len(donor), dtype=bool)
    else:
        if arm == "full_norm_random":
            raw = np.random.default_rng(seed).standard_normal(full_delta.shape).astype(np.float32)
            reference = np.linalg.norm(full_delta, axis=-1)
        else:
            tq = bases["target"].q
            target_delta = (full_delta @ tq) @ tq.T
            reference = np.linalg.norm(target_delta, axis=-1)
            q = bases[arm.removeprefix("projected_")].q
            raw = (full_delta @ q) @ q.T
        matched, available = match_delta_norm(torch.from_numpy(raw), torch.from_numpy(reference))
        delta, valid = matched.numpy(), available.numpy()

    def factory(begin, end):
        change = torch.as_tensor(delta[begin:end], device=device)
        available = torch.as_tensor(valid[begin:end], device=device)

        def operation(value):
            return value + change.to(value.dtype), {"delta": change, "valid": available}

        return [("source", SOURCE, operation)]

    return factory


def _availability(arrays):
    valid = np.ones(len(arrays["prediction"]), dtype=bool)
    for key, value in arrays.items():
        if key.endswith("_valid"):
            valid &= value.all(axis=1)
    return valid


def score_rows(arrays, baseline, plan, arm, labels=FAMILIES, label_key="family"):
    """All eligible rows first; original-correct strata remain descriptive only."""
    base_correct = (baseline["prediction"] == plan["answer"]) & baseline["ended"]
    correct = (arrays["prediction"] == plan["answer"]) & arrays["ended"]
    available = _availability(arrays)
    rows = []
    for code, family in enumerate(labels):
        for population in ("all", "ordinary", "old_exception"):
            requested = plan[label_key] == code
            if population != "all":
                requested &= plan["recipient_old_exception"] == (population == "old_exception")
            chosen = requested & available
            n, nb = int(chosen.sum()), int((chosen & base_correct).sum())
            before, after = int((chosen & base_correct).sum()), int((chosen & correct).sum())
            row = {
                "arm": arm,
                "family": family,
                "population": population,
                "requested": int(requested.sum()),
                "available": n,
                "unavailable": int((requested & ~available).sum()),
                "baseline_correct": before,
                "correct": after,
                "accuracy": after / n if n else None,
                "delta_accuracy": (after - before) / n if n else None,
                "known_became_wrong": int((chosen & base_correct & ~correct).sum()),
                "wrong_became_correct": int((chosen & ~base_correct & correct).sum()),
                "known_denominator": nb,
                "known_retention": int((chosen & base_correct & correct).sum()) / nb
                if nb
                else None,
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
            for label in ("source", "downstream"):
                for kind in ("delta", "realized_delta"):
                    key = f"{label}_{kind}_norm"
                    row[f"mean_{key}"] = (
                        float(arrays[key][chosen, 0].mean()) if n and key in arrays else None
                    )
            rows.append(row)
    return rows


def paired_score_rows(
    left, right, baseline, plan, left_name, right_name, labels=FAMILIES, label_key="family"
):
    """Contrasts use a common availability mask, never unmatched denominators."""
    correct_left = (left["prediction"] == plan["answer"]) & left["ended"]
    correct_right = (right["prediction"] == plan["answer"]) & right["ended"]
    correct_clean = (baseline["prediction"] == plan["answer"]) & baseline["ended"]
    available_left, available_right = _availability(left), _availability(right)
    rows = []
    for code, family in enumerate(labels):
        for population in ("all", "ordinary", "old_exception"):
            requested = plan[label_key] == code
            if population != "all":
                requested &= plan["recipient_old_exception"] == (population == "old_exception")
            chosen = requested & available_left & available_right
            n = int(chosen.sum())
            nl, nr = int((chosen & correct_left).sum()), int((chosen & correct_right).sum())
            known_right_wrong = chosen & correct_clean & ~correct_right
            denominator = int(known_right_wrong.sum())
            rows.append(
                {
                    "left": left_name,
                    "right": right_name,
                    "family": family,
                    "population": population,
                    "requested": int(requested.sum()),
                    "common_available": n,
                    "left_unavailable": int((requested & ~available_left).sum()),
                    "right_unavailable": int((requested & ~available_right).sum()),
                    "left_correct": nl,
                    "right_correct": nr,
                    "accuracy_difference": (nl - nr) / n if n else None,
                    "right_wrong_left_correct": int((chosen & ~correct_right & correct_left).sum()),
                    "right_correct_left_wrong": int((chosen & correct_right & ~correct_left).sum()),
                    "original_known_right_wrong": denominator,
                    "original_known_right_wrong_left_recovered": int(
                        (known_right_wrong & correct_left).sum()
                    ),
                    "conditional_recovery_auxiliary": int((known_right_wrong & correct_left).sum())
                    / denominator
                    if denominator
                    else None,
                }
            )
    return rows


def _check_predictions(generated, archived, ids):
    for key in ("prediction", "ended"):
        if not np.array_equal(generated[key], archived[key][ids]):
            raise ValueError(f"Clean frozen-model generation differs from archive: {key}")


def record_archive_comparison(path, generated, archived, ids):
    """Archive differences are audited, never used for calibration or row selection."""
    ids = np.asarray(ids)
    if generated["prediction"].shape != ids.shape or generated["ended"].shape != ids.shape:
        raise ValueError("Local generation shape differs from its declared query plan")
    if generated["ended"].dtype != bool or archived["ended"].dtype != bool:
        raise ValueError("Archive and local termination masks must be boolean")
    _same_arrays(
        path,
        {
            "query_id": ids,
            "local_prediction": generated["prediction"],
            "local_ended": generated["ended"],
            "archive_prediction": archived["prediction"][ids],
            "archive_ended": archived["ended"][ids],
            "prediction_changed": generated["prediction"] != archived["prediction"][ids],
            "ended_changed": generated["ended"] != archived["ended"][ids],
        },
    )


def verify_original_replay(model, world, archived, output, device):
    """Strong identity/numerical gate; original query order and batch remain exact."""
    ids = np.arange(len(world.answers))
    replay = _cached_generate(output / "original-replay.npz", model, world, ids, device, 512)
    _check_predictions(replay, archived, ids)
    _freeze_json(
        output / "original-replay.json",
        {
            "status": "exact_archive_value_and_EOS_replay",
            "queries": len(ids),
            "batch_size": 512,
            "prediction_differences": 0,
            "ended_differences": 0,
            "extra_identity_gate_cost": _cost(replay),
            "raw_sha256": file_sha256(output / "original-replay.npz"),
        },
    )
    return replay


def _cost(arrays):
    return {
        key: int(arrays[key])
        for key in (
            "forward_calls",
            "forward_examples",
            "padded_token_positions",
            "logical_token_positions",
        )
    }


def _cached_generate(path, model, world, ids, device, batch_size, **kwargs):
    if path.exists():
        with np.load(path) as stored:
            result = dict(stored)
        if not np.array_equal(result.pop("query_id"), ids):
            raise ValueError("Cached generation query plan differs")
        return result
    result = generate_queries(
        model, world.prompts[ids], world.lengths[ids], device, batch_size, **kwargs
    )
    atomic_numpy_save(path, query_id=ids, **result)
    return result


def _fit_chain(model, world, chain, pair, dest, device, batch_size, archived):
    plan = calibration_plan(
        world.actual_ids[chain],
        world.person,
        world.membership_ids[chain],
        pair["E"],
        pair["replay"],
    )
    _same_arrays(
        dest / "calibration-plan.npz", {key: np.asarray(value) for key, value in plan.items()}
    )
    actual = _cached_generate(
        dest / "calibration-actual.npz",
        model,
        world,
        plan["actual_query_ids"],
        device,
        batch_size,
        capture_sites=(SOURCE, DOWNSTREAM),
    )
    membership = _cached_generate(
        dest / "calibration-membership.npz",
        model,
        world,
        plan["membership_query_ids"],
        device,
        batch_size,
    )
    record_archive_comparison(
        dest / "archive-calibration-actual.npz", actual, archived, plan["actual_query_ids"]
    )
    record_archive_comparison(
        dest / "archive-calibration-membership.npz",
        membership,
        archived,
        plan["membership_query_ids"],
    )
    prefix = f"{CHAINS[chain]}:"
    allowed = np.array(
        [i for i, label in enumerate(world.token_labels) if label.startswith(prefix)]
    )
    groups, valid = valid_predicted_groups(membership["prediction"], membership["ended"], allowed)
    _same_arrays(dest / "self-predicted-groups.npz", {"prediction": groups, "valid": valid})
    bases, reports, basis_arrays = {}, {}, {}
    for label, site in (("source", SOURCE), ("downstream", DOWNSTREAM)):
        basis, report = fit_predicted_group_bases(
            actual[f"state_{site.layer}_{site.position}"], groups, valid, rank=RANK, seed=74 + chain
        )
        bases[label], reports[label] = basis, report
        if basis is not None:
            for kind, value in basis.items():
                for field in ("q", "mean", "scale"):
                    basis_arrays[f"{label}_{kind}_{field}"] = getattr(value, field)
    _same_arrays(dest / "bases.npz", basis_arrays)
    information = {
        "S_count": len(pair["E"]),
        "R_count": len(pair["replay"]),
        "actual_queries": len(plan["actual_query_ids"]),
        "additional_membership_queries": len(plan["membership_query_ids"]),
        "derived_calibration_queries": 0,
        "true_memberships_supplied": False,
        "membership_syntax_rule": prefix,
        "invalid_groups_never_replaced": True,
        "calibration_cost": {"actual": _cost(actual), "membership": _cost(membership)},
        "basis_reports": reports,
        "variant": "group_means",
        "basis_sha256": file_sha256(dest / "bases.npz"),
    }
    _freeze_json(dest / "calibration.json", information)
    # Truth is read only after Q and its information contract are frozen. These
    # labels score calibration quality; they never select rows or alter Q.
    membership_correct = (
        membership["prediction"] == world.answers[plan["membership_query_ids"]]
    ) & membership["ended"]
    _freeze_json(
        dest / "calibration-diagnostics.json",
        {
            "role": "offline scoring after frozen Q; never fitting or row selection",
            "queries": len(groups),
            "valid_predictions": int(valid.sum()),
            "valid_rate": float(valid.mean()),
            "correct_predictions": int(membership_correct.sum()),
            "accuracy": float(membership_correct.mean()),
            "accuracy_given_valid_auxiliary": float(membership_correct[valid].mean())
            if valid.any()
            else None,
            "basis_sha256": information["basis_sha256"],
        },
    )
    return bases


def run_chain(model, world, chain, pair, dest, device, batch_size, per_population, archived):
    dest.mkdir(parents=True, exist_ok=True)
    bases = _fit_chain(model, world, chain, pair, dest, device, batch_size, archived)
    transfer = make_transfer_plan(world, chain, per_population)
    _same_arrays(dest / "transfer-plan.npz", transfer)
    recipients = transfer["recipient"][transfer["pair_type"] == 0]
    plan = make_probe_plan(world, chain, recipients)
    plan["wrong_source_index"] = wrong_source_indices(plan["family"])
    plan["wrong_source_query_id"] = plan["query_id"][plan["wrong_source_index"]]
    plan["wrong_source_same_answer"] = plan["answer"] == plan["answer"][plan["wrong_source_index"]]
    _same_arrays(dest / "probe-plan.npz", plan)
    ids = plan["query_id"]
    query_site_roles(world.lengths[ids], SOURCE.position)
    if downstream_routing_site(SOURCE, len(model.blocks), world.lengths[ids] - 1) != DOWNSTREAM:
        raise ValueError("Candidate/downstream site mismatch")
    clean = _cached_generate(
        dest / "clean.npz",
        model,
        world,
        ids,
        device,
        batch_size,
        capture_sites=(SOURCE, DOWNSTREAM),
    )
    record_archive_comparison(dest / "archive-probe-clean.npz", clean, archived, ids)
    rows, states, costs, arm_arrays = [], {}, {"clean": _cost(clean)}, {}
    clean_down = clean[f"state_{DOWNSTREAM.layer}_{DOWNSTREAM.position}"]
    for arm in ARMS:
        if arm not in ("clean", "sham") and bases["source"] is None:
            states[arm] = "unavailable_source_rank"
            continue
        if "rescue" in arm and bases["downstream"] is None:
            states[arm] = "unavailable_downstream_rank"
            continue
        if arm == "clean":
            generated = clean
        else:
            operations = lesion_operations(
                arm,
                bases["source"],
                bases["downstream"],
                clean_down,
                plan["wrong_source_index"],
                [world.seed, 979, chain],
                device,
            )
            generated = _cached_generate(
                dest / f"{arm}.npz", model, world, ids, device, batch_size, operations=operations
            )
        if arm in ("sham", "rescue_only"):
            for key in ("prediction", "ended"):
                if not np.array_equal(generated[key], clean[key]):
                    raise ValueError(f"No-lesion control changed output: {arm}/{key}")
        rows.extend(score_rows(generated, clean, plan, arm))
        costs[arm], states[arm] = _cost(generated), "complete"
        arm_arrays[arm] = generated
    comparisons = [
        (arm, "clean")
        for arm in ("sham", "target_remove", "random_remove", "complement_remove", "rescue_only")
    ]
    comparisons += [("target_remove", arm) for arm in ("random_remove", "complement_remove")]
    comparisons += [
        (arm, "target_remove")
        for arm in ("remove_rescue", "remove_wrong_source_rescue", "remove_random_rescue")
    ]
    comparisons += [
        ("remove_rescue", arm) for arm in ("remove_wrong_source_rescue", "remove_random_rescue")
    ]
    paired_rows = []
    for left, right in comparisons:
        if left in arm_arrays and right in arm_arrays:
            paired_rows.extend(
                paired_score_rows(arm_arrays[left], arm_arrays[right], clean, plan, left, right)
            )
    # Donor activations are read only after both bases and their provenance are frozen.
    valid = transfer["donor"] >= 0
    selected = {key: value[valid] for key, value in transfer.items()}
    people = np.unique(np.r_[selected["recipient"], selected["donor"]])
    donor_ids = world.derived_ids[chain, people]
    donor_clean = _cached_generate(
        dest / "transfer-clean.npz",
        model,
        world,
        donor_ids,
        device,
        batch_size,
        capture_sites=(SOURCE,),
    )
    record_archive_comparison(dest / "archive-transfer-cache.npz", donor_clean, archived, donor_ids)
    ri, di = (
        np.searchsorted(people, selected["recipient"]),
        np.searchsorted(people, selected["donor"]),
    )
    source_state = donor_clean[f"state_{SOURCE.layer}_{SOURCE.position}"]
    # v2: the causal baseline has exactly the intervention query order and batch shape.
    # The donor/recipient source cache and intervention delta definitions stay unchanged.
    local_ids = world.derived_ids[chain, selected["recipient"]]
    transfer_local = _cached_generate(
        dest / "transfer-local-clean.npz",
        model,
        world,
        local_ids,
        device,
        batch_size,
        capture_sites=(SOURCE,),
    )
    record_archive_comparison(
        dest / "archive-transfer-local-clean.npz", transfer_local, archived, local_ids
    )
    _same_arrays(
        dest / "transfer-reference-drift.npz",
        {
            "query_id": local_ids,
            "cache_vs_local_state_norm": np.linalg.norm(
                source_state[ri] - transfer_local["state_0_2"], axis=1
            ),
            "cache_prediction": donor_clean["prediction"][ri],
            "cache_ended": donor_clean["ended"][ri],
            "local_prediction": transfer_local["prediction"],
            "local_ended": transfer_local["ended"],
        },
    )
    donor_baseline = {key: transfer_local[key] for key in ("prediction", "ended")}
    donor_plan = {**selected, "answer": selected["recipient_default"]}
    transfer_rows, transfer_status = [], {}
    costs["transfer_clean"] = _cost(donor_clean)
    costs["transfer_local_clean"] = _cost(transfer_local)
    for arm in TRANSFER_ARMS:
        if arm.startswith("projected_") and bases["source"] is None:
            transfer_status[arm] = "unavailable_source_rank"
            continue
        ops = transfer_operations(
            arm,
            bases["source"],
            source_state[ri],
            source_state[di],
            [world.seed, 980, chain],
            device,
        )
        generated = _cached_generate(
            dest / f"transfer-{arm}.npz",
            model,
            world,
            world.derived_ids[chain, selected["recipient"]],
            device,
            batch_size,
            operations=ops,
        )
        if arm == "sham":
            for key in ("prediction", "ended"):
                if not np.array_equal(generated[key], donor_baseline[key]):
                    raise ValueError("Transfer sham changed output")
        scored = score_rows(
            generated, donor_baseline, donor_plan, arm, labels=TRANSFER_TYPES, label_key="pair_type"
        )
        for row in scored:
            code = TRANSFER_TYPES.index(row["family"])
            chosen = (selected["pair_type"] == code) & _availability(generated)
            requested = transfer["pair_type"] == code
            if row["population"] != "all":
                exception = row["population"] == "old_exception"
                chosen &= selected["recipient_old_exception"] == exception
                requested &= transfer["recipient_old_exception"] == exception
            row["plan_requested"] = int(requested.sum())
            row["missing_donors"] = int((requested & ~valid).sum())
            for role in ("recipient_actual", "donor_default", "donor_actual"):
                matched = (generated["prediction"] == selected[role]) & generated["ended"]
                base_matched = (donor_baseline["prediction"] == selected[role]) & donor_baseline[
                    "ended"
                ]
                row[f"matches_{role}"] = int((chosen & matched).sum())
                row[f"baseline_matches_{role}"] = int((chosen & base_matched).sum())
            # Donor's own clean response is descriptive, not an eligibility filter.
            row["donor_answered_default"] = int(
                (
                    chosen
                    & donor_clean["ended"][di]
                    & (donor_clean["prediction"][di] == selected["donor_default"])
                ).sum()
            )
        transfer_rows.extend(scored)
        transfer_status[arm], costs[f"transfer_{arm}"] = "complete", _cost(generated)
    for filename, contents in (
        ("necessity.csv", rows),
        ("paired-necessity.csv", paired_rows),
        ("transfer.csv", transfer_rows),
    ):
        with (dest / filename).open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(contents[0]))
            writer.writeheader()
            writer.writerows(contents)
    write_json(
        dest / "status.json",
        {
            "arms": states,
            "transfer_arms": transfer_status,
            "evaluation_cost": costs,
            "information_boundary": "truth-matched donor states are diagnostic only",
        },
    )
    archive_rows = {}
    for path in sorted(dest.glob("archive-*.npz")):
        with np.load(path) as comparison:
            archive_rows[path.stem] = {
                "queries": len(comparison["query_id"]),
                "prediction_differences": int(comparison["prediction_changed"].sum()),
                "ended_differences": int(comparison["ended_changed"].sum()),
                "case_exclusion": False,
            }
    write_json(dest / "archive-discrepancies.json", archive_rows)
    return {"chain": CHAINS[chain], "arms": states, "transfer_arms": transfer_status}


def scientific_outputs(output):
    """Seal only producer-owned artifacts, excluding mutable queue logs and locks."""
    names = {
        "archive-calibration-actual.npz",
        "archive-calibration-membership.npz",
        "archive-probe-clean.npz",
        "archive-transfer-cache.npz",
        "archive-transfer-local-clean.npz",
        "archive-discrepancies.json",
        "transfer-local-clean.npz",
        "transfer-reference-drift.npz",
        "calibration-plan.npz",
        "calibration-actual.npz",
        "calibration-membership.npz",
        "self-predicted-groups.npz",
        "bases.npz",
        "calibration.json",
        "calibration-diagnostics.json",
        "transfer-plan.npz",
        "probe-plan.npz",
        "transfer-clean.npz",
        "necessity.csv",
        "paired-necessity.csv",
        "transfer.csv",
        "status.json",
    }
    names.update(f"{arm}.npz" for arm in ARMS)
    names.update(f"transfer-{arm}.npz" for arm in TRANSFER_ARMS)
    paths = [
        output / "contract.json",
        output / "original-replay.npz",
        output / "original-replay.json",
    ]
    for chain in CHAINS:
        paths.extend(
            output / chain / name for name in sorted(names) if (output / chain / name).exists()
        )
    return {str(path.relative_to(output)): file_sha256(path) for path in paths}


def run_model(run, output, lock_path, *, device="cuda", batch_size=256, per_population=64):
    """One original pre-edit checkpoint, both chains; separate outputs, restartable nodes."""
    run, output, lock_path = Path(run).resolve(), Path(output).resolve(), Path(lock_path).resolve()
    if output == run or run in output.parents:
        raise ValueError("Never write diagnostics inside the frozen parent")
    lock = json.loads(lock_path.read_text())
    validate_lock(lock)
    config = json.loads((run / "config.json").read_text())
    if (
        config["world"] not in (0, 1)
        or config["model"]["width"] not in (256, 768)
        or config["seed"] not in (0, 1)
        or config["condition"] not in (*CHAINS, "neither")
    ):
        raise ValueError("The original 24-model development matrix is required")
    step = config["study"]["steps"]
    if step != 15360 or per_population != 64:
        raise ValueError("The parent step and 64+64 recipient budget are fixed")
    for key, current in (("torch", torch.__version__), ("cuda", torch.version.cuda)):
        if config[key] != current:
            raise ValueError(f"Parent runtime differs: {key}")
    device = torch.device(device)
    weights = run / f"model-{step}.pt"
    contract = {
        "numerical_policy": NUMERICAL_POLICY,
        "run": str(run),
        "step": step,
        "world": config["world"],
        "seed": config["seed"],
        "width": config["model"]["width"],
        "condition": config["condition"],
        "parent_config_sha256": file_sha256(run / "config.json"),
        "parent_weights_sha256": file_sha256(weights),
        "lock": lock,
        "lock_sha256": file_sha256(lock_path),
        "sources": _sources(),
        "source_site": [0, 2],
        "downstream_site": [1, 2],
        "rank": RANK,
        "per_population": per_population,
        "batch_size": batch_size,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "numpy": np.__version__,
        "device_type": device.type,
        "checkpoint_role": "pre_edit_clean_only",
        "calibration": "S_union_R_actual_queries_with_self_predicted_membership_groups",
        "families": list(FAMILIES),
        "arms": list(ARMS),
        "transfer_types": list(TRANSFER_TYPES),
        "transfer_arms": list(TRANSFER_ARMS),
        "wrong_source_rule": "cyclic_previous_query_within_family_no_truth_or_prediction_selection",
        "cross_query_reference": "The actual-calibration mean is reused for all query families",
        "recovery_source": "Same checkpoint and same query clean post-L1/pos2 target projection",
        "rescue_only_role": "No-lesion downstream sham; cannot establish necessity",
        "random_recovery": "Random rank32 downstream basis; same-query recovery delta norm",
        "rank_failure": "Unavailable, never padded or replaced; report coverage by organization",
        "interpretation": (
            "Artificial-lesion behavior recovery; no claim of edit repair "
            "or organizational specificity"
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    _freeze_json(output / "contract.json", contract)
    complete_path = output / "complete.json"
    if complete_path.exists():
        complete = json.loads(complete_path.read_text())
        for name, digest in complete["outputs"].items():
            if file_sha256(output / name) != digest:
                raise ValueError(f"Completed diagnostic artifact changed: {name}")
        return complete
    repository = Path(__file__).resolve().parents[2]
    world = make_cross_world(config["world"], repository / "data/bios-organization-v1")
    if (
        array_hash(world.answers) != config["truth_sha256"]
        or array_hash(world.prompts) != config["prompts_sha256"]
    ):
        raise ValueError("Parent generated world changed")
    saved = torch.load(weights, map_location="cpu", weights_only=False)
    final = json.loads((run / "learning-complete.json").read_text())
    if (
        saved["config"] != config["model"]
        or saved["step"] != step
        or state_hash(saved["model"]) != final["model_sha256"]
    ):
        raise ValueError("Parent model identity mismatch")
    model = CausalLM(ModelConfig(**saved["config"]))
    model.load_state_dict(saved["model"])
    model.to(device).eval()
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    with np.load(run / f"predictions-{step}.npz") as stored:
        archived = {key: stored[key] for key in ("prediction", "ended")}
    verify_original_replay(model, world, archived, output, device)
    completed = []
    for chain, name in enumerate(CHAINS):
        pair = edit_pair(world, chain)
        reference = run / "edits" / f"{name}-exception-mlp"
        with np.load(reference / "sets.npz") as old:
            for key in ("E", "replay", "exception"):
                if not np.array_equal(old[key], pair[key]):
                    raise ValueError(
                        f"Original exception calibration permission changed: {name}/{key}"
                    )
        if not (reference / "complete.json").exists():
            raise ValueError("Original exception case is incomplete")
        completed.append(
            run_chain(
                model,
                world,
                chain,
                pair,
                output / name,
                device,
                batch_size,
                per_population,
                archived,
            )
        )
        print(
            json.dumps({"event": "necessity_chain_complete", "run": str(run), "chain": name}),
            flush=True,
        )
    if _sources() != contract["sources"]:
        raise ValueError("Diagnostic implementation changed during execution")
    outputs = scientific_outputs(output)
    complete = {"status": "complete", "chains": completed, "outputs": outputs}
    write_json(complete_path, complete)
    return complete
