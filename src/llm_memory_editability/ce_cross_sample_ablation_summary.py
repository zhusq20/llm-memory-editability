"""Continuous-only paired summaries without invented isolated references."""

import numpy as np

from .ce_cross_sample_summary import (
    _bootstrap,
    _clustered_endpoint,
    _conditional,
    _conditional_endpoint,
    _layout,
    _metrics,
    _mixed,
    _old,
    _rate,
    _retention,
    _vectors,
    compare_architectures,
)


def _validate_streams(records, panel, facts):
    expected = 1 + 2 * len(facts)
    if len(records) != expected or len({r["key"] for r in records}) != expected:
        raise ValueError(f"Exactly {expected} distinct continuous records are required")
    if any(r["stream"] not in ("baseline", "forward", "reverse") for r in records):
        raise ValueError("Continuous-only summary does not accept isolated or unknown streams")
    baselines = [r for r in records if r["stream"] == "baseline"]
    if (
        len(baselines) != 1
        or baselines[0]["key"] != "baseline"
        or baselines[0]["step"] != 0
        or baselines[0]["written"]
    ):
        raise ValueError("Exactly one unchanged baseline is required")
    streams = {}
    expected_facts = {f["fact_id"] for f in facts}
    for name in ("forward", "reverse"):
        order = panel["streams"][name]
        if len(order) != len(facts) or set(order) != expected_facts:
            raise ValueError(f"Invalid frozen {name} order")
        states = sorted([r for r in records if r["stream"] == name], key=lambda r: r["step"])
        if [r["step"] for r in states] != list(range(1, len(facts) + 1)):
            raise ValueError(f"Incomplete or duplicate {name} prefix")
        for step, record in enumerate(states, 1):
            if record["written"] != order[:step]:
                raise ValueError(f"{name} step {step} differs from the frozen written prefix")
        streams[name] = states
    if panel["streams"]["reverse"] != panel["streams"]["forward"][::-1]:
        raise ValueError("Reverse must reverse the entire forward stream")
    return baselines[0], streams


def _two_step(record, pairs):
    reference = record.get("two_step", record["readout"].get("explicit_two_step"))
    if reference is not None and (
        len(reference) != len(pairs)
        or [r["pair_id"] for r in reference] != [p["pair_id"] for p in pairs]
    ):
        raise ValueError("Explicit two-step reference differs from the fixed pair panel")
    return reference


def _atom_endpoint(pair_rows, settings, initial=False):
    field = "just_written_atoms" if initial else "endpoint_atoms"
    values = [
        [value for order in pair["orders"].values() for value in order[field]] for pair in pair_rows
    ]
    return {
        **_rate(value for pair in values for value in pair),
        "pair_bootstrap": _bootstrap([np.mean(pair) for pair in values], settings),
    }


def summarize_streams(records, panel, evaluation):
    """Require one baseline and every forward/reverse prefix, with no isolated arm.

    With the frozen 32-pair panel this is exactly 129 records. Raw query scoring,
    original-correct retention and fact-pair resampling match the original study.
    """
    pairs, facts = _layout(panel)
    baseline, streams = _validate_streams(records, panel, facts)
    fact_ids = [f["fact_id"] for f in facts]
    if "old_eval_pools" in panel:
        old = baseline["readout"]["old"]
        if set(old) != set(panel["old_eval_pools"]):
            raise ValueError("Baseline old pool names differ from the frozen panel")
        for pool, rows in panel["old_eval_pools"].items():
            if old[pool].get("rows") != rows:
                raise ValueError("Baseline old pool rows differ from the frozen panel")
    baseline_atoms, baseline_comp, baseline_both = _vectors(baseline, pairs, facts)
    result = {
        "schema": "ce-cross-sample-continuous-only-summary-v1",
        "pairs": len(pairs),
        "facts": len(facts),
        "records": len(records),
        "isolated_references": "not included; none synthesized",
        "statistical_unit": "fact pair; both stream orders and all prefixes remain within pair",
        "generalization_scope": evaluation.get("generalization_scope"),
        "baseline": {
            "atom_target_hits": _rate(baseline_atoms),
            "final_target_hits": _metrics(baseline_comp, baseline_both),
            "old": _old(baseline, baseline),
        },
        "streams": {},
        "pair_endpoint_values": [],
        "fact_endpoint_values": [],
    }
    age_buckets = {"atoms": {}, "compositions": {}}
    stream_info = {}
    for name, states in streams.items():
        write_steps = {fact: i for i, fact in enumerate(panel["streams"][name], 1)}
        closure_steps = [max(write_steps[p[role]["fact_id"]] for role in ("A", "B")) for p in pairs]
        vectors = [_vectors(r, pairs, facts) for r in states]
        trajectories = []
        for step, (record, (atoms, composition, both)) in enumerate(
            zip(states, vectors, strict=True), 1
        ):
            written = [i for i, fact in enumerate(fact_ids) if write_steps[fact] <= step]
            closed = [i for i, closure in enumerate(closure_steps) if closure <= step]
            atom_at_write = [vectors[write_steps[fact_ids[i]] - 1][0][i] for i in written]
            comp_at_close = [vectors[closure_steps[i] - 1][1][i] for i in closed]
            new_index = fact_ids.index(panel["streams"][name][step - 1])
            trajectories.append(
                {
                    "key": record["key"],
                    "step": step,
                    "just_written_fact_id": fact_ids[new_index],
                    "just_written_correct": atoms[new_index],
                    "write_info": record.get("write_info"),
                    "written_atoms": _rate(atoms[i] for i in written),
                    "written_atom_retention": _retention(
                        atom_at_write, [atoms[i] for i in written]
                    ),
                    "unwritten_atom_target_hits": _rate(
                        atoms[i] for i in range(len(facts)) if i not in written
                    ),
                    "closed_pairs": len(closed),
                    "final_correct": _metrics(
                        [composition[i] for i in closed], [both[i] for i in closed]
                    ),
                    "closed_composition_retention": _retention(
                        comp_at_close, [composition[i] for i in closed]
                    ),
                    "old": _old(record, baseline),
                    "mixed": _mixed(record, panel),
                }
            )
            for index in written:
                age = step - write_steps[fact_ids[index]]
                bucket = age_buckets["atoms"].setdefault(age, [[], []])
                bucket[0].append(atoms[index])
                bucket[1].append(vectors[write_steps[fact_ids[index]] - 1][0][index])
            for index in closed:
                age = step - closure_steps[index]
                bucket = age_buckets["compositions"].setdefault(age, [[], []])
                bucket[0].append(composition[index])
                bucket[1].append(vectors[closure_steps[index] - 1][1][index])
        first_atoms = [vectors[write_steps[f] - 1][0][i] for i, f in enumerate(fact_ids)]
        first_comps = [vectors[closure - 1][1][i] for i, closure in enumerate(closure_steps)]
        answer_atoms, answer_comps, answer_both = _vectors(
            states[-1], pairs, facts, "answer_correct"
        )
        reference = _two_step(states[-1], pairs)
        result["streams"][name] = {
            "trajectory": trajectories,
            "endpoint": {
                **trajectories[-1],
                "answer_correct": _metrics(answer_comps, answer_both),
                "all_atoms_answer_correct": _rate(answer_atoms),
                "explicit_two_step": reference,
                "explicit_two_step_strict_correct": (
                    _rate(r["strict_correct"] for r in reference) if reference is not None else None
                ),
            },
            "just_written_atoms": _rate(first_atoms),
            "endpoint_atomic_retention": _retention(first_atoms, vectors[-1][0]),
            "endpoint_composition_retention": _retention(first_comps, vectors[-1][1]),
            "just_closed_composition": _metrics(
                first_comps, [vectors[closure - 1][2][i] for i, closure in enumerate(closure_steps)]
            ),
        }
        stream_info[name] = (write_steps, closure_steps, vectors, answer_comps, answer_both)
    for i, fact in enumerate(facts):
        row = {"fact_id": fact["fact_id"], "pair_id": pairs[i // 2]["pair_id"], "orders": {}}
        for name, (writes, _, vectors, _, _) in stream_info.items():
            write_step = writes[fact["fact_id"]]
            row["orders"][name] = {
                "write_step": write_step,
                "write_age": len(facts) - write_step,
                "just_written_correct": vectors[write_step - 1][0][i],
                "endpoint_correct": vectors[-1][0][i],
            }
        result["fact_endpoint_values"].append(row)
    for i, pair in enumerate(pairs):
        row = {"pair_id": pair["pair_id"], "orders": {}}
        for name, (writes, closures, vectors, answer_comp, answer_both) in stream_info.items():
            atoms, comp, both = vectors[-1]
            row["orders"][name] = {
                "write_order": (
                    "A_then_B"
                    if writes[pair["A"]["fact_id"]] < writes[pair["B"]["fact_id"]]
                    else "B_then_A"
                ),
                "closure_step": closures[i],
                "closure_age": len(facts) - closures[i],
                "double_new": comp[i],
                "both_atoms": both[i],
                "both_atoms_and_composition": comp[i] and both[i],
                "double_new_answer": answer_comp[i],
                "both_atoms_answer": answer_both[i],
                "just_closed_double_new": vectors[closures[i] - 1][1][i],
                "just_closed_both_atoms": vectors[closures[i] - 1][2][i],
                "endpoint_atoms": atoms[2 * i : 2 * i + 2],
                "just_written_atoms": [
                    vectors[writes[pair[role]["fact_id"]] - 1][0][2 * i + j]
                    for j, role in enumerate(("A", "B"))
                ],
            }
        result["pair_endpoint_values"].append(row)
    rows = result["pair_endpoint_values"]
    result["primary_endpoint"] = {
        metric: _clustered_endpoint(rows, metric, evaluation)
        for metric in (
            "double_new",
            "both_atoms",
            "both_atoms_and_composition",
            "double_new_answer",
            "both_atoms_answer",
            "just_closed_double_new",
        )
    }
    result["primary_endpoint"].update(
        all_written_atoms=_atom_endpoint(rows, evaluation),
        just_written_atoms=_atom_endpoint(rows, evaluation, initial=True),
        composition_given_both_atoms=_conditional_endpoint(rows, evaluation),
    )
    initial, retained, denominators = [], [], []
    for pair in rows:
        starts = [v for order in pair["orders"].values() for v in order["just_written_atoms"]]
        ends = [v for order in pair["orders"].values() for v in order["endpoint_atoms"]]
        initial.extend(starts)
        retained.extend(ends)
        denominators.append(sum(starts))
    numerators = [
        sum(
            end and start
            for order in pair["orders"].values()
            for start, end in zip(order["just_written_atoms"], order["endpoint_atoms"], strict=True)
        )
        for pair in rows
    ]
    result["primary_endpoint"]["initially_correct_new_atoms_retained"] = {
        **_conditional(retained, initial),
        "pair_bootstrap": _bootstrap(numerators, evaluation, denominators),
    }
    result["age_retention"] = {
        kind: [
            {
                "age": age,
                "accuracy": _rate(current),
                "correct_when_first_written_or_closed": _rate(first),
                "initially_correct_retained": _conditional(current, first),
            }
            for age, (current, first) in sorted(ages.items())
        ]
        for kind, ages in age_buckets.items()
    }
    result["endpoint_minus_just_closed"] = _bootstrap(
        [
            np.mean(
                [
                    int(r["double_new"]) - int(r["just_closed_double_new"])
                    for r in p["orders"].values()
                ]
            )
            for p in rows
        ],
        evaluation,
    )
    references = [result["streams"][s]["endpoint"]["explicit_two_step"] for s in streams]
    if all(r is not None for r in references):
        result["primary_endpoint"]["explicit_two_step_strict_correct"] = {
            **_rate(r["strict_correct"] for reference in references for r in reference),
            "pair_bootstrap": _bootstrap(
                [np.mean([r[i]["strict_correct"] for r in references]) for i in range(len(pairs))],
                evaluation,
            ),
        }
    return result


def compare_with_control(control, variants, panel, evaluation):
    """Compare each fixed variant with its architecture's original recipe.

    Inputs may be continuous record lists or outputs from ``summarize_streams``.
    No winner selection or pooled percentage over distinct old baseline sets.
    """
    control = (
        summarize_streams(control, panel, evaluation) if isinstance(control, list) else control
    )
    summaries = {
        name: summarize_streams(value, panel, evaluation) if isinstance(value, list) else value
        for name, value in variants.items()
    }
    result = {"schema": "ce-cross-sample-ablation-comparison-v1", "variants": {}}
    for name, variant in summaries.items():
        comparison = compare_architectures(
            {"variant": variant, "control": control}, panel, evaluation
        )
        comparison["baseline_old_coverage_identical"] = (
            variant["baseline"]["old"] == control["baseline"]["old"]
        )
        atom_differences = [
            np.mean(
                [
                    int(v) - int(c)
                    for stream in ("forward", "reverse")
                    for v, c in zip(
                        variant_row["orders"][stream]["endpoint_atoms"],
                        control_row["orders"][stream]["endpoint_atoms"],
                        strict=True,
                    )
                ]
            )
            for variant_row, control_row in zip(
                variant["pair_endpoint_values"], control["pair_endpoint_values"], strict=True
            )
        ]
        comparison["paired_differences"]["variant_minus_control"]["all_written_atoms"] = _bootstrap(
            atom_differences, evaluation
        )
        result["variants"][name] = comparison
    return result
