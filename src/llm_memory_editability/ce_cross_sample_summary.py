"""Pair-clustered statistics for continuous cross-sample CE adaptation.

Both orders, their prefixes and repeated old questions are measurements of the
same fixed panel. Confidence intervals resample fact pairs, never state/query
observations. ``None`` denotes an empty prerequisite or original-correct set.
This module deliberately has no model or filesystem access.
"""

from itertools import combinations

import numpy as np


def _rate(values):
    values = list(values)
    return {
        "correct": int(sum(values)),
        "n": len(values),
        "accuracy": float(np.mean(values)) if values else None,
    }


def _conditional(outcomes, mask):
    result = _rate(o for o, keep in zip(outcomes, mask, strict=True) if keep)
    result.update(
        coverage_n=int(sum(mask)),
        full_n=len(mask),
        coverage=float(np.mean(mask)) if mask else None,
    )
    return result


def _bootstrap(values, settings, denominators=None):
    """A mean, or ratio of clustered sums, with the same paired resampling."""
    values = np.asarray(values, dtype=float)
    repetitions = int(settings.get("bootstrap_repetitions", 10000))
    if repetitions < 1:
        raise ValueError("bootstrap_repetitions must be positive")
    result = {"pairs": len(values), "repetitions": repetitions}
    if not len(values):
        return {**result, "mean": None, "ci95_pair_bootstrap": None}
    draws = np.random.RandomState(settings.get("bootstrap_seed", 2026101611)).randint(
        len(values), size=(repetitions, len(values))
    )
    if denominators is None:
        estimates = values[draws].mean(axis=1)
        mean = float(values.mean())
    else:
        denominators = np.asarray(denominators, dtype=float)
        if denominators.shape != values.shape:
            raise ValueError("bootstrap numerators and denominators differ")
        counts = denominators[draws].sum(axis=1)
        valid = counts > 0
        estimates = values[draws].sum(axis=1)[valid] / counts[valid]
        mean = float(values.sum() / denominators.sum()) if denominators.sum() else None
        result["defined_bootstrap_repetitions"] = int(sum(valid))
    return {
        **result,
        "mean": mean,
        "ci95_pair_bootstrap": (
            np.quantile(estimates, [0.025, 0.975]).tolist() if len(estimates) else None
        ),
    }


def _layout(panel):
    pairs = [p for p in panel["pairs"] if p.get("split", "evaluation") == "evaluation"]
    if not pairs:
        raise ValueError("panel has no evaluation pairs")
    ids = [p["pair_id"] for p in pairs]
    facts = [p[role] for p in pairs for role in ("A", "B")]
    if len(set(ids)) != len(ids) or len({f["fact_id"] for f in facts}) != len(facts):
        raise ValueError("panel has duplicate pair or fact identifiers")
    return pairs, facts


def _scores(record, kind, expected_rows, field="final_correct"):
    item = record["readout"][kind]
    scores = item[field]
    if len(scores) != len(expected_rows):
        raise ValueError(f"{record['key']} {kind}.{field}: wrong panel length")
    if any(not isinstance(v, (bool, np.bool_)) for v in scores):
        raise ValueError(f"{record['key']} {kind}.{field}: scores must be booleans")
    rows = item.get("rows")
    if rows is not None:
        # Accept both full rows and query-only rows, but never silently reorder.
        if len(rows) != len(expected_rows) or any(
            list(row) not in (list(expected), list(expected[:-1]))
            for row, expected in zip(rows, expected_rows, strict=True)
        ):
            raise ValueError(f"{record['key']} {kind}: rows differ from fixed panel order")
    return [bool(v) for v in scores]


def _vectors(record, pairs, facts, field="final_correct"):
    atoms = _scores(record, "atoms", [f["fact"] for f in facts], field)
    composition = _scores(record, "double_new", [p["query"] + [p["answer"]] for p in pairs], field)
    both = [atoms[2 * i] and atoms[2 * i + 1] for i in range(len(pairs))]
    return atoms, composition, both


def _metrics(composition, both):
    return {
        "double_new": _rate(composition),
        "both_atoms": _rate(both),
        "both_atoms_and_composition": _rate(
            a and c for a, c in zip(both, composition, strict=True)
        ),
        "composition_given_both_atoms": _conditional(composition, both),
    }


def _retention(before, after):
    return {
        "before": _rate(before),
        "after": _rate(after),
        "originally_correct_retained": _conditional(after, before),
        "originally_correct_lost": int(
            sum(b and not a for b, a in zip(before, after, strict=True))
        ),
        "newly_correct": int(sum(a and not b for b, a in zip(before, after, strict=True))),
    }


def _old(record, baseline):
    pools = {}
    categories = {"atoms": [[], []], "compositions": [[], []]}
    if set(record["readout"]["old"]) != set(baseline["readout"]["old"]):
        raise ValueError("old pool names differ from baseline")
    for pool, start in baseline["readout"]["old"].items():
        end = record["readout"]["old"][pool]
        if len(start["final_correct"]) != len(end["final_correct"]):
            raise ValueError(f"old pool {pool} has changed size")
        if start.get("rows") != end.get("rows"):
            raise ValueError(f"old pool {pool} rows differ from baseline")
        pools[pool] = {
            "final_correct": _retention(start["final_correct"], end["final_correct"]),
            "answer_correct": _retention(start["answer_correct"], end["answer_correct"]),
        }
        category = "atoms" if pool.startswith("atomic") else "compositions"
        categories[category][0].extend(start["final_correct"])
        categories[category][1].extend(end["final_correct"])
    return {
        "pools": pools,
        "categories": {key: _retention(*values) for key, values in categories.items()},
    }


def _mixed(record, panel):
    queries = panel.get("mixed_queries", [])
    if not queries:
        return {}
    values = _scores(record, "mixed", [q["query"] + [q["answer"]] for q in queries])
    written = set(record["written"])
    companion = record["readout"]["mixed"].get("old_companion")
    if companion is not None and len(companion["final_correct"]) != len(queries):
        raise ValueError("mixed old_companion length differs from fixed panel")
    result = {}
    for direction in sorted({q["direction"] for q in queries}):
        indices = [i for i, q in enumerate(queries) if q["direction"] == direction]
        active = [i for i in indices if queries[i]["fact_id"] in written]
        summary = {
            "all_targets": _rate(values[i] for i in indices),
            "supplied_fact_composition": _rate(values[i] for i in active),
            "old_companion_evaluated": companion is not None,
        }
        if companion is not None:
            old = companion["final_correct"]
            summary["necessary_old_atom"] = _rate(old[i] for i in active)
            summary["given_necessary_old_atom"] = _conditional(
                [values[i] for i in active], [old[i] for i in active]
            )
        result[direction] = summary
    return result


def _validate(records, panel, pairs, facts):
    by_key = {r["key"]: r for r in records}
    if len(by_key) != len(records):
        raise ValueError("duplicate record key")
    baseline = by_key.get("baseline")
    if (
        baseline is None
        or baseline["stream"] != "baseline"
        or baseline["written"]
        or baseline["step"] != 0
    ):
        raise ValueError("one unchanged baseline record is required")
    if any(r["stream"] not in ("baseline", "forward", "reverse", "isolated") for r in records):
        raise ValueError("unknown record stream")
    if sum(r["stream"] == "baseline" for r in records) != 1:
        raise ValueError("exactly one baseline record is required")
    streams = {}
    expected_facts = {f["fact_id"] for f in facts}
    for name in ("forward", "reverse"):
        order = panel["streams"][name]
        if len(order) != len(facts) or set(order) != expected_facts:
            raise ValueError(f"invalid {name} panel order")
        states = sorted([r for r in records if r["stream"] == name], key=lambda r: r["step"])
        if [r["step"] for r in states] != list(range(1, len(facts) + 1)):
            raise ValueError(f"incomplete or duplicate {name} prefix records")
        for step, record in enumerate(states, 1):
            if record["written"] != order[:step]:
                raise ValueError(f"{name} step {step} written facts differ from frozen order")
        streams[name] = states
    isolated = {}
    for record in records:
        if record["stream"] != "isolated":
            continue
        key = (record["pair_id"], record["order"])
        if key in isolated:
            raise ValueError("duplicate isolated reference")
        isolated[key] = record
    for pair in pairs:
        a, b = pair["A"]["fact_id"], pair["B"]["fact_id"]
        for order, written in (
            ("A_only", [a]),
            ("B_only", [b]),
            ("A_then_B", [a, b]),
            ("B_then_A", [b, a]),
        ):
            record = isolated.get((pair["pair_id"], order))
            if record is None or record["written"] != written or record["step"] != len(written):
                raise ValueError(f"missing or invalid isolated {pair['pair_id']} {order}")
    if len(isolated) != 4 * len(pairs):
        raise ValueError("unexpected isolated reference")
    return baseline, streams, isolated


def _clustered_endpoint(pair_rows, field, settings):
    values = [[r[field] for r in pair["orders"].values()] for pair in pair_rows]
    return {
        **_rate(v for pair in values for v in pair),
        "pair_bootstrap": _bootstrap([np.mean(pair) for pair in values], settings),
    }


def _conditional_endpoint(pair_rows, settings, mask_field="both_atoms"):
    numerators, denominators = [], []
    outcomes, masks = [], []
    for pair in pair_rows:
        rows = list(pair["orders"].values())
        numerators.append(sum(r["double_new"] and r[mask_field] for r in rows))
        denominators.append(sum(r[mask_field] for r in rows))
        outcomes.extend(r["double_new"] for r in rows)
        masks.extend(r[mask_field] for r in rows)
    return {
        **_conditional(outcomes, masks),
        "pair_bootstrap": _bootstrap(numerators, settings, denominators),
    }


def summarize(records, panel, settings):
    """Summarize a complete architecture's baseline, 2 streams and 4 refs/pair.

    Required readouts are boolean ``final_correct`` and ``answer_correct`` for
    ``atoms``, ``double_new``, ``mixed`` and each ``old`` pool. Rows, when present,
    must follow the fixed evaluation panel. ``mixed.old_companion`` is optional
    and is marked unevaluated when absent. Returned values are JSON serializable.
    """
    pairs, facts = _layout(panel)
    baseline, streams, isolated = _validate(records, panel, pairs, facts)
    fact_ids = [f["fact_id"] for f in facts]
    result = {
        "schema": "ce-cross-sample-summary-v1",
        "pairs": len(pairs),
        "facts": len(facts),
        "records": len(records),
        "statistical_unit": "fact pair; orders and prefixes stay within pair",
        "generalization_scope": settings.get("generalization_scope"),
        "baseline": {},
        "streams": {},
        "isolated": {},
        "pair_endpoint_values": [],
    }
    _, baseline_composition, baseline_both = _vectors(baseline, pairs, facts)
    result["baseline"] = {
        "final_target_hits": _metrics(baseline_composition, baseline_both),
        "old": _old(baseline, baseline),
    }
    atom_ages, composition_ages = {}, {}
    stream_info = {}
    for name, states in streams.items():
        write_steps = {fact: index for index, fact in enumerate(panel["streams"][name], 1)}
        closure_steps = [max(write_steps[p[role]["fact_id"]] for role in ("A", "B")) for p in pairs]
        vectors = [_vectors(r, pairs, facts) for r in states]
        trajectories = []
        for step, (record, (atoms, composition, both)) in enumerate(
            zip(states, vectors, strict=True), 1
        ):
            written = [i for i, fact in enumerate(fact_ids) if write_steps[fact] <= step]
            closed = [i for i, closure in enumerate(closure_steps) if closure <= step]
            trajectories.append(
                {
                    "key": record["key"],
                    "step": step,
                    "written_atoms": _rate(atoms[i] for i in written),
                    "unwritten_atom_target_hits": _rate(
                        atoms[i] for i in range(len(facts)) if i not in written
                    ),
                    "closed_pairs": len(closed),
                    "final_correct": _metrics(
                        [composition[i] for i in closed], [both[i] for i in closed]
                    ),
                    "old": _old(record, baseline),
                    "mixed": _mixed(record, panel),
                }
            )
            for index in written:
                first_step = write_steps[fact_ids[index]]
                first_correct = vectors[first_step - 1][0][index]
                bucket = atom_ages.setdefault(step - first_step, [[], []])
                bucket[0].append(atoms[index])
                bucket[1].append(first_correct)
            for index in closed:
                closure = closure_steps[index]
                first_correct = vectors[closure - 1][1][index]
                bucket = composition_ages.setdefault(step - closure, [[], []])
                bucket[0].append(composition[index])
                bucket[1].append(first_correct)
        stream_info[name] = (write_steps, closure_steps, vectors)
        answer_atoms, answer_composition, answer_both = _vectors(
            states[-1], pairs, facts, "answer_correct"
        )
        explicit_two_step = states[-1].get(
            "two_step", states[-1]["readout"].get("explicit_two_step")
        )
        if explicit_two_step is not None:
            if len(explicit_two_step) != len(pairs) or [
                row["pair_id"] for row in explicit_two_step
            ] != [p["pair_id"] for p in pairs]:
                raise ValueError("Explicit two-step reference differs from fixed pair order")
        result["streams"][name] = {
            "trajectory": trajectories,
            "endpoint": {
                **trajectories[-1],
                "answer_correct": _metrics(answer_composition, answer_both),
                "all_atoms_answer_correct": _rate(answer_atoms),
                "explicit_two_step": explicit_two_step,
                "explicit_two_step_strict_correct": (
                    _rate(row["strict_correct"] for row in explicit_two_step)
                    if explicit_two_step is not None
                    else None
                ),
            },
            "just_written_atoms": _rate(
                vectors[write_steps[fact] - 1][0][index] for index, fact in enumerate(fact_ids)
            ),
            "endpoint_atomic_retention": _retention(
                [vectors[write_steps[fact] - 1][0][index] for index, fact in enumerate(fact_ids)],
                vectors[-1][0],
            ),
            "endpoint_composition_retention": _retention(
                [vectors[closure - 1][1][i] for i, closure in enumerate(closure_steps)],
                vectors[-1][1],
            ),
            "just_closed_composition": _metrics(
                [vectors[closure - 1][1][i] for i, closure in enumerate(closure_steps)],
                [vectors[closure - 1][2][i] for i, closure in enumerate(closure_steps)],
            ),
        }
    for index, pair in enumerate(pairs):
        row = {"pair_id": pair["pair_id"], "orders": {}}
        for name, states in streams.items():
            write_steps, closure_steps, vectors = stream_info[name]
            order = (
                "A_then_B"
                if write_steps[pair["A"]["fact_id"]] < write_steps[pair["B"]["fact_id"]]
                else "B_then_A"
            )
            _, comp, both = vectors[-1]
            _, entity_comp, entity_both = _vectors(states[-1], pairs, facts, "answer_correct")
            _, isolated_comp, isolated_both = _vectors(
                isolated[(pair["pair_id"], order)], pairs, facts
            )
            closure = closure_steps[index]
            row["orders"][name] = {
                "isolated_order": order,
                "closure_step": closure,
                "closure_age": len(facts) - closure,
                "double_new": comp[index],
                "both_atoms": both[index],
                "both_atoms_and_composition": comp[index] and both[index],
                "double_new_answer": entity_comp[index],
                "both_atoms_answer": entity_both[index],
                "just_closed_double_new": vectors[closure - 1][1][index],
                "just_closed_both_atoms": vectors[closure - 1][2][index],
                "isolated_double_new": isolated_comp[index],
                "isolated_both_atoms": isolated_both[index],
            }
        result["pair_endpoint_values"].append(row)
    pair_rows = result["pair_endpoint_values"]
    result["primary_endpoint"] = {
        key: _clustered_endpoint(pair_rows, key, settings)
        for key in (
            "double_new",
            "both_atoms",
            "both_atoms_and_composition",
            "double_new_answer",
            "both_atoms_answer",
            "just_closed_double_new",
            "isolated_double_new",
        )
    }
    result["primary_endpoint"]["composition_given_both_atoms"] = _conditional_endpoint(
        pair_rows, settings
    )
    references = [
        result["streams"][name]["endpoint"]["explicit_two_step"] for name in ("forward", "reverse")
    ]
    if all(reference is not None for reference in references):
        result["primary_endpoint"]["explicit_two_step_strict_correct"] = {
            **_rate(row["strict_correct"] for reference in references for row in reference),
            "pair_bootstrap": _bootstrap(
                [
                    np.mean([reference[i]["strict_correct"] for reference in references])
                    for i in range(len(pairs))
                ],
                settings,
            ),
        }
    result["continuous_minus_isolated"] = _bootstrap(
        [
            np.mean(
                [int(r["double_new"]) - int(r["isolated_double_new"]) for r in p["orders"].values()]
            )
            for p in pair_rows
        ],
        settings,
    )
    result["endpoint_minus_just_closed"] = _bootstrap(
        [
            np.mean(
                [
                    int(r["double_new"]) - int(r["just_closed_double_new"])
                    for r in p["orders"].values()
                ]
            )
            for p in pair_rows
        ],
        settings,
    )
    for order in ("A_only", "B_only", "A_then_B", "B_then_A"):
        comp, both = [], []
        for index, pair in enumerate(pairs):
            _, values, prerequisites = _vectors(isolated[(pair["pair_id"], order)], pairs, facts)
            comp.append(values[index])
            both.append(prerequisites[index])
        result["isolated"][order] = {
            "all_target_hits": _metrics(comp, both),
            "closed_pairs": len(pairs) if order in ("A_then_B", "B_then_A") else 0,
            "composition_after_both_supplied": (
                _metrics(comp, both) if order in ("A_then_B", "B_then_A") else None
            ),
            "double_new_pair_bootstrap": _bootstrap(comp, settings),
        }
    result["age_retention"] = {}
    for kind, ages in (("atoms", atom_ages), ("compositions", composition_ages)):
        result["age_retention"][kind] = [
            {
                "age": age,
                "accuracy": _rate(current),
                "correct_when_first_written_or_closed": _rate(initial),
                "initially_correct_retained": _conditional(current, initial),
            }
            for age, (current, initial) in sorted(ages.items())
        ]
    return result


def compare_architectures(results, panel, settings):
    """Paired differences and a same-pair/same-order prerequisite comparison.

    ``results`` maps architecture names to complete record lists or summaries.
    Old retention stays separated by architecture, because their baseline
    original-correct populations differ. No mean of old pool percentages is used.
    """
    summaries = {
        name: summarize(value, panel, settings) if isinstance(value, list) else value
        for name, value in results.items()
    }
    if len(summaries) < 2:
        raise ValueError("architecture comparison requires at least two architectures")
    pairs, _ = _layout(panel)
    names = list(summaries)
    expected_ids = [p["pair_id"] for p in pairs]
    for name, summary in summaries.items():
        if [p["pair_id"] for p in summary["pair_endpoint_values"]] != expected_ids:
            raise ValueError(f"{name} has a different pair panel")
    comparison = {
        "schema": "ce-cross-sample-architecture-comparison-v1",
        "architectures": names,
        "pairs": len(pairs),
        "paired_differences": {},
        "common_atomic_prerequisites": {},
        "old_retention_by_architecture": {
            name: {
                stream: summary["streams"][stream]["endpoint"]["old"]
                for stream in ("forward", "reverse")
            }
            for name, summary in summaries.items()
        },
    }
    values = {name: summary["pair_endpoint_values"] for name, summary in summaries.items()}
    for left, right in combinations(names, 2):
        differences = {}
        for metric in ("double_new", "both_atoms", "both_atoms_and_composition"):
            differences[metric] = _bootstrap(
                [
                    np.mean(
                        [
                            int(a["orders"][stream][metric]) - int(b["orders"][stream][metric])
                            for stream in ("forward", "reverse")
                        ]
                    )
                    for a, b in zip(values[left], values[right], strict=True)
                ],
                settings,
            )
        comparison["paired_differences"][f"{left}_minus_{right}"] = differences
    common = []
    for i, pair in enumerate(pairs):
        common.append(
            {
                "pair_id": pair["pair_id"],
                "orders": {
                    stream: all(values[name][i]["orders"][stream]["both_atoms"] for name in names)
                    for stream in ("forward", "reverse")
                },
            }
        )
    for name in names:
        rows = []
        for pair, masks in zip(values[name], common, strict=True):
            rows.append(
                {
                    "pair_id": pair["pair_id"],
                    "orders": {
                        stream: {**row, "common_both_atoms": masks["orders"][stream]}
                        for stream, row in pair["orders"].items()
                    },
                }
            )
        comparison["common_atomic_prerequisites"][name] = _conditional_endpoint(
            rows, settings, "common_both_atoms"
        )
    comparison["common_atomic_prerequisite_mask"] = common
    comparison["common_prerequisite_paired_differences"] = {}
    denominators = [sum(pair["orders"].values()) for pair in common]
    for left, right in combinations(names, 2):
        numerators = [
            sum(
                (
                    int(values[left][i]["orders"][stream]["double_new"])
                    - int(values[right][i]["orders"][stream]["double_new"])
                )
                * common[i]["orders"][stream]
                for stream in ("forward", "reverse")
            )
            for i in range(len(pairs))
        ]
        comparison["common_prerequisite_paired_differences"][f"{left}_minus_{right}"] = _bootstrap(
            numerators, settings, denominators
        )
    return comparison
