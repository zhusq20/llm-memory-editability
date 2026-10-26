"""Contracts that change conclusions about composition and retention."""

import copy
import json

import pytest

from llm_memory_editability.ce_cross_sample_summary import compare_architectures, summarize

SETTINGS = {"bootstrap_repetitions": 200, "bootstrap_seed": 17}


def fixture():
    pairs = [
        {
            "pair_id": f"p{i}",
            "split": "evaluation",
            "A": {"fact_id": f"a{i}", "fact": [10 * i, 0, 10 * i + 1]},
            "B": {"fact_id": f"b{i}", "fact": [10 * i + 1, 1, 10 * i + 2]},
            "query": [10 * i, 0, 1],
            "answer": 10 * i + 2,
        }
        for i in range(2)
    ]
    panel = {
        "pairs": pairs,
        "streams": {"forward": ["a0", "a1", "b0", "b1"], "reverse": ["b1", "b0", "a1", "a0"]},
        "mixed_queries": [
            {
                "fact_id": p[role]["fact_id"],
                "direction": direction,
                "query": [100 + i, 2, 3],
                "answer": 4,
            }
            for i, p in enumerate(pairs)
            for role, direction in (("A", "new_old"), ("B", "old_new"))
        ],
    }
    facts = [p[role] for p in pairs for role in ("A", "B")]

    def score(values, rows):
        return {"final_correct": list(values), "answer_correct": list(values), "rows": rows}

    def record(key, stream, written, pair_id=None, order=None):
        atoms = [f["fact_id"] in written for f in facts]
        comp = [all(p[role]["fact_id"] in written for role in ("A", "B")) for p in pairs]
        readout = {
            "atoms": score(atoms, [f["fact"] for f in facts]),
            "double_new": score(comp, [p["query"] + [p["answer"]] for p in pairs]),
            "mixed": score(
                [q["fact_id"] in written for q in panel["mixed_queries"]],
                [q["query"] + [q["answer"]] for q in panel["mixed_queries"]],
            ),
            "old": {
                "atomic_id": score([True, False], [[0, 0, 1], [1, 0, 2]]),
                "test_oo": score([False, False], [[0, 0, 1, 2], [1, 0, 1, 3]]),
            },
        }
        readout["mixed"]["old_companion"] = score([True, False, True, False], [])
        return {
            "key": key,
            "stream": stream,
            "step": len(written),
            "written": list(written),
            "pair_id": pair_id,
            "order": order,
            "readout": readout,
            "model_digest": "fixed",
            "parameter_digest": key,
        }

    records = [record("baseline", "baseline", [])]
    for stream, sequence in panel["streams"].items():
        for step in range(1, 5):
            records.append(record(f"{stream}-{step}", stream, sequence[:step]))
    for pair in pairs:
        a, b = pair["A"]["fact_id"], pair["B"]["fact_id"]
        for order, written in (
            ("A_only", [a]),
            ("B_only", [b]),
            ("A_then_B", [a, b]),
            ("B_then_A", [b, a]),
        ):
            records.append(
                record(f"{pair['pair_id']}-{order}", "isolated", written, pair["pair_id"], order)
            )
    return panel, records


def get(records, key):
    return next(r for r in records if r["key"] == key)


def test_pair_clustered_primary_and_entity_eos_separation():
    panel, records = fixture()
    get(records, "forward-4")["readout"]["double_new"]["final_correct"][0] = False
    result = summarize(records, panel, SETTINGS)
    primary = result["primary_endpoint"]
    assert primary["double_new"]["accuracy"] == 0.75
    assert primary["double_new"]["pair_bootstrap"]["pairs"] == 2
    assert primary["double_new"]["pair_bootstrap"]["ci95_pair_bootstrap"] == [0.5, 1.0]
    assert primary["double_new_answer"]["accuracy"] == 1.0
    assert result["continuous_minus_isolated"]["mean"] == -0.25
    assert result["continuous_minus_isolated"]["ci95_pair_bootstrap"] == [-0.5, 0.0]
    assert json.loads(json.dumps(result)) == result
    assert result == summarize(records, panel, SETTINGS)


def test_explicit_two_step_reference_keeps_raw_pairs_and_averages_orders():
    panel, records = fixture()
    forward = [
        {"pair_id": "p0", "predicted_bridge": 1, "strict_correct": True},
        {"pair_id": "p1", "predicted_bridge": 0, "strict_correct": False},
    ]
    reverse = [
        {"pair_id": "p0", "predicted_bridge": 1, "strict_correct": True},
        {"pair_id": "p1", "predicted_bridge": 11, "strict_correct": True},
    ]
    get(records, "forward-4")["two_step"] = forward
    get(records, "reverse-4")["two_step"] = reverse
    result = summarize(records, panel, SETTINGS)
    endpoint = result["streams"]["forward"]["endpoint"]
    assert endpoint["explicit_two_step"] == forward
    assert endpoint["explicit_two_step_strict_correct"]["accuracy"] == 0.5
    primary = result["primary_endpoint"]["explicit_two_step_strict_correct"]
    assert primary["accuracy"] == 0.75
    assert primary["pair_bootstrap"]["ci95_pair_bootstrap"] == [0.5, 1.0]


def test_conditional_subset_keeps_full_denominator_and_zero_coverage_is_na():
    panel, records = fixture()
    for stream in ("forward", "reverse"):
        final = get(records, f"{stream}-4")["readout"]
        final["atoms"]["final_correct"] = [False] * 4
        final["double_new"]["final_correct"] = [True] * 2
    result = summarize(records, panel, SETTINGS)["primary_endpoint"]
    assert result["double_new"]["accuracy"] == 1.0
    assert result["both_atoms_and_composition"]["accuracy"] == 0.0
    conditional = result["composition_given_both_atoms"]
    assert conditional["full_n"] == 4
    assert conditional["coverage_n"] == 0
    assert conditional["accuracy"] is None
    assert conditional["pair_bootstrap"]["ci95_pair_bootstrap"] is None
    assert conditional["pair_bootstrap"]["defined_bootstrap_repetitions"] == 0


def test_old_zero_baseline_stays_na_and_loss_is_visible_despite_equal_accuracy():
    panel, records = fixture()
    get(records, "forward-4")["readout"]["old"]["atomic_id"]["final_correct"] = [False, True]
    pools = summarize(records, panel, SETTINGS)["streams"]["forward"]["endpoint"]["old"]["pools"]
    old = pools["atomic_id"]["final_correct"]
    assert old["before"]["accuracy"] == old["after"]["accuracy"] == 0.5
    assert old["originally_correct_retained"]["accuracy"] == 0.0
    assert old["originally_correct_lost"] == old["newly_correct"] == 1
    assert pools["test_oo"]["final_correct"]["originally_correct_retained"]["accuracy"] is None


def test_unclosed_chains_never_count_as_failures_or_retention_observations():
    panel, records = fixture()
    # A lucky target hit before either premise is supplied is a target hit only.
    get(records, "forward-1")["readout"]["double_new"]["final_correct"] = [True, True]
    result = summarize(records, panel, SETTINGS)
    trajectory = result["streams"]["forward"]["trajectory"]
    assert trajectory[0]["closed_pairs"] == 0
    assert trajectory[0]["final_correct"]["double_new"]["n"] == 0
    assert trajectory[1]["closed_pairs"] == 0
    assert trajectory[2]["closed_pairs"] == 1
    ages = result["age_retention"]["compositions"]
    assert [(r["age"], r["accuracy"]["n"]) for r in ages] == [(0, 4), (1, 2)]
    assert result["isolated"]["A_only"]["composition_after_both_supplied"] is None


def test_age_retention_filters_atoms_that_were_not_correct_when_written():
    panel, records = fixture()
    # a0 fails when first written, then succeeds; this is learning, not retention.
    get(records, "forward-1")["readout"]["atoms"]["final_correct"][0] = False
    # a1 succeeds when written and is subsequently lost.
    get(records, "forward-3")["readout"]["atoms"]["final_correct"][2] = False
    result = summarize(records, panel, SETTINGS)
    age_one = result["age_retention"]["atoms"][1]
    assert age_one["accuracy"]["n"] == 6
    assert age_one["initially_correct_retained"]["n"] == 5
    assert age_one["initially_correct_retained"]["correct"] == 4
    assert result["streams"]["forward"]["just_written_atoms"]["accuracy"] == 0.75
    endpoint = result["streams"]["forward"]["endpoint_atomic_retention"]
    assert endpoint["before"]["correct"] == 3
    assert endpoint["originally_correct_retained"]["n"] == 3
    assert endpoint["originally_correct_retained"]["accuracy"] == 1.0


def test_continuous_matches_isolated_order_in_each_stream():
    panel, records = fixture()
    for pair in ("p0", "p1"):
        get(records, f"{pair}-B_then_A")["readout"]["double_new"]["final_correct"] = [False, False]
    result = summarize(records, panel, SETTINGS)
    assert result["continuous_minus_isolated"]["mean"] == 0.5
    assert result["continuous_minus_isolated"]["ci95_pair_bootstrap"] == [0.5, 0.5]
    for pair in result["pair_endpoint_values"]:
        assert pair["orders"]["forward"]["isolated_order"] == "A_then_B"
        assert pair["orders"]["reverse"]["isolated_order"] == "B_then_A"


def test_mixed_requires_new_fact_and_keeps_necessary_old_atom_coverage():
    panel, records = fixture()
    result = summarize(records, panel, SETTINGS)
    first = result["streams"]["forward"]["trajectory"][0]["mixed"]
    assert first["new_old"]["all_targets"]["n"] == 2
    assert first["new_old"]["supplied_fact_composition"]["n"] == 1
    assert first["old_new"]["supplied_fact_composition"]["n"] == 0
    final = result["streams"]["forward"]["endpoint"]["mixed"]
    assert final["old_new"]["necessary_old_atom"]["accuracy"] == 0.0
    assert final["old_new"]["given_necessary_old_atom"]["accuracy"] is None


def test_architecture_common_prerequisites_use_same_pair_and_order():
    panel, first = fixture()
    second, third = copy.deepcopy(first), copy.deepcopy(first)
    get(first, "forward-4")["readout"]["atoms"]["final_correct"][0] = False
    get(second, "reverse-4")["readout"]["atoms"]["final_correct"][2] = False
    get(third, "forward-4")["readout"]["double_new"]["final_correct"] = [False, False]
    comparison = compare_architectures(
        {"transformer": first, "loop": second, "discoloop": third}, panel, SETTINGS
    )
    for name in ("transformer", "loop", "discoloop"):
        conditional = comparison["common_atomic_prerequisites"][name]
        assert conditional["coverage_n"] == 2
        assert conditional["full_n"] == 4
    assert comparison["common_atomic_prerequisites"]["discoloop"]["accuracy"] == 0.5
    diff = comparison["paired_differences"]["transformer_minus_discoloop"]["double_new"]
    assert diff["mean"] == 0.5
    assert diff["ci95_pair_bootstrap"] == [0.5, 0.5]
    common_diff = comparison["common_prerequisite_paired_differences"][
        "transformer_minus_discoloop"
    ]
    assert common_diff["mean"] == 0.5
    assert set(comparison["old_retention_by_architecture"]) == {"transformer", "loop", "discoloop"}
    json.dumps(comparison)


@pytest.mark.parametrize(
    "failure", ["missing_prefix", "duplicate", "wrong_written", "wrong_rows", "missing_isolated"]
)
def test_invalid_or_incomplete_panels_fail_before_statistics(failure):
    panel, records = fixture()
    if failure == "missing_prefix":
        records.remove(get(records, "forward-2"))
    elif failure == "duplicate":
        records.append(copy.deepcopy(records[0]))
    elif failure == "wrong_written":
        get(records, "forward-2")["written"].reverse()
    elif failure == "wrong_rows":
        get(records, "forward-2")["readout"]["atoms"]["rows"].reverse()
    elif failure == "missing_isolated":
        records.remove(get(records, "p0-A_then_B"))
    with pytest.raises(ValueError):
        summarize(records, panel, SETTINGS)
