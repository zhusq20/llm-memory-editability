"""Pure-graph first/second-hop truth, interference isolation and zero coverage."""

import copy
from types import SimpleNamespace

import numpy as np
import pytest

from llm_memory_editability.ttcd_merge_hop_panel import build_panel


def graph(*, first_candidates=1, missing_new=False, edited_old=False, edited_new=False):
    """A sparse graph with the same 32 distinct targets in both query roles."""
    atoms = [[100 * i + 10, 0, 100 * i + 20] for i in range(32)]
    if edited_old:
        atoms[0][2] = atoms[1][0]
    new_tails = [100 * i + 30 for i in range(32)]
    if edited_new:
        new_tails[0] = atoms[1][0]
    lookup = np.full((4000, 16), -1, dtype=np.int64)
    for index, (head, relation, _) in enumerate(atoms):
        lookup[head, relation] = index

    def add_atom(head, relation, tail):
        if lookup[head, relation] >= 0:
            return int(lookup[head, relation])
        index = len(atoms)
        atoms.append([head, relation, tail])
        lookup[head, relation] = index
        return index

    chains, first, second = [], [], []

    def add_chain(a, b):
        first.append(a)
        second.append(b)
        chains.append([atoms[a][0], atoms[a][1], atoms[b][1], atoms[b][2]])
        return len(chains) - 1

    facts = []
    for i in range(32):
        predecessor = add_atom(100 * i + 40, 15, atoms[i][0])
        original_d = add_chain(predecessor, i)
        for candidate in range(first_candidates):
            relation = candidate + 1
            if i == 0 and (edited_old or edited_new):
                relation = 0
            old_successor = add_atom(atoms[i][2], relation, 100 * i + 50 + candidate)
            if not (i == 0 and missing_new):
                add_atom(new_tails[i], relation, 100 * i + 70 + candidate)
            add_chain(i, old_successor)
        facts.append(
            {
                "case_id": f"evaluation-{i}",
                "group": "id" if i < 16 else "ood",
                "atom": i,
                "old_tail": atoms[i][2],
                "new_tail": new_tails[i],
                "D": [original_d],
            }
        )
    support = np.zeros(len(chains), dtype=bool)
    world = SimpleNamespace(
        atoms=np.asarray(atoms, dtype=np.int64),
        chains=np.asarray(chains, dtype=np.int64),
        first=np.asarray(first, dtype=np.int64),
        second=np.asarray(second, dtype=np.int64),
        lookup=lookup,
        entities=len(lookup),
        support=lambda _phi: support,
    )
    panel = {
        "schema": "unchanged-parent",
        "facts": facts,
        "edited_atoms": list(range(32)),
        "orders": [list(range(i, 32)) + list(range(i)) for i in range(32)],
        "U_atoms": [len(atoms) - 1],
        "U_chains": [],
        "extra_original_field": {"preserve": [1, 2, 3]},
    }
    return panel, world, support


SETTINGS = {"first_hop_panel": {"seed": 20261015, "max_queries_per_fact": 8}}


def test_same_fixed_facts_have_both_hop_roles_without_mutating_parent():
    parent, world, _ = graph()
    before = copy.deepcopy(parent)
    result = build_panel(parent, world, SETTINGS, 7.2)
    assert parent == before
    assert {key: result[key] for key in parent} == parent
    assert len(result["facts"]) == len(result["first_hop"]) == 32
    assert [fact["group"] for fact in result["facts"]] == ["id"] * 16 + ["ood"] * 16
    for fact, entry in zip(parent["facts"], result["first_hop"], strict=True):
        assert int(world.second[fact["D"][0]]) == fact["atom"]
        query = entry["D"][0]
        assert int(world.first[query]) == fact["atom"]
        old_second, new_second = entry["original_successor_atoms"][0], entry["successor_atoms"][0]
        assert old_second not in parent["edited_atoms"]
        assert new_second not in parent["edited_atoms"]
        assert world.atoms[new_second, 0] == fact["new_tail"]
        assert entry["new_answers"] == [int(world.atoms[new_second, 2])]
        assert entry["new_answers"][0] != int(world.chains[query, 3])
    result["facts"][0]["D"].append(999)
    assert parent == before


@pytest.mark.parametrize("condition", ["missing_new", "edited_old", "edited_new"])
def test_missing_or_edited_successors_keep_fact_with_zero_coverage(condition):
    parent, world, _ = graph(**{condition: True})
    result = build_panel(parent, world, SETTINGS, 7.2)
    assert result["facts"] == parent["facts"]
    assert result["first_hop"][0]["D"] == []
    coverage = result["first_hop_coverage"]
    assert coverage["empty_fact_indices"] == [0]
    assert coverage["facts_total"] == 32 and coverage["facts_covered"] == 31
    assert coverage["queries_total"] == 31
    assert coverage["per_fact"][0]["candidates"] == 0


def test_unchanged_final_answer_is_not_a_first_hop_edit_probe():
    parent, world, _ = graph()
    new_successor = world.lookup[parent["facts"][0]["new_tail"], 1]
    query = int(np.flatnonzero(world.first == 0)[0])
    world.atoms[new_successor, 2] = world.chains[query, 3]
    result = build_panel(parent, world, SETTINGS, 7.2)
    assert result["first_hop"][0]["D"] == []
    assert result["first_hop_coverage"]["empty_fact_indices"] == [0]


def test_training_support_excluded_for_requested_phi():
    parent, world, support = graph()
    query = int(np.flatnonzero(world.first == 0)[0])
    support[query] = True
    calls = []

    def actual_support(phi):
        calls.append(phi)
        return support

    world.support = actual_support
    result = build_panel(parent, world, SETTINGS, 3.5)
    assert calls == [3.5]
    assert result["first_hop"][0]["D"] == []


def test_all_zero_coverage_retains_full_fact_and_control_panel():
    parent, world, support = graph()
    support[np.isin(world.first, parent["edited_atoms"])] = True
    result = build_panel(parent, world, SETTINGS, 7.2)
    assert {key: result[key] for key in parent} == parent
    coverage = result["first_hop_coverage"]
    assert coverage["facts_total"] == 32 and coverage["facts_covered"] == 0
    assert coverage["empty_fact_indices"] == list(range(32))
    assert coverage["queries_total"] == coverage["candidate_queries_total"] == 0
    assert all(entry["D"] == [] for entry in result["first_hop"])


def test_selection_is_deterministic_capped_and_reports_unselected_candidates():
    parent, world, _ = graph(first_candidates=12)
    first = build_panel(parent, world, SETTINGS, 7.2)
    assert first == build_panel(parent, world, SETTINGS, 7.2)
    assert all(len(entry["D"]) == 8 for entry in first["first_hop"])
    coverage = first["first_hop_coverage"]
    assert coverage["candidate_queries_total"] == 32 * 12
    assert coverage["queries_total"] == 32 * 8
    assert all(row["candidates"] == 12 and row["selected"] == 8 for row in coverage["per_fact"])


@pytest.mark.parametrize("invalid", ["first_hop", "trained", "other_target"])
def test_original_second_hop_panel_is_strictly_validated(invalid):
    parent, world, support = graph()
    if invalid == "first_hop":
        parent["facts"][0]["D"] = np.flatnonzero(world.first == 0).tolist()
    elif invalid == "trained":
        support[parent["facts"][0]["D"][0]] = True
    else:
        parent["facts"][0]["D"] = parent["facts"][1]["D"]
    with pytest.raises(ValueError, match="only the second hop of an untrained chain"):
        build_panel(parent, world, SETTINGS, 7.2)


def test_original_second_hop_rejects_any_edited_first_key():
    parent, world, _ = graph(edited_old=True)
    query = int(np.flatnonzero(world.first == 0)[0])
    assert world.second[query] == parent["facts"][1]["atom"]
    parent["facts"][1]["D"] = [query]
    with pytest.raises(ValueError, match="only the second hop"):
        build_panel(parent, world, SETTINGS, 7.2)


@pytest.mark.parametrize("seed,limit", [(1, 8), (20261015, 9), (20261015, 0)])
def test_selection_contract_rejects_changed_seed_or_excess_budget(seed, limit):
    parent, world, _ = graph()
    settings = {"first_hop_panel": {"seed": seed, "max_queries_per_fact": limit}}
    with pytest.raises(ValueError, match="seed 20261015"):
        build_panel(parent, world, settings, 7.2)
