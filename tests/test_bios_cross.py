"""Scientific contracts for the relation/organization crossover."""

import numpy as np
import pytest

from llm_memory_editability.bios_cross import (
    CONDITIONS,
    audit,
    documents,
    edit_pair,
    epoch_documents,
    qa_schedule,
    render,
)


@pytest.fixture(scope="module")
def cross_world():
    # Reuse a generated source world with no dependency on local downloaded data.
    from pathlib import Path
    from tempfile import TemporaryDirectory
    from unittest.mock import patch

    from test_bios_organization_data import organization_world

    from llm_memory_editability.bios_cross import make_cross_world

    class Factory:
        def __init__(self, root):
            self.root = root

        def mktemp(self, name):
            path = self.root / name
            path.mkdir()
            return path

    with TemporaryDirectory() as tmp:
        old = organization_world.__wrapped__(Factory(Path(tmp)))
        with patch("llm_memory_editability.bios_cross.load_world", return_value=old):
            return make_cross_world(19)


def test_crossed_organization_preserves_facts_and_balances_positions(cross_world):
    w = cross_world
    assert audit(w)["passed"]
    expected = np.ones(w.n_base, dtype=int)
    expected[w.root_ids.ravel()] = 32
    for condition in CONDITIONS:
        d = documents(w, condition)
        np.testing.assert_array_equal(np.bincount(d.ravel(), minlength=w.n_base), expected)
        for chain, slot in enumerate([1, 8]):
            linked = d[:, slot] == w.root_ids[chain, w.memberships[chain]]
            assert linked.all() if condition == ("company", "project")[chain] else not linked.any()
        slots = np.zeros((w.n_base, 10), dtype=int)
        for epoch in range(10):
            a = epoch_documents(d, w.seed, epoch)
            np.add.at(slots, (a, np.arange(10)[None, :]), 1)
        np.testing.assert_array_equal(slots, np.repeat(expected[:, None], 10, axis=1))


def test_split_excludes_heldout_qa_and_balances_tasks(cross_world):
    w = cross_world
    schedule = qa_schedule(w, 1280)
    assert schedule.shape == (1280, 40)
    assert not np.isin(schedule, w.heldout_ids).any()
    for chain in range(2):
        assert np.isin(schedule[:, chain * 20 : (chain + 1) * 20], w.train_ids[chain]).all()
        counts = np.bincount(schedule.ravel(), minlength=len(w.answers))[w.train_ids[chain]]
        assert counts.max() - counts.min() <= 1
        np.testing.assert_array_equal(
            np.union1d(w.train_ids[chain], w.heldout_ids[chain]), w.derived_ids[chain]
        )


@pytest.mark.parametrize("chain", [0, 1])
def test_edits_are_paired_and_do_not_leak_propagation(cross_world, chain):
    w = cross_world
    p = edit_pair(w, chain)
    assert len(p["E"]) == 93 and len(p["D"]) == 96 and len(p["conflict_D"]) == 45
    for kind in ["coherent", "exception"]:
        target = p[kind]
        changed = np.flatnonzero(target != w.answers)
        np.testing.assert_array_equal(changed, np.union1d(p["E"], p["D"]))
        np.testing.assert_array_equal(
            target[w.derived_ids[1 - chain]], w.answers[w.derived_ids[1 - chain]]
        )
        assert not np.intersect1d(p["replay"], np.union1d(p["D"], p["heldout"])).size
        assert (p["replay"] < w.n_base).all()
        members = np.searchsorted(w.derived_ids[chain], p["D"])
        expected = target[w.root_ids[chain, w.memberships[chain, members]]]
        np.testing.assert_array_equal(target[p["D"]], expected)
    np.testing.assert_array_equal(np.sort(p["coherent"][p["E"]]), np.sort(p["exception"][p["E"]]))


def test_answer_generation_inputs_and_document_labels(cross_world):
    import torch

    from llm_memory_editability.bios_cross_train import tensor_queries

    w = cross_world
    d = documents(w, "company")[:3]
    batch = render(w, d)
    np.testing.assert_array_equal(
        batch["tokens"][np.arange(3)[:, None], batch["positions"] + 1], batch["labels"]
    )
    data = tensor_queries(w, torch.device("cpu"))
    rows = torch.arange(len(w.answers))
    assert torch.equal(data["tokens"][rows, data["lengths"]], data["labels"][:, 0])
    assert data["prompts"].shape == (len(w.answers), 5)


def test_damage_counts_do_not_include_preexisting_errors(cross_world):
    from llm_memory_editability.bios_cross_train import edit_metrics

    w = cross_world
    p = edit_pair(w, 0)
    old = np.ones(len(w.answers), dtype=bool)
    chosen = np.flatnonzero(p["strata"] == 1)[:2]
    old[chosen[0]] = False
    current = np.ones_like(old)
    current[chosen] = False
    metrics = edit_metrics(w, p, {"correct": current}, old)
    assert metrics["U_full"]["1"]["broken"] == 1
    assert metrics["E"] == metrics["D"] == 1
