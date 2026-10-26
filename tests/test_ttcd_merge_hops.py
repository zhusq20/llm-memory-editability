"""Independent-delta arithmetic, both hop labels, reuse and complete state replay."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from test_ttcd_architecture_eval import make_reader
from test_ttcd_continual_eval import experiment as material

from llm_memory_editability import ttcd_continual_eval as original
from llm_memory_editability import ttcd_merge_hops as merge
from llm_memory_editability.ttcd_editing import tensor_digest
from llm_memory_editability.ttcd_merge_hop_panel import build_panel

experiment = material
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module", autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def settings():
    return json.loads((ROOT / "configs/ttcd-merge-hops-v1.json").read_text())


def test_independent_merge_sums_deltas_without_mutating_donors_or_averaging():
    zero = {0: torch.zeros(2, 2), 1: torch.zeros(2, 2)}
    first = {0: torch.tensor([[1.0, 2.0], [3.0, 4.0]]), 1: torch.ones(2, 2)}
    second = {0: torch.tensor([[5.0, 6.0], [7.0, 8.0]]), 1: torch.ones(2, 2) * 3}
    before = [tensor_digest(value) for value in (zero, first, second)]
    result = merge.add_increment(merge.add_increment(zero, first), second)
    torch.testing.assert_close(result[0], torch.tensor([[6.0, 8.0], [10.0, 12.0]]), rtol=0, atol=0)
    torch.testing.assert_close(result[1], torch.ones(2, 2) * 4, rtol=0, atol=0)
    assert before == [tensor_digest(value) for value in (zero, first, second)]
    assert result[0].data_ptr() not in (
        zero[0].data_ptr(),
        first[0].data_ptr(),
        second[0].data_ptr(),
    )
    with pytest.raises(ValueError, match="keys"):
        merge.add_increment(zero, {0: first[0]})
    with pytest.raises(ValueError, match="shape/dtype/device"):
        merge.add_increment(zero, {0: first[0].double(), 1: first[1]})


def test_two_methods_and_hop_roles_have_one_shared_fast_state_matrix():
    panel = {"facts": [{}, {}], "orders": [[0, 1], [1, 0]]}
    plan = merge.state_plan(panel)
    assert len(plan) == 11
    assert sum(p["kind"] == "independent_merge" for p in plan.values()) == 4
    assert plan["continuous/r001/t002"]["parent_key"] == "main/r001/t002"
    assert plan["independent_merge/r001/t002"]["parent_key"] is None
    assert plan["continuous/r001/t002"]["indices"] == plan["independent_merge/r001/t002"]["indices"]


@pytest.mark.parametrize("architecture", ("d8", "loop4x2", "disco4x2"))
def test_both_roles_full_merge_and_reused_continuous_states_replay(
    architecture, experiment, settings, tmp_path, monkeypatch
):
    config, world, _ = experiment
    config = copy.deepcopy(config)
    config["writer"]["steps"] = 1
    reader = make_reader(architecture, vocab_size=world.metadata["vocab_size"], dtype=torch.float32)
    world.lookup = np.full((world.entities, world.metadata["relations"]), -1, dtype=np.int64)
    world.lookup[world.atoms[:, 0], world.atoms[:, 1]] = np.arange(len(world.atoms))
    base_panel = original.build_panel(config, world)
    panel = build_panel(base_panel, world, settings, config["phi"])
    source, out = tmp_path / "parent", tmp_path / "new"
    source.mkdir()
    out.mkdir()

    def loader(*_):
        return reader, world

    def identity(*_):
        return {"engineering_only": True, "writer_steps": 1}

    kwargs = {"reader_loader": loader, "identity_factory": identity}
    original.calibrate(config, "stream", source, "cpu", **kwargs)
    original.evaluate(config, "stream", source, "cpu", **kwargs)
    audit = original.audit(config, "stream", source, "cpu", **kwargs)
    (source / "result-audit.json").write_text(json.dumps(audit))
    (source / "result-cloud-audit.json").write_text(json.dumps({"engineering_only": True}))
    (out / "panel.json").write_text(json.dumps(panel))
    parent_hashes = {
        str(p.relative_to(source)): p.read_bytes() for p in source.rglob("*") if p.is_file()
    }
    monkeypatch.setattr(merge, "source_context", lambda *_: (reader, world, config, panel))
    result = merge.evaluate(config, settings, architecture, source, out, "cpu")
    assert result["states"] == 37 and result["new_merged_states"] == 16
    assert result["reused_one_step_donors"] == 4
    assert not list((out / "states/continuous").rglob("fast.pt"))
    assert len(list((out / "states/independent_merge").rglob("fast.pt"))) == 16
    assert result["first_hop_coverage"]["facts_total"] == 4
    for method in ("continuous", "independent_merge"):
        assert result["E_by_age"][method]["0"]["all"]["n"] == 16
        assert set(result["roles"]["second"][method]) == {"1", "2", "3", "4"}
        assert (
            result["roles"]["first"][method]["4"]["all"]["n"]
            == 4 * panel["first_hop_coverage"]["queries_total"]
        )
    checked = merge.audit(config, settings, architecture, source, out, "cpu")
    assert checked["passed"] and checked["independent_donors_rewritten"] == 4
    assert parent_hashes == {
        str(p.relative_to(source)): p.read_bytes() for p in source.rglob("*") if p.is_file()
    }
    for spec in merge.state_plan(panel).values():
        if spec["kind"] in ("baseline", "single", "continuous"):
            assert spec["parent_key"] is not None
    if architecture == "d8":
        directory = out / "states/independent_merge/r000/t002"
        payload = torch.load(directory / "fast.pt", weights_only=True)
        payload["fast"] = {k: v / 2 for k, v in payload["fast"].items()}
        payload["fast_digest"] = tensor_digest(payload["fast"])
        torch.save(payload, directory / "fast.pt")
        with pytest.raises(AssertionError, match="declared sum"):
            merge.audit(config, settings, architecture, source, out, "cpu")


def test_baseline_condition_and_fact_equal_readout_have_explicit_denominators():
    rows = []
    for fact, baseline, after in ((1, True, True), (1, True, False), (2, False, True)):
        rows.append(
            {
                "fact": fact,
                "baseline_correct": baseline,
                "new_correct": after,
                "baseline_other_correct": True,
                "other_correct": True,
                "edited_atom_correct": True,
                "old_answer": not after,
            }
        )
    result = merge.summarize_rows(rows)
    assert result["baseline_correct"]["n"] == 2
    assert result["baseline_correct"]["accuracy"] == 0.5
    assert result["baseline_correct"]["coverage"] == 2 / 3
    assert result["all"]["fact_equal_accuracy"] == 0.75
    assert merge.summarize_rows([])["baseline_correct"]["accuracy"] is None


def test_source_state_requires_one_update_and_matching_provenance(experiment, tmp_path):
    _, _, reader = experiment
    from llm_memory_editability.ttcd_editing import save_fast

    directory = tmp_path / "states/single/f000"
    directory.mkdir(parents=True)
    spec = {"kind": "single"}
    fast = reader.empty_fast()
    save_fast(directory / "fast.pt", reader, fast, spec)
    record = {
        "key": "single/f000",
        "spec": spec,
        "fast_digest": tensor_digest(fast),
        "write_history": {"updates": [{}, {}]},
    }
    (directory / "record.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="one update"):
        merge.source_state(reader, tmp_path, "single/f000")


def test_method_difference_pairs_the_same_questions_and_orders():
    before = [
        {"query": [1, 2, 3, 4], "rotation": 0, "baseline_correct": True, "new_correct": False},
        {"query": [1, 2, 3, 4], "rotation": 1, "baseline_correct": True, "new_correct": True},
        {"query": [2, 3, 4, 5], "rotation": 0, "baseline_correct": False, "new_correct": True},
    ]
    after = copy.deepcopy(before)
    after[0]["new_correct"] = True
    after[2]["new_correct"] = False
    result = merge.paired_difference(before, after)
    assert result["all"]["merge_minus_continuous_pp"] == 0
    assert result["baseline_correct"]["merge_minus_continuous_pp"] == 50
    assert result["all"]["improved"] == result["all"]["regressed"] == 1
    with pytest.raises(ValueError, match="identical"):
        merge.paired_difference(before, after[:2])
    assert merge.paired_difference([], [])["all"]["merge_minus_continuous_pp"] is None
