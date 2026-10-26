"""One-step budget, preserved parent identity and second-hop/retention denominators."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from test_ttcd_architecture_eval import make_reader
from test_ttcd_continual_eval import experiment as material

from llm_memory_editability import ttcd_continual_eval as evaluation
from llm_memory_editability import ttcd_one_step as one
from llm_memory_editability.ttcd_editing import tensor_digest
from llm_memory_editability.ttcd_experiment import packet_for

ROOT = Path(__file__).resolve().parents[1]

experiment = material


@pytest.fixture
def settings():
    return json.loads((ROOT / "configs/ttcd-one-step-second-hop-v1.json").read_text())


@pytest.fixture(scope="module", autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_parent_config_is_unchanged_and_only_eval_budget_changes(settings):
    parent = {"writer": {"steps": 4, "learning_rate": 0.001}, "training": {"steps": 10000}}
    before = copy.deepcopy(parent)
    view = one.evaluation_view(parent, settings)
    assert parent == before
    assert view["writer"]["steps"] == 1
    view["writer"]["steps"] = 4
    assert view == parent
    with pytest.raises(ValueError, match="actual unchanged parent"):
        one.evaluation_view({"writer": {"steps": 1}}, settings)


def test_loader_checks_the_original_training_identity(settings, monkeypatch):
    parent = {"writer": {"steps": 4}, "architecture": {"name": "loop"}}
    seen = []
    engine = SimpleNamespace(load_reader=lambda config, phase, device: seen.append(config))
    monkeypatch.setattr(one, "parent_engine", lambda _: engine)
    view, loader, _ = one.interfaces(parent, settings, "loop")
    loader(view, "stream", "cpu")
    assert seen == [parent] and seen[0] is parent
    assert view["writer"]["steps"] == 1 and parent["writer"]["steps"] == 4
    with pytest.raises(ValueError, match="fixed stream"):
        loader(view, "common", "cpu")


@pytest.mark.parametrize("architecture", ("d8", "loop4x2", "disco4x2"))
def test_each_fact_is_exactly_one_local_update(architecture, experiment):
    config, world, _ = experiment
    config = copy.deepcopy(config)
    config["writer"]["steps"] = 1
    reader = make_reader(architecture)
    before = tensor_digest(reader.backbone.state_dict())
    case = evaluation.case_bank(config)[0]
    fast, history = evaluation.write_fact(reader, reader.empty_fast(), config, world, case, "cpu")
    packet = torch.tensor([packet_for(config, world, case)])
    direct, _ = reader.write(
        packet,
        reader.empty_fast(),
        teacher_window=32,
        student_window=16,
        learning_rate=config["writer"]["learning_rate"],
    )
    assert len(history["updates"]) == 1
    assert tensor_digest(fast) == tensor_digest(direct)
    assert tensor_digest(reader.backbone.state_dict()) == before
    assert not any(p.requires_grad for p in reader.parameters())


def test_panel_rejects_first_hop_editing_and_trained_compositions():
    world = SimpleNamespace(
        first=np.array([40]), second=np.array([1]), support=lambda _: np.array([False])
    )
    panel = {"edited_atoms": list(range(32)), "orders": [[]] * 32, "facts": [{"atom": 1, "D": [0]}]}
    one.validate_panel(panel, world)
    world.first[0] = 1
    with pytest.raises(ValueError, match="only the second hop"):
        one.validate_panel(panel, world)
    world.first[0] = 40
    world.support = lambda _: np.array([True])
    with pytest.raises(ValueError, match="untrained chain"):
        one.validate_panel(panel, world)


def test_entity_offsets_eos_and_empty_condition_are_scored_explicitly():
    pred = {"answer": [8, 9], "stop": [2, 1], "new_answers": [5, 6]}
    assert one.scored(pred, {"entity_offset": 3, "end_marker": 2}) == [True, False]
    result = one.summarize_rows(
        [
            {
                "new_correct": True,
                "baseline_correct": False,
                "baseline_first_correct": True,
                "first_correct": True,
                "edited_atom_correct": True,
                "old_answer": False,
            }
        ]
    )
    assert result["all"] == {"n": 1, "correct": 1, "accuracy": 1.0}
    assert result["baseline_correct"]["n"] == 0
    assert result["baseline_correct"]["accuracy"] is None
    assert result["baseline_correct"]["coverage"] == 0


def test_baseline_correct_condition_does_not_count_preexisting_failures():
    rows = []
    for baseline, after in ((True, True), (True, False), (False, True), (False, True)):
        rows.append(
            {
                "new_correct": after,
                "baseline_correct": baseline,
                "baseline_first_correct": True,
                "first_correct": True,
                "edited_atom_correct": True,
                "old_answer": not after,
            }
        )
    report = one.summarize_rows(rows)
    assert report["all"]["accuracy"] == 0.75
    assert report["baseline_correct"] == {"n": 2, "correct": 1, "accuracy": 0.5, "coverage": 0.5}


def test_complete_one_step_states_and_secondary_analysis_replay(experiment, tmp_path):
    config, world, reader = experiment
    config = copy.deepcopy(config)
    config["writer"]["steps"] = 1

    def loader(*_):
        return reader, world

    def identity(*_):
        return {"engineering": True, "writer_steps": 1}

    kwargs = {"reader_loader": loader, "identity_factory": identity}
    out = tmp_path / "evaluation"
    out.mkdir()
    evaluation.calibrate(config, "stream", out, "cpu", **kwargs)
    summary = evaluation.evaluate(config, "stream", out, "cpu", **kwargs)
    first, paired = one.analyze(out, world.metadata)
    records = one.read_records(out)
    for record in records:
        if record["write_history"] is not None:
            assert len(record["write_history"]["updates"]) == 1
    assert len(records) == summary["states"]
    assert first["single"]["D"]["all"]["n"] == len(paired["baseline"])
    assert set(first["continuous"]) == {"1", "2", "3", "4"}
    assert len(paired["continuous_final"]) == 4 * len(paired["baseline"])
    assert first["by_age"]["0"]["previously_successful_E"]["n"] <= 16
    evaluation.audit(config, "stream", out, "cpu", **kwargs)
    second, repeated = one.analyze(out, world.metadata)
    assert first == second and paired == repeated


def test_cross_architecture_common_subset_keeps_zero_coverage(tmp_path):
    paths = {}
    for name in ("transformer", "loop", "discoloop"):
        out = tmp_path / name
        out.mkdir()
        data = {
            "baseline": [
                {"query": [1, 2, 3, 4], "correct": name != "transformer", "first_correct": True}
            ],
            "single": [],
            "continuous_final": [],
        }
        (out / "paired-queries.json").write_text(json.dumps(data))
        paths[name] = out
    result = one.cross_architecture_report(paths, tmp_path / "aggregate")
    assert result["common_baseline_correct"] == 0 and result["coverage"] == 0
    assert result["architectures"]["loop"]["single"]["all"]["accuracy"] is None
    data["baseline"][0]["query"][0] = 99
    (paths["discoloop"] / "paired-queries.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="universes differ"):
        one.cross_architecture_report(paths, tmp_path / "bad")
