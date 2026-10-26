"""Continual panels, current truth, fact weighting, isolation and full replay."""

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import GPT2Config

from llm_memory_editability import ttcd_continual_eval as ce
from llm_memory_editability.supervised_alignment import SupervisedAlignmentGPT
from llm_memory_editability.ttcd_editing import TTCDReader, tensor_digest


@pytest.fixture(scope="module", autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def experiment(tmp_path):
    entities, relations = 12, 2
    atoms = np.array(
        [[h, r, (h + r + 1) % entities] for h in range(entities) for r in range(relations)]
    )
    first, second, chains = [], [], []
    for i, (h, r, b) in enumerate(atoms):
        for r2 in range(relations):
            j = int(b * relations + r2)
            first.append(i)
            second.append(j)
            chains.append([h, r, r2, atoms[j, -1]])
    world = SimpleNamespace(
        atoms=atoms,
        chains=np.asarray(chains),
        first=np.asarray(first),
        second=np.asarray(second),
        entities=entities,
        eo=3,
        ro=entities + 3,
        metadata={
            "entity_offset": 3,
            "relation_offset": entities + 3,
            "end_marker": 2,
            "pad_token": 0,
            "entities": entities,
            "relations": relations,
            "vocab_size": entities + relations + 3,
        },
    )
    world.support = lambda _: np.zeros(len(world.chains), dtype=bool)
    bank = []
    for split, ids in (("calibration", [17, 21]), ("evaluation", [1, 5, 9, 13])):
        for i, atom in enumerate(ids):
            bank.append(
                {
                    "case_id": f"{split}-{atom}",
                    "split": split,
                    "group": "id" if i % 2 == 0 else "ood",
                    "atom": atom,
                    "new_tail": int((atoms[atom, -1] + 3) % entities),
                    "D": np.flatnonzero(world.second == atom).tolist(),
                }
            )
    (tmp_path / "training-split.json").write_text(json.dumps({"case_bank": bank}))
    config = {
        "data_dir": str(tmp_path),
        "phi": 2.0,
        "continual": {
            "train_length": 2,
            "eval_length": 4,
            "rotations": 4,
            "affected": 2,
            "controls": 2,
            "order_seed": 1729,
        },
        "writer": {
            "teacher_window": 32,
            "student_window": 16,
            "gap": 17,
            "steps": 2,
            "learning_rate": 0.001,
        },
        "reading_gate": {"minimum_cases_per_group": 1, "minimum_accuracy": 0.0},
    }
    torch.manual_seed(19)
    cfg = GPT2Config(
        vocab_size=world.metadata["vocab_size"],
        n_positions=40,
        n_embd=8,
        n_head=2,
        n_layer=2,
        resid_pdrop=0,
        embd_pdrop=0,
        attn_pdrop=0,
    )
    reader = TTCDReader(SupervisedAlignmentGPT(cfg, repeats=2, discoloop=True)).eval()
    return config, world, reader


def test_frozen_panel_covers_all_facts_positions_and_excludes_all_keys(experiment):
    config, world, _ = experiment
    panel = ce.build_panel(config, world)
    assert panel == ce.build_panel(config, world)
    edited = panel["edited_atoms"]
    assert set(edited) == {c["atom"] for c in ce.case_bank(config) if c["split"] == "evaluation"}
    assert np.all(np.sort(panel["orders"], axis=0) == np.arange(4)[:, None])
    for fact in panel["facts"]:
        d = fact["D"]
        assert np.all(world.second[d] == fact["atom"])
        assert not np.isin(world.first[d], edited).any()
        assert not world.support(config["phi"])[d].any()
        assert fact["revision_tail"] not in (fact["old_tail"], fact["new_tail"])
    assert not np.isin(panel["U_atoms"], edited).any()
    assert not np.isin(world.first[panel["U_chains"]], edited).any()
    assert not np.isin(world.second[panel["U_chains"]], edited).any()


def test_panel_failure_does_not_drop_a_fact_or_silently_crop_controls(experiment):
    config, world, _ = experiment
    changed = copy.deepcopy(config)
    changed["continual"]["eval_length"] = 3
    changed["continual"]["rotations"] = 3
    with pytest.raises(ValueError, match="every distinct"):
        ce.build_panel(changed, world)
    changed = copy.deepcopy(config)
    changed["continual"]["controls"] = 1000
    with pytest.raises(ValueError, match="Insufficient U"):
        ce.build_panel(changed, world)
    world.first[:] = 1
    with pytest.raises(ValueError, match="No eligible D"):
        ce.build_panel(config, world)


def test_preflight_panel_must_equal_reconstructed_graph_selection(experiment, tmp_path):
    config, world, _ = experiment
    panel = ce.build_panel(config, world)
    path = tmp_path / "panel.json"
    path.write_text(json.dumps(panel))
    config["continual"]["panel_path"] = str(path)
    assert ce._panel(config, world) == panel
    panel["facts"][0]["new_tail"] = 11
    path.write_text(json.dumps(panel))
    with pytest.raises(ValueError, match="preflight"):
        ce._panel(config, world)


def test_revision_truth_and_manifest_include_every_intermediate_state(experiment):
    config, world, _ = experiment
    panel = ce.build_panel(config, world)
    plan = ce.state_plan(config, panel)
    assert len(plan) == 1 + 4 + 4 * 4 + 4 * 2
    for r, order in enumerate(panel["orders"]):
        before = plan[f"main/r{r:03d}/t002"]
        first = plan[f"revision/r{r:03d}/t003"]
        last = plan[f"revision/r{r:03d}/t004"]
        assert first["indices"] == before["indices"]
        assert first["targets"][0] == panel["facts"][order[0]]["revision_tail"]
        assert first["targets"][1:] == before["targets"][1:]
        assert last["indices"] == order[:3]
        assert last["targets"][1:] == [panel["facts"][i]["new_tail"] for i in order[1:3]]


def test_full_contract_saves_1313_states_but_reads_only_revision_endpoints(experiment):
    config, world, _ = experiment
    panel = ce.build_panel(config, world)
    panel["facts"] = panel["facts"] * 8
    panel["orders"] = [list(range(i, 32)) + list(range(i)) for i in range(32)]
    config["continual"]["train_length"] = 8
    plan = ce.state_plan(config, panel)
    assert len(plan) == 1313
    assert sum(spec["read"] for spec in plan.values()) == 1121
    branch = [s for s in plan.values() if s["kind"] == "revision" and s["rotation"] == 0]
    assert [s["t"] for s in branch if s["read"]] == [9, 16]
    assert len(branch[-1]["indices"]) == 15


def test_reads_are_stateless_and_joint_answer_EOS_is_preserved(experiment):
    config, world, reader = experiment
    panel = ce.build_panel(config, world)
    base = tensor_digest(reader.backbone.state_dict())
    fast, _ = ce.write_fact(reader, reader.empty_fast(), config, world, panel["facts"][0], "cpu")
    digest = tensor_digest(fast)
    targets = [f["new_tail"] for f in panel["facts"][:2]]
    before = ce.read_state(reader, fast, world, panel, [0, 1], targets, "cpu")
    ce.read_state(reader, fast, world, panel, [2], [panel["facts"][2]["new_tail"]], "cpu")
    after = ce.read_state(reader, fast, world, panel, [0, 1], targets, "cpu")
    assert before == after
    assert tensor_digest(fast) == digest
    assert tensor_digest(reader.backbone.state_dict()) == base
    assert before["metrics"]["D_fact_equal"] == np.mean(
        [p["D"]["new_accuracy"] for p in before["per_fact"]]
    )
    for pool, p in before["predictions"].items():
        expected = np.mean(
            (np.asarray(p["answer"]) == np.asarray(p["new_answers"]) + world.eo)
            & (np.asarray(p["stop"]) == world.metadata["end_marker"])
        )
        assert before["metrics"][pool]["new_accuracy"] == expected


def test_full_evaluation_and_reload_enforce_complete_matrix(experiment, monkeypatch, tmp_path):
    config, world, reader = experiment
    out = tmp_path / "evaluation"
    monkeypatch.setattr(ce, "_load_reader", lambda *_: (copy.deepcopy(reader), world))
    monkeypatch.setattr(ce, "fingerprint", lambda *_: {"engineering_fixture": True})
    calibration = ce.calibrate(config, "single", out, "cpu")
    assert calibration["passed"]
    assert not calibration["formal_D_used"]
    summary = ce.evaluate(config, "single", out, "cpu")
    assert summary["states"] == 29
    assert summary["readout_states"] == 29
    assert summary["primary_t"] == 2
    assert summary["parameters"]["base_trainable"] == 0
    audit = ce.audit(config, "single", out, "cpu")
    assert audit["passed"] and audit["answer_and_EOS_exact"]
    assert audit["states"] == 29
    path = out / "states/main/r000/t002/record.json"
    record = json.loads(path.read_text())
    record["readout"]["predictions"]["D"]["stop"][0] += 1
    path.write_text(json.dumps(record))
    with pytest.raises(AssertionError, match="token/label"):
        ce.audit(config, "single", out, "cpu")
    path.unlink()
    with pytest.raises(ValueError, match="state matrix"):
        ce.audit(config, "single", out, "cpu")


def test_primary_weights_facts_equally_when_D_counts_differ():
    # Two facts occur in each of two repeated orders: fact 1 has eight D,
    # fact 2 only one. Its success must still carry half the primary weight.
    pred = {"answer": [3], "stop": [2], "old_answers": [0]}
    baseline = {"predictions": {p: pred for p in ("U_atoms", "U_chains")}}
    records = []
    for rotation in range(2):
        facts = [
            {"atom": 1, "E": {"new_accuracy": 1.0}, "D": {"new_accuracy": 0.0}},
            {"atom": 2, "E": {"new_accuracy": 1.0}, "D": {"new_accuracy": 1.0}},
        ]
        for t in (1, 2):
            records.append(
                {
                    "spec": {"kind": "main", "t": t, "rotation": rotation},
                    "readout": {
                        "per_fact": facts if t == 2 else facts[rotation : rotation + 1],
                        "metrics": {p: {"old_accuracy": 1.0} for p in ("U_atoms", "U_chains")},
                        "predictions": baseline["predictions"],
                    },
                }
            )
    result = ce.summarize(
        {"continual": {"train_length": 2}},
        {"facts": [{}, {}]},
        records,
        baseline,
        {"end_marker": 2, "entity_offset": 3},
    )
    assert result["primary"]["D"] == 0.5
