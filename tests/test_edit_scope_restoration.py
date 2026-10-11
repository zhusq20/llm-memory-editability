"""Contracts for fixed saved-edit restoration, using real tiny GPT-2 blocks."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import GPT2Config

from llm_memory_editability import edit_scope_restoration as restoration
from llm_memory_editability.edit_scope_development import (
    CODE_FILES as NATIVE_CODE_FILES,
)
from llm_memory_editability.edit_scope_development import (
    _write_prediction,
    make_edit_model,
    parameter_digest,
    select_case_bank,
)
from llm_memory_editability.grok_depth import write_json
from llm_memory_editability.grokking_reproduction import ReproductionGPT, build_graph, encode_rows
from llm_memory_editability.path_overlap import World, sha256

PROJECT = Path(__file__).resolve().parents[1]


@pytest.fixture
def parent():
    torch.set_num_threads(2)
    torch.manual_seed(191)
    config = GPT2Config(
        vocab_size=97,
        n_positions=16,
        n_embd=16,
        n_layer=4,
        n_head=4,
        embd_pdrop=0,
        resid_pdrop=0,
        attn_pdrop=0,
        use_cache=False,
    )
    return ReproductionGPT(config, repeats=2).eval().requires_grad_(False)


@pytest.fixture
def metadata():
    return {"pad_token": 0, "entity_offset": 20, "relation_offset": 60, "end_marker": 90}


def joint_change(parent):
    joint = make_edit_model(parent, "joint")
    with torch.no_grad():
        for parameter in joint.parameters():
            if parameter.requires_grad:
                parameter.add_(torch.randn_like(parameter) * 0.015)
    selected = {name: p.detach().clone() for name, p in joint.named_parameters() if p.requires_grad}
    return joint.eval().requires_grad_(False), selected


@pytest.mark.parametrize("mode", restoration.MODES)
def test_literal_weights_and_parent_preservation(parent, mode):
    _, selected = joint_change(parent)
    original = {name: p.clone() for name, p in parent.named_parameters()}
    before = parameter_digest(parent)
    model = restoration.make_restoration_model(parent)
    restoration.apply_restoration(model, parent, selected, mode)
    for name, parameter in model.named_parameters():
        if name.startswith("transformer.h."):
            _, _, index, suffix = name.split(".", 3)
            source = f"transformer.h.{int(index) % 4}.{suffix}"
            expected = selected[source] if mode[int(index) // 4] == "1" else original[source]
        else:
            expected = original[name]
        torch.testing.assert_close(parameter, expected, atol=0, rtol=0)
        assert parameter.grad is None and not parameter.requires_grad
    assert parameter_digest(parent) == before
    assert (
        model.transformer.h[0].mlp.c_proj.weight.data_ptr()
        != model.transformer.h[4].mlp.c_proj.weight.data_ptr()
    )
    assert parent.repeats == 2 and len(parent.transformer.h) == 4


def test_native_baseline_and_joint_expansion_are_exact(parent, metadata):
    joint, selected = joint_change(parent)
    model = restoration.make_restoration_model(parent)
    for mode, reference in (("00", parent), ("11", joint)):
        restoration.apply_restoration(model, parent, selected, mode)
        for rows in (np.array([[2, 1, 7]]), np.array([[2, 1, 3, 7]])):
            x, positions, _ = map(torch.as_tensor, encode_rows(rows, metadata))
            with torch.no_grad():
                torch.testing.assert_close(
                    model(x, positions), reference(x, positions), atol=0, rtol=0
                )


def test_restore_modes_reset_previous_endpoint_and_reject_bad_states(parent):
    _, selected = joint_change(parent)
    model = restoration.make_restoration_model(parent)
    restoration.apply_restoration(model, parent, selected, "11")
    restoration.apply_restoration(model, parent, selected, "00")
    for index, block in enumerate(model.transformer.h):
        for name, p in block.named_parameters():
            torch.testing.assert_close(
                p, dict(parent.transformer.h[index % 4].named_parameters())[name], atol=0, rtol=0
            )
    with pytest.raises(ValueError, match="mode"):
        restoration.apply_restoration(model, parent, selected, "12")
    with pytest.raises(ValueError, match="parameter names"):
        restoration.apply_restoration(model, parent, {}, "11")
    wrong = dict(selected)
    key = next(iter(wrong))
    wrong[key] = wrong[key].flatten()[:1]
    with pytest.raises(ValueError, match="shape/dtype"):
        restoration.apply_restoration(model, parent, wrong, "11")


def test_exact_gate_catches_discrete_and_continuous_changes():
    raw = {"answer": np.array([31]), "margin": np.array([1.0], np.float32)}
    assert restoration.compare_predictions(raw, copy.deepcopy(raw))["passed"]
    changed = copy.deepcopy(raw)
    changed["margin"][0] = np.nextafter(np.float32(1.0), np.float32(2.0))
    assert restoration.compare_predictions(raw, changed)["mismatched"] == ["margin"]
    changed = copy.deepcopy(raw)
    changed["answer"][0] += 1
    assert not restoration.compare_predictions(raw, changed)["passed"]
    assert not restoration.compare_predictions(raw, {"answer": raw["answer"]})["passed"]


def test_paired_contrasts_use_fixed_denominators_and_case_mean():
    metrics = {
        mode: {"D": {"n": 8, "new_accuracy": value}}
        for mode, value in zip(restoration.MODES, (0.0, 0.8, 0.2, 0.3), strict=True)
    }
    contrasts = restoration.paired_contrasts(metrics)["D"]["new_accuracy"]
    assert contrasts["restore_first_loss"] == pytest.approx(0.5)
    assert contrasts["restore_second_loss"] == pytest.approx(0.6)
    assert contrasts["interaction"] == pytest.approx(0.3)
    other = copy.deepcopy(metrics)
    other["00"]["D"]["n"] = 2
    with pytest.raises(ValueError, match="pools changed"):
        restoration.paired_contrasts(other)
    second = copy.deepcopy(metrics)
    for mode in restoration.MODES:
        second[mode]["D"]["n"] = 1
        second[mode]["D"]["new_accuracy"] = 0.0
    records = [
        {"group": "id", "metrics": item, "contrasts": restoration.paired_contrasts(item)}
        for item in (metrics, second)
    ]
    summary = restoration.summarize_records(records)
    assert summary["all"]["modes"]["11"]["D"]["new_accuracy"] == pytest.approx(0.4)
    assert summary["ood"]["case_count"] == 0


class CorrectOracle:
    """Truth-selection fixture only; not evidence the random model knows facts."""

    def correct(self, kind, indices):
        return np.ones(len(indices), dtype=bool)


@pytest.fixture
def source_batch(tmp_path, parent, metadata, monkeypatch):
    root = tmp_path / "parent"
    data = root / "data"
    data.mkdir(parents=True)
    graph = build_graph(917, entities=40, relations=12, degree=8)
    np.savez(data / "world.npz", **graph)
    np.savez(data / "panels.npz")
    world_hash = sha256(data / "world.npz")
    write_json(
        data / "complete.json",
        {
            **metadata,
            "entities": 40,
            "relations": 12,
            "atomic_id": int((~graph["ood"]).sum()),
            "world_sha256": world_hash,
        },
    )
    world = World(data)
    checkpoint = root / "checkpoint.pt"
    checkpoint.write_bytes(b"immutable synthetic fixture checkpoint")
    batch = tmp_path / "source"
    config_path = batch / "frozen/config.json"
    config_path.parent.mkdir(parents=True)
    config = {"source": {"root": str(root), "run": "loop4x2-phi7.2-wd0.1", "step": 1500000}}
    write_json(config_path, config)
    bank = select_case_bank(
        world,
        "first",
        CorrectOracle(),
        {
            "seed": 71,
            "id_cases": 16,
            "ood_cases": 16,
            "affected_chains": 1,
            "unrelated_atoms": 1,
            "unrelated_chains": 1,
            "same_answer_atoms": 0,
            "replay_atoms": 1,
            "replay_chains": 1,
        },
    )
    assert len(bank["cases"]) == 32
    run = batch / "runs/first-joint"
    run.mkdir(parents=True)
    write_json(run / "case-bank.json", bank)
    native = {name: sha256(PROJECT / name) for name in NATIVE_CODE_FILES}
    hashes = {
        "config_sha256": sha256(config_path),
        "checkpoint_sha256": sha256(checkpoint),
        "world_sha256": world_hash,
        "case_bank_sha256": sha256(run / "case-bank.json"),
    }
    joint, selected = joint_change(parent)
    records = []
    for case in bank["cases"]:
        for rate in restoration.LEARNING_RATES:
            folder = run / "cases" / case["case_id"] / f"lr-{rate:.0e}"
            folder.mkdir(parents=True)
            baseline, prediction, state = (
                folder / "predictions-000.npz",
                folder / "predictions-100.npz",
                folder / "selected-100.pt",
            )
            _write_prediction(baseline, parent, world, case, torch.device("cpu"), "float32")
            _write_prediction(prediction, joint, world, case, torch.device("cpu"), "float32")
            torch.save(
                {
                    "selected": selected,
                    "scope": "joint",
                    "case_id": case["case_id"],
                    "learning_rate": rate,
                    "step": 100,
                    "frozen_parameter_sha256": restoration._frozen_native_digest(parent),
                    **hashes,
                },
                state,
            )
            records.append(
                {
                    "case_id": case["case_id"],
                    "step": 100,
                    "learning_rate": rate,
                    "selected_state_path": str(state.relative_to(run)),
                    "selected_state_sha256": sha256(state),
                    "prediction_path": str(prediction.relative_to(run)),
                    "prediction_sha256": sha256(prediction),
                }
            )
    write_json(
        run / "summary.json",
        {
            "arm": "first-joint",
            "case_count": 32,
            "records": records,
            "settings": {
                "learning_rates": list(restoration.LEARNING_RATES),
                "save_endpoints": [100],
                "precision": "float32",
            },
            "code_sha256": native,
            **hashes,
        },
    )
    write_json(run / "audit.json", {"passed": True, "checked": 64, "expected": 64})
    write_json(
        run / "complete.json",
        {
            "complete": True,
            "independent_audit_complete": True,
            "summary_sha256": sha256(run / "summary.json"),
            "audit_sha256": sha256(run / "audit.json"),
        },
    )
    monkeypatch.setattr(
        restoration,
        "load_model",
        lambda *args: (parent, {"phi": 7.2, "weight_decay": 0.1}, checkpoint),
    )
    return batch


def test_saved_state_smoke_and_independent_replay_have_zero_updates(
    source_batch, tmp_path, monkeypatch
):
    def forbidden(*args, **kwargs):
        raise AssertionError("Restoration cannot optimize or select cases")

    monkeypatch.setattr(torch.optim, "AdamW", forbidden)
    monkeypatch.setattr(torch.Tensor, "backward", forbidden)
    out, audit = tmp_path / "readout", tmp_path / "audit"
    out.mkdir()
    (out / "science-process.json").write_text("{}")
    (out / "tracking.json").write_text("{}")
    result = restoration.run_shard(source_batch, "first", 1e-5, out, "cpu", limit=1)
    assert result["case_count"] == 1 and result["smoke"]
    assert result["optimization_updates"] == 0
    assert result["mode_readouts"] == 4
    assert set(result["records"][0]["metrics"]) == set(restoration.MODES)
    assert all(result["records"][0]["gates"][mode]["passed"] for mode in ("00", "11"))
    replay = restoration.audit_shard(out, audit, "cpu")
    assert replay["aggregates"] == result["aggregates"]
    assert json.loads((audit / "audit.json").read_text())["passed"]
    assert json.loads((out / "complete.json").read_text())["passed"]
    assert json.loads((audit / "complete.json").read_text())["passed"]
    with pytest.raises(RuntimeError, match="scientific attempt"):
        restoration.run_shard(source_batch, "first", 1e-5, out, "cpu", limit=1)


def test_failed_native_gate_stops_before_every_hybrid(source_batch, tmp_path, monkeypatch):
    monkeypatch.setattr(
        restoration,
        "compare_predictions",
        lambda *args: {"passed": False, "mismatched": ["answer"]},
    )
    out = tmp_path / "failure"
    with pytest.raises(RuntimeError, match="native replay gate"):
        restoration.run_shard(source_batch, "first", 1e-4, out, "cpu", limit=1)
    assert (out / "failure.json").exists()
    assert not list(out.rglob("predictions-10.npz"))
    assert not list(out.rglob("predictions-01.npz"))
    assert not (out / "complete.json").exists()


def test_source_shards_keep_full_bank_and_detect_state_mutation(source_batch):
    _, _, endpoints, _ = restoration._source_manifest(source_batch, "first", 1e-5)
    assert len(endpoints) == 32
    assert sum(endpoint["case"]["group"] == "id" for endpoint in endpoints) == 16
    Path(endpoints[0]["paths"]["selected"]).write_bytes(b"corrupted saved weights")
    with pytest.raises(ValueError, match="state or raw native"):
        restoration._source_manifest(source_batch, "first", 1e-5)


def test_full_shard_gates_every_endpoint_before_hybrids(source_batch, tmp_path, monkeypatch):
    native = restoration._write_prediction
    modes = []

    def log_readout(path, *args, **kwargs):
        modes.append(Path(path).stem.removeprefix("predictions-"))
        return native(path, *args, **kwargs)

    monkeypatch.setattr(restoration, "_write_prediction", log_readout)
    result = restoration.run_shard(source_batch, "first", 1e-4, tmp_path / "full", "cpu")
    assert result["case_count"] == 32 and not result["smoke"]
    assert modes[:64] == ["00", "11"] * 32
    assert modes[64:] == ["10", "01"] * 32
    assert result["aggregates"]["id"]["case_count"] == 16
    assert result["aggregates"]["ood"]["case_count"] == 16
