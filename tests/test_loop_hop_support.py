"""Contracts for nested support, matched exposure, depth and baseline reuse."""

import copy
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from llm_memory_editability.grokking_reproduction import MergedBatchStream
from llm_memory_editability.loop_hop_data import (
    audit_data,
    exposure_summary,
    load,
    make_table,
    pack_rows,
    prepare,
    sha256,
)
from llm_memory_editability.loop_hop_support import (
    MatchedSupportStream,
    expand_support,
    matched_exposure_summary,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
runner = importlib.import_module("run_loop_hop_support")
launcher = importlib.import_module("launch_loop_hop_support")
torch.set_num_threads(1)


@pytest.fixture
def task(tmp_path):
    config = json.loads((ROOT / "configs/loop-hop-depth-support-v1.json").read_text())
    config["data"].update(
        entities_per_graph=8,
        relations=4,
        train_compositions_per_hop=24,
        evaluation_templates=2,
        training_panel_size=8,
    )
    config["model"].update(
        n_embd=16, n_head=2, n_positions=16, n_ctx=16, attn_pdrop=0, resid_pdrop=0, embd_pdrop=0
    )
    config["supports"] = {"base": 24, "more": 72}
    prepare(config, tmp_path / "base")
    expand_support(tmp_path / "base", tmp_path / "more", multiplier=3, seed=71)
    low, high = load(tmp_path / "base"), load(tmp_path / "more")
    config["parent"]["panels_sha256"] = low[2]["panels_sha256"]
    return config, tmp_path, low, high


def test_support_adds_only_unique_nonreserved_A_queries_and_preserves_test_bytes(task):
    _, root, low, high = task
    low_world, _, metadata = low
    world, panels, high_meta = high
    assert sha256(root / "base/panels.npz") == sha256(root / "more/panels.npz")
    for key in ("atoms", "permutations"):
        np.testing.assert_array_equal(world[key], low_world[key])
    for hop in (2, 3, 4):
        np.testing.assert_array_equal(world[f"train_{hop}"][:24], low_world[f"train_{hop}"])
        assert len(world[f"train_{hop}"]) == 72
        assert np.all(world[f"train_{hop}"][:, [0, -1]] < metadata["entities_per_graph"])
    assert audit_data(world, panels, high_meta)["passed"]


def test_every_batch_preserves_atoms_categories_partial_masks_and_supervised_tokens(task):
    _, _, low, high = task
    lw, _, lm = low
    hw, _, hm = high
    atoms, support = len(lw["atoms"]), lm["train_compositions_per_hop"]
    small = MergedBatchStream(lm["training_examples"], 19)
    large = MatchedSupportStream(
        hm["training_examples"], 19, atoms=atoms, base_support=support, multiplier=3
    )
    counts = [np.zeros(m["training_examples"], dtype=np.int64) for m in (lm, hm)]
    tables = [make_table(w, m, "cpu") for w, m in ((lw, lm), (hw, hm))]
    for step in range(1, 89):
        a, na = small.batch(19)
        b, nb = large.batch(19)
        assert na == nb
        np.testing.assert_array_equal(a < atoms, b < atoms)
        np.testing.assert_array_equal(a[a < atoms], b[b < atoms])
        np.testing.assert_array_equal(
            (a[:na] - atoms) // support,
            np.where(
                b[:nb] < atoms, (b[:nb] - atoms) // support, (b[:nb] - atoms) // (3 * support)
            ),
        )
        if step <= 8:
            for left, right in zip(tables[0], tables[1], strict=True):
                torch.testing.assert_close(left[a], right[b], rtol=0, atol=0)
        for target, indices in zip(counts, (a, b), strict=True):
            target += np.bincount(indices, minlength=len(target) + 1)[: len(target)]
        x = exposure_summary(counts[0], lw, lm, step, 19)
        y = matched_exposure_summary(counts[1], hw, hm, step, 19)
        for key in (
            "examples",
            "counts",
            "effective_input_tokens",
            "supervised_tokens",
            "executed_input_tokens",
            "masked_batch_slots",
        ):
            assert x[key] == y[key]
    for category in range(3):
        low_counts = counts[0][atoms + category * support : atoms + (category + 1) * support]
        high_counts = counts[1][
            atoms + category * support * 3 : atoms + (category + 1) * support * 3
        ]
        np.testing.assert_array_equal(low_counts, high_counts.reshape(3, support).sum(0))


def test_stream_restoration_includes_support_phase(task):
    _, _, low, high = task
    stream = MatchedSupportStream(
        high[2]["training_examples"], 83, atoms=len(low[0]["atoms"]), base_support=24, multiplier=3
    )
    for _ in range(19):
        stream.batch(19)
    saved = copy.deepcopy(stream.state_dict())
    expected = [stream.batch(19)[0] for _ in range(13)]
    restored = MatchedSupportStream(
        high[2]["training_examples"], 1, atoms=len(low[0]["atoms"]), base_support=24, multiplier=3
    )
    restored.load_state_dict(saved)
    for wanted in expected:
        np.testing.assert_array_equal(restored.batch(19)[0], wanted)


@pytest.mark.parametrize(
    "name", list(json.loads((ROOT / "configs/loop-hop-depth-support-v1.json").read_text())["arms"])
)
def test_all_structures_execute_real_layers_channels_and_preserve_causality(task, name):
    config, _, (_, panels, metadata), _ = task
    model, spec, arm, _ = runner.core.native.initialize(config, name, metadata, torch.device("cpu"))
    model.eval()
    counts = {"attention": 0, "mlp": 0, "channel": 0}
    handles = []

    def count(key):
        def hook(*_):
            counts[key] += 1

        return hook

    for block in model.transformer.h:
        handles += [
            block.attn.c_attn.register_forward_hook(count("attention")),
            block.mlp.register_forward_hook(count("mlp")),
        ]
    boundary = model._discoloop_boundary

    def channel(x):
        counts["channel"] += 1
        return boundary(x)

    model._discoloop_boundary = channel
    tokens, pos, _ = pack_rows(panels["id_4"][:2], metadata)
    x, positions = torch.from_numpy(tokens), torch.from_numpy(pos)
    with torch.no_grad():
        before = model(x, positions)[:, 0]
    assert counts["attention"] == counts["mlp"] == spec["unique_layers"] * spec["repeats"]
    assert counts["channel"] == (spec["repeats"] - 1 if arm["discoloop"] else 0)
    for handle in handles:
        handle.remove()
    x[:, 5] = (x[:, 5] + 1) % metadata["vocab_size"]
    with torch.no_grad():
        torch.testing.assert_close(before, model(x, positions)[:, 0], rtol=0, atol=0)


def test_same_layers_share_initial_weights_across_architecture_repeats_support(task):
    config, _, (_, _, metadata), _ = task
    hashes = {2: set(), 3: set()}
    for name, spec in config["arms"].items():
        model, _, _, initial = runner.core.native.initialize(
            config, name, metadata, torch.device("cpu")
        )
        hashes[spec["unique_layers"]].add(initial)
        assert not torch.equal(
            model.transformer.h[0].mlp.c_fc.weight, model.transformer.h[1].mlp.c_fc.weight
        )
    assert all(len(values) == 1 for values in hashes.values())


def test_sampling_patch_is_scoped_and_rejects_wrong_mounted_condition(task):
    config, root, _, _ = task
    original = runner.core.native.MergedBatchStream
    config["input_root"] = str(root / "more")
    with runner.sampling(config, "loop-l2-r2-more"):
        assert runner.core.native.MergedBatchStream is not original
    assert runner.core.native.MergedBatchStream is original
    with pytest.raises(ValueError, match="Mounted support"):
        with runner.sampling(config, "loop-l2-r3-base"):
            pass
    assert runner.core.native.MergedBatchStream is original


def test_only_requested_structures_and_four_support_controls_are_added():
    config = json.loads((ROOT / "configs/loop-hop-depth-support-v1.json").read_text())
    launcher.validate(config)
    assert len(config["arms"]) == 14
    assert "loop-l2-r2-base" not in config["arms"]
    assert len([a for a in config["arms"].values() if a["data_condition"] == "more"]) == 4
    changed = copy.deepcopy(config)
    changed["runtime"]["gpu_candidates"] = [8, 9]
    with pytest.raises(ValueError):
        launcher.validate(changed)


def test_full_frozen_inputs_and_reused_L2_initialization_when_mounted():
    if not Path("/expanded-inputs").exists():
        pytest.skip("Full-size mounted data are checked by the frozen queued controller")
    low, high = load("/base-inputs"), load("/expanded-inputs")
    assert low[2]["training_examples"] == 40000
    assert high[2]["training_examples"] == 100000
    assert low[2]["panels_sha256"] == high[2]["panels_sha256"]
    assert audit_data(*high)["passed"]
    for k in (2, 3, 4):
        np.testing.assert_array_equal(high[0][f"train_{k}"][:10000], low[0][f"train_{k}"])
    config = json.loads((ROOT / "configs/loop-hop-depth-support-v1.json").read_text())
    _, _, _, initial = runner.core.native.initialize(
        config, "loop-l2-r3-base", low[2], torch.device("cpu")
    )
    parent = json.loads(Path("/parent/runs/loop-l2-r2/run.json").read_text())
    assert initial == parent["initial_model_sha256"]


@pytest.mark.parametrize("reason", ("foreign", "capacity"))
def test_foreign_gpu_reservation_and_eight_member_cap_block_admission(
    tmp_path, monkeypatch, reason
):
    config = json.loads((ROOT / "configs/loop-hop-depth-support-v1.json").read_text())
    controller = launcher.Controller(config, tmp_path)
    rows = {9: {"uuid": "GPU-nine", "used": 20000, "free": 70000, "utilization": 100}}
    blocked = {9: ["foreign reservation"] if reason == "foreign" else []}
    if reason == "capacity":
        controller.leases[9] = {f"member-{i}" for i in range(8)}
    monkeypatch.setattr(launcher, "snapshot", lambda *_: (rows, blocked))

    def yield_wait(*_):
        raise RuntimeError("waiting without admission")

    monkeypatch.setattr(launcher.time, "sleep", yield_wait)
    name = next(iter(config["arms"]))
    with pytest.raises(RuntimeError, match="without admission"):
        controller.acquire(name)
    assert name not in controller.leases.get(9, set())


def test_only_registered_own_members_can_fill_the_last_gpu_slot(tmp_path, monkeypatch):
    config = json.loads((ROOT / "configs/loop-hop-depth-support-v1.json").read_text())
    controller = launcher.Controller(config, tmp_path)
    controller.leases[9] = {f"member-{i}" for i in range(7)}
    rows = {9: {"uuid": "GPU-nine", "used": 20000, "free": 70000, "utilization": 100}}
    monkeypatch.setattr(launcher, "snapshot", lambda *_: (rows, {9: []}))
    name = next(iter(config["arms"]))
    assert controller.acquire(name) == (9, "GPU-nine")
    assert len(controller.leases[9]) == 8
    assert controller.state["batch"] == "loop-hop-depth-support-v1"
