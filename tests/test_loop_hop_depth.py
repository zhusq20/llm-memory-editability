"""Scientific contracts for truth, isolation, depth, scoring and resource admission."""

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from llm_memory_editability.grokking_reproduction import MergedBatchStream, model_digest
from llm_memory_editability.loop_hop_data import (
    exposure_summary,
    load,
    make_table,
    pack_rows,
    path_details,
    prepare,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
runner = importlib.import_module("run_loop_hop_depth")
native = runner.native
launcher = importlib.import_module("launch_loop_hop_depth")


def configuration():
    value = json.loads((ROOT / "configs/loop-hop-unified-v1.json").read_text())
    value["data"].update(
        entities_per_graph=8,
        relations=4,
        train_compositions_per_hop=24,
        evaluation_templates=2,
        training_panel_size=8,
    )
    value["model"].update(n_embd=16, n_head=2, resid_pdrop=0, attn_pdrop=0, embd_pdrop=0)
    return value


@pytest.fixture
def task(tmp_path):
    config = configuration()
    prepare(config, tmp_path / "task")
    return config, load(tmp_path / "task")


def test_all_targets_follow_only_atomic_facts_and_graph_b_has_no_composites(task):
    _, (world, panels, metadata) = task
    n = metadata["entities_per_graph"]
    facts = {(int(h), int(r)): int(t) for h, r, t in world["atoms"]}
    for name, rows in {**panels, **{f"full_{k}": world[f"train_{k}"] for k in (2, 3, 4)}}.items():
        for row in rows:
            entity = int(row[0])
            for relation in row[1:-1]:
                entity = facts[entity, int(relation)]
            assert entity == row[-1], name
    for k in (2, 3, 4):
        assert np.all(world[f"train_{k}"][:, [0, -1]] < n)
        assert not {tuple(r[:-1]) for r in world[f"train_{k}"]} & {
            tuple(r[:-1]) for r in panels[f"id_{k}"]
        }


def test_relation_sequences_cannot_predict_endpoint_without_start(task):
    _, (_, panels, metadata) = task
    n = metadata["entities_per_graph"]
    for k in (2, 3, 4):
        for split, graph in (("id", 0), ("ood", 1)):
            rows = panels[f"{split}_{k}"]
            for start in range(0, len(rows), n):
                selected = rows[start : start + n]
                np.testing.assert_array_equal(np.sort(selected[:, -1]), np.arange(n) + graph * n)
                assert len(set(map(tuple, selected[:, 1:-1]))) == 1


def test_padding_and_supervision_only_answer_eos(task):
    _, (world, panels, metadata) = task
    for p in ("atomic_id", "id_2", "id_3", "id_4"):
        rows = panels[p]
        tokens, positions, labels = pack_rows(rows, metadata)
        k = rows.shape[1] - 2
        assert positions[0].tolist() == [k, k + 1]
        assert np.all(tokens[:, k + 2 :] == metadata["pad_token"])
        np.testing.assert_array_equal(labels[:, 0], rows[:, -1] + metadata["entity_offset"])
        assert np.all(labels[:, 1] == metadata["end_marker"])
    table = make_table(world, metadata, "cpu")
    assert table[2][-1].tolist() == [-100, -100]
    assert len(table[0]) == metadata["training_examples"] + 1


@pytest.mark.parametrize(
    "name",
    [
        "loop-l1-r2",
        "disco-l1-r2",
        "loop-l1-r3",
        "disco-l1-r3",
        "loop-l1-r4",
        "disco-l1-r4",
        "loop-l2-r2",
        "disco-l2-r2",
    ],
)
def test_real_layer_count_and_causal_answer_without_future_leak(task, name):
    config, (_, panels, metadata) = task
    model, spec, _, _ = native.initialize(config, name, metadata, torch.device("cpu"))
    model.eval()
    counts = {"attention": 0, "mlp": 0, "channel": 0}
    handles = []

    def increment(key):
        def hook(*_):
            counts[key] += 1

        return hook

    for block in model.transformer.h:
        handles.append(block.attn.c_attn.register_forward_hook(increment("attention")))
        handles.append(block.mlp.register_forward_hook(increment("mlp")))
    boundary = model._discoloop_boundary

    def record_boundary(state):
        counts["channel"] += 1
        return boundary(state)

    model._discoloop_boundary = record_boundary
    tokens, positions, _ = pack_rows(panels["id_4"][:2], metadata)
    x, pos = torch.as_tensor(tokens), torch.as_tensor(positions)
    with torch.no_grad():
        answer = model(x, pos)[:, 0]
    assert counts["attention"] == counts["mlp"] == spec["unique_layers"] * spec["repeats"]
    assert counts["channel"] == (spec["repeats"] - 1 if name.startswith("disco") else 0)
    for handle in handles:
        handle.remove()
    changed = x.clone()
    changed[:, 5] = (changed[:, 5] + 1) % metadata["vocab_size"]
    with torch.no_grad():
        torch.testing.assert_close(model(changed, pos)[:, 0], answer, atol=0, rtol=0)


def test_paired_weights_and_independent_layers(task):
    config, (_, _, metadata) = task
    hashes = []
    for r in (2, 3, 4):
        for arch in ("loop", "disco"):
            model, _, _, initial = native.initialize(
                config, f"{arch}-l1-r{r}", metadata, torch.device("cpu")
            )
            hashes.append(initial)
    assert len(set(hashes)) == 1
    model, _, _, _ = native.initialize(config, "loop-l2-r2", metadata, torch.device("cpu"))
    assert not torch.equal(
        model.transformer.h[0].mlp.c_fc.weight, model.transformer.h[1].mlp.c_fc.weight
    )


def test_sweep_restores_training_horizon_weights_and_rng(task):
    config, (world, panels, metadata) = task
    model, _, _, _ = native.initialize(config, "disco-l1-r3", metadata, torch.device("cpu"))
    before, rng = model_digest(model), torch.get_rng_state()
    metrics, predictions = runner.sweep(model, world, panels, metadata, torch.device("cpu"), 32)
    assert set(metrics) == set(predictions) == {"1", "2", "3", "4"}
    assert model.repeats == 3 and model.training
    assert model_digest(model) == before and torch.equal(rng, torch.get_rng_state())
    assert all("premise_coverage" in metrics[r]["ood_4"] for r in metrics)


def test_answer_at_one_loop_is_identical_with_discrete_channel_disabled(task):
    config, (_, panels, metadata) = task
    plain, _, _, _ = native.initialize(config, "loop-l1-r2", metadata, torch.device("cpu"))
    disco, _, _, _ = native.initialize(config, "disco-l1-r2", metadata, torch.device("cpu"))
    plain.repeats = disco.repeats = 1
    plain.eval()
    disco.eval()
    tokens, positions, _ = pack_rows(panels["id_3"][:2], metadata)
    x, pos = torch.as_tensor(tokens), torch.as_tensor(positions)
    with torch.no_grad():
        torch.testing.assert_close(plain(x, pos), disco(x, pos), rtol=0, atol=0)


def test_premise_indices_and_masked_exposure_accounting(task):
    config, (world, panels, metadata) = task
    nodes, edges = path_details(panels["ood_4"], metadata, world["permutations"])
    np.testing.assert_array_equal(world["atoms"][edges, -1], nodes[:, 1:])
    count = metadata["training_examples"]
    stream, exposures = MergedBatchStream(count, 19), np.zeros(count, dtype=np.int64)
    for _ in range(7):
        indices, _ = stream.batch(17)
        exposures += np.bincount(indices, minlength=count + 1)[:count]
    report = exposure_summary(exposures, world, metadata, 7, 17)
    assert report["examples"] + report["masked_batch_slots"] == 7 * 17
    assert report["supervised_tokens"] == 2 * report["examples"]
    launcher.validate(config)


def test_pending_foreign_gpu_reservation_blocks_admission_without_cuda_process():
    rows = {7: {"uuid": "GPU-seven"}, 8: {"uuid": "GPU-eight"}}
    containers = [
        {
            "Name": "/old-evaluation-queue",
            "HostConfig": {"DeviceRequests": [{"DeviceIDs": ["GPU-seven"]}]},
        }
    ]
    blocked = launcher.blocked_gpus(rows, [("GPU-eight", "123")], containers)
    assert blocked[7] == ["/old-evaluation-queue"]
    assert blocked[8] == ["active CUDA process 123"]


def test_only_own_container_processes_may_share_the_selected_gpu():
    rows = {9: {"uuid": "GPU-nine"}}
    own = {
        "Name": "/loop-hop-unified-v1-loop-l1-r2-train",
        "State": {"Pid": 111},
        "HostConfig": {"DeviceRequests": [{"DeviceIDs": ["GPU-nine"]}]},
    }
    blocked = launcher.blocked_gpus(rows, [("GPU-nine", "111")], [own], "loop-hop-unified-v1-")
    assert blocked[9] == []
    blocked = launcher.blocked_gpus(rows, [("GPU-nine", "222")], [own], "loop-hop-unified-v1-")
    assert blocked[9] == ["active CUDA process 222"]


def test_sweep_restores_horizon_after_failure(task, monkeypatch):
    config, (world, panels, metadata) = task
    model, _, _, _ = native.initialize(config, "loop-l1-r2", metadata, torch.device("cpu"))
    monkeypatch.setattr(
        native, "measure_panels", lambda *args: (_ for _ in ()).throw(ValueError("test"))
    )
    with pytest.raises(ValueError, match="test"):
        runner.sweep(model, world, panels, metadata, torch.device("cpu"), 32)
    assert model.repeats == 2 and model.training
