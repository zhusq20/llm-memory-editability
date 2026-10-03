"""Scientific contracts for nested support, paired compute and complete scoring."""

import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from llm_memory_editability.grok_depth import EpochStream
from llm_memory_editability.multihop_scaling import (
    COMMON_SPLITS,
    EVALUATION_SPLITS,
    SOURCE_FILES,
    TRAIN_SPLITS,
    _validate_output,
    audit,
    audit_world,
    build_world,
    common_evaluation_digest,
    construct,
    data_digest,
    evaluate,
    run_name,
    train,
    truth_path_details,
)
from llm_memory_editability.storage_composition import file_hash


@pytest.fixture(scope="module")
def spec():
    return {
        "world_seed": 761011,
        "initialization": 762011,
        "stream_seed": 763011,
        "width": 32,
        "heads": 4,
        "layers": 2,
        "repeats": 1,
        "phi": 1.0,
        "batch_size": 128,
        "steps": 64000,
        "nodes": [0, 256, 512, 1000, 2000, 4000, 8000, 16000, 32000, 64000],
        "checkpoint_nodes": [0, 8000, 32000, 64000],
        "lr": 0.001,
        "weight_decay": 0.01,
        "warmup": 200,
        "schedule": "cosine",
        "min_lr_ratio": 0.1,
        "clip": 1.0,
    }


@pytest.fixture(scope="module")
def pair(spec):
    return build_world(spec), build_world(dict(spec, phi=4.0))


def executor():
    module_spec = importlib.util.spec_from_file_location(
        "execute_multihop_scaling", Path("scripts/execute_multihop_scaling.py")
    )
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def test_low_support_nested_and_tests_graph_ids_ood_identical(pair):
    low, high = pair
    assert len(low["atomic"]) == 256
    assert low["metadata"]["id_mask"] == high["metadata"]["id_mask"]
    assert sum(low["metadata"]["id_mask"]) == 192
    for name in COMMON_SPLITS:
        np.testing.assert_array_equal(low[name], high[name])
    for hops, high_size in ((2, 472), (3, 768), (4, 768)):
        name = f"train_{hops}"
        assert len(low[name]) == 192 and len(high[name]) == high_size
        np.testing.assert_array_equal(low[name], high[name][:192])
    assert common_evaluation_digest(low) == common_evaluation_digest(high)
    assert data_digest(low) != data_digest(high)
    assert audit_world(low)["same_atomic_id_ood_and_heldout_pools_across_support"]
    assert high["metadata"]["composition_support"]["2"]["capped"]


def test_strict_facts_have_no_composition_role_at_any_hop_and_support(pair):
    for world in pair:
        trained = set()
        for hop in (2, 3, 4):
            trained.update(truth_path_details(world, world[f"train_{hop}"])[1].ravel())
        for hop in (2, 3, 4):
            strict = truth_path_details(world, world[f"strict_{hop}"])[1]
            assert not set(strict.ravel()) & trained
        familiar_queries = {
            hop: set(map(tuple, world[f"familiar_{hop}"][:, :-1])) for hop in (2, 3, 4)
        }
        for hop in (2, 3, 4):
            assert not set(map(tuple, world[f"train_{hop}"][:, :-1])) & familiar_queries[hop]


@pytest.mark.parametrize("change", ["support_order", "test_pool", "truth", "id"])
def test_corrupted_support_or_common_pools_rejected(pair, change):
    world = copy.deepcopy(pair[0])
    if change == "support_order":
        world["train_3"][[0, 1]] = world["train_3"][[1, 0]]
    elif change == "test_pool":
        world["familiar_2"] = world["familiar_2"][1:]
    elif change == "truth":
        world["strict_4"][0, -1] = 3 + (int(world["strict_4"][0, -1]) - 2) % 64
    else:
        world["metadata"]["id_mask"][0] = not world["metadata"]["id_mask"][0]
    with pytest.raises(ValueError):
        audit_world(world)


def test_frozen_data_and_common_pool_hashes_enforced(spec, pair):
    world = pair[0]
    frozen = dict(
        spec,
        frozen_data_sha256=data_digest(world),
        common_evaluation_sha256=common_evaluation_digest(world),
    )
    assert data_digest(build_world(frozen)) == data_digest(world)
    with pytest.raises(ValueError, match="frozen"):
        build_world(dict(frozen, frozen_data_sha256="invalid"))
    with pytest.raises(ValueError, match="evaluation"):
        build_world(dict(frozen, common_evaluation_sha256="invalid"))


def test_same_atomic_stream_positions_hop_counts_and_total_tokens(spec, pair):
    draws = []
    for world in pair:
        streams = [
            EpochStream(len(world[name]), spec["stream_seed"] + i)
            for i, name in enumerate(TRAIN_SPLITS)
        ]
        batches = [stream.take(32 * 128).reshape(128, 32) for stream in streams]
        draws.append(batches)
        assert np.concatenate(batches, axis=1).shape == (128, 128)
        for batch, name in zip(batches, TRAIN_SPLITS, strict=True):
            count = np.bincount(batch.ravel(), minlength=len(world[name]))
            assert count.sum() == 4096 and np.ptp(count) <= 1
    np.testing.assert_array_equal(draws[0][0], draws[1][0])
    assert not np.array_equal(draws[0][2], draws[1][2])


def test_initial_weights_shared_by_support_and_match_first_block_at_equal_compute(spec):
    torch.set_num_threads(1)
    for depth in (2, 4):
        standard = construct(dict(spec, layers=depth, repeats=1), "cpu")
        loop = construct(dict(spec, layers=1, repeats=depth), "cpu")
        high = construct(dict(spec, layers=depth, repeats=1, phi=4.0), "cpu")
        for key, value in standard.state_dict().items():
            torch.testing.assert_close(value, high.state_dict()[key], rtol=0, atol=0)
        for key, value in standard.blocks[0].state_dict().items():
            torch.testing.assert_close(value, loop.blocks[0].state_dict()[key], rtol=0, atol=0)
        for key in ("token.weight", "position.weight", "ln_final.weight", "ln_final.bias"):
            torch.testing.assert_close(
                standard.state_dict()[key], loop.state_dict()[key], rtol=0, atol=0
            )
        assert loop.effective_depth == standard.effective_depth == depth
        assert all(parameter.requires_grad for parameter in loop.parameters())
        assert sum(p.numel() for p in standard.parameters()) > sum(
            p.numel() for p in loop.parameters()
        )


def test_complete_predefined_matrix_and_world_level_seed_pairing():
    module = executor()
    dev, confirmation = module.specifications(), module.specifications("confirmation")
    assert len(dev) == 16 and len(confirmation) == 96
    assert len({run_name(spec) for spec in dev + confirmation}) == 112
    assert {s["world_seed"] for s in dev} == {761011}
    assert {s["initialization"] for s in dev} == {762011}
    assert {s["stream_seed"] for s in dev} == {763011}
    assert {s["world_seed"] for s in confirmation} == {761101, 761102, 761103}
    assert {s["initialization"] for s in confirmation} == {762101, 762102}
    for world in (761101, 761102, 761103):
        paired = [spec for spec in confirmation if spec["world_seed"] == world]
        assert len(paired) == 32 and {spec["stream_seed"] for spec in paired} == {world + 2000}
    assert sum(s["steps"] for s in dev) == 1_024_000
    assert sum(s["steps"] for s in confirmation) == 6_144_000


class TruthOracle(torch.nn.Module):
    def __init__(self, world, wrong_eos=False):
        super().__init__()
        self.world, self.wrong_eos = world, wrong_eos
        self.lookup = {
            (int(head), int(relation)): int(tail) for head, relation, tail in world["atomic"]
        }

    def forward(self, tokens):
        logits = torch.full((*tokens.shape, self.world["metadata"]["vocab_size"]), -20.0)
        for i, row in enumerate(tokens.tolist()):
            if row[-1] == self.world["metadata"]["separator_token"]:
                result = row[1]
                for relation in row[2:-1]:
                    result = self.lookup.get((result, relation), 3)
            else:
                result = 0 if self.wrong_eos else 1
            logits[i, -1, result] = 20.0
        return logits


def test_full_generation_atomic_premises_and_autonomous_feedback(pair):
    for world in pair:
        metrics, predictions = evaluate(TruthOracle(world), world, "cpu")
        assert set(metrics) == set(EVALUATION_SPLITS)
        for name in EVALUATION_SPLITS:
            assert metrics[name]["accuracy"] == 1.0
            assert predictions[name + "_correct"].all()
        bad_metrics, _ = evaluate(TruthOracle(world, wrong_eos=True), world, "cpu")
        assert bad_metrics["atomic"]["answer_accuracy"] == 1.0
        assert bad_metrics["atomic"]["accuracy"] == 0.0
        assert bad_metrics["familiar_4"]["conditional_accuracy"] is None
        assert bad_metrics["familiar_4"]["autonomous_two_calls"] == 0.0


def test_output_artifacts_and_scheduler_no_overwrite(tmp_path):
    (tmp_path / "input-spec.json").write_text("{}")
    (tmp_path / "train-process.log").write_text("log")
    _validate_output(tmp_path)
    existing = tmp_path / "model-000000.pt"
    existing.write_bytes(b"original")
    with pytest.raises(FileExistsError):
        _validate_output(tmp_path)
    assert existing.read_bytes() == b"original"
    with pytest.raises(FileExistsError):
        executor().run_one({}, {}, tmp_path, "cpu")


def test_cpu_engineering_training_independent_reload_and_exact_exposure(tmp_path, spec):
    spec = dict(
        spec,
        entities=8,
        relations=2,
        degree=2,
        width=16,
        steps=2,
        nodes=[0, 1, 2],
        checkpoint_nodes=[0, 2],
        warmup=1,
    )
    world = build_world(spec)
    spec["frozen_data_sha256"] = data_digest(world)
    spec["common_evaluation_sha256"] = common_evaluation_digest(world)
    source = {path: file_hash(path) for path in SOURCE_FILES}
    result = train(spec, tmp_path, source, "cpu")
    assert result["supervised_tokens"] == 2 * 832
    assert result["padded_input_tokens"] == 2 * 1024
    assert result["environment"]["device"] == "cpu"
    assert audit(tmp_path, "cpu")["all_nodes_exposure_counters_exact"]
    with pytest.raises(FileExistsError):
        train(spec, tmp_path, source, "cpu")
    with pytest.raises(FileExistsError):
        audit(tmp_path, "cpu")
