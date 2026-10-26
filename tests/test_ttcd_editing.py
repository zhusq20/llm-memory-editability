"""Scientific contracts: local gradient, causal writes, tied reads and isolation."""

import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import GPT2Config

from llm_memory_editability.supervised_alignment import SupervisedAlignmentGPT
from llm_memory_editability.ttcd_editing import (
    TTCDReader,
    build_case_bank,
    correction_packet,
    load_fast,
    local_ttcd_update,
    save_fast,
    tensor_digest,
)


@pytest.fixture(scope="module", autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def reader():
    torch.manual_seed(15)
    cfg = GPT2Config(
        vocab_size=29,
        n_positions=40,
        n_embd=16,
        n_head=2,
        n_layer=2,
        resid_pdrop=0,
        embd_pdrop=0,
        attn_pdrop=0,
    )
    return TTCDReader(SupervisedAlignmentGPT(cfg, repeats=2, discoloop=True).double()).eval()


def test_local_rule_matches_stopped_target_autograd():
    torch.manual_seed(3)
    z = torch.randn(3, 7, dtype=torch.double)
    zt = torch.randn_like(z)
    w = torch.randn(7, 5, dtype=torch.double, requires_grad=True)
    teacher = zt @ w
    student = z @ w
    loss = (student - teacher.detach()).square().sum() / (2 * len(z))
    gradient = torch.autograd.grad(loss, w)[0]
    actual = local_ttcd_update(z, teacher, student, 0.03)
    torch.testing.assert_close(actual, -0.03 * gradient)
    assert not actual.requires_grad
    assert torch.count_nonzero(local_ttcd_update(z, teacher, student, 0)) == 0


def test_native_causal_opt_in_preserves_ordinary_and_windowed_computation(reader, monkeypatch):
    calls = []
    original = torch.nn.functional.scaled_dot_product_attention

    def observe(*args, **kwargs):
        calls.append((kwargs.get("attn_mask") is None, kwargs.get("is_causal", False)))
        return original(*args, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", observe)
    tokens = torch.tensor([[2, 7, 8, 4], [3, 8, 4, 1]])
    legacy = reader(tokens)
    assert calls and all(not no_mask and not causal for no_mask, causal in calls)
    bounded = reader(tokens, window=2)
    reader.native_causal_attention = True
    calls.clear()
    observed = reader(tokens)
    assert calls and all(no_mask and causal for no_mask, causal in calls)
    torch.testing.assert_close(observed, legacy, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(observed, reader.backbone(tokens), rtol=0, atol=0)
    calls.clear()
    torch.testing.assert_close(reader(tokens, window=2), bounded, rtol=0, atol=0)
    assert calls and all(not no_mask and not causal for no_mask, causal in calls)


@pytest.mark.parametrize("disco", [False, True])
def test_zero_state_matches_native_and_materialized_update(reader, disco):
    reader.backbone.discoloop = disco
    tokens = torch.tensor([[2, 7, 8, 4], [3, 8, 4, 1]])
    original = tensor_digest(reader.backbone.state_dict())
    fast = reader.empty_fast()
    torch.testing.assert_close(
        reader(tokens, fast=fast), reader.backbone(tokens), rtol=1e-10, atol=1e-10
    )
    for weight in fast.values():
        weight.normal_(std=0.002)
    materialized = copy.deepcopy(reader.backbone)
    with torch.no_grad():
        for layer, block in enumerate(materialized.transformer.h):
            block.mlp.c_proj.weight.add_(fast[layer])
    torch.testing.assert_close(
        reader(tokens, fast=fast), materialized(tokens), rtol=1e-10, atol=1e-10
    )
    assert tensor_digest(reader.backbone.state_dict()) == original
    assert all(not p.requires_grad for p in reader.parameters())


def test_capture_is_causal_and_first_round_only(reader):
    x = torch.tensor([[1, 2, 3, 4, 5, 6, 7]])
    fast = reader.empty_fast()
    capture = {"student_window": 2, "positions": torch.tensor([4])}
    reader(x, fast=fast, window=6, capture=capture)
    changed = x.clone()
    changed[:, 5:] = torch.tensor([[19, 20]])
    other = {"student_window": 2, "positions": torch.tensor([4])}
    reader(changed, fast=fast, window=6, capture=other)
    assert len(capture["features"]) == len(reader.backbone.transformer.h)
    for layer in capture["features"]:
        for left, right in zip(capture["features"][layer], other["features"][layer], strict=True):
            torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_commit_uses_old_state_once_and_shared_layer_updates(reader):
    packet = torch.tensor([[1, 2, 3, 4, 5, 6, 7]])
    fast = reader.empty_fast()
    before = tensor_digest(fast)
    parent = tensor_digest(reader.backbone.state_dict())
    capture = {"student_window": 2, "positions": torch.tensor([6])}
    reader(packet, fast=fast, window=7, capture=capture)
    actual, info = reader.write(
        packet, fast, teacher_window=7, student_window=2, learning_rate=0.001
    )
    for layer, features in capture["features"].items():
        torch.testing.assert_close(actual[layer], local_ttcd_update(*features, 0.001))
    assert info["layers_written"] == 2  # Not doubled for two loop executions.
    assert tensor_digest(fast) == before
    assert tensor_digest(reader.backbone.state_dict()) == parent


def test_record_outside_student_changes_teacher_not_first_layer_student_key(reader):
    x = torch.tensor([[1, 2, 3, 4, 5, 6, 7]])
    altered = x.clone()
    altered[:, 2] = 20
    captures = []
    for packet in (x, altered):
        c = {"student_window": 2, "positions": torch.tensor([6])}
        reader(packet, window=7, capture=c)
        captures.append(c["features"][0])
    torch.testing.assert_close(captures[0][0], captures[1][0], atol=0, rtol=0)
    assert not torch.equal(captures[0][1], captures[1][1])


def test_fast_reload_is_identical_and_read_order_cannot_mutate_it(reader, tmp_path):
    packet = torch.tensor([[1, 2, 3, 4, 5, 6, 7]])
    fast, _ = reader.write(
        packet, reader.empty_fast(), teacher_window=7, student_window=2, learning_rate=0.001
    )
    path = tmp_path / "fast.pt"
    save_fast(path, reader, fast, {"case": "engineering"})
    restored, meta = load_fast(path, reader)
    assert meta == {"case": "engineering"}
    state_hash = tensor_digest(restored)
    expected = reader(packet[:, :3], fast=restored)
    reader(packet[:, 2:5], fast=restored)
    actual = reader(packet[:, :3], fast=restored)
    torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    assert tensor_digest(restored) == state_hash
    with torch.no_grad():
        next(reader.backbone.parameters()).add_(0.1)
    with pytest.raises(ValueError, match="different backbone"):
        load_fast(path, reader)


def test_packet_excludes_answer_after_query_and_checks_visibility():
    metadata = {
        "entities": 8,
        "entity_offset": 3,
        "relation_offset": 11,
        "end_marker": 2,
        "pad_token": 0,
    }
    packet = correction_packet(
        [1, 2, 3], 5, [1, 2], metadata, gap=17, teacher_window=32, student_window=16
    )
    assert packet[:4] == [4, 13, 8, 2]
    assert packet[-2:] == [4, 13]
    assert len(packet) == 23
    assert packet[4:-2] == [0] * 17
    with pytest.raises(ValueError, match="visible"):
        correction_packet(
            [1, 2, 3], 5, [1, 2], metadata, gap=0, teacher_window=32, student_window=16
        )
    with pytest.raises(ValueError, match="different entity"):
        correction_packet(
            [1, 2, 3], 3, [1, 2], metadata, gap=17, teacher_window=32, student_window=16
        )


def toy_world():
    atoms = np.array([[0, 0, 1], [1, 1, 2], [2, 0, 3], [3, 1, 0], [4, 0, 1], [5, 0, 3]])
    chains = np.array(
        [[0, 0, 1, 2], [4, 0, 1, 2], [2, 0, 1, 0], [5, 0, 1, 0], [1, 1, 0, 3], [3, 1, 0, 1]]
    )
    world = SimpleNamespace(
        atoms=atoms,
        chains=chains,
        first=np.array([0, 4, 2, 5, 1, 3]),
        second=np.array([1, 1, 3, 3, 2, 0]),
        entities=6,
        ood=np.array([False, False, True, True, False, False]),
    )
    world.support = lambda _: np.zeros(len(chains), dtype=bool)
    return world


def test_bank_is_fixed_disjoint_and_held_out_without_outcome_selection():
    world = toy_world()
    kwargs = dict(seed=3, calibration_cases=1, evaluation_cases=1, affected=2, controls=2, phi=7.2)
    bank = build_case_bank(world, **kwargs)
    assert bank == build_case_bank(world, **kwargs)
    calibration = {c["atom"] for c in bank if c["split"] == "calibration"}
    evaluation = {c["atom"] for c in bank if c["split"] == "evaluation"}
    assert not calibration & evaluation
    for case in bank:
        assert np.all(world.second[case["D"]] == case["atom"])
        assert not world.support(7.2)[case["D"]].any()
        assert case["new_tail"] != world.atoms[case["atom"], 2]
        assert not np.isin(world.second[case["U_chains"]], case["atom"]).any()
        assert not np.isin(world.first[case["U_chains"]], case["atom"]).any()


@pytest.mark.parametrize("rate", [-1, float("inf"), float("nan")])
def test_invalid_learning_rates_fail(rate):
    with pytest.raises(ValueError):
        local_ttcd_update(torch.ones(1, 2), torch.ones(1, 3), torch.ones(1, 3), rate)
