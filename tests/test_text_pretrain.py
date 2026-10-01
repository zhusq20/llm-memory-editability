"""Scientific contracts for the paired text-pretraining comparison."""

import json
from pathlib import Path

import numpy as np
import torch

from llm_memory_editability.text_pretrain import (
    atomic_sentence,
    build_world,
    composite_sentence,
    construct,
    pack_documents,
    training_table,
)


def spec():
    return json.loads(Path("configs/text-pretrain-development-v1.json").read_text())["base"]


def test_global_pair_holdout_and_composition_truth():
    world = build_world(spec())
    train = np.concatenate([world["background_train"], world["target_train"]])
    test = np.concatenate([world["background_test"], world["target_test"]])
    assert not {tuple(r[[0, 4]]) for r in train} & {tuple(r[[0, 4]]) for r in test}
    lookup = {(h, r): t for h, r, t in world["atomic"]}
    for h, r1, b, r2, t in world["all_paths"]:
        assert lookup[h, r1] == b
        assert lookup[b, r2] == t


def test_p0_p1_exact_tokens_targets_and_positions_only_attention_changes():
    world = build_world(spec())
    a, bounds = training_table(world, "P0")
    b, bounds_b = training_table(world, "P1")
    c, bounds_c = training_table(world, "P2")
    assert bounds == bounds_b == bounds_c
    for column in (0, 1, 3):
        np.testing.assert_array_equal(a[column], b[column])
    np.testing.assert_array_equal(a[2][: bounds[2]], b[2][: bounds[2]])
    assert not np.array_equal(a[2][bounds[2] :], b[2][bounds[2] :])
    for column in (0, 1, 3):
        np.testing.assert_array_equal(b[column][:, :16], c[column][:, :16])
    for t in (a, b, c):
        assert (t[3][: bounds[1]] != -100).sum(1).tolist() == [7] * bounds[1]
        assert (t[3][bounds[1] : bounds[2]] != -100).sum(1).tolist() == [9] * (
            bounds[2] - bounds[1]
        )
        assert (t[3][bounds[2] :] != -100).sum(1).tolist() == [23] * (bounds[3] - bounds[2])


def test_document_mask_causal_and_neutral_or_conclusion_isolated():
    world = build_world(spec())
    for arm in ("P0", "P1", "P2"):
        table, bounds = training_table(world, arm)
        mask = table[2][bounds[2], 0]
        assert not np.triu(mask, 1).any()
        assert not mask[16:, :16].any()
        assert mask[8:16, :8].any() == (arm != "P0")
        assert table[3][bounds[2], 7] == -100
        assert table[3][bounds[2], 15] == -100


def test_p0_second_fact_has_no_causal_access_to_first_fact():
    cfg = dict(spec(), width=16, heads=2, layers=1, repeats=1, dropout=0.0)
    model = construct(cfg, "cpu").eval()
    world = build_world(cfg)
    a, bounds = training_table(world, "P0")
    x, pos, mask, _ = [torch.as_tensor(v[bounds[2] : bounds[2] + 1]) for v in a]
    changed = x.clone()
    changed[0, 1] = int(world["atomic_first"][8, 0])
    with torch.no_grad():
        left = model(x, pos, mask)
        right = model(changed, pos, mask)
    torch.testing.assert_close(left[:, 8:16], right[:, 8:16], rtol=0, atol=0)
    assert not torch.equal(left[:, :8], right[:, :8])


def test_full_loss_has_prompt_targets_and_autoregressive_final_answer():
    world = build_world(spec())
    row = world["target_train"][0]
    sentence = composite_sentence(row)
    assert sentence[-3:] == [row[-1], 5, 1]
    x, _pos, _mask, y = pack_documents([sentence], [0])
    np.testing.assert_array_equal(y[: len(sentence) - 1], sentence[1:])
    assert x[0] == 2 and y[0] == row[0]
    assert atomic_sentence([row[0], row[1], row[2]])[-3] == row[2]


def test_pair_initialization_and_self_patch_identity():
    cfg = dict(spec(), width=16, heads=2, layers=1, repeats=2, dropout=0.0)
    a = construct(cfg, "cpu").eval()
    b = construct(cfg, "cpu").eval()
    for key, val in a.state_dict().items():
        torch.testing.assert_close(val, b.state_dict()[key], rtol=0, atol=0)
    rows = build_world(cfg)["target_test"][:3]
    x = torch.tensor([composite_sentence(r)[:-3] for r in rows])
    with torch.no_grad():
        cache = {}
        expected = a(x, cache=cache)
        for layer in range(2):
            for component in ("attention", "mlp", "full"):
                value = cache[layer][component][:, 3]
                actual = a(x, patch=dict(layer=layer, component=component, position=3, value=value))
                torch.testing.assert_close(expected, actual, rtol=0, atol=0)


def test_pure_prefix_state_cannot_contain_second_relation_or_answer():
    cfg = dict(spec(), width=16, heads=2, layers=2, repeats=2, dropout=0.0)
    model = construct(cfg, "cpu").eval()
    rows = build_world(cfg)["target_test"][:4]
    x = torch.tensor([composite_sentence(r)[:-3] for r in rows])
    altered = x.clone()
    altered[:, 4:] = 0
    with torch.no_grad():
        full, prefix = {}, {}
        model(x, cache=full)
        model(altered, cache=prefix)
    for layer in full:
        for component in ("attention", "mlp", "full"):
            torch.testing.assert_close(
                full[layer][component][:, 3], prefix[layer][component][:, 3], rtol=0, atol=0
            )


def test_counterfactual_donors_obey_holdout_and_preserve_relation():
    import importlib.util

    module_spec = importlib.util.spec_from_file_location(
        "text_analysis", "scripts/analyze_text_pretrain.py"
    )
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    world = build_world(spec())
    held = {tuple(r[[0, 4]]) for r in world["background_test"]}
    for kind, pairs in module.select_donors(world, spec()["world"]).items():
        for index, donor in pairs:
            recipient = world["target_test"][index]
            assert donor[0] != recipient[0]
            assert donor[1] == recipient[1] and donor[3] == recipient[3]
            if kind == "same_bridge":
                assert donor[2] == recipient[2] and donor[4] == recipient[4]
            else:
                assert donor[2] != recipient[2] and donor[4] != recipient[4]
                assert (donor[0], donor[4]) in held
