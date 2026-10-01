"""Contracts that affect load, knowledge availability and architecture conclusions."""

import numpy as np
import pytest
import torch

from llm_memory_editability.grok_depth import EpochStream
from llm_memory_editability.storage_composition import (
    ARCHITECTURES,
    audit_world,
    build_world,
    construct,
    data_digest,
    generate_rows,
    pack_sentences,
)


@pytest.fixture
def spec():
    return {
        "world": 7,
        "heads_n": 24,
        "bridges_n": 24,
        "tails_n": 16,
        "familiar_n": 8,
        "strict_n": 4,
        "holdout_fraction": 0.3,
        "width": 32,
        "heads": 4,
        "dropout": 0.0,
        "architecture": "standard1",
        "initialization": 17,
    }


def test_world_truth_pair_holdout_and_zero_strict_fact_roles(spec):
    world = build_world(spec)
    audit_world(world)
    assert len(world["common_atomic"]) == 8 * 12
    assert len(world["extra_atomic"]) == 4 * (24 + 24) - 8 * 12
    other = build_world(dict(spec, load="high"))
    assert data_digest(world) == data_digest(other)
    lookup = {(int(h), int(r)): int(t) for h, r, t in world["common_atomic"]}
    for h, r1, b, r2, t in world["strict_test"]:
        assert lookup[int(h), int(r1)] == b and lookup[int(b), int(r2)] == t
    corrupt = {key: value.copy() for key, value in world.items()}
    corrupt["strict_test"][0, -1] += 1
    with pytest.raises(AssertionError):
        audit_world(corrupt)


def test_full_next_token_objective_and_composite_prompt_hides_bridge(spec):
    world = build_world(spec)
    x, labels = pack_sentences(world["common_atomic"])
    assert np.all((labels != -100).sum(axis=1) == 7)
    assert np.all(x[:, 7:] == 0)
    assert np.all(labels[:, 6] == 1)
    comp = world["strict_test"][:2].copy()
    a = pack_sentences(comp)
    comp[:, 2] += 1
    b = pack_sentences(comp)
    assert all(np.array_equal(left, right) for left, right in zip(a, b, strict=True))
    assert np.all((a[1] != -100).sum(axis=1) == 9)


def test_common_exposures_match_exactly_across_loads(spec):
    world = build_world(spec)
    low, high = [], []
    for store, extra in ((low, world["background_atomic"]), (high, world["extra_atomic"])):
        sizes = [len(world["common_atomic"]), len(world["train_composite"]), len(extra)]
        for i, size in enumerate(sizes):
            stream = EpochStream(size, 41 + i)
            indices = stream.take(3200)
            store.append(indices)
            assert np.ptp(np.bincount(indices, minlength=size)) <= 1
    np.testing.assert_array_equal(low[0], high[0])
    np.testing.assert_array_equal(low[1], high[1])


def test_standard_and_loop_have_identical_trainable_weights_at_equal_unique_depth(spec):
    standard = construct(spec, "cpu").eval()
    loop = construct(dict(spec, architecture="loop1x2"), "cpu").eval()
    loop3 = construct(dict(spec, architecture="loop1x3"), "cpu").eval()
    for key, value in standard.state_dict().items():
        assert torch.equal(value, loop.state_dict()[key])
        assert torch.equal(value, loop3.state_dict()[key])
    assert all(p.requires_grad for p in loop.parameters())
    block = loop.blocks[0]
    assert block.attention.qkv.out_features == spec["width"] * 3
    assert block.mlp.up.out_features == spec["width"] * 4
    assert list(loop.iter_blocks())[0] is list(loop.iter_blocks())[1]
    x = torch.tensor([[2, 3, 4, 5]])
    torch.testing.assert_close(standard(x), loop(x, repeats=1), rtol=0, atol=0)
    assert len(ARCHITECTURES) == 5


def test_standard_forward_matches_huggingface_gpt2(spec):
    from transformers import GPT2Config, GPT2LMHeadModel

    ours = construct(dict(spec, architecture="standard2"), "cpu").eval()
    cfg = ours.config
    reference = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=cfg.vocab_size,
            n_positions=cfg.context,
            n_embd=cfg.width,
            n_layer=cfg.layers,
            n_head=cfg.heads,
            activation_function="gelu_new",
            resid_pdrop=0,
            embd_pdrop=0,
            attn_pdrop=0,
        )
    ).eval()
    with torch.no_grad():
        reference.transformer.wte.weight.copy_(ours.token.weight)
        reference.transformer.wpe.weight.copy_(ours.position.weight)
        reference.transformer.ln_f.load_state_dict(ours.ln_final.state_dict())
        for left, right in zip(ours.blocks, reference.transformer.h, strict=True):
            right.ln_1.load_state_dict(left.ln1.state_dict())
            right.ln_2.load_state_dict(left.ln2.state_dict())
            for source, target in (
                (left.attention.qkv, right.attn.c_attn),
                (left.attention.proj, right.attn.c_proj),
                (left.mlp.up, right.mlp.c_fc),
                (left.mlp.down, right.mlp.c_proj),
            ):
                target.weight.copy_(source.weight.T)
                target.bias.copy_(source.bias)
    tokens = torch.tensor([[2, 22, 3, 13, 4, 49, 5], [2, 23, 3, 14, 4, 50, 5]])
    with torch.no_grad():
        torch.testing.assert_close(ours(tokens), reference(tokens).logits, rtol=1e-5, atol=1e-6)


def test_generated_tokens_are_fed_back_for_period_and_eos(spec):
    class RecordingModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.prompts = []

        def forward(self, tokens):
            self.prompts.append(tokens.clone())
            output = torch.zeros(len(tokens), tokens.shape[1], 100)
            output[:, -1, (42, 5, 1)[len(self.prompts) - 1]] = 10
            return output

    model = RecordingModel()
    metrics, pred = generate_rows(model, np.array([[22, 13, 42]]), "cpu")
    assert metrics["accuracy"] == 1
    assert model.prompts[1][0, -1] == 42
    assert model.prompts[2][0, -1] == 5
    np.testing.assert_array_equal(pred["generated"], [[42, 5, 1]])
