"""Contracts for position-restricted MLP writes."""

import pytest
import torch
from transformers import GPT2Config

from llm_memory_editability.supervised_alignment import SupervisedAlignmentGPT
from llm_memory_editability.ttcd_editing import TTCDReader
from llm_memory_editability.write_position import layer_values, masked_logits


def make_reader(repeats, discoloop):
    torch.manual_seed(20261020)
    cfg = GPT2Config(
        vocab_size=29,
        n_positions=16,
        n_embd=16,
        n_head=2,
        n_layer=2,
        resid_pdrop=0,
        embd_pdrop=0,
        attn_pdrop=0,
    )
    model = SupervisedAlignmentGPT(cfg, repeats=repeats, discoloop=discoloop).double()
    return TTCDReader(model, native_causal_attention=True).eval()


def shifted_values(reader, scale=0.05):
    names = [n for n, _ in reader.backbone.named_parameters() if ".mlp." in n]
    return {
        n: reader.backbone.get_parameter(n).detach()
        + scale * torch.randn_like(reader.backbone.get_parameter(n))
        for n in names
    }


@pytest.mark.parametrize(("repeats", "discoloop"), [(1, False), (2, False), (2, True)])
def test_empty_and_full_masks_reproduce_the_reader(repeats, discoloop):
    reader = make_reader(repeats, discoloop)
    tokens = torch.randint(0, 27, (5, 4))
    positions = torch.tensor([[1, 2]] * 5)
    values = shifted_values(reader)
    new = layer_values(values, 2)
    empty = torch.zeros(tokens.shape, dtype=torch.bool)
    with torch.no_grad():
        before = reader(tokens, positions)
    assert torch.allclose(masked_logits(reader, tokens, positions, new, empty), before, atol=1e-10)
    original = {n: reader.backbone.get_parameter(n).detach().clone() for n in values}
    for name, value in values.items():
        reader.backbone.get_parameter(name).data.copy_(value)
    with torch.no_grad():
        after = reader(tokens, positions)
    for name, value in original.items():
        reader.backbone.get_parameter(name).data.copy_(value)
    full = masked_logits(reader, tokens, positions, new, ~empty)
    assert torch.allclose(full, after, atol=1e-10)
    assert not torch.allclose(before, after)


def test_mask_is_causal_and_position_local():
    reader = make_reader(2, True)
    tokens = torch.randint(0, 27, (3, 4))
    positions = torch.tensor([[1, 2]] * 3)
    new = layer_values(shifted_values(reader, 0.5), 2)
    late = torch.zeros(tokens.shape, dtype=torch.bool)
    late[:, 2:] = True
    with torch.no_grad():
        before = reader(tokens, positions)
    # A write restricted to positions after the answer position cannot change its score.
    out = masked_logits(reader, tokens, positions, new, late)
    assert torch.allclose(out[:, 0], before[:, 0], atol=1e-10)
    subject = torch.zeros(tokens.shape, dtype=torch.bool)
    subject[:, 0] = True
    assert not torch.allclose(masked_logits(reader, tokens, positions, new, subject), before)
