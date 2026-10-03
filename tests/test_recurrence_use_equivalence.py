"""Deletion is fewer loops for a single shared autonomous block."""

import pytest
import torch

from llm_memory_editability.bios_model import ModelConfig
from llm_memory_editability.grok_loop_model import LoopGPT
from llm_memory_editability.recurrence_use import trace


@pytest.mark.parametrize("repeats", [2, 4, 6])
def test_every_whole_block_deletion_equals_native_one_fewer_loop(repeats):
    torch.manual_seed(911)
    model = (
        LoopGPT(
            ModelConfig(vocab_size=14, width=8, layers=1, heads=2, context=9),
            repeats=repeats - 1,
            dropout=0.0,
        )
        .eval()
        .double()
    )
    model.requires_grad_(False)
    tokens = torch.tensor([[2, 3, 11, 12, 13], [2, 4, 12, 11, 13]])
    expected = model(tokens)
    for source in range(repeats):
        actual, _ = trace(model, tokens, repeats, skip=("all", source))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_local_deletion_is_not_whole_iteration_removal():
    torch.manual_seed(912)
    model = (
        LoopGPT(
            ModelConfig(vocab_size=14, width=8, layers=1, heads=2, context=9),
            repeats=3,
            dropout=0.0,
        )
        .eval()
        .double()
    )
    tokens = torch.tensor([[2, 3, 11, 12, 13]])
    with torch.no_grad():
        fewer = model(tokens)[:, -1]
        local, _ = trace(model, tokens, 4, skip=("r1", 0))
    assert not torch.equal(local[:, -1], fewer)
