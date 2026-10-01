"""Training contracts that affect composition-learning conclusions."""

import copy

import numpy as np
import pytest
import torch

from llm_memory_editability.bios_model import ModelConfig
from llm_memory_editability.grok_depth import EpochStream, SmallGPT, evaluate_rows, pack_rows


@pytest.mark.parametrize("rows", [[[2, 8, 5]], [[2, 8, 9, 5]]])
def test_pack_supervises_tail_and_eos_after_its_actual_input(rows):
    tokens, positions, labels = pack_rows(rows)
    length = len(rows[0])
    np.testing.assert_array_equal(tokens[:, :length], rows)
    np.testing.assert_array_equal(tokens[:, length:], 0)
    np.testing.assert_array_equal(positions, [[length - 2, length - 1]])
    np.testing.assert_array_equal(labels, [[5, 1]])
    assert tokens[0, positions[0, 0]] == rows[0][-2]
    assert tokens[0, positions[0, 1]] == labels[0, 0]


def test_answers_cannot_see_tail_and_atomic_predictions_cannot_see_right_padding():
    torch.manual_seed(91)
    model = SmallGPT(ModelConfig(vocab_size=16, width=16, layers=2, heads=2, context=4))
    model.eval()
    x, pos, _ = (torch.as_tensor(a) for a in pack_rows([[2, 8, 5], [3, 9, 6]]))
    original = model(x, pos).detach()
    changed_tail = x.clone()
    changed_tail[:, 2] = torch.tensor([11, 12])
    torch.testing.assert_close(model(changed_tail, pos)[:, 0], original[:, 0], rtol=0, atol=0)
    changed_padding = x.clone()
    changed_padding[:, 3] = torch.tensor([13, 14])
    torch.testing.assert_close(model(changed_padding, pos), original, rtol=0, atol=0)
    torch.testing.assert_close(model(x[:, :3], pos), original, rtol=1e-6, atol=1e-7)
    comp, pos, _ = (torch.as_tensor(a) for a in pack_rows([[2, 8, 9, 5]]))
    baseline = model(comp, pos).detach()
    comp[:, 3] = 12
    torch.testing.assert_close(model(comp, pos)[:, 0], baseline[:, 0], rtol=0, atol=0)


class GeneratedAnswerEOS(torch.nn.Module):
    """EOS is correct only when evaluation feeds the generated (wrong) answer back."""

    def forward(self, x, positions=None):
        logits = torch.full((len(x), 2, 16), -10.0, device=x.device)
        logits[:, 0, 7] = 10.0
        tails = x[torch.arange(len(x)), positions[:, 1]]
        stop = (tails == 7).long()
        logits[torch.arange(len(x)), 1, stop] = 10.0
        return logits


def test_eos_evaluation_really_continues_generated_answer():
    model = GeneratedAnswerEOS()
    metrics, pred = evaluate_rows(model, np.array([[2, 8, 5], [3, 9, 6]]), "cpu")
    np.testing.assert_array_equal(pred["answer"], [7, 7])
    np.testing.assert_array_equal(pred["stop"], [1, 1])
    assert metrics["answer_accuracy"] == metrics["accuracy"] == 0.0


def test_epoch_stream_restores_partial_epoch_and_cross_epoch_batches():
    stream = EpochStream(13, 123)
    first = stream.take(13)
    assert sorted(first.tolist()) == list(range(13))
    stream.take(5)
    saved = copy.deepcopy(stream.state_dict())
    expected = [stream.take(n) for n in [2, 19, 31, 1]]
    resumed = EpochStream(13, 999)
    resumed.load_state_dict(saved)
    for n, values in zip([2, 19, 31, 1], expected, strict=True):
        np.testing.assert_array_equal(resumed.take(n), values)
    wrong_size = EpochStream(14, 123)
    with pytest.raises(AssertionError):
        wrong_size.load_state_dict(saved)
