"""Loss weighting must preserve logical supervision mass across the four arms."""

import torch
from torch.nn import functional as F

from llm_memory_editability.grok_usage_train import weighted_loss


def test_main_arm_loss_equals_historical_answer_and_eos_mean():
    torch.manual_seed(17)
    logits = torch.randn(7, 2, 19, requires_grad=True)
    labels = torch.randint(19, (7, 2))
    actual = weighted_loss(logits, labels, torch.ones(7), 7)
    expected = F.cross_entropy(logits.flatten(0, 1), labels.flatten())
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual, logits, retain_graph=True)[0],
        torch.autograd.grad(expected, logits)[0],
    )


def test_two_half_weight_substitutes_preserve_one_slot_loss_and_gradient():
    torch.manual_seed(21)
    logits = torch.randn(3, 2, 11, requires_grad=True)
    labels = torch.randint(11, (3, 2))
    indices = torch.tensor([0, 1, 2, 2])
    repeated = weighted_loss(
        logits[indices], labels[indices], torch.tensor([1.0, 1.0, 0.5, 0.5]), 3
    )
    original = weighted_loss(logits, labels, torch.ones(3), 3)
    torch.testing.assert_close(repeated, original)
    torch.testing.assert_close(
        torch.autograd.grad(repeated, logits, retain_graph=True)[0],
        torch.autograd.grad(original, logits)[0],
    )
