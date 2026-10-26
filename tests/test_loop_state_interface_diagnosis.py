"""Input-only scale diagnosis must not change the tied decoder or soft feedback."""

import importlib.util
import os
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("LOOP_DIAGNOSTIC_SOURCE", str(ROOT))
os.environ["LOOP_REGRESSION_MODULE"] = str(ROOT / "scripts/diagnose_loop_state_regression.py")
spec = importlib.util.spec_from_file_location(
    "state_interfaces", ROOT / "scripts/diagnose_loop_state_interfaces.py"
)
diag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diag)


def test_norm_rollback_changes_only_B_input_lookup_and_restores_without_parameter_changes():
    torch.manual_seed(29)
    embedding = torch.nn.Embedding(12, 4)
    original = embedding.weight.detach().clone()
    early = original.clone()
    early[5:8] *= 0.3
    tokens = torch.arange(12)
    before = embedding(tokens).detach()
    handle = diag.input_norm_hook(embedding, early, {"entities_per_graph": 3, "entity_offset": 2})
    during = embedding(tokens).detach()
    keep = [i for i in range(12) if i not in (5, 6, 7)]
    assert torch.equal(during[keep], before[keep])
    torch.testing.assert_close(during[5:8].norm(dim=-1), early[5:8].norm(dim=-1))
    assert torch.equal(embedding.weight, original)
    handle.remove()
    assert torch.equal(embedding(tokens), before)


def test_norm_rollback_leaves_decoder_logits_and_soft_feedback_exact():
    torch.manual_seed(30)
    embedding = torch.nn.Embedding(12, 4)
    early = embedding.weight.detach().clone() * 0.25
    states, probability = torch.randn(2, 4), torch.randn(2, 12).softmax(-1)
    readout = F.linear(states, embedding.weight)
    feedback = F.linear(probability, embedding.weight.T)
    handle = diag.input_norm_hook(embedding, early, {"entities_per_graph": 3, "entity_offset": 2})
    try:
        assert torch.equal(F.linear(states, embedding.weight), readout)
        assert torch.equal(F.linear(probability, embedding.weight.T), feedback)
    finally:
        handle.remove()
