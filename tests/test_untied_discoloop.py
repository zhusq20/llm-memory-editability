import json
import sys
from pathlib import Path

import pytest
import torch
from transformers import GPT2Config

from llm_memory_editability.supervised_alignment import SupervisedAlignmentGPT
from llm_memory_editability.untied_discoloop import UntiedDiscoLoopGPT

ROOT = Path(__file__).resolve().parents[1]


def config(layers):
    return GPT2Config(
        vocab_size=37, n_positions=8, n_embd=32, n_layer=layers, n_head=4, use_cache=False
    )


def test_copied_blocks_reproduce_tied_discoloop_exactly():
    torch.manual_seed(0)
    tied = SupervisedAlignmentGPT(config(4), repeats=2, discoloop=True).eval()
    untied = UntiedDiscoLoopGPT(config(8), discoloop_split=4).eval()
    state = untied.state_dict()
    for name, value in tied.state_dict().items():
        state[name] = value.clone()
        if name.startswith("transformer.h."):
            index, rest = name.split(".", 3)[2], name.split(".", 3)[3]
            state[f"transformer.h.{int(index) + 4}.{rest}"] = value.clone()
    untied.load_state_dict(state, strict=True)
    tokens = torch.randint(0, 37, (5, 4))
    positions = torch.tensor([[2, 3]] * 5)
    with torch.no_grad():
        a_logits, a_first, a_final = tied.forward_with_states(tokens, positions)
        b_logits, b_first, b_final = untied.forward_with_states(tokens, positions)
    assert torch.equal(a_first, b_first)
    assert torch.equal(a_final, b_final)
    assert torch.equal(a_logits, b_logits)


def test_channel_sits_after_split_and_changes_output():
    torch.manual_seed(1)
    untied = UntiedDiscoLoopGPT(config(8), discoloop_split=4).eval()
    plain = SupervisedAlignmentGPT(config(8), repeats=1).eval()
    plain.load_state_dict(untied.state_dict(), strict=True)
    tokens = torch.randint(0, 37, (3, 4))
    with torch.no_grad():
        logits, first, _ = untied.forward_with_states(tokens)
        plain_first = plain._embedded(tokens)
        for block in plain.transformer.h[:4]:
            plain_first = untied._blocks(plain_first, [block])
        assert torch.equal(first, plain_first)
        assert not torch.allclose(logits, plain(tokens))


def test_rejects_unauthorized_shapes():
    with pytest.raises(ValueError):
        UntiedDiscoLoopGPT(config(8), repeats=2)
    with pytest.raises(ValueError):
        UntiedDiscoLoopGPT(config(8), discoloop_split=8)


def test_runner_patches_model_flags_and_flops():
    sys.path.insert(0, str(ROOT / "scripts"))
    import run_untied_discoloop

    cfg = json.loads((ROOT / "configs/untied-discoloop-development-v1.json").read_text())
    native = run_untied_discoloop.patched_runner(cfg)
    arm = cfg["arms"]["untied_discoloop"]
    native.validate_arm(arm)
    assert native.SupervisedAlignmentGPT is UntiedDiscoLoopGPT
    assert native.architecture_flags(arm)["discoloop_split"] == 4
    assert native.extra_discoloop_training_flops(cfg["spec"], 768, 1013, arm) > 0
