import numpy as np
import torch
from transformers import AutoTokenizer

from llm_memory_editability.qwen_path_learning import (
    CONFIG,
    ROOT,
    encode_fact,
    fisher_quadratic,
    random_positions,
    read,
    sequence_mean,
)
from llm_memory_editability.qwen_path_train import schedule


def test_full_answer_eos_and_prompt_only_roles():
    tokenizer = AutoTokenizer.from_pretrained(
        ROOT / read(CONFIG)["model_source"], local_files_only=True
    )
    first = encode_fact(tokenizer, "Alan Turing", "Alan Turing was born in", "London")
    second = encode_fact(tokenizer, "Alan Turing", "Alan Turing was born in", "Manchester")
    n = first["answer_start"]
    assert first["roles"][:n] == second["roles"][:n]
    assert first["input_ids"][:n] == second["input_ids"][:n]
    assert first["roles"].count("S_end") == 1
    assert first["roles"][n - 1] == "L"
    assert first["roles"].index("S_end") < n - 1
    assert first["input_ids"][-1] == tokenizer.eos_token_id
    assert sum(first["loss_mask"]) == first["answer_tokens"] + 1
    assert all(x == "A" for x in first["roles"][n:])


def test_random_controls_match_positions_and_do_not_use_answer_history():
    roles = ["W"] * 6 + ["S_other", "S_end"] + ["C"] * 6 + ["L", "A", "A"]
    positions, info = random_positions(roles, "S_end", "prompt", 0)
    again, _ = random_positions(roles[:-2] + ["A"] * 10, "S_end", "prompt", 0)
    assert positions == again and info["valid"]
    assert len(positions) == 1 and positions[0] != 7
    assert positions[0] * 2 // 14 == 7 * 2 // 14
    invalid, reason = random_positions(["W"] * 8 + ["L", "A"], "W", "prompt", 0)
    assert invalid == [] and not reason["valid"]


def test_sequence_weighting_is_not_token_weighting():
    values = torch.tensor([1.0, 3.0, 3.0, 3.0])
    b = torch.tensor([0, 1, 1, 1])
    assert sequence_mean(values, b, 2).mean() == 2
    assert values.mean() == 2.5


def test_kl_quadratic_keeps_cancelling_cross_terms():
    logp = torch.tensor([[-0.3, 0.4, 1.0]], dtype=torch.float64).log_softmax(-1)
    a = torch.tensor([[1.0, -1.0, 0.1]], dtype=torch.float64)
    b = -a
    assert fisher_quadratic(logp, a).item() > 0
    assert fisher_quadratic(logp, a + b).item() == 0
    for eps in (0.001, 0.0005):
        new_logp = (logp + eps * a).log_softmax(-1)
        actual = (logp.exp() * (logp - new_logp)).sum()
        predicted = fisher_quadratic(logp, eps * a).sum()
        assert torch.isclose(actual, predicted, rtol=0.001, atol=1e-12)


def test_exposure_schedule_is_paired_and_balanced():
    cfg = read(CONFIG)
    a = schedule(cfg, 0, 0)
    assert a == schedule(cfg, 0, 0)
    assert a != schedule(cfg, 1, 0)
    r_counts = np.bincount([i for step in a for i in step["R"]], minlength=64)
    t_counts = np.bincount([i for step in a for i in step["text"]], minlength=64)
    assert set(r_counts) == {32}
    assert set(t_counts) == {8}
