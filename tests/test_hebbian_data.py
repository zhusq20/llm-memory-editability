import json

import pytest

from llm_memory_editability.hebbian_data import TEMPLATE, encode_answer, normalize, score_answer
from llm_memory_editability.hebbian_learning import DATA


def test_complete_answer_scoring_rejects_prefix_and_explanation():
    assert score_answer("  PARIS.\nAnother line", ["Paris"]) == 1
    assert score_answer("New York", ["New York City"]) == 0
    assert score_answer("Paris, France", ["Paris"]) == 0
    assert normalize(" Ａlice  SMITH ") == "alice smith"


def test_actual_tokenizer_masks_prompt_and_includes_eos():
    pytest.importorskip("transformers")
    from transformers import AutoTokenizer

    path = DATA / "source/qwen3-0.6b-base"
    if not path.exists():
        pytest.skip("Locked tokenizer not downloaded")
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    prompt = TEMPLATE.format("The birthplace of Example is")
    encoded = encode_answer(tok, prompt, "New York City")
    start = encoded["answer_start"]
    assert not any(encoded["loss_mask"][:start])
    assert all(encoded["loss_mask"][start:])
    assert encoded["input_ids"][-1] == tok.eos_token_id
    assert tok.decode(encoded["input_ids"][start:-1]).strip() == "New York City"


def test_locked_subjects_views_labels_and_episodes():
    if not (DATA / "pools.json").exists():
        pytest.skip("No frozen experimental data")
    pools = json.loads((DATA / "pools.json").read_text())
    episodes = json.loads((DATA / "episodes.json").read_text())
    raw = {r["case_id"]: r for r in json.loads((DATA / "source/counterfact.json").read_text())}
    seen = set()
    for name, records in pools.items():
        groups = {r["subject_group"] for r in records}
        assert not seen & groups
        seen |= groups
        for r in records:
            assert r["answer"] == raw[r["case_id"]]["requested_rewrite"]["target_true"]["str"]
            assert len(r["views"]) == 3
            if name.endswith("keep"):
                assert r["baseline"]["views"][0]["answer_em"] == 1
        if name in episodes:
            assert sorted(i for episode in episodes[name] for i in episode) == sorted(
                r["case_id"] for r in records
            )
