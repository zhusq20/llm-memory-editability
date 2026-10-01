import numpy as np

from llm_memory_editability.bios_model import CausalLM, ModelConfig
from llm_memory_editability.hebbian_followup import CONFIG, fixed_world, parse_entity
from llm_memory_editability.hebbian_future import read, training_arrays


def test_focal_content_and_marginals_are_fixed():
    cfg = read(CONFIG)["chains"]
    for world in cfg["worlds"]:
        worlds = [fixed_world(world, rho, cfg) for rho in cfg["correlations"]]
        for w in worlds:
            focal = w["common_conflict"]
            np.testing.assert_array_equal(w["home_y"][focal], worlds[0]["home_y"][focal])
            np.testing.assert_array_equal(w["composite_y"], worlds[0]["composite_y"])
            np.testing.assert_array_equal(training_arrays(w)[0], training_arrays(worlds[0])[0])
            for mask in (w["train_people"], ~w["train_people"]):
                assert (w["home_y"][mask] == w["composite_y"][mask]).mean() == w["correlation"]
                assert np.ptp(np.bincount(w["home"][mask], minlength=4)) == 0


def test_near_equal_parameter_budget():
    cfg = read(CONFIG)["chains"]
    counts = {}
    for arch in cfg["architectures"]:
        model = CausalLM(ModelConfig(122, arch["width"], arch["depth"], arch["heads"], 8))
        counts[arch["name"]] = sum(p.numel() for p in model.parameters())
    assert abs(counts["d2w135"] / counts["d4w96"] - 1) < 0.005
    assert counts["d2w96"] < 0.52 * counts["d4w96"]


def test_full_alias_and_abstention():
    inventory = {
        "NY": {"aliases": ["New York", "New York City"]},
        "NJ": {"aliases": ["New Jersey"]},
        "FR": {"aliases": ["French"]},
    }
    assert parse_entity(" New York City. He later moved.", inventory)["entity"] == "NY"
    assert parse_entity("New York and New Jersey", inventory)["status"] == "ambiguous"
    assert parse_entity("not New York", inventory)["entity"] is None
    assert parse_entity("Frenchman", inventory)["entity"] is None
    assert parse_entity("The answer is New York", inventory)["status"] == "unrecognized"


def test_candidate_scoring_uses_all_answer_tokens_without_eos(monkeypatch):
    import importlib

    import torch
    from transformers import AutoTokenizer

    from llm_memory_editability.hebbian_future import ROOT
    from llm_memory_editability.hebbian_model import QwenExperiment

    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    runner = importlib.import_module("run_hebbian_followup")
    engine = QwenExperiment.__new__(QwenExperiment)
    engine.device = torch.device("cpu")
    engine.tokenizer = AutoTokenizer.from_pretrained(
        ROOT / "data/hebbian-learning-v1/source/qwen3-0.6b-base", local_files_only=True
    )
    engine.tokenizer.pad_token = engine.tokenizer.eos_token
    engine.model = torch.nn.Module()
    engine.model.lm_head = torch.nn.Linear(1, len(engine.tokenizer), bias=False)
    engine.model.lm_head.weight.data.zero_()
    engine.hidden = lambda ids, mask: torch.zeros(*ids.shape, 1)
    answers = ["New York City", "French", "United States", "Barcelona"]
    scores = runner.candidate_scores(engine, ["The answer is"], [answers], 4)[0]
    lengths = np.array(
        [len(engine.tokenizer.encode(" " + a, add_special_tokens=False)) for a in answers]
    )
    np.testing.assert_array_equal(scores[:, 2], lengths)
    np.testing.assert_allclose(scores[:, 1], -np.log(len(engine.tokenizer)), rtol=1e-6)
    np.testing.assert_allclose(scores[:, 0], scores[:, 1] * lengths, rtol=1e-6)
    assert len(set(lengths)) > 1
