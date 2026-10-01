"""Research-critical information, scoring, split and intervention contracts."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location(
    "frozen_runner", ROOT / "scripts/run_twohop_frozen.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_scoring_does_not_accept_answer_substrings_or_cooccurrence():
    assert runner.grade("France and Germany", "France")["em"] is False
    assert runner.grade("Not France", "France")["em"] is False
    assert runner.grade("Paris", "Paris, Texas")["em"] is False
    assert runner.grade("reason\nFinal answer: The United Kingdom.", "United Kingdom")["em"]
    assert runner.grade("UK", "United Kingdom", ["UK"])["alias_em"]
    assert not runner.grade("UK", "United Kingdom", ["UK"])["em"]


def test_patch_only_changes_selected_position_and_prefill():
    layer = SimpleNamespace(mlp=torch.nn.Identity())
    engine = runner.Engine.__new__(runner.Engine)
    engine.model = SimpleNamespace(model=SimpleNamespace(layers=[layer]))
    x = torch.arange(24).reshape(2, 3, 4).float()
    delta = torch.ones((2, 4))
    with engine.patch(0, delta):
        y = layer.mlp(x)
        assert torch.equal(y[:, :-1], x[:, :-1])
        assert torch.equal(y[:, -1], x[:, -1] + delta)
        cached = x[:, :1]
        assert torch.equal(layer.mlp(cached), cached)
    assert torch.equal(layer.mlp(x), x)
    with engine.patch(0, delta, torch.tensor([0, 1])):
        y = layer.mlp(x)
        assert torch.equal(y[0, 0], x[0, 0] + 1)
        assert torch.equal(y[1, 1], x[1, 1] + 1)
        assert torch.equal(y[:, 2], x[:, 2])


def test_frozen_samples_and_donors_preserve_split_and_target_contracts():
    cases = runner.read(runner.DATA / "cases.json")
    cfg = runner.read(ROOT / "configs/twohop-frozen-v1.json")
    assert len(cases) == 288
    for dataset in ["mquake", "2wiki"]:
        groups = {}
        for split in ["development", "evaluation"]:
            part = [r for r in cases if r["dataset"] == dataset and r["split"] == split]
            assert len(part) == cfg["behavior_per_dataset"][split]
            assert sum(r["mechanism"] for r in part) == cfg["mechanism_per_dataset"][split]
            groups[split] = {r["group"] for r in part}
            assert len(groups[split]) == len(part)
            for r in part:
                donor = r["wrong_donor"]
                if donor:
                    assert donor["id"] != r["id"]
                    assert runner.normalize(donor["answer"]) != runner.normalize(r["answer"])
        assert not groups["development"] & groups["evaluation"]


def test_autonomous_prompt_does_not_require_gold_fields():
    content = runner.prompt_content("Who directed the film starring this actor?", proposal=True)
    assert "Proposed intermediate entity:" not in content
    question = "What is the capital of the country where X was born?"
    content = runner.prompt_content(question, bridge="MODEL GENERATED ENTITY")
    assert question in content and "MODEL GENERATED ENTITY" in content
