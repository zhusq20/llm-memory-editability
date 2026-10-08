"""Scientific scoring and eligibility contracts for the original editing benchmarks."""

from pathlib import Path

import pytest

from llm_memory_editability.mquake_reproduction import aggregate as aggregate_mquake
from llm_memory_editability.mquake_reproduction import load_prompts, queries
from llm_memory_editability.paper_editing_runtime import (
    answer_text,
    canonical_block_arguments,
    cloze_alias_match,
    exact_alias_match,
    ripple_match,
    source_ripple_match,
)
from llm_memory_editability.ripple_reproduction import (
    AXES,
    FrozenQuery,
    aggregate,
    fact_labels,
    frozen_aliases,
    relation_templates,
)

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "docs/development-artifacts/paper-reproductions-parallel-v1"


def test_keyword_input_adapter_preserves_gptj_forward_and_gradient():
    import importlib.util

    import torch
    from transformers import GPTJConfig
    from transformers.models.gptj.modeling_gptj import GPTJBlock

    spec = importlib.util.spec_from_file_location(
        "original_nethook", UPSTREAM / "memit-upstream/util/nethook.py"
    )
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    torch.manual_seed(7)
    block = GPTJBlock(GPTJConfig(n_embd=64, n_head=4, rotary_dim=16, n_positions=32)).eval()
    input_a = torch.randn(1, 3, 64, requires_grad=True)
    input_b = input_a.detach().clone().requires_grad_()
    positions = torch.arange(3).unsqueeze(0)
    first = block(hidden_states=input_a, position_ids=positions)[0]
    first.sum().backward()
    block.register_forward_pre_hook(canonical_block_arguments, with_kwargs=True)
    with original.Trace(block, retain_input=True) as trace:
        second = block(hidden_states=input_b, position_ids=positions)[0]
    second.sum().backward()
    assert torch.equal(first, second)
    assert torch.equal(input_a.grad, input_b.grad)
    assert torch.equal(trace.input, input_b)


def test_cot_mentions_are_not_final_answers():
    raw = " France is in Europe.\nAnswer: Asia\n\nQuestion: next"
    assert answer_text(raw, "cot") == "Asia"
    assert not exact_alias_match(answer_text(raw, "cot"), ["Europe"])
    assert answer_text(" Europe is relevant but no answer marker", "cot") == ""
    assert answer_text(" Croatia\nQ: different question A: France", "line") == "Croatia"


def test_cloze_object_prefix_handles_continuation_without_accepting_other_names():
    assert cloze_alias_match(" Croatia, and she lives there.", ["Croatia"])
    assert cloze_alias_match("El Campu.", ["El Campu"])
    assert not cloze_alias_match("Indiana", ["India"])
    assert not cloze_alias_match("the old answer was Croatia", ["Croatia"])
    assert not cloze_alias_match("an arbitrary answer", [""])


def test_ripple_paper_multiobject_rule_and_prompt_leakage():
    answers = [["old object", "alias"], ["another object"]]
    assert ripple_match("an alias remains", answers)
    assert not source_ripple_match("an alias remains", answers)
    query = FrozenQuery(
        {
            "prompt": "The award which is not old object is",
            "answers": [{"value": "old object", "aliases": []}],
        }
    )
    assert not ripple_match("nothing known", query.get_answers())
    assert not ripple_match("anything", [[]])


def test_mquake_edit_denominator_is_facts_and_multihop_is_any_question():
    def predictions(edits, facts, multi):
        return {
            "edit": [{"correct": c} for c in edits],
            "single": [{"correct": c} for c in facts],
            "multi": [{"correct": c} for c in multi],
            "cot": [{"correct": False}] * 3,
        }

    first = predictions([True], [True, True], [False, True, False])
    second = predictions([False, True, True], [True, False, True], [False] * 3)
    scores = aggregate_mquake(
        [
            {"hops": 2, "baseline": {"predictions": first}, "edited": {"predictions": first}},
            {"hops": 3, "baseline": {"predictions": second}, "edited": {"predictions": second}},
        ]
    )["edited"]
    assert scores["edit_wise_success"] == 3 / 4
    assert scores["instance_wise_accuracy"] == 1 / 2
    assert scores["multi_hop_accuracy"] == 1 / 2
    assert scores["all_facts_and_edits_known_coverage"] == 1 / 2
    assert scores["conditional_multi_hop_accuracy"] == 1


def test_ripple_macro_average_and_coverage_do_not_count_unexecuted_as_wrong():
    def record(passed, failed, filtered, edit=True):
        size = passed + failed + filtered
        result = {
            "edit_success": edit,
            "executed_fraction": (passed + failed) / size,
            "test_count": size,
            "paper_accuracy": passed / (passed + failed) if passed + failed else 0,
            "outcomes": {
                "PASSED": list(range(passed)),
                "FAILED": list(range(failed)),
                "NOT_EXECUTED": list(range(filtered)),
            },
        }
        return {"axes": {axis: {"result": result} for axis in AXES.values()}}

    result = aggregate(
        [
            record(9, 1, 90),
            record(0, 1, 0),
            record(0, 0, 5),
            record(5, 0, 0, edit=False),
            {"excluded": True},
        ]
    )
    assert result["LG"]["paper_accuracy"] == pytest.approx(0.45)
    assert result["LG"]["successful_edit_eligible_cases"] == 2
    assert result["LG"]["paper_case_coverage"] == 2 / 4
    assert result["LG"]["precondition_filtered_test_cases"] == 95
    assert result["pool"]["invalid_label_cases"] == 1


def test_all_released_mquake_chains_and_prompts_are_valid():
    import json

    root = UPSTREAM / "mquake-upstream"
    cases = json.loads((root / "datasets/MQuAKE-CF-3k.json").read_text())
    prompts = load_prompts({"prompt_dir": str(root / "prompts")})
    assert len(cases) == 3000
    for case in cases:
        for edited in (False, True):
            recipe = queries(case, prompts, edited)
            assert len(recipe["multi"]) == len(recipe["cot"]) == 3
            assert len(recipe["edit"]) == len(case["requested_rewrite"])
            assert len(recipe["single"]) == len(case["single_hops"])


def test_ripple_uses_released_labels_and_aliases_without_api():
    import json

    source = UPSTREAM / "ripple-upstream"
    templates = relation_templates(source)
    valid_counts = []
    for split in ("recent", "random", "popular"):
        cases = json.loads((source / f"data/benchmark/{split}.json").read_text())
        aliases = frozen_aliases(cases)
        count = 0
        for case in cases:
            subject, target, prompt = fact_labels(case["edit"], templates)
            assert prompt + " " + target + "." == case["edit"]["prompt"]
            if subject and target:
                count += 1
            for field in AXES:
                for test in case[field]:
                    for data in test["test_queries"] + test["condition_queries"]:
                        query = FrozenQuery(data)
                        assert query.get_query_prompt() == data["prompt"]
                        assert query.to_dict() is data
        assert aliases
        valid_counts.append(count)
    assert valid_counts == [1839, 1922, 885]
