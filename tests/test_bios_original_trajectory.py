"""Native token boundaries and attribute-only likelihood define the trajectory measure."""

import json
import math
import runpy
from pathlib import Path

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer

from llm_memory_editability.bios_original_data import source_templates

MODULE = runpy.run_path("scripts/run_bios_original_trajectory.py")


def test_native_cases_use_original_sentence_tokens_and_name_before_date():
    root = Path("data/bios-original-v1/world-142701")
    people = json.loads((root / "people.json").read_text())
    ids = np.load(root / "template-ids.npy", mmap_mode="r")
    tokens = np.load(root / "sentence-tokens.npy", mmap_mode="r")
    lengths = np.load(root / "sentence-lengths.npy", mmap_mode="r")
    tokenizer = Tokenizer.from_file("data/bios-original-v1/tokenizer/tokenizer.json")
    templates = source_templates()
    for p in people:
        try:
            rows = [
                MODULE["native_case"](
                    p,
                    attr,
                    templates,
                    ids[p["id"], 0],
                    tokens[p["id"], 0],
                    lengths[p["id"], 0],
                    tokenizer,
                )
                for attr in ("date", "birthcity")
            ]
        except ValueError:
            continue
        assert p["name"] in rows[0]["prefix"]
        assert rows[0]["target"] not in rows[0]["prefix"]
        assert rows[1]["contains_true_date_prefix"]
        assert rows[0]["target"] in rows[1]["prefix"]
        assert rows[1]["prompt_ids"][:1] == [50256]
        return
    pytest.fail("No valid native template case")


def test_full_attribute_probability_excludes_boundary_and_eos():
    class Uniform:
        def hidden(self, ids):
            return ids.float().unsqueeze(-1)

        def logits(self, hidden):
            return torch.zeros((*hidden.shape[:-1], 5))

    class Decoder:
        def decode(self, tokens, **kwargs):
            return str(tokens)

    row = dict(prompt_ids=[4, 1], target_ids=[2, 3], boundary_token=4)
    result = MODULE["measure"](Uniform(), [row], Decoder(), "cpu", 4)[0]
    assert result["full_attribute_nll"] == pytest.approx(2 * math.log(5))
    assert result["full_attribute_probability"] == pytest.approx(1 / 25)
    assert len(result["generated_ids"]) == 4
    assert not result["attribute_exact"]
