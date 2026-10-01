"""Guard regression for both worlds; science and optimizer code are unchanged."""

import ast
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from llm_memory_editability import bios_shortcut_edit_v2 as v2
from llm_memory_editability.bios_cross import make_cross_world
from llm_memory_editability.bios_data import array_hash


@pytest.mark.parametrize("world_seed,expected_vocab", [(0, 4804), (1, 4805)])
def test_both_world_vocabularies_pass_guard_before_checkpoint_read(
    tmp_path, monkeypatch, world_seed, expected_vocab
):
    world = make_cross_world(world_seed)
    assert world.vocab_size == expected_vocab
    config = {
        "model": dict(vocab_size=expected_vocab, width=256, layers=8, heads=4, context=128),
        "study": {
            "protocol": "v2.6-development-crossover",
            "steps": 15360,
            "lr": 1e-4,
            "documents_per_step": 16,
            "facts_per_document": 10,
            "QA_per_chain_per_step": 20,
            "document_QA_weights": [0.8, 0.2],
        },
        "world": world_seed,
        "seed": 0,
        "condition": "company",
        "sources": v2.source_hashes(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "truth_sha256": array_hash(world.answers),
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    np.savez(
        tmp_path / "predictions-15360.npz",
        prediction=world.answers,
        ended=np.ones(len(world.answers), dtype=bool),
        correct=np.ones(len(world.answers), dtype=bool),
        value_nll=np.zeros(len(world.answers)),
    )

    def reached_checkpoint(*args, **kwargs):
        raise RuntimeError("passed world-dependent vocabulary guard")

    monkeypatch.setattr(torch, "load", reached_checkpoint)
    with pytest.raises(RuntimeError, match="passed world-dependent"):
        v2.load_parent(tmp_path, "low")
    config["model"]["vocab_size"] = 4805 if world_seed == 0 else 4804
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="requires original width256"):
        v2.load_parent(tmp_path, "low")


def test_v2_changes_only_parent_validation_and_source_self_identity():
    directory = Path(v2.__file__).parent
    a, b = [
        ast.parse((directory / name).read_text())
        for name in ("bios_shortcut_edit.py", "bios_shortcut_edit_v2.py")
    ]
    a_functions = {
        node.name: ast.dump(node) for node in a.body if isinstance(node, ast.FunctionDef)
    }
    b_functions = {
        node.name: ast.dump(node) for node in b.body if isinstance(node, ast.FunctionDef)
    }
    assert a_functions.keys() == b_functions.keys()
    changed = {name for name in a_functions if a_functions[name] != b_functions[name]}
    assert changed == {"load_parent", "editing_sources"}
    a_rest = ast.Module(
        body=[node for node in a.body if not isinstance(node, ast.FunctionDef)], type_ignores=[]
    )
    b_rest = ast.Module(
        body=[node for node in b.body if not isinstance(node, ast.FunctionDef)], type_ignores=[]
    )
    assert ast.dump(a_rest) == ast.dump(b_rest)
    # The actual execution path, checkpoints, samplers, optimizer updates, score
    # definitions, and resume handling are byte-for-byte equal ASTs above.
