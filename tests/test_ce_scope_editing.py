"""Contracts for matched exposure, scope, old distillation, and isolated edits."""

import copy

import pytest
import torch
from transformers import GPT2Config

from llm_memory_editability.ce_scope_editing import (
    SCOPES,
    arrays,
    atomic_ce_edit,
    choose_recipe,
    load_edit,
    parameter_names,
    restore,
    teacher_batches,
)
from llm_memory_editability.supervised_alignment import SupervisedAlignmentGPT
from llm_memory_editability.ttcd_editing import TTCDReader, tensor_digest

META = {"entity_offset": 0, "relation_offset": 10, "end_marker": 28, "pad_token": 27}
SETTINGS = {
    "steps": 3,
    "learning_rate": 0.002,
    "kl_factor": 10.0,
    "norm_constraint": 0.0005,
    "gradient_clip": 1.0,
    "old_per_kind_per_step": 1,
    "batch_seed": 4,
}


@pytest.fixture
def reader():
    torch.set_num_threads(1)
    torch.manual_seed(4)
    config = GPT2Config(
        vocab_size=29,
        n_positions=8,
        n_embd=8,
        n_head=2,
        n_layer=2,
        resid_pdrop=0,
        embd_pdrop=0,
        attn_pdrop=0,
    )
    return TTCDReader(
        SupervisedAlignmentGPT(config, repeats=2, discoloop=True).double(),
        native_causal_attention=True,
    ).eval()


def protected(reader):
    from types import SimpleNamespace

    import numpy as np

    world = SimpleNamespace(
        metadata=META,
        atoms=np.array([[0, 0, 2], [2, 1, 3], [3, 0, 1]]),
        chains=np.array([[0, 0, 1, 3], [2, 1, 0, 1], [3, 0, 0, 2]]),
    )
    split = {"protection_atoms": [0, 1, 2], "protection_chains": [0, 1, 2]}
    return teacher_batches(reader, world, split, SETTINGS)


def test_parameter_scopes_are_nested_and_full_counts_tied_embeddings_once(reader):
    down, mlp, all_names = (set(parameter_names(reader.backbone, s)) for s in SCOPES)
    assert down < mlp < all_names
    assert len(down) == 2
    assert all(n.endswith(".mlp.c_proj.weight") for n in down)
    assert any(n.endswith(".mlp.c_fc.weight") for n in mlp)
    assert any("attn.c_attn" in n for n in all_names)
    assert "transformer.wte.weight" in all_names
    assert sum(p.numel() for p in reader.backbone.parameters()) == sum(
        reader.backbone.get_parameter(n).numel() for n in all_names
    )


@pytest.mark.parametrize("scope", SCOPES)
def test_exact_budget_scope_bound_and_independent_reload(reader, scope):
    base = {k: v.detach().clone() for k, v in reader.backbone.state_dict().items()}
    teachers = protected(reader)
    before_teacher = tensor_digest(
        {str(i): b["log_probability"] for i, bs in enumerate(teachers) for b in bs[:1]}
    )
    values, info = atomic_ce_edit(reader, [1, 2, 4], META, scope, SETTINGS, teachers)
    assert info["steps"] == SETTINGS["steps"]
    assert info["new_supervised_tokens"] == 3
    assert info["old_atomic_exposures"] == info["old_composite_exposures"] == 3
    assert info["updated_composite_supervision"] == info["early_entity_supervision"] == 0
    assert info["maximum_absolute_parameter_change"] <= SETTINGS["norm_constraint"] + 1e-14
    assert set(values) == set(parameter_names(reader.backbone, scope))
    assert all(not p.requires_grad and p.grad is None for p in reader.parameters())
    for name, p in reader.backbone.named_parameters():
        if name not in values:
            assert torch.equal(p, base[name])
    saved = tensor_digest(reader.backbone.state_dict())
    restore(reader, base)
    assert tensor_digest(reader.backbone.state_dict()) == tensor_digest(base)
    load_edit(reader, values, scope)
    assert tensor_digest(reader.backbone.state_dict()) == saved
    assert (
        tensor_digest(
            {str(i): b["log_probability"] for i, bs in enumerate(teachers) for b in bs[:1]}
        )
        == before_teacher
    )


def test_all_scopes_use_identical_old_targets_and_exposure(reader):
    teachers, base = protected(reader), copy.deepcopy(reader.backbone.state_dict())
    infos = []
    for scope in SCOPES:
        restore(reader, base)
        _, info = atomic_ce_edit(reader, [1, 2, 4], META, scope, SETTINGS, teachers)
        infos.append(info)
    for key in (
        "steps",
        "new_atomic_exposures",
        "new_supervised_tokens",
        "old_atomic_exposures",
        "old_composite_exposures",
        "old_distillation_slots",
    ):
        assert len({i[key] for i in infos}) == 1
    for step in teachers:
        assert {b["kind"] for b in step} == {"atoms", "chains"}
        for batch in step:
            x, p = batch["tokens"], batch["positions"]
            restore(reader, base)
            with torch.no_grad():
                assert torch.equal(reader(x, p).log_softmax(-1), batch["log_probability"])


def test_atomic_input_has_no_label_leak_and_no_eos_supervision(reader):
    x1, p1, _ = arrays([[1, 2, 3]], META, "cpu")
    x2, p2, _ = arrays([[1, 2, 4]], META, "cpu")
    assert torch.equal(x1[:, :2], x2[:, :2])
    assert torch.equal(p1[:, :1], p2[:, :1])
    with torch.no_grad():
        assert torch.equal(reader(x1[:, :2], p1[:, :1]), reader(x2[:, :2], p2[:, :1]))


def test_composite_new_labels_cannot_be_passed_to_editor(reader):
    with pytest.raises(ValueError, match="atomic fact"):
        atomic_ce_edit(reader, [1, 2, 3, 4], META, "all", SETTINGS, protected(reader))


def test_development_selection_protects_old_knowledge_and_excludes_composition():
    def record(lr, atomic, old):
        return {
            "learning_rate": lr,
            "atomic_accuracy": atomic,
            "minimum_old_retention": old,
            "retention_by_pool": {"atomic_id": old},
        }

    records = [record(1e-3, 1, 0.9), record(1e-4, 0.75, 1), record(1e-5, 0.5, 1)]
    assert choose_recipe(records, 0.995)["learning_rate"] == 1e-4
    assert choose_recipe(records[:1], 0.995)["retention_threshold_met"] is False
    # Without an eligible candidate, old retention has priority over atomic accuracy.
    assert (
        choose_recipe([record(1e-3, 1, 0.9), record(1e-4, 0.5, 0.98)], 0.995)["learning_rate"]
        == 1e-4
    )
    with pytest.raises(ValueError, match="only atomic"):
        choose_recipe([{**records[0], "two_hop_accuracy": 1}], 0.995)


def test_saved_scope_mismatch_rejected(reader):
    values = {
        name: reader.backbone.get_parameter(name).detach().clone()
        for name in parameter_names(reader.backbone, "down")
    }
    with pytest.raises(ValueError, match="exactly"):
        load_edit(reader, values, "all")
