"""Contracts for architecture-independent endpoint scoring and paired coverage."""

import copy
import sys
from pathlib import Path

import pytest
import torch
from transformers import GPT2Config

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_ttcd_novel_fact_transfer as readouts  # noqa: E402

from llm_memory_editability.ce_architecture import (  # noqa: E402
    inspect_endpoints,
    old_retention_rates,
    paired_report,
    summarize_endpoints,
)
from llm_memory_editability.ce_scope_editing import (  # noqa: E402
    atomic_ce_edit,
    parameter_names,
    restore,
    teacher_batches,
)
from llm_memory_editability.supervised_alignment import SupervisedAlignmentGPT  # noqa: E402
from llm_memory_editability.ttcd_editing import TTCDReader, tensor_digest  # noqa: E402

META = {"entity_offset": 0, "relation_offset": 10, "end_marker": 28, "pad_token": 27}
STAT = {"bootstrap_seed": 1, "bootstrap_draws": 100}


def test_zero_baseline_old_pool_is_na_and_does_not_raise_retention_score():
    rates, minimum = old_retention_rates(
        {
            "test_oo": {"retained": 0, "originally_correct": 0},
            "atomic_id": {"retained": 99, "originally_correct": 100},
        }
    )
    assert rates["test_oo"] is None
    assert minimum == 0.99
    with pytest.raises(ValueError, match="undefined"):
        old_retention_rates({"test_oo": {"retained": 0, "originally_correct": 0}})


@pytest.mark.parametrize("layers,repeats", [(4, 1), (2, 2)])
def test_plain_architecture_endpoint_matches_inherited_greedy_and_preserves_model(layers, repeats):
    torch.set_num_threads(1)
    torch.manual_seed(9)
    cfg = GPT2Config(
        vocab_size=29,
        n_positions=8,
        n_embd=8,
        n_head=2,
        n_layer=layers,
        resid_pdrop=0,
        embd_pdrop=0,
        attn_pdrop=0,
    )
    reader = TTCDReader(
        SupervisedAlignmentGPT(cfg, repeats, discoloop=False),
        native_causal_attention=True,
    ).eval()
    before = tensor_digest(reader.state_dict())
    for rows in ([[0, 0, 2], [2, 1, 3]], [[0, 0, 1, 3], [2, 1, 0, 1]]):
        expected = readouts.greedy_rows(reader, reader.empty_fast(), rows, META)
        actual = inspect_endpoints(
            reader,
            reader.empty_fast(),
            rows,
            META,
            [r[-1] for r in rows],
            kind="test",
            bridges=[2, 3],
        )
        assert actual["answer"] == expected["answer"]
        assert actual["stop"] == expected["stop"]
        assert actual["final_new_joint"] == expected["final_correct"]
        assert actual["stages"] == {}
    assert tensor_digest(reader.state_dict()) == before
    from types import SimpleNamespace

    import numpy as np

    world = SimpleNamespace(
        metadata=META,
        atoms=np.array([[0, 0, 2], [2, 1, 3]]),
        chains=np.array([[0, 0, 1, 3], [2, 1, 0, 1]]),
    )
    settings = dict(
        steps=2,
        learning_rate=0.002,
        kl_factor=10.0,
        norm_constraint=0.0005,
        gradient_clip=1.0,
        old_per_kind_per_step=1,
        batch_seed=4,
    )
    teachers = teacher_batches(
        reader,
        world,
        {"protection_atoms": [0, 1], "protection_chains": [0, 1]},
        settings,
    )
    base = copy.deepcopy(reader.backbone.state_dict())
    values, info = atomic_ce_edit(reader, [1, 2, 4], META, "mlp", settings, teachers)
    assert len(parameter_names(reader.backbone, "mlp")) == 4 * layers
    assert info["steps"] == 2 and info["updated_composite_supervision"] == 0
    for name, parameter in reader.backbone.named_parameters():
        if name not in values:
            assert torch.equal(parameter, base[name])
    restore(reader, base)
    assert tensor_digest(reader.state_dict()) == before


def records(success=True):
    result = []
    for index in range(32):
        item = {"fact": {"case_id": index}, "before": {}, "after": {}}
        for condition in ("before", "after"):
            for kind in ("novel_atom", "first_hop", "second_hop"):
                row = {"final_correct": [success if condition == "after" else False]}
                if kind != "novel_atom":
                    row.update(
                        companion={"final_correct": [True]},
                        explicit_two_step={"final_correct": [success]},
                    )
                item[condition][kind] = row
            for kind in (
                "U_atoms",
                "U_chains",
                "old_matched_atom",
                "old_matched_chains",
            ):
                item[condition][kind] = {"final_correct": [True]}
            item[condition]["extended_old"] = {
                "atomic_id": {"final_correct": [True]},
                "test_oo": {"final_correct": [True]},
            }
        result.append(item)
    return result


def test_unsuccessful_writes_have_zero_coverage_and_na_conditional_accuracy():
    summary = summarize_endpoints(records(False), STAT)
    outcome = summary["conditional"]["second_hop"]["both_atoms_current_correct"]
    assert outcome["coverage_n"] == outcome["n"] == 0
    assert outcome["accuracy"] is None


def test_common_prerequisites_require_all_architectures_and_retain_full_results():
    data = {"discoloop": records(), "loop": records(), "transformer": records(False)}
    report = paired_report(data, STAT)
    assert report["common_prerequisites"]["second_hop"]["coverage_n"] == 0
    assert (
        report["architectures"]["discoloop"]["conditions"]["after"]["second_hop"]["correct"] == 32
    )
    assert report["architectures"]["transformer"]["conditions"]["after"]["second_hop"]["n"] == 32
    assert (
        report["paired_fact_differences"]["discoloop_minus_transformer"]["second_hop"]["fact_mean"]
        == 1.0
    )


def test_pairing_rejects_changed_fact_panels():
    left, right = records(), records()
    right[0]["fact"]["case_id"] = 99
    with pytest.raises(AssertionError, match="identical32"):
        paired_report({"loop": left, "transformer": right}, STAT)
