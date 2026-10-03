"""Changed truth, untouched closure, output positions and local edit scope."""

import copy

import numpy as np
import torch

from llm_memory_editability.grokking_dynamics_mechanism import (
    PARAMETER,
    answer_batch,
    editor,
    select_cases,
    select_donors,
    serialize_case,
    taught_atoms,
)
from llm_memory_editability.latent_scaling import build_world, construct
from llm_memory_editability.text_pretrain import atomic_sentence, composite_sentence


def sample_spec():
    return {
        "world": 103,
        "heads_n": 32,
        "bridges_n": 32,
        "tails_n": 16,
        "familiar_n": 8,
        "strict_n": 4,
        "holdout_fraction": 0.25,
        "low_extra": "anchors",
        "anchor_n": 4,
        "composition_count": "all",
        "width": 8,
        "layers": 1,
        "repeats": 2,
        "heads": 2,
        "dropout": 0,
        "initialization": 104,
    }


def test_cases_are_graph_selected_and_all_update_truth_and_untouched_paths_are_correct():
    world = build_world(sample_spec())
    cases = select_cases(world)
    assert len(cases) == 8
    assert [serialize_case(c) for c in cases] == [serialize_case(c) for c in select_cases(world)]
    atoms = taught_atoms(world)
    for case in cases:
        lookup = {tuple(map(int, row[:2])): int(row[2]) for row in atoms}
        old, new = case["old_fact"], case["new_fact"]
        assert old[:2] == new[:2] and old[2] != new[2]
        lookup[tuple(new[:2])] = new[2]
        for split in ["familiar", "strict"]:
            for name in ["D_" + split, "U_" + split]:
                for h, r1, b, r2, t in case["tasks"][name]:
                    assert lookup[int(h), int(r1)] == b
                    assert lookup[int(b), int(r2)] == t
            assert len(case["tasks"]["D_" + split]) + len(case["tasks"]["U_" + split]) == len(
                world[split + "_test"]
            )
        e, r, k, u = (
            {case["atomic_index"]},
            set(case["replay_indices"]),
            set(case["keep_indices"]),
            set(case["unused_indices"]),
        )
        assert not (e & r or e & k or e & u or r & k or r & u or k & u)
        assert e | r | k | u == set(range(len(atoms)))


def test_template_answer_positions_and_pure_first_hop_donor_information():
    world = build_world(sample_spec())
    for rows, start in [(world["common_atomic"][:3], 4), (world["familiar_test"][:3], 6)]:
        _tokens, positions, labels = answer_batch(rows, "cpu")
        assert positions[:, 0].eq(start).all()
        np.testing.assert_array_equal(labels[:, 0], rows[:, -1])
        assert labels[:, 1].eq(5).all() and labels[:, 2].eq(1).all()
    rows = world["familiar_test"]
    for style, values in select_donors(world, rows).items():
        for row, index, tail in zip(rows, values["indices"], values["tails"], strict=True):
            if index < 0:
                continue
            atom = world["common_atomic"][index]
            assert atom[1] == row[1]
            assert 13 <= atom[1] <= 16 and 17 <= row[3] <= 20
            prefix = atomic_sentence(atom)[:5]
            assert row[3] not in prefix and tail not in prefix
            assert len(prefix) == 5 and len(composite_sentence(row)[:7]) == 7
            assert (atom[2] == row[2]) == (style == "same_bridge")


def test_editor_only_changes_local_down_weight_and_exact_tensor_replays():
    world = build_world(sample_spec())
    model = construct(sample_spec(), "cpu").eval()
    before = copy.deepcopy(model.state_dict())
    case = select_cases(world, n_per_group_role=1, calibration=True)[0]
    record, raw, final = editor(model, case, "cpu", "edit", 0.001, nodes=(0, 1))
    assert record["changed_tensors"] == [PARAMETER]
    assert record["exact_weight_reload_passed"]
    assert final.shape == before[PARAMETER].shape
    assert "step1_D_familiar_generated" in raw
    assert all(torch.equal(model.state_dict()[key], value) for key, value in before.items())
