"""No contradictory old composites or hidden direct edit targets in preservation."""

import torch

from llm_memory_editability.bios_direction_cross import cross_tasks


def test_cross_edit_closure_and_information_partitions():
    tasks = cross_tasks(0, 0, "cpu")
    assert len(tasks) == 12
    for task in tasks:
        sets = task.sets
        assert len(sets["E"]) == 2 and len(sets["R"]) == 48 and len(sets["V"]) == 48
        assert len(sets["D"]) == 32 and len(sets["D_heldout"]) == 16
        assert set(sets["D_focal"]) <= set(sets["D_heldout"])
        all_ids = sum((sets[k] for k in ("E", "R", "V", "U", "D")), [])
        assert len(all_ids) == len(set(all_ids)) == len(task.labels)
        assert torch.equal(task.labels[sets["R"]], task.old_labels[sets["R"]])
        assert torch.equal(task.labels[sets["U"]], task.old_labels[sets["U"]])
        assert torch.all(task.labels[sets["E"], 0] != task.old_labels[sets["E"], 0])
        assert torch.all(task.labels[sets["D"], 0] == task.metadata["a"])
        if task.metadata["kind"] == "independent":
            assert int(task.labels[sets["E"][1], 0]) == task.metadata["b"]


def test_bounded_covariance_matches_original_operation():
    from llm_memory_editability.bios_direction import (
        batch_setup,
        minimal_task,
        representation_geometry,
    )
    from llm_memory_editability.bios_direction_cross import bounded_representation_geometry
    from llm_memory_editability.bios_model import CausalLM, ModelConfig

    torch.manual_seed(41)
    model = CausalLM(ModelConfig(41, width=8, layers=2, heads=2, context=8)).double()
    for p in model.parameters():
        p.requires_grad_(False)
    task = minimal_task(0, 0, "independent", "cpu")
    suffix, labels, ew, rw, ref, cov, null, ranks, spectrum, w0 = batch_setup(
        model, 0, [task], [{}]
    )
    actual = bounded_representation_geometry(suffix, (rw > 0).double(), w0)
    expected = representation_geometry(suffix, (rw > 0).double(), w0)
    torch.testing.assert_close(actual[0], expected[0], rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(actual[1], expected[1], rtol=1e-5, atol=1e-6)
    assert torch.equal(actual[2], expected[2])
