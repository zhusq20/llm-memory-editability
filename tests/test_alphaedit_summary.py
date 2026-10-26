"""Teacher-free reporting regression for an atomic-only editing experiment."""

import copy

from llm_memory_editability.alphaedit_summary import summarize


def test_summary_accepts_complete_edit_records_without_a_ttcd_teacher():
    atom = {
        "final_correct": [True],
        "stages": {"round1_end": {"answer": {"expected_correct": [True]}}},
    }
    hop = {
        "final_correct": [True, False],
        "stages": {
            "round1_end": {
                "answer": {"expected_correct": [True, False]},
                "bridge": {"expected_correct": [True, True]},
            }
        },
        "companion": {"final_correct": [True, True]},
        "explicit_two_step": {"final_correct": [True, True]},
    }
    old = {"final_correct": [True], "answer": [1], "stop": [28]}
    state = {
        "novel_atom": atom,
        "first_hop": hop,
        "second_hop": hop,
        **{name: old for name in ("U_atoms", "U_chains", "old_matched_atom", "old_matched_chains")},
    }
    records = [{"before": copy.deepcopy(state), "after": copy.deepcopy(state)} for _ in range(2)]
    report = summarize(records, {"bootstrap_seed": 7, "bootstrap_draws": 50})
    assert "teacher_record_visible" not in report
    assert report["conditions"]["after"]["first_hop"]["accuracy"] == 0.5
    assert report["retention"]["U_atoms"]["initially_correct_retained"]["accuracy"] == 1
