"""Contracts of the pre-registered hop-position analysis (no model outputs needed)."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from llm_memory_editability.mquake_reproduction import queries

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "analyze_path_hop_position", ROOT / "scripts/analyze_path_hop_position.py"
)
hop = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hop)


def mquake_case(case_id, edited_hop):
    """Two-hop chain a -r1-> b -r2-> c with one counterfactual edit at hop 1 or 2."""
    triples = [["a", "r1", "b"], ["b", "r2", "c"]]
    labeled = [["A", "R1", "B"], ["B", "R2", "C"]]
    if edited_hop == 1:
        new = [["a", "r1", "x"], ["x", "r2", "y"]]
        new_labeled = [["A", "R1", "X"], ["X", "R2", "Y"]]
        request = {"subject": "A", "relation_id": "r1", "target_new": {"str": "X", "id": "x"}}
        request["target_true"] = {"str": "B", "id": "b"}
    else:
        new = [["a", "r1", "b"], ["b", "r2", "y"]]
        new_labeled = [["A", "R1", "B"], ["B", "R2", "Y"]]
        request = {"subject": "B", "relation_id": "r2", "target_new": {"str": "Y", "id": "y"}}
        request["target_true"] = {"str": "C", "id": "c"}
    request["prompt"] = "{} is"
    hop_text = [{"question": "q", "answer": "B", "answer_alias": []}] * 2
    return {
        "case_id": case_id,
        "requested_rewrite": [request],
        "questions": ["q1", "q2", "q3"],
        "answer": "C",
        "answer_alias": [],
        "new_answer": "Y",
        "new_answer_alias": [],
        "single_hops": hop_text,
        "new_single_hops": hop_text,
        "orig": {
            "triples": triples,
            "triples_labeled": labeled,
            "new_triples": new,
            "new_triples_labeled": new_labeled,
            "edit_triples": [],
        },
    }


def mquake_record(case, direct, cot, baseline=True):
    def rows(correct, text):
        return [{"correct": correct, "answer_text": text}] + [{"correct": False, "answer_text": ""}]

    def phase(multi, chain_cot):
        return {
            "predictions": {
                "edit": [{"correct": True}],
                "single": [{"correct": True}, {"correct": True}],
                "multi": rows(multi, "Y" if multi else "C"),
                "cot": rows(chain_cot, ""),
            }
        }

    return {
        "case_id": case["case_id"],
        "case_sha256": hashlib.sha256(json.dumps(case, sort_keys=True).encode()).hexdigest(),
        "baseline": phase(baseline, baseline),
        "edited": phase(direct, cot),
    }


def test_edit_position_uses_reproduction_matching():
    prompts = {"relations": {"r1": "", "r2": ""}, "multi": "", "cot": ""}
    for edited_hop in (1, 2):
        case = mquake_case(edited_hop, edited_hop)
        assert hop.mquake_edit_positions(case) == [edited_hop]
        assert len(queries(case, prompts, True)["edit"]) == 1
    assert hop.mquake_group(mquake_case(0, 1)) == "first"
    assert hop.mquake_group(mquake_case(0, 2)) == "later"


def test_gap_interaction_and_old_answer():
    first = [mquake_case(i, 1) for i in range(4)]
    later = [mquake_case(10 + i, 2) for i in range(4)]
    rows = [hop.mquake_row(c, mquake_record(c, i < 3, i < 2)) for i, c in enumerate(first)]
    rows += [hop.mquake_row(c, mquake_record(c, i < 1, i < 2)) for i, c in enumerate(later)]
    gaps = hop.mquake_gaps(rows, [2])
    assert gaps == {"direct": 0.5, "cot": 0.0, "interaction": 0.5}
    cells = hop.cell_summary(rows)
    assert cells["2-hop/later"]["old_answer_retained"] == 0.75
    assert hop.mquake_gaps([r for r in rows if r["group"] == "first"], [2]) is None


def test_bootstrap_is_deterministic():
    rows = [{"hops": 2, "value": v} for v in [0, 1, 1, 0, 1]]
    settings = {"iterations": 200, "seed": 3, "interval": [0.025, 0.975]}

    def statistic(sample):
        return {"mean": sum(r["value"] for r in sample) / len(sample)}

    first = hop.bootstrap(rows, statistic, settings, "hops")
    assert first == hop.bootstrap(rows, statistic, settings, "hops")
    assert first["mean"]["low"] <= 0.6 <= first["mean"]["high"]
    assert hop.bootstrap([], statistic, settings) == {}


def ripple_case():
    query = {"query_type": "two_hop", "target_ids": ["t"]}
    return {
        "edit": {"subject_id": "s"},
        "Compositionality_I": [
            {"test_queries": [dict(query, subject_id="s")]},
            {"test_queries": [dict(query, subject_id="other")]},
        ],
        "Compositionality_II": [
            {"test_queries": [dict(query, subject_id="u", target_ids=["s"])]},
            {"test_queries": [dict(query, subject_id="s", target_ids=["s"])]},
        ],
    }


def test_ripple_structure_and_counts():
    case = ripple_case()
    assert hop.ripple_structure(case, "CI") == [True, False]
    assert hop.ripple_structure(case, "CII") == [True, False]
    result = {"edit_success": True, "outcomes": {"PASSED": [0, 1], "FAILED": []}}
    record = {"case": case, "axes": {"CI": {"result": result}, "CII": {"result": result}}}
    assert hop.ripple_case(record) == {"CI": (1, 1), "CII": (1, 1)}
    record["axes"]["CII"]["result"] = dict(result, edit_success=False)
    assert hop.ripple_case(record) == {"CI": (1, 1)}
    sample = [{"CI": (3, 4), "CII": (1, 4)}, {"CI": (1, 4)}]
    assert hop.ripple_gap(sample) == {"CI": 0.5, "CII": 0.25, "CI_minus_CII": 0.25}


def test_analysis_refuses_unaudited_runs(tmp_path):
    with pytest.raises(SystemExit):
        hop.require_audited(tmp_path)
    (tmp_path / "complete.json").write_text("{}")
    (tmp_path / "audit.json").write_text(json.dumps({"passed": False}))
    with pytest.raises(SystemExit):
        hop.require_audited(tmp_path)
    (tmp_path / "audit.json").write_text(json.dumps({"passed": True}))
    hop.require_audited(tmp_path)


def test_mquake_end_to_end(tmp_path):
    cases = [mquake_case(i, 1 + i % 2) for i in range(8)]
    data = tmp_path / "MQuAKE.json"
    data.write_text(json.dumps(cases))
    for index, case in enumerate(cases, 1):
        directory = tmp_path / "run/cases" / f"{index:05d}"
        directory.mkdir(parents=True)
        record = mquake_record(case, case["case_id"] % 2 == 0, True)
        (directory / "record.json").write_text(json.dumps(record))
    config = {
        "bootstrap": {"iterations": 50, "seed": 1, "interval": [0.025, 0.975]},
        "mquake": {"data_sha256": hop.sha(data), "max_cases": 8, "primary_hop_counts": [2]},
    }
    result = hop.analyze_mquake(config, tmp_path / "run", data)
    estimate = result["primary_full_pool"]["estimate"]
    assert estimate == {"direct": 1.0, "cot": 0.0, "interaction": 1.0}
    assert result["secondary_prerequisite_subset"]["coverage"] == 1.0


def test_frozen_contract_counts_are_complete():
    config = json.loads((ROOT / "configs/path-hop-analysis-v1.json").read_text())
    cells = config["mquake"]["data_only_cell_counts"]
    assert sum(cells.values()) == config["mquake"]["max_cases"]
    for hops in config["mquake"]["primary_hop_counts"]:
        assert cells[f"{hops}-hop/edited-hop-1"] > 0
        assert any(k.startswith(f"{hops}-hop/edited-hop-") and not k.endswith("-1") for k in cells)
    counts = config["rippleedits"]["data_only_counts"]
    assert set(counts) == set(config["rippleedits"]["data_sha256"])
    for subset in counts.values():
        assert subset["CI"]["structure_valid_tests"] == subset["CI"]["tests"]
