import json

import pytest

from llm_memory_editability import bios_original_diagnostics as diagnostics
from llm_memory_editability.bios_original_data import make_people
from llm_memory_editability.bios_original_diagnostics import (
    audit_batches,
    decode_operation,
    operation_prompt,
    parse_date,
    read_selected_batches,
    select_people,
    symbolic_operation,
)


def test_pair_selection_is_development_only_and_complete():
    people = make_people(142701, 1000)
    selected = select_people(people, 142701)
    assert len(selected) == len(set(selected)) == 24
    assert all(people[i]["split"] == "dev" for i in selected)
    assert all(people[i]["partner"] in selected for i in selected)
    assert selected == select_people(people, 142701)


def test_oracle_and_autonomous_share_exact_rule_prompts():
    assert operation_prompt("parity", "March 2, 1971") == (
        "Question: Is March an even-numbered month?\nAnswer:"
    )
    assert operation_prompt("comparison", "May 4, 1999", "June 1, 2000") == (
        "Question: Which date is earlier, May 4, 1999 or June 1, 2000?\nAnswer:"
    )
    # A malformed extraction is retained as input, never repaired with ground truth.
    assert operation_prompt("parity", "I don't know") == (
        "Question: Is I don't know an even-numbered month?\nAnswer:"
    )
    assert parse_date("March 31, 2001") is None
    assert parse_date("date: March 2, 2001; Answer: March 2, 2001") is None


def test_date_mapping_does_not_recover_truth_from_wrong_model_choice():
    names = ("A", "B")
    assert decode_operation("comparison", "May 1, 2000", "May 1, 2000", "May 1, 1990", names) == "A"
    assert symbolic_operation("comparison", "May 1, 2000", "May 1, 1990", names) == "B"
    assert (
        decode_operation("comparison", "May 1, 2000", "May 1, 2000", "May 1, 2000", names) is None
    )
    assert symbolic_operation("comparison", "bad", "May 1, 1990", names) is None
    assert symbolic_operation("parity", "May 1, 1990") == "no"


def test_audit_replays_complete_original_batches(tmp_path):
    people = [dict(split="train"), dict(split="test"), dict(split="dev")]
    assert audit_batches(people, ["date"], False, 4) == [0, 4]
    path = tmp_path / "rows.jsonl"
    path.write_text("".join(json.dumps(dict(value=i)) + "\n" for i in range(12)))
    batches = read_selected_batches(path, [0, 8], 4)
    assert [[r["source_row"] for r in b["rows"]] for b in batches] == [[0, 1, 2, 3], [8, 9, 10, 11]]
    with pytest.raises(ValueError, match="Incomplete"):
        read_selected_batches(path, [10], 4)


def test_complete_diagnostic_keeps_bad_extraction_and_all_cases(tmp_path, monkeypatch):
    people = [
        dict(id=0, name="A", partner=1, year=1990, month=3, day=1),
        dict(id=1, name="B", partner=0, year=2000, month=4, day=1),
    ]
    path = tmp_path / "people.json"
    path.write_text(json.dumps(people))
    plan = dict(worlds={"1": dict(people_path=str(path), person_ids=[0, 1])}, views=[0])

    def fake_generate(model, tokenizer, rows, plan, device):
        for row in rows:
            if "mode" not in row:
                output = "bad date" if row["person_id"] == 0 else "May 1, 2010"
            else:
                output = "yes" if row["task"] == "parity" else "March 1, 1990"
            row.update(generated=output, eos=True)

    monkeypatch.setattr(diagnostics, "generate_rows", fake_generate)
    result = diagnostics.run_diagnostics(
        None, None, dict(world=1, condition="S"), plan, tmp_path, "cpu"
    )
    assert result["rows"] == 16
    assert result["extraction_rows"] == 2
    rows = [json.loads(line) for line in (tmp_path / "diagnostics.jsonl").read_text().splitlines()]
    automatic = [r for r in rows if r["mode"] == "autonomous_rule" and r["person_id"] == 0]
    assert all("bad date" in r["prompt"] for r in automatic)
    assert all(not r["inputs_parseable"] and not r["inputs_correct"] for r in automatic)
    assert result["totals"]["comparison/view-0/autonomous_rule"]["n"] == 2
    assert result["totals"]["comparison/view-0/autonomous_rule"]["inputs_parseable_n"] == 0
