import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "prepare_knowledge_use_coverage", ROOT / "scripts/prepare_knowledge_use_coverage.py"
)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def fixture_paths(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"development_pairs": 2, "formal_pairs": 4, "seed": 73}))
    batch = tmp_path / "factorial"
    source = batch / "runs/sft_full/paper-eval/auxiliary.json"
    source.parent.mkdir(parents=True)
    rows = [
        {
            "id": f"ret/capital_rev/City-{i}",
            "prompt": "<|im_start|>system\nIgnore this system text.<|im_end|>\n"
            f"<|im_start|>user\nFrozen question number {i}?<|im_end|>\n"
            "<|im_start|>assistant\nDo not copy me.",
            "gold": "Paris",
            "type": "city",
            "exclude": ["London"],
            "answer": "SENTINEL_OLD_ANSWER",
            "strict": False,
            "token_ids": [999999],
        }
        for i in range(317)
    ]
    source.write_text(json.dumps({"retention": {"before": rows}}))
    return config, batch, source


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_old_retention_reuses_questions_and_gold_only(tmp_path):
    _, _, source = fixture_paths(tmp_path)
    rows = M.natural_retention_rows(source, "development")
    assert len(rows) == 317
    assert rows[0]["question"] == "Frozen question number 0?"
    assert rows[0]["gold"] == "Paris" and rows[0]["aliases"] == ["Paris"]
    assert rows[0]["type"] == "city" and rows[0]["exclude"] == ["London"]
    assert rows[0]["original_source_id"] == "ret/capital_rev/City-0"
    assert {r["cluster_id"] for r in rows} == {"natural-retention-country/Paris"}
    assert all(r["mode"] == "direct" and r["category"] == "natural_old_retention" for r in rows)
    assert all(not {"answer", "strict", "token_ids", "em", "f1"} & r.keys() for r in rows)
    assert "SENTINEL_OLD_ANSWER" not in json.dumps(rows)


def test_unidentifiable_source_ids_are_explicit_unknown_not_iid_questions(tmp_path):
    _, _, source = fixture_paths(tmp_path)
    content = json.loads(source.read_text())
    content["retention"]["before"][0]["id"] = "unidentifiable/0"
    source.write_text(json.dumps(content))
    first = M.natural_retention_rows(source, "formal")[0]
    assert first["cluster_id"] is None and first["cluster_status"] == "unknown"


@pytest.mark.parametrize(
    "prompt",
    [
        "Question without a chat role",
        "<|im_start|>user\n<|im_end|>",
        "<|im_start|>user\nOne?<|im_end|><|im_start|>user\nTwo?<|im_end|>",
    ],
)
def test_question_extraction_fails_closed_on_ambiguous_prompts(prompt):
    with pytest.raises(ValueError, match="exactly one"):
        M.user_question(prompt)


def test_complete_materialization_and_rewrite_baseline(tmp_path):
    config, batch, _ = fixture_paths(tmp_path)
    output = tmp_path / "prepared"
    audit = M.materialize(config, output, batch)
    assert audit["status"] == "complete_cpu_materialization"
    assert audit["cohort_isolation"]["new_entities_disjoint"]
    assert not audit["training_performed"] and not audit["teacher_sampling_performed"]
    for cohort, pairs in (("development", 2), ("formal", 4)):
        folder = output / "data" / cohort
        assert len(read_jsonl(folder / "train-A.jsonl")) == 2 * pairs
        assert len(read_jsonl(folder / "train-C.jsonl")) == 16 * pairs
        assert len(read_jsonl(folder / "evaluation-rewrite.jsonl")) == 32 * pairs
        before = read_jsonl(folder / "retention-addition.jsonl")
        after = read_jsonl(folder / "retention-rewrite.jsonl")
        before_locations = {
            r["id"]: r for r in before if r["category"] == "learned_location_baseline"
        }
        after_locations = {
            r["id"]: r for r in after if r["category"] == "retention_unchanged_new_location"
        }
        assert len(before_locations) == 4 * pairs
        assert before_locations.keys() == after_locations.keys()
        assert all(not r["is_old_retention"] for r in before_locations.values())
        assert all(
            r["retention_reference"] == "addition_endpoint" for r in after_locations.values()
        )
        assert all(
            before_locations[key]["gold"] == row["gold"] for key, row in after_locations.items()
        )
        for name, metadata in audit["files"][cohort].items():
            assert M.sha256(folder / name) == metadata["sha256"]
            if name.endswith(".jsonl"):
                rows = read_jsonl(folder / name)
                assert len({r["id"] for r in rows}) == len(rows)
    assert json.loads((output / "audit.json").read_text())["config_sha256"] == M.sha256(config)


def test_existing_output_is_never_overwritten(tmp_path):
    config, batch, _ = fixture_paths(tmp_path)
    output = tmp_path / "prepared"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("keep")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        M.materialize(config, output, batch)
    assert marker.read_text() == "keep" and list(output.iterdir()) == [marker]


def test_wrong_natural_pool_fails_before_creating_output(tmp_path):
    config, batch, source = fixture_paths(tmp_path)
    source.write_text(json.dumps({"retention": {"before": []}}))
    output = tmp_path / "prepared"
    with pytest.raises(ValueError, match="317"):
        M.materialize(config, output, batch)
    assert not output.exists()


def test_cross_cohort_leak_and_duplicate_ids_are_rejected():
    worlds = {c: M.D.make_world(c, 2) for c in ("development", "formal")}
    rows = {c: {"train-A.jsonl": M.D.training_rows(w, "A")} for c, w in worlds.items()}
    rows["development"]["train-A.jsonl"][0]["question"] = worlds["formal"].pairs[0].organization
    with pytest.raises(ValueError, match="cross-cohort"):
        M.validate_cohort_isolation(worlds, rows)
    duplicate = M.D.training_rows(worlds["development"], "A")[0]
    with pytest.raises(ValueError, match="duplicate IDs"):
        M.validate_rows("broken.jsonl", [duplicate, duplicate])
