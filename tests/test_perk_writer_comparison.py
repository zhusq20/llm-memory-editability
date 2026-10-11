import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "writer_execution", ROOT / "scripts/execute_perk_writer_comparison.py"
)
executor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(executor)


def test_authorized_matrix_reuses_one_common_baseline():
    plan = json.loads((ROOT / "configs/perk-writer-comparison-v1.json").read_text())
    rows = executor.scientific_tasks(plan)
    assert len(rows) == 5 and sum(condition == "fixed" for condition, _ in rows) == 1
    assert (
        sum(plan["outer_updates_per_arm"] for condition, _ in rows if condition != "fixed") == 8192
    )
    plan["checkpoint_rule"] = "best_test_accuracy"
    with pytest.raises(ValueError, match="Test-dependent"):
        executor.scientific_tasks(plan)


def test_prepared_hash_change_prevents_execution(tmp_path):
    source = tmp_path / "source-preparation"
    source.mkdir()
    file = source / "source.py"
    file.write_text("fixed")
    plan = tmp_path / "plan-preparation.json"
    plan.write_text("{}")
    executor.write(
        tmp_path / "preparation-manifest.json",
        {
            "plan_sha256": executor.sha(plan),
            "source_files": {"source.py": executor.sha(file)},
            "data": {"files": {}},
        },
    )
    executor.verify_preparation(tmp_path)
    file.write_text("changed")
    with pytest.raises(ValueError, match="Prepared source changed"):
        executor.verify_preparation(tmp_path)


def test_no_duplicate_controller_submission(tmp_path):
    (tmp_path / "controller-process.json").write_text("{}")
    with pytest.raises(RuntimeError, match="registration already"):
        executor.launch(tmp_path)


def test_fixed_condition_and_budget_cannot_silently_change():
    plan = json.loads((ROOT / "configs/perk-writer-comparison-v1.json").read_text())
    plan["outer_updates_per_arm"] += 1
    with pytest.raises(ValueError, match="budget differs"):
        executor.scientific_tasks(plan)
    plan["outer_updates_per_arm"] -= 1
    plan["conditions"].append("loop")
    with pytest.raises(ValueError, match="three conditions"):
        executor.scientific_tasks(plan)


def test_paired_world_analysis_and_actual_material_streams(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "writer_analysis", ROOT / "scripts/analyze_perk_writer_comparison.py"
    )
    analysis = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analysis)
    effect = analysis.paired_difference({"a": 1.0, "b": 0.5}, {"a": 0.5, "b": 0.0}, 100)
    assert effect["worlds"] == 2 and effect["difference_pp"] == 50.0
    assert effect["world_bootstrap_95_pp"] == [50.0, 50.0]
    with pytest.raises(ValueError, match="Paired worlds"):
        analysis.paired_difference({"a": 1.0}, {"b": 1.0})
    first, second = tmp_path / "first.jsonl", tmp_path / "second.jsonl"
    first.write_text(
        json.dumps({"outer_updates": 1, "material_id": "a", "material_sha256": "b"}) + "\n"
    )
    second.write_bytes(first.read_bytes())
    assert analysis.paired_material_stream(first, second) == 1
    second.write_text(first.read_text().replace('"b"', '"c"'))
    with pytest.raises(ValueError, match="streams differ"):
        analysis.paired_material_stream(first, second)


def test_extension_masks_padding_and_retains_full_eos():
    torch = pytest.importorskip("torch")
    from llm_memory_editability.perk_writer_experiment import encode_training_rows

    class Tokenizer:
        eos_token = "!"
        pad_token_id = 0
        model_max_length = 100

        def encode(self, text, add_special_tokens):
            return [ord(c) for c in text]

    batch = encode_training_rows(Tokenizer(), [("p:", "short"), ("p:", "longer answer")])
    assert (batch["labels"][batch["attention_mask"] == 0] == -100).all()
    assert (batch["labels"][:, :2] == -100).all()
    for index in range(2):
        length = int(batch["attention_mask"][index].sum())
        assert int(batch["labels"][index, length - 1]) == ord("!")
        assert batch["attention_mask"][index, 0] == 1
    assert torch.equal(batch["input_ids"][:, :2], torch.tensor([[112, 58], [112, 58]]))
