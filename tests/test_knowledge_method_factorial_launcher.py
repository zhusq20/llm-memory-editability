import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "factorial_launcher", ROOT / "scripts/launch_knowledge_method_factorial.py"
)
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


def plan():
    return json.loads(M.DEFAULT_PLAN.read_text())


def test_eight_arms_and_disjoint_thousand_case_coverage():
    cfg = M.validate(plan())
    assert cfg["expected_training_rows"] == 297840
    assert cfg["expected_training_rows"] // cfg["batch_size"] * cfg["epochs"] == 372300
    bad = plan()
    bad["arms"] = bad["arms"][:-1]
    with pytest.raises(ValueError, match="eight arms"):
        M.validate(bad)


def test_new_container_identity_and_old_batch_read_only():
    p = plan()
    names = []
    for arm in M.ARMS:
        name, argv = M.command(p, "train", arm, p["lanes"][0])
        names.append(name)
        assert name.startswith("kmf-v1-") and not name.startswith("kmd-v1-")
        assert argv[-2:] == ["--arm", arm]
        assert any("target=/old-batch,readonly" in value for value in argv)
        assert any(
            p["runtime"]["deps"] in value and "target=/batch/deps,readonly" in value
            for value in argv
        )
        assert p["wandb_token_file"] not in " ".join(argv)
    assert len(set(names)) == 8


def test_preparation_and_regrading_use_correct_paths():
    p = plan()
    _, prep = M.command(p, "prepare-shard", "shard-1", p["lanes"][1])
    assert prep[-2:] == ["--shard", "shard-1"]
    _, grade = M.command(p, "regrade-old", "regrade-development", p["lanes"][0])
    assert grade[-4:] == ["--batch", "/old-batch", "--output", "/batch/regrade-development"]
