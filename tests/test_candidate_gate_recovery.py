"""Do not turn a free GPU into an extra user GPU allocation."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from recover_candidate_update_gate import capacity_available, dependency_check  # noqa: E402


@pytest.mark.parametrize(
    "current,selected,allowed",
    [
        ({"a", "b", "c", "d"}, "e", False),
        ({"a", "b", "c"}, "d", True),
        ({"a", "b", "c", "d"}, "d", True),
    ],
)
def test_capacity(current, selected, allowed):
    assert capacity_available(current, selected, 4) is allowed


def test_predecessor_requires_successful_train_and_grid(tmp_path):
    dependency = {"root": str(tmp_path), "arm": "parent", "config_sha256": "frozen"}
    assert not dependency_check.predecessor_ready(dependency)
    for stage in ["train", "grid"]:
        folder = tmp_path / "execution/parent" / stage
        folder.mkdir(parents=True)
        (folder / "exit.json").write_text(json.dumps({"code": 0}))
    run = tmp_path / "runs/parent"
    run.mkdir(parents=True)
    complete = {
        "state": "complete",
        "step": 1500000,
        "budget": 1500000,
        "arm": "parent",
        "model_sha256": "model",
        "config_sha256": "frozen",
        "world_sha256": "world",
    }
    (run / "complete.json").write_text(json.dumps(complete))
    (run / "grid.json").write_text(json.dumps(complete))
    assert dependency_check.predecessor_ready(dependency)
    (tmp_path / "execution/parent/grid/exit.json").write_text('{"code": 1}')
    with pytest.raises(RuntimeError, match="failed"):
        dependency_check.predecessor_ready(dependency)
