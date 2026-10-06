"""A scheduling amendment cannot change treatments or overlap the running batch."""

import copy
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "oo_parallel", ROOT / "scripts/execute_oo_parallel.py"
)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def configs():
    base = module.read(ROOT / "configs/oo-optimizer-attribution-v1.json")
    base["execution_lock"] = "original-lock.json"
    execution = copy.deepcopy(base)
    execution.update(gpus=[6, 7], execution_lock="parallel-lock.json")
    execution.pop("wait_for_results_root")
    execution.pop("wait_for_runs")
    return base, execution, {"gpus": [2, 3, 4, 5]}


def test_resource_change_preserves_full_matrix_and_order():
    base, execution, architecture = configs()
    module.validate_override(base, execution, architecture)
    assert execution["runs"] == base["runs"]
    assert sum(s["steps"] for s in execution["runs"]) == 1_800_000


@pytest.mark.parametrize("change", ["budget", "seed", "missing_arm", "image", "data", "gpu"])
def test_reject_scientific_or_unapproved_resource_changes(change):
    base, execution, architecture = configs()
    if change == "budget":
        execution["runs"][0]["steps"] += 1
    elif change == "seed":
        execution["runs"][0]["sampling_seed"] += 1
    elif change == "missing_arm":
        execution["runs"].pop()
    elif change == "image":
        execution["runtime"]["image"] = "different-image"
    elif change == "data":
        execution["data_file"] = "other.json"
    elif change == "gpu":
        execution["gpus"] = [5, 6]
    with pytest.raises(AssertionError):
        module.validate_override(base, execution, architecture)


def test_reject_gpu_overlap_and_real_predecessor_dependency():
    base, execution, architecture = configs()
    with pytest.raises(AssertionError):
        module.validate_override(base, execution, {"gpus": [2, 3, 4, 5, 6]})
    base["reused_baselines"][0]["path"] = base["wait_for_results_root"] + "/runs/live"
    execution["reused_baselines"] = copy.deepcopy(base["reused_baselines"])
    with pytest.raises(AssertionError):
        module.validate_override(base, execution, architecture)
