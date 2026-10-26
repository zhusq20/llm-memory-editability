"""The GPU1 migration cannot silently alter scientific or quota contracts."""

import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import launch_candidate_gpu1 as migration  # noqa: E402


@pytest.fixture
def configs():
    return tuple(
        json.loads((ROOT / f"configs/{name}.json").read_text())
        for name in (
            "candidate-update-gate-gpu1-v3",
            "candidate-update-gate-multihop-resource-recovery-v2",
        )
    )


def test_scientific_contract_unchanged(configs):
    migration.validate_migration(*configs)


@pytest.mark.parametrize("field", ["model", "training", "data", "arms", "evaluation_nodes"])
def test_scientific_changes_rejected(configs, field):
    config, previous = copy.deepcopy(configs)
    config[field] = {}
    with pytest.raises(ValueError, match="scientific"):
        migration.validate_migration(config, previous)


@pytest.mark.parametrize("key,value", [("gpu", 0), ("cpu_first", 24), ("image", "another")])
def test_wrong_device_or_runtime_rejected(configs, key, value):
    config, previous = copy.deepcopy(configs)
    config["runtime"][key] = value
    with pytest.raises(ValueError):
        migration.validate_migration(config, previous)


def test_no_fifth_gpu(configs):
    config, previous = copy.deepcopy(configs)
    config["scheduling_recovery"]["maximum_concurrent_owned_gpus"] = 5
    with pytest.raises(ValueError):
        migration.validate_migration(config, previous)
