import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from migrate_gate_extension_gpu4 import (  # noqa: E402
    PLACEHOLDERS,
    filter_snapshot,
    inactive,
    select_jobs,
)


def test_only_frozen_depth1_matrix_is_selected():
    path = Path(__file__).resolve().parents[1] / "configs/gate-generalization-extension-v1.json"
    plan = json.loads(path.read_text())
    identifiers = [f"depth-1--{arm}" for arm in ("paper", "fixed-update", "token-update")]
    jobs = select_jobs(plan, identifiers)
    assert sum(job["steps"] for job in jobs) == 900000
    with pytest.raises(ValueError):
        select_jobs(plan, ["confirmation-3--paper"])
    changed = copy.deepcopy(plan)
    next(job for job in changed["jobs"] if job["id"] == identifiers[0])["steps"] = 1500000
    with pytest.raises(ValueError):
        select_jobs(changed, identifiers)


def test_long_budget_preserves_science_and_original_evaluation_nodes():
    root = Path(__file__).resolve().parents[1] / "configs"
    old = json.loads((root / "gate-extension/depth-1.json").read_text())
    new = json.loads((root / "gate-depth-long-gpu4-v1.json").read_text())
    assert new["training"]["steps"] == 1500000
    assert new["evaluation_nodes"] == old["evaluation_nodes"] + list(range(400000, 1500001, 100000))
    assert new["inference_repeats"] == [1, 2, 3, 4, 6, 8]
    new["training"]["steps"] = old["training"]["steps"]
    for key in ("batch", "authorization", "primary", "evaluation_nodes"):
        new[key] = old[key]
    assert new == old


def container(identifier, name):
    return {
        "Id": identifier,
        "Name": name,
        "State": {
            "Status": "created",
            "StartedAt": "0001-01-01T00:00:00Z",
            "Pid": 0,
            "Running": False,
        },
        "HostConfig": {"RestartPolicy": {"Name": "no"}},
    }


def test_changed_or_started_reservation_is_never_ignored():
    _, identifier, name = PLACEHOLDERS[0]
    original = container(identifier, name)
    assert inactive(original, identifier, name)
    for key, value in (
        ("Status", "running"),
        ("Pid", 123),
        ("Running", True),
        ("StartedAt", "2026-10-18T00:00:00Z"),
    ):
        changed = copy.deepcopy(original)
        changed["State"][key] = value
        assert not inactive(changed, identifier, name)
    assert not inactive(original, "different-id", name)
    changed = copy.deepcopy(original)
    changed["HostConfig"]["RestartPolicy"]["Name"] = "always"
    assert not inactive(changed, identifier, name)


def test_active_process_and_unrelated_reservations_remain_blocking():
    records = {identifier: container(identifier, name) for _, identifier, name in PLACEHOLDERS}
    # A historical placeholder that has started must also stay blocking.
    records[PLACEHOLDERS[0][1]]["State"]["Running"] = True
    common = SimpleNamespace(
        docker=lambda _, identifier, **kwargs: SimpleNamespace(
            stdout=json.dumps([records[identifier]])
        )
    )
    original = {
        4: [entry[2] for entry in PLACEHOLDERS] + ["active CUDA process 123", "/other"],
        5: ["/untouched"],
    }
    _, blocked = filter_snapshot(common, {4: {}}, original)
    assert blocked[4] == [PLACEHOLDERS[0][2], "active CUDA process 123", "/other"]
    assert blocked[5] == ["/untouched"]
    assert len(original[4]) == 6
