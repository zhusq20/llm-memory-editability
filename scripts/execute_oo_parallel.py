"""Run the frozen optimizer attribution on separately authorized GPU slots."""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import importlib.util
import json
import sys
import traceback
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_override(base, execution, architecture):
    """Permit only the authorized resource change, retaining the entire experiment."""
    assert base["batch"] == "oo-optimizer-attribution-v1"
    assert base["phase"] == "runs" and base["gpus"] == [2, 3, 4, 5]
    assert execution["gpus"] == [6, 7]
    assert execution["execution_lock"] != base["execution_lock"]
    expected = copy.deepcopy(base)
    expected.update(gpus=[6, 7], execution_lock=execution["execution_lock"])
    expected.pop("wait_for_results_root")
    expected.pop("wait_for_runs")
    assert execution == expected, "Scheduling amendment changed the scientific configuration"
    assert execution["runtime"]["docker_context"] == "lm-memory"
    assert not set(execution["gpus"]) & set(architecture["gpus"])
    predecessor = Path(base["wait_for_results_root"]).resolve()
    for baseline in execution["reused_baselines"]:
        assert not Path(baseline["path"]).resolve().is_relative_to(predecessor)


def load_frozen_controller(base):
    scripts = Path(base["source_root"]) / "scripts"
    sys.path.insert(0, str(scripts))
    spec = importlib.util.spec_from_file_location(
        "frozen_oo_controller", scripts / "execute_oo_optimizer_attribution.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert (
        Path(module.scheduler.__file__).resolve()
        == (scripts / "execute_parametric_architecture.py").resolve()
    )
    return module


def run(schedule_path):
    schedule = read(schedule_path)
    assert schedule["predecessor_barrier"] == "waived_for_disjoint_gpus_by_user"
    for path, expected in schedule["files_sha256"].items():
        assert digest(path) == expected, f"Scheduling artifact changed: {path}"
    assert str(Path(__file__).resolve()) in schedule["files_sha256"]
    base_path = Path(schedule["base_config"])
    base = read(base_path)
    assert digest(base_path) == read(base["execution_lock"])["config_sha256"]
    execution_path = Path(schedule["execution_config"])
    execution = read(execution_path)
    architecture = read(Path(base["wait_for_results_root"]) / "frozen-config.json")
    validate_override(base, execution, architecture)
    controller = load_frozen_controller(base)
    root = Path(execution["results_root"])
    with (root / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            controller.write(
                root / "schedule-amendment.json",
                {
                    "schedule": str(Path(schedule_path).resolve()),
                    "schedule_sha256": digest(schedule_path),
                    "base_config_sha256": digest(base_path),
                    "execution_config_sha256": digest(execution_path),
                    "gpus": execution["gpus"],
                    "predecessor_barrier": schedule["predecessor_barrier"],
                    "utc": controller.now(),
                },
            )
            # The original validator still checks all six runs and both old baselines.
            evidence = controller.verify_inputs(base, Path(base["source_root"]))
            controller.write(root / "parallel-input-verification.json", evidence)
            controller.scheduler.control_locked(
                execution, execution_path, root, command_builder=controller.container_command
            )
            if read(root / "controller-state.json")["state"] == "complete":
                controller.report(execution)
        except BaseException:
            state_path = root / "controller-state.json"
            state = read(state_path) if state_path.exists() else {}
            state.update(
                state="controller_error", error=traceback.format_exc(), updated_utc=controller.now()
            )
            controller.write(state_path, state)
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", required=True, type=Path)
    run(parser.parse_args().schedule.resolve())
