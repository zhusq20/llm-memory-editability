import copy
import fcntl
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


policy = module(ROOT / "scripts/ttcd_gpu8_resources.py", "gpu8_policy")
sharing = module(ROOT / "scripts/share_ttcd_gpu8.py", "gpu8_sharing")


@pytest.fixture
def resource():
    class ResourceUnavailable(Exception):
        pass

    row = {
        "uuid": "GPU8",
        "total_mib": 97887,
        "memory": 22253,
        "free_mib": 74998,
        "utilization": 99,
    }
    container = {
        "Name": f"/{policy.BATCH}-transformer-stream-train",
        "Id": "original",
        "Mounts": [{"Destination": "/frozen", "Source": str(policy.SOURCE), "RW": False}],
        "Config": {"User": "1015:1016", "Cmd": ["python", policy.SCRIPT, "train"]},
        "HostConfig": {"DeviceRequests": [{"DeviceIDs": ["GPU8"]}]},
    }
    containers, processes = [container], [{"uuid": "GPU8", "pid": 100}]
    cpu_calls = []
    helpers = SimpleNamespace(
        ResourceUnavailable=ResourceUnavailable,
        gpu_snapshot=lambda: ({8: row}, processes),
        running_containers=lambda _: containers,
        container_uses_gpu=lambda c, index, uuid: True,
        docker=lambda *args: SimpleNamespace(stdout="PID\n100\n"),
        admit_cpu=lambda config, arm: (cpu_calls.append((config, arm)) or containers, 999999999),
        utc_now=lambda: "now",
    )
    return SimpleNamespace(
        helpers=helpers, row=row, containers=containers, processes=processes, cpu_calls=cpu_calls
    )


def test_active_transformer_allows_loop_with_original_cpu_budget(resource):
    runtime = {"cpu_sets": {"loop": list(range(44, 48))}, "memory_gib": 96}
    result = policy.admission(resource.helpers, runtime, "loop")
    assert result["gpu"] == 8 and result["gpu_sharing"] is True
    assert result["free_mib"] == 74998 and result["cpu_set"] == list(range(44, 48))
    assert result["wallclock_is_exclusive_gpu_benchmark"] is False
    assert resource.cpu_calls == [({"runtime": runtime}, "loop")]


def test_low_memory_and_outside_cuda_are_rejected(resource):
    resource.row["free_mib"] = 32767
    with pytest.raises(resource.helpers.ResourceUnavailable, match="32GiB"):
        policy.admission(resource.helpers, {"cpu_sets": {"loop": []}}, "loop")
    resource.row["free_mib"] = 74998
    resource.processes.append({"uuid": "GPU8", "pid": 200})
    with pytest.raises(resource.helpers.ResourceUnavailable, match="outside CUDA"):
        policy.cohort(resource.helpers)


@pytest.mark.parametrize(
    "change", ["unrelated", "writable", "owner", "command", "device", "source"]
)
def test_sharing_checks_frozen_container_identity(resource, change):
    c = resource.containers[0]
    if change == "unrelated":
        c["Name"] = "/another-project"
    elif change == "writable":
        c["Mounts"][0]["RW"] = True
    elif change == "owner":
        c["Config"]["User"] = "0"
    elif change == "command":
        c["Config"]["Cmd"] = ["python", "other.py"]
    elif change == "device":
        c["HostConfig"]["DeviceRequests"][0]["DeviceIDs"] = ["GPU9"]
    else:
        c["Mounts"][0]["Source"] = "/other-source"
    with pytest.raises(resource.helpers.ResourceUnavailable):
        policy.cohort(resource.helpers)


def test_duplicate_architecture_and_unaccounted_memory_rejected(resource):
    resource.containers.append(copy.deepcopy(resource.containers[0]))
    with pytest.raises(resource.helpers.ResourceUnavailable, match="duplicate"):
        policy.cohort(resource.helpers)
    resource.containers.clear()
    resource.processes.clear()
    with pytest.raises(resource.helpers.ResourceUnavailable, match="unaccounted"):
        policy.cohort(resource.helpers)


def test_cohort_leases_share_one_lock_until_both_stages_finish(tmp_path):
    path = tmp_path / "gpu-8.lock"
    pool = policy.CohortSlots(path)
    first, second = pool.take("loop"), pool.take("transformer")
    with pytest.raises(ValueError, match="at most one"):
        pool.take("loop")
    with path.open("a") as external:
        with pytest.raises(BlockingIOError):
            fcntl.flock(external, fcntl.LOCK_EX | fcntl.LOCK_NB)
        first.close()
        with pytest.raises(BlockingIOError):
            fcntl.flock(external, fcntl.LOCK_EX | fcntl.LOCK_NB)
        second.close()
        fcntl.flock(external, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert pool.handle is None


def test_other_controller_gpu_lock_is_respected(tmp_path):
    path = tmp_path / "gpu-8.lock"
    with path.open("a") as external:
        fcntl.flock(external, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            policy.CohortSlots(path).take("loop")


def test_command_only_adds_sharing_metadata(tmp_path, monkeypatch):
    command = [
        "docker",
        "--context",
        "lm-memory",
        "run",
        "--gpus",
        "device=GPU8",
        "--env",
        "WANDB_TAGS=old",
        "image",
        "python",
        policy.SCRIPT,
        "train",
    ]
    ctx = SimpleNamespace(
        helpers=SimpleNamespace(),
        controller=SimpleNamespace(build_command=lambda *args: ("same", list(command))),
    )
    engine = SimpleNamespace()
    monkeypatch.setattr(sharing, "sha", lambda _: "schedule-sha")
    sharing.install(engine, ctx, policy)
    name, actual = ctx.controller.build_command(
        {"runtime": {"image": "image"}}, "loop", {}, tmp_path, None, {"gpu": 8}
    )
    assert name == "same" and actual[actual.index("image") :] == command[command.index("image") :]
    assert actual[actual.index("--gpus") + 1] == "device=GPU8"
    assert "LM_GPU_SHARING=1" in actual
    assert "WANDB_TAGS=ttcd-architecture,loop,gpu-8,shared,user-authorized" in actual
    assert not any("target=/run/experiment" in item for item in actual)
