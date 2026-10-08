"""Adopt existing loop containers and use newly released GPUs without retraining."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from llm_memory_editability.grok_depth import utc, write_json

ROOT = Path("/ossfs/workspace/llm-memory-editability")
RESULTS = ROOT / "results/loop-block-depth-v1"
ARTIFACT = ROOT / "docs/development-artifacts/loop-block-depth-parallel-20261006"
FROZEN = ROOT / "docs/development-artifacts/loop-block-depth-v1/frozen-config.json"


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def driver():
    config = read(FROZEN)
    path = Path(config["source_root"]) / "scripts/run_loop_block_depth.py"
    spec = importlib.util.spec_from_file_location("frozen_loop_driver", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, module.checked_config(FROZEN)


def inspect(name):
    return json.loads(
        subprocess.check_output(["docker", "--context", "lm-memory", "inspect", name], text=True)
    )[0]


def ownership(info, config, gpu, name):
    labels = info["Config"]["Labels"]
    requests = info["HostConfig"]["DeviceRequests"]
    assert labels["batch"] == config["batch"] and labels["run"] == name
    assert info["Image"] == config["runtime"]["image"]
    assert any(r.get("DeviceIDs") == [str(gpu)] for r in requests)
    assert str(FROZEN) in info["Config"]["Cmd"]


def preflight():
    base, config = driver()
    prior = read(ROOT / "results/composition-data-curves-v1/controller-state.json")
    assert prior["state"] == "complete" and not prior["active"] and not prior["queued"]
    assert not prior["failed"]
    assert all(base.gpu_free(g) for g in [0, 1, 2])

    def one(gpu):
        p, log, out, _name = base.container(config, gpu, config["specs"][0], engineering=True)
        code = p.wait()
        log.close()
        return dict(
            gpu=gpu,
            passed=code == 0 and read(out / "audit.json")["passed"],
            evidence=str(out),
            returncode=code,
        )

    with ThreadPoolExecutor(max_workers=3) as pool:
        checks = list(pool.map(one, [0, 1, 2]))
    result = dict(passed=all(r["passed"] for r in checks), checks=checks, utc=utc())
    write_json(ARTIFACT / "preflight.json", result)
    print(json.dumps(result))


def controller():
    base, config = driver()
    schedule = read(ARTIFACT / "schedule.json")
    assert sha(FROZEN) == schedule["science_config_sha256"]
    assert sha(__file__) == schedule["controller_sha256"]
    assert read(ARTIFACT / "preflight.json")["passed"]
    handle = (RESULTS / "controller.lock").open("a")
    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = read(RESULTS / "controller-state.json")
    active = {int(gpu): item for gpu, item in state["active"].items()}
    adopted = []
    for gpu, item in active.items():
        info = inspect(item["container"])
        ownership(info, config, gpu, item["name"])
        adopted.append(
            dict(
                gpu=gpu,
                name=item["name"],
                container=info["Id"],
                host_pid=info["State"]["Pid"],
                state=info["State"],
            )
        )
    write_json(ARTIFACT / "adopted-containers.json", dict(utc=utc(), containers=adopted))
    known = set(state["completed"] + state["failed"]) | {v["name"] for v in active.values()}
    pending = [s for s in config["specs"] if s["name"] not in known]
    processes = {}
    state.update(gpus=schedule["gpus"], schedule_amendment=str(ARTIFACT / "schedule.json"))
    while pending or active:
        for gpu, item in list(active.items()):
            info = inspect(item["container"])
            ownership(info, config, gpu, item["name"])
            if info["State"]["Running"]:
                continue
            out = RESULTS / "runs" / item["name"]
            ok = (
                info["State"]["ExitCode"] == 0
                and (out / "audit.json").exists()
                and read(out / "audit.json")["passed"]
                and (out / "worker-completion.json").exists()
            )
            state["completed" if ok else "failed"].append(item["name"])
            write_json(
                out / "parallel-container-completion.json",
                dict(passed=ok, state=info["State"], utc=utc()),
            )
            if gpu in processes:
                process, log = processes.pop(gpu)
                process.wait(timeout=20)
                log.close()
            del active[gpu]
        for gpu in schedule["gpus"]:
            if pending and gpu not in active and base.gpu_free(gpu):
                spec = pending.pop(0)
                p, log, _out, name = base.container(config, gpu, spec)
                active[gpu] = dict(name=spec["name"], container=name)
                processes[gpu] = (p, log)
        state.update(
            state="running",
            active={str(k): v for k, v in active.items()},
            queued=[s["name"] for s in pending],
            updated_utc=utc(),
        )
        write_json(RESULTS / "controller-state.json", state)
        time.sleep(5)
    command = [
        sys.executable,
        config["source_root"] + "/scripts/report_loop_block_depth.py",
        "--config",
        str(FROZEN),
    ]
    code = subprocess.run(command, env=base.environment(config), check=False).returncode
    state.update(
        state="finished_with_failures" if state["failed"] or code else "complete",
        report_returncode=code,
        finished_utc=utc(),
    )
    write_json(RESULTS / "controller-state.json", state)


def launch():
    base, config = driver()
    schedule = read(ARTIFACT / "schedule.json")
    assert sha(FROZEN) == schedule["science_config_sha256"]
    assert sha(__file__) == schedule["controller_sha256"]
    assert read(ARTIFACT / "preflight.json")["passed"]
    previous = read(RESULTS / "controller-process.json")
    pid = previous["pid"]
    command_path = Path(f"/proc/{pid}/cmdline")
    expected = str(Path(config["source_root"]) / "scripts/run_loop_block_depth.py")
    assert command_path.exists() and expected.encode() in command_path.read_bytes()
    write_json(ARTIFACT / "previous-controller-process.json", previous)
    write_json(ARTIFACT / "state-before-takeover.json", read(RESULTS / "controller-state.json"))
    # Only stop the scheduler PID. Docker workers and the W&B sidecar continue.
    os.kill(pid, signal.SIGTERM)
    for _ in range(100):
        if not command_path.exists() or not command_path.read_bytes():
            break
        time.sleep(0.1)
    else:
        raise RuntimeError("Old scheduler did not exit; do not start a competing controller")
    script = ARTIFACT / "execute_loop_block_parallel.py"
    command = [sys.executable, "-u", str(script), "controller"]
    with (RESULTS / "parallel-controller.log").open("a") as log:
        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=base.environment(config),
        )
    write_json(
        RESULTS / "controller-process.json", dict(pid=process.pid, command=command, utc=utc())
    )
    write_json(RESULTS / "schedule-amendment.json", schedule)
    print(json.dumps({"pid": process.pid, "gpus": schedule["gpus"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["preflight", "launch", "controller"])
    args = parser.parse_args()
    {"preflight": preflight, "launch": launch, "controller": controller}[args.mode]()
