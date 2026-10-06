"""User-authorized early main runs; retain the frozen trainer and final selector."""

import fcntl
import importlib.util
import math
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEV = REPO / "results/parametric-architecture-development-v1"
ART = REPO / "docs/development-artifacts/parametric-architecture-rolling-20261004"


def load_controller(config):
    spec = importlib.util.spec_from_file_location(
        "frozen_controller",
        Path(config["source_root"]) / "scripts/execute_parametric_architecture.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def early_config(c, config):
    candidates, choices, runs = [], {}, []
    for architecture in ("M8", "W8", "IHC8"):
        specs = [s for s in config["runs"] if s["architecture"] == architecture]
        assert len(specs) == 2 and {s["learning_rate"] for s in specs} == {5e-5, 1e-4}
        rows = []
        for spec in specs:
            out = DEV / "runs" / spec["name"]
            assert c.read(out / "audit.json")["passed"] is True
            record = c.read(out / "learning.json")[-1]
            assert record["step"] == spec["steps"] == 32000
            metrics = record["metrics"]
            score = (metrics["atomic"]["nll"] + metrics["train_composition"]["nll"]) / 2
            assert math.isfinite(score)
            rows.append(
                dict(
                    architecture=architecture,
                    learning_rate=spec["learning_rate"],
                    score=score,
                    run=spec["name"],
                )
            )
        selected = min(rows, key=lambda r: (r["score"], r["learning_rate"]))
        candidates.extend(rows)
        choices[architecture] = selected
        spec = dict(next(s for s in specs if s["name"] == selected["run"]))
        spec.update(
            name=architecture.lower() + "-init1",
            steps=300000,
            initialization=811201,
            sampling_seed=812201,
            dropout_seed=813201,
        )
        runs.append(spec)
    t = config["transition"]
    main = dict(config)
    main.pop("transition")
    main.update(
        batch=t["batch"],
        phase="main",
        data_file=t["data_file"],
        data_sha256=t["data_sha256"],
        results_root=t["results_root"],
        evaluation_nodes=[0, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000, 200000, 300000],
        development_selection=str(ART / "early-selection.json"),
        tracking={**config["tracking"], "group": t["batch"]},
        runs=runs,
        execution_lock=str(ART / "early-lock.json"),
    )
    main["reporting"] = {
        **t.get("reporting", {}),
        "historical_reuse_verified": t["historical_reuse_verified"],
    }
    return main, dict(rule=t["selection_rule"], candidates=candidates, choices=choices)


def inspect(c, name):
    return c.json.loads(
        subprocess.check_output(["docker", "--context", "lm-memory", "inspect", name], text=True)
    )[0]


def run():
    config = __import__("json").loads((DEV / "frozen-config.json").read_text())
    c = load_controller(config)
    ART.mkdir(parents=True, exist_ok=True)
    assert c.digest(DEV / "frozen-config.json") == c.read(config["execution_lock"])["config_sha256"]
    for path, digest in config["source_files"].items():
        assert c.digest(path) == digest
    early, selection = early_config(c, config)
    assert c.digest(early["data_file"]) == early["data_sha256"]
    root = Path(early["results_root"])
    assert not (root / "controller-process.json").exists(), "Main controller already exists"
    state = c.read(DEV / "controller-state.json")
    assert not state["failed"] and not state.get("tracking_failure") and not state["queued"]
    assert list(state["active"]) == ["5"]
    dev_item = state["active"]["5"]
    assert inspect(c, dev_item["container"])["State"]["Running"]
    devices = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"], text=True
    )
    for line in devices.splitlines():
        gpu, memory = map(int, line.split(","))
        if gpu in (2, 3, 4):
            assert memory < 1024, (gpu, memory)
    old = c.read(DEV / "controller-process.json")
    assert "execute_parametric_architecture.py" in Path(f"/proc/{old['pid']}/cmdline").read_text()
    early_path = ART / "early-config.json"
    assert not early_path.exists(), "Existing early freeze: inspect recovery before relaunch"
    c.write(early_path, early)
    c.write(ART / "early-selection.json", selection)
    c.write(
        early["execution_lock"],
        dict(
            config_sha256=c.digest(early_path),
            frozen_utc=c.now(),
            environment={"torch": "2.6.0+cu126", "transformers": "4.40.0"},
        ),
    )
    c.write(
        ART / "handoff.json",
        dict(
            old_controller=old,
            development_state=state,
            early_config_sha256=c.digest(early_path),
            scheduler_sha256=c.digest(__file__),
            authorization="User requested starting main experiments on idle GPUs immediately",
            utc=c.now(),
        ),
    )
    # Signal only the scheduler PID. The active Docker client, container and tracker survive.
    os.kill(old["pid"], signal.SIGTERM)
    dev_lock = (DEV / "controller.lock").open("a")
    fcntl.flock(dev_lock, fcntl.LOCK_EX)
    root.mkdir(parents=True, exist_ok=True)
    main_lock = (root / "controller.lock").open("a")
    fcntl.flock(main_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    c.write(root / "controller-process.json", dict(pid=os.getpid(), command=sys.argv))
    c.write(DEV / "rolling-controller.json", dict(pid=os.getpid(), command=sys.argv))
    c.write(root / "frozen-config.json", early)
    ms = dict(
        state="running",
        phase="main",
        started_utc=c.now(),
        gpus=[2, 3, 4, 5],
        completed=[],
        failed=[],
        active={},
        queued=[],
        pending_development=["HC8"],
    )
    c.write(root / "controller-state.json", ms)
    tracking = [
        str(REPO / ".venv-wandb/bin/python"),
        "-u",
        "-m",
        "llm_memory_editability.interface_tracking",
        "--root",
        str(root),
        "--defaults",
        str(Path(config["source_root"]) / "configs/experiment-tracking-defaults.json"),
    ]
    log = (root / "tracking.log").open("a")
    tracker = subprocess.Popen(
        tracking,
        env=c.host_env(config["source_root"]),
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    c.write(root / "tracking-process.json", dict(pid=tracker.pid, command=tracking))
    deadline = time.monotonic() + 45
    while not (root / "tracking-ready.json").exists():
        assert tracker.poll() is None and time.monotonic() < deadline
        time.sleep(0.2)
    active, queue = {}, [(s, early, early_path) for s in early["runs"]]
    final_path = None
    try:
        while active or queue or final_path is None:
            assert tracker.poll() is None, "Main W&B tracker exited"
            if final_path is None:
                os.kill(c.read(DEV / "tracking-process.json")["pid"], 0)
                info = inspect(c, dev_item["container"])
                if not info["State"]["Running"]:
                    out = DEV / "runs" / dev_item["name"]
                    c.write(out / "container-inspect.json", [info])
                    assert info["State"]["ExitCode"] == 0 and c.read(out / "audit.json")["passed"]
                    state["completed"].append(dev_item["name"])
                    state.update(
                        state="complete", active={}, finished_utc=c.now(), updated_utc=c.now()
                    )
                    c.write(DEV / "controller-state.json", state)
                    final_path = c.select_main(config, DEV)
                    final = c.read(final_path)
                    for spec in early["runs"]:
                        assert spec == next(s for s in final["runs"] if s["name"] == spec["name"])
                    for key in ("data_sha256", "source_files", "evaluation_nodes", "runtime"):
                        assert early[key] == final[key]
                    c.write(root / "frozen-config.json", final)
                    c.write(
                        ART / "final-compatibility.json",
                        dict(
                            passed=True,
                            utc=c.now(),
                            final_config_sha256=c.digest(final_path),
                            early_runs=[s["name"] for s in early["runs"]],
                        ),
                    )
                    early_names = {s["name"] for s in early["runs"]}
                    queue.extend(
                        (s, final, final_path)
                        for s in final["runs"]
                        if s["name"] not in early_names
                    )
                    ms["pending_development"] = []
            for gpu, item in list(active.items()):
                if item["process"].poll() is None:
                    continue
                item["log"].close()
                out = root / "runs" / item["name"]
                c.write(out / "container-inspect.json", [inspect(c, item["container"])])
                passed = item["process"].returncode == 0 and (out / "audit.json").exists()
                passed = passed and c.read(out / "audit.json").get("passed") is True
                ms["completed" if passed else "failed"].append(item["name"])
                del active[gpu]
            if ms["failed"]:
                queue.clear()
            for gpu in [2, 3, 4] + ([5] if final_path is not None else []):
                if gpu in active or not queue:
                    continue
                spec, cfg, path = queue.pop(0)
                out = root / "runs" / spec["name"]
                assert not out.exists(), f"Duplicate run {out}"
                out.mkdir(parents=True)
                name, command = c.container_command(cfg, path, spec, out, gpu, 1)
                c.write(out / "container-command-1.json", dict(name=name, command=command))
                worker_log = (out / "worker.log").open("a")
                proc = subprocess.Popen(command, stdout=worker_log, stderr=subprocess.STDOUT)
                active[gpu] = dict(process=proc, name=spec["name"], container=name, log=worker_log)
                for _ in range(60):
                    try:
                        info = inspect(c, name)
                    except subprocess.CalledProcessError:
                        time.sleep(0.2)
                        continue
                    if info["State"].get("Running"):
                        c.write(
                            out / "container-runtime.json",
                            dict(
                                physical_gpu=gpu,
                                host_pid=info["State"]["Pid"],
                                container_id=info["Id"],
                                image=info["Image"],
                                cpus=cfg["runtime"]["cpus"],
                                memory=cfg["runtime"]["memory"],
                            ),
                        )
                        break
                    time.sleep(0.2)
            ms.update(
                active={
                    str(g): {k: x[k] for k in ("name", "container")} for g, x in active.items()
                },
                queued=[s["name"] for s, _, _ in queue],
                updated_utc=c.now(),
            )
            c.write(root / "controller-state.json", ms)
            time.sleep(3)
        if not ms["failed"]:
            subprocess.run(
                [
                    sys.executable,
                    str(Path(config["source_root"]) / "scripts/report_parametric_architecture.py"),
                    "--config",
                    str(final_path),
                    "--out",
                    str(root / "report"),
                ],
                env=c.host_env(config["source_root"]),
                check=True,
            )
        ms.update(
            state="finished_with_failures" if ms["failed"] else "complete", finished_utc=c.now()
        )
        c.write(root / "controller-state.json", ms)
    except BaseException:
        ms.update(state="controller_error", error=traceback.format_exc(), updated_utc=c.now())
        c.write(root / "controller-state.json", ms)
        raise


if __name__ == "__main__":
    run()
