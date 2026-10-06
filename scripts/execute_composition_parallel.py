"""Extend only the frozen composition controller's resource scheduling."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import time
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def eligible_gpus(schedule):
    result = list(schedule["immediate_gpus"])
    for dependency in schedule["deferred_gpus"]:
        root = Path(dependency["results_root"])
        state = read(root / "controller-state.json")
        if (
            state.get("state") != "complete"
            or state.get("failed")
            or state.get("active")
            or state.get("queued")
            or set(state.get("completed", [])) != set(dependency["required_runs"])
        ):
            continue
        if not all(
            (root / "runs" / name / "audit.json").exists()
            and read(root / "runs" / name / "audit.json").get("passed") is True
            for name in dependency["required_runs"]
        ):
            continue
        result.extend(dependency["gpus"])
    return result


def pending_specs(specs, completed, active_names, label):
    names = [spec["name"] for spec in specs]
    if len(names) != len(set(names)) or not set(active_names) <= set(names):
        raise ValueError("Duplicate run or recovered run outside the registered stage")
    if set(active_names) & set(completed):
        raise ValueError("A completed run still has an active container")
    queue = [s for s in specs if s["name"] not in set(completed) | set(active_names)]
    if label == "load":
        queue.sort(key=lambda s: -s["entities"])
    return queue


class ContainerProcess:
    def __init__(self, name):
        self.name = name
        self.pid = None
        self.returncode = None

    def poll(self):
        info = read_container(self.name)
        self.returncode = None if info["State"]["Running"] else info["State"]["ExitCode"]
        return self.returncode


def read_container(name):
    return json.loads(
        subprocess.check_output(["docker", "--context", "lm-memory", "inspect", name], text=True)
    )[0]


class ExistingTracker:
    """Keep the existing W&B writer; completion evidence supplies its exit status."""

    def __init__(self, pid, root):
        self.pid, self.root, self.returncode = pid, root, None

    def wait(self, timeout):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if (self.root / "tracking-completion.json").exists():
                self.returncode = 0
                return 0
            stat = Path(f"/proc/{self.pid}/stat")
            if not stat.exists() or stat.read_text().rsplit(")", 1)[1].split()[0] == "Z":
                self.returncode = 1
                return 1
            time.sleep(3)
        raise subprocess.TimeoutExpired("existing W&B tracker", timeout)


class ParallelScheduler:
    def __init__(self, module, schedule):
        self.m, self.schedule = module, schedule
        self.original_tracking = module.start_tracking

    def tracking(self, config, root):
        path = root / "tracking-process.json"
        if path.exists():
            previous = read(path)
            cmdline = Path(f"/proc/{previous['pid']}/cmdline")
            if cmdline.exists():
                actual = cmdline.read_bytes().replace(b"\0", b" ").decode()
                if "llm_memory_editability.curve_tracking" in actual and str(root) in actual:
                    return ExistingTracker(previous["pid"], root), (root / "tracking.log").open("a")
        return self.original_tracking(config, root)

    def recover(self, config, state):
        """Adopt live workers without waiting for them or interrupting training."""
        names = subprocess.check_output(
            [
                "docker",
                "--context",
                "lm-memory",
                "ps",
                "--filter",
                "label=batch=" + config["batch"],
                "--format",
                "{{.Names}}",
            ],
            text=True,
        ).split()
        candidates = {item["container"] for item in state.get("active", {}).values()} | set(names)
        active = {}
        allowed = set(self.schedule["immediate_gpus"])
        for dependency in self.schedule["deferred_gpus"]:
            allowed.update(dependency["gpus"])
        for name in sorted(candidates):
            info = read_container(name)
            labels = info["Config"]["Labels"]
            ids = info["HostConfig"]["DeviceRequests"][0]["DeviceIDs"]
            if (
                labels["batch"] != config["batch"]
                or info["Image"] != config["runtime"]["image"]
                or len(ids) != 1
                or int(ids[0]) not in allowed
                or ids[0] in active
            ):
                raise RuntimeError("Unexpected recovered container identity or GPU overlap")
            active[ids[0]] = {"name": labels["run"], "container": name}
        state["active"] = active

    def stage(self, config, specs, label, state):
        m = self.m
        root = Path(config["results_root"])
        path = Path(config["source_root"]).parent / f"{label}-config.json"
        materialized = {**config, "runs": specs, "phase": label}
        if path.exists():
            if read(path) != materialized:
                raise ValueError("Attempted to change a materialized scientific stage")
        else:
            m.write_json(path, materialized)
            m.write_json(
                path.with_name(label + "-execution-lock.json"),
                {"config_sha256": digest(path), "utc": m.utc()},
            )
        by_name = {s["name"]: s for s in specs}
        active_names = [item["name"] for item in state.get("active", {}).values()]
        if set(active_names) - set(by_name) and set(by_name) <= set(state["completed"]):
            # On restart during load, leave its workers intact while the original
            # controller revisits the completed support gate.
            return
        queue = pending_specs(specs, state["completed"], active_names, label)
        active = {
            int(gpu): {
                "spec": by_name[item["name"]],
                "container": item["container"],
                "process": ContainerProcess(item["container"]),
                "log": None,
            }
            for gpu, item in state.get("active", {}).items()
        }
        state.update(stage=label, state="running", stage_runs=[s["name"] for s in specs])
        while queue or active:
            for gpu, item in list(active.items()):
                if item["process"].poll() is None:
                    continue
                if item["log"] is not None:
                    item["log"].close()
                out = root / "runs" / item["spec"]["name"]
                m.write_json(out / "container-inspect.json", [read_container(item["container"])])
                passed = item["process"].returncode == 0 and (out / "audit.json").exists()
                passed = passed and read(out / "audit.json").get("passed") is True
                target = state["completed" if passed else "failed"]
                if item["spec"]["name"] not in target:
                    target.append(item["spec"]["name"])
                del active[gpu]
            if state["failed"] and not active:
                raise RuntimeError("Registered scientific job failed; queue preserved")
            spent = sum(
                read(f)["wall_seconds"] for f in (root / "runs").glob("*/worker-completion.json")
            )
            if spent > config["budget"]["maximum_gpu_hours"] * 3600:
                raise RuntimeError("Frozen cumulative GPU time budget reached")
            eligible = eligible_gpus(self.schedule)
            for gpu in eligible:
                if gpu in active or not queue or state["failed"] or not m.gpu_free(gpu):
                    continue
                spec = queue.pop(0)
                out = root / "runs" / spec["name"]
                if (out / "failure.json").exists():
                    raise RuntimeError("Recorded failure must not be silently overwritten")
                if (out / "audit.json").exists() and read(out / "audit.json").get("passed"):
                    state["completed"].append(spec["name"])
                    continue
                out.mkdir(parents=True, exist_ok=True)
                attempt = len(list(out.glob("container-command-*.json"))) + 1
                name, command = m.container_command(config, path, spec, out, gpu, attempt)
                m.write_json(
                    out / f"container-command-{attempt}.json", {"name": name, "command": command}
                )
                log = (out / f"worker-{attempt}.log").open("a")
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                active[gpu] = {"process": process, "log": log, "container": name, "spec": spec}
                for _ in range(60):
                    inspect = subprocess.run(
                        ["docker", "--context", "lm-memory", "inspect", name],
                        capture_output=True,
                        text=True,
                    )
                    if inspect.returncode == 0:
                        info = json.loads(inspect.stdout)[0]
                        if info["State"].get("Running"):
                            m.write_json(
                                out / "container-runtime.json",
                                {
                                    "physical_gpu": gpu,
                                    "host_pid": info["State"]["Pid"],
                                    "container_id": info["Id"],
                                    "image": info["Image"],
                                    "cpus": config["runtime"]["cpus"],
                                    "memory": config["runtime"]["memory"],
                                },
                            )
                            break
                    if process.poll() is not None:
                        break
                    time.sleep(0.2)
            state.update(
                active={
                    str(g): {
                        "name": i["spec"]["name"],
                        "container": i["container"],
                        "pid": i["process"].pid,
                    }
                    for g, i in active.items()
                },
                queued=[s["name"] for s in queue],
                updated_utc=m.utc(),
                consumed_gpu_hours=spent / 3600,
                gpus=eligible,
                schedule_amendment=self.schedule["schedule_path"],
            )
            m.write_json(root / "controller-state.json", state)
            time.sleep(3)


def run(path):
    assert digest(path) == read(path.with_name("schedule-lock.json"))["sha256"]
    schedule = read(path)
    for file, expected in schedule["files_sha256"].items():
        if digest(file) != expected:
            raise RuntimeError(f"Scheduling artifact changed: {file}")
    assert str(Path(__file__).resolve()) in schedule["files_sha256"]
    for file in schedule["cuda_preflights"]:
        assert read(file)["passed"] is True
    schedule["schedule_path"] = str(path)
    config_path = Path(schedule["base_config"])
    config = read(config_path)
    assert config["gpus"] == [0, 1] and config["batch"] == "composition-data-curves-v1"
    script = Path(config["source_root"]) / "scripts/execute_composition_curves.py"
    spec = importlib.util.spec_from_file_location("frozen_composition_controller", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    scheduler = ParallelScheduler(module, schedule)
    module.stage, module.recover_active, module.start_tracking = (
        scheduler.stage,
        scheduler.recover,
        scheduler.tracking,
    )
    module.controller(config_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", required=True, type=Path)
    run(parser.parse_args().schedule.resolve())
