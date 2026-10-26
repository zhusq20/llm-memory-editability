#!/usr/bin/env python3
"""Submit disjoint data preparation and a complete eight-arm matrix, once."""

from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
import queue
import shutil
import socket
import subprocess
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import launch_knowledge_method_diagnosis as L  # noqa: E402

DEFAULT_PLAN = ROOT / "configs/knowledge-method-factorial-launch-v1.json"
ARMS = (
    "pd_lora",
    "sft_lora",
    "pd_full",
    "sft_full",
    "pd_down",
    "sft_down",
    "pd_down_lora",
    "sft_down_lora",
)


def validate(plan):
    if plan["batch"] != "knowledge-method-factorial-1000-v1" or set(plan["arms"]) != set(ARMS):
        raise ValueError("The expanded batch requires exactly eight arms")
    if len({x["gpu_uuid"] for x in plan["lanes"]}) != len(plan["lanes"]):
        raise ValueError("One worker per unique GPU")
    scientific = json.loads((ROOT / plan["scientific_config"]).read_text())
    if set(scientific["arms"]) != set(ARMS) or scientific["n_cases"] != 1000:
        raise ValueError("Scientific matrix or data size differs")
    if (
        scientific.get("model") != "Qwen/Qwen3-1.7B"
        or scientific.get("enable_thinking") is not False
        or scientific.get("judge_model") != "Qwen/Qwen3-4B"
        or scientific.get("judge_enable_thinking") is not False
        or scientific.get("answer_format") != "qwen_final_answer_only"
    ):
        raise ValueError("The requested Qwen model and non-thinking contract differ")
    intervals = [(0, scientific["reuse_cases"])]
    intervals += [(v["start"], v["stop"]) for v in scientific["prepare_shards"].values()]
    coverage = [i for a, b in intervals for i in range(a, b)]
    if sorted(coverage) != list(range(1000)):
        raise ValueError("Disjoint preparation coverage is incomplete")
    return scientific


def command(plan, stage, name, lane):
    # Reuse the checked resource limits/secret bootstrap; replace batch-specific names and mounts.
    template = dict(plan, commands={stage: ["python"]})
    _, argv = L.container_command(template, stage, name, lane)
    label = f"kmf-v1-{name}-{stage}"
    argv[argv.index("--name") + 1] = label
    mount_positions = [i + 1 for i, x in enumerate(argv) if x == "--mount"]
    for pos in mount_positions:
        if "target=/batch/deps," in argv[pos]:
            argv[pos] = f"type=bind,source={plan['runtime']['deps']},target=/batch/deps,readonly"
        elif "target=/batch/upstream," in argv[pos]:
            argv[pos] = (
                f"type=bind,source={plan['runtime']['upstream']},target=/batch/upstream,readonly"
            )
    image_pos = argv.index(plan["runtime"]["image"])
    old_batch = str(Path(plan["runtime"]["reuse_shared"]).parent)
    argv[image_pos:image_pos] = [
        "--mount",
        f"type=bind,source={old_batch},target=/old-batch,readonly",
    ]
    common = ["--config", f"/project/{plan['scientific_config']}", "--batch", "/batch"]
    if stage == "regrade-old":
        runner = [
            "python",
            "-u",
            "scripts/run_knowledge_method_paper_eval.py",
            "--stage",
            "regrade-old",
            "--config",
            f"/project/{plan['scientific_config']}",
            "--batch",
            "/old-batch",
            "--output",
            "/batch/regrade-development",
        ]
    else:
        runner = [
            "python",
            "-u",
            "scripts/run_knowledge_method_factorial.py",
            "--stage",
            stage,
            *common,
        ]
        if stage == "prepare-shard":
            runner += ["--shard", name]
        elif stage in ["train", "audit", "evaluate"]:
            runner += ["--arm", name]
    argv[-1:] = runner
    return label, argv


class Controller(L.Controller):
    def __init__(self, plan, scientific):
        self.plan, self.scientific = plan, scientific
        self.output = Path(plan["output"])
        self.mutex = threading.Lock()
        names = ["regrade-development", "merge", *scientific["prepare_shards"], *ARMS]
        self.state = {
            "batch": plan["batch"],
            "state": "preparing",
            "jobs": {name: {"state": "pending"} for name in names},
        }

    def run_stage(self, arm, stage, lane):
        folder = self.output / "execution" / arm / stage
        if folder.exists():
            raise FileExistsError(f"Attempt already exists: {folder}")
        label, argv = command(self.plan, stage, arm, lane)
        if L.call(
            [*L.DOCKER, "ps", "-a", "--filter", f"name=^/{label}$", "-q"], timeout=30
        ).stdout.strip():
            raise FileExistsError(f"Container already exists: {label}")
        L.write(folder / "command.json", {"argv": argv, "utc": L.utc()})
        L.call(argv, timeout=120)
        subprocess.run(
            [*L.DOCKER, "cp", "--archive", "-", f"{label}:/"],
            input=L.credential_archive(self.plan["wandb_token_file"]),
            check=True,
            capture_output=True,
            timeout=120,
        )
        L.call([*L.DOCKER, "start", label], timeout=120)
        self.update(
            arm,
            state="running",
            stage=stage,
            gpu=lane["gpu"],
            gpu_uuid=lane["gpu_uuid"],
            container=label,
            started_utc=L.utc(),
        )
        with (folder / "stdout.log").open("w") as log:
            follower = subprocess.Popen(
                [*L.DOCKER, "logs", "--follow", label], stdout=log, stderr=subprocess.STDOUT
            )
            try:
                code = int(L.call([*L.DOCKER, "wait", label]).stdout.strip())
            finally:
                try:
                    follower.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    follower.terminate()
                    follower.wait(timeout=10)
        L.write(folder / "exit.json", {"code": code, "utc": L.utc()})
        if code:
            raise RuntimeError(f"{label} exited {code}; preserve attempt")

    def parallel(self, jobs):
        pending = queue.Queue()
        for job in jobs:
            pending.put(job)

        def lane_worker(lane):
            outcomes = []
            while True:
                try:
                    name, stages = pending.get_nowait()
                except queue.Empty:
                    return all(outcomes)
                try:
                    outcomes.append(self.worker(name, stages, lane))
                finally:
                    pending.task_done()

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.plan["lanes"])) as pool:
            return all(list(pool.map(lane_worker, self.plan["lanes"])))

    def run(self):
        jobs = [("regrade-development", ["regrade-old"])]
        jobs += [(name, ["prepare-shard"]) for name in self.scientific["prepare_shards"]]
        ok = self.parallel(jobs)
        if ok:
            ok = self.worker("merge", ["merge"], self.plan["lanes"][0])
        if ok and L.prepared(self.output):
            self.state["state"] = "training"
            L.write(self.output / "controller-state.json", self.state)
            ok = self.parallel([(name, ["train", "audit", "evaluate"]) for name in ARMS])
        else:
            ok = False
            for name in ARMS:
                self.update(name, state="blocked_by_prepare")
        self.state["state"] = "complete" if ok else "needs_attention"
        self.state["finished_utc"] = L.utc()
        L.write(self.output / "controller-state.json", self.state)
        L.write(self.output / "controller-exit.json", self.state)
        return 0 if ok else 1


def freeze(plan):
    destination = Path(plan["output"]) / "source"
    if destination.exists():
        raise FileExistsError("Existing source snapshot is preserved")
    sources = list((ROOT / "src/llm_memory_editability").rglob("*.py"))
    sources += [
        ROOT / "scripts" / x
        for x in [
            "launch_knowledge_method_factorial.py",
            "start_knowledge_method_factorial_when_ready.py",
            "run_knowledge_method_factorial.py",
            "run_knowledge_method_paper_eval.py",
            "launch_knowledge_method_diagnosis.py",
            "launch_paper_update_gate.py",
            "launch_dual_gate.py",
            "resume_loop_query_control_transport.py",
        ]
    ]
    sources += [ROOT / plan["scientific_config"], DEFAULT_PLAN, ROOT / "pyproject.toml"]
    sources += list((ROOT / "tests").glob("test_knowledge_method*.py"))
    sources += list(
        (ROOT / "docs/development-artifacts/knowledge-method-factorial-1000-v1").glob("*.md")
    )

    def copy(source):
        relative = source.relative_to(ROOT)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        return str(relative), hashlib.sha256(target.read_bytes()).hexdigest()

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        manifest = dict(pool.map(copy, sorted(set(sources))))
    L.write(
        Path(plan["output"]) / "source-manifest.json", {"utc": L.utc(), "source_sha256": manifest}
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    scientific = validate(plan)
    if not args.execute:
        print(
            json.dumps(
                {"execute": False, "plan": plan, "per_arm_steps": 372300, "total_updates": 2978400},
                indent=2,
            )
        )
        return 0
    if socket.gethostname() != plan["host"]:
        raise ValueError("Host differs from inspected resource map")
    output = Path(plan["output"])
    for name in ["engineering-gate.json", "judge-download-complete.json"]:
        if not json.loads((output / name).read_text()).get("passed"):
            raise ValueError(f"Prerequisite failed: {name}")
    downloaded = json.loads((output / "judge-download-complete.json").read_text())
    if (downloaded.get("model"), downloaded.get("revision")) != (
        scientific["judge_model"],
        scientific["judge_revision"],
    ):
        raise ValueError("Downloaded judge identity differs from the scientific configuration")
    for name in ("deps", "upstream"):
        # Root-squashed persistent storage cannot create nested mount targets as daemon root.
        (output / name).mkdir(exist_ok=True)
    with (output / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output / "controller-process.json").exists():
            raise FileExistsError("Already submitted; do not restart")
        L.write(
            output / "controller-process.json",
            {"pid": os.getpid(), "host": socket.gethostname(), "utc": L.utc(), "plan": plan},
        )
        L.write(output / "controller-state.json", {"state": "freezing_source", "utc": L.utc()})
        freeze(plan)
        return Controller(plan, scientific).run()


if __name__ == "__main__":
    os.umask(0o022)
    raise SystemExit(main())
