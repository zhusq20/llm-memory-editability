"""Freeze, run and monitor the direct CE/projection addition experiment."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import io
import json
import os
import shutil
import subprocess
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BATCH = "ce-down-novel-fact-v1"
EXTRAS = (
    "src/llm_memory_editability/ce_down_projection.py",
    "scripts/run_ce_down_novel.py",
    "scripts/launch_ce_down_novel.py",
    "scripts/run_ttcd_novel_fact_transfer.py",
    "scripts/check_discoloop_edit_rounds.py",
    "tests/test_ce_down_projection.py",
    f"docs/development-artifacts/{BATCH}/protocol.md",
)


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def docker(*args, **kwargs):
    return subprocess.run(["docker", "--context", "lm-memory", *args], check=True,
                          capture_output=True, timeout=60, **kwargs)


def freeze(batch, plan_path):
    source = batch / "source"
    if source.exists():
        raise FileExistsError("This experiment is already frozen; do not resubmit")
    plan = json.loads(plan_path.read_text())
    original = json.loads(Path(plan["parent_source_manifest"]).read_text())
    source.mkdir(parents=True)
    for relative, expected in original["files"].items():
        path = Path(plan["parent_source"]) / relative
        if path.is_symlink() or sha(path) != expected:
            raise ValueError(f"Frozen parent changed: {relative}")
        target = source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    for relative in EXTRAS:
        target = source / relative
        if target.exists() and target.read_bytes() != (ROOT / relative).read_bytes():
            raise ValueError(f"Cannot replace frozen parent code: {relative}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    target = source / f"configs/{BATCH}.json"
    shutil.copyfile(plan_path, target)
    files = {str(p.relative_to(source)): sha(p) for p in sorted(source.rglob("*")) if p.is_file()}
    manifest = {"batch": BATCH, "files": files, "parent_manifest_sha256": sha(plan["parent_source_manifest"]),
                "engineering": plan["engineering"], "scientific_base_updates": 0}
    write(source / "source-manifest.json", manifest)
    return source, sha(source / "source-manifest.json")


def archive(source):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        for path in sorted(source.rglob("*")):
            if not path.is_file():
                continue
            relative = str(Path("frozen") / path.relative_to(source))
            content = path.read_bytes()
            info = tarfile.TarInfo(relative)
            info.size, info.mode = len(content), 0o444
            tar.addfile(info, io.BytesIO(content))
    return buffer.getvalue()


def idle_gpu():
    result = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu",
                             "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True)
    rows = [tuple(v.strip() for v in line.split(",")) for line in result.stdout.splitlines()]
    processes = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
                               check=True, capture_output=True, text=True).stdout
    for index, uuid, memory, utilization in rows:
        if int(memory) <= 1024 and int(utilization) <= 5 and uuid not in processes:
            lock_path = Path(f"/tmp/lm-memory-gpu-locks-{os.getuid()}/gpu-{index}.lock")
            lock_path.parent.mkdir(exist_ok=True)
            lock = lock_path.open("a")
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                lock.close()
                continue
            return {"index": int(index), "uuid": uuid}, lock
    return None, None


def run_stage(batch, plan, source, manifest_sha, stage, gpu):
    name = f"{BATCH}-{stage}"
    output = batch / "output"
    output.mkdir(exist_ok=True)
    args = ["create", "--name", name, "--user", "1015:1016", "--cpuset-cpus", "84-87",
            "--cpus", "4", "--memory", "24g", "--read-only", "--tmpfs", "/tmp:rw,size=2g",
            "--env", "PYTHONDONTWRITEBYTECODE=1", "--env", "PYTHONPATH=/frozen/src:/frozen/scripts",
            "--env", f"LM_SOURCE_MANIFEST_SHA256={manifest_sha}", "--env", "HOME=/tmp/home",
            "--mount", "type=bind,source=/ossfs/workspace/llm-memory-editability,target=/ossfs/workspace/llm-memory-editability,readonly",
            "--mount", f"type=bind,source={output},target=/out",
            "--mount", "type=bind,source=/home/siqizhu4/.cache/llm-memory-ttcd/wandb-token,target=/run/secrets/wandb,readonly",
            "--workdir", "/frozen"]
    if stage != "cloud":
        args += ["--gpus", f"device={gpu['uuid']}"]
    args += [plan["image"], "python", "scripts/run_ce_down_novel.py", stage,
             "--plan", f"configs/{BATCH}.json", "--root", "/out", "--device", "cuda:0"]
    created = docker(*args).stdout.decode().strip()
    copied = subprocess.run(["docker", "--context", "lm-memory", "cp", "-", f"{name}:/"],
                            input=archive(source), check=True, capture_output=True, timeout=60)
    if copied.returncode:
        raise RuntimeError("Frozen source copy failed")
    docker("start", name)
    write(batch / f"launch-{stage}.json", {"name": name, "id": created, "gpu": gpu, "argv": args})
    started = time.monotonic()
    while True:
        state = json.loads(docker("inspect", "--format", "{{json .State}}", name).stdout)
        write(batch / "controller-state.json", {"stage": stage, "container": name, "container_state": state,
                                               "elapsed_seconds": time.monotonic() - started, "gpu": gpu})
        if not state["Running"]:
            (batch / f"{stage}.log").write_bytes(docker("logs", name).stdout)
            write(batch / f"exit-{stage}.json", state)
            if state["ExitCode"]:
                raise RuntimeError(f"Stage failed: {stage}; preserve failure, no automatic retry")
            return
        if time.monotonic() - started > plan["stage_timeout_seconds"]:
            docker("stop", "--time", "30", name)
            raise TimeoutError(f"New experiment stage exceeded its timeout: {stage}")
        time.sleep(15)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=ROOT / f"configs/{BATCH}.json")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    batch = Path(plan["output_root"])
    batch.mkdir(parents=True, exist_ok=True)
    with (batch / "controller.lock").open("a") as controller_lock:
        fcntl.flock(controller_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (batch / "controller-process.json").exists():
            raise FileExistsError("Submission already exists; preserve its recovery state")
        source, manifest_sha = freeze(batch, args.plan)
        write(batch / "controller-process.json", {"pid": os.getpid(), "executed": args.execute,
                                                  "manifest_sha256": manifest_sha, "scope": plan["scope"]})
        if not args.execute:
            return
        gpu, lock, waiting = None, None, time.monotonic()
        try:
            while gpu is None:
                gpu, lock = idle_gpu()
                if gpu is None:
                    write(batch / "controller-state.json", {"state": "waiting_for_exclusive_idle_gpu"})
                    if time.monotonic() - waiting > plan["resource_wait_seconds"]:
                        raise TimeoutError("No exclusive idle GPU became available")
                    time.sleep(30)
            for stage in ("development", "evaluation", "audit", "cloud"):
                run_stage(batch, plan, source, manifest_sha, stage, gpu)
            write(batch / "completion-manifest.json", {"passed": True, "source_sha256": manifest_sha,
                                                       "base_parameter_updates": 0,
                                                       "stages": ["development", "evaluation", "audit", "cloud"]})
            write(batch / "controller-state.json", {"state": "complete", "passed": True, "gpu": gpu})
        except BaseException as error:
            write(batch / "controller-failure.json", {"error": repr(error)})
            raise
        finally:
            if lock is not None:
                lock.close()


if __name__ == "__main__":
    main()
