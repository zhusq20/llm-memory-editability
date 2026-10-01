"""Prepare, test, launch, resume, and inspect the plan 14.27 development batch."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

from tokenizers import Tokenizer

from llm_memory_editability.bios_original_data import (
    ATTRS,
    digest,
    load_json,
    prepare_world,
    verify_source,
    write_json,
)


def tokenizer(cfg):
    return Tokenizer.from_file(str(Path(cfg["data_root"]) / "tokenizer/tokenizer.json"))


def prepare(cfg):
    manifest = verify_source()
    tok = tokenizer(cfg)
    assert tok.get_vocab_size() == cfg["model"]["vocab_size"] == 50257
    audits = [prepare_world(cfg, world, tok) for world in cfg["world_seeds"]]
    artifact = Path("docs/development-artifacts/bios-original-v1")
    artifact.mkdir(parents=True, exist_ok=True)
    write_json(
        artifact / "data-audit.json",
        dict(
            worlds=audits,
            source_revision=manifest["revision"],
            tokenizer_sha256=digest(Path(cfg["data_root"]) / "tokenizer/tokenizer.json"),
        ),
    )
    print(json.dumps(audits, indent=2))


def worker(cfg, world, init, condition, device, phases):
    from llm_memory_editability.bios_original_train import evaluate, finetune, pretrain

    root = Path(cfg["result_root"]) / f"world-{world}" / f"init-{init}" / condition
    root.mkdir(parents=True, exist_ok=True)
    with (root / "worker.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            write_json(
                root / "worker-status.json",
                dict(state="running", phase="pretrain", pid=os.getpid(), updated_unix=time.time()),
            )
            checkpoint = pretrain(cfg, world, init, condition, device)
            if phases == "pretrain":
                write_json(
                    root / "worker-status.json",
                    dict(
                        state="complete", phases=phases, pid=os.getpid(), updated_unix=time.time()
                    ),
                )
                return
            write_json(
                root / "worker-status.json",
                dict(state="running", phase="adapt", pid=os.getpid(), updated_unix=time.time()),
            )
            tok = tokenizer(cfg)
            adapted = finetune(cfg, world, init, condition, "adapt", checkpoint, device, tok)
            # Task training starts promptly; full evaluation is an explicit later phase.
            branches = [condition] if condition != "MP" else ["MP", "MP-R", "MP-Random", "MP-CoT"]
            for branch in branches:
                write_json(
                    root / "worker-status.json",
                    dict(
                        state="running",
                        phase=f"task/{branch}",
                        pid=os.getpid(),
                        updated_unix=time.time(),
                    ),
                )
                finetune(cfg, world, init, branch, "task", adapted, device, tok)
            if phases == "all":
                evaluate(cfg, world, adapted, root / "adapt/evaluation", device, tok, tasks=ATTRS)
                for branch in branches:
                    path = root.parent / branch / "task"
                    evaluate(
                        cfg,
                        world,
                        path / "final.pt",
                        path / "evaluation",
                        device,
                        tok,
                        cot=branch == "MP-CoT",
                    )
            write_json(
                root / "worker-status.json",
                dict(state="complete", phases=phases, pid=os.getpid(), updated_unix=time.time()),
            )
        except Exception:
            write_json(
                root / "worker-status.json",
                dict(
                    state="failed",
                    error=traceback.format_exc(),
                    pid=os.getpid(),
                    updated_unix=time.time(),
                ),
            )
            raise


def launch(cfg, config_path, gpus, phases):
    verify_source()
    for world in cfg["world_seeds"]:
        data = Path(cfg["data_root"]) / f"world-{world}"
        for file, sha in load_json(data / "audit.json")["files"].items():
            if digest(data / file) != sha:
                raise ValueError(f"Frozen data changed: {data / file}")
    audit = load_json("docs/development-artifacts/bios-original-v1/data-audit.json")
    if digest(Path(cfg["data_root"]) / "tokenizer/tokenizer.json") != audit["tokenizer_sha256"]:
        raise ValueError("Tokenizer changed")
    if (
        load_json("docs/development-artifacts/bios-original-v1/pipeline-preflight.json")["status"]
        != "passed"
    ):
        raise ValueError("End-to-end preflight has not passed")
    root = Path(cfg["result_root"])
    root.mkdir(parents=True, exist_ok=True)
    assignments = [
        (w, i, c)
        for w in cfg["world_seeds"]
        for i in cfg["initialization_seeds"]
        for c in cfg["pretrain"]["conditions"]
    ]
    if len(gpus) < len(assignments):
        raise ValueError("Provide one free GPU per development pretraining worker")
    # Refuse to overlap an active process, including one belonging to another project.
    query = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    usage = {
        int(row.split(",")[0]): [int(x) for x in row.split(",")[1:]]
        for row in query.strip().splitlines()
    }
    for gpu in gpus[: len(assignments)]:
        if usage[gpu][0] > 1000 or usage[gpu][1] > 5:
            raise RuntimeError(f"GPU {gpu} is occupied: {usage[gpu]}")
    snapshot = root / "execution-source"
    if not snapshot.exists():
        for filename in [
            "scripts/run_bios_original.py",
            "configs/bios-original-development-v1.json",
            "src/llm_memory_editability/bios_original_data.py",
            "src/llm_memory_editability/bios_original_model.py",
            "src/llm_memory_editability/bios_original_train.py",
            "docs/hebbian-learning-plan-v1.md",
        ]:
            dest = snapshot / filename
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(filename, dest)
        package = snapshot / "src/llm_memory_editability/__init__.py"
        package.write_text("")
        files = {
            str(p.relative_to(snapshot)): digest(p) for p in snapshot.rglob("*") if p.is_file()
        }
        write_json(snapshot / "manifest.json", files)
        (snapshot / "environment.txt").write_text(
            subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
        )
    else:
        for file, sha in load_json(snapshot / "manifest.json").items():
            if digest(snapshot / file) != sha:
                raise ValueError("Execution snapshot changed")
        if load_json(snapshot / config_path) != cfg:
            raise ValueError("Configuration differs from execution snapshot")
    jobs = []
    for (world, init, condition), gpu in zip(assignments, gpus, strict=False):
        run = root / f"world-{world}" / f"init-{init}" / condition
        run.mkdir(parents=True, exist_ok=True)
        env = dict(
            os.environ,
            CUDA_VISIBLE_DEVICES=str(gpu),
            OMP_NUM_THREADS="4",
            TOKENIZERS_PARALLELISM="false",
            PYTHONPATH=str((snapshot / "src").resolve()),
        )
        cmd = [
            sys.executable,
            "-u",
            str(snapshot / "scripts/run_bios_original.py"),
            "--config",
            str(snapshot / config_path),
            "worker",
            "--world",
            str(world),
            "--init",
            str(init),
            "--condition",
            condition,
            "--device",
            "cuda:0",
            "--phases",
            phases,
        ]
        with (run / "worker.log").open("a") as out:
            proc = subprocess.Popen(
                cmd,
                env=env,
                stdout=out,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
        jobs.append(
            dict(
                world=world,
                init=init,
                condition=condition,
                gpu=gpu,
                pid=proc.pid,
                command=cmd,
                log=str(run / "worker.log"),
            )
        )
    record = dict(launched_unix=time.time(), phases=phases, jobs=jobs)
    write_json(root / "launch.json", record)
    write_json("docs/development-artifacts/bios-original-v1/launch.json", record)
    print(json.dumps(record, indent=2))


def status(cfg):
    root = Path(cfg["result_root"])
    for path in sorted(root.glob("world-*/init-*/*/worker-status.json")):
        rec = load_json(path)
        rec["run"] = str(path.parent)
        try:
            os.kill(rec["pid"], 0)
            rec["pid_exists"] = True
        except ProcessLookupError:
            rec["pid_exists"] = False
        progress = path.parent / "pretrain/status.json"
        if progress.exists():
            rec["pretrain"] = load_json(progress)
        print(json.dumps(rec))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/bios-original-development-v1.json")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    sub.add_parser("status")
    launch_parser = sub.add_parser("launch")
    launch_parser.add_argument("--gpus", type=int, nargs="+", required=True)
    launch_parser.add_argument("--phases", choices=["pretrain", "train", "all"], default="all")
    worker_parser = sub.add_parser("worker")
    worker_parser.add_argument("--world", type=int, required=True)
    worker_parser.add_argument("--init", type=int, default=1427)
    worker_parser.add_argument("--condition", choices=["S", "M", "MP"], required=True)
    worker_parser.add_argument("--device", default="cuda:0")
    worker_parser.add_argument("--phases", choices=["pretrain", "train", "all"], default="all")
    args = parser.parse_args()
    cfg = load_json(args.config)
    if args.command == "prepare":
        prepare(cfg)
    elif args.command == "launch":
        launch(cfg, args.config, args.gpus, args.phases)
    elif args.command == "worker":
        worker(cfg, args.world, args.init, args.condition, args.device, args.phases)
    else:
        status(cfg)


if __name__ == "__main__":
    main()
