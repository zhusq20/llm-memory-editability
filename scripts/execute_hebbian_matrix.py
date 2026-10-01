#!/usr/bin/env python3
"""Resource-local coordinator; every stage waits for complete prerequisite receipts."""

from __future__ import annotations

import argparse
import itertools
import os
import subprocess
import sys
import time

import numpy as np

from llm_memory_editability.hebbian_learning import (
    ARTIFACTS,
    DATA,
    RESULTS,
    ROOT,
    config,
    now,
    read_json,
    sha256,
    write_json,
)


def worker(task_file, device):
    from llm_memory_editability.hebbian_model import QwenExperiment
    from llm_memory_editability.hebbian_train import train_run

    pools, texts = read_json(DATA / "pools.json"), read_json(DATA / "text.json")
    records = {r["case_id"]: r for rows in pools.values() for r in rows}
    engine = QwenExperiment(device)
    for task in read_json(task_file):
        task = dict(task)
        selected = [records[i] for i in task.pop("case_ids")]
        if task.get("parent_path"):
            task["parent_path"] = RESULTS / task["parent_path"]
        train_run(engine, records=selected, pools=pools, texts=texts, **task)


def task(phase, run_id, case_ids, condition, seed, episode_id, split, lr, steps, **kwargs):
    return dict(
        phase=phase,
        run_id=run_id,
        case_ids=case_ids,
        condition=condition,
        seed=seed,
        episode_id=episode_id,
        split=split,
        lr=lr,
        steps=steps,
        **kwargs,
    )


def run_stage(stage, tasks, gpus):
    dispatch = ARTIFACTS / "matrix"
    dispatch.mkdir(exist_ok=True)
    tasks = [t for t in tasks if not (RESULTS / t["run_id"] / "complete.json").exists()]
    if not tasks:
        print(stage, "already complete", flush=True)
        return
    launches = []
    processes = []
    for shard, gpu in enumerate(gpus):
        assigned = tasks[shard :: len(gpus)]
        if not assigned:
            continue
        path = dispatch / f"{stage}-slot{shard}-gpu{gpu}.json"
        write_json(path, assigned)
        log = (RESULTS / "dispatch" / f"{stage}-slot{shard}-gpu{gpu}.log").open("ab")
        command = [
            sys.executable,
            "-u",
            str(ROOT / "scripts/execute_hebbian_matrix.py"),
            "--worker",
            str(path),
            "--device",
            f"cuda:{gpu}",
        ]
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
        launches.append(
            {"pid": process.pid, "gpu": gpu, "tasks": len(assigned), "command": command}
        )
        processes.append(process)
    write_json(dispatch / f"{stage}-launch.json", {"time": now(), "launches": launches})
    print(stage, "started", len(tasks), "runs", flush=True)
    while any(p.poll() is None for p in processes):
        write_json(
            ARTIFACTS / "matrix-status.json",
            {
                "time": now(),
                "stage": stage,
                "running_workers": sum(p.poll() is None for p in processes),
                "new_runs_complete": sum(
                    (RESULTS / t["run_id"] / "complete.json").exists() for t in tasks
                ),
                "new_runs_expected": len(tasks),
                "status": "running",
            },
        )
        time.sleep(10)
    failures = [p.returncode for p in processes if p.returncode != 0]
    if failures:
        write_json(
            ARTIFACTS / "matrix-status.json",
            {"time": now(), "stage": stage, "status": "failed", "exit_codes": failures},
        )
        raise RuntimeError(f"{stage} failed: {failures}; inspect dispatch logs")
    if not all((RESULTS / t["run_id"] / "complete.json").exists() for t in tasks):
        raise RuntimeError(f"{stage}: missing receipts")
    print(stage, "complete", flush=True)


def matrix(gpus):
    from llm_memory_editability.hebbian_statistics import freeze_b
    from llm_memory_editability.hebbian_train import (
        frozen_training_sources,
        run_summary,
        select_config,
    )

    (RESULTS / "dispatch").mkdir(exist_ok=True)
    pools, episodes = read_json(DATA / "pools.json"), read_json(DATA / "episodes.json")
    form_ids = [r["case_id"] for r in pools["F_form"]]
    b_dev_paths = [
        RESULTS / f"B/dev-lr{lr:g}-e{e}"
        for lr in config()["adapt"]["learning_rates"]
        for e in range(len(episodes["B_dev"]))
    ]
    if not all((p / "complete.json").exists() for p in b_dev_paths):
        raise RuntimeError("B_dev must finish before this coordinator starts")
    if not (ARTIFACTS / "B-lock.json").exists():
        freeze_b()
    b_lr = read_json(ARTIFACTS / "B-lock.json")["learning_rate"]
    tasks = [
        task(
            "adapt", f"B/eval-lr{b_lr:g}-e{e}", ids, "B", 0, e, "eval", b_lr, 128, diagnostics=True
        )
        for e, ids in enumerate(episodes["B_eval"])
    ]
    run_stage("B-eval", tasks, gpus)
    if not (ARTIFACTS / "C-lock.json").exists():
        tasks = [
            task("form", f"C/dev-C0-lr{lr:g}", form_ids, "C0", 10, -1, "dev", lr, 256)
            for lr in config()["form"]["learning_rates"]
        ]
        run_stage("C-dev-learning-rate", tasks, gpus)
        lr_candidates = [
            {"learning_rate": t["lr"], "run_id": t["run_id"], **run_summary(RESULTS / t["run_id"])}
            for t in tasks
        ]
        chosen = select_config(lr_candidates, "answer_nll", False)
        form_lr = chosen["learning_rate"]
        tasks = [
            task(
                "form",
                f"C/dev-C1-lambda{lam:g}",
                form_ids,
                "C1",
                10,
                -1,
                "dev",
                form_lr,
                256,
                lambda_geo=lam,
            )
            for lam in config()["form"]["lambdas"]
        ]
        run_stage("C-dev-geometry", tasks, gpus)
        candidates = [{"condition": "C0", "lambda_geo": 0.0, **chosen}] + [
            {
                "condition": "C1",
                "lambda_geo": t["lambda_geo"],
                "run_id": t["run_id"],
                **run_summary(RESULTS / t["run_id"]),
            }
            for t in tasks
        ]
        tasks = []
        for item in candidates:
            for e, ids in enumerate(episodes["C_dev"]):
                name = f"C/adapt-dev-{item['condition']}-lambda{item['lambda_geo']:g}-e{e}"
                tasks.append(
                    task(
                        "adapt",
                        name,
                        ids,
                        item["condition"],
                        10,
                        e,
                        "dev",
                        b_lr,
                        64,
                        parent_path=item["run_id"] + "/checkpoint-0256.pt",
                    )
                )
        run_stage("C-dev-adaptation", tasks, gpus)
        for item in candidates:
            summaries = [
                run_summary(
                    RESULTS / f"C/adapt-dev-{item['condition']}-lambda{item['lambda_geo']:g}-e{e}"
                )
                for e in range(len(episodes["C_dev"]))
            ]
            item["future_q_auc"] = float(np.mean([r["q_auc"] for r in summaries]))
            item["adapt_results"] = summaries
        chosen = select_config(
            [r for r in candidates if r["condition"] == "C1"], "future_q_auc", True
        )
        lam = chosen["lambda_geo"]
        run_stage(
            "C-dev-group-control",
            [
                task(
                    "form",
                    "C/dev-C2-engineering",
                    form_ids,
                    "C2",
                    10,
                    -1,
                    "dev",
                    form_lr,
                    256,
                    lambda_geo=lam,
                )
            ],
            gpus,
        )
        write_json(
            ARTIFACTS / "C-lock.json",
            {
                "time": now(),
                "learning_rate": form_lr,
                "lambda_geo": lam,
                "adapt_learning_rate": b_lr,
                "lr_candidates": lr_candidates,
                "formation_candidates": candidates,
                "source_hashes": frozen_training_sources(),
                "coordinator_sha256": sha256(__file__),
                "data_lock_sha256": sha256(ARTIFACTS / "data-lock.json"),
                "seeds": [0, 1, 2],
                "conditions": ["C0", "C1", "C2"],
                "episodes": episodes["C_eval"],
                "status": "locked",
            },
        )
    lock = read_json(ARTIFACTS / "C-lock.json")
    tasks = [
        task(
            "form",
            f"C/form-{c}-s{s}",
            form_ids,
            c,
            s,
            -1,
            "eval",
            lock["learning_rate"],
            256,
            lambda_geo=0.0 if c == "C0" else lock["lambda_geo"],
        )
        for c, s in itertools.product(["C0", "C1", "C2"], [0, 1, 2])
    ]
    run_stage("C-form", tasks, gpus)
    tasks = [
        task(
            "adapt",
            f"C/adapt-{c}-s{s}-e{e}",
            ids,
            c,
            s,
            e,
            "eval",
            b_lr,
            128,
            parent_path=f"C/form-{c}-s{s}/checkpoint-0256.pt",
        )
        for c, s, (e, ids) in itertools.product(
            ["C0", "C1", "C2"], [0, 1, 2], enumerate(episodes["C_eval"])
        )
    ]
    run_stage("C-adapt", tasks, gpus)
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/report_hebbian_learning.py")], check=True, cwd=ROOT
    )
    write_json(ARTIFACTS / "matrix-status.json", {"time": now(), "status": "complete"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker")
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--gpus", default="3,4,5,7,8,9")
    args = parser.parse_args()
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    if args.worker:
        worker(args.worker, args.device)
    else:
        matrix([int(x) for x in args.gpus.split(",")])
