"""Freeze and execute a paired replication of state alignment in ordinary GPT."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from llm_memory_editability.grok_depth import utc, write_json

REPOSITORY = Path("/ossfs/workspace/llm-memory-editability")
BATCH = "independent-alignment-v1"
ARTIFACT = REPOSITORY / "docs/development-artifacts" / BATCH
RESULTS = REPOSITORY / "results" / BATCH
SCRIPT = "scripts/run_independent_alignment.py"


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def specifications():
    parent = read(REPOSITORY / "configs/representation-alignment-confirmation-v1.json")
    base = next(s for s in parent["specs"] if s["arm"] == "bridge_ce")
    result = []
    worlds = [(770011, "development", [771011])]
    worlds += [
        (w, "confirmation", [107270821, 107270822]) for w in [107270811, 107270812, 107270813]
    ]
    for world, phase, initializations in worlds:
        for initialization in initializations:
            for arm, weight in [("bridge_ce", 0.0), ("aligned", 0.3)]:
                spec = dict(
                    base,
                    layers=2,
                    repeats=1,
                    world=world,
                    phase=phase,
                    initialization=initialization,
                    arm=arm,
                    bridge_weight=0.3,
                    alignment_weight=weight,
                )
                spec["name"] = f"{phase}-w{world}-i{initialization}-{arm}"
                result.append(spec)
    return result


def freeze():
    from llm_memory_editability.independent_alignment import validate_model
    from llm_memory_editability.latent_scaling import build_world, data_digest, model_digest
    from llm_memory_editability.representation_alignment import new_model

    destination = ARTIFACT / "source"
    if destination.exists():
        raise FileExistsError(destination)
    parent = read(REPOSITORY / "configs/representation-alignment-confirmation-v1.json")
    scientific = [
        p for p in parent["source"] if p.startswith("src/") and "experiment_tracking" not in p
    ]
    assert all(sha(REPOSITORY / p) == parent["source"][p] for p in scientific)
    specs = specifications()
    for spec in specs:
        model = new_model(spec, "cpu")
        validate_model(spec, model)
        spec.update(
            initial_model_sha256=model_digest(model),
            parameters=sum(p.numel() for p in model.parameters()),
            data_sha256=data_digest(build_world(spec)),
        )
    for world, initialization in {(s["world"], s["initialization"]) for s in specs}:
        pair = [s for s in specs if (s["world"], s["initialization"]) == (world, initialization)]
        assert len(pair) == 2
        assert len({s["initial_model_sha256"] for s in pair}) == 1
        assert len({s["data_sha256"] for s in pair}) == 1
    paths = [
        *REPOSITORY.glob("src/llm_memory_editability/*.py"),
        REPOSITORY / SCRIPT,
        REPOSITORY / "scripts/run_shared_cache_branch.py",
        REPOSITORY / "scripts/report_independent_alignment.py",
        REPOSITORY / "scripts/audit_independent_alignment_tracking.py",
        REPOSITORY / "tests/test_independent_alignment.py",
        REPOSITORY / "configs/experiment-tracking-defaults.json",
        REPOSITORY / "configs/representation-alignment-confirmation-v1.json",
        ARTIFACT / "protocol.md",
        ARTIFACT / "literature-methods.md",
    ]
    hashes = {}
    for path in paths:
        relative = path.relative_to(REPOSITORY)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        hashes[str(relative)] = sha(target)
    config = dict(
        batch=BATCH,
        created_utc=utc(),
        repository=str(REPOSITORY),
        source_root=str(destination),
        source=hashes,
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        historical_scientific_sources_exact=True,
        historical_config_sha256=sha(
            REPOSITORY / "configs/representation-alignment-confirmation-v1.json"
        ),
        specs=specs,
        gpus=list(range(8)),
        runtime=read(REPOSITORY / "configs/composition-data-curves-v1.json")["runtime"],
        maximum_run_seconds=1200,
        maximum_engineering_seconds=300,
        development_prerequisite=0.99,
        updates=sum(s["steps"] for s in specs),
        maximum_allocated_gpu_hours=6,
    )
    write_json(ARTIFACT / "frozen-config.json", config)
    write_json(
        ARTIFACT / "execution-lock.json", dict(config_sha256=sha(ARTIFACT / "frozen-config.json"))
    )
    print(
        json.dumps(
            {"runs": len(specs), "updates": config["updates"], "parameters": specs[0]["parameters"]}
        )
    )


def checked_config(path):
    path = Path(path)
    assert sha(path) == read(path.with_name("execution-lock.json"))["config_sha256"]
    config = read(path)
    for relative, expected in config["source"].items():
        assert sha(Path(config["source_root"]) / relative) == expected, relative
    return config


def environment(config):
    return dict(
        os.environ,
        PYTHONPATH=config["source_root"] + "/src",
        LD_LIBRARY_PATH="/lib64"
        + (":" + os.environ["LD_LIBRARY_PATH"] if os.environ.get("LD_LIBRARY_PATH") else ""),
    )


def preflight(spec, out):
    import numpy as np
    import torch

    from llm_memory_editability.grok_depth import make_optimizer
    from llm_memory_editability.independent_alignment import validate_model
    from llm_memory_editability.latent_scaling import build_world, model_digest
    from llm_memory_editability.representation_alignment import (
        RepresentationStep,
        new_model,
        objective,
        pack_training,
    )

    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    assert torch.cuda.device_count() == 1
    for dtype in (torch.float32, torch.bfloat16):
        x = torch.randn(64, 64, device="cuda", dtype=dtype, requires_grad=True)
        loss = (x @ x.T).float().square().mean()
        loss.backward()
        torch.cuda.synchronize()
        assert torch.isfinite(loss) and torch.isfinite(x.grad).all()
    model = new_model(spec, "cuda:0")
    validate_model(spec, model)
    initial = model_digest(model)
    table = tuple(torch.as_tensor(a, device="cuda") for a in pack_training(build_world(spec))[0])
    lr = torch.tensor(spec["lr"], device="cuda")
    optimizer = make_optimizer(model, lr, spec["weight_decay"])
    graph = RepresentationStep(model, optimizer, table, spec["batch_size"], spec)
    assert model_digest(model) == initial
    index = torch.arange(spec["batch_size"], device="cuda")
    graph(index)
    reference = new_model(spec, "cuda:0")
    other_lr = torch.tensor(spec["lr"], device="cuda")
    other_opt = make_optimizer(reference, other_lr, spec["weight_decay"])
    other_opt.zero_grad(set_to_none=False)
    loss, _ = objective(
        reference, *(part[index] for part in table), spec["bridge_weight"], spec["alignment_weight"]
    )
    loss.backward()
    torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.0, foreach=True)
    other_opt.step()
    error = max(
        float((a - b).abs().max())
        for a, b in zip(model.parameters(), reference.parameters(), strict=True)
    )
    assert error <= 1e-6, error
    assert np.isfinite(error)
    write_json(
        out / "preflight.json",
        dict(
            passed=True,
            cuda_computation=True,
            fp32=True,
            bf16=True,
            graph_update_max_parameter_error=error,
            independent_parameters=True,
            torch=torch.__version__,
        ),
    )


def train_or_audit(config, name, out, mode, engineering):
    import torch

    from llm_memory_editability import independent_alignment as experiment

    spec = next(s for s in config["specs"] if s["name"] == name)
    if engineering:
        spec = dict(spec, steps=8, nodes=[0, 8])
    if mode == "preflight":
        preflight(spec, out)
    elif mode == "train":
        experiment.train(spec, out, config["source"], torch.device("cuda:0"))
    else:
        experiment.audit(out, torch.device("cuda:0"))


def worker(config, name, out, engineering):
    arguments = [
        "--config",
        str(ARTIFACT / "frozen-config.json"),
        "--name",
        name,
        "--out",
        str(out),
    ]
    if engineering:
        arguments.append("--engineering")
    budget = config["maximum_engineering_seconds" if engineering else "maximum_run_seconds"]
    began = time.monotonic()
    try:
        modes = ["preflight", "train", "audit"] if engineering else ["train", "audit"]
        for mode in modes:
            subprocess.run(
                [sys.executable, "-u", str(Path(config["source_root"]) / SCRIPT), mode, *arguments],
                check=True,
                timeout=max(1, budget - (time.monotonic() - began)),
            )
        write_json(
            out / "worker-completion.json",
            dict(passed=True, allocated_seconds=time.monotonic() - began),
        )
    except BaseException:
        write_json(out / "failure.json", dict(utc=utc(), traceback=traceback.format_exc()))
        raise


def container(config, gpu, spec, engineering=False):
    name = spec["name"]
    out = (
        RESULTS
        / ("engineering" if engineering else "runs")
        / (f"gpu{gpu}" if engineering else name)
    )
    out.mkdir(parents=True, exist_ok=False)
    cname = f"lm-independent-alignment-{'eng-' if engineering else ''}{gpu}-{name}"
    cmd = [
        "docker",
        "--context",
        config["runtime"]["docker_context"],
        "run",
        "--name",
        cname,
        "--label",
        "project=llm-memory-editability",
        "--label",
        "batch=" + BATCH,
        "--label",
        "run=" + name,
        "--network=none",
        "--gpus",
        f"device={gpu}",
        "--cpus",
        str(config["runtime"]["cpus"]),
        "--memory",
        config["runtime"]["memory"],
        "--shm-size=2g",
        "--read-only",
        "--tmpfs",
        "/tmp:rw,size=2g",
        "--mount",
        f"type=bind,src={REPOSITORY},dst={REPOSITORY},readonly",
        "--mount",
        f"type=bind,src={out},dst={out}",
        "--workdir",
        str(REPOSITORY),
    ]
    for key, value in dict(
        PHYSICAL_GPU=gpu,
        PYTHONDONTWRITEBYTECODE=1,
        OMP_NUM_THREADS=1,
        MKL_NUM_THREADS=1,
        PYTHONPATH=config["source_root"] + "/src",
        XDG_CACHE_HOME="/tmp/cache",
    ).items():
        cmd += ["--env", f"{key}={value}"]
    cmd += [
        config["runtime"]["image"],
        config["runtime"]["python"],
        "-u",
        config["source_root"] + "/" + SCRIPT,
        "worker",
        "--config",
        str(ARTIFACT / "frozen-config.json"),
        "--name",
        name,
        "--out",
        str(out),
    ]
    if engineering:
        cmd.append("--engineering")
    write_json(out / "container-command.json", dict(command=cmd, gpu=gpu, container=cname))
    log = (out / "worker.log").open("a")
    p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    for _ in range(30):
        inspected = subprocess.run(
            ["docker", "--context", "lm-memory", "inspect", cname], capture_output=True, text=True
        )
        if inspected.returncode == 0:
            info = json.loads(inspected.stdout)[0]
            if info["State"]["Pid"]:
                write_json(
                    out / "container-runtime.json",
                    dict(
                        host_pid=info["State"]["Pid"],
                        physical_gpu=gpu,
                        image=info["Image"],
                        container=cname,
                    ),
                )
                break
        time.sleep(0.2)
    return p, log, out, cname, spec


def prerequisite(config):
    rows = []
    for spec in config["specs"]:
        if spec["phase"] != "development":
            continue
        out = RESULTS / "runs" / spec["name"]
        m = read(out / "complete.json")["metrics"]
        d = read(out / "state-diagnostics.json")
        values = [m[t]["accuracy"] for t in ["common_atomic", "train_composite"]]
        values += [
            d[t]["first_hop_original_readout_accuracy"] for t in ["common_atomic", "strict_test"]
        ]
        rows.append(
            dict(
                name=spec["name"],
                prerequisite_values=values,
                passed=min(values) >= config["development_prerequisite"],
            )
        )
    result = dict(
        passed=all(r["passed"] for r in rows),
        runs=rows,
        composition_test_not_used=True,
        recipe_not_reselected=True,
    )
    write_json(RESULTS / "development-prerequisite.json", result)
    return result["passed"]


def controller(config):
    from run_shared_cache_branch import gpu_free

    RESULTS.mkdir(parents=True, exist_ok=True)
    lock = (RESULTS / "controller.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (RESULTS / "controller-state.json").exists():
        raise FileExistsError("Existing execution history; do not silently restart")
    for b in ["residual-cache-comparison-v1", "knowledge-change-v1"]:
        previous = read(REPOSITORY / "results" / b / "controller-state.json")
        assert previous["state"] == "complete" and not previous["active"] and not previous["queued"]
    assert all(gpu_free(g) for g in config["gpus"])
    state = dict(
        state="preflight",
        started_utc=utc(),
        completed=[],
        failed=[],
        active={},
        queued=[s["name"] for s in config["specs"]],
        gpus=config["gpus"],
    )
    write_json(RESULTS / "controller-state.json", state)

    def check(gpu):
        job = container(
            config, gpu, next(s for s in config["specs"] if s["arm"] == "aligned"), engineering=True
        )
        code = job[0].wait()
        job[1].close()
        return code == 0 and read(job[2] / "audit.json")["passed"]

    with ThreadPoolExecutor(max_workers=8) as pool:
        checks = list(pool.map(check, config["gpus"]))
    write_json(
        RESULTS / "preflight.json", dict(passed=all(checks), checks=checks, gpus=config["gpus"])
    )
    if not all(checks):
        state.update(state="finished_with_failures", reason="engineering_preflight")
        write_json(RESULTS / "controller-state.json", state)
        return
    cmd = [
        str(REPOSITORY / ".venv-wandb/bin/python"),
        "-u",
        "-m",
        "llm_memory_editability.curve_tracking",
        "--root",
        str(RESULTS),
        "--runs-dir",
        str(RESULTS / "runs"),
        "--defaults",
        config["source_root"] + "/configs/experiment-tracking-defaults.json",
    ]
    with (RESULTS / "tracking.log").open("a") as log:
        tracker = subprocess.Popen(
            cmd, stdout=log, stderr=subprocess.STDOUT, env=environment(config)
        )
    write_json(RESULTS / "tracking-process.json", dict(pid=tracker.pid, command=cmd))
    allocated = 0.0
    for phase in ["development", "confirmation"]:
        pending = [s for s in config["specs"] if s["phase"] == phase]
        active = {}
        state.update(state="running", phase=phase)
        while pending or active:
            for gpu, (p, log, out, _cname, spec) in list(active.items()):
                if p.poll() is None:
                    continue
                log.close()
                ok = (
                    p.returncode == 0
                    and (out / "audit.json").exists()
                    and read(out / "audit.json")["passed"]
                )
                state["completed" if ok else "failed"].append(spec["name"])
                if ok:
                    allocated += read(out / "worker-completion.json")["allocated_seconds"] / 3600
                write_json(out / "process-status.json", dict(returncode=p.returncode, utc=utc()))
                del active[gpu]
            for gpu in config["gpus"]:
                if pending and gpu not in active and gpu_free(gpu):
                    spec = pending.pop(0)
                    active[gpu] = container(config, gpu, spec)
            state.update(
                active={str(g): dict(name=v[4]["name"], container=v[3]) for g, v in active.items()},
                queued=[s["name"] for s in pending]
                + (
                    [s["name"] for s in config["specs"] if s["phase"] == "confirmation"]
                    if phase == "development"
                    else []
                ),
                updated_utc=utc(),
                completed_allocated_gpu_hours=allocated,
            )
            write_json(RESULTS / "controller-state.json", state)
            time.sleep(2)
        if state["failed"]:
            break
        if phase == "development" and not prerequisite(config):
            state.update(state="development_prerequisite_not_met", active={}, finished_utc=utc())
            write_json(RESULTS / "controller-state.json", state)
            return
    report = subprocess.run(
        [
            sys.executable,
            config["source_root"] + "/scripts/report_independent_alignment.py",
            "--config",
            str(ARTIFACT / "frozen-config.json"),
        ],
        env=environment(config),
    )
    state.update(
        state="finished_with_failures" if state["failed"] or report.returncode else "complete",
        active={},
        queued=[],
        report_returncode=report.returncode,
        finished_utc=utc(),
    )
    write_json(RESULTS / "controller-state.json", state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=["freeze", "launch", "controller", "worker", "train", "audit", "preflight"]
    )
    parser.add_argument("--config", type=Path, default=ARTIFACT / "frozen-config.json")
    parser.add_argument("--name")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--engineering", action="store_true")
    args = parser.parse_args()
    if args.mode == "freeze":
        freeze()
        return
    config = checked_config(args.config)
    if args.mode == "launch":
        RESULTS.mkdir(parents=True, exist_ok=True)
        if (RESULTS / "controller-process.json").exists():
            raise FileExistsError("Existing launch record")
        cmd = [
            sys.executable,
            "-u",
            config["source_root"] + "/" + SCRIPT,
            "controller",
            "--config",
            str(args.config),
        ]
        with (RESULTS / "controller.log").open("a") as log:
            p = subprocess.Popen(
                cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env=environment(config),
            )
        write_json(RESULTS / "controller-process.json", dict(pid=p.pid, command=cmd, utc=utc()))
        print(json.dumps({"pid": p.pid, "results": str(RESULTS)}))
    elif args.mode == "controller":
        controller(config)
    elif args.mode == "worker":
        worker(config, args.name, args.out, args.engineering)
    else:
        train_or_audit(config, args.name, args.out, args.mode, args.engineering)


if __name__ == "__main__":
    main()
