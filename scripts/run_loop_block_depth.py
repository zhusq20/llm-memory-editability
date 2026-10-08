"""Paired four-block recurrence extension; historical trainers remain unchanged."""

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
BATCH = "loop-block-depth-v1"
ARTIFACT = REPOSITORY / "docs/development-artifacts" / BATCH
RESULTS = REPOSITORY / "results" / BATCH


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def specifications(parent):
    from llm_memory_editability.grok_loop_model import flops
    from llm_memory_editability.latent_scaling import construct

    base = next(
        s
        for s in parent["specs"]
        if s["width"] == 128
        and s["layers"] == 1
        and s["repeats"] == 2
        and s["composition_count"] == "all"
    )
    reference = construct(dict(base, layers=4, repeats=1), "cpu")
    budget = 128000 * flops(reference.config, 1, 192, 9, output_positions=9)
    result = []
    for layers, repeats in [(4, r) for r in [8, 6, 4, 3, 2, 1]] + [(2, 2), (2, 4), (8, 1)]:
        spec = dict(base, layers=layers, repeats=repeats)
        model = construct(spec, "cpu")
        per_step = flops(model.config, repeats, 192, 9, output_positions=9)
        matched = min(128000, budget // per_step)
        spec.update(
            name=f"l{layers}-r{repeats}",
            matched_compute_step=matched,
            reference_compute_budget=budget,
            per_step_flops=per_step,
            nodes=sorted(set(base["nodes"] + [matched])),
            checkpoint_nodes=sorted(set(base["checkpoint_nodes"] + [matched])),
            repeat_nodes=[32000, 128000],
            test_repeats=[1, 2, 3, 4, 6, 8, 12, 16],
        )
        result.append(spec)
    return result


def freeze():
    from llm_memory_editability.latent_scaling import (
        build_world,
        construct,
        data_digest,
        model_digest,
    )

    destination = ARTIFACT / "source"
    if destination.exists():
        raise FileExistsError(destination)
    parent = read(REPOSITORY / "configs/latent-scaling-v1.json")
    specs = specifications(parent)
    for spec in specs:
        model = construct(spec, "cpu")
        spec["initial_model_sha256"] = model_digest(model)
        spec["parameters"] = sum(p.numel() for p in model.parameters())
        spec["data_sha256"] = data_digest(build_world(spec))
    assert len({s["initial_model_sha256"] for s in specs if s["layers"] == 4}) == 1
    assert len({s["data_sha256"] for s in specs}) == 1
    prior = read(
        REPOSITORY
        / "results/latent-scaling-v1/w730011-i731011-d128-l1-r2-nall-s128000/complete.json"
    )
    assert specs[0]["data_sha256"] == prior["data_sha256"]
    paths = [
        *REPOSITORY.glob("src/llm_memory_editability/*.py"),
        REPOSITORY / "scripts/run_loop_block_depth.py",
        REPOSITORY / "scripts/report_loop_block_depth.py",
        REPOSITORY / "tests/test_loop_block_depth.py",
        REPOSITORY / "configs/latent-scaling-v1.json",
        REPOSITORY / "configs/experiment-tracking-defaults.json",
        ARTIFACT / "protocol.md",
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
        source_root=str(destination),
        source=hashes,
        specs=specs,
        gpus=[3, 4, 6, 7],
        created_utc=utc(),
        repository=str(REPOSITORY),
        runtime=read(REPOSITORY / "configs/composition-data-curves-v1.json")["runtime"],
        maximum_run_seconds=7200,
        updates=9 * 128000,
        historical_source_differences=[
            p for p, h in parent["source"].items() if sha(REPOSITORY / p) != h
        ],
    )
    write_json(ARTIFACT / "frozen-config.json", config)
    write_json(
        ARTIFACT / "execution-lock.json", {"config_sha256": sha(ARTIFACT / "frozen-config.json")}
    )
    print(json.dumps({"runs": len(specs), "config": str(ARTIFACT / "frozen-config.json")}))


def checked_config(path):
    path = Path(path)
    assert sha(path) == read(path.with_name("execution-lock.json"))["config_sha256"]
    config = read(path)
    for relative, expected in config["source"].items():
        assert sha(Path(config["source_root"]) / relative) == expected, relative
    return config


def train_or_audit(config, name, out, audit_only=False, engineering=False):
    from llm_memory_editability import latent_scaling as latent

    spec = next(s for s in config["specs"] if s["name"] == name)
    if engineering:
        spec = dict(
            spec,
            steps=8,
            nodes=[0, 8],
            checkpoint_nodes=[0, 8],
            repeat_nodes=[8],
            test_repeats=[1, 2, 8],
        )
    if audit_only:
        latent.audit(out)
        return
    initial = latent.construct(spec, "cpu")
    assert latent.model_digest(initial) == spec["initial_model_sha256"]
    world = latent.build_world(spec)
    assert latent.data_digest(world) == spec["data_sha256"]
    write_json(
        out / "run.json",
        dict(
            spec=spec,
            pid=os.getpid(),
            gpu=int(os.environ["PHYSICAL_GPU"]),
            world_sha256=spec["data_sha256"],
            initial_model_sha256=spec["initial_model_sha256"],
            parameters=spec["parameters"],
            tracking_group=BATCH,
            job_type="engineering" if engineering else "training",
        ),
    )
    latent.train(spec, out, config["source"])


def preflight(config, out):
    import numpy as np
    import torch

    from llm_memory_editability import latent_scaling as latent

    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    checks = {}
    for dtype in (torch.float32, torch.bfloat16):
        x = torch.randn(128, 128, device="cuda", dtype=dtype, requires_grad=True)
        y = x @ x.T
        y.float().square().mean().backward()
        torch.cuda.synchronize()
        assert torch.isfinite(y).all() and torch.isfinite(x.grad).all()
        checks[str(dtype)] = True
    parent = REPOSITORY / "results/latent-scaling-v1/w730011-i731011-d128-l1-r2-nall-s128000"
    result = read(parent / "complete.json")
    saved = torch.load(parent / "model-128000.pt", map_location="cpu", weights_only=False)
    model = latent.construct(saved["spec"], "cuda:0")
    assert (
        latent.model_digest(latent.construct(saved["spec"], "cpu"))
        == result["initial_model_sha256"]
    )
    model.load_state_dict(saved["model"])
    world = latent.build_world(saved["spec"])
    assert latent.data_digest(world) == result["data_sha256"]
    metrics, pred = latent.evaluate(model, world, "low", "cuda:0")
    latent.compare_metrics(metrics, result["endpoint"]["metrics"])
    with np.load(parent / "predictions-128000.npz") as original:
        maximum = latent.compare_predictions(pred, original)
    write_json(
        out / "cuda-and-parent-check.json",
        dict(
            passed=True,
            actual_compute=checks,
            parent_predictions_exact=True,
            max_nll_difference=maximum,
            utc=utc(),
        ),
    )


def worker(config, name, out, engineering):
    script = Path(config["source_root"]) / "scripts/run_loop_block_depth.py"
    arguments = [
        "--config",
        str(ARTIFACT / "frozen-config.json"),
        "--name",
        name,
        "--out",
        str(out),
    ]
    if engineering:
        arguments += ["--engineering"]
    started = time.monotonic()
    try:
        if engineering:
            subprocess.run(
                [sys.executable, str(script), "preflight", *arguments], check=True, timeout=300
            )
        for mode in ["train", "audit"]:
            subprocess.run(
                [sys.executable, "-u", str(script), mode, *arguments],
                check=True,
                timeout=config["maximum_run_seconds"] if mode == "train" else 900,
            )
        write_json(
            out / "worker-completion.json", dict(passed=True, seconds=time.monotonic() - started)
        )
    except BaseException:
        write_json(out / "failure.json", dict(utc=utc(), traceback=traceback.format_exc()))
        raise


def gpu_free(gpu):
    text = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"], text=True
    )
    used = {int(line.split(",")[0]): int(line.split(",")[1]) for line in text.splitlines()}
    if used[gpu] > 128:
        return False
    for context in ["default", "lm-memory", "d157"]:
        ids = subprocess.check_output(
            ["docker", "--context", context, "ps", "-q"], text=True
        ).split()
        if not ids:
            continue
        for item in json.loads(
            subprocess.check_output(["docker", "--context", context, "inspect", *ids], text=True)
        ):
            for request in item["HostConfig"].get("DeviceRequests") or []:
                if request.get("Count") == -1 or str(gpu) in (request.get("DeviceIDs") or []):
                    return False
    return True


def container(config, gpu, spec, engineering=False):
    name = spec["name"]
    out = (
        RESULTS
        / ("engineering" if engineering else "runs")
        / (f"gpu{gpu}" if engineering else name)
    )
    out.mkdir(parents=True, exist_ok=False)
    cname = f"lm-loop-depth-{'eng-' if engineering else ''}{gpu}-{name}"
    cmd = [
        "docker",
        "--context",
        "lm-memory",
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
        "2",
        "--memory",
        "12g",
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
        config["source_root"] + "/scripts/run_loop_block_depth.py",
        "worker",
        "--config",
        str(ARTIFACT / "frozen-config.json"),
        "--name",
        name,
        "--out",
        str(out),
    ]
    if engineering:
        cmd += ["--engineering"]
    write_json(out / "container-command.json", dict(command=cmd, gpu=gpu, container=cname))
    log = (out / "worker.log").open("a")
    process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    for _ in range(30):
        inspect = subprocess.run(
            ["docker", "--context", "lm-memory", "inspect", cname], capture_output=True, text=True
        )
        if inspect.returncode == 0:
            info = json.loads(inspect.stdout)[0]
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
    return process, log, out, cname


def environment(config):
    return dict(
        os.environ,
        PYTHONPATH=config["source_root"] + "/src",
        LD_LIBRARY_PATH="/lib64"
        + (":" + os.environ["LD_LIBRARY_PATH"] if os.environ.get("LD_LIBRARY_PATH") else ""),
    )


def controller(config):
    RESULTS.mkdir(parents=True, exist_ok=True)
    lock = (RESULTS / "controller.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (RESULTS / "controller-state.json").exists():
        raise FileExistsError(
            "Controller already has an execution history; do not silently restart"
        )
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
    prior = read(REPOSITORY / "results/composition-data-curves-v1/controller-state.json")
    assert not prior.get("queued") and prior.get("stage") == "load"
    assert all(gpu_free(g) for g in config["gpus"]), "Registered GPUs are not all idle"

    def check(gpu):
        p, log, out, _name = container(config, gpu, config["specs"][0], engineering=True)
        code = p.wait()
        log.close()
        return code == 0 and read(out / "audit.json")["passed"]

    with ThreadPoolExecutor(max_workers=4) as pool:
        passed = list(pool.map(check, config["gpus"]))
    write_json(
        RESULTS / "preflight.json", dict(passed=all(passed), gpus=config["gpus"], checks=passed)
    )
    if not all(passed):
        state.update(state="finished_with_failures", failure="preflight")
        write_json(RESULTS / "controller-state.json", state)
        return
    tracking_cmd = [
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
    tracking_log = (RESULTS / "tracking.log").open("a")
    tracker = subprocess.Popen(
        tracking_cmd, stdout=tracking_log, stderr=subprocess.STDOUT, env=environment(config)
    )
    write_json(RESULTS / "tracking-process.json", dict(pid=tracker.pid, command=tracking_cmd))
    pending, active = list(config["specs"]), {}
    state["state"] = "running"
    while pending or active:
        for gpu, (p, log, out, _cname, spec) in list(active.items()):
            if p.poll() is not None:
                log.close()
                ok = (
                    p.returncode == 0
                    and (out / "audit.json").exists()
                    and read(out / "audit.json")["passed"]
                )
                state["completed" if ok else "failed"].append(spec["name"])
                write_json(out / "process-status.json", dict(returncode=p.returncode, utc=utc()))
                del active[gpu]
        for gpu in config["gpus"]:
            if pending and gpu not in active and gpu_free(gpu):
                spec = pending.pop(0)
                active[gpu] = (*container(config, gpu, spec), spec)
        state.update(
            active={str(g): dict(name=v[4]["name"], container=v[3]) for g, v in active.items()},
            queued=[s["name"] for s in pending],
            updated_utc=utc(),
        )
        write_json(RESULTS / "controller-state.json", state)
        time.sleep(5)
    report_cmd = [
        sys.executable,
        config["source_root"] + "/scripts/report_loop_block_depth.py",
        "--config",
        str(ARTIFACT / "frozen-config.json"),
    ]
    report = subprocess.run(report_cmd, env=environment(config), check=False)
    state.update(
        state="finished_with_failures" if state["failed"] or report.returncode else "complete",
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
        command = [
            sys.executable,
            "-u",
            config["source_root"] + "/scripts/run_loop_block_depth.py",
            "controller",
            "--config",
            str(args.config),
        ]
        with (RESULTS / "controller.log").open("a") as log:
            p = subprocess.Popen(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env=environment(config),
            )
        write_json(RESULTS / "controller-process.json", dict(pid=p.pid, command=command, utc=utc()))
        print(json.dumps({"pid": p.pid, "results": str(RESULTS)}))
    elif args.mode == "controller":
        controller(config)
    elif args.mode == "worker":
        worker(config, args.name, args.out, args.engineering)
    elif args.mode == "preflight":
        preflight(config, args.out)
    else:
        train_or_audit(config, args.name, args.out, args.mode == "audit", args.engineering)


if __name__ == "__main__":
    main()
