"""Paired shared-KV training, inference ablation and atomic-edit propagation."""

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
BATCH = "shared-cache-branch-v1"
ARTIFACT = REPOSITORY / "docs/development-artifacts" / BATCH
RESULTS = REPOSITORY / "results" / BATCH


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def specifications(parent):
    from llm_memory_editability.grok_loop_model import flops
    from llm_memory_editability.shared_cache import ARMS, construct, executed_flops

    base = next(
        s
        for s in parent["specs"]
        if s["width"] == 128
        and s["layers"] == 1
        and s["repeats"] == 2
        and s["composition_count"] == "all"
    )
    result = []
    worlds = [
        (730011, "development", [2, 4]),
        (107260711, "confirmation", [4]),
        (107260712, "confirmation", [4]),
        (107260713, "confirmation", [4]),
    ]
    for world, phase, repeats in worlds:
        for repeat in repeats:
            for arm in ARMS:
                spec = dict(
                    base,
                    layers=4,
                    repeats=repeat,
                    world=world,
                    phase=phase,
                    memory_arm=arm,
                    window=2,
                    initialization=731011 if phase == "development" else 107260721,
                    stream_seed=732011 if phase == "development" else 107260731,
                )
                model = construct(spec, "cpu")
                budget = 128000 * flops(model.config, repeat, 192, 9, output_positions=9)
                per_step = executed_flops(model.config, repeat, 192, 9, arm)
                matched = budget // per_step
                spec.update(
                    name=f"w{world}-{arm}-r{repeat}",
                    job_type="training",
                    matched_compute_step=matched,
                    reference_compute_budget=budget,
                    per_step_flops=per_step,
                    nodes=sorted(set(base["nodes"] + [matched])),
                    checkpoint_nodes=sorted(set([0, 32000, 128000, matched])),
                    repeat_nodes=[128000],
                    test_repeats=[1, 2, 4, 8],
                )
                result.append(spec)
    return result


def freeze():
    from llm_memory_editability.latent_scaling import (
        build_world,
        data_digest,
        model_digest,
    )
    from llm_memory_editability.shared_cache import construct

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
    for world in {s["world"] for s in specs}:
        paired = [s for s in specs if s["world"] == world]
        assert len({s["initial_model_sha256"] for s in paired}) == 1
        assert len({s["data_sha256"] for s in paired}) == 1
    from llm_memory_editability.interface_editing import _hash_json, graph_cases

    editing = []
    for parent_spec in specs:
        if parent_spec["repeats"] != 4:
            continue
        cases, qualification = graph_cases(parent_spec, seed=107260741, per_cell=1, replay_n=32)
        editing.append(
            dict(
                name="edit-" + parent_spec["name"],
                job_type="editing",
                phase=parent_spec["phase"],
                world=parent_spec["world"],
                memory_arm=parent_spec["memory_arm"],
                parent_name=parent_spec["name"],
                parent_dir=str(RESULTS / "runs" / parent_spec["name"]),
                candidate_seed=107260741,
                per_cell=1,
                replay_n=32,
                nodes=[0, 32, 128, 512],
                lr=1e-4,
                replay_weight=1.0,
                parameter_scope="mlp",
                block_index=0,
                cases=cases,
                qualification=qualification,
                case_sha256=_hash_json(cases),
            )
        )
    prior = read(
        REPOSITORY
        / "results/latent-scaling-v1/w730011-i731011-d128-l1-r2-nall-s128000/complete.json"
    )
    assert specs[0]["data_sha256"] == prior["data_sha256"]
    paths = [
        *REPOSITORY.glob("src/llm_memory_editability/*.py"),
        REPOSITORY / "scripts/run_shared_cache_branch.py",
        REPOSITORY / "scripts/report_shared_cache_branch.py",
        REPOSITORY / "tests/test_shared_cache.py",
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
        editing_specs=editing,
        gpus=list(range(8)),
        created_utc=utc(),
        repository=str(REPOSITORY),
        runtime=read(REPOSITORY / "configs/composition-data-curves-v1.json")["runtime"],
        maximum_run_seconds=3600,
        maximum_edit_seconds=900,
        maximum_engineering_seconds=400,
        maximum_gpu_hours=25,
        updates=len(specs) * 128000,
        edit_updates=len(editing) * 8 * 512,
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
    spec = next(s for s in config["specs"] + config["editing_specs"] if s["name"] == name)
    if spec["job_type"] == "editing":
        from llm_memory_editability import interface_editing as editor
        from llm_memory_editability.shared_cache import construct

        editor.new_model = construct
        if not read(Path(spec["parent_dir"]) / "audit.json")["passed"]:
            raise RuntimeError("Editing requires an independently audited parent")
        if audit_only:
            editor.audit(out)
        else:
            editor.run(spec, out)
            manifest = read(out / "run.json")
            assert manifest["case_sha256"] == spec["case_sha256"]
        return
    from llm_memory_editability.shared_cache import install_training_adapter

    latent = install_training_adapter(spec)
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
        audit_interventions(spec, out)
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
    shutil.copy2(out / f"model-{spec['steps']:06d}.pt", out / "model.pt")
    evaluate_interventions(spec, out)
    if engineering:
        from llm_memory_editability import interface_editing as editor
        from llm_memory_editability.shared_cache import construct

        editor.new_model = construct
        edit_spec = dict(
            parent_dir=str(out),
            phase="engineering",
            candidate_seed=107260741,
            per_cell=1,
            replay_n=32,
            nodes=[0, 2],
            lr=1e-4,
            block_index=0,
        )
        editor.run(edit_spec, out / "edit-engineering")
        editor.audit(out / "edit-engineering")


def evaluate_interventions(spec, out, audit=False):
    import numpy as np
    import torch

    from llm_memory_editability import latent_scaling as latent
    from llm_memory_editability.shared_cache import construct

    torch.set_num_threads(1)
    saved = torch.load(out / "model.pt", map_location="cpu", weights_only=False)
    model = construct(saved["spec"], "cuda:0")
    model.load_state_dict(saved["model"])
    world = latent.build_world(spec)
    result = {}
    for condition in ["normal", "shared_off"]:
        model.disable_shared = condition == "shared_off"
        metrics, predictions = latent.evaluate(model, world, "low", "cuda:0")
        path = out / ("intervention-" + condition + ".npz")
        if audit:
            expected = read(out / "interventions.json")[condition]
            latent.compare_metrics(metrics, expected)
            with np.load(path) as original:
                latent.compare_predictions(predictions, original)
        else:
            result[condition] = metrics
            np.savez_compressed(path, **predictions)
    if not audit:
        write_json(out / "interventions.json", result)
    else:
        write_json(out / "intervention-audit.json", dict(passed=True, conditions=2, utc=utc()))


def audit_interventions(spec, out):
    evaluate_interventions(spec, out, audit=True)


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
    from llm_memory_editability.shared_cache import construct

    spec = next(s for s in config["specs"] if s["memory_arm"] == "shared_full")
    tokens = torch.tensor([[2, 3, 4, 5, 6, 7]], device="cuda:0")
    changed = tokens.clone()
    changed[:, 3:] = 20
    normal = construct(spec, "cuda:0")
    detached = construct(dict(spec, memory_arm="shared_detached"), "cuda:0")
    torch.testing.assert_close(normal(tokens), detached(tokens), rtol=0, atol=0)
    torch.testing.assert_close(normal(tokens)[:, :3], normal(changed)[:, :3], rtol=0, atol=0)
    normal(tokens)[:, -1].square().sum().backward()
    detached(tokens)[:, -1].square().sum().backward()
    assert (
        torch.linalg.vector_norm(
            normal.blocks[0].attention.qkv.weight.grad
            - detached.blocks[0].attention.qkv.weight.grad
        )
        > 1e-5
    )
    for arm in ["shared_full", "shared_window", "shared_detached"]:
        model_check = construct(dict(spec, memory_arm=arm), "cuda:0")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model_check(tokens).float().square().mean()
        loss.backward()
        assert torch.isfinite(loss)
        assert all(torch.isfinite(p.grad).all() for p in model_check.parameters())
    checks["shared_forward_backward_bf16"] = True
    checks["causal_shared_mask_and_detach"] = True
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
    script = Path(config["source_root"]) / "scripts/run_shared_cache_branch.py"
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
    job = next(s for s in config["specs"] + config["editing_specs"] if s["name"] == name)
    budget = (
        config["maximum_engineering_seconds"]
        if engineering
        else config["maximum_edit_seconds"]
        if job["job_type"] == "editing"
        else config["maximum_run_seconds"]
    )

    def remaining(limit):
        return max(1, min(limit, budget - (time.monotonic() - started)))

    try:
        if engineering:
            subprocess.run(
                [sys.executable, str(script), "preflight", *arguments],
                check=True,
                timeout=remaining(300),
            )
        for mode in ["train", "audit"]:
            subprocess.run(
                [sys.executable, "-u", str(script), mode, *arguments],
                check=True,
                timeout=remaining(budget if mode == "train" else 900),
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
    cname = f"lm-shared-cache-{'eng-' if engineering else ''}{gpu}-{name}"
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
        config["source_root"] + "/scripts/run_shared_cache_branch.py",
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
        queued=[s["name"] for s in config["specs"] + config["editing_specs"]],
        gpus=config["gpus"],
    )
    write_json(RESULTS / "controller-state.json", state)
    for batch in [
        "loop-block-depth-v1",
        "composition-data-curves-v1",
        "parametric-architecture-main-v1",
        "oo-optimizer-attribution-v1",
    ]:
        prior = read(REPOSITORY / "results" / batch / "controller-state.json")
        assert (
            prior["state"] == "complete" and not prior.get("active") and not prior.get("queued")
        ), batch
    assert all(gpu_free(g) for g in config["gpus"]), "Registered GPUs are not all idle"

    def check(gpu):
        p, log, out, _name = container(
            config,
            gpu,
            next(
                s
                for s in config["specs"]
                if s["memory_arm"] == "shared_window" and s["repeats"] == 4
            ),
            engineering=True,
        )
        code = p.wait()
        log.close()
        return code == 0 and read(out / "audit.json")["passed"]

    with ThreadPoolExecutor(max_workers=8) as pool:
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
    pending, active = list(config["specs"] + config["editing_specs"]), {}
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
                eligible = next(
                    (
                        s
                        for s in pending
                        if s["job_type"] == "training" or s["parent_name"] in state["completed"]
                    ),
                    None,
                )
                if eligible is not None:
                    pending.remove(eligible)
                    active[gpu] = (*container(config, gpu, eligible), eligible)
        blocked = [s for s in pending if s.get("parent_name") in state["failed"]]
        for spec in blocked:
            out = RESULTS / "runs" / spec["name"]
            out.mkdir(parents=True, exist_ok=True)
            write_json(
                out / "failure.json", dict(reason="parent_failed", parent=spec["parent_name"])
            )
            state["failed"].append(spec["name"])
            pending.remove(spec)
        state.update(
            active={str(g): dict(name=v[4]["name"], container=v[3]) for g, v in active.items()},
            queued=[s["name"] for s in pending],
            updated_utc=utc(),
        )
        write_json(RESULTS / "controller-state.json", state)
        time.sleep(5)
    report_cmd = [
        sys.executable,
        config["source_root"] + "/scripts/report_shared_cache_branch.py",
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
            config["source_root"] + "/scripts/run_shared_cache_branch.py",
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
