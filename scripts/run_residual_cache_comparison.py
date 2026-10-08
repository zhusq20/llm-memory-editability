"""Paired residual organization, first-loop KV access and local-edit propagation."""

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
BATCH = "residual-cache-comparison-v1"
ARTIFACT = REPOSITORY / "docs/development-artifacts" / BATCH
RESULTS = REPOSITORY / "results" / BATCH


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def specifications(parent):
    from llm_memory_editability.grok_loop_model import flops
    from llm_memory_editability.residual_cache import construct, executed_flops

    draft = read(ARTIFACT / "design-draft.json")
    worlds = {w["world"]: w for w in draft["world_specs"]}
    base = next(
        s
        for s in parent["specs"]
        if s["world"] == 730011 and s["repeats"] == 4 and s["memory_arm"] == "local"
    )
    result = []
    for cell in draft["training_cells"]:
        world = worlds[cell["world"]]
        spec = dict(base, **{k: v for k, v in cell.items() if k != "steps"})
        spec.update(
            repeats=cell["train_repeats"],
            job_type="training",
            steps=128000,
            initialization=world["initialization"],
            stream_seed=world["stream_seed"],
            controller_initialization_seed=world["controller_initialization_seed"],
        )
        model = construct(spec, "cpu")
        per_step = executed_flops(
            model.config, spec["repeats"], 192, 9, spec["memory_arm"], spec["residual_kind"]
        )
        budget = 128000 * flops(model.config, spec["repeats"], 192, 9, output_positions=9)
        cross = 128000 * flops(model.config, 4, 192, 9, output_positions=9)
        native_step, cross_step = budget // per_step, cross // per_step
        assert 0 < cross_step <= native_step <= 128000
        spec.update(
            matched_compute_step=native_step,
            reference_compute_budget=budget,
            cross_R_compute_step=cross_step,
            cross_R_compute_budget=cross,
            per_step_flops=per_step,
            nodes=sorted(set(draft["shared_setup"]["nodes"] + [native_step, cross_step])),
            checkpoint_nodes=sorted(set([0, 32000, 128000, native_step, cross_step])),
            repeat_nodes=[32000, 128000],
            test_repeats=[1, 2, 4, 8],
        )
        result.append(spec)
    return result


def freeze():
    import torch

    torch.set_num_threads(1)
    from llm_memory_editability.latent_scaling import (
        build_world,
        data_digest,
        model_digest,
    )
    from llm_memory_editability.residual_cache import construct

    destination = ARTIFACT / "source"
    if destination.exists():
        raise FileExistsError(destination)
    parent_path = (
        REPOSITORY / "docs/development-artifacts/shared-cache-branch-v1/frozen-config.json"
    )
    assert sha(parent_path) == read(ARTIFACT / "design-draft.json")["baseline"]["sha256"]
    parent = read(parent_path)
    specs = specifications(parent)
    reuse, rerun = [], []
    for spec in specs:
        model = construct(spec, "cpu")
        spec["initial_model_sha256"] = model_digest(model)
        spec["parameters"] = sum(p.numel() for p in model.parameters())
        assert spec["parameters"] == spec["analytical_parameters"]
        spec["data_sha256"] = data_digest(build_world(spec))
        core = {k: v for k, v in model.state_dict().items() if not k.startswith("connections.")}
        ordinary = construct(dict(spec, residual_kind="single", gamma=1), "cpu")
        import torch

        for key, value in ordinary.state_dict().items():
            torch.testing.assert_close(core[key], value, rtol=0, atol=0)
        spec["core_initial_model_sha256"] = model_digest(ordinary)
        world = next(
            w
            for w in read(ARTIFACT / "design-draft.json")["world_specs"]
            if w["world"] == spec["world"]
        )
        assert spec["data_sha256"] == world["data_sha256"]
        assert spec["core_initial_model_sha256"] == world["initial_model_sha256"]
        candidate = spec.get("historical_reuse_candidate")
        if candidate:
            old = REPOSITORY / "results" / candidate["batch"] / "runs" / candidate["run"]
            available = {r["step"] for r in read(old / "learning.json")}
            needed = {spec["matched_compute_step"], spec["cross_R_compute_step"]}
            if needed <= available and read(old / "audit.json")["passed"]:
                reuse.append(
                    dict(spec, reuse_dir=str(old), reuse_checkpoint_sha256=sha(old / "latest.pt"))
                )
                continue
            rerun.append(
                dict(
                    name=spec["name"],
                    reason="historical matched-compute nodes unavailable",
                    missing_nodes=sorted(needed - available),
                )
            )
    for world in {s["world"] for s in specs}:
        paired = [s for s in specs if s["world"] == world]
        assert len({s["core_initial_model_sha256"] for s in paired}) == 1
        assert len({s["data_sha256"] for s in paired}) == 1
    from llm_memory_editability.interface_editing import _hash_json, graph_cases

    editing = []
    for parent_spec in specs:
        if (
            parent_spec["repeats"] != 4
            or parent_spec["role"] != "primary"
            or parent_spec["residual_kind"] == "single"
        ):
            continue
        cases, qualification = graph_cases(parent_spec, seed=107260741, per_cell=1, replay_n=32)
        editing.append(
            dict(
                name="edit-" + parent_spec["name"],
                job_type="editing",
                phase=parent_spec["phase"],
                world=parent_spec["world"],
                memory_arm=parent_spec["memory_arm"],
                residual_kind=parent_spec["residual_kind"],
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
        old_edit = next(
            s
            for s in parent["editing_specs"]
            if s["world"] == parent_spec["world"] and s["memory_arm"] == parent_spec["memory_arm"]
        )
        assert editing[-1]["case_sha256"] == old_edit["case_sha256"]
    prior = read(
        REPOSITORY
        / "results/latent-scaling-v1/w730011-i731011-d128-l1-r2-nall-s128000/complete.json"
    )
    assert specs[0]["data_sha256"] == prior["data_sha256"]
    paths = [
        *REPOSITORY.glob("src/llm_memory_editability/*.py"),
        REPOSITORY / "scripts/run_residual_cache_comparison.py",
        REPOSITORY / "scripts/report_residual_cache_comparison.py",
        REPOSITORY / "tests/test_residual_cache.py",
        parent_path,
        REPOSITORY / "configs/experiment-tracking-defaults.json",
        ARTIFACT / "protocol.md",
        ARTIFACT / "design-draft.json",
        ARTIFACT / "container/Dockerfile",
        ARTIFACT / "container/runtime.json",
        ARTIFACT / "container/manifest.json",
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
        specs=[s for s in specs if s["name"] not in {r["name"] for r in reuse}],
        all_training_cells=specs,
        reuse_specs=reuse,
        reuse_rejections=rerun,
        editing_specs=editing,
        gpus=list(range(8)),
        preflight_gpus=[g for g in range(8) if gpu_free(g)],
        created_utc=utc(),
        repository=str(REPOSITORY),
        runtime=read(ARTIFACT / "container/runtime.json"),
        maximum_run_seconds=14400,
        maximum_edit_seconds=7200,
        maximum_engineering_seconds=900,
        maximum_gpu_hours=150,
        updates=(len(specs) - len(reuse)) * 128000,
        edit_updates=len(editing) * 8 * 512,
        bootstrap=read(RESULTS / "bootstrap" / "fused" / "benchmark.json"),
        historical_source_differences=[
            p
            for p, h in parent["source"].items()
            if (REPOSITORY / p).exists() and sha(REPOSITORY / p) != h
        ],
    )
    write_json(ARTIFACT / "frozen-config.json", config)
    write_json(
        ARTIFACT / "execution-lock.json", {"config_sha256": sha(ARTIFACT / "frozen-config.json")}
    )
    print(
        json.dumps(
            {
                "new_training": len(config["specs"]),
                "reused": len(reuse),
                "editing_parents": len(editing),
                "config": str(ARTIFACT / "frozen-config.json"),
            }
        )
    )


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
        from llm_memory_editability.residual_cache import construct

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
    from llm_memory_editability.residual_cache import install_training_adapter

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
    install_diagnostics(latent, spec, out)
    latent.train(spec, out, config["source"])
    shutil.copy2(out / f"model-{spec['steps']:06d}.pt", out / "model.pt")
    evaluate_interventions(spec, out)
    if engineering:
        from llm_memory_editability import interface_editing as editor
        from llm_memory_editability.residual_cache import construct

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
    from llm_memory_editability.residual_cache import construct

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


def install_diagnostics(latent, spec, out):
    import numpy as np
    import torch

    from llm_memory_editability.storage_composition import pack_sentences

    original = latent.evaluate
    graph_type = latent.FullTokenStep
    graph_holder, history = [], []

    def graph_factory(*args, **kwargs):
        graph = graph_type(*args, **kwargs)
        graph_holder.append(graph)
        return graph

    def evaluate(model, world, degree, device, repeats=None):
        result = original(model, world, degree, device, repeats)
        if repeats is None and len(history) < len(spec["nodes"]):
            # First eight rows per stratum, fixed by graph ordering, never by score.
            keys = ["common_atomic", "train_composite", "familiar_test", "strict_test"]
            tables = [pack_sentences(world[key][:8])[0] for key in keys]
            panel = torch.as_tensor(np.concatenate(tables), device=device)
            model.diagnostic_records = []
            model.diagnostic_product = None
            started = time.perf_counter()
            with torch.no_grad():
                model(panel)
            records = model.diagnostic_records
            model.diagnostic_records = None
            model.diagnostic_product = None
            row = dict(
                step=spec["nodes"][len(history)],
                panel_selection="first eight graph rows per four task strata",
                records=records,
                seconds=time.perf_counter() - started,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                gradient_norm_before_clip=float(graph_holder[0].grad_norm)
                if graph_holder
                else None,
            )
            history.append(row)
            write_json(out / "residual-diagnostics.json", history)
        return result

    latent.FullTokenStep = graph_factory
    latent.evaluate = evaluate


def reuse_audit(spec, out):
    """Read-only historical reload with the new adapter and fixed R scans."""
    import numpy as np
    import torch

    from llm_memory_editability import latent_scaling as latent
    from llm_memory_editability.residual_cache import construct

    old = Path(spec["reuse_dir"])
    assert sha(old / "latest.pt") == spec["reuse_checkpoint_sha256"]
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    payload = torch.load(old / "latest.pt", map_location="cpu", weights_only=False)
    historical_spec = payload["spec"]
    for key in [
        "world",
        "initialization",
        "stream_seed",
        "width",
        "layers",
        "heads",
        "dropout",
        "repeats",
        "steps",
        "batch_size",
        "lr",
        "weight_decay",
        "warmup",
        "schedule",
        "min_lr_ratio",
        "composition_count",
        "low_extra",
        "anchor_n",
    ]:
        assert historical_spec[key] == spec[key], key
    assert historical_spec.get("memory_arm", "local") == spec["memory_arm"]
    model = construct(spec, "cuda:0").eval()
    assert latent.model_digest(construct(spec, "cpu")) == spec["initial_model_sha256"]
    model.load_state_dict(payload["model"])
    world = latent.build_world(spec)
    assert latent.data_digest(world) == spec["data_sha256"] == payload["data_sha256"]
    metrics, predictions = latent.evaluate(model, world, "low", "cuda:0")
    latent.compare_metrics(metrics, read(old / "complete.json")["endpoint"]["metrics"])
    with np.load(old / "predictions-128000.npz") as saved:
        maximum = latent.compare_predictions(predictions, saved)
    repeats = {}
    for node in [32000, 128000]:
        saved = torch.load(old / f"model-{node:06d}.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model"])
        repeats[node] = {}
        for repeat in [1, 2, 4, 8]:
            measured, pred = latent.evaluate(model, world, "low", "cuda:0", repeat)
            repeats[node][repeat] = measured
            np.savez_compressed(out / f"repeat-{node:06d}-r{repeat:02d}.npz", **pred)
    write_json(out / "repeat-metrics.json", repeats)
    model.load_state_dict(payload["model"])
    install_diagnostics(latent, dict(spec, nodes=[128000]), out)
    latent.evaluate(model, world, "low", "cuda:0")
    write_json(
        out / "audit.json",
        dict(
            passed=True,
            source_dir=str(old),
            endpoint_tokens_exact=True,
            max_nll_difference=maximum,
            repeated_checks=8,
            utc=utc(),
        ),
    )


def benchmark(out):
    import numpy as np
    import torch

    from llm_memory_editability import latent_scaling as latent
    from llm_memory_editability.interface_editing import editable_parameters
    from llm_memory_editability.parametric_architecture import sinkhorn
    from llm_memory_editability.residual_cache import construct, make_optimizer, training_sinkhorn
    from llm_memory_editability.shared_cache import construct as old_construct
    from llm_memory_editability.storage_composition import FullTokenStep, pack_sentences

    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    parent = read(
        REPOSITORY / "docs/development-artifacts/shared-cache-branch-v1/frozen-config.json"
    )
    base = next(
        s
        for s in parent["specs"]
        if s["world"] == 730011 and s["repeats"] == 4 and s["memory_arm"] == "local"
    )
    tokens = torch.tensor([[2, 3, 4, 5, 6, 7]], device="cuda:0")
    changed = tokens.clone()
    changed[:, 3:] = 20
    for memory in ["local", "shared_full"]:
        old, new = (
            old_construct(dict(base, memory_arm=memory), "cuda:0"),
            construct(dict(base, memory_arm=memory), "cuda:0"),
        )
        torch.testing.assert_close(old(tokens), new(tokens), rtol=0, atol=0)
        old(tokens).square().sum().backward()
        new(tokens).square().sum().backward()
        for a, b in zip(old.parameters(), new.parameters(), strict=True):
            torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
    spec = dict(
        base,
        repeats=8,
        residual_kind="mhc",
        memory_arm="shared_full",
        gamma=1,
        controller_initialization_seed=107260751,
    )
    model = construct(spec, "cuda:0").eval()
    torch.testing.assert_close(model(tokens)[:, :3], model(changed)[:, :3], rtol=0, atol=0)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = model(tokens).float().square().mean()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    edited = construct(spec, "cuda:0")
    selected = editable_parameters(edited, "mlp", 0)
    assert selected and all(
        not p.requires_grad for n, p in edited.named_parameters() if n.startswith("connections.")
    )
    del edited, old, new, model
    # Capture a fresh model, as the scientific trainer does; engineering eager
    # backward on the legacy stream must not leave accumulation-stream history.
    model = construct(spec, "cuda:0")
    # Non-symmetric logits and a non-constant loss test the fused projection's
    # backward, rather than only its nearly identity initialization.
    logits = torch.randn(192, 9, 4, 4, device="cuda:0", requires_grad=True)
    target = torch.randn_like(logits)
    eager = sinkhorn(logits)
    eager_gradient = torch.autograd.grad((eager * target).sum(), logits)[0]
    fused = training_sinkhorn(logits)
    fused_gradient = torch.autograd.grad((fused * target).sum(), logits)[0]
    torch.testing.assert_close(eager, fused, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(eager_gradient, fused_gradient, rtol=1e-4, atol=1e-6)
    world = latent.build_world(spec)
    arrays = [
        pack_sentences(world[k][:64]) for k in ["common_atomic", "train_composite", "anchor_atomic"]
    ]
    table = tuple(
        torch.as_tensor(np.concatenate([a[i] for a in arrays]), device="cuda:0") for i in range(2)
    )
    lr = torch.tensor(1e-3, device="cuda:0")
    optimizer = make_optimizer(model, lr, 0.01)
    torch.cuda.reset_peak_memory_stats()
    began = time.perf_counter()
    graph = FullTokenStep(model, optimizer, table, 192)
    capture_seconds = time.perf_counter() - began
    indices = torch.arange(192, device="cuda:0")
    began = time.perf_counter()
    for _ in range(256):
        loss = graph(indices)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - began
    assert torch.isfinite(loss) and torch.isfinite(graph.grad_norm)
    model.eval()
    model.diagnostic_records = []
    model.diagnostic_product = None
    with torch.no_grad():
        model(tokens)
    records = model.diagnostic_records
    assert len(records) == 64
    write_json(
        out / "benchmark.json",
        dict(
            passed=True,
            longest_arm=spec["residual_kind"] + "/" + spec["memory_arm"] + "/R8",
            historical_forward_backward_exact=True,
            causal_mask_exact=True,
            bf16_backward_finite=True,
            edit_controllers_frozen=True,
            capture_seconds=capture_seconds,
            updates=256,
            seconds=seconds,
            updates_per_second=256 / seconds,
            projected_128k_training_seconds=seconds * 500,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            diagnostic_max_column_error=max(r["max_column_error"] for r in records),
            parameters=sum(p.numel() for p in model.parameters()),
            utc=utc(),
            sinkhorn_forward_and_backward_equivalent=True,
        ),
    )
    print(json.dumps(read(out / "benchmark.json")), flush=True)


def preflight(config, out):
    import torch

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

    gpu = int(os.environ["PHYSICAL_GPU"])
    for i, spec in enumerate(config["reuse_specs"]):
        if config["preflight_gpus"][i % len(config["preflight_gpus"])] == gpu:
            audit_out = out / ("reuse-" + spec["name"])
            audit_out.mkdir()
            reuse_audit(spec, audit_out)
    write_json(
        out / "cuda-and-parent-check.json",
        dict(passed=True, actual_compute=checks, historical_reuse_checked=True, utc=utc()),
    )


def worker(config, name, out, engineering):
    script = Path(config["source_root"]) / "scripts/run_residual_cache_comparison.py"
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
    cname = f"lm-residual-cache-{'eng-' if engineering else ''}{gpu}-{name}"
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
        "/tmp:rw,exec,size=2g",
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
        TORCHINDUCTOR_CACHE_DIR="/tmp/inductor",
        TRITON_CACHE_DIR="/tmp/triton",
    ).items():
        cmd += ["--env", f"{key}={value}"]
    cmd += [
        config["runtime"]["image"],
        config["runtime"]["python"],
        "-u",
        config["source_root"] + "/scripts/run_residual_cache_comparison.py",
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
        "shared-cache-branch-v1",
    ]:
        prior = read(REPOSITORY / "results" / batch / "controller-state.json")
        assert (
            prior["state"] == "complete" and not prior.get("active") and not prior.get("queued")
        ), batch
    assert config["preflight_gpus"], "No idle GPUs at freeze"
    assert all(gpu_free(g) for g in config["preflight_gpus"]), "Initial GPUs acquired elsewhere"

    def check(gpu):
        p, log, out, _name = container(
            config,
            gpu,
            next(
                s
                for s in config["specs"]
                if s["memory_arm"] == "shared_full"
                and s["repeats"] == 8
                and s["residual_kind"] == "mhc"
                and s["phase"] == "development"
                and s["gamma"] == 1
            ),
            engineering=True,
        )
        code = p.wait()
        log.close()
        return code == 0 and read(out / "audit.json")["passed"]

    with ThreadPoolExecutor(max_workers=len(config["preflight_gpus"])) as pool:
        passed = list(pool.map(check, config["preflight_gpus"]))
    write_json(
        RESULTS / "preflight.json",
        dict(passed=all(passed), gpus=config["preflight_gpus"], checks=passed),
    )
    if not all(passed):
        state.update(state="finished_with_failures", failure="preflight")
        write_json(RESULTS / "controller-state.json", state)
        return
    for i, spec in enumerate(config["reuse_specs"]):
        audit_out = (
            RESULTS
            / "engineering"
            / f"gpu{config['preflight_gpus'][i % len(config['preflight_gpus'])]}"
            / ("reuse-" + spec["name"])
        )
        assert read(audit_out / "audit.json")["passed"]
        destination = RESULTS / "reused" / spec["name"]
        destination.mkdir(parents=True)
        write_json(
            destination / "reference.json",
            dict(spec=spec, independent_audit_dir=str(audit_out), utc=utc()),
        )
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
    pending, active = (
        sorted(
            config["specs"],
            key=lambda s: (
                s["phase"] != "development",
                s["role"] != "primary",
                {"mhc": 0, "identity_mhc": 1, "single": 2}[s["residual_kind"]],
                -s["repeats"],
                s["memory_arm"],
            ),
        )
        + config["editing_specs"],
        {},
    )
    state["state"] = "running"
    state["reused"] = [s["name"] for s in config["reuse_specs"]]
    development = {s["name"] for s in config["specs"] if s["phase"] == "development"}
    validated_gpus = set(config["preflight_gpus"])
    gpu_seconds = (
        sum(
            read(RESULTS / "engineering" / f"gpu{g}" / "worker-completion.json")["seconds"]
            for g in config["preflight_gpus"]
        )
        + config["bootstrap"]["seconds"]
    )
    development_passed = False
    while pending or active:
        for gpu, (p, log, out, _cname, spec) in list(active.items()):
            if p.poll() is not None:
                log.close()
                ok = (
                    p.returncode == 0
                    and (out / "audit.json").exists()
                    and read(out / "audit.json")["passed"]
                )
                if spec.get("engineering_job"):
                    if ok:
                        validated_gpus.add(gpu)
                    else:
                        state.setdefault("gpu_validation_failures", []).append(gpu)
                else:
                    state["completed" if ok else "failed"].append(spec["name"])
                if (out / "worker-completion.json").exists():
                    gpu_seconds += read(out / "worker-completion.json")["seconds"]
                else:
                    gpu_seconds += (
                        config["maximum_edit_seconds"]
                        if spec["job_type"] == "editing"
                        else config["maximum_run_seconds"]
                    )
                write_json(out / "process-status.json", dict(returncode=p.returncode, utc=utc()))
                del active[gpu]
        if not development_passed and development <= set(state["completed"]):
            development_passed = True
            write_json(
                RESULTS / "development-audit.json",
                dict(
                    passed=True,
                    jobs=sorted(development),
                    admission="numerical and reload audit only; no accuracy selection",
                    utc=utc(),
                ),
            )
        budget_exhausted = gpu_seconds >= config["maximum_gpu_hours"] * 3600
        if budget_exhausted:
            for spec in pending:
                state["failed"].append(spec["name"])
            state["budget_unstarted"] = [s["name"] for s in pending]
            pending.clear()
        for gpu in config["gpus"]:
            if pending and gpu not in active and gpu_free(gpu):
                if gpu in state.get("gpu_validation_failures", []):
                    continue
                if gpu not in validated_gpus:
                    engineering_spec = next(
                        s
                        for s in config["specs"]
                        if s["phase"] == "development"
                        and s["residual_kind"] == "mhc"
                        and s["memory_arm"] == "shared_full"
                        and s["repeats"] == 8
                        and s["gamma"] == 1
                    )
                    active[gpu] = (
                        *container(config, gpu, engineering_spec, engineering=True),
                        dict(engineering_spec, engineering_job=True),
                    )
                    continue
                eligible = next(
                    (
                        s
                        for s in pending
                        if (
                            (s["phase"] == "development" or development_passed)
                            and (
                                s["job_type"] == "training"
                                or s["parent_name"] in state["completed"]
                            )
                        )
                    ),
                    None,
                )
                if eligible is not None:
                    pending.remove(eligible)
                    active[gpu] = (*container(config, gpu, eligible), eligible)
        blocked = [
            s
            for s in pending
            if s.get("parent_name") in state["failed"]
            or (set(state["failed"]) & development and s["phase"] != "development")
        ]
        for spec in blocked:
            out = RESULTS / "runs" / spec["name"]
            out.mkdir(parents=True, exist_ok=True)
            write_json(
                out / "failure.json",
                dict(reason="parent_or_development_audit_failed", parent=spec.get("parent_name")),
            )
            state["failed"].append(spec["name"])
            pending.remove(spec)
        state.update(
            active={str(g): dict(name=v[4]["name"], container=v[3]) for g, v in active.items()},
            queued=[s["name"] for s in pending],
            updated_utc=utc(),
            development_audited=development_passed,
            completed_gpu_hours=gpu_seconds / 3600,
            validated_gpus=sorted(validated_gpus),
        )
        write_json(RESULTS / "controller-state.json", state)
        time.sleep(5)
    report_cmd = [
        sys.executable,
        config["source_root"] + "/scripts/report_residual_cache_comparison.py",
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
        "mode",
        choices=[
            "benchmark",
            "freeze",
            "launch",
            "controller",
            "worker",
            "train",
            "audit",
            "preflight",
        ],
    )
    parser.add_argument("--config", type=Path, default=ARTIFACT / "frozen-config.json")
    parser.add_argument("--name")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--engineering", action="store_true")
    args = parser.parse_args()
    if args.mode == "benchmark":
        benchmark(args.out or RESULTS / "bootstrap")
        return
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
            config["source_root"] + "/scripts/run_residual_cache_comparison.py",
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
