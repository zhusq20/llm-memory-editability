#!/usr/bin/env python3
"""Native 300k Wang foundation with the token-local query controller.

The training/checkpoint/replay loop is adapted verbatim from the existing
supervised-alignment executor. Original merged sampling, CUDA graph restoration,
AdamW, bf16/dropout and answer/EOS evaluation are reused without changing frozen
historical sources. This batch adds one architecture, without auxiliary losses.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from transformers import GPT2Config

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from run_supervised_alignment import (  # noqa: E402
    PANELS,
    AlignmentStep,
    exposure_summary,
    gradient_diagnostic,
    graph_values,
    make_table,
    state_digest,
    validate_arm,
)

from llm_memory_editability.bios_model import ModelConfig, matmul_flops  # noqa: E402
from llm_memory_editability.grok_depth import utc, write_json  # noqa: E402
from llm_memory_editability.grokking_reproduction import (  # noqa: E402
    MergedBatchStream,
    ReproductionGPT,
    digest,
    evaluate_rows,
    learning_rate,
    model_digest,
    optimizer_for,
)
from llm_memory_editability.loop_query_control import (  # noqa: E402
    QueryControlGPT,
    control_accounting,
)


def architecture_flags(arm):
    return {
        "query_control": True,
        "control_dim": arm["control_dim"],
        "control_seed": arm["control_seed"],
        "shared_across_layers_and_rounds": True,
        "first_round_kv": False,
        "discoloop": False,
    }


def initialize(config, device, arm):
    root, spec = Path(config["input_root"]), config["spec"]
    metadata = json.loads((root / "data/complete.json").read_text())
    for name in ("world", "panels"):
        if digest(root / f"data/{name}.npz") != metadata[f"{name}_sha256"]:
            raise ValueError("Original graph/panels changed")
    saved = torch.load(config["initial_checkpoint"], map_location="cpu", weights_only=False)
    if saved["step"] != 0:
        raise ValueError("Pretraining must start at original step zero")
    for key in (
        "unique_layers",
        "repeats",
        "phi",
        "weight_decay",
        "initialization",
        "stream_seed",
        "lr",
        "warmup",
        "batch_size",
    ):
        if saved["spec"].get(key) != spec[key]:
            raise ValueError(f"Original recipe differs: {key}")
    torch.manual_seed(spec["initialization"])
    hf = GPT2Config.from_json_file(str(root / "reference/gpt2/config.json"))
    hf.vocab_size, hf.n_layer, hf.use_cache = metadata["vocab_size"], spec["unique_layers"], False
    model = QueryControlGPT(hf, spec["repeats"], arm["control_dim"], arm["control_seed"])
    source_hash = state_digest(saved["model"])
    constructed_base = {k: v for k, v in model.state_dict().items() if not k.startswith("control_")}
    if state_digest(constructed_base) != source_hash:
        raise ValueError("Constructor did not reconstruct the exact historical backbone")
    incompatible = model.load_state_dict(saved["model"], strict=False)
    expected = {f"control_{name}.weight" for name in ("input", "content", "recurrent", "query")}
    if set(incompatible.missing_keys) != expected or incompatible.unexpected_keys:
        raise ValueError("Only the four new controller matrices may be missing")
    model.to(device)
    copied_base = {k: v for k, v in model.state_dict().items() if not k.startswith("control_")}
    if state_digest(copied_base) != source_hash:
        raise AssertionError("Backbone tensor copy differs")
    torch.manual_seed(config["dropout_seed"])
    model.train()
    return (
        model,
        metadata,
        {
            "source_checkpoint": config["initial_checkpoint"],
            "source_checkpoint_sha256": digest(config["initial_checkpoint"]),
            "source_backbone_sha256": source_hash,
            "constructor_matches_source_weights": True,
            "initial_model_sha256": model_digest(model),
            "dropout_seed": config["dropout_seed"],
            "architecture": architecture_flags(arm),
            "rng_policy": "Explicit original CPU/CUDA seed after loading; new controller uses a "
            "separate preserved CPU RNG; identical cross-architecture dropout masks not assumed",
        },
    )


def validate_config(config):
    expected = {
        "unique_layers": 4,
        "repeats": 2,
        "phi": 7.2,
        "weight_decay": 0.1,
        "initialization": 791101,
        "stream_seed": 792101,
        "lr": 1e-4,
        "warmup": 2000,
        "batch_size": 512,
        "steps": 300000,
    }
    if config["batch"] != "loop-query-control-development-v1":
        raise ValueError("Wrong batch")
    if any(config["spec"].get(key) != value for key, value in expected.items()):
        raise ValueError("Preserve the original 300k recipe")
    if (
        config["arms"]
        != {
            "query_control": {
                "kind": "none",
                "coefficient": 0.0,
                "scope": "supervised",
                "first_round_kv": False,
                "discoloop": False,
                "control_dim": 32,
                "control_seed": 20261016,
            }
        }
        or config["dropout_seed"] != 20261012
    ):
        raise ValueError("One pure query-control arm; no DiscoLoop, KV or auxiliary loss")
    if config["evaluation_nodes"] != [0, 1000, 8000, 16000, 32000, 64000, 128000, 256000, 300000]:
        raise ValueError("Frozen curve nodes differ")


def preflight(config, out):
    """Full-size CUDA comparison and graph restoration; zero surviving updates."""
    device = torch.device("cuda:0")
    model, metadata, provenance = initialize(config, device, config["arms"]["query_control"])
    reference = ReproductionGPT(model.config, model.repeats).to(device).train()
    base = {k: v for k, v in model.state_dict().items() if not k.startswith("control_")}
    reference.load_state_dict(base, strict=True)
    world = dict(np.load(Path(config["input_root"]) / "data/world.npz"))
    table, _ = make_table(world, metadata, config["spec"], device)
    stream = MergedBatchStream(len(table[0]) - 1, config["spec"]["stream_seed"])
    indices = torch.as_tensor(stream.batch(config["spec"]["batch_size"])[0], device=device)
    x, positions, labels = (part[indices] for part in table)
    outputs, gradients, checks = [], {}, {}
    for candidate in (reference, model):
        torch.manual_seed(config["dropout_seed"])
        candidate.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = candidate(x, positions)
            loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), labels.flatten())
        loss.backward()
        torch.cuda.synchronize()
        outputs.append(logits.detach().cpu())
        if candidate is reference:
            gradients = {k: p.grad.detach().cpu().clone() for k, p in candidate.named_parameters()}
        else:
            checks["all_backbone_gradients_exact"] = all(
                torch.equal(gradients[k], p.grad.detach().cpu())
                for k, p in candidate.named_parameters()
                if not k.startswith("control_")
            )
            checks["query_map_receives_finite_nonzero_gradient"] = bool(
                torch.isfinite(candidate.control_query.weight.grad).all()
                and candidate.control_query.weight.grad.abs().sum() > 0
            )
    checks["initial_logits_exact"] = torch.equal(*outputs)
    del gradients, reference, outputs, candidate, logits, loss
    model.zero_grad(set_to_none=True)
    before = model_digest(model)
    torch.manual_seed(config["dropout_seed"])
    cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
    lr = torch.tensor(0.0, device=device)
    optimizer = optimizer_for(model, lr, config["spec"]["weight_decay"])
    graph = AlignmentStep(
        model, optimizer, table, config["spec"]["batch_size"], config["arms"]["query_control"]
    )
    checks["capture_restores_all_weights"] = before == model_digest(model)
    checks["capture_restores_cpu_rng"] = torch.equal(cpu_rng, torch.get_rng_state())
    checks["capture_restores_cuda_rng"] = torch.equal(cuda_rng, torch.cuda.get_rng_state())
    checks["capture_restores_adam_state"] = all(
        not bool(value.count_nonzero())
        for values in optimizer.state.values()
        for value in values.values()
        if torch.is_tensor(value)
    )
    del graph
    result = {
        "passed": all(checks.values()),
        "checks": checks,
        "provenance": provenance,
        "parameters": sum(p.numel() for p in model.parameters()),
        "surviving_training_updates": 0,
        "utc": utc(),
    }
    write_json(out / "result-preflight.json", result)
    if not result["passed"]:
        raise AssertionError("Full-size preflight failed")
    return result


def run(config, arm_name, out, smoke, config_path):
    out.mkdir(parents=True, exist_ok=False)
    try:
        _run(config, arm_name, out, smoke, config_path)
    except BaseException as error:
        failure = {
            "state": "failed",
            "updated_utc": utc(),
            "error": repr(error),
            "traceback": traceback.format_exc(),
        }
        write_json(out / "failure.json", failure)
        write_json(out / "status.json", failure)
        raise


def _run(config, arm_name, out, smoke, config_path):
    spec, arm = dict(config["spec"]), config["arms"][arm_name]
    validate_arm(arm)
    if smoke:
        raise ValueError("This batch has no short-training stage")
    nodes = [0, spec["steps"]] if smoke else config["evaluation_nodes"]
    if nodes != sorted(set(nodes)) or nodes[0] != 0 or nodes[-1] != spec["steps"]:
        raise ValueError("Evaluation nodes must include zero and the exact fixed endpoint")
    device = torch.device("cuda:0")
    started = time.perf_counter()
    write_json(out / "status.json", {"state": "initializing", "updated_utc": utc(), "step": 0})
    model, metadata, provenance = initialize(config, device, arm)
    input_root = Path(config["input_root"])
    world = dict(np.load(input_root / "data/world.npz"))
    panels = dict(np.load(input_root / "data/panels.npz"))
    if set(panels) != set(PANELS):
        raise ValueError("Source must contain the seven frozen panels")
    table, selected = make_table(world, metadata, spec, device)
    stream = MergedBatchStream(len(table[0]) - 1, spec["stream_seed"])
    counts = np.zeros(stream.size, dtype=np.int64)
    lr = torch.tensor(0.0, device=device)
    optimizer = optimizer_for(model, lr, spec["weight_decay"])
    diagnostic_stream = MergedBatchStream(stream.size, spec["stream_seed"])
    diagnostic_indices = torch.as_tensor(
        diagnostic_stream.batch(spec["batch_size"])[0], device=device
    )
    diagnostic = gradient_diagnostic(model, table, diagnostic_indices, arm, device)
    write_json(out / "gradient-diagnostic.json", diagnostic)
    graph_started = time.perf_counter()
    graph = AlignmentStep(model, optimizer, table, spec["batch_size"], arm)
    capture_seconds = time.perf_counter() - graph_started
    if model_digest(model) != provenance["initial_model_sha256"]:
        raise AssertionError("CUDA graph capture changed step-zero weights")
    parameters = sum(p.numel() for p in model.parameters())
    flop_step = matmul_flops(
        ModelConfig(
            metadata["vocab_size"],
            model.config.n_embd,
            model.config.n_layer * model.repeats,
            model.config.n_head,
            1024,
        ),
        spec["batch_size"],
        sequence=4,
        output_positions=2,
    )
    accounting = control_accounting(spec, model.config.n_embd, model.control_dim)
    extra_control_flops = accounting["additional_training_matmul_flops_per_step"]
    run_metadata = {
        "batch": config["batch"],
        "arm": arm_name,
        "objective": arm,
        "spec": spec,
        "config_sha256": digest(config_path),
        "engineering_only": smoke,
        "parameters": parameters,
        "active_parameters": parameters,
        "vocab_size": metadata["vocab_size"],
        "world_sha256": metadata["world_sha256"],
        "panels_sha256": metadata["panels_sha256"],
        "atomic_examples": len(world["atoms"]),
        "composition_examples": len(selected),
        "evaluation_scope": "seven original frozen panels; full training pool not evaluated",
        "evaluation_batch_size": config.get("evaluation_batch_size", 256),
        "supervision": "Original final-round answer/EOS CE only; no auxiliary or bridge labels",
        "control_accounting": accounting,
        "estimated_matmul_training_flops_per_step": {
            "base": flop_step,
            "query_control": extra_control_flops,
        },
        "estimated_additional_inference_matmul_flops_per_sequence": accounting[
            "additional_inference_matmul_flops_per_sequence"
        ],
        "dtype": "bf16 AMP; float32 model and Adam state",
        "pid": os.getpid(),
        "hostname": platform.node(),
        "gpu_name": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "started_utc": utc(),
        "capture_seconds": capture_seconds,
        "flops_note": "Matrix products only; train=3*forward. Includes input/controller/query "
        "maps and padded slots; excludes RMSNorm, tanh, softmax, additions and memory traffic. "
        "The query delta is computed once per round and reused across physical layers.",
        **provenance,
    }
    write_json(out / "run.json", run_metadata)
    history, training_seconds, evaluation_seconds, step = [], 0.0, 0.0, 0

    def checkpoint(final=False):
        saved = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "stream": stream.state_dict(),
            "counts": counts,
            "step": step,
            "spec": spec,
            "arm": arm,
            "config": config,
            "arm_name": arm_name,
            "architecture": architecture_flags(arm),
            "cpu_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(device),
            "training_seconds": training_seconds,
            "evaluation_seconds": evaluation_seconds,
        }
        torch.save(saved, out / "latest.tmp.pt")
        (out / "latest.tmp.pt").replace(out / "latest.pt")
        if final:
            torch.save(
                {
                    "model": model.state_dict(),
                    "spec": spec,
                    "step": step,
                    "arm_name": arm_name,
                    "arm": arm,
                    "architecture": architecture_flags(arm),
                    "model_sha256": model_digest(model),
                },
                out / f"checkpoint-{step:07d}.pt",
            )
        np.savez_compressed(out / "exposure.npz", counts=counts, selected=selected)

    def measure(values=None):
        nonlocal evaluation_seconds
        torch.cuda.synchronize()
        before = time.perf_counter()
        metrics = {}
        for name in PANELS:
            metrics[name], prediction = evaluate_rows(
                model, panels[name], metadata, device, config.get("evaluation_batch_size", 256)
            )
            np.savez_compressed(out / f"predictions-{step:07d}-{name}.npz", **prediction)
        torch.cuda.synchronize()
        evaluation_seconds += time.perf_counter() - before
        record = {
            "step": step,
            "created_utc": utc(),
            "metrics": metrics,
            "training_seconds": training_seconds,
            "evaluation_seconds": evaluation_seconds,
            "wall_seconds": time.perf_counter() - started,
            "estimated_base_matmul_training_flops": step * flop_step,
            "estimated_extra_control_flops": step * extra_control_flops,
            "estimated_matmul_training_flops": step * (flop_step + extra_control_flops),
            **exposure_summary(counts, len(world["atoms"]), step, spec["batch_size"]),
            **(values or {}),
        }
        history.append(record)
        write_json(out / "learning.json", history)
        checkpoint(final=step in config.get("checkpoint_nodes", []) or step == spec["steps"])
        return record

    measure()
    values = {}
    with (out / "progress.jsonl").open("x") as progress:
        for end in nodes[1:]:
            while step < end:
                heartbeat = min(
                    end,
                    ((step // config.get("heartbeat_steps", 1000)) + 1)
                    * config.get("heartbeat_steps", 1000),
                )
                torch.cuda.synchronize()
                segment = time.perf_counter()
                while step < heartbeat:
                    n = min(128, heartbeat - step)
                    indices = np.stack([stream.batch(spec["batch_size"])[0] for _ in range(n)])
                    counts += np.bincount(indices.ravel(), minlength=len(counts) + 1)[: len(counts)]
                    gpu_indices = torch.as_tensor(indices, device=device)
                    for offset in range(n):
                        lr.fill_(learning_rate(spec, step + offset + 1))
                        graph(gpu_indices[offset])
                    step += n
                torch.cuda.synchronize()
                training_seconds += time.perf_counter() - segment
                values = graph_values(graph, arm["coefficient"])
                status = {
                    "state": "running",
                    "arm": arm_name,
                    "step": step,
                    "budget": spec["steps"],
                    "training_seconds": training_seconds,
                    "updated_utc": utc(),
                    **values,
                }
                progress.write(json.dumps(status, allow_nan=False) + "\n")
                progress.flush()
                write_json(out / "status.json", status)
                print(json.dumps(status, allow_nan=False), flush=True)
            measure(values)
    final_hash = model_digest(model)
    if final_hash == provenance["initial_model_sha256"]:
        raise AssertionError("No parameter update occurred")
    final = {
        "state": "complete",
        "step": step,
        "budget": spec["steps"],
        "arm": arm_name,
        "engineering_only": smoke,
        "parameters": parameters,
        "training_seconds": training_seconds,
        "evaluation_seconds": evaluation_seconds,
        "wall_seconds": time.perf_counter() - started,
        "updated_utc": utc(),
        "endpoint": history[-1],
        "model_sha256": final_hash,
        "final_checkpoint_sha256": digest(out / f"checkpoint-{step:07d}.pt"),
        "initial_model_sha256": provenance["initial_model_sha256"],
        "architecture": architecture_flags(arm),
        "independent_reload": "pending",
        **values,
    }
    write_json(out / "complete.json", final)
    write_json(out / "status.json", final)
    print(json.dumps({key: value for key, value in final.items() if key != "endpoint"}), flush=True)


def audit(config, arm_name, out, config_path):
    if (out / "audit.json").exists():
        raise FileExistsError("Independent audit already exists; preserve original audit")
    started = time.perf_counter()
    findings = {}
    try:
        run_metadata = json.loads((out / "run.json").read_text())
        if run_metadata["config_sha256"] != digest(config_path):
            raise AssertionError(
                "Audit configuration differs from the frozen training configuration"
            )
        complete = json.loads((out / "complete.json").read_text())
        endpoint = complete["endpoint"]
        if complete["arm"] != arm_name or run_metadata["objective"] != config["arms"][arm_name]:
            raise AssertionError("Audit arm configuration mismatch")
        checkpoint = out / f"checkpoint-{complete['step']:07d}.pt"
        findings["checkpoint_sha256_matches"] = (
            digest(checkpoint) == complete["final_checkpoint_sha256"]
        )
        model, metadata, _ = initialize(config, torch.device("cuda:0"), config["arms"][arm_name])
        saved = torch.load(checkpoint, map_location="cuda:0", weights_only=False)
        if (
            saved["arm"] != config["arms"][arm_name]
            or saved["spec"] != run_metadata["spec"]
            or saved["step"] != complete["step"]
            or saved["architecture"] != architecture_flags(config["arms"][arm_name])
            or run_metadata["architecture"] != saved["architecture"]
            or complete["architecture"] != saved["architecture"]
        ):
            raise AssertionError("Checkpoint architecture/objective flags or budget mismatch")
        findings["architecture"] = architecture_flags(config["arms"][arm_name])
        model.load_state_dict(saved["model"], strict=True)
        del saved
        findings["model_sha256_matches"] = model_digest(model) == complete["model_sha256"]
        panels = dict(np.load(Path(config["input_root"]) / "data/panels.npz"))
        atol, rtol = config.get("audit_loss_atol", 1e-6), config.get("audit_loss_rtol", 1e-6)
        findings["panels"] = {}
        for name in PANELS:
            metrics, prediction = evaluate_rows(
                model, panels[name], metadata, "cuda:0", run_metadata["evaluation_batch_size"]
            )
            with np.load(out / f"predictions-{complete['step']:07d}-{name}.npz") as original:
                rows_equal = np.array_equal(original["rows"], panels[name])
                answer_equal = np.array_equal(original["answer"], prediction["answer"])
                stop_equal = np.array_equal(original["stop"], prediction["stop"])
                loss_close = np.allclose(original["nll"], prediction["nll"], atol=atol, rtol=rtol)
                max_loss_delta = float(np.abs(original["nll"] - prediction["nll"]).max())
            scalar_matches = all(
                bool(np.isclose(value, endpoint["metrics"][name][key], atol=atol, rtol=rtol))
                for key, value in metrics.items()
            )
            findings["panels"][name] = {
                "n": len(panels[name]),
                "rows_exact": rows_equal,
                "answer_ids_exact": answer_equal,
                "eos_ids_exact": stop_equal,
                "loss_within_tolerance": bool(loss_close),
                "max_absolute_loss_difference": max_loss_delta,
                "scalar_metrics_match": scalar_matches,
                "passed": rows_equal
                and answer_equal
                and stop_equal
                and bool(loss_close)
                and scalar_matches,
                "metrics": metrics,
            }
        with np.load(out / "exposure.npz") as exposure:
            observed = exposure_summary(
                exposure["counts"],
                run_metadata["atomic_examples"],
                complete["step"],
                run_metadata["spec"]["batch_size"],
            )
            findings["exposure_counts_match"] = all(
                observed[key] == endpoint[key] for key in observed
            )
        passed = (
            findings["checkpoint_sha256_matches"]
            and findings["model_sha256_matches"]
            and findings["exposure_counts_match"]
            and all(panel["passed"] for panel in findings["panels"].values())
        )
        result = {
            "passed": passed,
            "arm": arm_name,
            "step": complete["step"],
            "process_id": os.getpid(),
            "audited_utc": utc(),
            "seconds": time.perf_counter() - started,
            "loss_atol": atol,
            "loss_rtol": rtol,
            "findings": findings,
            "scope": "fresh process reload; exact answer/EOS token IDs on all seven panels",
        }
    except BaseException as error:
        result = {
            "passed": False,
            "audited_utc": utc(),
            "error": repr(error),
            "traceback": traceback.format_exc(),
            "findings": findings,
        }
    write_json(out / "audit.json", result)
    print(json.dumps(result, allow_nan=False), flush=True)
    if not result["passed"]:
        raise SystemExit(1)


def main():
    import fcntl
    import threading

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("preflight", "train", "audit", "cloud-audit"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    validate_config(config)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"result-{args.stage}.json"
    with (args.out / "execution.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if path.exists() or (args.out / f"failure-{args.stage}.json").exists():
            raise FileExistsError("Preserve completed/failed stages; do not resubmit")
        science = args.out / "query_control"
        tracking_run, stop, observer = None, threading.Event(), None
        try:
            if args.stage in ("train", "audit"):
                import wandb

                tracking_run = wandb.init(
                    entity="zhusq20",
                    project="llm-memory-editability",
                    group=config["batch"],
                    name=f"{config['batch']}/{args.stage}",
                    job_type=args.stage,
                    config={**config, "stage": args.stage},
                    dir=str(args.out),
                    save_code=False,
                    settings=wandb.Settings(disable_git=True),
                )
                write_json(
                    args.out / f"wandb-{args.stage}.json",
                    {"id": tracking_run.id, "url": tracking_run.url},
                )
                tracking_run.define_metric("training/step")
                tracking_run.define_metric("*", step_metric="training/step")
                print(json.dumps({"wandb": tracking_run.url}), flush=True)
            observer_errors = []
            seen_curves, seen_progress = set(), set()

            def drain():
                if tracking_run is None:
                    return
                learning = science / "learning.json"
                if learning.exists():
                    for record in json.loads(learning.read_text()):
                        if record["step"] in seen_curves:
                            continue
                        values = {"training/step": record["step"]}
                        for panel, metrics in record["metrics"].items():
                            values.update({f"{panel}/{k}": v for k, v in metrics.items()})
                        values.update(
                            {
                                f"cost/{k}": v
                                for k, v in record.items()
                                if isinstance(v, (int, float))
                            }
                        )
                        tracking_run.log(values)
                        seen_curves.add(record["step"])
                progress = science / "progress.jsonl"
                if progress.exists():
                    for line in progress.read_text().splitlines():
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError:
                            break
                        if record["step"] not in seen_progress:
                            tracking_run.log(
                                {
                                    "training/step": record["step"],
                                    **{
                                        f"train/{k}": v
                                        for k, v in record.items()
                                        if isinstance(v, (int, float))
                                    },
                                }
                            )
                            seen_progress.add(record["step"])

            def observe():
                try:
                    while not stop.wait(15):
                        drain()
                except BaseException:
                    observer_errors.append(traceback.format_exc())

            if args.stage == "preflight":
                result = preflight(config, args.out)
            elif args.stage == "train":
                observer = threading.Thread(target=observe, daemon=True)
                observer.start()
                run(config, "query_control", science, False, args.config)
                stop.set()
                observer.join(timeout=60)
                drain()
                if observer_errors:
                    raise RuntimeError(observer_errors[0])
                result = json.loads((science / "complete.json").read_text())
            elif args.stage == "audit":
                audit(config, "query_control", science, args.config)
                result = json.loads((science / "audit.json").read_text())
            else:
                import wandb

                findings = []
                api = wandb.Api(timeout=60)
                for stage in ("train", "audit"):
                    tracking = json.loads((args.out / f"wandb-{stage}.json").read_text())
                    remote = api.run(f"zhusq20/llm-memory-editability/{tracking['id']}")
                    canonical = args.out / f"result-{stage}.json"
                    expected = digest(canonical)
                    if (
                        remote.state != "finished"
                        or remote.summary.get("result_sha256") != expected
                    ):
                        raise ValueError("Cloud state/result digest mismatch")
                    if remote.summary.get("terminal_step") != 300000:
                        raise ValueError("Cloud terminal step mismatch")
                    folder = args.out / "cloud-verification" / stage
                    folder.mkdir(parents=True, exist_ok=True)
                    downloaded = remote.file(canonical.name).download(
                        root=str(folder), replace=True
                    )
                    if digest(downloaded.name) != expected:
                        raise ValueError("Cloud canonical JSON bytes differ")
                    findings.append({**tracking, "state": remote.state, "sha256": expected})
                result = {"passed": True, "full_json_exact": True, "runs": findings, "utc": utc()}
            write_json(path, result)
            if tracking_run:
                tracking_run.summary.update(
                    {
                        "result_sha256": digest(path),
                        "terminal_step": result["step"],
                        "independent_audit_passed": result.get("passed", False),
                    }
                )
                tracking_run.save(str(path), base_path=str(args.out), policy="now")
                tracking_run.finish(exit_code=0)
            print(
                json.dumps(
                    {
                        "stage": args.stage,
                        "step": result.get("step"),
                        "passed": result.get("passed", True),
                    }
                ),
                flush=True,
            )
        except BaseException as error:
            stop.set()
            if observer:
                observer.join(timeout=60)
            write_json(
                args.out / f"failure-{args.stage}.json",
                {"error": repr(error), "traceback": traceback.format_exc(), "utc": utc()},
            )
            if tracking_run:
                tracking_run.finish(exit_code=1)
            raise


if __name__ == "__main__":
    main()
