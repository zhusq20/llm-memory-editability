#!/usr/bin/env python3
"""Use the audited native GPT/DiscoLoop training machinery on mixed 2/3/4-hop data."""

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
from torch.nn import functional as F
from transformers import GPT2Config

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from run_supervised_alignment import (  # noqa: E402
    AlignmentStep,
    architecture_flags,
    gradient_diagnostic,
    graph_values,
)

from llm_memory_editability.bios_model import ModelConfig, matmul_flops  # noqa: E402
from llm_memory_editability.grok_depth import utc, write_json  # noqa: E402
from llm_memory_editability.grokking_reproduction import (  # noqa: E402
    MergedBatchStream,
    ReproductionGPT,
    digest,
    learning_rate,
    model_digest,
    optimizer_for,
)
from llm_memory_editability.loop_hop_data import (  # noqa: E402
    exposure_summary,
    load,
    make_table,
    pack_rows,
    panel_names,
    path_details,
)
from llm_memory_editability.supervised_alignment import SupervisedAlignmentGPT  # noqa: E402


def arm_spec(config, name):
    structure = config["arms"][name]
    spec = {**config["training"], **structure, "name": name}
    arm = {
        "kind": "none",
        "coefficient": 0.0,
        "scope": "supervised",
        "first_round_kv": False,
        "discoloop": structure["architecture"] == "discoloop",
        "discoloop_alpha": 1.0,
        "discoloop_temperature": 1.0,
        "discoloop_eps": 1e-6,
    }
    return spec, arm


def initialize(config, name, metadata, device):
    spec, arm = arm_spec(config, name)
    torch.manual_seed(spec["initialization"])
    hf = GPT2Config(
        **config["model"],
        n_layer=spec["unique_layers"],
        vocab_size=metadata["vocab_size"],
        bos_token_id=metadata["pad_token"],
        eos_token_id=metadata["end_marker"],
        pad_token_id=metadata["pad_token"],
        use_cache=False,
    )
    model = SupervisedAlignmentGPT(hf, spec["repeats"], **architecture_flags(arm)).to(device)
    initial = model_digest(model)
    torch.manual_seed(config["dropout_seed"])
    return model, spec, arm, initial


@torch.no_grad()
def evaluate(model, rows, metadata, device, batch_size):
    was_training = model.training
    model.eval()
    packed = pack_rows(rows, metadata)
    answers, stops, nlls = [], [], []
    try:
        for begin in range(0, len(rows), batch_size):
            x, pos, labels = [
                torch.as_tensor(part[begin : begin + batch_size], device=device) for part in packed
            ]
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(x, pos)
                nll = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), reduction="none")
                answer = logits[:, 0].argmax(-1)
                x[torch.arange(len(x), device=device), pos[:, 1]] = answer
                stop = model(x, pos)[:, 1].argmax(-1)
            answers.append(answer.cpu().numpy())
            stops.append(stop.cpu().numpy())
            nlls.append(nll.view(-1, 2).float().cpu().numpy())
    finally:
        model.train(was_training)
    answer, stop, nll = np.concatenate(answers), np.concatenate(stops), np.concatenate(nlls)
    correct = answer == rows[:, -1] + metadata["entity_offset"]
    prediction = {"rows": rows, "answer": answer, "stop": stop, "nll": nll}
    return {
        "n": len(rows),
        "answer_accuracy": float(correct.mean()),
        "accuracy": float((correct & (stop == metadata["end_marker"])).mean()),
        "answer_nll": float(nll[:, 0].mean()),
        "nll": float(nll.mean()),
    }, prediction


def premise_metrics(rows, prediction, atomic_correct, world, metadata):
    nodes, edges = path_details(rows, metadata, world["permutations"])
    eligible = atomic_correct[edges].all(axis=1)
    correct = prediction["answer"] == rows[:, -1] + metadata["entity_offset"]
    nonrepeat = np.array([len(set(path)) == len(path) for path in nodes])
    return {
        "premise_correct_n": int(eligible.sum()),
        "premise_coverage": float(eligible.mean()),
        "answer_accuracy_given_premises": float(correct[eligible].mean())
        if eligible.any()
        else None,
        "nonrepeated_n": int(nonrepeat.sum()),
        "nonrepeated_answer_accuracy": float(correct[nonrepeat].mean())
        if nonrepeat.any()
        else None,
    }


def measure_panels(model, world, panels, metadata, device, batch_size):
    metrics, predictions = {}, {}
    for name in panel_names():
        metrics[name], predictions[name] = evaluate(
            model, panels[name], metadata, device, batch_size
        )
    atomic_correct = np.concatenate(
        [
            predictions[name]["answer"] == panels[name][:, -1] + metadata["entity_offset"]
            for name in ("atomic_id", "atomic_ood")
        ]
    )
    for name in panel_names()[2:]:
        metrics[name].update(
            premise_metrics(panels[name], predictions[name], atomic_correct, world, metadata)
        )
    return metrics, predictions


def accounting(model, spec, arm, metadata):
    base = matmul_flops(
        ModelConfig(
            metadata["vocab_size"],
            model.config.n_embd,
            spec["unique_layers"] * spec["repeats"],
            model.config.n_head,
            model.config.n_positions,
        ),
        spec["batch_size"],
        sequence=metadata["sequence_length"],
        output_positions=2,
    )
    channel = (
        12
        * spec["batch_size"]
        * metadata["sequence_length"]
        * model.config.n_embd
        * metadata["vocab_size"]
        * (spec["repeats"] - 1)
        if arm["discoloop"]
        else 0
    )
    return {
        "parameters": sum(p.numel() for p in model.parameters()),
        "executed_layers": spec["unique_layers"] * spec["repeats"],
        "channel_calls_per_forward": spec["repeats"] - 1 if arm["discoloop"] else 0,
        "base_matmul_training_flops_per_step": base,
        "channel_matmul_training_flops_per_step": channel,
        "estimated_matmul_training_flops_per_step": base + channel,
        "flops_note": "Dense padded T=6, two supervised output slots; training factor3; "
        "includes both differentiable vocabulary projections per channel. Excludes "
        "softmax, layer/RMS normalization, dropout, reductions and memory traffic.",
    }


def preflight(config, name, base):
    world, panels, metadata = load(config["input_root"])
    device = torch.device("cuda:0")
    model, spec, arm, initial = initialize(config, name, metadata, device)
    table = make_table(world, metadata, device)
    stream = MergedBatchStream(len(table[0]) - 1, spec["stream_seed"])
    index = torch.as_tensor(stream.batch(spec["batch_size"])[0], device=device)
    reference = ReproductionGPT(model.config, spec["repeats"]).to(device).train()
    reference.load_state_dict(model.state_dict(), strict=True)
    disabled = SupervisedAlignmentGPT(model.config, spec["repeats"], discoloop=False).to(device)
    disabled.load_state_dict(model.state_dict(), strict=True)
    hashes = [model_digest(reference), model_digest(disabled)]
    gradients, outputs = [], []
    for current in (reference, disabled):
        torch.manual_seed(config["dropout_seed"])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = current(table[0][index], table[1][index])
            loss = F.cross_entropy(logits.flatten(0, 1), table[2][index].flatten())
        loss.backward()
        gradients.append({n: p.grad.clone() for n, p in current.named_parameters()})
        outputs.append(logits.detach())
    baseline_exact = torch.equal(*outputs) and all(
        torch.equal(gradients[0][n], gradients[1][n]) for n in gradients[0]
    )
    del reference, disabled, gradients, outputs
    torch.manual_seed(config["dropout_seed"])
    diagnostic = gradient_diagnostic(model, table, index, arm, device)
    lr = torch.tensor(0.0, device=device)
    optimizer = optimizer_for(model, lr, spec["weight_decay"])
    cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
    graph = AlignmentStep(model, optimizer, table, spec["batch_size"], arm)
    findings = {
        "baseline_forward_and_all_gradients_exact": baseline_exact,
        "baseline_initial_weights_equal": hashes[0] == hashes[1] == initial,
        "graph_model_restored": model_digest(model) == initial,
        "graph_cpu_rng_restored": torch.equal(cpu_rng, torch.get_rng_state()),
        "graph_cuda_rng_restored": torch.equal(cuda_rng, torch.cuda.get_rng_state()),
        "graph_optimizer_state_zero": all(
            not torch.count_nonzero(v)
            for state in optimizer.state.values()
            for v in state.values()
            if torch.is_tensor(v)
        ),
    }
    # A real full-size captured update, backwards and synchronization.
    lr.fill_(spec["lr"])
    graph(index)
    torch.cuda.synchronize()
    findings["captured_update_finite"] = all(
        np.isfinite(v) for v in graph_values(graph, 0.0).values()
    )
    findings["captured_update_changes_parameters"] = model_digest(model) != initial
    result = {
        "passed": all(findings.values()),
        "engineering_only": True,
        "arm": name,
        "findings": findings,
        "initial_model_sha256": initial,
        "gradient_diagnostic": diagnostic,
        "accounting": accounting(model, spec, arm, metadata),
        "utc": utc(),
    }
    destination = base / "preflight" / f"{name}.json"
    if destination.exists():
        raise FileExistsError(destination)
    write_json(destination, result)
    if not result["passed"]:
        raise RuntimeError("Full-size CUDA preflight failed")


def train(config, name, out, smoke, config_path, base):
    out.mkdir(parents=True, exist_ok=False)
    world, panels, metadata = load(config["input_root"])
    device = torch.device("cuda:0")
    model, spec, arm, initial = initialize(config, name, metadata, device)
    if smoke:
        spec["steps"] = config["smoke_steps"]
    table = make_table(world, metadata, device)
    stream = MergedBatchStream(len(table[0]) - 1, spec["stream_seed"])
    counts = np.zeros(stream.size, dtype=np.int64)
    lr = torch.tensor(0.0, device=device)
    optimizer = optimizer_for(model, lr, spec["weight_decay"])
    graph = AlignmentStep(model, optimizer, table, spec["batch_size"], arm)
    if model_digest(model) != initial:
        raise AssertionError("Capture altered the initial scientific weights")
    cost = accounting(model, spec, arm, metadata)
    runtime_path = base / "runtime" / f"{name}-{'smoke' if smoke else 'train'}.json"
    runtime = json.loads(runtime_path.read_text()) if runtime_path.exists() else {}
    if runtime:
        write_json(out / "container-runtime.json", runtime)
    run_metadata = {
        "batch": config["batch"],
        "spec": spec,
        "arm": name,
        "architecture": architecture_flags(arm),
        "model": config["model"],
        "config_sha256": digest(config_path),
        "world_sha256": metadata["world_sha256"],
        "panels_sha256": metadata["panels_sha256"],
        "initial_model_sha256": initial,
        "engineering_only": smoke,
        "dtype": "float32 parameters and Adam state; bf16 AMP; TF32 enabled",
        "pid": os.getpid(),
        "gpu_name": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "hostname": platform.node(),
        "started_utc": utc(),
        "rng_policy": "Same explicit initialization and stream within paired architecture; "
        "independent unique layers; same dropout seed does not promise equal masks across R",
        **cost,
    }
    write_json(out / "run.json", run_metadata)
    started = time.perf_counter()
    history, training_seconds, step = [], 0.0, 0
    nodes = [0, spec["steps"]] if smoke else config["evaluation_nodes"]

    def measure(values):
        metrics, predictions = measure_panels(
            model, world, panels, metadata, device, config["evaluation_batch_size"]
        )
        for panel, prediction in predictions.items():
            np.savez_compressed(out / f"predictions-{step:07d}-{panel}.npz", **prediction)
        resource_path = base / "resources" / f"{name}.json"
        resource = json.loads(resource_path.read_text()) if resource_path.exists() else {}
        record = {
            "step": step,
            "metrics": metrics,
            "training_seconds": training_seconds,
            "wall_seconds": time.perf_counter() - started,
            "estimated_matmul_training_flops": step
            * cost["estimated_matmul_training_flops_per_step"],
            **exposure_summary(counts, world, metadata, step, spec["batch_size"]),
            **values,
            **(
                {
                    "physical_gpu": resource["gpu"],
                    "gpu_memory_used_mib": resource["used_mib"],
                    "gpu_utilization_percent": resource["utilization"],
                }
                if resource
                else {}
            ),
        }
        history.append(record)
        write_json(out / "learning.json", history)
        saved = {
            "model": model.state_dict(),
            "spec": spec,
            "arm": arm,
            "step": step,
            "architecture": architecture_flags(arm),
            "config_sha256": digest(config_path),
            "world_sha256": metadata["world_sha256"],
            "model_sha256": model_digest(model),
        }
        torch.save(saved, out / f"checkpoint-{step:07d}.pt")
        # Full recovery state: never reuse or overwrite an earlier checkpoint.
        temporary = out / "latest.tmp.pt"
        torch.save(
            {
                **saved,
                "optimizer": optimizer.state_dict(),
                "stream": stream.state_dict(),
                "counts": counts,
                "cpu_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(),
                "training_seconds": training_seconds,
            },
            temporary,
        )
        temporary.replace(out / "latest.pt")
        np.savez_compressed(out / "exposure.npz", counts=counts)
        return record

    measure({})
    values = {}
    for end in nodes[1:]:
        while step < end:
            boundary = min(end, (step // config["heartbeat_steps"] + 1) * config["heartbeat_steps"])
            torch.cuda.synchronize()
            segment = time.perf_counter()
            while step < boundary:
                size = min(128, boundary - step)
                indices = np.stack([stream.batch(spec["batch_size"])[0] for _ in range(size)])
                counts += np.bincount(indices.ravel(), minlength=len(counts) + 1)[: len(counts)]
                gpu_indices = torch.as_tensor(indices, device=device)
                for offset in range(size):
                    lr.fill_(learning_rate(spec, step + offset + 1))
                    graph(gpu_indices[offset])
                step += size
            torch.cuda.synchronize()
            training_seconds += time.perf_counter() - segment
            values = graph_values(graph, 0.0)
            status = {
                "state": "running",
                "arm": name,
                "step": step,
                "budget": spec["steps"],
                "training_seconds": training_seconds,
                "updated_utc": utc(),
                **values,
            }
            write_json(out / "status.json", status)
            with (out / "progress.jsonl").open("a") as progress:
                progress.write(json.dumps(status, allow_nan=False) + "\n")
            print(json.dumps(status, allow_nan=False), flush=True)
        measure(values)
    checkpoint = out / f"checkpoint-{step:07d}.pt"
    final = {
        "state": "complete",
        "arm": name,
        "step": step,
        "budget": spec["steps"],
        "engineering_only": smoke,
        "endpoint": history[-1],
        "initial_model_sha256": initial,
        "model_sha256": model_digest(model),
        "final_checkpoint_sha256": digest(checkpoint),
        "config_sha256": digest(config_path),
        "world_sha256": metadata["world_sha256"],
        "parameters": cost["parameters"],
        "training_seconds": training_seconds,
        "wall_seconds": time.perf_counter() - started,
        "updated_utc": utc(),
        "independent_reload": "pending",
    }
    if final["model_sha256"] == initial:
        raise AssertionError("No scientific parameter update occurred")
    write_json(out / "complete.json", final)
    write_json(out / "status.json", final)


def audit(config, name, out, config_path):
    if (out / "audit.json").exists():
        raise FileExistsError("Preserve the existing independent audit")
    world, panels, metadata = load(config["input_root"])
    complete = json.loads((out / "complete.json").read_text())
    checkpoint = out / f"checkpoint-{complete['step']:07d}.pt"
    model, spec, arm, _ = initialize(config, name, metadata, torch.device("cuda:0"))
    saved = torch.load(checkpoint, map_location="cuda:0", weights_only=False)
    if (
        saved["arm"] != arm
        or saved["architecture"] != architecture_flags(arm)
        or saved["spec"] != {**spec, "steps": complete["budget"]}
        or saved["step"] != complete["step"]
        or saved["world_sha256"] != metadata["world_sha256"]
        or saved["config_sha256"] != digest(config_path)
        or complete["config_sha256"] != digest(config_path)
    ):
        raise ValueError("Saved architecture, budget or data/config identity changed")
    model.load_state_dict(saved["model"], strict=True)
    if model_digest(model) != complete["model_sha256"]:
        raise ValueError("Reloaded model hash differs")
    metrics, predictions = measure_panels(
        model, world, panels, metadata, torch.device("cuda:0"), config["evaluation_batch_size"]
    )
    findings = {}
    for panel, prediction in predictions.items():
        with np.load(out / f"predictions-{complete['step']:07d}-{panel}.npz") as original:
            exact = all(
                np.array_equal(original[k], prediction[k]) for k in ("rows", "answer", "stop")
            )
            losses_close = np.allclose(original["nll"], prediction["nll"], atol=1e-6, rtol=1e-6)
        previous = complete["endpoint"]["metrics"][panel]
        scalar_match = all(
            previous[k] is v if v is None else np.isclose(previous[k], v, atol=1e-6, rtol=1e-6)
            for k, v in metrics[panel].items()
        )
        findings[panel] = {
            "passed": bool(exact and losses_close and scalar_match),
            "n": len(panels[panel]),
            "answer_and_eos_ids_exact": bool(exact),
            "metrics": metrics[panel],
        }
    with np.load(out / "exposure.npz") as exposure:
        observed = exposure_summary(
            exposure["counts"], world, metadata, complete["step"], spec["batch_size"]
        )
    exposure_match = all(complete["endpoint"][k] == v for k, v in observed.items())
    result = {
        "passed": all(row["passed"] for row in findings.values())
        and exposure_match
        and digest(checkpoint) == complete["final_checkpoint_sha256"],
        "arm": name,
        "step": complete["step"],
        "process_id": os.getpid(),
        "audited_utc": utc(),
        "panels": findings,
        "exposure_match": exposure_match,
        "scope": "fresh container; all full panels; exact answer/EOS IDs; all premise coverage",
    }
    write_json(out / "audit.json", result)
    if not result["passed"]:
        raise RuntimeError("Independent full-panel reload failed")


@torch.no_grad()
def traced_states(model, tokens, positions):
    """Observe the actual forward; no substitute attention/MLP implementation.

    ln_2 receives U and the native MLP returns MLP(ln_2(U)), including
    its dropout. Their sum is exactly the complete block residual state.
    The next first-layer ln_1 receives the post-boundary state.
    """
    states, pending, handles = [], {}, []
    layers = len(model.transformer.h)
    executed = [0]

    def first_input(module, args):
        if executed[0] and executed[0] % layers == 0 and model.discoloop:
            states.append(("after_boundary", executed[0], args[0].detach().clone()))

    handles.append(model.transformer.h[0].ln_1.register_forward_pre_hook(first_input))
    for index, block in enumerate(model.transformer.h):

        def before_norm(module, args, index=index):
            pending[index] = args[0]

        def after_mlp(module, args, output, index=index):
            executed[0] += 1
            states.append(("after_layer", executed[0], (pending[index] + output).detach().clone()))

        handles.append(block.ln_2.register_forward_pre_hook(before_norm))
        handles.append(block.mlp.register_forward_hook(after_mlp))
    try:
        logits, _, final = model.forward_with_states(tokens, positions)
        if executed[0] != layers * model.repeats or not torch.equal(states[-1][2], final):
            raise AssertionError("Observed block states disagree with the actual forward")
        return logits, states
    finally:
        for handle in handles:
            handle.remove()


def trace(config, name, out):
    if (out / "prefix-trace.json").exists():
        raise FileExistsError("Preserve the original prefix trace")
    world, panels, metadata = load(config["input_root"])
    complete = json.loads((out / "complete.json").read_text())
    if not json.loads((out / "audit.json").read_text())["passed"]:
        raise ValueError("Trace requires the independent endpoint audit")
    device = torch.device("cuda:0")
    model, spec, _, _ = initialize(config, name, metadata, device)
    saved = torch.load(
        out / f"checkpoint-{complete['step']:07d}.pt", map_location=device, weights_only=False
    )
    model.load_state_dict(saved["model"], strict=True)
    model.eval()
    summaries = {}
    arrays = {}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for k in (2, 3, 4):
            rows = panels[f"ood_{k}"][: config["trace_examples_per_hop"]]
            x, positions, _ = [torch.as_tensor(v, device=device) for v in pack_rows(rows, metadata)]
            direct = model(x, positions)
            observed, states = traced_states(model, x, positions)
            if not torch.equal(direct, observed):
                raise AssertionError("Observation changed native output")
            nodes, _ = path_details(rows, metadata, world["permutations"])
            targets = nodes[:, 1:] + metadata["entity_offset"]
            records = []
            for where, depth, state in states:
                hidden = model.transformer.ln_f(state)[:, 1 : k + 1]
                answer = F.linear(hidden, model.transformer.wte.weight).argmax(-1).cpu().numpy()
                arrays[f"k{k}-{where}-d{depth}"] = answer
                records.append(
                    {
                        "where": where,
                        "executed_layers": depth,
                        "loop": (depth - 1) // spec["unique_layers"] + 1,
                        "prefix_accuracy": (answer == targets).mean(axis=0).tolist(),
                    }
                )
            arrays[f"k{k}-rows"] = rows
            arrays[f"k{k}-targets"] = targets
            summaries[str(k)] = {"n": len(rows), "records": records}
    if model_digest(model) != complete["model_sha256"]:
        raise AssertionError("Trace altered model parameters")
    np.savez_compressed(out / "prefix-trace.npz", **arrays)
    write_json(
        out / "prefix-trace.json",
        {
            "passed": True,
            "arm": name,
            "panels": summaries,
            "scope": "First fixed256 graphB examples per hop; native tied vocabulary head; "
            "actual forward hooks; no learned probe or intermediate supervision. Decodability "
            "alone is not evidence that the state is causally used.",
            "utc": utc(),
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("preflight", "smoke", "train", "audit", "trace"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    torch.set_num_threads(config["runtime"]["cpu_threads"])
    torch.set_num_interop_threads(1)
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    smoke = args.stage == "smoke" or args.smoke
    out = args.base / ("smoke" if smoke else "runs") / args.arm
    try:
        if args.stage == "preflight":
            preflight(config, args.arm, args.base)
        elif args.stage == "audit":
            audit(config, args.arm, out, args.config)
        elif args.stage == "trace":
            trace(config, args.arm, out)
        else:
            train(config, args.arm, out, smoke, args.config, args.base)
    except BaseException as error:
        failure = {
            "state": "failed",
            "utc": utc(),
            "error": repr(error),
            "traceback": traceback.format_exc(),
        }
        path = args.base / "failures" / f"{args.arm}-{args.stage}{'-smoke' if smoke else ''}.json"
        write_json(path, failure)
        if out.exists():
            write_json(out / "failure.json", failure)
        raise


if __name__ == "__main__":
    main()
