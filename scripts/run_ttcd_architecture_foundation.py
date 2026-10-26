#!/usr/bin/env python3
"""Rebuild and independently reload the two authorized native foundations."""

from __future__ import annotations

import argparse
import fcntl
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
from run_discoloop_restore import CurveTracker, native_runner  # noqa: E402

from llm_memory_editability.discoloop_restore import (  # noqa: E402
    PANELS_SHA256,
    WORLD_SHA256,
)
from llm_memory_editability.grok_depth import utc, write_json  # noqa: E402
from llm_memory_editability.path_overlap import sha256  # noqa: E402

BATCH = "ttcd-architecture-comparison-v1"
INITIAL = {
    "transformer": (
        "11b1c7254c2c277676239094ab835bb3dca302a9d7197cec4e98380308d7fa3f",
        97782528,
        8,
        1,
    ),
    "loop": ("ba0af28431d36db482f2676c1e111a725cf51041c3dca9d006dd10d7a7c18518", 69431040, 4, 2),
}


def read_json(path):
    return json.loads(Path(path).read_text())


def validate(config):
    name = config["architecture_name"]
    if config["batch"] != BATCH or name not in INITIAL or set(config["arms"]) != {name}:
        raise ValueError("Only the authorized independent D8 and Loop4x2 may be rebuilt")
    digest, parameters, layers, repeats = INITIAL[name]
    expected = {
        "unique_layers": layers,
        "repeats": repeats,
        "phi": 7.2,
        "weight_decay": 0.1,
        "initialization": 791101,
        "stream_seed": 792101,
        "lr": 1e-4,
        "warmup": 2000,
        "batch_size": 512,
        "steps": 300000,
    }
    if any(config["spec"].get(key) != value for key, value in expected.items()):
        raise ValueError("The successful original native recipe must be preserved")
    arm = config["arms"][name]
    if (
        arm
        != {
            "kind": "none",
            "coefficient": 0.0,
            "scope": "supervised",
            "first_round_kv": False,
            "discoloop": False,
            "discoloop_alpha": 1.0,
            "discoloop_temperature": 1.0,
            "discoloop_eps": 1e-6,
        }
        or config["dropout_seed"] != 20261012
    ):
        raise ValueError("Native CE only, original dropout and no channel/KV")
    if config["initial_model_sha256"] != digest or config["parameters"] != parameters:
        raise ValueError("The historical independent initialization must be bound")
    if config["evaluation_batch_size"] != 256 or config["smoke_steps"] != 32:
        raise ValueError("Preserve original native scoring and the 32-step development check")


def preflight(config, runner):
    validate(config)
    metadata = read_json(Path(config["input_root"]) / "data/complete.json")
    if metadata["world_sha256"] != WORLD_SHA256 or metadata["panels_sha256"] != PANELS_SHA256:
        raise ValueError("The foundation must use exactly the restored Wang graph")
    model, observed, provenance = runner.initialize(
        config, torch.device("cuda:0"), config["arms"][config["architecture_name"]]
    )
    parameters = sum(p.numel() for p in model.parameters())
    if observed != metadata or parameters != config["parameters"]:
        raise ValueError("Native shape or metadata mismatch")
    if provenance["initial_model_sha256"] != config["initial_model_sha256"]:
        raise ValueError("Original initialization differs")
    if not provenance["constructor_matches_source_weights"]:
        raise ValueError("Original seed must independently reconstruct the initial tensors")
    layer_hashes = [runner.state_digest(layer.state_dict()) for layer in model.transformer.h]
    if len(set(layer_hashes)) != len(layer_hashes):
        raise ValueError("Physical layers must be independently initialized")
    probe = torch.randn(64, 64, device="cuda:0", requires_grad=True)
    probe.square().matmul(probe).mean().backward()
    torch.cuda.synchronize()
    if not torch.isfinite(probe.grad).all():
        raise FloatingPointError("CUDA forward/backward/synchronize failed")
    return {
        "passed": True,
        "engineering_only": True,
        "scientific_training_updates": 0,
        "parameters": parameters,
        "physical_layer_sha256": layer_hashes,
        "independent_physical_initialization": True,
        "metadata": metadata,
        "provenance": provenance,
        "cuda_forward_backward_synchronized": True,
    }


def numerical_receipt(config, out, science, *, engineering=False):
    complete, audit = read_json(science / "complete.json"), read_json(science / "audit.json")
    step = config["smoke_steps"] if engineering else config["spec"]["steps"]
    checkpoint = science / f"checkpoint-{step:07d}.pt"
    if (
        complete["state"] != "complete"
        or complete["step"] != step
        or complete["engineering_only"] is not engineering
        or not audit["passed"]
        or audit["step"] != step
        or sha256(checkpoint) != complete["final_checkpoint_sha256"]
        or complete["initial_model_sha256"] != config["initial_model_sha256"]
        or complete["parameters"] != config["parameters"]
    ):
        raise ValueError("Native endpoint and independent numerical audit must agree")
    receipt = {
        "passed": True,
        "gate_semantics": "identity_and_independent_numerical_replay_only",
        "capability_thresholds": None,
        "checkpoint_sha256": sha256(checkpoint),
        "model_sha256": complete["model_sha256"],
        "step": step,
        "initial_model_sha256": complete["initial_model_sha256"],
        "parameters": complete["parameters"],
        "native_metrics": complete["endpoint"]["metrics"],
        "audit_sha256": sha256(science / "audit.json"),
        "engineering_only": engineering,
        "utc": utc(),
    }
    write_json(out / ("smoke-identity.json" if engineering else "identity.json"), receipt)
    return receipt


def cloud_audit(config, out):
    import wandb

    api, findings = wandb.Api(timeout=60), []
    for stage in ("train", "audit"):
        tracking = read_json(out / f"wandb-{stage}.json")
        run = api.run(f"zhusq20/llm-memory-editability/{tracking['id']}")
        path = out / f"result-{stage}.json"
        expected = sha256(path)
        if (
            run.state != "finished"
            or run.config.get("batch") != BATCH
            or run.config.get("architecture_name") != config["architecture_name"]
            or run.config.get("stage") != stage
            or run.summary.get("result_sha256") != expected
        ):
            raise ValueError("Foundation cloud state/identity/digest differs")
        download = out / "cloud-verification" / stage
        download.mkdir(parents=True, exist_ok=True)
        saved = run.file(path.name).download(root=str(download), replace=True)
        if sha256(saved.name) != expected:
            raise ValueError("Foundation canonical cloud JSON differs")
        if stage == "train" and run.summary.get("terminal_step") != 300000:
            raise ValueError("Foundation cloud terminal update differs")
        findings.append({**tracking, "state": run.state, "stage": stage, "sha256": expected})
    return {"passed": True, "full_json_exact": True, "runs": findings}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=("preflight", "smoke", "smoke-audit", "train", "audit", "cloud-audit")
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tracking", choices=("online", "disabled"), default="online")
    args = parser.parse_args()
    config = read_json(args.config)
    validate(config)
    torch.set_num_threads(config["runtime"]["cpu_threads"])
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    torch.cuda.set_device(0)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    run = None
    with (out / "execution.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = out / f"result-{args.stage}.json"
        if path.exists() or (out / f"failure-{args.stage}.json").exists():
            raise FileExistsError("Preserve claimed/completed/failed stages; no automatic retry")
        try:
            if args.stage in ("train", "audit") and args.tracking == "online":
                import wandb

                run = wandb.init(
                    entity="zhusq20",
                    project="llm-memory-editability",
                    group=BATCH,
                    name=f"{BATCH}/{config['architecture_name']}/foundation/{args.stage}",
                    job_type=f"foundation-{args.stage}",
                    config={**config, "stage": args.stage},
                    dir=str(out),
                    save_code=False,
                    settings=wandb.Settings(disable_git=True),
                )
                write_json(out / f"wandb-{args.stage}.json", {"id": run.id, "url": run.url})
                print(json.dumps({"wandb": run.url}), flush=True)
            native = native_runner()
            science = out / config["architecture_name"]
            smoke = out / "smoke" / config["architecture_name"]
            if args.stage == "preflight":
                result = preflight(config, native)
            elif args.stage in ("train", "smoke"):
                destination = science if args.stage == "train" else smoke
                destination.parent.mkdir(parents=True, exist_ok=True)
                with CurveTracker(run, destination):
                    native.run(
                        config,
                        config["architecture_name"],
                        destination,
                        args.stage == "smoke",
                        args.config,
                    )
                result = read_json(destination / "complete.json")
            elif args.stage in ("audit", "smoke-audit"):
                destination = science if args.stage == "audit" else smoke
                native.audit(config, config["architecture_name"], destination, args.config)
                result = numerical_receipt(
                    config, out, destination, engineering=args.stage == "smoke-audit"
                )
            else:
                result = cloud_audit(config, out)
            write_json(path, result)
            if run:
                run.summary["result_sha256"] = sha256(path)
                run.summary["terminal_step"] = result["step"]
                run.save(str(path), base_path=str(out), policy="now")
                run.finish(exit_code=0)
            print(
                json.dumps(
                    {
                        "stage": args.stage,
                        "passed": result.get("passed", True),
                        "step": result.get("step"),
                    }
                ),
                flush=True,
            )
        except BaseException as error:
            write_json(
                out / f"failure-{args.stage}.json",
                {"error": repr(error), "traceback": traceback.format_exc(), "utc": utc()},
            )
            if run:
                run.finish(exit_code=1)
            raise


if __name__ == "__main__":
    main()
