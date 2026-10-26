#!/usr/bin/env python3
"""Calibrate, edit, or independently audit the TTCD second-hop experiment."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import platform
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from llm_memory_editability.grok_depth import utc, write_json  # noqa: E402
from llm_memory_editability.ttcd_experiment import (  # noqa: E402
    audit,
    calibrate,
    evaluate,
    load_inputs,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=("preflight", "train", "train-audit", "calibrate", "evaluate", "audit")
    )
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/ttcd-second-hop-scratch-v1.json"
    )
    parser.add_argument("--model", choices=("discoloop", "loop"), default="discoloop")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--tracking", choices=("online", "offline", "disabled"), default="online")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    out = args.out or ROOT / "results" / config["batch"] / args.model
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(config["runtime"]["cpu_threads"])
    torch.manual_seed(config["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if str(args.device).startswith("cuda"):
        # Actual matmul, backward, synchronization on the selected device.
        probe = torch.randn(32, 32, device=args.device, requires_grad=True)
        probe.square().matmul(probe).mean().backward()
        torch.cuda.synchronize(args.device)
        if not torch.isfinite(probe.grad).all():
            raise RuntimeError("CUDA compute check failed")
        del probe
    with (out / "execution.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (out / "summary.json").exists() and args.stage in ("calibrate", "evaluate"):
            raise RuntimeError("Existing experiment output: do not repeat or overwrite the batch")
        write_json(
            out / f"environment-{args.stage}.json",
            {
                "utc": utc(),
                "hostname": platform.node(),
                "python": sys.version,
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "device": args.device,
                "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "gpu_name": torch.cuda.get_device_name(args.device)
                if str(args.device).startswith("cuda")
                else None,
                "cpu_affinity": sorted(os.sched_getaffinity(0)),
            },
        )
        run = None
        try:
            if args.stage == "preflight":
                if "scratch" in config:
                    from llm_memory_editability.ttcd_scratch import validate_scratch_inputs

                    world = validate_scratch_inputs(config)
                    result = {
                        "passed": True,
                        "atoms": len(world.atoms),
                        "repeats": config["model"]["repeats"],
                        "checkpoint_required": False,
                    }
                else:
                    reader, world = load_inputs(config, args.model, args.device)
                    result = {
                        "passed": True,
                        "atoms": len(world.atoms),
                        "repeats": reader.backbone.repeats,
                    }
                write_json(out / "preflight.json", result)
            else:
                if args.tracking != "disabled":
                    import wandb

                    defaults = json.loads(
                        (ROOT / "configs/experiment-tracking-defaults.json").read_text()
                    )
                    run = wandb.init(
                        entity=defaults["entity"],
                        project=defaults["project"],
                        name=f"{config['batch']}/{args.model}/{args.stage}",
                        group=config["batch"],
                        job_type=args.stage,
                        config={**config, "model_name": args.model, "stage": args.stage},
                        mode=args.tracking,
                        dir=str(out),
                        save_code=False,
                        settings=wandb.Settings(disable_git=True),
                    )
                    write_json(
                        out / f"wandb-{args.stage}.json",
                        {"id": run.id, "url": run.url, "mode": args.tracking},
                    )
                    print(json.dumps({"wandb": run.url}), flush=True)

                def log(step, record):
                    if run:
                        if args.stage == "train":
                            values = {
                                "training/step": step,
                                "training/loss": record["loss"],
                                "training/gradient_norm": record["gradient_norm"],
                                "training/lr": record["lr"],
                                "training/seconds": record["training_seconds"],
                                "wall_seconds": record["wall_seconds"],
                            }
                            values.update({f"loss/{k}": v for k, v in record["losses"].items()})
                            values.update({f"exposure/{k}": v for k, v in record["counts"].items()})
                            values.update({f"compute/{k}": v for k, v in record["compute"].items()})
                            for key, metrics in record.get("validation", {}).items():
                                values[f"old_world/{key}"] = metrics["old_accuracy"]
                            for group, metrics in record.get("calibration", {}).items():
                                values.update(
                                    {f"calibration/{group}/{k}": v for k, v in metrics.items()}
                                )
                            run.log(values, step=step)
                            return
                        metrics = record["after"]["metrics"]
                        prefix = f"{record['method']}/{record['case']['group']}"
                        run.log(
                            {
                                "completed_case_methods": step,
                                f"{prefix}/E_new": metrics["E"]["new_accuracy"],
                                f"{prefix}/D_new": metrics["D"]["new_accuracy"],
                                f"{prefix}/U_chains_old": metrics["U_chains"]["old_accuracy"],
                                f"{prefix}/same_answer_old": metrics["same_answer_atoms"][
                                    "old_accuracy"
                                ],
                                f"{prefix}/wall_seconds": record["wall_seconds"],
                                f"{prefix}/fast_updates": record["exposure"]["updates"],
                                f"{prefix}/writer_tokens": record["exposure"][
                                    "writer_input_tokens"
                                ],
                            },
                            step=step,
                        )

                if args.stage == "train":
                    if "branch" in config:
                        from llm_memory_editability.ttcd_branch import train_branch

                        result = train_branch(config, args.model, out, args.device, log)
                    else:
                        from llm_memory_editability.ttcd_scratch import train_scratch

                        result = train_scratch(config, args.model, out, args.device, log)
                elif args.stage == "train-audit":
                    from llm_memory_editability.ttcd_scratch import audit_scratch_training

                    result = audit_scratch_training(config, args.model, out, args.device)
                elif args.stage == "calibrate":
                    result = calibrate(config, args.model, out, args.device)
                elif args.stage == "evaluate":
                    result = evaluate(config, args.model, out, args.device, log)
                else:
                    result = audit(config, args.model, out, args.device)
                if run:
                    run.summary["result"] = (
                        result
                        if args.stage != "calibrate"
                        else {"passed": result["passed"], "groups": result["groups"]}
                    )
            write_json(
                out / f"status-{args.stage}.json",
                {"utc": utc(), "stage": args.stage, "state": result.get("status", "complete")},
            )
            if args.stage == "calibrate" and not result["passed"]:
                print(
                    json.dumps({"state": "reading_needs_attention", "groups": result["groups"]}),
                    flush=True,
                )
                if run:
                    run.finish(exit_code=3)
                return 3
            if run:
                run.finish()
            print(
                json.dumps({"stage": args.stage, "state": "complete", "out": str(out)}), flush=True
            )
            return 0
        except Exception:
            write_json(
                out / f"failure-{args.stage}.json",
                {"utc": utc(), "traceback": traceback.format_exc()},
            )
            if run:
                run.finish(exit_code=1)
            raise


if __name__ == "__main__":
    raise SystemExit(main())
