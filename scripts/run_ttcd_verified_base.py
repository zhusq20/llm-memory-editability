#!/usr/bin/env python3
"""Execute one frozen stage of successful-base TTCD adaptation and replay."""

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

from llm_memory_editability import ttcd_continual_eval as evaluation  # noqa: E402
from llm_memory_editability import ttcd_verified_base as experiment  # noqa: E402
from llm_memory_editability.grok_depth import utc, write_json  # noqa: E402
from llm_memory_editability.path_overlap import sha256  # noqa: E402


def cloud_audit(config, phase, out):
    """Independently verify finished cloud state and the exact persisted JSON."""
    import wandb

    stages = (
        ("train", "train-audit", "gate")
        if phase == "common"
        else (
            "train",
            "train-audit",
            "calibrate",
            "evaluate",
            "audit",
        )
    )
    api, records = wandb.Api(timeout=60), []
    for stage in stages:
        tracking = json.loads((out / f"wandb-{stage}.json").read_text())
        run = api.run(f"zhusq20/llm-memory-editability/{tracking['id']}")
        path = out / f"result-{stage}.json"
        expected = sha256(path)
        if (
            run.state != "finished"
            or run.config.get("arm") != phase
            or run.config.get("batch") != config["batch"]
        ):
            raise ValueError(f"Cloud endpoint identity/state differs for {stage}")
        if run.summary.get("result_sha256") != expected:
            raise ValueError(f"Cloud result digest differs for {stage}")
        download = out / "cloud-verification" / stage
        download.mkdir(parents=True, exist_ok=True)
        saved = run.file(path.name).download(root=str(download), replace=True)
        if sha256(saved.name) != expected:
            raise ValueError(f"Cloud full JSON differs for {stage}")
        if stage == "train":
            steps = config["common"]["steps"] if phase == "common" else config["training"]["steps"]
            if (
                run.summary.get("terminal_step") != steps
                or run.summary.get("base_parameter_updates") != steps
            ):
                raise ValueError("Cloud terminal step/exposure differs")
        records.append(
            {
                "stage": stage,
                "id": tracking["id"],
                "url": tracking["url"],
                "state": run.state,
                "result_sha256": expected,
            }
        )
    result = {"passed": True, "runs": records, "full_json_exact": True}
    write_json(out / "cloud-audit.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=(
            "prepare",
            "preflight",
            "train",
            "train-audit",
            "gate",
            "calibrate",
            "evaluate",
            "audit",
            "cloud-audit",
        ),
    )
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/ttcd-successful-base-v1.json"
    )
    parser.add_argument("--model", choices=("common", "single", "stream"), required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tracking", choices=("online", "offline", "disabled"), default="online")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    out = args.out or ROOT / "results" / config["batch"] / args.model
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(config["runtime"]["cpu_threads"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if str(args.device).startswith("cuda"):
        probe = torch.randn(32, 32, device=args.device, requires_grad=True)
        probe.square().matmul(probe).mean().backward()
        torch.cuda.synchronize(args.device)
        if not torch.isfinite(probe.grad).all():
            raise RuntimeError("Actual CUDA forward/backward failed")
        del probe
    run = None
    with (out / "execution.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result_path = out / f"result-{args.stage}.json"
        if result_path.exists():
            raise FileExistsError("Stage result already exists; no automatic overwrite/retry")
        write_json(
            out / f"environment-{args.stage}.json",
            {
                "utc": utc(),
                "hostname": platform.node(),
                "python": sys.version,
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "device": args.device,
                "cpu_affinity": sorted(os.sched_getaffinity(0)),
                "tf32": False,
                "adaptation_forward": "bf16 deterministic eval; fast matrices float32",
            },
        )
        try:
            if (
                args.stage not in ("prepare", "preflight", "cloud-audit")
                and args.tracking != "disabled"
            ):
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
                    config={**config, "arm": args.model, "stage": args.stage},
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
                if run is None:
                    return
                if args.stage == "train":
                    values = {
                        f"training/{key}": record[key]
                        for key in (
                            "step",
                            "loss",
                            "lr",
                            "gradient_norm",
                            "training_seconds",
                            "wall_seconds",
                        )
                    }
                    for source, namespace in (
                        ("losses", "loss"),
                        ("counts", "exposure"),
                        ("compute", "compute"),
                    ):
                        values.update(
                            {f"{namespace}/{key}": value for key, value in record[source].items()}
                        )
                    values.update(
                        {
                            f"old_world/{key}": value["old_accuracy"]
                            for key, value in record.get("native_metrics", {}).items()
                        }
                    )
                else:
                    values = {"evaluation/completed": step}
                run.log(values, step=step)

            if args.stage == "prepare":
                result = {"passed": True, "metadata": experiment.prepare_data(config)}
            elif args.stage == "preflight":
                result = experiment.preflight(config, out, args.device)
            elif args.stage == "train":
                result = experiment.train(config, args.model, out, args.device, log)
            elif args.stage == "train-audit":
                result = experiment.audit_training(config, args.model, out, args.device)
            elif args.stage == "gate":
                if args.model != "common":
                    raise ValueError("Only the common reader stage has the branch gate")
                result = experiment.common_gate(config, out, args.device)
            elif args.stage in ("calibrate", "evaluate", "audit"):
                if args.model == "common":
                    raise ValueError("Formal write evaluation requires a paired branch")
                callback = getattr(evaluation, args.stage)
                kwargs = {
                    "reader_loader": experiment.load_reader,
                    "identity_factory": experiment.evaluation_identity,
                }
                if args.stage == "evaluate":
                    kwargs["log"] = log
                result = callback(config, args.model, out, args.device, **kwargs)
            else:
                result = cloud_audit(config, args.model, out)
            write_json(result_path, result)
            code = 3 if args.stage in ("gate", "calibrate") and not result["passed"] else 0
            write_json(
                out / f"status-{args.stage}.json",
                {"utc": utc(), "stage": args.stage, "state": "gate_failed" if code else "complete"},
            )
            if run:
                run.summary["result_sha256"] = sha256(result_path)
                run.summary["passed"] = result.get("passed", True)
                if args.stage == "train":
                    run.summary["terminal_step"] = result["step"]
                    run.summary["base_parameter_updates"] = result["counts"][
                        "base_parameter_updates"
                    ]
                run.save(str(result_path), base_path=str(out), policy="now")
                run.finish(exit_code=code)
            print(
                json.dumps(
                    {
                        "stage": args.stage,
                        "phase": args.model,
                        "state": "gate_failed" if code else "complete",
                    }
                ),
                flush=True,
            )
            return code
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
