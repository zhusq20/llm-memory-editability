#!/usr/bin/env python3
"""Run one audited stage of the paired cross-sample TTCD experiment."""

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
from llm_memory_editability.path_overlap import sha256  # noqa: E402
from llm_memory_editability.ttcd_continual import (  # noqa: E402
    audit_training,
    compare_replay,
    identity,
    train,
    validate_parent,
)
from llm_memory_editability.ttcd_continual_eval import (  # noqa: E402
    audit,
    build_panel,
    calibrate,
    evaluate,
)


def plain_summary(value):
    """W&B SummarySubDict exposes items, but is not a Python dict/iterable."""
    if hasattr(value, "items"):
        return {key: plain_summary(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain_summary(item) for item in value]
    return value


def cloud_audit(config, name, out):
    """Fresh online API reads of each completed stage, without leaking credentials."""
    import wandb

    api = wandb.Api(timeout=60)
    records = []
    for stage in ("train", "train-audit", "calibrate", "evaluate", "audit"):
        metadata = json.loads((out / f"wandb-{stage}.json").read_text())
        run = api.run(f"zhusq20/llm-memory-editability/{metadata['id']}")
        if (
            run.state != "finished"
            or run.config.get("arm") != name
            or run.config.get("batch") != config["batch"]
        ):
            raise ValueError(f"Cloud endpoint identity/state mismatch for {stage}")
        result = json.loads((out / f"result-{stage}.json").read_text())
        summary = plain_summary(run.summary)
        compare_replay(result, summary["result"])
        if stage == "train":
            if summary.get("training/step") != config["models"][name]["step"]:
                raise ValueError("Cloud training step differs")
            if summary.get("exposure/base_parameter_updates") != config["parent"]["steps"]:
                raise ValueError("Cloud additional update count differs")
        records.append(dict(stage=stage, id=metadata["id"], url=metadata["url"], state=run.state))
    result = dict(passed=True, status="cloud_endpoints_independently_verified", runs=records)
    write_json(out / "cloud-audit.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=(
            "preflight",
            "train",
            "train-audit",
            "calibrate",
            "evaluate",
            "audit",
            "cloud-audit",
        ),
    )
    parser.add_argument("--config", type=Path, default=ROOT / "configs/ttcd-cross-sample-v1.json")
    parser.add_argument("--model", choices=("single", "stream"), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", type=Path)
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
            raise RuntimeError("Actual CUDA computation failed")
        del probe
    run = None
    with (out / "execution.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (out / f"result-{args.stage}.json").exists():
            raise FileExistsError("Stage output already exists; no automatic replay or overwrite")
        write_json(
            out / f"environment-{args.stage}.json",
            dict(
                utc=utc(),
                hostname=platform.node(),
                python=sys.version,
                torch=torch.__version__,
                torch_cuda=torch.version.cuda,
                device=args.device,
                cpu_affinity=sorted(os.sched_getaffinity(0)),
            ),
        )
        try:
            if args.stage not in ("preflight", "cloud-audit") and args.tracking != "disabled":
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
                    dict(id=run.id, url=run.url, mode=args.tracking),
                )
                print(json.dumps({"wandb": run.url}), flush=True)

            def log(step, record):
                if run is None:
                    return
                if args.stage == "train":
                    values = {
                        f"training/{k}": record[k]
                        for k in (
                            "step",
                            "additional_step",
                            "loss",
                            "lr",
                            "gradient_norm",
                            "training_seconds",
                            "wall_seconds",
                        )
                    }
                    values.update({f"loss/{k}": v for k, v in record["losses"].items()})
                    values.update({f"exposure/{k}": v for k, v in record["counts"].items()})
                    values.update({f"compute/{k}": v for k, v in record["compute"].items()})
                    for key, value in record.get("validation", {}).items():
                        values[f"old_world/{key}"] = value["old_accuracy"]
                    for group, values_group in record.get("calibration", {}).items():
                        values.update(
                            {f"calibration/{group}/{k}": v for k, v in values_group.items()}
                        )
                    run.log(values, step=step)
                else:
                    # Evaluation raw matrices are persisted independently; the
                    # stage summary is authoritative for aggregate cloud checks.
                    run.log({"evaluation/completed": step}, step=step)

            if args.stage == "preflight":
                parent, provenance, world = validate_parent(config, args.model)
                panel = build_panel(config, world)
                panel_path = config["continual"].get("panel_path")
                if panel_path and panel != json.loads(Path(panel_path).read_text()):
                    raise ValueError("Prepared formal panel differs from frozen data")
                write_json(out / "preflight-panel.json", panel)
                groups = {
                    g: [
                        x["open"]["new_correct"]
                        for x in parent["readout"]["calibration_atoms"]
                        if x["group"] == g
                    ]
                    for g in ("id", "ood")
                }
                result = dict(
                    passed=True,
                    parent=provenance,
                    identity=identity(config),
                    panel_sha256=sha256(out / "preflight-panel.json"),
                    parent_open_accuracy={g: sum(v) / len(v) for g, v in groups.items()},
                )
            elif args.stage == "train":
                result = train(config, args.model, out, args.device, log)
            elif args.stage == "train-audit":
                result = audit_training(config, args.model, out, args.device)
            elif args.stage == "calibrate":
                result = calibrate(config, args.model, out, args.device)
            elif args.stage == "evaluate":
                result = evaluate(config, args.model, out, args.device, log)
            elif args.stage == "audit":
                result = audit(config, args.model, out, args.device)
            else:
                result = cloud_audit(config, args.model, out)
            write_json(out / f"result-{args.stage}.json", result)
            code = 3 if args.stage == "calibrate" and not result["passed"] else 0
            write_json(
                out / f"status-{args.stage}.json",
                dict(
                    utc=utc(),
                    stage=args.stage,
                    state="calibration_needs_attention" if code else "complete",
                ),
            )
            if run:
                run.summary["result"] = result
                run.finish(exit_code=code)
            print(
                json.dumps(
                    {
                        "stage": args.stage,
                        "state": "complete" if not code else "calibration_needs_attention",
                    }
                ),
                flush=True,
            )
            return code
        except Exception:
            write_json(
                out / f"failure-{args.stage}.json",
                dict(utc=utc(), traceback=traceback.format_exc()),
            )
            if run:
                run.finish(exit_code=1)
            raise


if __name__ == "__main__":
    raise SystemExit(main())
