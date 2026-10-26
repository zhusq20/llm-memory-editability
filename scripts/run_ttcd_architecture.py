#!/usr/bin/env python3
"""Run one frozen ordinary/Loop TTCD stage using the existing audited engine."""

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

from llm_memory_editability import ttcd_architecture as architecture  # noqa: E402
from llm_memory_editability import ttcd_continual_eval as evaluation  # noqa: E402
from llm_memory_editability import ttcd_double_new_eval as double_new  # noqa: E402
from llm_memory_editability.grok_depth import utc, write_json  # noqa: E402
from llm_memory_editability.path_overlap import World, sha256  # noqa: E402

STAGES = (
    "prepare",
    "preflight",
    "train",
    "train-audit",
    "gate",
    "calibrate",
    "evaluate",
    "audit",
    "cloud-audit",
    "double-prepare",
    "double-evaluate",
    "double-audit",
    "double-cloud-audit",
)


def cloud_audit(config, phase, out, *, double=False):
    import wandb

    stages = (
        ("double-evaluate", "double-audit")
        if double
        else (
            ("train", "train-audit", "gate")
            if phase == "common"
            else ("train", "train-audit", "calibrate", "evaluate", "audit")
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
            or run.config.get("batch") != config["batch"]
            or run.config.get("arm") != phase
            or run.config.get("architecture") != config["architecture"]
            or run.summary.get("result_sha256") != expected
        ):
            raise ValueError(f"Cloud architecture/phase/state/result identity differs: {stage}")
        destination = out / "cloud-verification" / stage
        destination.mkdir(parents=True, exist_ok=True)
        saved = run.file(path.name).download(root=str(destination), replace=True)
        if sha256(saved.name) != expected:
            raise ValueError("Cloud canonical result JSON differs")
        if stage == "train":
            step = config["common"]["steps"] if phase == "common" else config["training"]["steps"]
            if (
                run.summary.get("terminal_step") != step
                or run.summary.get("base_parameter_updates") != step
            ):
                raise ValueError("Cloud terminal step/exposure differs")
        records.append({"stage": stage, **tracking, "result_sha256": expected, "state": run.state})
    result = {"passed": True, "runs": records, "full_json_exact": True}
    write_json(out / ("double-cloud-audit.json" if double else "cloud-audit.json"), result)
    return result


def dispatch(engine, config, phase, stage, out, device, log=None, *, settings=None, panel=None):
    if phase not in ("common", "stream"):
        raise ValueError("Only common plus the continual stream is authorized")
    if stage == "prepare":
        return {"passed": True, "metadata": engine.prepare_data(config)}
    if stage == "preflight":
        result = engine.preflight(config, out, device)
        reader, _, _, _ = engine.foundation_inputs(config, device)
        result["architecture_accounting"] = architecture.architecture_accounting(reader)
        return result
    if stage == "train":
        return engine.train(config, phase, out, device, log)
    if stage == "train-audit":
        return engine.audit_training(config, phase, out, device)
    if stage == "gate":
        if phase != "common":
            raise ValueError("Only the common stage produces its numerical/advisory gate")
        return engine.common_gate(config, out, device)
    if stage == "cloud-audit":
        return cloud_audit(config, phase, out)
    if phase != "stream":
        raise ValueError("Write evaluation is reserved for the continual endpoint")
    if stage == "calibrate":
        return architecture.calibrate(engine, config, phase, out, device)
    if stage in ("evaluate", "audit"):
        kwargs = {
            "reader_loader": engine.load_reader,
            "identity_factory": engine.evaluation_identity,
        }
        if stage == "evaluate":
            kwargs["log"] = log
        return getattr(evaluation, stage)(config, phase, out, device, **kwargs)
    if stage.startswith("double-"):
        if settings is None or panel is None:
            raise ValueError("Double-new evaluation requires frozen settings and panel paths")
        if stage == "double-cloud-audit":
            return cloud_audit(config, phase, out, double=True)
        if stage == "double-prepare":
            if Path(panel).exists():
                raise FileExistsError("Preserve the existing graph-only double-new panel")
            prepared = double_new.build_panel(config, settings, World(config["data_dir"]))
            Path(panel).parent.mkdir(parents=True, exist_ok=True)
            write_json(Path(panel), prepared)
            return {
                "passed": True,
                "panel_sha256": sha256(panel),
                "model_queries": 0,
                "scientific_training_updates": 0,
            }
        kwargs = {
            "reader_loader": engine.load_reader,
            "identity_factory": lambda cfg, opts, name, selected: (
                architecture.supplementary_identity(engine, cfg, opts, name, selected)
            ),
        }
        if stage == "double-evaluate":
            kwargs["log"] = log
        return getattr(double_new, stage.removeprefix("double-"))(
            config, settings, phase, out, panel, device, **kwargs
        )
    raise ValueError(f"Unknown architecture stage: {stage}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", choices=("common", "stream"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tracking", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--settings", type=Path)
    parser.add_argument("--panel", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    engine = architecture.build_engine(config)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(config["runtime"]["cpu_threads"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if str(args.device).startswith("cuda"):
        probe = torch.randn(32, 32, device=args.device, requires_grad=True)
        probe.square().matmul(probe).mean().backward()
        torch.cuda.synchronize(args.device)
        if not torch.isfinite(probe.grad).all():
            raise FloatingPointError("Actual CUDA forward/backward failed")
        del probe
    with (out / "execution.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = out / f"result-{args.stage}.json"
        if path.exists() or (out / f"failure-{args.stage}.json").exists():
            raise FileExistsError("Preserve existing stage results/failures; no automatic retry")
        write_json(
            out / f"environment-{args.stage}.json",
            {
                "utc": utc(),
                "python": sys.version,
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "hostname": platform.node(),
                "cpu_affinity": sorted(os.sched_getaffinity(0)),
                "device": args.device,
                "architecture": config["architecture"],
                "tf32": False,
                "forward": "Existing BF16 native objective; FP32 fast matrices and evaluation",
            },
        )
        run = None
        try:
            if args.tracking != "disabled" and args.stage not in (
                "prepare",
                "preflight",
                "cloud-audit",
                "double-prepare",
                "double-cloud-audit",
            ):
                import wandb

                defaults = json.loads(
                    (ROOT / "configs/experiment-tracking-defaults.json").read_text()
                )
                run = wandb.init(
                    entity=defaults["entity"],
                    project=defaults["project"],
                    name=f"{config['batch']}/{config['architecture']['name']}/{args.model}/{args.stage}",
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
                if run:
                    values = {
                        f"training/{k}": v for k, v in record.items() if isinstance(v, (int, float))
                    }
                    for key in ("losses", "counts", "compute"):
                        values.update({f"{key}/{k}": v for k, v in record.get(key, {}).items()})
                    if args.stage != "train":
                        values["evaluation/completed"] = step
                    run.log(values, step=step)

            result = dispatch(
                engine,
                config,
                args.model,
                args.stage,
                out,
                args.device,
                log,
                settings=json.loads(args.settings.read_text()) if args.settings else None,
                panel=args.panel,
            )
            write_json(path, result)
            code = 3 if result.get("passed") is False else 0
            write_json(
                out / f"status-{args.stage}.json",
                {"utc": utc(), "stage": args.stage, "state": "failed" if code else "complete"},
            )
            if run:
                run.summary["result_sha256"] = sha256(path)
                run.summary["passed"] = result.get("passed", True)
                if args.stage == "train":
                    run.summary["terminal_step"] = result["step"]
                    run.summary["base_parameter_updates"] = result["counts"][
                        "base_parameter_updates"
                    ]
                run.save(str(path), base_path=str(out), policy="now")
                run.finish(exit_code=code)
            print(
                json.dumps({"stage": args.stage, "phase": args.model, "returncode": code}),
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
