"""Execute one frozen stage of matched one-step TTCD training and evaluation."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import platform
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGES = (
    "prepare",
    "preflight",
    "train",
    "train-audit",
    "train-cloud-audit",
    "eval-preflight",
    "calibrate",
    "evaluate",
    "audit",
    "cloud-audit",
    "aggregate",
)
TRACKED = ("train", "train-audit", "calibrate", "evaluate", "audit")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_runtime():
    manifest_path = ROOT / "source-manifest.json"
    if digest(manifest_path) != os.environ.get("LM_TTCD_SOURCE_MANIFEST_SHA256"):
        raise ValueError("Frozen container source manifest differs")
    manifest = json.loads(manifest_path.read_text())
    for relative, expected in manifest["files"].items():
        path = ROOT / relative
        if path.is_symlink() or not path.resolve().is_relative_to(ROOT.resolve()):
            raise ValueError("Frozen runtime file escaped its source")
        if digest(path) != expected:
            raise ValueError(f"Frozen runtime bytes changed: {relative}")
    return manifest


def cloud_audit(task, out, stages):
    import wandb

    api, records = wandb.Api(timeout=60), []
    for stage in stages:
        tracking = json.loads((out / f"wandb-{stage}.json").read_text())
        result_path = out / f"result-{stage}.json"
        expected = digest(result_path)
        run = api.run(f"zhusq20/llm-memory-editability/{tracking['id']}")
        if (
            run.state != "finished"
            or run.config.get("batch") != task["batch"]
            or run.config.get("architecture") != task["architecture_name"]
            or run.config.get("training_writer_steps") != 1
            or run.config.get("evaluation_writer_steps") != 1
            or run.summary.get("result_sha256") != expected
        ):
            raise ValueError(f"Cloud identity/state/canonical-result mismatch: {stage}")
        destination = out / "cloud-verification" / stage
        destination.mkdir(parents=True, exist_ok=True)
        downloaded = run.file(result_path.name).download(root=str(destination), replace=False)
        if digest(downloaded.name) != expected:
            raise ValueError("Cloud full result JSON differs")
        if stage == "train" and (
            run.summary.get("terminal_step") != 10000
            or run.summary.get("base_parameter_updates") != 10000
            or run.summary.get("fast_state_updates") != 1280000
        ):
            raise ValueError("Cloud one-step training exposure differs")
        records.append({"stage": stage, **tracking, "result_sha256": expected})
    return {"passed": True, "runs": records, "canonical_json_exact": True}


def dispatch(stage, task, settings, out, device, log=None, comparison_root=None):
    from llm_memory_editability import ttcd_one_step_evaluation as evaluation
    from llm_memory_editability import ttcd_one_step_training as training

    if stage == "aggregate":
        if comparison_root is None:
            raise ValueError("Three-architecture comparison root required")
        return evaluation.aggregate(
            {arm: comparison_root / arm / "evaluation" for arm in settings["architectures"]},
            out,
            control_outputs={
                arm: ROOT / "results/ttcd-merge-hops-v1" / arm / "evaluation"
                for arm in settings["architectures"]
            },
        )
    if stage == "prepare":
        return evaluation.prepare(task, settings, out)
    if stage == "preflight":
        return training.preflight(task, out, device)
    if stage in ("train", "train-audit"):
        engine = training.build_engine(task)
        if stage == "train":
            return engine.train(engine.config, "stream", out, device, log)
        return engine.audit_training(engine.config, "stream", out, device)
    if stage == "train-cloud-audit":
        return cloud_audit(task, out, ("train", "train-audit"))
    if stage == "cloud-audit":
        return cloud_audit(task, out, ("calibrate", "evaluate", "audit"))
    if stage == "eval-preflight":
        return evaluation.preflight(task, settings, out, device)
    if stage == "evaluate":
        return evaluation.evaluate(task, settings, out, device, log)
    if stage in ("calibrate", "audit"):
        return getattr(evaluation, stage)(task, settings, out, device)
    raise ValueError("Unsupported one-step stage")


def main():
    manifest = verify_runtime()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--task", type=Path)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument(
        "--architecture", choices=("transformer", "loop", "discoloop"), required=True
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--comparison-root", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tracking", choices=("online", "disabled"), default="online")
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT / "src"))
    import torch

    from llm_memory_editability.grok_depth import utc, write_json

    settings = json.loads(args.settings.read_text())
    task = json.loads(args.task.read_text()) if args.task else None
    if args.stage != "aggregate" and (
        task is None or task["architecture_name"] != args.architecture
    ):
        raise ValueError("This stage requires its exact architecture task")
    torch.set_num_threads(settings["runtime"]["cpu_threads"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    with (out / "execution.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result_path, failure = out / f"result-{args.stage}.json", out / f"failure-{args.stage}.json"
        if result_path.exists() or failure.exists():
            raise FileExistsError("Preserve original stage results/failures; no automatic retry")
        run, stats, training_stats = None, None, None
        try:
            write_json(
                out / f"environment-{args.stage}.json",
                {
                    "utc": utc(),
                    "python": sys.version,
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                    "hostname": platform.node(),
                    "device": args.device,
                    "cpu_affinity": sorted(os.sched_getaffinity(0)),
                    "training_writer_steps": 1,
                    "evaluation_writer_steps": 1,
                    "source_manifest_sha256": digest(ROOT / "source-manifest.json"),
                    "source_architecture": manifest["architecture"],
                    "tf32": False,
                    "actual_gpu": os.environ.get("LM_ACTUAL_GPU_INDEX"),
                    "gpu_sharing": os.environ.get("LM_GPU_SHARING"),
                },
            )
            with contextlib.ExitStack() as stack:
                if args.stage == "train":
                    from llm_memory_editability.ttcd_train_speedup import enabled

                    training_stats = stack.enter_context(enabled())
                if args.stage in ("evaluate", "audit"):
                    from llm_memory_editability.ttcd_eval_speedup import install

                    undo, stats = install()
                    stack.callback(undo)
                if args.stage in TRACKED and args.tracking == "online":
                    import wandb

                    run = wandb.init(
                        entity="zhusq20",
                        project="llm-memory-editability",
                        name=f"{settings['batch']}/{args.architecture}/{args.stage}",
                        group=settings["batch"],
                        job_type=args.stage,
                        config={
                            "batch": settings["batch"],
                            "architecture": args.architecture,
                            "training_writer_steps": 1,
                            "evaluation_writer_steps": 1,
                            "training_method": "stream",
                            "task": task,
                            "settings": settings,
                            "source_manifest_sha256": digest(ROOT / "source-manifest.json"),
                        },
                        mode="online",
                        dir=str(out),
                        save_code=False,
                        settings=wandb.Settings(disable_git=True),
                    )
                    write_json(out / f"wandb-{args.stage}.json", {"id": run.id, "url": run.url})
                    print(json.dumps({"wandb": run.url}), flush=True)

                def log(step, record):
                    if not run:
                        return
                    values = {
                        f"training/{k}": v for k, v in record.items() if isinstance(v, (int, float))
                    }
                    for name in ("losses", "counts", "compute"):
                        values.update({f"{name}/{k}": v for k, v in record.get(name, {}).items()})
                    if args.stage != "train":
                        values["evaluation/states"] = step
                    run.log(values, step=step)

                result = dispatch(
                    args.stage, task, settings, out, args.device, log, args.comparison_root
                )
            if stats is not None:
                result["evaluation_runtime"] = stats
            if training_stats is not None:
                result["training_runtime"] = training_stats
            write_json(result_path, result)
            code = 3 if result.get("passed") is False else 0
            write_json(
                out / f"status-{args.stage}.json",
                {
                    "utc": utc(),
                    "state": "failed" if code else "complete",
                    "stage": args.stage,
                },
            )
            if run:
                run.summary["result_sha256"] = digest(result_path)
                run.summary["passed"] = result.get("passed", True)
                if args.stage == "train":
                    run.summary["terminal_step"] = result["step"]
                    run.summary["base_parameter_updates"] = result["counts"][
                        "base_parameter_updates"
                    ]
                    run.summary["fast_state_updates"] = result["counts"]["fast_state_updates"]
                run.save(str(result_path), base_path=str(out), policy="now")
                run.finish(exit_code=code)
            print(
                json.dumps(
                    {"stage": args.stage, "architecture": args.architecture, "returncode": code}
                ),
                flush=True,
            )
            return code
        except BaseException:
            write_json(failure, {"utc": utc(), "traceback": traceback.format_exc()})
            if run:
                run.finish(exit_code=1)
            raise


if __name__ == "__main__":
    raise SystemExit(main())
