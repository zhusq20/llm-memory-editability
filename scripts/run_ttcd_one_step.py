#!/usr/bin/env python3
"""Run one evaluation-only stage on an unchanged audited four-step parent."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from llm_memory_editability import ttcd_one_step as experiment  # noqa: E402
from llm_memory_editability.grok_depth import utc, write_json  # noqa: E402
from llm_memory_editability.path_overlap import sha256  # noqa: E402

STAGES = ("preflight", "calibrate", "evaluate", "audit", "cloud-audit", "aggregate")


def verify_runtime():
    manifest_path = ROOT / "source-manifest.json"
    expected = os.environ.get("LM_TTCD_SOURCE_MANIFEST_SHA256")
    if expected is None or sha256(manifest_path) != expected:
        raise ValueError("Frozen container source manifest identity differs")
    manifest = json.loads(manifest_path.read_text())
    for relative, wanted in manifest["files"].items():
        path = ROOT / relative
        if path.is_symlink() or not path.resolve().is_relative_to(ROOT.resolve()):
            raise ValueError("Frozen runtime path escaped its source")
        if sha256(path) != wanted:
            raise ValueError(f"Frozen runtime bytes changed: {relative}")


def cloud_audit(settings, architecture, out):
    import wandb

    api, records = wandb.Api(timeout=60), []
    for stage in ("calibrate", "evaluate", "audit"):
        metadata = json.loads((out / f"wandb-{stage}.json").read_text())
        path = out / f"result-{stage}.json"
        expected = sha256(path)
        run = api.run(f"zhusq20/llm-memory-editability/{metadata['id']}")
        if (
            run.state != "finished"
            or run.config.get("batch") != settings["batch"]
            or run.config.get("architecture") != architecture
            or run.config.get("writer_steps") != 1
            or run.summary.get("result_sha256") != expected
        ):
            raise ValueError(f"Cloud one-step identity/status differs: {stage}")
        destination = out / "cloud-verification" / stage
        destination.mkdir(parents=True, exist_ok=True)
        downloaded = run.file(path.name).download(root=str(destination), replace=False)
        if sha256(downloaded.name) != expected:
            raise AssertionError("Cloud canonical result JSON differs")
        records.append({"stage": stage, **metadata, "result_sha256": expected})
    return {"passed": True, "runs": records, "canonical_json_exact": True}


def dispatch(stage, parent, settings, architecture, out, device, log):
    module = experiment.evaluation
    if stage == "cloud-audit":
        return cloud_audit(settings, architecture, out)
    view, loader, identity = experiment.interfaces(parent, settings, architecture)
    if stage == "preflight":
        return experiment.preflight(parent, settings, architecture, out, device)
    kwargs = {"reader_loader": loader, "identity_factory": identity}
    if stage == "calibrate":
        result = module.calibrate(view, "stream", out, device, **kwargs)
        result.update(quality_passed=result["passed"], passed=True, quality_is_advisory=True)
        write_json(out / "calibration.json", result)
        return result
    world = experiment.parent_engine(parent).validate_inputs(parent)
    if stage == "evaluate":
        result = module.evaluate(view, "stream", out, device, log, **kwargs)
        analysis, paired = experiment.analyze(out, world.metadata)
        write_json(out / "second-hop-analysis.json", analysis)
        write_json(out / "paired-queries.json", paired)
        result.update(
            second_hop_analysis_sha256=sha256(out / "second-hop-analysis.json"),
            paired_queries_sha256=sha256(out / "paired-queries.json"),
        )
        return result
    if stage == "audit":
        result = module.audit(view, "stream", out, device, **kwargs)
        analysis, paired = experiment.analyze(out, world.metadata)
        module._compare_nested(analysis, json.loads((out / "second-hop-analysis.json").read_text()))
        module._compare_nested(paired, json.loads((out / "paired-queries.json").read_text()))
        result.update(second_hop_conditions_recomputed=True, writer_steps=1)
        return result
    raise ValueError("Unsupported stage")


def main():
    verify_runtime()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--parent-config", type=Path)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument(
        "--architecture", choices=("transformer", "loop", "discoloop"), required=True
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--comparison-root", type=Path)
    parser.add_argument("--tracking", choices=("online", "disabled"), default="online")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    settings = json.loads(args.settings.read_text())
    experiment.validate_settings(settings)
    torch.set_num_threads(settings["runtime"]["cpu_threads"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    with (out / "execution.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = out / f"result-{args.stage}.json"
        failure = out / f"failure-{args.stage}.json"
        if path.exists() or failure.exists():
            raise FileExistsError("Preserve prior results/failures; no automatic retry")
        run, undo, cache_stats = None, None, None
        try:
            if args.stage == "aggregate":
                if args.comparison_root is None:
                    raise ValueError("Cross-architecture aggregation requires all three outputs")
                result = experiment.cross_architecture_report(
                    {
                        name: args.comparison_root / name / "evaluation"
                        for name in settings["architectures"]
                    },
                    out,
                )
                write_json(path, result)
                return 0
            if args.parent_config is None:
                raise ValueError("The unchanged parent config is required")
            parent = json.loads(args.parent_config.read_text())
            write_json(
                out / f"environment-{args.stage}.json",
                {
                    "utc": utc(),
                    "torch": torch.__version__,
                    "device": args.device,
                    "cpu_affinity": sorted(os.sched_getaffinity(0)),
                    "parent_writer_steps": parent["writer"]["steps"],
                    "evaluation_writer_steps": 1,
                    "new_base_parameter_updates": 0,
                },
            )
            if args.stage not in ("preflight", "cloud-audit") and args.tracking == "online":
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
                        "writer_steps": 1,
                        "parent_writer_steps": 4,
                        "settings": settings,
                    },
                    dir=str(out),
                    save_code=False,
                    settings=wandb.Settings(disable_git=True),
                )
                write_json(out / f"wandb-{args.stage}.json", {"id": run.id, "url": run.url})
                print(json.dumps({"wandb": run.url}), flush=True)

            def log(step, _record):
                if run:
                    run.log({"evaluation/completed_states": step}, step=step)

            if args.stage in ("evaluate", "audit"):
                from llm_memory_editability.ttcd_eval_speedup import install

                undo, cache_stats = install()
            result = dispatch(
                args.stage, parent, settings, args.architecture, out, args.device, log
            )
            if undo:
                undo()
                undo = None
                write_json(out / f"cache-{args.stage}.json", cache_stats)
            write_json(path, result)
            if run:
                run.summary["result_sha256"] = sha256(path)
                run.summary["new_base_parameter_updates"] = 0
                run.save(str(path), base_path=str(out), policy="now")
                run.finish(exit_code=0)
            write_json(out / f"status-{args.stage}.json", {"state": "complete", "utc": utc()})
            print(json.dumps({"stage": args.stage, "state": "complete"}), flush=True)
            return 0
        except BaseException:
            if undo:
                undo(verify=False)
            write_json(failure, {"utc": utc(), "traceback": traceback.format_exc()})
            if run:
                run.finish(exit_code=1)
            raise


if __name__ == "__main__":
    raise SystemExit(main())
