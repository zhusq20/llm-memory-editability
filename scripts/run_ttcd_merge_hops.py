#!/usr/bin/env python3
"""Run a frozen evaluation-only merge/hop stage using audited one-step parents."""

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
from run_ttcd_one_step import verify_runtime  # noqa: E402

from llm_memory_editability import ttcd_merge_hops as experiment  # noqa: E402
from llm_memory_editability.grok_depth import utc, write_json  # noqa: E402
from llm_memory_editability.path_overlap import sha256  # noqa: E402

STAGES = ("prepare", "preflight", "evaluate", "audit", "cloud-audit", "aggregate")


def cloud_audit(settings, architecture, out):
    import wandb

    api, records = wandb.Api(timeout=60), []
    for stage in ("evaluate", "audit"):
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
            raise ValueError("Cloud merge/hop identity, completion or canonical result differs")
        directory = out / "cloud-verification" / stage
        directory.mkdir(parents=True, exist_ok=True)
        downloaded = run.file(path.name).download(root=str(directory), replace=False)
        if sha256(downloaded.name) != expected:
            raise AssertionError("Cloud canonical JSON differs")
        records.append({"stage": stage, **metadata, "result_sha256": expected})
    return {"passed": True, "runs": records, "canonical_json_exact": True}


def main():
    verify_runtime()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--parent-config", type=Path)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument(
        "--architecture", choices=("transformer", "loop", "discoloop"), required=True
    )
    parser.add_argument("--parent-evaluation", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--comparison-root", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tracking", choices=("online", "disabled"), default="online")
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
        result_path, failure = out / f"result-{args.stage}.json", out / f"failure-{args.stage}.json"
        if result_path.exists() or failure.exists():
            raise FileExistsError("Preserve existing results/failures; no automatic retry")
        run, undo, stats = None, None, None
        try:
            if args.stage == "aggregate":
                if args.comparison_root is None:
                    raise ValueError("Three-architecture comparison root required")
                result = experiment.cross_architecture_report(
                    {
                        name: args.comparison_root / name / "evaluation"
                        for name in settings["architectures"]
                    },
                    out,
                )
                write_json(result_path, result)
                return 0
            if args.parent_config is None:
                raise ValueError("Unchanged model-parent configuration required")
            parent = json.loads(args.parent_config.read_text())
            write_json(
                out / f"environment-{args.stage}.json",
                {
                    "utc": utc(),
                    "torch": torch.__version__,
                    "device": args.device,
                    "cpu_affinity": sorted(os.sched_getaffinity(0)),
                    "writer_steps": 1,
                    "new_base_parameter_updates": 0,
                    "parent_writer_training_steps": 4,
                },
            )
            if args.stage == "prepare":
                result = experiment.prepare(parent, settings, out)
            elif args.stage == "cloud-audit":
                result = cloud_audit(settings, args.architecture, out)
            else:
                if args.parent_evaluation is None:
                    raise ValueError("Audited one-step parent states required")
                if args.stage in ("evaluate", "audit"):
                    from llm_memory_editability.ttcd_eval_speedup import install

                    undo, stats = install()
                    if args.tracking == "online":
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
                                "settings": settings,
                            },
                            dir=str(out),
                            save_code=False,
                            settings=wandb.Settings(disable_git=True),
                        )
                        write_json(out / f"wandb-{args.stage}.json", {"id": run.id, "url": run.url})
                        print(json.dumps({"wandb": run.url}), flush=True)

                def log(step, record):
                    if run:
                        kind = record["spec"]["kind"]
                        base = record["readout"]["metrics"]
                        values = {
                            "evaluation/completed_states": step,
                            f"{kind}/written_facts": record["spec"]["t"],
                            f"{kind}/second_hop_D": base["D_fact_equal"],
                            f"{kind}/E": base["E_fact_equal"],
                            f"{kind}/U_atoms": base["U_atoms"]["old_accuracy"],
                            f"{kind}/U_chains": base["U_chains"]["old_accuracy"],
                        }
                        run.log(values, step=step)

                function = getattr(experiment, args.stage)
                kwargs = {"log": log} if args.stage == "evaluate" else {}
                result = function(
                    parent,
                    settings,
                    args.architecture,
                    args.parent_evaluation,
                    out,
                    args.device,
                    **kwargs,
                )
            if undo:
                undo()
                undo = None
                write_json(out / f"cache-{args.stage}.json", stats)
            write_json(result_path, result)
            if run:
                run.summary["result_sha256"] = sha256(result_path)
                run.summary["new_base_parameter_updates"] = 0
                run.save(str(result_path), base_path=str(out), policy="now")
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
