#!/usr/bin/env python3
"""Execute graph preparation, supplementary evaluation or independent audit."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import platform
import subprocess
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from llm_memory_editability import ttcd_double_new_eval as experiment  # noqa: E402
from llm_memory_editability.grok_depth import utc, write_json  # noqa: E402
from llm_memory_editability.path_overlap import World, sha256  # noqa: E402


def cloud_audit(settings, name, out):
    import wandb

    api, records = wandb.Api(timeout=60), []
    for stage in ("evaluate", "audit"):
        tracking = json.loads((out / f"wandb-{stage}.json").read_text())
        run = api.run(f"zhusq20/llm-memory-editability/{tracking['id']}")
        path = out / f"result-{stage}.json"
        expected = sha256(path)
        if (
            run.state != "finished"
            or run.config.get("batch") != settings["batch"]
            or run.config.get("arm") != name
            or run.summary.get("result_sha256") != expected
        ):
            raise ValueError("Supplementary cloud identity/state/result hash differs")
        download = out / "cloud-verification" / stage
        download.mkdir(parents=True, exist_ok=True)
        saved = run.file(path.name).download(root=str(download), replace=True)
        if sha256(saved.name) != expected:
            raise ValueError("Supplementary canonical cloud JSON differs")
        records.append({"stage": stage, **tracking, "result_sha256": expected})
    return {"passed": True, "runs": records, "full_json_exact": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=("prepare", "preflight", "evaluate", "audit", "cloud-audit")
    )
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--model", choices=("single", "stream"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tracking", choices=("online", "offline", "disabled"), default="online")
    args = parser.parse_args()
    config, settings = [json.loads(p.read_text()) for p in (args.base_config, args.settings)]
    if config["batch"] != settings["parent_batch"]:
        raise ValueError("Supplementary evaluator has another parent")
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(settings["runtime"]["cpu_threads"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with (out / "execution.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result_path = out / f"result-{args.stage}.json"
        if result_path.exists():
            raise FileExistsError("Supplementary stage already completed; preserve it")
        write_json(
            out / f"environment-{args.stage}.json",
            {
                "utc": utc(),
                "hostname": platform.node(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "python": sys.version,
                "cpu_affinity": sorted(os.sched_getaffinity(0)),
                "device": args.device,
                "tf32": False,
                "scientific_training_updates": 0,
            },
        )
        run = None
        try:
            if args.stage in ("evaluate", "audit") and args.tracking != "disabled":
                import wandb

                defaults = json.loads(
                    (ROOT / "configs/experiment-tracking-defaults.json").read_text()
                )
                run = wandb.init(
                    entity=defaults["entity"],
                    project=defaults["project"],
                    name=f"{settings['batch']}/{args.model}/{args.stage}",
                    group=settings["batch"],
                    job_type=args.stage,
                    config={
                        **settings,
                        "arm": args.model,
                        "stage": args.stage,
                        "base_config_sha256": sha256(args.base_config),
                    },
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
                    eligible = [p for p in record["readout"]["pairs"] if p["eligible"]]
                    run.log(
                        {"evaluation/states": step, "evaluation/eligible_pairs": len(eligible)},
                        step=step,
                    )

            if args.stage == "prepare":
                if args.panel.exists():
                    raise FileExistsError("Preserve the frozen supplementary panel")
                panel = experiment.build_panel(config, settings, World(config["data_dir"]))
                args.panel.parent.mkdir(parents=True, exist_ok=True)
                write_json(args.panel, panel)
                result = {
                    "passed": True,
                    "panel_sha256": sha256(args.panel),
                    "facts": len(panel["facts"]),
                    "pairs": len(panel["pairs"]),
                    "states_per_arm": len(experiment.state_plan(panel)),
                    "model_queries": 0,
                    "scientific_training_updates": 0,
                }
            elif args.stage == "preflight":
                if not str(args.device).startswith("cuda"):
                    raise ValueError("Real trained-weight development preflight requires CUDA")
                subprocess.run(
                    [
                        sys.executable,
                        str(
                            ROOT
                            / "docs/development-artifacts/ttcd-double-new-eval-v1/verify_cuda.py"
                        ),
                        "--base-config",
                        str(args.base_config),
                        "--out",
                        str(out / "preflight"),
                    ],
                    check=True,
                    timeout=600,
                )
                result = json.loads((out / "preflight/receipt.json").read_text())
                if result.get("passed") is not True:
                    raise ValueError("Development CUDA preflight failed")
            elif args.stage == "cloud-audit":
                result = cloud_audit(settings, args.model, out)
            else:
                callback = getattr(experiment, args.stage)
                kwargs = {"log": log} if args.stage == "evaluate" else {}
                result = callback(
                    config, settings, args.model, out, args.panel, args.device, **kwargs
                )
            write_json(result_path, result)
            if run:
                run.summary["result_sha256"] = sha256(result_path)
                run.summary["states"] = result.get("states")
                run.summary["scientific_training_updates"] = 0
                run.save(str(result_path), base_path=str(out), policy="now")
                run.finish(exit_code=0)
            # Keep large predictions on disk, not the terminal.
            print(
                json.dumps(
                    {
                        "stage": args.stage,
                        "result": str(result_path),
                        "passed": result.get("passed"),
                        "states": result.get("states"),
                    }
                ),
                flush=True,
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
