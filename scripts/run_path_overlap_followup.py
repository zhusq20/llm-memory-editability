"""Execute one frozen, zero-training post-hoc Z1 diagnostic node."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from llm_memory_editability.grok_depth import utc, write_json
from llm_memory_editability.path_overlap import sha256
from llm_memory_editability.path_overlap_followup import (
    CODE_FILES,
    compare_audit,
    run_diagnostic,
)

PROJECT = Path(__file__).resolve().parents[1]


def run(args):
    config = json.loads(Path(args.config).read_text())
    torch.set_num_threads(int(config.get("execution", {}).get("cpu_threads", 4)))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    out = Path(args.out)
    source = Path(config["source"]["science_root"]).resolve()
    if out.resolve() == source or source in out.resolve().parents:
        raise ValueError("Diagnostics must never write inside the completed Z1 source")
    if args.smoke and "smoke" not in out.parts:
        raise ValueError("Smoke output requires a separate directory named smoke")
    if args.audit:
        prior_hashes = json.loads((out / "hashes.json").read_text())
        if not prior_hashes["complete"] or prior_hashes["config_sha256"] != sha256(args.config):
            raise ValueError("Audit config differs or original diagnostic is incomplete")
        for name, digest in prior_hashes["outputs_sha256"].items():
            if sha256(out / name) != digest:
                raise ValueError(f"Audit source output hash differs: {name}")
        old_summary = json.loads((out / "summary.json").read_text())
        with np.load(out / "predictions.npz", allow_pickle=False) as data:
            old_arrays = {k: data[k] for k in data.files}
    elif (out / "hashes.json").exists():
        raise ValueError("Existing output is immutable; use --audit or a new attempt directory")
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    code = {name: sha256(PROJECT / name) for name in CODE_FILES}
    if args.audit and prior_hashes["code_sha256"] != code:
        raise ValueError("Audit implementation differs from diagnostic implementation")
    tracking = config.get("tracking", {})
    wandb_run = None
    if tracking.get("enabled") and not args.audit and not args.smoke:
        import wandb

        node = config["nodes"][args.node_index]
        name = node.get("tracking_name", f"{config['batch']}-{node['run']}-{node['step']}")
        wandb_run = wandb.init(
            entity=tracking["entity"],
            project=tracking["project"],
            name=name,
            group=config["batch"],
            job_type="zero-training-post-hoc-diagnostic",
            mode="online",
            dir=str(out),
            config={
                "batch": config["batch"],
                "node": node,
                "analysis": config["analysis"],
                "config_sha256": sha256(args.config),
                "code_sha256": code,
                "optimizer_updates": 0,
                "precision": "float32, explicit causal softmax; TF32 disabled",
            },
        )
    failed = False
    try:
        summary, arrays = run_diagnostic(config, args.node_index, device, smoke=args.smoke)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        if args.audit:
            result = compare_audit(old_summary, old_arrays, summary, arrays)
            write_json(
                out / "audit.json",
                {
                    **result,
                    "pid": os.getpid(),
                    "created_utc": utc(),
                    "seconds": elapsed,
                    "config_sha256": sha256(args.config),
                    "code_sha256": code,
                    "audited_outputs_sha256": {
                        name: sha256(out / name) for name in ("summary.json", "predictions.npz")
                    },
                },
            )
        else:
            if wandb_run is not None:
                metrics = {"optimizer_updates": 0, "execution_seconds": elapsed}
                for label in ("primary", "same_bridge"):
                    for mode, kinds in summary[label].items():
                        for kind, values in kinds.items():
                            for metric in ("T", "R", "delta", "facts", "eligible_queries"):
                                value = values[metric]
                                if value is not None:
                                    metrics[f"{label}/{kind}/{mode}/{metric}"] = value
                wandb_run.log(metrics, step=0)
                wandb_run.config.update(summary["source"])
                summary["wandb"] = {"id": wandb_run.id, "url": wandb_run.url}
            write_json(out / "summary.json", summary)
            np.savez_compressed(out / "predictions.npz", **arrays)
            write_json(
                out / "hashes.json",
                {
                    "complete": True,
                    "audit_pending": True,
                    "run": summary["run"],
                    "step": summary["step"],
                    "smoke": args.smoke,
                    "config_sha256": sha256(args.config),
                    "code_sha256": code,
                    "source": summary["source"],
                    "outputs_sha256": {
                        name: sha256(out / name) for name in ("summary.json", "predictions.npz")
                    },
                    "seconds": elapsed,
                    "optimizer_updates": 0,
                    "device": torch.cuda.get_device_name(device)
                    if device.type == "cuda"
                    else "cpu",
                    "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device))
                    if device.type == "cuda"
                    else None,
                    "torch": torch.__version__,
                    "pid": os.getpid(),
                    "completed_utc": utc(),
                    "wandb": summary.get("wandb"),
                },
            )
        print(
            json.dumps(
                {
                    "run": summary["run"],
                    "step": summary["step"],
                    "audit": args.audit,
                    "smoke": args.smoke,
                    "seconds": elapsed,
                    "out": str(out),
                }
            ),
            flush=True,
        )
    except Exception as exc:
        failed = True
        write_json(
            out / ("audit-failure.json" if args.audit else "failure.json"),
            {"error": repr(exc), "created_utc": utc(), "config_sha256": sha256(args.config)},
        )
        raise
    finally:
        if wandb_run is not None:
            wandb_run.finish(exit_code=1 if failed else 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--node-index", type=int, required=True)
    parser.add_argument("--out", required=True, help="Dedicated output directory for this node")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
