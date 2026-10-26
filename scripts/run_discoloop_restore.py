#!/usr/bin/env python3
"""Run one isolated restoration stage using the original scientific trainer."""

from __future__ import annotations

import argparse
import fcntl
import importlib.util
import json
import os
import platform
import sys
import threading
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from llm_memory_editability.discoloop_restore import (  # noqa: E402
    INITIAL_MODEL_SHA256,
    PARAMETERS,
    foundation_gate,
    validate_recipe,
)
from llm_memory_editability.grok_depth import utc, write_json  # noqa: E402
from llm_memory_editability.path_overlap import sha256  # noqa: E402


def native_runner():
    spec = importlib.util.spec_from_file_location(
        "restored_native_supervised", ROOT / "scripts/run_supervised_alignment.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_json(path):
    return json.loads(Path(path).read_text())


def preflight(config, runner, device):
    root = Path(config["input_root"])
    metadata = read_json(root / "data/complete.json")
    validate_recipe(config, metadata, read_json(root / "reference/gpt2/config.json"))
    for name in ("world", "panels"):
        if sha256(root / f"data/{name}.npz") != metadata[f"{name}_sha256"]:
            raise ValueError(f"Original {name} bytes differ")
    checkpoint = torch.load(config["initial_checkpoint"], map_location="cpu", weights_only=True)
    if checkpoint["step"] != 0:
        raise ValueError("Original initialization is required")
    digest = runner.state_digest(checkpoint["model"])
    # Tied LM output/embedding tensors appear twice in state_dict.
    state = checkpoint["model"]
    parameters = sum(t.numel() for k, t in state.items() if k != "lm_head.weight")
    if digest != INITIAL_MODEL_SHA256 or parameters != PARAMETERS:
        raise ValueError("Exact historical initialization or parameter count differs")
    for key, value in config["spec"].items():
        if key != "name" and checkpoint["spec"].get(key) != value:
            raise ValueError(f"Initial checkpoint recipe differs: {key}")
    del checkpoint, state
    probe = torch.randn(64, 64, device=device, requires_grad=True)
    probe.square().matmul(probe).mean().backward()
    torch.cuda.synchronize(device)
    if not torch.isfinite(probe.grad).all():
        raise RuntimeError("Actual CUDA forward/backward failed")
    return {
        "passed": True,
        "initial_model_sha256": digest,
        "parameters": parameters,
        "original_world_sha256": metadata["world_sha256"],
        "original_panels_sha256": metadata["panels_sha256"],
        "cuda_forward_backward_synchronized": True,
        "environment_identical_to_history": False,
        "scope": "Exact initial tensors and successful original scientific recipe; "
        "new environment and training trajectory",
    }


class CurveTracker:
    """Observe emitted files without changing the original CUDA training path."""

    def __init__(self, run, path):
        self.run, self.path = run, path
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.observe, daemon=True)
        self.logged = set()
        self.error = None

    def drain(self):
        if self.run is None or not (self.path / "learning.json").exists():
            return
        for record in read_json(self.path / "learning.json"):
            step = record["step"]
            if step in self.logged:
                continue
            values = {"training/step": step}
            for name, panel in record["metrics"].items():
                values.update({f"{name}/{key}": value for key, value in panel.items()})
            for key, value in record.items():
                if isinstance(value, (float, int)):
                    values[f"cost/{key}"] = value
            self.run.log(values, step=step)
            self.logged.add(step)

    def observe(self):
        try:
            while not self.stop.wait(15):
                self.drain()
        except BaseException:
            self.error = traceback.format_exc()

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join(timeout=60)
        self.drain()
        if self.error:
            raise RuntimeError(f"W&B curve observer failed: {self.error}")


def cloud_audit(config, out):
    """Compare canonical uploaded result bytes, avoiding SDK summary coercions."""
    import wandb

    records, api = [], wandb.Api(timeout=60)
    for stage in ("train", "audit", "foundation-gate"):
        tracking = read_json(out / f"wandb-{stage}.json")
        run = api.run(f"zhusq20/llm-memory-editability/{tracking['id']}")
        result_path = out / f"result-{stage}.json"
        expected = sha256(result_path)
        if (
            run.state != "finished"
            or run.config.get("batch") != config["batch"]
            or run.config.get("stage") != stage
            or run.summary.get("result_sha256") != expected
        ):
            raise ValueError(f"Cloud state or result digest differs: {stage}")
        folder = out / "cloud-verification" / stage
        folder.mkdir(parents=True, exist_ok=True)
        downloaded = run.file(result_path.name).download(root=str(folder), replace=True)
        if sha256(downloaded.name) != expected:
            raise ValueError(f"Canonical full result differs: {stage}")
        if stage == "train" and run.summary.get("terminal_step") != 300000:
            raise ValueError("Cloud training endpoint differs")
        records.append({"stage": stage, **tracking, "state": run.state, "result_sha256": expected})
    return {"passed": True, "runs": records, "full_json_exact": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=(
            "preflight",
            "smoke",
            "smoke-audit",
            "train",
            "audit",
            "foundation-gate",
            "cloud-audit",
        ),
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tracking", choices=("online", "offline", "disabled"), default="online")
    args = parser.parse_args()
    config = read_json(args.config)
    validate_recipe(config)
    if args.device != "cuda:0":
        raise ValueError("The original native runner uses the container's isolated cuda:0")
    torch.set_num_threads(config["runtime"].get("intraop_threads", 4))
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.set_device(0)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    with (out / "execution.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result_path = out / f"result-{args.stage}.json"
        if result_path.exists() or (out / f"failure-{args.stage}.json").exists():
            raise FileExistsError("Preserve completed or failed stages; no automatic retry")
        write_json(
            out / f"environment-{args.stage}.json",
            {
                "utc": utc(),
                "python": sys.version,
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "hostname": platform.node(),
                "cpu_affinity": sorted(os.sched_getaffinity(0)),
                "tf32": True,
                "intraop_threads": torch.get_num_threads(),
                "config_sha256": sha256(args.config),
            },
        )
        run = None
        try:
            if args.stage in ("train", "audit", "foundation-gate") and args.tracking != "disabled":
                import wandb

                defaults = read_json(ROOT / "configs/experiment-tracking-defaults.json")
                run = wandb.init(
                    entity=defaults["entity"],
                    project=defaults["project"],
                    name=f"{config['batch']}/discoloop/{args.stage}",
                    group=config["batch"],
                    job_type=args.stage,
                    config={**config, "stage": args.stage, "arm": "discoloop"},
                    mode=args.tracking,
                    dir=str(out),
                    save_code=False,
                    settings=wandb.Settings(disable_git=True),
                )
                write_json(out / f"wandb-{args.stage}.json", {"id": run.id, "url": run.url})
                print(json.dumps({"wandb": run.url}), flush=True)
            native = native_runner()
            science = out / "discoloop"
            smoke = out / "smoke/discoloop"
            if args.stage == "preflight":
                result = preflight(config, native, args.device)
            elif args.stage in ("train", "smoke"):
                destination = science if args.stage == "train" else smoke
                destination.parent.mkdir(parents=True, exist_ok=True)
                with CurveTracker(run, destination):
                    native.run(config, "discoloop", destination, args.stage == "smoke", args.config)
                result = read_json(destination / "complete.json")
            elif args.stage in ("audit", "smoke-audit"):
                destination = science if args.stage == "audit" else smoke
                native.audit(config, "discoloop", destination, args.config)
                result = read_json(destination / "audit.json")
            elif args.stage == "foundation-gate":
                checkpoint = science / "checkpoint-0300000.pt"
                result = foundation_gate(
                    config,
                    read_json(science / "complete.json"),
                    read_json(science / "audit.json"),
                    sha256(checkpoint),
                )
                result.update(
                    {
                        "checkpoint": str(checkpoint),
                        "created_utc": utc(),
                        "audit_sha256": sha256(science / "audit.json"),
                        "complete_sha256": sha256(science / "complete.json"),
                        "config_sha256": sha256(args.config),
                    }
                )
                write_json(out / "foundation-gate.json", result)
            else:
                result = cloud_audit(config, out)
            write_json(result_path, result)
            if run is not None:
                run.summary.update(
                    {
                        "result_sha256": sha256(result_path),
                        "passed": result.get("passed", result.get("state") == "complete"),
                        "terminal_step": result.get("step"),
                        "base_parameter_updates": result.get("step")
                        if args.stage == "train"
                        else 0,
                    }
                )
                run.save(str(result_path), base_path=str(out), policy="now")
                run.finish()
                run = None
            print(
                json.dumps(
                    {
                        "stage": args.stage,
                        "result": str(result_path),
                        "passed": result.get("passed"),
                    }
                ),
                flush=True,
            )
            if result.get("passed") is False:
                raise SystemExit(3)
        except BaseException as error:
            write_json(
                out / f"failure-{args.stage}.json",
                {
                    "error": repr(error),
                    "traceback": traceback.format_exc(),
                    "utc": utc(),
                },
            )
            if run is not None:
                run.finish(exit_code=1)
            raise


if __name__ == "__main__":
    main()
