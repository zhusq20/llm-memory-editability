#!/usr/bin/env python3
"""Run an unchanged frozen TTCD CLI with separately verified runtime shortcuts."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import runpy
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_extension(name):
    path = ROOT / "src/llm_memory_editability" / f"{name}.py"
    full_name = f"llm_memory_editability.{name}"
    spec = importlib.util.spec_from_file_location(full_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = module
    spec.loader.exec_module(module)
    return module


def ensure_eval_dependencies():
    """Older parent snapshots predate the supplementary evaluation module."""
    import llm_memory_editability

    name = "ttcd_double_new_eval"
    if importlib.util.find_spec(f"llm_memory_editability.{name}") is None:
        setattr(llm_memory_editability, name, load_extension(name))


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temporary.replace(path)


def main():
    args = sys.argv[1:]
    if args and args[0] == "--":
        args = args[1:]
    if len(args) < 2 or not args[0].startswith("/frozen/scripts/run_ttcd_"):
        raise ValueError(
            "Wrap only the registered unchanged frozen TTCD stage commands"
        )
    script, stage = Path(args[0]), args[1]
    if script.name not in (
        "run_ttcd_verified_base.py",
        "run_ttcd_architecture.py",
        "run_ttcd_double_new_eval.py",
    ):
        raise ValueError("No native foundation or unrelated program may be wrapped")
    if "--out" not in args:
        raise ValueError("An explicit original output directory is required")
    out = Path(args[args.index("--out") + 1])
    manifest_path = Path(os.environ["LM_TTCD_SPEEDUP_MANIFEST"])
    if sha(manifest_path) != os.environ["LM_TTCD_SPEEDUP_MANIFEST_SHA256"]:
        raise ValueError(
            "The mounted runtime manifest differs from the reviewed amendment"
        )
    manifest = json.loads(manifest_path.read_text())
    for relative, expected in manifest["files"].items():
        if sha(ROOT / relative) != expected:
            raise ValueError(
                f"Runtime extension changed after verification: {relative}"
            )
    if not manifest["numerical_verification"]["passed"]:
        raise ValueError(
            "The runtime needs an exact fixed-checkpoint numerical verification"
        )
    out.mkdir(parents=True, exist_ok=True)
    provenance_path = out / f"runtime-acceleration-{stage}.json"
    if provenance_path.exists():
        raise FileExistsError(
            "Preserve the original runtime receipt; no automatic retry"
        )
    provenance = {
        "runtime_manifest_sha256": sha(manifest_path),
        "frozen_cli": str(script),
        "frozen_cli_sha256": sha(script),
        "original_argv": args,
        "scientific_configuration_changed": False,
        "optimizer_steps_for_verification": 0,
        "training": "Prune only discarded post-capture outputs; original loss/gradient unchanged",
        "evaluation": "Cache immutable model digest; retain every fast-state and prediction audit",
        "pid": os.getpid(),
        "started_unix": time.time(),
        "state": "running",
    }
    write(provenance_path, provenance)
    train_module = load_extension("ttcd_train_speedup")
    ensure_eval_dependencies()
    eval_module = load_extension("ttcd_eval_speedup")
    train_undo = train_module.install()
    eval_undo, eval_stats = eval_module.install()
    # Preserve the frozen scientific identity; record actual runtime separately
    # in every new full training checkpoint and in the online run configuration.
    import torch

    original_save = torch.save

    def tracked_save(obj, *save_args, **kwargs):
        if isinstance(obj, dict) and obj.get("schema") == "ttcd-verified-base-v1":
            obj = {**obj, "runtime_acceleration": dict(provenance)}
        return original_save(obj, *save_args, **kwargs)

    torch.save = tracked_save
    import wandb

    original_init = wandb.init

    def tracked_init(*init_args, **kwargs):
        kwargs["config"] = {
            **kwargs.get("config", {}),
            "runtime_acceleration": dict(provenance),
        }
        return original_init(*init_args, **kwargs)

    wandb.init = tracked_init
    original_argv = sys.argv
    try:
        sys.argv = args
        runpy.run_path(str(script), run_name="__main__")
        provenance["state"] = "complete"
    except SystemExit as error:
        provenance["state"] = "complete" if error.code in (None, 0) else "failed"
        provenance["exit_code"] = error.code
        raise
    except BaseException:
        provenance.update(state="failed", traceback=traceback.format_exc())
        raise
    finally:
        sys.argv = original_argv
        torch.save, wandb.init = original_save, original_init
        try:
            eval_undo()
        finally:
            train_undo()
        provenance.update(
            finished_unix=time.time(),
            training_optimization=train_undo.provenance,
            evaluation_optimization=eval_stats,
        )
        write(provenance_path, provenance)


if __name__ == "__main__":
    main()
