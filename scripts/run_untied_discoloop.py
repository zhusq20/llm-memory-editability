#!/usr/bin/env python3
"""Train and reload DiscoLoop's vocabulary handoff on an untied 8-layer Transformer.

Reuses the native supervised-alignment runner unchanged except for three
module-level substitutions in this process: the model class, the architecture
flags (adds the channel position) and the channel FLOP estimate (one boundary).
"""

from __future__ import annotations

import argparse
import fcntl
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import torch  # noqa: E402
from run_discoloop_restore import CurveTracker, native_runner  # noqa: E402

from llm_memory_editability.grok_depth import utc, write_json  # noqa: E402
from llm_memory_editability.path_overlap import sha256  # noqa: E402
from llm_memory_editability.untied_discoloop import UntiedDiscoLoopGPT  # noqa: E402

BATCH = "untied-discoloop-development-v1"
ARM = "untied_discoloop"


def read_json(path):
    return json.loads(Path(path).read_text())


def patched_runner(config):
    native = native_runner()
    arm = config["arms"][ARM]
    split = arm["discoloop_split"]
    original_flags = native.architecture_flags

    def architecture_flags(arm):
        return {**original_flags(arm), "discoloop_split": arm["discoloop_split"]}

    def extra_discoloop_training_flops(spec, width, vocab_size, arm, sequence=4):
        # Two differentiable vocabulary matmuls at the single in-pass boundary.
        return 12 * spec["batch_size"] * sequence * width * vocab_size

    native.architecture_flags = architecture_flags
    native.extra_discoloop_training_flops = extra_discoloop_training_flops
    native.SupervisedAlignmentGPT = UntiedDiscoLoopGPT
    if split != 4 or config["spec"]["unique_layers"] != 8 or config["spec"]["repeats"] != 1:
        raise ValueError("Authorized design: 8 independent blocks, channel after block 4")
    return native


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("smoke", "train", "audit"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tracking", choices=("online", "disabled"), default="online")
    args = parser.parse_args()
    config = read_json(args.config)
    if config["batch"] != BATCH or set(config["arms"]) != {ARM}:
        raise ValueError("Unexpected configuration")
    torch.set_num_threads(config["runtime"]["cpu_threads"])
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    torch.cuda.set_device(0)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    native = patched_runner(config)
    run = None
    with (out / "execution.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = out / f"result-{args.stage}.json"
        if path.exists() or (out / f"failure-{args.stage}.json").exists():
            raise FileExistsError("Preserve completed/failed stages; relaunch in a new directory")
        try:
            if args.stage in ("train", "audit") and args.tracking == "online":
                import wandb

                tracking = config["tracking"]
                run = wandb.init(
                    entity=tracking["entity"],
                    project=tracking["project"],
                    group=BATCH,
                    name=f"{BATCH}/{ARM}/{args.stage}",
                    job_type=args.stage,
                    config={**config, "stage": args.stage},
                    dir=str(out),
                    save_code=False,
                    settings=wandb.Settings(disable_git=True),
                )
                write_json(out / f"wandb-{args.stage}.json", {"id": run.id, "url": run.url})
                print(json.dumps({"wandb": run.url}), flush=True)
            science = out / ARM if args.stage != "smoke" else out / "smoke" / ARM
            if args.stage in ("train", "smoke"):
                science.parent.mkdir(parents=True, exist_ok=True)
                with CurveTracker(run, science):
                    native.run(config, ARM, science, args.stage == "smoke", args.config)
                result = read_json(science / "complete.json")
            else:
                native.audit(config, ARM, science, args.config)
                result = read_json(science / "audit.json")
            write_json(path, result)
            if run:
                run.summary["result_sha256"] = sha256(path)
                run.summary["terminal_step"] = result["step"]
                run.save(str(path), base_path=str(out), policy="now")
                run.finish(exit_code=0)
            print(json.dumps({"stage": args.stage, "step": result.get("step")}), flush=True)
        except BaseException as error:
            write_json(
                out / f"failure-{args.stage}.json",
                {"error": repr(error), "traceback": traceback.format_exc(), "utc": utc()},
            )
            if run:
                run.finish(exit_code=1)
            raise


if __name__ == "__main__":
    main()
