#!/usr/bin/env python3
"""Reuse frozen full-GPT training, adding only matched support sampling."""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import run_loop_hop_depth as core  # noqa: E402
import torch  # noqa: E402

from llm_memory_editability.grok_depth import utc, write_json  # noqa: E402
from llm_memory_editability.loop_hop_data import load  # noqa: E402
from llm_memory_editability.loop_hop_support import (  # noqa: E402
    MatchedSupportStream,
    matched_exposure_summary,
)


@contextlib.contextmanager
def sampling(config, name):
    native = core.native
    original_stream, original_summary = native.MergedBatchStream, native.exposure_summary
    _, _, metadata = load(config["input_root"])
    condition = config["arms"][name]["data_condition"]
    if metadata["train_compositions_per_hop"] != config["supports"][condition]:
        raise ValueError("Mounted support condition disagrees with the frozen arm")
    if metadata["panels_sha256"] != config["parent"]["panels_sha256"]:
        raise ValueError("The common frozen evaluation panels changed")
    try:
        if condition == "more":

            def stream(size, seed):
                return MatchedSupportStream(
                    size,
                    seed,
                    atoms=metadata["atomic_examples"],
                    base_support=metadata["base_train_compositions_per_hop"],
                    multiplier=metadata["support_multiplier"],
                )

            native.MergedBatchStream = stream
            native.exposure_summary = matched_exposure_summary
        yield
    finally:
        native.MergedBatchStream, native.exposure_summary = original_stream, original_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=(
            "preflight",
            "smoke",
            "smoke-audit",
            "smoke-grid",
            "smoke-grid-audit",
            "train",
            "audit",
            "grid",
            "grid-audit",
            "cloud",
        ),
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.stage == "cloud":
        core.cloud(config, args.arm, args.base)
        return
    torch.set_num_threads(config["runtime"]["cpu_threads"])
    torch.set_num_interop_threads(1)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(config["runtime"]["gpu_memory_fraction"], 0)
    torch.backends.cuda.matmul.allow_tf32 = True
    smoke = args.stage.startswith("smoke")
    out = args.base / ("smoke" if smoke else "runs") / args.arm
    try:
        with sampling(config, args.arm):
            if args.stage == "preflight":
                core.native.preflight(config, args.arm, args.base)
            elif args.stage == "smoke":
                core.native.train(config, args.arm, out, True, args.config, args.base)
            elif args.stage in ("audit", "smoke-audit"):
                core.native.audit(config, args.arm, out, args.config)
            elif "grid" in args.stage:
                core.endpoint(
                    config, args.arm, args.base, args.config, smoke, args.stage.endswith("audit")
                )
            else:
                core.tracked_train(config, args.arm, args.base, args.config)
    except BaseException as error:
        import traceback

        write_json(
            args.base / "failures" / f"{args.arm}-{args.stage}.json",
            {"utc": utc(), "error": repr(error), "traceback": traceback.format_exc()},
        )
        raise


if __name__ == "__main__":
    main()
