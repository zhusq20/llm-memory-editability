"""Run the frozen descriptive/intervention or MLP-only editing development analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from llm_memory_editability.depth_step_mechanism import edit_analysis, load_run, trace_analysis
from llm_memory_editability.grok_depth import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--mode", choices=("trace", "edit"), required=True)
    arguments = parser.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = True
    model, world, source = load_run(arguments.run_dir, arguments.checkpoint, arguments.device)
    if arguments.mode == "trace":
        report = trace_analysis(model, world, arguments.out, arguments.device)
    else:
        report = edit_analysis(model, world, arguments.out, arguments.device)
    report["source"] = source
    write_json(arguments.out / f"{arguments.mode}-summary.json", report)
    print(f"{arguments.mode} analysis complete: {arguments.out}", flush=True)


if __name__ == "__main__":
    main()
