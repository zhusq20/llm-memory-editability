"""Freeze and execute paired full-token text-pretraining experiments."""

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

from llm_memory_editability.grok_depth import write_json
from llm_memory_editability.text_pretrain import run, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["freeze", "run"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--run")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    if args.command == "freeze":
        target = Path(cfg["source_lock"])
        if target.exists():
            raise FileExistsError(target)
        sources = [
            Path("scripts/run_text_pretrain.py"),
            *[
                Path(f"src/llm_memory_editability/{s}.py")
                for s in ("text_pretrain", "bios_model", "grok_depth", "grok_loop_model")
            ],
        ]
        for file in sources:
            dst = target.parent / "source" / file
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file, dst)
        write_json(
            target,
            dict(
                created_unix=time.time(),
                config_sha256=sha(args.config),
                sources={str(p): sha(p) for p in sources},
                git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            ),
        )
    else:
        selected = cfg["runs"] if args.run is None else {args.run: cfg["runs"][args.run]}
        for name, overrides in selected.items():
            spec = dict(
                cfg["base"], **overrides, source_lock=cfg["source_lock"], config_path=args.config
            )
            run(spec, Path(cfg["output_root"]) / name, args.device)


if __name__ == "__main__":
    main()
