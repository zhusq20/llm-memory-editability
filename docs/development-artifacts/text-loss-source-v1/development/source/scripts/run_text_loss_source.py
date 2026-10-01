"""Freeze or run a paired loss-source continuation experiment."""

import argparse
import json
import shutil
import time
from pathlib import Path

import torch

from llm_memory_editability.grok_depth import write_json
from llm_memory_editability.text_loss_source import run
from llm_memory_editability.text_pretrain import sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["freeze", "run"])
    p.add_argument("--config", required=True)
    p.add_argument("--run")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    cfg["config_path"] = args.config
    if args.command == "freeze":
        target = Path(cfg["lock"])
        assert not target.exists()
        files = [
            Path("scripts/run_text_loss_source.py"),
            *[
                Path("src/llm_memory_editability") / (n + ".py")
                for n in [
                    "text_loss_source",
                    "text_pretrain",
                    "grok_depth",
                    "grok_loop_model",
                    "bios_model",
                ]
            ],
        ]
        for f in files:
            out = target.parent / "source" / f
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, out)
        parents = sorted({str(Path(v["parent"]) / "latest.pt") for v in cfg["runs"].values()})
        inputs = sorted(
            {
                str(Path(v["parent"]) / f)
                for v in cfg["runs"].values()
                for f in ["world.npz", "training-table.npz", "learning.json", "spec.json"]
            }
        )
        write_json(
            target,
            dict(
                created_unix=time.time(),
                config_sha256=sha(args.config),
                sources={str(f): sha(f) for f in files},
                parents={f: sha(f) for f in parents},
                inputs={f: sha(f) for f in inputs},
                plan_sha256=sha(cfg["plan"]),
            ),
        )
    else:
        torch.set_num_threads(2)
        torch.cuda.set_device(args.device)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.set_per_process_memory_fraction(0.1, args.device)
        for name in [args.run] if args.run else cfg["runs"]:
            run(cfg, name, args.device)


if __name__ == "__main__":
    main()
