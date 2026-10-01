"""Verify text-structure runs using the frozen FP32/TF32 execution settings."""

import argparse
import json
import sys
from pathlib import Path

import torch
from analyze_text_structure import main as analyze
from run_text_structure import audit, report

from llm_memory_editability.grok_depth import utc, write_json
from llm_memory_editability.text_pretrain import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    output = Path(cfg["source_lock"]).parent
    lock = output / "verification-lock.json"
    script = "scripts/verify_text_structure.py"
    if not lock.exists():
        write_json(
            lock,
            dict(
                created_utc=utc(),
                sources={script: sha(script)},
                config_sha256=sha(args.config),
                precision="FP32 with CUDA matmul and cuDNN TF32 enabled",
            ),
        )
    frozen = json.loads(lock.read_text())
    assert frozen["sources"][script] == sha(script)
    assert frozen["config_sha256"] == sha(args.config)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    audit(args.config, args.device)
    sys.argv = [
        "scripts/analyze_text_structure.py",
        "analyze",
        "--config",
        args.config,
        "--device",
        args.device,
    ]
    analyze()
    result = report(args.config)
    write_json(
        output / "verified-completion.json",
        dict(
            status="complete",
            utc=utc(),
            runs=len(cfg["runs"]),
            verification_lock_sha256=sha(lock),
            artifacts={
                str(output / p): sha(output / p)
                for p in (
                    "lock.json",
                    "audit.json",
                    "summary.json",
                    "secondary-analysis-lock.json",
                    "secondary-analysis.json",
                    "completion-manifest.json",
                    "comparison.png",
                )
            },
        ),
    )
    print(json.dumps(result["scores"]), flush=True)


if __name__ == "__main__":
    main()
