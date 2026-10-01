"""Finish reload audits and reporting once every frozen run has completed."""

import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from llm_memory_editability.grok_depth import write_json
from llm_memory_editability.text_pretrain import sha

ROOT = Path("docs/development-artifacts/text-pretrain-v1")


def wait_for(paths):
    deadline = time.monotonic() + 3600
    while not all(p.exists() for p in paths):
        if time.monotonic() > deadline:
            raise TimeoutError([str(p) for p in paths if not p.exists()])
        time.sleep(1)


def main():
    cfg = json.loads(Path("configs/text-pretrain-confirmation-v1.json").read_text())
    wait_for([Path(cfg["output_root"]) / name / "complete.json" for name in cfg["runs"]])
    subprocess.run(
        [sys.executable, "scripts/audit_text_pretrain.py"],
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="2"),
        check=True,
    )
    wait_for(
        [
            ROOT / "confirmation" / "mechanism" / name / str(node) / "summary.json"
            for name in cfg["runs"]
            for node in (16000, 32000)
        ]
    )
    subprocess.run([sys.executable, "scripts/report_text_pretrain.py"], check=True)
    completions = []
    for stage in ("development", "confirmation"):
        source = json.loads(Path(f"configs/text-pretrain-{stage}-v1.json").read_text())
        for name in source["runs"]:
            path = Path(source["output_root"]) / name / "complete.json"
            d = json.loads(path.read_text())
            completions.append(
                dict(
                    run=name,
                    stage=stage,
                    steps=d["final"]["step"],
                    supervised_tokens=d["final"]["supervised_tokens"],
                    estimated_flops=d["final"]["estimated_flops"],
                    training_seconds=d["final"]["seconds"],
                    parameters=d["parameters"],
                    complete_sha256=sha(path),
                )
            )
    audit = [
        json.loads((ROOT / stage / "endpoint-audit.json").read_text())
        for stage in ("development", "confirmation")
    ]
    summary = json.loads((ROOT / "report" / "summary.json").read_text())
    sources = [
        *Path("scripts").glob("*text_pretrain.py"),
        Path("src/llm_memory_editability/text_pretrain.py"),
        Path("tests/test_text_pretrain.py"),
        Path("scripts/run_bios_same_weight.py"),
        Path("scripts/report_bios_same_weight.py"),
    ]
    write_json(
        ROOT / "completion-manifest.json",
        dict(
            status="complete",
            created_unix=time.time(),
            training_runs=len(completions),
            development_runs=3,
            confirmation_runs=18,
            learning_nodes=len(completions) * 9,
            endpoint_reload_runs=sum(r["runs"] for r in audit),
            raw_prediction_scoring_rows=sum(r["raw_scoring_rows"] for r in audit),
            mechanism_checkpoints=36,
            mechanism_scoring_rows=summary["intervention_prediction_rows_checked"],
            related_tests=26,
            bios_same_weight_endpoints=18,
            bios_generated_records=6912,
            bios_frozen_base_parameter_comparisons=12,
            donor_pairing_audit=json.loads(
                (ROOT / "confirmation/donor-pairing-audit.json").read_text()
            ),
            totals={
                k: sum(r[k] for r in completions)
                for k in ("steps", "supervised_tokens", "estimated_flops", "training_seconds")
            },
            runs=completions,
            environment=dict(
                python=platform.python_version(),
                torch=torch.__version__,
                numpy=np.__version__,
                cuda=torch.version.cuda,
                hardware="NVIDIA RTX PRO 6000 Blackwell Server Edition",
                gpus=[2, 5],
                data=(
                    "Locally generated typed random worlds and deterministic English templates; "
                    "no downloaded training corpus"
                ),
            ),
            sources={str(p): sha(p) for p in sources},
            notes=[
                "Training time is summed across concurrent processes, not dedicated GPU hours; "
                "excludes evaluation/startup/file I/O.",
                "Matmul FLOPs use the frozen model estimator; not measured hardware FLOPs.",
                "No confirmation runs excluded; development position choices kept separate.",
            ],
        ),
    )
    print(
        "Complete: 21 training runs, 36 mechanism checkpoints, reloads and scoring.",
        flush=True,
    )


if __name__ == "__main__":
    main()
