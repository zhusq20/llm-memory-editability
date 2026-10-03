"""Zero-training early-exit/recurrence comparison on real and synthetic weights."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time

import diagnose_realworld_loop as diagnosis
import numpy as np
import torch

from llm_memory_editability import grokking_reproduction as synthetic

ROOT = diagnosis.PROJECT / "results/realworld-loop-depth-diagnosis-v1"


def wait_gpu():
    while True:
        path = diagnosis.ROOT / "controller-state.json"
        if path.exists():
            state = json.loads(path.read_text())
            if not state.get("queued"):
                memory = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=index,memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                )
                for row in memory.splitlines():
                    gpu, used = map(int, row.split(","))
                    if (
                        gpu in [3, 4, 5, 6, 7]
                        and str(gpu) not in state.get("active", {})
                        and used < 100
                    ):
                        return gpu
        time.sleep(10)


def measure(gpu):
    device = diagnosis.configure(gpu)
    data, added, manifest = diagnosis.load_data("native")
    tokenizer = diagnosis.GPT2TokenizerFast.from_pretrained(
        diagnosis.BASE["tokenizer"], local_files_only=True
    )
    settings = [
        ("standard8", 1, 4),
        ("standard8", 1, 8),
        ("loop4x2", 1, 4),
        ("loop4x2", 2, 4),
        ("loop4x2", 4, 4),
    ]
    for arch, repeats, blocks in settings:
        name = f"real-{arch}-b{blocks}-r{repeats}"
        out = ROOT / "development" / name
        out.mkdir(parents=True, exist_ok=True)
        model = diagnosis.construct(diagnosis.BASE["model"], diagnosis.spec_for(arch), device)
        source = torch.load(diagnosis.checkpoint_path(arch), map_location="cpu", weights_only=False)
        model.load_state_dict(source["model"])
        del source
        model.transformer.h = torch.nn.ModuleList(list(model.transformer.h)[:blocks])
        model.repeats = repeats
        digest = diagnosis.model_digest(model)
        diagnosis.write_json(
            out / "run.json",
            {
                "spec": {
                    "dataset": "real",
                    "arch": arch,
                    "repeats": repeats,
                    "blocks": blocks,
                    "scientific_training_updates": 0,
                },
                "pid": os.getpid(),
                "gpu": gpu,
                "world_sha256": manifest["prepared_sha256"],
                "initial_model_sha256": digest,
                "tracking_group": "realworld-loop-diagnosis-v1",
                "job_type": "depth-evaluation",
            },
        )
        diagnosis.write_json(out / "learning.json", [])
        full = (arch == "standard8" and blocks == 8) or (arch == "loop4x2" and repeats in [2, 4])
        metrics, predictions = diagnosis.evaluate_selected(
            model, data, added, tokenizer, device, full
        )
        control = None
        if (arch == "standard8" and blocks == 8) or (arch == "loop4x2" and repeats == 2):
            path = diagnosis.checkpoint_path(arch).parent / "endpoint-predictions.json"
            expected = json.loads(path.read_text())["test_all"]
            control = sum(
                a["generated_tokens"] != b["generated_tokens"]
                for a, b in zip(expected, predictions["test_all"], strict=True)
            )
            assert control == 0, "Unmodified numerical control differs"
        assert diagnosis.model_digest(model) == digest
        diagnosis.write_json(out / "predictions.json", predictions)
        diagnosis.write_json(out / "learning.json", [{"step": 0, "metrics": metrics}])
        diagnosis.write_json(
            out / "complete.json",
            {
                "passed": True,
                "full": full,
                "unchanged_parameters": True,
                "control_mismatches": control,
            },
        )
        print(
            json.dumps(
                {
                    "name": name,
                    "metrics": {
                        k: metrics[k]["alias_em"] for k in ["atomic", "test_ii", "test_oo"]
                    },
                }
            ),
            flush=True,
        )
        del model
        torch.cuda.empty_cache()
    config = json.loads(
        (diagnosis.PROJECT / "configs/grokking-reproduction-development-v1.json").read_text()
    )
    spec = next(s for s in config["runs"] if s["name"] == "loop4x2-phi7.2-wd0.1")
    metadata = json.loads((synthetic.ROOT / "data/complete.json").read_text())
    panels = dict(np.load(synthetic.ROOT / "data/panels.npz"))
    for repeats in [1, 2, 4]:
        out = ROOT / "development" / f"synthetic-loop-r{repeats}"
        out.mkdir(parents=True, exist_ok=True)
        model = synthetic.construct(spec, metadata, device)
        checkpoint = synthetic.ROOT / "development" / spec["name"] / "checkpoint-1500000.pt"
        source = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(source["model"])
        del source
        model.repeats = repeats
        digest = diagnosis.model_digest(model)
        diagnosis.write_json(
            out / "run.json",
            {
                "spec": {
                    "dataset": "synthetic",
                    "repeats": repeats,
                    "scientific_training_updates": 0,
                },
                "pid": os.getpid(),
                "gpu": gpu,
                "world_sha256": metadata["world_sha256"],
                "initial_model_sha256": digest,
                "tracking_group": "realworld-loop-diagnosis-v1",
                "job_type": "depth-evaluation",
            },
        )
        diagnosis.write_json(out / "learning.json", [])
        metrics = {}
        for name in ["atomic_id", "atomic_ood", "test_ii", "test_oo"]:
            metrics[name], pred = synthetic.evaluate_rows(model, panels[name], metadata, device)
            np.savez_compressed(out / f"{name}.npz", **pred)
        assert diagnosis.model_digest(model) == digest
        diagnosis.write_json(out / "learning.json", [{"step": 0, "metrics": metrics}])
        diagnosis.write_json(out / "complete.json", {"passed": True, "unchanged_parameters": True})
        print(json.dumps({"name": out.name, "metrics": metrics}), flush=True)
        del model
        torch.cuda.empty_cache()
    diagnosis.write_json(ROOT / "controller-state.json", {"state": "complete"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int)
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    diagnosis.write_json(
        ROOT / "controller-state.json", {"state": "waiting_for_idle_gpu", "pid": os.getpid()}
    )
    chosen = args.gpu if args.gpu is not None else wait_gpu()
    diagnosis.write_json(
        ROOT / "controller-state.json", {"state": "running", "gpu": chosen, "pid": os.getpid()}
    )
    measure(chosen)
