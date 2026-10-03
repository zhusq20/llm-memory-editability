"""Re-evaluate canonical checkpoints with their own generated bridge text."""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import diagnose_realworld_loop as common
import torch

ROOT = common.PROJECT / "results/realworld-loop-bridge-feedback-v1"


def main(arch, gpu):
    preceding = (
        common.PROJECT
        / "results/realworld-loop-entity-v1/development"
        / f"{arch}-natural"
        / "complete.json"
    )
    while True:
        memory = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            text=True,
        )
        used = {int(a): int(b) for a, b in (r.split(",") for r in memory.splitlines())}
        if preceding.exists() and used[gpu] < 100:
            break
        time.sleep(10)
    device = common.configure(gpu)
    source = common.ROOT / "development" / f"{arch}-canonical"
    assert json.loads((source / "audit.json").read_text())["passed"]
    data, _, _ = common.load_data("canonical")
    saved = json.loads((source / "endpoint-predictions.json").read_text())
    atoms = {r["id"]: r for r in saved["atomic"]}
    tokenizer = common.GPT2TokenizerFast.from_pretrained(
        common.BASE["tokenizer"], local_files_only=True
    )
    records = []
    for row in data["evaluation_compositions"]:
        bridge = atoms[row["atom_ids"][0]]["prediction"]
        question = common.canonical_question(bridge, [row["edges"][1][1]])
        encoded = common.encode_example(tokenizer, question, row["answer"])
        assert len(encoded["input"]) <= 256
        records.append({**row, "question": question, "encoded": encoded})
    model = common.construct(common.BASE["model"], common.spec_for(arch), device)
    model.load_state_dict(
        torch.load(source / "latest.pt", map_location="cpu", weights_only=False)["model"]
    )
    before = common.model_digest(model)
    metrics, predictions = common.evaluate(model, records, tokenizer, device)
    assert common.model_digest(model) == before
    by_id = {r["id"]: r for r in data["evaluation_compositions"]}
    for pred in predictions:
        pred["generated_bridge"] = atoms[by_id[pred["id"]]["atom_ids"][0]]["prediction"]
    grouped = {"test_all": metrics}
    for role in ["II", "IO", "OI", "OO"]:
        grouped["test_" + role.lower()] = common.aggregate(
            [r for r in predictions if r["role"] == role]
        )
    out = ROOT / "development" / arch
    common.write_json(
        out / "run.json",
        {
            "spec": {**common.spec_for(arch), "steps": 0},
            "pid": os.getpid(),
            "model": common.BASE["model"],
            "tracking_group": "realworld-loop-diagnosis-v1",
            "job_type": "bridge-feedback-evaluation",
            "source": str(source),
            "model_sha256": before,
            "source_predictions_sha256": common.sha256(source / "endpoint-predictions.json"),
            "code_sha256": common.sha256(Path(__file__)),
        },
    )
    common.write_json(out / "learning.json", [{"step": 0, "metrics": grouped}])
    common.write_json(out / "endpoint.json", grouped)
    common.write_json(out / "endpoint-predictions.json", predictions)
    common.write_json(
        out / "complete.json",
        {
            "passed": True,
            "utc": common.utc(),
            "model_unchanged": True,
            "uses_audited_first_call_predictions": True,
        },
    )
    print(json.dumps({k: v["alias_em"] for k, v in grouped.items()}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", choices=["standard8", "loop4x2"], required=True)
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()
    main(args.arch, args.gpu)
