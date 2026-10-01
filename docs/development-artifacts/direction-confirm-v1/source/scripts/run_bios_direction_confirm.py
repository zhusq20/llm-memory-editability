"""Independent worlds: frozen predictors and paired learning intervention."""

import argparse
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from predict_bios_direction import features, predict

from llm_memory_editability.bios_data import write_json
from llm_memory_editability.bios_direction import ROOT, digest
from llm_memory_editability.bios_direction_formation import edit_parent, train

PRED = ROOT / "docs/development-artifacts/direction-prediction-v1/predictor.json"
ART = ROOT / "docs/development-artifacts/direction-confirm-v1"
OUT = ROOT / "results/bios-direction-confirm-v1"


def freeze():
    predictor = json.loads(PRED.read_text())
    files = [
        PRED,
        Path(__file__),
        ROOT / "scripts/predict_bios_direction.py",
        ROOT / "src/llm_memory_editability/bios_direction_formation.py",
        ROOT / "src/llm_memory_editability/bios_direction.py",
        ROOT / "src/llm_memory_editability/bios_model.py",
        ROOT / "docs/development-artifacts/direction-v1/selected.json",
    ]
    payload = dict(
        created=datetime.now(timezone.utc).isoformat(),
        worlds=predictor["independent_worlds"],
        seeds=[0, 1],
        train_arms=predictor["train_arms"],
        methods=predictor["methods"],
        training_trajectories=32,
        editing_trajectories=768,
        primary_endpoint="Final E/local/U joint success across task and training arms.",
        primary_comparisons=[
            ["func-hard-gn", "func-soft-adam"],
            ["func-soft-gn", "func-soft-adam"],
        ],
        statistics=(
            "Within-world average; paired world-level sign flips; "
            "Holm across two primary comparisons. All eight worlds required."
        ),
        predictor_endpoint="Brier and method-selection success by attempt 256; secondary.",
        causal_training_endpoint="Selected versus baseline, fixed AdamW; secondary.",
        files={str(p.relative_to(ROOT)): digest(p) for p in files},
    )
    if (ART / "lock.json").exists():
        assert json.loads((ART / "lock.json").read_text())["files"] == payload["files"]
        return
    write_json(ART / "lock.json", payload)
    (ART / "preregistration.md").write_text((ROOT / "docs/experimental-protocol.md").read_text())
    for p in files:
        target = ART / "source" / p.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(p.read_bytes())


def prediction_file(world, seed, arm, directory, predictor):
    learning = json.loads((directory / "learning.json").read_text())
    rows = []
    for person, kind, method in itertools.product(
        (0, 1, 2), ("coherent", "independent"), predictor["methods"]
    ):
        row = dict(world=world, seed=seed, train_arm=arm, person=person, kind=kind, method=method)
        for model_name in ("baseline", "mechanism"):
            x = features(learning, person, kind, method, model_name == "mechanism")
            row[model_name] = float(predict(predictor["models"][model_name], x[None])[0])
        rows.append(row)
    file = directory / "predictions-before-edit.json"
    if not file.exists():
        assert not (directory / "edits").exists(), "Predictions must precede all editing"
        write_json(
            file,
            dict(
                created=datetime.now(timezone.utc).isoformat(),
                parent_hash=digest(directory / "model-1024.pt"),
                predictor_hash=digest(PRED),
                rows=rows,
            ),
        )
    return rows


def run(worlds, device):
    lock = json.loads((ART / "lock.json").read_text())
    predictor = json.loads(PRED.read_text())
    for p, h in lock["files"].items():
        assert digest(ROOT / p) == h, p
    assert set(worlds) <= set(lock["worlds"])
    for world, seed, arm in itertools.product(worlds, (0, 1), lock["train_arms"]):
        directory = OUT / f"world-{world}-seed-{seed}" / arm
        train(world, seed, arm, device, directory)
        prediction_file(world, seed, arm, directory, predictor)
        edit_parent(world, seed, arm, device, directory)
        print(
            json.dumps(dict(event="complete-parent", world=world, seed=seed, arm=arm)), flush=True
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("freeze", "run"))
    parser.add_argument("--worlds", nargs="+", type=int)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.command == "freeze":
        freeze()
    else:
        run(args.worlds, torch.device(args.device))


if __name__ == "__main__":
    main()
