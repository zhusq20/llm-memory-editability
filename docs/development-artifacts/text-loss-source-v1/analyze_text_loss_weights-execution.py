"""Reciprocal weight interchange between paired A and N continuations."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from llm_memory_editability.grok_depth import write_json
from llm_memory_editability.text_pretrain import construct, evaluate, sha

GROUPS = ("token", "attention", "mlp", "norm_position")


def group_of(name):
    if name == "token.weight":
        return "token"
    if ".attention." in name:
        return "attention"
    if ".mlp." in name:
        return "mlp"
    return "norm_position"


@torch.inference_mode()
def analyze(config, parent, output, device):
    cfg = json.loads(Path(config).read_text())
    output = Path(output)
    if (output / "summary.json").exists():
        return
    output.mkdir(exist_ok=True, parents=True)
    names = {v["arm"]: k for k, v in cfg["runs"].items() if v["parent"] == parent}
    roots = {a: Path(cfg["output_root"]) / names[a] for a in ("A", "N")}
    weights = {
        a: torch.load(
            r / f"weights-{cfg['steps']:06d}.pt", map_location=device, weights_only=False
        )["model"]
        for a, r in roots.items()
    }
    spec = json.loads((Path(parent) / "spec.json").read_text())
    world = np.load(Path(parent) / "world.npz")
    model = construct(spec, device).eval()
    records = []
    raw = {}
    for recipient, donor in [("A", "N"), ("N", "A")]:
        for group in ("self", *GROUPS, "all"):
            state = {
                k: (weights[donor][k] if group == "all" or group_of(k) == group else v)
                for k, v in weights[recipient].items()
            }
            model.load_state_dict(state)
            for k, v in model.state_dict().items():
                assert torch.equal(v, state[k])
            metrics = {}
            for split in ("atomic", "background_train", "background_test", "target_test"):
                score, pred = evaluate(model, world[split], device, composite=split != "atomic")
                metrics[split] = score
                for k, v in pred.items():
                    raw[f"{recipient}_{group}_{split}_{k}"] = v
                if group in ("self", "all"):
                    expected = np.load(
                        roots[recipient if group == "self" else donor]
                        / f"predictions-{cfg['steps']:06d}.npz"
                    )
                    for key in ("answer", "stops", "target"):
                        np.testing.assert_array_equal(pred[key], expected[split + "_" + key])
                    assert (
                        np.max(np.abs(pred["probability"] - expected[split + "_probability"]))
                        < 1e-5
                    )
            records.append(dict(recipient=recipient, donor=donor, group=group, metrics=metrics))
    np.savez_compressed(output / "predictions.npz", **raw)
    write_json(
        output / "summary.json",
        dict(
            world=spec["world"],
            initialization=spec["initialization"],
            config=config,
            parent=parent,
            groups={g: [k for k in weights["A"] if group_of(k) == g] for g in GROUPS},
            checkpoints={a: sha(r / f"weights-{cfg['steps']:06d}.pt") for a, r in roots.items()},
            sources={__file__: sha(__file__)},
            records=records,
            claim=(
                "Conditional effects of transplanting paired training-induced "
                "parameter differences; "
                "not independent semantic storage modules."
            ),
        ),
    )
    print(
        json.dumps(
            [
                dict(
                    recipient=r["recipient"],
                    group=r["group"],
                    target=r["metrics"]["target_test"]["accuracy"],
                    atomic=r["metrics"]["atomic"]["accuracy"],
                )
                for r in records
            ]
        ),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--parent", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    torch.set_num_threads(2)
    torch.cuda.set_device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    analyze(args.config, args.parent, args.output, args.device)
