"""Secondary analysis: exactly replay saved mature updates and inspect competition."""

import argparse
import csv
import json

import numpy as np
import torch

from llm_memory_editability.bios_cross import make_cross_world
from llm_memory_editability.bios_model import CausalLM, ModelConfig
from llm_memory_editability.bios_path_minimal import CONFIG as MINIMAL_CONFIG
from llm_memory_editability.bios_path_minimal import verify
from llm_memory_editability.bios_path_transfer import (
    CONFIG,
    ROOT,
    cases,
    example_data,
    inner,
    sha,
    stamp,
    tensor_sha,
    verify_lock,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text())
    minimal = json.loads(MINIMAL_CONFIG.read_text())
    verify(minimal)
    verify_lock(config)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    records = []
    for world_id in config["worlds"]:
        world = make_cross_world(world_id)
        selected = cases(world)
        alternatives = {
            (case["chain"], case["group"]): case["targets"][0]
            for case in selected
            if case["kind"] == "conflict"
        }
        for seed in config["seeds"]:
            name = f"world-{world_id}-seed-{seed}-neither"
            parent = torch.load(
                ROOT / config["parent_root"] / name / "model-15360.pt",
                map_location=device,
                weights_only=False,
            )
            model = CausalLM(ModelConfig(**parent["config"])).to(device)
            model.load_state_dict(parent["model"])
            model.eval()
            for pname, parameter in model.named_parameters():
                parameter.requires_grad_(pname == config["target_parameter"])
            weight = model.blocks[config["target_layer"]].mlp.down.weight
            before_weight = weight.detach().clone()
            for case in selected:
                directory = ROOT / config["output"] / name / "step-15360" / case["name"]
                saved = json.loads((directory / "measurements.json").read_text())
                updates = [
                    row
                    for row in saved["updates"]
                    if row["probe"] == 1
                    and row["scale"] == "matched_norm"
                    and row["fraction"] == config["primary_fraction"]
                ]
                data = example_data(world, case, device)
                a, b, c = (
                    case["targets"][1],
                    alternatives[case["chain"], case["group"]],
                    case["old_targets"][1],
                )
                logits = model(data["prompts"][1:2], (data["lengths"][1:2] - 1)[:, None])[0, 0]
                margin = logits[a] - logits[b]
                gradient = torch.autograd.grad(margin, weight)[0]
                p_before = logits.detach().double().softmax(-1)[[a, b, c]].cpu().tolist()
                margin_before = margin.detach().item()
                with np.load(directory / "factors.npz") as factors:
                    for update in updates:
                        arm = update["arm"]
                        z = torch.tensor(factors[arm + "_z"], device=device)
                        delta = torch.tensor(factors[arm + "_delta"], device=device)
                        direction = torch.einsum("btd,btk->btdk", delta, z).sum(1)[0]
                        with torch.no_grad():
                            weight.copy_(before_weight - update["eta"] * direction)
                            if tensor_sha(weight) != update["weight_sha256"]:
                                raise ValueError(
                                    "Secondary outcome was not evaluated at the original update"
                                )
                            after = model(
                                data["prompts"][1:2], (data["lengths"][1:2] - 1)[:, None]
                            )[0, 0].double()
                            p_after = after.softmax(-1)[[a, b, c]].cpu().tolist()
                            margin_after = (after[a] - after[b]).item()
                            prediction = inner(gradient, weight - before_weight)
                            weight.copy_(before_weight)
                        records.append(
                            {
                                "world": world_id,
                                "seed": seed,
                                "case": case["name"],
                                "kind": case["kind"],
                                "arm": arm,
                                "margin_before": margin_before,
                                "margin_after": margin_after,
                                "margin_change": margin_after - margin_before,
                                "predicted_margin_change": prediction,
                                **{
                                    f"p_{label}_before": p_before[i]
                                    for i, label in enumerate("abc")
                                },
                                **{f"p_{label}_after": p_after[i] for i, label in enumerate("abc")},
                                "original_CE_change": update["observed_change"],
                                "weight_hash_verified": True,
                            }
                        )
    out = ROOT / minimal["artifacts"]
    with (out / "mature-margin-secondary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (out / "margin-audit.json").write_text(
        json.dumps(
            {
                "finished_at": stamp(),
                "updates": len(records),
                "all_original_weight_hashes_verified": True,
                "analysis_source_sha256": sha(__file__),
            },
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps({"replayed_updates": len(records), "all_original_weight_hashes_verified": True})
    )


if __name__ == "__main__":
    main()
