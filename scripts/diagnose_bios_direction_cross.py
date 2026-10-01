"""Autonomous two-step and answer competition on the original composition task."""

import csv
import json

import torch

from llm_memory_editability.bios_cross import make_cross_world
from llm_memory_editability.bios_data import write_json
from llm_memory_editability.bios_direction import ROOT, load_parent


@torch.no_grad()
def answer(model, prompt):
    x = prompt[None]
    value = model(x)[:, -1].argmax(-1)
    ended = model(torch.cat((x, value[:, None]), -1))[:, -1].argmax(-1).eq(3)
    return int(value), bool(ended)


def main():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    rows = []
    for world_id in (0, 1):
        world = make_cross_world(world_id, ROOT / "data/bios-organization-v1")
        for seed in (0, 1):
            path = (
                ROOT
                / "results/bios-cross-scale-dev-v1/width-256"
                / f"world-{world_id}-seed-{seed}-neither/model-15360.pt"
            )
            model = load_parent(path, device)
            for directory in sorted(
                (
                    ROOT / f"results/bios-direction-cross-v1/main/world-{world_id}-seed-{seed}"
                ).iterdir()
            ):
                records = json.loads((directory / "metrics.json").read_text())
                state = torch.load(directory / "state.pt", map_location=device, weights_only=False)
                for index, r in enumerate(records):
                    with torch.no_grad():
                        model.blocks[4].mlp.down.weight.copy_(state["weights"][index])
                    meta = r["task"]["metadata"]
                    chain = meta["chain"]
                    person = meta["person"]
                    group = meta["group"]
                    membership = world.membership_ids[chain, person]
                    root = world.root_ids[chain, group]
                    first, first_eos = answer(
                        model, torch.tensor(world.prompts[membership, :4], device=device)
                    )
                    second_prompt = torch.tensor(world.prompts[root, :4], device=device)
                    second_prompt[1] = first
                    second, second_eos = answer(model, second_prompt)
                    oracle, oracle_eos = answer(
                        model, torch.tensor(world.prompts[root, :4], device=device)
                    )
                    focal = world.derived_ids[chain, person]
                    direct, direct_eos = answer(
                        model, torch.tensor(world.prompts[focal], device=device)
                    )
                    final = r["timeline"][-1]
                    assert (
                        int(direct == meta["a"] and direct_eos)
                        == final["sets"]["D_focal"]["correct"]
                    )
                    predicted = next(
                        (name for name in ("a", "b", "c") if direct == meta[name]), "other"
                    )
                    rows.append(
                        dict(
                            world=world_id,
                            seed=seed,
                            chain=chain,
                            group=group,
                            kind=meta["kind"],
                            arm=r["arm"],
                            E_joint=final["e_joint"],
                            local_broken=final["sets"]["local"]["broken"],
                            direct=int(direct == meta["a"] and direct_eos),
                            direct_value=predicted,
                            membership=int(first == world.answers[membership] and first_eos),
                            autonomous=int(second == meta["a"] and second_eos and first_eos),
                            intermediate_oracle=int(oracle == meta["a"] and oracle_eos),
                        )
                    )
    dest = ROOT / "docs/development-artifacts/direction-cross-v1"
    with (dest / "two-step.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    write_json(
        dest / "two-step-audit.json",
        dict(
            n=len(rows),
            all_direct_recomputed=True,
            autonomous_uses_model_generated_intermediate=True,
            oracle_uses_true_intermediate=True,
        ),
    )
    print("Two-step diagnostics:", len(rows))


if __name__ == "__main__":
    main()
