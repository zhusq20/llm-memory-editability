"""Pre-edit activation geometry and controlled mean-component interventions.

Company means are candidate shared components, not assumed learned mechanisms.
All donors, recipients and controls depend only on the old world and fixed seeds.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .bios_data import EOS, load_world, rng_for, write_json
from .bios_model import CausalLM, ModelConfig
from .bios_train import code_fingerprint, precision, tensor_data


@torch.no_grad()
def activation_summary(model, world, data):
    device = data["tokens"].device
    collected = [[] for _ in model.blocks]
    detailed = {}
    first_batch = [True]
    handles = []
    # Actual-city atomic queries have final prompt position 3.
    for layer, block in enumerate(model.blocks):

        def capture(_module, _inputs, output, layer=layer):
            collected[layer].append(output[:, 3].float().cpu())
            if first_batch[0]:
                detailed[f"z_positions_layer_{layer}"] = (
                    output[:128, [1, 2, 3]].float().cpu().numpy()
                )

        handles.append(block.mlp.activation.register_forward_hook(capture))

        def capture_mlp(_module, inputs, output, layer=layer):
            if first_batch[0]:
                detailed[f"x_positions_layer_{layer}"] = (
                    inputs[0][:128, [1, 2, 3]].float().cpu().numpy()
                )
                detailed[f"h_positions_layer_{layer}"] = (
                    output[:128, [1, 2, 3]].float().cpu().numpy()
                )

        handles.append(block.mlp.register_forward_hook(capture_mlp))
    with precision(device):
        for begin in range(2112, 4160, 256):
            model(
                data["prompts"][begin : begin + 256], data["lengths"][begin : begin + 256, None] - 1
            )
            first_batch[0] = False
    for handle in handles:
        handle.remove()
    activations = [torch.cat(parts) for parts in collected]
    summaries, means = [], []
    for layer, z in enumerate(activations):
        mu = torch.stack([z[world.employers == c].mean(0) for c in range(64)])
        centered = mu - z.mean(0)
        residual = z - mu[world.employers]
        shared_variance = centered.square().mean().item()
        individual_variance = residual.square().mean().item()
        cosine = F.cosine_similarity(z, mu[world.employers], dim=-1)
        summaries.append(
            {
                "layer": layer,
                "mean_norm": z.norm(dim=-1).mean().item(),
                "company_mean_variance": shared_variance,
                "within_company_variance": individual_variance,
                "shared_variance_fraction": shared_variance
                / (shared_variance + individual_variance + 1e-20),
                "member_to_company_cosine": cosine.mean().item(),
            }
        )
        means.append(mu)
    detailed["query_ids"] = np.arange(2112, 2240)
    detailed["token_positions"] = np.array([1, 2, 3])
    return summaries, means, activations, detailed


def probe_plan(world):
    rng = rng_for(world.seed, 51)
    recipients = rng.permutation(64)[:8]
    result = []
    for c in recipients:
        same = np.flatnonzero((world.defaults == world.defaults[c]) & (np.arange(64) != c))[0]
        donor = rng.choice(np.flatnonzero(world.defaults != world.defaults[c]))
        wrong = rng.choice(
            np.flatnonzero(~np.isin(world.defaults, [world.defaults[c], world.defaults[donor]]))
        )
        unrelated = rng.choice(np.flatnonzero(~np.isin(np.arange(64), [c, donor, same, wrong])))
        members = np.flatnonzero(world.employers == c)
        other = np.flatnonzero(world.employers == unrelated)
        ids = np.concatenate([2112 + members, 6208 + members, 2112 + other])
        # 6208 is the first birth-city row: employer 2048 + defaults 64 + actual/date 4096.
        group = np.concatenate(
            [np.where(world.exceptions[members], 1, 0), np.full(32, 2), np.full(32, 3)]
        )
        result.append(
            {
                "recipient": int(c),
                "donor": int(donor),
                "same_answer": int(same),
                "wrong_donor": int(wrong),
                "unrelated": int(unrelated),
                "ids": ids,
                "group": group,
            }
        )
    return result


@torch.no_grad()
def causal_probes(model, world, data, means, behavior=False):
    device = data["tokens"].device
    results = []
    for plan_index, plan in enumerate(probe_plan(world)):
        ids = torch.as_tensor(plan["ids"], device=device)
        tokens = data["prompts"][ids]
        positions = data["lengths"][ids, None] - 1
        donor_answer = int(world.city_tokens[world.defaults[plan["donor"]]])

        def observe(
            tokens=tokens, positions=positions, donor_answer=donor_answer, ids=ids, plan=plan
        ):
            with precision(device):
                logits = model(tokens, positions)[:, 0].float()
                probability = logits.softmax(-1)[:, donor_answer]
                if not behavior:
                    return probability, None
                first = logits.argmax(-1)
                continuation = torch.zeros((len(ids), 6), dtype=torch.long, device=device)
                continuation[:, :5] = tokens
                rows = torch.arange(len(ids), device=device)
                lengths = data["lengths"][ids]
                continuation[rows, lengths] = first
                ended = model(continuation, lengths[:, None])[:, 0].argmax(-1).eq(EOS)
                truth = torch.as_tensor(world.answers[plan["ids"]], device=device)
                return probability, {
                    "old_correct": (first.eq(truth) & ended).cpu().numpy(),
                    "donor_generated": (first.eq(donor_answer) & ended).cpu().numpy(),
                    "ended": ended.cpu().numpy(),
                }

        baseline, baseline_behavior = observe()
        for layer, mu_cpu in enumerate(means):
            mu = mu_cpu.to(device)
            c = plan["recipient"]
            desired = mu[plan["donor"]] - mu[c]
            generator = torch.Generator(device=device).manual_seed(
                world.seed * 10000 + 100 * plan_index + layer
            )
            random = torch.randn(desired.shape, generator=generator, device=device)
            random = random * (desired.norm() / random.norm().clamp_min(1e-20))
            shifts = {
                "donor": desired,
                "same_answer_donor": mu[plan["same_answer"]] - mu[c],
                "wrong_donor": mu[plan["wrong_donor"]] - mu[c],
                "random_equal_norm": random,
                "ablation": -(mu[c] - mu.mean(0)),
            }
            for kind, shift in shifts.items():

                def patch(_module, _inputs, output, shift=shift):
                    edited = output.clone()
                    edited[:, 3] = edited[:, 3] + shift.to(output.dtype)
                    return edited

                handle = model.blocks[layer].mlp.activation.register_forward_hook(patch)
                try:
                    changed, changed_behavior = observe()
                finally:
                    handle.remove()
                delta = (changed - baseline).cpu().numpy()
                by_group = {
                    name: float(delta[plan["group"] == group].mean())
                    for group, name in enumerate(
                        (
                            "same_company_nonexception",
                            "old_exception",
                            "independent_attribute",
                            "unrelated_company",
                        )
                    )
                }
                result = {
                    "recipient": c,
                    "donor": plan["donor"],
                    "layer": layer,
                    "intervention": kind,
                    "shift_norm": shift.norm().item(),
                    **by_group,
                    "selective_shared_score": by_group["same_company_nonexception"]
                    - (by_group["independent_attribute"] + by_group["unrelated_company"]) / 2,
                }
                if behavior:
                    result["generation"] = {}
                    for group, name in enumerate(by_group):
                        mask = plan["group"] == group
                        known = mask & baseline_behavior["old_correct"]
                        result["generation"][name] = {
                            "n": int(mask.sum()),
                            "baseline_old_accuracy": float(
                                baseline_behavior["old_correct"][mask].mean()
                            ),
                            "changed_old_accuracy": float(
                                changed_behavior["old_correct"][mask].mean()
                            ),
                            "donor_generated_delta": float(
                                changed_behavior["donor_generated"][mask].mean()
                                - baseline_behavior["donor_generated"][mask].mean()
                            ),
                            "old_known": int(known.sum()),
                            "old_broken": int((known & ~changed_behavior["old_correct"]).sum()),
                            "nontermination_rate": float((~changed_behavior["ended"][mask]).mean()),
                        }
                results.append(result)
    return results


def run(args):
    started = time.perf_counter()
    torch.set_num_threads(4)
    device = torch.device(args.device)
    world = load_world(args.world)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = CausalLM(ModelConfig(**checkpoint["config"])).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    data = tensor_data(world, device)
    summary, means, activations, detailed = activation_summary(model, world, data)
    probes = causal_probes(model, world, data, means, behavior=args.behavior)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "actual-city-activations.npz",
        **{f"z_{i}": z.numpy() for i, z in enumerate(activations)},
    )
    np.savez_compressed(out / "input-activation-output-position-sample.npz", **detailed)
    write_json(
        out / "measurements.json",
        {
            "step": checkpoint["step"],
            "device": str(device),
            "precision": "BF16 autocast" if device.type == "cuda" else "FP32 CPU diagnostic",
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
            "measurement_seconds": time.perf_counter() - started,
            "torch_version": torch.__version__,
            "hardware": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
            "behavior": args.behavior,
            **code_fingerprint(),
            "geometry": summary,
            "probes": probes,
            "scope": (
                "candidate company-mean components at final query token; "
                "exploratory development measurement"
            ),
        },
    )
    print(
        json.dumps(
            {"event": "measurement_complete", "step": checkpoint["step"], "probes": len(probes)}
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--behavior", action="store_true", help="Also score complete generated answers"
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
