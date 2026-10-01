"""Separately frozen exact output-Jacobian calibration at all six toy parent pairs."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from llm_memory_editability.bios_data import write_json
from llm_memory_editability.bios_model import CausalLM, ModelConfig
from llm_memory_editability.bios_path_minimal import CONFIG, gradients, probe_data, verify
from llm_memory_editability.bios_path_transfer import ROOT, inner, norm, sha, stamp


def freeze(config, directory):
    path = directory / "jacobian-lock.json"
    if path.exists():
        raise FileExistsError("Cannot replace Jacobian prospective lock")
    protocol = (
        "### 18.8 " + (ROOT / "docs/experimental-protocol.md").read_text().split("### 18.8 ", 1)[1]
    )
    (directory / "jacobian-preregistration.md").write_text(protocol)
    sources = {str(Path(__file__).relative_to(ROOT)): sha(__file__)}
    for seed in config["seeds"]:
        parent = ROOT / config["output"] / f"seed-{seed}" / "parent.pt"
        sources[str(parent.relative_to(ROOT))] = sha(parent)
    write_json(
        path,
        {
            "created_at": stamp(),
            "sources": sources,
            "minimal_lock_sha256": verify(config),
            "protocol_sha256": sha(directory / "jacobian-preregistration.md"),
            "fractions": [0.0001, 0.0003, 0.001],
            "seeds": config["seeds"],
            "people": config["people_to_edit"],
            "arms": config["arms"],
        },
    )
    return {"lock": str(path), "sha256": sha(path)}


def run(config, directory, device):
    lock = json.loads((directory / "jacobian-lock.json").read_text())
    if verify(config) != lock["minimal_lock_sha256"]:
        raise ValueError("Minimal parent contract changed")
    for name, digest in lock["sources"].items():
        if sha(ROOT / name) != digest:
            raise ValueError("Frozen Jacobian source changed")
    if sha(directory / "jacobian-preregistration.md") != lock["protocol_sha256"]:
        raise ValueError("Jacobian protocol changed")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    summaries, updates, matrices = [], [], {}
    for seed in lock["seeds"]:
        saved = torch.load(
            ROOT / config["output"] / f"seed-{seed}" / "parent.pt",
            map_location=device,
            weights_only=False,
        )
        model = CausalLM(ModelConfig(**saved["config"])).to(device)
        model.load_state_dict(saved["model"])
        model.eval()
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name == config["target_parameter"])
        w = model.blocks[0].mlp.down.weight
        parent = w.detach().clone()
        for person in lock["people"]:
            tokens, labels, abc = probe_data(config, person, "conflict", device)
            a, b, c = abc
            logits = model(tokens[:2])[:, -1]
            p = logits.detach().double().softmax(-1)
            e = p.clone()
            e[0, b] -= 1
            e[1, a] -= 1
            jacobians = []
            for query in range(2):
                jacobians.append(
                    torch.stack(
                        [
                            torch.autograd.grad(logits[query, value], w, retain_graph=True)[
                                0
                            ].flatten()
                            for value in range(logits.shape[-1])
                        ]
                    ).double()
                )
            m = jacobians[1] @ jacobians[0].T
            kernel = (e[1] @ m @ e[0]).item()
            reference = gradients(model, tokens, labels, "full")
            direct = inner(reference["g"][1], reference["g"][0])
            scale = norm(reference["g"][1]) * norm(reference["g"][0])
            error = abs(kernel - direct) / max(scale, 1e-30)
            if error > 5e-5:
                raise ValueError("Exact Jacobian identity failed")
            pieces = [
                (p[1] @ m @ p[0]).item(),
                -(p[1] @ m[:, b]).item(),
                -(m[a] @ p[0]).item(),
                m[a, b].item(),
            ]
            if abs(sum(pieces) - kernel) > 1e-9 * max(scale, 1):
                raise ValueError("Four-term metric decomposition failed")
            zi, zj = reference["z"][0, -1], reference["z"][1, -1]
            di, dj = reference["delta"][0, -1], reference["delta"][1, -1]
            summaries.append(
                {
                    "seed": seed,
                    "person": person,
                    "euclidean_error_inner": inner(e[1], e[0]),
                    "jacobian_weighted_inner": kernel,
                    "direct_gradient_inner": direct,
                    "normalized_reconstruction_error": error,
                    "feature_cosine": inner(zi, zj) / (norm(zi) * norm(zj)),
                    "backward_cosine": inner(di, dj) / (norm(di) * norm(dj)),
                    "probability_probability_term": pieces[0],
                    "probability_target_b_term": pieces[1],
                    "target_a_probability_term": pieces[2],
                    "target_a_target_b_term": pieces[3],
                    "p_source_old": p[0, c].item(),
                    "p_derived_old": p[1, c].item(),
                }
            )
            prefix = f"seed{seed}_person{person}"
            matrices[prefix + "_metric"] = m.cpu().numpy()
            matrices[prefix + "_probabilities"] = p.cpu().numpy()
            for arm in lock["arms"]:
                source = (
                    reference if arm == "full" else gradients(model, tokens[:1], labels[:1], arm)
                )
                g = source["g"][0]
                for fraction in lock["fractions"]:
                    eta = fraction * norm(parent) / norm(g)
                    with torch.no_grad():
                        w.copy_(parent - eta * g)
                        delta = w - parent
                        after_logits = model(tokens)[:, -1].double()
                        after = F.cross_entropy(after_logits, labels, reduction="none")
                        predicted = inner(reference["g"][1], delta)
                        predicted_source = inner(reference["g"][0], delta)
                        w.copy_(parent)
                    updates.append(
                        {
                            "seed": seed,
                            "person": person,
                            "arm": arm,
                            "fraction": fraction,
                            "kernel": inner(reference["g"][1], g),
                            "predicted_derived_change": predicted,
                            "observed_derived_change": (after[1] - reference["loss"][1]).item(),
                            "predicted_source_change": predicted_source,
                            "observed_source_change": (after[0] - reference["loss"][0]).item(),
                        }
                    )
    for filename, rows in (
        ("jacobian-geometry.csv", summaries),
        ("jacobian-small-steps.csv", updates),
    ):
        with (directory / filename).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    np.savez_compressed(directory / "jacobian-metrics.npz", **matrices)
    result = {
        "finished_at": stamp(),
        "fact_pairs": len(summaries),
        "small_steps": len(updates),
        "max_normalized_identity_error": max(
            r["normalized_reconstruction_error"] for r in summaries
        ),
        "positive_euclidean_negative_true_pairs": sum(
            r["euclidean_error_inner"] > 0 and r["jacobian_weighted_inner"] < 0 for r in summaries
        ),
        "small_step_sign_agreement": sum(
            r["predicted_derived_change"] * r["observed_derived_change"] > 0 for r in updates
        )
        / len(updates),
        "source_improved_count": sum(r["observed_source_change"] < 0 for r in updates),
    }
    write_json(directory / "jacobian-audit.json", result)
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("freeze", "run"))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text())
    directory = ROOT / config["artifacts"]
    if args.command == "freeze":
        print(json.dumps(freeze(config, directory)))
    else:
        run(config, directory, torch.device(args.device))


if __name__ == "__main__":
    main()
