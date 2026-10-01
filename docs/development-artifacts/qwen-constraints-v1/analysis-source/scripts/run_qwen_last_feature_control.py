#!/usr/bin/env python3
"""Follow-up: preserving last-token features versus real all-position responses."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import torch

from llm_memory_editability.qwen_constraints import feature_basis, module_projection, update_scale


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--format", choices=["qa", "plain"], required=True)
    parser.add_argument("--device", required=True)
    args = parser.parse_args()
    source = Path(
        "docs/development-artifacts/qwen-constraints-v1/source/scripts/run_qwen_constraints.py"
    )
    spec = importlib.util.spec_from_file_location("frozen_constraints", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    exp = module.Experiment(args.layer, args.format, args.device)
    features = torch.load(exp.out / "activations.pt", weights_only=True)["features"]
    exp.out = module.RESULTS / "module-last" / f"layer-{args.layer}-{args.format}"
    exp.art = module.ART / "module-last" / f"layer-{args.layer}-{args.format}"
    exp.out.mkdir(parents=True, exist_ok=True)
    exp.art.mkdir(parents=True, exist_ok=True)
    if (exp.art / "predictions.json").exists():
        raise RuntimeError("Preserve completed follow-up predictions")
    ne, nr = exp.cfg["counts"]["E"], exp.cfg["counts"]["R"]
    exp.evaluate(list(range(len(exp.rows))))
    parameter = exp.parameters["down"]
    parameter.requires_grad_(True)
    gradients = torch.empty((ne + nr, parameter.numel()), device=exp.device)
    for i in range(ne + nr):
        ids, attention = exp.batch([i])
        h = exp.hidden(ids, attention)[0]
        readout = (
            exp.model.lm_head.weight[exp.rows[i]["target"]]
            - exp.model.lm_head.weight[exp.competitors[i]]
        )
        gradients[i] = torch.autograd.grad(h @ readout, parameter)[0].detach().flatten()
    parameter.requires_grad_(False)
    keys = torch.stack([v[-1] for v in features[ne:]]).to(exp.device)
    exp.keep_features = torch.cat(features[ne:]).to(exp.device)
    basis, singular = feature_basis(keys, exp.cfg["feature_svd_rtol"])
    directions, predictions, diagnostic = {}, [], []
    for i in range(ne):
        direction = module_projection(gradients[i].reshape_as(parameter), basis).flatten()
        directions[("down", i, "module_last")] = direction
        energy = float(direction.double().square().sum() / gradients[i].double().square().sum())
        residual = keys @ direction.reshape_as(parameter).T
        diagnostic.append(
            dict(
                case_id=exp.rows[i]["case_id"],
                rank=len(basis),
                retained_energy=energy,
                equal_target_cost_factor=1 / energy**0.5,
                relative_last_feature_leak=float(
                    residual.norm() / (keys.norm() * direction.norm())
                ),
            )
        )
        for metric in ["equal_norm", "equal_target"]:
            for step in exp.cfg["steps"]:
                scale, capped = update_scale(
                    direction,
                    gradients[i],
                    step,
                    float(exp.base["down"].norm()),
                    exp.cfg["relative_parameter_cap"],
                    metric,
                )
                delta = direction * scale
                predictions.append(
                    dict(
                        index=len(predictions),
                        case_id=exp.rows[i]["case_id"],
                        target_index=i,
                        group="down",
                        method="module_last",
                        metric=metric,
                        route="all",
                        step=step,
                        scale=scale,
                        capped=capped,
                        delta_norm=float(delta.norm()),
                        relative_delta_norm=float(delta.norm() / exp.base["down"].norm()),
                        predicted_target=float(gradients[i].double() @ delta.double()),
                        predicted_R=(gradients[ne:].double() @ delta.double()).tolist(),
                    )
                )
    module.write_json(
        exp.art / "geometry.json",
        dict(rows=diagnostic, feature_shape=list(keys.shape), singular_values=singular.tolist()),
    )
    module.write_json(
        exp.art / "predictions.json",
        dict(
            time=module.now(),
            rows=predictions,
            scope="Follow-up predictions saved before its finite interventions; no U/W gradients",
        ),
    )
    exp.intervene({"down": gradients}, directions, predictions)


if __name__ == "__main__":
    main()
