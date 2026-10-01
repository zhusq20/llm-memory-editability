#!/usr/bin/env python3
"""Measure zero-update batching and routing noise without changing frozen experiments."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np


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
    experiment = module.Experiment(args.layer, args.format, args.device)
    result_path = experiment.art / "zero-update-calibration.json"
    if result_path.exists():
        raise RuntimeError("Preserve existing zero-update calibration")
    experiment.evaluate(list(range(len(experiment.rows))))
    ne = experiment.cfg["counts"]["E"]
    records = []
    handle = experiment.mlp.register_forward_hook(experiment.route_hook)
    try:
        for i in range(ne):
            indices = [i, *range(ne, len(experiment.rows))]
            for route in ["all", "last", "earlier"]:
                experiment.route = route
                values = experiment.evaluate(indices)
                shift = values[:, 0] - experiment.base_margin[indices].numpy()
                row = dict(
                    case_id=experiment.rows[i]["case_id"],
                    route=route,
                    target_zero_shift=float(shift[0]),
                    roles={},
                )
                for role in ["R", "U", "W"]:
                    positions = [
                        j for j, k in enumerate(indices) if experiment.rows[k]["role"] == role
                    ]
                    row["roles"][role] = dict(
                        margin_changes=shift[positions].tolist(),
                        margin_rms=float(np.sqrt(np.mean(shift[positions] ** 2))),
                        margin_max=float(np.max(np.abs(shift[positions]))),
                        kl_mean=float(values[positions, 1].mean()),
                    )
                records.append(row)
    finally:
        handle.remove()
    assert experiment.local_hash() == experiment.original_hash
    module.write_json(
        result_path,
        dict(
            time=module.now(),
            rows=records,
            scope="Same evaluation batches/routes as interventions, zero parameter update",
            unchanged_parameter_hash=experiment.original_hash,
            source_sha256=module.digest(Path(__file__)),
        ),
    )
    print(result_path, flush=True)


if __name__ == "__main__":
    main()
