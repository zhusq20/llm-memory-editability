#!/usr/bin/env python3
"""Full-size, calibration-only two-fact write/save/reload engineering check."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from llm_memory_editability import ttcd_continual_eval as original  # noqa: E402
from llm_memory_editability import ttcd_double_new_eval as supplementary  # noqa: E402
from llm_memory_editability.grok_depth import write_json  # noqa: E402
from llm_memory_editability.ttcd_architecture import (  # noqa: E402
    architecture_accounting,
    build_engine,
)
from llm_memory_editability.ttcd_editing import load_fast, save_fast, tensor_digest  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if not config.get("engineering_only") or config["training"]["steps"] != 16:
        raise ValueError("This entrypoint is calibration-only engineering after16 updates")
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "receipt.json").exists():
        raise FileExistsError("Preserve engineering evidence")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    probe = torch.randn(32, 32, device="cuda:0", requires_grad=True)
    probe.square().matmul(probe).mean().backward()
    torch.cuda.synchronize()
    if not torch.isfinite(probe.grad).all():
        raise FloatingPointError("Real CUDA forward/backward failed")
    engine = build_engine(config)
    reader, world = engine.load_reader(config, "stream", "cuda:0")
    bank = original.case_bank(config)
    development = [case for case in bank if case["split"] == "calibration"]
    reserved = {case["atom"] for case in bank}
    selected = None
    for first in development:
        for second in development:
            if first["atom"] == second["atom"]:
                continue
            bridge, relation, _ = map(int, world.atoms[second["atom"]])
            choices = np.flatnonzero(
                (world.first == first["atom"]) & (world.chains[:, 2] == relation)
            )
            if len(choices) and bridge != int(world.atoms[first["atom"], -1]):
                selected = first, second, int(choices[0])
                break
        if selected:
            break
    if selected is None:
        raise ValueError("No graph-only development pair exists")
    first, second, query = selected
    facts = []
    for case in (first, second):
        fact = copy.deepcopy(case)
        fact["old_tail"] = int(world.atoms[case["atom"], -1])
        fact["D"] = np.flatnonzero(
            (world.second == case["atom"]) & ~np.isin(world.first, list(reserved))
        )[:2].tolist()
        facts.append(fact)
    facts[0]["new_tail"] = int(world.atoms[second["atom"], 0])
    panel = {
        "facts": facts,
        "orders": [[0, 1], [1, 0]],
        "U_atoms": [i for i in range(len(world.atoms)) if i not in reserved][:4],
        "U_chains": np.flatnonzero(
            ~np.isin(world.first, list(reserved)) & ~np.isin(world.second, list(reserved))
        )[:4].tolist(),
        "pairs": [
            {
                "pair_id": "development-pair",
                "first": 0,
                "second": 1,
                "type": "development",
                "row": world.chains[query].tolist(),
                "first_only_target": facts[1]["old_tail"],
                "target": facts[1]["new_tail"],
            }
        ],
    }
    before = tensor_digest(reader.backbone.state_dict())
    checks = []
    for order, indices in enumerate(panel["orders"]):
        fast = reader.empty_fast()
        for i in indices:
            fast, _ = original.write_fact(reader, fast, config, world, facts[i], "cuda:0")
        observed = supplementary.read_state(
            reader, fast, world, panel, {"indices": indices}, "cuda:0"
        )
        path = args.out / f"state-{order}.pt"
        digest = save_fast(path, reader, fast, {"indices": indices})
        fresh, fresh_world = engine.load_reader(config, "stream", "cuda:0")
        loaded, provenance = load_fast(path, fresh)
        if tensor_digest(loaded) != digest or provenance != {"indices": indices}:
            raise AssertionError("Fast state or provenance changed on reload")
        replay = supplementary.read_state(fresh, loaded, fresh_world, panel, provenance, "cuda:0")
        original._compare_nested(replay, observed, "fresh-reader-reload")
        rebuilt = fresh.empty_fast()
        for i in indices:
            rebuilt, _ = original.write_fact(
                fresh, rebuilt, config, fresh_world, facts[i], "cuda:0"
            )
        if tensor_digest(rebuilt) != digest:
            raise AssertionError("Independent write transition differs")
        checks.append({"order": indices, "state_exact": True, "tokens_and_scores_exact": True})
        del fresh, loaded, rebuilt
    if tensor_digest(reader.backbone.state_dict()) != before:
        raise AssertionError("Development writes changed the slow backbone")
    result = {
        "passed": True,
        "engineering_only": True,
        "checks": checks,
        "development_atoms": [first["atom"], second["atom"]],
        "formal_model_queries": 0,
        "scientific_training_updates": 0,
        "base_changed": False,
        "cuda_forward_backward_synchronized": True,
        "architecture_accounting": architecture_accounting(reader),
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
    }
    write_json(args.out / "receipt.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
