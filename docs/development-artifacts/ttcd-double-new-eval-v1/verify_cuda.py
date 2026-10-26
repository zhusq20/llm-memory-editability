"""Development-only real-foundation two-fact numerical engineering check."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from llm_memory_editability import ttcd_continual_eval as original
from llm_memory_editability import ttcd_double_new_eval as supplement
from llm_memory_editability import ttcd_verified_base as base
from llm_memory_editability.grok_depth import write_json
from llm_memory_editability.ttcd_editing import load_fast, save_fast, tensor_digest

parser = argparse.ArgumentParser()
parser.add_argument("--base-config", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=False)
config = json.loads(args.base_config.read_text())
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
probe = torch.randn(32, 32, device="cuda", requires_grad=True)
probe.square().matmul(probe).mean().backward()
torch.cuda.synchronize()
assert torch.isfinite(probe.grad).all()
reader, world, _, _ = base.foundation_inputs(config, "cuda:0")
bank = original.case_bank(config)
development = [c for c in bank if c["split"] == "calibration"]
reserved = {c["atom"] for c in bank}
selected = None
for a in development:
    for b in development:
        if a["atom"] == b["atom"]:
            continue
        h, r, old_bridge = map(int, world.atoms[a["atom"]])
        bridge, r2, old_second = map(int, world.atoms[b["atom"]])
        ids = np.flatnonzero((world.first == a["atom"]) & (world.chains[:, 2] == r2))
        if len(ids) and bridge != old_bridge:
            selected = a, b, int(ids[0])
            break
    if selected:
        break
assert selected is not None
a, b, j = selected
facts = []
for case in (a, b):
    fact = copy.deepcopy(case)
    fact["old_tail"] = int(world.atoms[case["atom"], -1])
    fact["D"] = np.flatnonzero(
        (world.second == case["atom"]) & ~np.isin(world.first, list(reserved))
    )[:2].tolist()
    assert fact["D"]
    facts.append(fact)
facts[0]["new_tail"] = int(world.atoms[b["atom"], 0])
legal_atoms = [i for i in range(len(world.atoms)) if i not in reserved][:4]
legal_chains = np.flatnonzero(
    ~np.isin(world.first, list(reserved)) & ~np.isin(world.second, list(reserved))
)[:4].tolist()
panel = {
    "facts": facts,
    "orders": [[0, 1], [1, 0]],
    "U_atoms": legal_atoms,
    "U_chains": legal_chains,
    "pairs": [
        {
            "pair_id": "development-pair",
            "first": 0,
            "second": 1,
            "type": "development",
            "row": world.chains[j].tolist(),
            "first_only_target": facts[1]["old_tail"],
            "target": facts[1]["new_tail"],
        }
    ],
}
base_hash = tensor_digest(reader.backbone.state_dict())
checks = []
for direction, indices in enumerate(panel["orders"]):
    state = reader.empty_fast()
    for i in indices:
        state, _ = original.write_fact(reader, state, config, world, facts[i], "cuda:0")
    spec = {"indices": indices}
    observed = supplement.read_state(reader, state, world, panel, spec, "cuda:0")
    digest = save_fast(args.out / f"state-{direction}.pt", reader, state, spec)
    loaded, provenance = load_fast(args.out / f"state-{direction}.pt", reader)
    assert provenance == spec and tensor_digest(loaded) == digest
    replay = supplement.read_state(reader, loaded, world, panel, spec, "cuda:0")
    original._compare_nested(replay, observed, "reload")
    rebuilt = reader.empty_fast()
    for i in indices:
        rebuilt, _ = original.write_fact(reader, rebuilt, config, world, facts[i], "cuda:0")
    assert tensor_digest(rebuilt) == digest
    checks.append(
        {
            "direction": direction,
            "finite": True,
            "state_exact": True,
            "tokens_and_scores_exact": True,
        }
    )
assert tensor_digest(reader.backbone.state_dict()) == base_hash
result = {
    "passed": True,
    "checks": checks,
    "foundation_model_sha256": base_hash,
    "development_atoms": [a["atom"], b["atom"]],
    "formal_model_queries": 0,
    "scientific_training_updates": 0,
    "base_changed": False,
    "cuda_probe_backward": True,
    "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
}
write_json(args.out / "receipt.json", result)
print(json.dumps(result))
