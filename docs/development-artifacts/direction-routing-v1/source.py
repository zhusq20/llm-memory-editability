"""Frozen inference-only attention-bias screen; never used as an editor baseline."""

import argparse
import contextlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from types import MethodType

import torch
from torch.nn import functional as F

from llm_memory_editability.bios_data import write_json
from llm_memory_editability.bios_direction import ROOT, digest, load_parent
from llm_memory_editability.bios_direction_cross import cross_tasks

ART = ROOT / "docs/development-artifacts/direction-routing-v1"
OUT = ROOT / "results/bios-direction-routing-v1"


@contextlib.contextmanager
def route_bias(model, layer, positions, target, strength):
    module = model.blocks[layer].attention
    original = module.forward

    def forward(self, x):
        n, t, width = x.shape
        q, k, v = self.qkv(x).reshape(n, t, 3, self.heads, width // self.heads).unbind(2)
        mask = torch.full((n, 1, t, t), -torch.inf, device=x.device, dtype=x.dtype)
        causal = (
            torch.arange(t, device=x.device)[:, None] >= torch.arange(t, device=x.device)[None, :]
        )
        mask.masked_fill_(causal, 0)
        source = positions - 1 if target == "relation" else torch.ones_like(positions)
        mask[torch.arange(n, device=x.device), 0, positions, source] += math.log(strength)
        h = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=mask
        )
        return self.proj(h.transpose(1, 2).reshape(n, t, width))

    module.forward = MethodType(forward, module)
    try:
        yield
    finally:
        module.forward = original


@torch.no_grad()
def evaluate(model, task, layer, target, strength):
    ids = task.sets["E"] + task.sets["local"] + task.sets["D_focal"]
    tokens = task.tokens[ids].clone()
    positions = task.positions[ids, 0]
    context = (
        contextlib.nullcontext()
        if layer < 0
        else route_bias(model, layer, positions, target, strength)
    )
    with context:
        logits = model(tokens[:, :5], positions[:, None])[:, 0]
        pred = logits.argmax(-1)
        tokens[torch.arange(len(ids), device=tokens.device), positions + 1] = pred
        ended = model(tokens, (positions + 1)[:, None])[:, 0].argmax(-1).eq(3)
    correct = pred.eq(task.labels[ids, 0]) & ended
    ne, nl = len(task.sets["E"]), len(task.sets["local"])
    return dict(
        E=int(correct[:ne].sum()),
        local=int(correct[ne : ne + nl].sum()),
        local_n=nl,
        D=int(correct[-1]),
        prediction=int(pred[-1]),
        joint=bool(correct.all()),
        margin_ab=float(logits[-1, task.metadata["a"]] - logits[-1, task.metadata["b"]]),
    )


def freeze():
    files = [
        Path(__file__),
        ROOT / "src/llm_memory_editability/bios_direction_cross.py",
        ROOT / "src/llm_memory_editability/bios_direction.py",
    ]
    write_json(
        ART / "lock.json",
        dict(
            created=datetime.now(timezone.utc).isoformat(),
            parent_editor="func-soft-adam",
            worlds=[0, 1],
            seeds=[0, 1],
            layers=list(range(8)),
            targets=["relation", "entity"],
            strengths=[0.25, 4.0],
            controls="Unmodified forward and a bias of 1; all edits and weights unchanged.",
            prediction=(
                "Underweighting the final relation predicts a selective D benefit from "
                "amplifying relation attention, compared with amplifying entity attention."
            ),
            selection=(
                "World 0: max D gains subject to no total E/local loss versus baseline. "
                "Test selected intervention on world 1. Retain all 32 interventions."
            ),
            status="Exploratory on existing worlds; no independent confirmation.",
            files={str(p.relative_to(ROOT)): digest(p) for p in files},
        ),
    )
    (ART / "source.py").write_bytes(Path(__file__).read_bytes())


def run(world, device):
    lock = json.loads((ART / "lock.json").read_text())
    for p, h in lock["files"].items():
        assert digest(ROOT / p) == h
    results = []
    for seed in (0, 1):
        parent = (
            ROOT
            / "results/bios-cross-scale-dev-v1/width-256"
            / f"world-{world}-seed-{seed}-neither/model-15360.pt"
        )
        model = load_parent(parent, device)
        tasks = cross_tasks(world, seed, device)
        d = ROOT / f"results/bios-direction-cross-v1/main/world-{world}-seed-{seed}/func-soft-adam"
        state = torch.load(d / "state.pt", map_location="cpu", weights_only=False)
        records = json.loads((d / "metrics.json").read_text())
        for i, task in enumerate(tasks):
            with torch.no_grad():
                model.blocks[4].mlp.down.weight.copy_(state["weights"][i].to(device))
            baseline = evaluate(model, task, -1, "relation", 1.0)
            assert baseline["D"] == records[i]["timeline"][-1]["sets"]["D_focal"]["correct"]
            sham = evaluate(model, task, 5, "relation", 1.0)
            assert all(sham[k] == baseline[k] for k in ("E", "local", "D", "prediction"))
            result = dict(metadata=task.metadata, baseline=baseline, conditions=[])
            for layer in range(8):
                for target in ("relation", "entity"):
                    for strength in (0.25, 4.0):
                        result["conditions"].append(
                            dict(
                                layer=layer,
                                target=target,
                                strength=strength,
                                **evaluate(model, task, layer, target, strength),
                            )
                        )
            results.append(result)
    write_json(OUT / f"world-{world}.json", results)
    print(json.dumps(dict(world=world, cases=len(results), conditions=32)), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("freeze", "run"))
    parser.add_argument("--world", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    if args.command == "freeze":
        freeze()
    else:
        run(args.world, torch.device(args.device))


if __name__ == "__main__":
    main()
