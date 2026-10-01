"""Check nonzero Adam-state restoration and captured/eager loss-source updates."""

import copy
import json
from pathlib import Path

import numpy as np
import torch

from llm_memory_editability.grok_depth import EpochStream, make_optimizer, write_json
from llm_memory_editability.text_loss_source import (
    SourceGraphStep,
    loss_sources,
    weighted_loss,
    weights_for,
)
from llm_memory_editability.text_pretrain import construct


def main():
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    root = Path("results/text-pretrain-v1/development/development-P1")
    parent = torch.load(root / "latest.pt", map_location="cuda:0", weights_only=False)
    spec = parent["spec"]
    world = np.load(root / "world.npz")
    stored = np.load(root / "training-table.npz")
    table = tuple(stored[k] for k in ("tokens", "positions", "mask", "labels"))
    bounds = np.cumsum(
        [0, len(world["atomic"]), len(world["background_train"]), len(world["target_train"])]
    )
    sources = loss_sources(table[3], bounds)
    indices = []
    streams = [EpochStream(bounds[i + 1] - bounds[i], 0) for i in range(3)]
    for stream, state in zip(streams, parent["streams"], strict=True):
        stream.load_state_dict(state)
    for _ in range(3):
        indices.append(
            torch.tensor(
                np.concatenate(
                    [
                        s.take(n) + lo
                        for s, n, lo in zip(streams, spec["batch_parts"], bounds[:-1], strict=True)
                    ]
                ),
                device="cuda:0",
            )
        )
    records = []
    for arm in ["AB", "A", "B", "N"]:
        models = []
        optimizers = []
        for _ in range(2):
            model = construct(spec, "cuda:0").train()
            model.load_state_dict(parent["model"])
            opt = make_optimizer(
                model, torch.tensor(spec["lr"], device="cuda:0"), spec["weight_decay"]
            )
            opt.load_state_dict(copy.deepcopy(parent["optimizer"]))
            models.append(model)
            optimizers.append(opt)
        gpu = tuple(
            torch.as_tensor(v, device="cuda:0") for v in (*table, weights_for(sources, arm))
        )
        torch.set_rng_state(parent["cpu_rng"].cpu())
        torch.cuda.set_rng_state(parent["cuda_rng"].cpu())
        graph = SourceGraphStep(models[0], optimizers[0], gpu, 128)
        assert all(torch.equal(v, parent["model"][k]) for k, v in models[0].state_dict().items())
        # Nonzero momentum and variance must survive capture initialization exactly.
        for a, b in zip(
            optimizers[0].state_dict()["state"].values(),
            parent["optimizer"]["state"].values(),
            strict=True,
        ):
            for key in a:
                assert torch.equal(a[key], b[key])
        for idx in indices:
            graph(idx)
        torch.cuda.synchronize()
        torch.set_rng_state(parent["cpu_rng"].cpu())
        torch.cuda.set_rng_state(parent["cuda_rng"].cpu())
        for idx in indices:
            opt = optimizers[1]
            opt.zero_grad(set_to_none=False)
            x, pos, mask, labels, weights = [v[idx] for v in gpu]
            logits = models[1](x, position_ids=pos, attention_mask=mask)
            loss = weighted_loss(logits, labels, weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(models[1].parameters(), 1.0, foreach=True)
            opt.step()
        errors = [
            float((a - b).abs().max())
            for a, b in zip(models[0].parameters(), models[1].parameters(), strict=True)
        ]
        assert max(errors) < 1e-6, (arm, max(errors))
        records.append(
            dict(
                arm=arm,
                capture_parent_and_optimizer_restored=True,
                three_step_max_parameter_error=max(errors),
            )
        )
    write_json(
        "docs/development-artifacts/text-loss-source-v1/preflight.json",
        dict(
            status="passed",
            records=records,
            torch=torch.__version__,
            gpu=torch.cuda.get_device_name(),
        ),
    )
    print(json.dumps(records))


if __name__ == "__main__":
    main()
