"""Prospective learning-rate interventions and factor measurements on new worlds."""

import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .bios_data import write_json
from .bios_direction import ROOT, digest, edit_batch, minimal_task
from .bios_direction_cross import CROSS_ARMS
from .bios_model import CausalLM, ModelConfig

TRAIN_NODES = (0, 8, 32, 128, 256, 512, 1024)
TRAIN_ARMS = ("baseline", "qk-0.25", "qk-4", "up-0.25", "up-4", "post-0.25", "post-4")


def world_labels(world):
    rng = np.random.default_rng(np.random.SeedSequence([271523, world]))
    r1 = np.r_[
        np.arange(3), rng.permutation(np.r_[np.repeat(0, 10), np.repeat(1, 10), np.repeat(2, 9)])
    ]
    exceptions = np.concatenate(
        [
            rng.choice(np.flatnonzero((r1 == v) & (np.arange(32) >= 3)), n, replace=False)
            for v, n in enumerate((3, 3, 2))
        ]
    )
    r2 = r1.copy()
    r2[exceptions] = np.roll(r1[exceptions], -3)
    r3 = rng.permutation(np.arange(32) % 3)
    labels = np.stack((r1, r2, r3), -1)
    assert (r1 == r2).sum() == 24 and np.all(r1[:3] == r2[:3])
    for col in labels.T:
        assert np.ptp(np.bincount(col, minlength=3)) <= 1
    return torch.tensor(labels.flatten() + 38, dtype=torch.long)


def optimizer_views(model):
    """Separate Q/K and V learning rates without changing forward computation.

    Optimizer leaf views share parameter storage; their gradients are assigned
    from the original model's autograd gradients. A frozen view gets grad=None,
    so its optimizer moments and step count genuinely stop.
    """
    groups = {k: [] for k in ("qk", "up", "post", "other")}
    mapping = []
    for name, p in model.named_parameters():
        if name.startswith("blocks.0.attention.qkv."):
            pieces = [("qk", slice(0, 128)), ("other", slice(128, None))]
        else:
            group = (
                "up"
                if name.startswith("blocks.0.mlp.up.")
                else ("post" if name.startswith("blocks.1.") else "other")
            )
            pieces = [(group, slice(None))]
        for group, region in pieces:
            view = nn.Parameter(p.detach()[region])
            groups[group].append(view)
            mapping.append((group, p, region, view))
    optimizer = torch.optim.AdamW(
        [dict(params=params, lr=0.001, name=name) for name, params in groups.items()],
        weight_decay=0.1,
        foreach=True,
    )
    return optimizer, mapping


def multiplier(arm, step):
    if arm == "baseline":
        return None, 1.0
    group, mode = arm.split("-")
    if mode == "freezeearly":
        return group, 0.0 if step < 256 else 1.0
    if mode == "freezelate":
        return group, 0.0 if step >= 768 else 1.0
    if mode == "freezeall":
        return group, 0.0
    return group, float(mode) if step < 256 else 1.0


def factor_measure(model, seed, labels, device):
    records = []
    for person in (0, 1, 2):
        task = minimal_task(seed, person, "independent", device, labels)
        ids = task.metadata["pair"]
        captured = {}

        def capture(module, inputs, output, captured=captured):
            captured.update(z=inputs[0], h=output)

        handle = model.blocks[0].mlp.down.register_forward_hook(capture)
        tokens = task.tokens[ids]
        logits = model(tokens)[:, -1]
        handle.remove()
        a, b, c = (task.metadata[x] for x in ("a", "b", "c"))
        margin = logits[:, a] - logits[:, b]
        gradients = []
        deltas = []
        for i in range(2):
            g, d = torch.autograd.grad(
                margin[i], (model.blocks[0].mlp.down.weight, captured["h"]), retain_graph=True
            )
            gradients.append(g.detach())
            deltas.append(d[i].detach())
        g = torch.stack(gradients).flatten(1)
        kernel = g @ g.T
        z = captured["z"].detach()
        d = torch.stack(deltas)
        rebuilt = torch.einsum("itd,itk->idk", d, z).flatten(1)
        error = (rebuilt - g).norm() / g.norm().clamp_min(1e-20)
        common = float((kernel[0, 0] + kernel[1, 1] + 2 * kernel[0, 1]) / 2)
        difference = float((kernel[0, 0] + kernel[1, 1] - 2 * kernel[0, 1]) / 2)
        # Actual GELU slope at the preactivation, not a binary activation mask.
        pre = model.blocks[0].mlp.up
        activations = {}

        def pre_hook(module, inputs, output, activations=activations):
            activations.update(input=inputs[0], pre=output)

        handle = pre.register_forward_hook(pre_hook)
        model(tokens)
        handle.remove()
        y = activations["pre"]
        gate = torch.autograd.grad(model.blocks[0].mlp.activation(y).sum(), y)[0].detach()

        def cosine(x):
            return float(F.cosine_similarity(x[0].flatten(), x[1].flatten(), dim=0))

        records.append(
            dict(
                person=person,
                common=common,
                difference=difference,
                ratio=difference / max(common, 1e-20),
                mixing=float((kernel[0, 0] - kernel[1, 1]) / 2),
                eigenvalues=torch.linalg.eigvalsh(kernel.double()).tolist(),
                feature_cos=cosine(z),
                downstream_cos=cosine(d),
                gate_cos=cosine(gate),
                relation_input_difference=float(
                    (activations["input"][0] - activations["input"][1]).detach().norm()
                ),
                feature_difference=float((z[0] - z[1]).norm()),
                gate_difference=float((gate[0] - gate[1]).norm()),
                reconstruction_error=float(error),
                new_target_nll=float(F.cross_entropy(logits, task.labels[ids, 0]).detach()),
                margin_ab=margin.detach().tolist(),
                kernel=kernel.detach().cpu().tolist(),
            )
        )
    return records


def train(world, seed, arm, device, out):
    out = Path(out)
    if (out / "complete.json").exists():
        return
    out.mkdir(parents=True, exist_ok=True)
    labels = world_labels(world).to(device)
    torch.manual_seed(seed)
    model = CausalLM(ModelConfig(41, width=64, layers=2, heads=2, context=8)).to(device)
    tokens = minimal_task(seed, 0, "independent", device, labels).tokens
    opt, mapping = optimizer_views(model)
    timeline = []
    for step in range(1025):
        if step in TRAIN_NODES:
            with torch.no_grad():
                logits = model(tokens)[:, -1]
                acc = float(logits.argmax(-1).eq(labels).float().mean())
                loss = float(F.cross_entropy(logits, labels))
            factors = factor_measure(model, seed, labels, device)
            timeline.append(dict(step=step, accuracy=acc, loss=loss, factors=factors))
            torch.save(
                dict(
                    model={k: v.detach().cpu() for k, v in model.state_dict().items()},
                    config=model.config_dict(),
                    optimizer=opt.state_dict(),
                    step=step,
                    world=world,
                    seed=seed,
                    arm=arm,
                    labels=labels.cpu(),
                ),
                out / f"model-{step}.pt",
            )
        if step == 1024:
            break
        model.zero_grad(set_to_none=True)
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(tokens)[:, -1], labels)
        loss.backward()
        selected, factor = multiplier(arm, step)
        for group in opt.param_groups:
            group["lr"] = 0.001 * (factor if group["name"] == selected else 1.0)
        for group, p, region, view in mapping:
            view.grad = None if group == selected and factor == 0 else p.grad[region]
        opt.step()
    write_json(out / "learning.json", timeline)
    write_json(
        out / "complete.json",
        dict(
            world=world,
            seed=seed,
            arm=arm,
            files={p.name: digest(p) for p in out.iterdir() if p.suffix in (".json", ".pt")},
        ),
    )


def edit_parent(world, seed, train_arm, device, out, edit_arms=CROSS_ARMS, steps=1024):
    from .bios_direction import load_parent

    out = Path(out)
    saved = torch.load(out / "model-1024.pt", map_location="cpu", weights_only=False)
    labels = saved["labels"].to(device)
    model = load_parent(out / "model-1024.pt", device)
    selected = json.loads(
        (ROOT / "docs/development-artifacts/direction-v1/selected.json").read_text()
    )["choices"]
    tasks = [
        minimal_task(seed, p, k, device, labels)
        for p in (0, 1, 2)
        for k in ("coherent", "independent")
    ]
    for task in tasks:
        task.metadata.update(world=world, train_arm=train_arm)
    for arm in edit_arms:
        edit_batch(
            model,
            0,
            tasks,
            arm,
            [copy.deepcopy(selected[arm]) for _ in tasks],
            steps,
            out / "edits" / arm,
        )
