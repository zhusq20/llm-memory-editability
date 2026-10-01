"""A paired four-cell test of which fact controls a composite answer.

The editor is unchanged. Derivative probes use answer contrasts, never a
composite target loss. Isolated directions preserve the OTHER direct query's
two a/c and b/c contrasts to first order, not its entire output distribution.
"""

from dataclasses import replace

import torch

from .bios_direction import Suffix
from .bios_direction_cross import cross_tasks

CELLS = ("aa", "ab", "ba", "bb")
LAYER = 4


def factorial_tasks(world, seed, device):
    """Rows select root targets; columns select personal targets; a/b stay fixed."""
    originals = cross_tasks(world, seed, device)
    result = []
    for pair_index in range(6):
        base = originals[2 * pair_index]
        for cell in CELLS:
            labels, tokens = base.labels.clone(), base.tokens.clone()
            root, actual = base.sets["E"]
            labels[root, 0] = base.metadata[cell[0]]
            labels[actual, 0] = base.metadata[cell[1]]
            labels[base.sets["D"], 0] = base.metadata[cell[0]]
            rows = torch.arange(len(tokens), device=tokens.device)
            tokens[rows, base.positions[:, 0] + 1] = labels[:, 0]
            metadata = dict(
                base.metadata,
                cell=cell,
                pair_index=pair_index,
                kind="coherent" if cell[0] == cell[1] else "independent",
                root_target=base.metadata[cell[0]],
                actual_target=base.metadata[cell[1]],
            )
            result.append(
                replace(
                    base,
                    name=base.name.rsplit("-", 1)[0] + "-" + cell,
                    labels=labels,
                    tokens=tokens,
                    metadata=metadata,
                )
            )
    return result


def check_reused_task(task, manifest):
    """Compare actual information, including teacher-forced EOS prefixes."""
    current = task.manifest()
    for key in ("sets", "labels", "old_labels", "tokens_sha256"):
        assert current[key] == manifest[key], key
    for key in ("world", "seed", "chain", "group", "person", "a", "b", "c", "original_ids"):
        assert current["metadata"][key] == manifest["metadata"][key], key


def make_probe(model, task):
    ids = task.sets["E"] + task.sets["D_focal"]
    positions = task.positions[ids, :1]
    tokens = task.tokens[ids, :5].clone()
    # No desired answer is present even at masked future positions.
    tokens.masked_fill_(torch.arange(5, device=tokens.device)[None] > positions, 0)
    return Suffix(model, LAYER, tokens, positions)


def values(probe, weight, metadata):
    logits = probe(weight[None])[0, :, 0]
    a, b, c = (metadata[k] for k in ("a", "b", "c"))
    contrasts = torch.stack((logits[:, a] - logits[:, c], logits[:, b] - logits[:, c]), -1)
    return logits, contrasts


def derivatives(probe, weight, metadata):
    w = weight.detach().clone().requires_grad_()
    logits, contrasts = values(probe, w, metadata)
    gradients = torch.stack(
        [
            torch.autograd.grad(contrasts[i, j], w, retain_graph=True)[0]
            for i in range(3)
            for j in range(2)
        ]
    ).reshape(3, 2, *w.shape)
    return logits.detach(), contrasts.detach(), gradients.detach()


def isolated_directions(gradients):
    """Minimum-norm direct-fact controls; D is excluded from construction.

    A unit control changes the chosen fact's (a-c,b-c) by (+.5,-.5),
    hence its a-b contrast by 1; the other fact has both contrasts fixed.
    """
    g = gradients[:2].flatten(0, 1).flatten(1).double()
    gram = g @ g.T
    eig, vec = torch.linalg.eigh(gram)
    keep = eig > eig[-1].clamp_min(1e-30) * 1e-10
    inverse = (vec * torch.where(keep, 1 / eig.clamp_min(1e-300), 0)[None]) @ vec.T
    targets = torch.tensor([[0.5, -0.5, 0, 0], [0, 0, 0.5, -0.5]], device=g.device)
    controls = targets.double() @ inverse @ g
    achieved = controls @ g.T
    gd = (gradients[2, 0] - gradients[2, 1]).flatten().double()
    coefficients = controls @ gd
    residual = gd - (gd @ g.T) @ inverse @ g
    summary = {
        "rank": int(keep.sum()),
        "spectrum": eig.cpu().tolist(),
        "control_residual": float((achieved - targets).abs().max()),
        "root_response": float(coefficients[0]),
        "actual_response": float(coefficients[1]),
        "unexplained_D_gradient_fraction": float(residual.norm() / gd.norm().clamp_min(1e-30)),
        "root_control_norm": float(controls[0].norm()),
        "actual_control_norm": float(controls[1].norm()),
    }
    return controls.reshape(2, *gradients.shape[-2:]).to(gradients.dtype), summary


@torch.no_grad()
def calibrate_controls(probe, weight, metadata, gradients, magnitudes):
    controls, summary = isolated_directions(gradients)
    baseline_logits, baseline = values(probe, weight, metadata)
    rows = []
    if summary["control_residual"] > 1e-5:
        return summary, rows
    for index, name in enumerate(("root", "actual")):
        for magnitude in magnitudes:
            for sign in (-1, 1):
                delta = sign * magnitude * controls[index]
                cap = min(1.0, float(1e-3 * weight.norm() / delta.norm().clamp_min(1e-30)))
                delta = delta * cap
                changed, after = values(probe, weight + delta, metadata)
                observed = after - baseline
                predicted = (gradients.double() * delta.double()).sum((-1, -2))
                other = 1 - index
                logp = baseline_logits.double().log_softmax(-1)
                new_logp = changed.double().log_softmax(-1)
                kl = (logp.exp() * (logp - new_logp)).sum(-1)
                rows.append(
                    dict(
                        control=name,
                        magnitude=magnitude,
                        sign=sign,
                        cap=cap,
                        actual_target_increment=sign * magnitude * cap,
                        delta_norm=float(delta.norm()),
                        predicted_ab=(predicted[:, 0] - predicted[:, 1]).cpu().tolist(),
                        observed_ab=(observed[:, 0] - observed[:, 1]).cpu().tolist(),
                        other_ac_bc_drift=float(observed[other].abs().max()),
                        other_output_kl=float(kl[other]),
                        direct_argmax_changes=int(
                            (changed[:2].argmax(-1) != baseline_logits[:2].argmax(-1)).sum()
                        ),
                    )
                )
    return summary, rows


def first_adam_update(model, task, config):
    """Exact first-step algebra of the frozen soft-functional Adam editor."""
    ids = task.sets["E"] + task.sets["R"]
    suffix = Suffix(model, LAYER, task.tokens[ids], task.positions[ids])
    w = model.blocks[LAYER].mlp.down.weight.detach().clone().requires_grad_()
    logits = suffix(w[None])[0]
    logp = logits.double().log_softmax(-1)
    labels = task.labels[ids]
    ce = -logp.gather(-1, labels[..., None])[..., 0].mean(-1)
    n = len(task.sets["E"])
    # The original-parent KL has exactly zero derivative at the parent.
    loss = ce[:n].mean() + ce[n:].mean()
    gradient = torch.autograd.grad(loss, w)[0]
    m, v = 0.1 * gradient, 0.001 * gradient.square()
    t = torch.ones((), device=w.device)
    raw = -(m / (1 - 0.9**t)) / ((v / (1 - 0.999**t)).sqrt() + 1e-8)
    return config["lr"] * raw


def factorial_effects(margins):
    aa, ab, ba, bb = (float(margins[cell]) for cell in CELLS)
    return dict(
        intercept=(aa + ab + ba + bb) / 4,
        root=(aa + ab - ba - bb) / 4,
        actual=(aa - ab + ba - bb) / 4,
        interaction=(aa - ab - ba + bb) / 4,
    )
