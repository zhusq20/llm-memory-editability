"""Target-complete direction experiments; batched independent down-only editors.

The randomized generalized Gauss--Newton sketch is a diagnostic solver, not
AlphaEdit/CrispEdit. Every finite hard-constraint proposal is checked against the
original parent. Training batches contain E/R only; evaluation has no gradients.
"""

import hashlib
import itertools
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .bios_data import write_json
from .bios_model import CausalLM, ModelConfig

ROOT = Path(__file__).resolve().parents[2]
NODES = (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
ARMS = tuple(
    "-".join(x) for x in itertools.product(("func", "repr"), ("soft", "hard"), ("adam", "gn"))
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stable_order(values, *key):
    return sorted(values, key=lambda i: hashlib.sha256(f"{key}:{int(i)}".encode()).hexdigest())


def load_parent(path, device):
    saved = torch.load(path, map_location=device, weights_only=False)
    model = CausalLM(ModelConfig(**saved["config"])).to(device).eval()
    model.load_state_dict(saved["model"])
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


@dataclass
class Task:
    name: str
    tokens: torch.Tensor
    positions: torch.Tensor
    labels: torch.Tensor
    old_labels: torch.Tensor
    sets: dict
    metadata: dict

    def manifest(self):
        return {
            "name": self.name,
            "sets": self.sets,
            "metadata": self.metadata,
            "labels": self.labels.cpu().tolist(),
            "old_labels": self.old_labels.cpu().tolist(),
            "tokens_sha256": hashlib.sha256(self.tokens.cpu().numpy().tobytes()).hexdigest(),
        }


def minimal_task(seed, person, kind, device, labels=None):
    people = torch.arange(32, device=device).repeat_interleave(3)
    relation = torch.arange(3, device=device).repeat(32)
    tokens = torch.stack(
        (torch.ones_like(people), 3 + people, 35 + relation, torch.full_like(people, 2)), -1
    )
    old = 38 + (people + (relation == 2)) % 3 if labels is None else labels.to(device)
    old = old[:, None]
    desired = old.clone()
    first, second = (0, 1) if seed % 2 == 0 else (1, 0)
    i, j, k = 3 * person + first, 3 * person + second, 3 * person + 2
    c = int(old[i, 0])
    assert int(old[j, 0]) == c
    a, b = 38 + (c - 38 + 1) % 3, 38 + (c - 38 + 2) % 3
    desired[i] = a if kind == "coherent" else b
    e, local = [i], [k]
    if kind != "selective":
        desired[j] = a
        e.append(j)
    else:
        local.append(j)
    remaining = stable_order([x for x in range(32) if x != person], "split", person, 271520)

    def expand(xs):
        return [3 * p + r for p in xs for r in range(3)]

    sets = {
        "E": e,
        "R": local + expand(remaining[:15]),
        "V": expand(remaining[15:23]),
        "U": expand(remaining[23:]),
        "local": local,
    }
    return Task(
        f"seed-{seed}-person-{person}-{kind}",
        tokens,
        torch.full((96, 1), 3, device=device),
        desired,
        old,
        sets,
        {"seed": seed, "person": person, "kind": kind, "a": a, "b": b, "c": c, "pair": [i, j]},
    )


class Suffix:
    """Frozen prefix cached for independent weight matrices (batch axis zero)."""

    def __init__(self, model, layer, tokens, positions):
        self.model, self.layer, self.positions = model, layer, positions
        if tokens.ndim == 2:
            tokens = tokens[None]
            self.positions = positions[None]
        self.n, self.q, self.t = tokens.shape
        with torch.no_grad():
            flat = tokens.reshape(-1, self.t)
            x = model.token(flat) + model.position(torch.arange(self.t, device=flat.device))
            for block in model.blocks[:layer]:
                x = block(x)
            block = model.blocks[layer]
            x = x + block.attention(block.ln1(x))
            self.base = (x + block.mlp.down.bias).reshape(self.n, self.q, self.t, -1).detach()
            self.z = (
                block.mlp.activation(block.mlp.up(block.ln2(x)))
                .reshape(self.n, self.q, self.t, -1)
                .detach()
            )
        self.forward_calls = 0

    def __call__(self, w):
        self.forward_calls += 1
        n = w.shape[0]
        x = self.base + torch.einsum("nqtk,ndk->nqtd", self.z.expand(n, -1, -1, -1), w)
        x = x.reshape(n * self.q, self.t, -1)
        for block in self.model.blocks[self.layer + 1 :]:
            x = block(x)
        x = self.model.ln_final(x).reshape(n, self.q, self.t, -1)
        positions = self.positions.expand(n, -1, -1)
        x = x.gather(2, positions[..., None].expand(-1, -1, -1, x.shape[-1]))
        return F.linear(x, self.model.token.weight)


def candidate_configs(arm):
    obj, form, solver = arm.split("-")
    rates = (1e-4, 1e-3, 1e-2) if solver == "adam" else (0.1, 0.3, 1.0)
    levels = (
        (0.0, 1.0, 10.0, 100.0)
        if form == "soft"
        else ((0.0, 1e-6, 1e-4, 1e-2) if obj == "repr" else (1e-6, 1e-4, 1e-2, 0.1))
    )
    return [
        dict(lr=lr, level=level, damping=0.01, rank=8, refresh=16, trust_fraction=0.05)
        for lr, level in itertools.product(rates, levels)
    ]


def gaussian_newton_sketch(logits, w, weights, rank, generator):
    """E[U^T U] = weighted CE GGN; independent probes per example/logit.

    B = diag(sqrt(p)) - p sqrt(p)^T satisfies BB^T = diag(p)-pp^T.
    Frozen probes are differentiated through logits only, never through p.
    """
    probability = logits.detach().softmax(-1)
    result = []
    for _ in range(rank):
        noise = torch.randn(probability.shape, device=w.device, dtype=w.dtype, generator=generator)
        raw = probability.sqrt() * noise
        probe = raw - probability * raw.sum(-1, keepdim=True)
        probe *= weights.sqrt()[..., None] / np.sqrt(rank)
        result.append(
            torch.autograd.grad(logits, w, probe, retain_graph=True)[0].detach().flatten(1)
        )
    return torch.stack(result, 1)


def inverse_sketch(v, sketch, damping):
    """(damping I + U^T U)^-1 v via an exact rank-r dual solve."""
    shape = v.shape
    flat = v.flatten(1)
    if sketch is None:
        return v / damping
    k = sketch @ sketch.transpose(-1, -2)
    k.diagonal(dim1=-2, dim2=-1).add_(damping)
    coeff = torch.linalg.solve(k.double(), (sketch @ flat[..., None]).double()).to(v.dtype)
    return ((flat - (sketch.transpose(-1, -2) @ coeff)[..., 0]) / damping).reshape(shape)


def representation_geometry(suffix, rmask, w0):
    z = suffix.z
    mask = rmask[..., None].expand(-1, -1, z.shape[2])
    # Fixed sequence length; in cross tasks only positions <= last supervised index.
    valid = (
        torch.arange(z.shape[2], device=z.device)[None, None, :]
        <= suffix.positions.max(-1).values[..., None]
    )
    mask = mask * valid
    covariance = (
        torch.einsum("nqtk,nqtl,nqt->nkl", z.double(), z.double(), mask.double())
        / mask.sum((1, 2))[:, None, None]
    )
    output = torch.einsum("nqtk,dk->nqtd", z, w0) + suffix.model.blocks[suffix.layer].mlp.down.bias
    scale = (output.square().sum(-1) * mask).sum((1, 2)) / mask.sum((1, 2))
    covariance = covariance / scale[:, None, None].clamp_min(1e-12)
    suffix.repr_factor = (
        z * (mask / (mask.sum((1, 2)) * scale).clamp_min(1e-12)[:, None, None]).sqrt()[..., None]
    )
    vals, vecs = torch.linalg.eigh(covariance.double())
    threshold = vals[:, -1:] * 1e-10
    keep = vals > threshold
    null = (vecs * (~keep)[:, None, :]) @ vecs.transpose(-1, -2)
    return covariance.to(w0.dtype), null.to(w0.dtype), keep.sum(-1), vals


def batch_setup(model, layer, tasks, configs):
    ids = [t.sets["E"] + t.sets["R"] for t in tasks]
    assert len(set(map(len, ids))) == 1
    toks = torch.stack([t.tokens[i] for t, i in zip(tasks, ids, strict=True)])
    positions = torch.stack([t.positions[i] for t, i in zip(tasks, ids, strict=True)])
    labels = torch.stack([t.labels[i] for t, i in zip(tasks, ids, strict=True)])
    suffix = Suffix(model, layer, toks, positions)
    n, q, o = labels.shape
    ermask = torch.zeros((n, q), device=toks.device)
    for row, task in enumerate(tasks):
        ermask[row, : len(task.sets["E"])] = 1
    rmask = 1 - ermask
    eweight, rweight = ermask / ermask.sum(-1, keepdim=True), rmask / rmask.sum(-1, keepdim=True)
    w0 = model.blocks[layer].mlp.down.weight.detach().clone()
    with torch.no_grad():
        ref = suffix(w0[None].expand(n, -1, -1)).double().log_softmax(-1)
    covariance, null, ranks, spectrum = representation_geometry(suffix, rmask, w0)
    return suffix, labels, eweight, rweight, ref, covariance, null, ranks, spectrum, w0


def quantities(logits, w, w0, labels, eweight, rweight, ref, covariance, obj, repr_factor):
    logp = logits.double().log_softmax(-1)
    ce = -logp.gather(-1, labels[..., None])[..., 0].mean(-1)
    le, lr = (ce * eweight).sum(-1), (ce * rweight).sum(-1)
    if obj == "func":
        distance = ((ref.exp() * (ref - logp)).sum(-1).mean(-1) * rweight).sum(-1).clamp_min(0)
    else:
        delta = w - w0
        change = torch.einsum("nqtk,ndk->nqtd", repr_factor, delta)
        distance = change.double().square().sum((1, 2, 3))
    return le, lr, distance


@torch.no_grad()
def evaluate_weights(model, layer, tasks, weights):
    result = []
    # One run at a time keeps evaluation memory bounded for the 8-layer model.
    for task, w in zip(tasks, weights, strict=True):
        suffix = Suffix(model, layer, task.tokens, task.positions)
        logits = suffix(w[None])[0]
        parent_logits = suffix(model.blocks[layer].mlp.down.weight[None])[0]
        pred = logits.argmax(-1)
        old_pred = parent_logits.argmax(-1)
        correct = pred.eq(task.labels).all(-1)
        old_correct = old_pred.eq(task.old_labels).all(-1)
        if task.labels.shape[1] == 2:
            # EOS must be generated after the model's own answer, not the target.
            tokens = task.tokens.clone()
            lengths = task.positions[:, 0] + 1
            tokens[torch.arange(len(tokens), device=tokens.device), lengths] = pred[:, 0]
            generated = Suffix(model, layer, tokens, lengths[:, None])(w[None])[0, :, 0].argmax(-1)
            tokens[torch.arange(len(tokens), device=tokens.device), lengths] = old_pred[:, 0]
            old_generated = Suffix(model, layer, tokens, lengths[:, None])(
                model.blocks[layer].mlp.down.weight[None]
            )[0, :, 0].argmax(-1)
            correct = pred[:, 0].eq(task.labels[:, 0]) & generated.eq(3)
            old_correct = old_pred[:, 0].eq(task.old_labels[:, 0]) & old_generated.eq(3)
        nll = -logits.log_softmax(-1).gather(-1, task.labels[..., None])[..., 0].mean(-1)
        row = {"sets": {}, "prediction": pred[:, 0].cpu().tolist()}
        for key, ids in task.sets.items():
            indices = torch.tensor(ids, device=w.device, dtype=torch.long)
            known = old_correct[indices]
            row["sets"][key] = {
                "n": len(ids),
                "correct": int(correct[indices].sum()),
                "old_known": int(known.sum()),
                "broken": int((known & ~correct[indices]).sum()),
                "nll": float(nll[indices].mean()) if ids else None,
            }
        e, local, u = (row["sets"][x] for x in ("E", "local", "U"))
        row["joint"] = e["correct"] == e["n"] and local["broken"] == 0 and u["broken"] == 0
        row["e_joint"] = e["correct"] == e["n"]
        pair = task.metadata.get("pair", task.sets["E"][:2])
        a, b, c = (task.metadata[x] for x in ("a", "b", "c"))
        row["margins"] = {
            "ab": (logits[pair, 0, a] - logits[pair, 0, b]).cpu().tolist(),
            "ac": (logits[pair, 0, a] - logits[pair, 0, c]).cpu().tolist(),
            "bc": (logits[pair, 0, b] - logits[pair, 0, c]).cpu().tolist(),
        }
        result.append(row)
    return result


def edit_batch(model, layer, tasks, arm, configs, steps, directory, evaluate=True):
    """Each batch member is an independent run; no cross-run loss/normalization."""
    directory = Path(directory)
    if (directory / "complete.json").exists():
        return json.loads((directory / "metrics.json").read_text())
    directory.mkdir(parents=True, exist_ok=True)
    obj, form, solver = arm.split("-")
    setup = batch_setup(model, layer, tasks, configs)
    suffix, labels, ew, rw, ref, cov, null, ranks, spectrum, w0 = setup
    device = w0.device
    n = len(tasks)
    w = w0[None].expand(n, -1, -1).clone().requires_grad_()
    m, v = torch.zeros_like(w), torch.zeros_like(w)
    lr = torch.tensor([c["lr"] for c in configs], device=device)
    level = torch.tensor([c["level"] for c in configs], device=device)
    damping = configs[0]["damping"]
    rank, refresh = configs[0]["rank"], configs[0]["refresh"]
    strict = (level == 0) & (form == "hard") & (obj == "repr")
    age = torch.zeros(n, device=device, dtype=torch.long)
    failed = torch.zeros_like(age)
    active = torch.ones(n, device=device, dtype=torch.bool)
    sketch = None
    generator = torch.Generator(device=device).manual_seed(271521)
    records = [
        dict(task=t.manifest(), config=c, arm=arm, timeline=[])
        for t, c in zip(tasks, configs, strict=True)
    ]
    traces, checkpoints = [], {}
    pair_columns = torch.tensor(
        [[(t.sets["E"] + t.sets["R"]).index(i) for i in t.metadata["pair"]] for t in tasks],
        device=device,
    )
    answers_a = torch.tensor([t.metadata["a"] for t in tasks], device=device)
    answers_b = torch.tensor([t.metadata["b"] for t in tasks], device=device)
    rows = torch.arange(n, device=device)
    counters = dict(
        forward_batches=0,
        backward_batches=0,
        ggn_vjps=0,
        line_search_batches=0,
        covariance_eigh=1,
        train_examples_per_batch=labels.shape[1],
        batch_runs=n,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.monotonic()
    limit = 11 * steps + 10
    for attempt in range(limit + 1):
        if (
            attempt == 0
            or (attempt in NODES and attempt <= steps)
            or not active.any()
            or attempt == limit
        ):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            metrics = (
                evaluate_weights(model, layer, tasks, w.detach())
                if evaluate
                else [dict() for _ in tasks]
            )
            for j, metric in enumerate(metrics):
                records[j]["timeline"].append(
                    dict(
                        attempt=attempt,
                        accepted=int(age[j]),
                        seconds=time.monotonic() - start,
                        **metric,
                    )
                )
            checkpoints[str(attempt)] = w.detach().cpu().clone()
            if attempt and (not active.any() or attempt == limit):
                break
        if attempt >= steps and torch.all((age >= steps) | ~active):
            break
        logits = suffix(w)
        le, lretain, distance = quantities(
            logits, w, w0, labels, ew, rw, ref, cov, obj, suffix.repr_factor
        )
        pair_gradients = None
        if attempt in NODES:
            pair_margin = torch.stack(
                [
                    logits[rows, pair_columns[:, i], 0, answers_a]
                    - logits[rows, pair_columns[:, i], 0, answers_b]
                    for i in range(2)
                ],
                -1,
            )
            pair_gradients = torch.stack(
                [
                    torch.autograd.grad(pair_margin[:, i].sum(), w, retain_graph=True)[0].detach()
                    for i in range(2)
                ],
                1,
            )
            counters["backward_batches"] += 2
        data_loss = le + lretain
        objective = data_loss + level * distance if form == "soft" else data_loss
        if solver == "gn" and (sketch is None or attempt % refresh == 0 or failed.any()):
            weights = ((ew + rw) / labels.shape[2])[..., None].expand_as(labels)
            if form == "soft" and obj == "func":
                weights = ((ew + (1 + level[:, None]) * rw) / labels.shape[2])[..., None].expand_as(
                    labels
                )
            sketch = gaussian_newton_sketch(logits, w, weights, rank, generator)
            if strict.any():
                projected = (sketch.reshape(n, rank, *w0.shape) @ null[:, None]).flatten(2)
                sketch = torch.where(strict[:, None, None], projected, sketch)
            counters["ggn_vjps"] += rank
        g = torch.autograd.grad(objective.sum(), w, retain_graph=form == "hard")[0]
        counters["backward_batches"] += 1
        if form == "hard":
            a = torch.autograd.grad(distance.sum(), w)[0]
            counters["backward_batches"] += 1
        else:
            a = None
        if solver == "adam":
            m_new = 0.9 * m + 0.1 * g
            v_new = 0.999 * v + 0.001 * g.square()
            t = (age + 1).float()[:, None, None]
            raw = -(m_new / (1 - 0.9**t)) / ((v_new / (1 - 0.999**t)).sqrt() + 1e-8)
            proposal = lr[:, None, None] * raw
        else:
            if strict.any():
                g = torch.where(strict[:, None, None], g @ null, g)
            if obj == "repr" and form == "soft":
                # Exact representation Hessian plus damped sketched data GGN, CG.
                def hv(x, sketch=sketch):
                    flat = x.flatten(1)
                    return (
                        damping * x
                        + (sketch.transpose(-1, -2) @ (sketch @ flat[..., None]))[
                            ..., 0
                        ].reshape_as(x)
                        + 2 * level[:, None, None] * (x @ cov)
                    )

                direction = torch.zeros_like(g)
                residual, search = -g.clone(), -g.clone()
                rr = residual.square().sum((-1, -2))
                for _ in range(16):
                    hp = hv(search)
                    alpha = rr / (search * hp).sum((-1, -2)).clamp_min(1e-30)
                    direction += alpha[:, None, None] * search
                    residual -= alpha[:, None, None] * hp
                    new_rr = residual.square().sum((-1, -2))
                    search = residual + (new_rr / rr.clamp_min(1e-30))[:, None, None] * search
                    rr = new_rr
                proposal = lr[:, None, None] * direction
            else:
                proposal = -lr[:, None, None] * inverse_sketch(g, sketch, damping)
            # Fixed Euclidean trust radius, no hidden per-run gradient scaling.
            radius = configs[0]["trust_fraction"] * w0.norm()
            proposal *= (radius / proposal.flatten(1).norm(dim=1).clamp_min(1e-12)).clamp(max=1)[
                :, None, None
            ]
            m_new, v_new = m, v
        if form == "hard":
            if strict.any():
                proposal = torch.where(strict[:, None, None], proposal @ null, proposal)
            metric_a = a if solver == "adam" else inverse_sketch(a, sketch, damping)
            # Sequential quadratic subproblem with a linearized ORIGINAL distance.
            # Interior restoration avoids accumulating positive second-order drift.
            slack = torch.where(distance > 0.8 * level, -0.05 * level, 0.8 * (level - distance))
            excess = (a * proposal).sum((-1, -2)) - slack
            correction = excess.clamp_min(0) / (a * metric_a).sum((-1, -2)).clamp_min(1e-20)
            adjusted = proposal - correction[:, None, None] * metric_a
            proposal = torch.where(strict[:, None, None], proposal, adjusted).to(w.dtype)
        proposal = torch.where(active[:, None, None], proposal, torch.zeros_like(proposal))
        accepted = torch.zeros_like(active)
        chosen = w.detach().clone()
        chosen_logits = logits.detach().clone()
        picked_scale = torch.zeros(n, device=device)
        actual_d = distance.detach().clone()
        for backtrack in range(10):
            with torch.no_grad():
                candidate = w.detach() + (0.5**backtrack) * proposal
                if strict.any():
                    exact = w0 + (candidate - w0) @ null
                    candidate = torch.where(strict[:, None, None], exact, candidate)
                candidate_logits = suffix(candidate)
                cle, clr, cd = quantities(
                    candidate_logits,
                    candidate,
                    w0,
                    labels,
                    ew,
                    rw,
                    ref,
                    cov,
                    obj,
                    suffix.repr_factor,
                )
                co = cle + clr + level * cd if form == "soft" else cle + clr
                tolerance = torch.where(strict, torch.full_like(level, 1e-10), 1e-8 + 1e-5 * level)
                feasible = torch.ones_like(active) if form == "soft" else cd <= level + tolerance
                descent = co <= objective.detach() + 1e-7 if solver == "gn" else torch.isfinite(co)
                if form == "hard" and solver == "gn":
                    restoration = (
                        (distance.detach() > 0.8 * level)
                        & (cd < distance.detach() - 1e-4 * level)
                        & (co <= 1.05 * objective.detach() + 1e-6)
                        & ~strict
                    )
                    descent = descent | restoration
                good = feasible & descent & active & ~accepted & torch.isfinite(co)
                chosen[good] = candidate[good]
                chosen_logits[good] = candidate_logits[good]
                actual_d[good] = cd[good]
                picked_scale[good] = 0.5**backtrack
                accepted |= good
                counters["line_search_batches"] += 1
                if torch.all(accepted | ~active):
                    break
        delta = chosen - w.detach()
        if pair_gradients is not None:
            predicted = (pair_gradients * delta[:, None]).sum((-1, -2))
            updated_margin = torch.stack(
                [
                    chosen_logits[rows, pair_columns[:, i], 0, answers_a]
                    - chosen_logits[rows, pair_columns[:, i], 0, answers_b]
                    for i in range(2)
                ],
                -1,
            )
            observed = updated_margin - pair_margin.detach()
            flat = pair_gradients.flatten(2)
            kernel = flat @ flat.transpose(-1, -2)
            for j in range(n):
                records[j]["timeline"][-1]["local_step"] = {
                    "kernel": kernel[j].cpu().tolist(),
                    "predicted_margin_change": predicted[j].cpu().tolist(),
                    "observed_margin_change": observed[j].cpu().tolist(),
                    "actual_delta_norm": float(delta[j].norm()),
                }

        with torch.no_grad():
            w.copy_(chosen)
            m.copy_(torch.where(accepted[:, None, None], m_new, m))
            v.copy_(torch.where(accepted[:, None, None], v_new, v))
            age += accepted
            failed = torch.where(accepted, torch.zeros_like(failed), failed + 1)
            active = (age < steps) & (failed < 10)
        traces.append(
            torch.stack(
                (
                    le.detach(),
                    lretain.detach(),
                    distance.detach(),
                    actual_d,
                    delta.flatten(1).norm(dim=1),
                    picked_scale,
                    accepted.float(),
                    age.float(),
                ),
                -1,
            ).cpu()
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.monotonic() - start
    counters["forward_batches"] = suffix.forward_calls
    counters["elapsed_seconds_including_evaluation"] = elapsed
    for j, record in enumerate(records):
        record.update(
            accepted=int(age[j]),
            failed=bool(failed[j] >= 10),
            elapsed_batch_seconds=elapsed,
            representation_rank=int(ranks[j]),
            representation_spectrum=spectrum[j].cpu().tolist(),
        )
    write_json(directory / "metrics.json", records)
    torch.save(
        dict(
            weights=w.detach().cpu(),
            parent=w0.cpu(),
            checkpoints=checkpoints,
            m=m.cpu(),
            v=v.cpu(),
            age=age.cpu(),
            sketch=None if sketch is None else sketch.cpu(),
            generator_state=generator.get_state(),
            arm=arm,
            configs=configs,
        ),
        directory / "state.pt",
    )
    np.savez_compressed(
        directory / "steps.npz",
        values=torch.stack(traces).numpy(),
        columns=np.array(
            [
                "E_CE",
                "R_CE",
                "distance_before",
                "distance_after",
                "delta_norm",
                "backtrack_scale",
                "accepted",
                "accepted_count",
            ]
        ),
    )
    write_json(directory / "cost.json", counters)
    write_json(
        directory / "complete.json",
        dict(
            files={
                name: digest(directory / name)
                for name in ("metrics.json", "state.pt", "steps.npz", "cost.json")
            }
        ),
    )
    return records
