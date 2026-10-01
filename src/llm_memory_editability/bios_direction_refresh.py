"""V1 numerical audit: refresh curvature per trajectory and freeze idle counters.

This keeps the original randomized batched probe stream, finite objectives,
constraints and solver settings. Unused fresh probe rows are discarded rather
than changing another trajectory's sketch. Frozen v1 artifacts are untouched.
"""

import json
import time
from pathlib import Path

import numpy as np
import torch

from .bios_data import write_json
from .bios_direction import (
    NODES,
    batch_setup,
    digest,
    evaluate_weights,
    gaussian_newton_sketch,
    inverse_sketch,
    quantities,
)


def refresh_mask(active, failed, attempt, refresh, missing):
    due = missing or attempt % refresh == 0
    return active & (torch.full_like(active, due) | (failed > 0))


def update_failure_counts(active, accepted, failed):
    return torch.where(active, torch.where(accepted, torch.zeros_like(failed), failed + 1), failed)


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
        refresh_rows = refresh_mask(active, failed, attempt, refresh, sketch is None)
        if solver == "gn" and refresh_rows.any():
            weights = ((ew + rw) / labels.shape[2])[..., None].expand_as(labels)
            if form == "soft" and obj == "func":
                weights = ((ew + (1 + level[:, None]) * rw) / labels.shape[2])[..., None].expand_as(
                    labels
                )
            fresh = gaussian_newton_sketch(logits, w, weights, rank, generator)
            if strict.any():
                projected = (fresh.reshape(n, rank, *w0.shape) @ null[:, None]).flatten(2)
                fresh = torch.where(strict[:, None, None], projected, fresh)
            sketch = (
                fresh if sketch is None else torch.where(refresh_rows[:, None, None], fresh, sketch)
            )
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
            failed = update_failure_counts(active, accepted, failed)
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
            failed=bool(failed[j] >= 10 and age[j] < steps),
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
