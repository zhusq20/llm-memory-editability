"""Frozen-weight update deletion and semantic first-hop interventions.

Responses are measurements of computational dependence, not learned predictors.
Truth graphs select donors before inference; query means prevent donor frequency
from creating additional scientific observations.
"""

from __future__ import annotations

import time

import numpy as np
import torch
from torch.nn import functional as F

from .depth_step import EOS, prompt_rows, truth_path_details
from .latent_scaling import model_digest

COMPONENTS = ("full", "attention", "mlp")
TASKS = ("atomic",) + tuple(f"{s}_{h}" for h in (2, 3, 4) for s in ("familiar", "strict"))


@torch.no_grad()
def trace(model, tokens, repeats, *, skip=None, patch=None):
    """Apply one occurrence deletion or one r1 replacement, with native SDPA."""
    if model.training or repeats < 1:
        raise ValueError("Evaluation mode and positive recurrence are required")
    blocks = list(model.iter_blocks(repeats))
    if skip is not None:
        scope, source = skip
        if scope not in ("all", "r1") or source not in range(len(blocks)):
            raise ValueError("Unknown skip scope/occurrence")
    if patch is not None and patch[0] not in ("full", "mlp"):
        raise ValueError("Unknown patch component")
    x = model.token(tokens) + model.position(torch.arange(tokens.shape[1], device=tokens.device))
    cache = {name: [] for name in ("input", "full", "attention", "mlp", "output")}
    for index, block in enumerate(blocks):
        incoming = x
        z = block.ln1(x)
        batch, length, width = z.shape
        a = block.attention
        q, k, v = a.qkv(z).view(batch, length, 3, a.heads, width // a.heads).unbind(2)
        attention = a.proj(
            F.scaled_dot_product_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                is_causal=True,
                dropout_p=0.0,
            )
            .transpose(1, 2)
            .reshape(batch, length, width)
        )
        x = x + attention
        mlp = block.mlp(block.ln2(x))
        if index == 0 and patch is not None and patch[0] == "mlp":
            mlp = mlp.clone()
            if patch[1].shape != mlp[:, 2].shape:
                raise ValueError("Replacement shape differs")
            mlp[:, 2] = patch[1]
        x = x + mlp
        if index == 0 and patch is not None and patch[0] == "full":
            x = x.clone()
            if patch[1].shape != x[:, 2].shape:
                raise ValueError("Replacement shape differs")
            x[:, 2] = patch[1]
        if skip is not None and index == skip[1]:
            if skip[0] == "all":
                x, attention, mlp = incoming, torch.zeros_like(attention), torch.zeros_like(mlp)
            else:
                x, attention, mlp = x.clone(), attention.clone(), mlp.clone()
                x[:, 2], attention[:, 2], mlp[:, 2] = incoming[:, 2], 0, 0
        for name, value in zip(cache, (incoming, x - incoming, attention, mlp, x), strict=True):
            cache[name].append(value)
    logits = F.linear(model.ln_final(x), model.token.weight)
    return logits, {name: torch.stack(values) for name, values in cache.items()}


def interaction(numerator, denominator):
    """Ratio of query-averaged norms; an absent denominator is not zero influence."""
    numerator, denominator = np.asarray(numerator), np.asarray(denominator)
    if numerator.shape != denominator.shape or not len(numerator):
        raise ValueError("Require nonempty aligned response arrays")
    n, b = float(numerator.mean()), float(denominator.mean())
    return {"N": n, "B": b, "C": n / b if b > 1e-8 else None}


def prediction_metrics(prediction, targets):
    generated = prediction["generated"]
    logits = prediction["logits"].astype(np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    maximum = logits.max(1)
    partition = maximum + np.log(np.exp(logits - maximum[:, None]).sum(1))
    target_logits = logits[np.arange(len(targets)), targets]
    competitors = logits.copy()
    competitors[np.arange(len(targets)), targets] = -np.inf
    return {
        "answer": (generated[:, 0] == targets).astype(float),
        "complete": ((generated[:, 0] == targets) & (generated[:, 1] == EOS)).astype(float),
        "eos": (generated[:, 1] == EOS).astype(float),
        "nll": partition - target_logits,
        "margin": target_logits - competitors.max(1),
    }


def mean_metrics(metrics, mask=None):
    length = len(next(iter(metrics.values())))
    mask = np.ones(length, bool) if mask is None else np.asarray(mask, dtype=bool)
    if mask.shape != (length,):
        raise ValueError("Mask must align with observations")
    return {
        "n": int(mask.sum()),
        "coverage": float(mask.mean()) if length else 0.0,
        **{k: float(v[mask].mean()) if mask.any() else None for k, v in metrics.items()},
    }


def query_means(values, indices, n):
    counts = np.bincount(indices, minlength=n)
    sums = np.bincount(indices, weights=values, minlength=n)
    out = np.full(n, np.nan)
    np.divide(sums, counts, out=out, where=counts > 0)
    return out


def role_counts(world):
    counts = np.zeros((len(world["atomic"]), 4), dtype=np.int64)
    for hop in (2, 3, 4):
        _, edges = truth_path_details(world, world[f"train_{hop}"])
        for position in range(hop):
            np.add.at(counts[:, position], edges[:, position], 1)
    return counts


def select_donors(low, high):
    """Enumerate shared bridge donors and select answer-blind changed paths."""
    atoms = low["atomic"]
    for name in TASKS:
        np.testing.assert_array_equal(low[name], high[name])
    id_mask = np.asarray(low["metadata"]["id_mask"], bool)
    np.testing.assert_array_equal(id_mask, high["metadata"]["id_mask"])
    low_roles, high_roles = role_counts(low), role_counts(high)
    if low_roles[~id_mask].any() or high_roles[~id_mask].any():
        raise ValueError("Strict fact has composition training exposure")
    used_first = (low_roles[:, 0] > 0) & (high_roles[:, 0] > 0)
    strict = ~id_mask & (low_roles.sum(1) == 0) & (high_roles.sum(1) == 0)
    result = {"low_roles": low_roles, "high_roles": high_roles}
    rows = low["strict_2"]
    nodes, _ = truth_path_details(low, rows)
    for label, eligible in (("experienced", used_first), ("strict", strict)):
        recipients, indices = [], []
        for i, (row, path) in enumerate(zip(rows, nodes, strict=True)):
            mask = eligible & (atoms[:, 1] == row[1]) & (atoms[:, 2] == path[1])
            mask &= (atoms[:, 0] != row[0]) & (atoms[:, 0] != row[-1])
            choices = sorted(np.flatnonzero(mask), key=lambda k: tuple(atoms[k]))
            recipients.extend([i] * len(choices))
            indices.extend(choices)
        result[label + "_recipients"] = np.asarray(recipients, dtype=np.int64)
        result[label + "_atoms"] = np.asarray(indices, dtype=np.int64)
    result["common"] = (np.bincount(result["experienced_recipients"], minlength=len(rows)) > 0) & (
        np.bincount(result["strict_recipients"], minlength=len(rows)) > 0
    )
    lookup = {(int(h), int(r)): (i, int(t)) for i, (h, r, t) in enumerate(atoms)}
    trained = {tuple(row[:-1]) for w in (low, high) for row in w["train_2"]}
    order = sorted(range(len(atoms)), key=lambda i: tuple(atoms[i]))
    for label in ("familiar", "strict"):
        rows = low[label + "_2"]
        nodes, edges = truth_path_details(low, rows)
        recipients, donors, targets, successors = [], [], [], []
        for i, (row, path) in enumerate(zip(rows, nodes, strict=True)):
            for fact in order:
                h, r1, bridge = map(int, atoms[fact])
                successor = lookup.get((bridge, int(row[2])))
                if successor is None:
                    continue
                next_index, target = successor
                if (
                    r1 != row[1]
                    or h in (row[0], row[-1], target)
                    or bridge == path[1]
                    or target == row[-1]
                    or (h, r1, int(row[2])) in trained
                    or id_mask[fact] != id_mask[edges[i, 0]]
                    or id_mask[next_index] != id_mask[edges[i, 1]]
                ):
                    continue
                recipients.append(i)
                donors.append(fact)
                targets.append(target)
                successors.append(next_index)
                break
        result[label + "_changed_recipients"] = np.asarray(recipients, dtype=np.int64)
        result[label + "_changed_atoms"] = np.asarray(donors, dtype=np.int64)
        result[label + "_changed_targets"] = np.asarray(targets, dtype=np.int64)
        result[label + "_changed_successors"] = np.asarray(successors, dtype=np.int64)
    return result


@torch.no_grad()
def generate(model, rows, separator, repeats, *, skip=None, donor_atoms=None, component="full"):
    """Free answer and free EOS; a donor sees only the atomic question prefix."""
    rows = np.asarray(rows, np.int64)
    if not len(rows):
        return {
            "generated": np.empty((0, 2), np.int64),
            "logits": np.empty((0, model.config.vocab_size), np.float32),
        }, None
    device = next(model.parameters()).device
    tokens = torch.as_tensor(prompt_rows(rows, separator), device=device)

    def forward(inputs):
        patch = None
        if donor_atoms is not None:
            donor = torch.zeros_like(inputs)
            donor[:, :3] = torch.as_tensor(
                prompt_rows(donor_atoms, separator)[:, :3], device=device
            )
            _, donor_cache = trace(model, donor, 1)
            field = "output" if component == "full" else "mlp"
            patch = (component, donor_cache[field][0, :, 2])
        return trace(model, inputs, repeats, skip=skip, patch=patch)

    logits, cache = forward(tokens)
    answer_logits = logits[:, -1]
    answer = answer_logits.argmax(-1)
    eos_logits, _ = forward(torch.cat((tokens, answer[:, None]), dim=1))
    prediction = {
        "generated": torch.stack((answer, eos_logits[:, -1].argmax(-1)), 1).cpu().numpy(),
        "logits": answer_logits.cpu().numpy(),
    }
    return prediction, cache


def response_arrays(baseline, changed, source):
    arrays, rows = {}, []
    for component in COMPONENTS:
        for site, position in (("all", None), ("r1", 2), ("answer", -1)):
            a, b = baseline[component], changed[component]
            n, d = (b - a).norm(dim=-1), a.norm(dim=-1)
            n, d = (
                (n.mean(-1), d.mean(-1))
                if position is None
                else (n[..., position], d[..., position])
            )
            for target in range(source + 1, len(a)):
                numerator, denominator = n[target].cpu().numpy(), d[target].cpu().numpy()
                key = f"{component}_{site}_i{source}_j{target}"
                arrays[key + "_N"], arrays[key + "_B"] = numerator, denominator
                rows.append(
                    {
                        "component": component,
                        "site": site,
                        "source": source,
                        "target": target,
                        **interaction(numerator, denominator),
                    }
                )
    return arrays, rows


@torch.no_grad()
def evaluate(model, world, selection, repeats):
    started = time.perf_counter()
    if any(p.requires_grad for p in model.parameters()) or model.training:
        raise ValueError("All weights must be frozen in evaluation mode")
    before = model_digest(model)
    model.repeats = repeats
    separator = world["metadata"]["separator_token"]
    arrays, summaries, engineering = {}, {}, {}
    predictions, caches, metrics = {}, {}, {}
    conditions = [("baseline", None)] + [
        (f"skip_{scope}_{i}", (scope, i)) for scope in ("all", "r1") for i in range(repeats)
    ]
    for task in TASKS:
        rows = world[task]
        arrays[task + "_rows"] = rows
        predictions[task], metrics[task] = {}, {}
        for name, skip in conditions:
            pred, cache = generate(model, rows, separator, repeats, skip=skip)
            m = prediction_metrics(pred, rows[:, -1])
            predictions[task][name], metrics[task][name] = pred, m
            for field, values in {**pred, **m}.items():
                arrays[f"{task}_{name}_{field}"] = values
            summaries[f"{task}/{name}"] = {"full_pool": mean_metrics(m)}
            if name == "baseline":
                caches[task] = cache
                inputs = torch.as_tensor(
                    prompt_rows(rows, separator), device=next(model.parameters()).device
                )
                native = model(inputs)[:, -1].cpu().numpy()
                if not np.array_equal(native, pred["logits"]):
                    raise AssertionError("Traced logits differ from native forward")
                engineering[task + "_native_logits_exact"] = True
            elif task != "atomic":
                response, cells = response_arrays(caches[task], cache, skip[1])
                arrays.update({f"{task}_{name}_{k}": v for k, v in response.items()})
                summaries[f"{task}/{name}"]["responses"] = cells
            if name == f"skip_r1_{repeats - 1}":
                for field in ("generated", "logits"):
                    np.testing.assert_array_equal(pred[field], predictions[task]["baseline"][field])
                engineering[task + "_last_r1_no_downstream_effect"] = True
        del cache
    for task in TASKS[1:]:
        _, edges = truth_path_details(world, world[task])
        arrays[task + "_atomic_indices"] = edges
        common = np.ones(len(world[task]), bool)
        for name, _ in conditions:
            common &= metrics["atomic"][name]["complete"][edges].all(1)
        arrays[task + "_common_atomic_mask"] = common
        for name, _ in conditions:
            cell = summaries[f"{task}/{name}"]
            cell["necessary_atomic_coverage"] = float(
                metrics["atomic"][name]["complete"][edges].all(1).mean()
            )
            cell["common_atomic_pool"] = mean_metrics(metrics[task][name], common)
    # Self-prefix patches at equal physical length verify absence of later-relation information.
    for task in ("familiar_2", "strict_2"):
        rows = world[task]
        atoms = np.c_[rows[:, :2], np.zeros(len(rows), np.int64)]
        for component in ("full", "mlp"):
            pred, _ = generate(
                model, rows, separator, repeats, donor_atoms=atoms, component=component
            )
            for field in ("generated", "logits"):
                np.testing.assert_array_equal(pred[field], predictions[task]["baseline"][field])
            engineering[f"{task}_self_prefix_{component}_exact"] = True
    rows = world["strict_2"]
    _, edges = truth_path_details(world, rows)
    common = selection["common"]
    donor_atom_ok = np.ones(len(world["atomic"]), bool)
    for label in ("experienced", "strict"):
        recipients, facts = selection[label + "_recipients"], selection[label + "_atoms"]
        donor_atom_ok[facts] = metrics["atomic"]["baseline"]["complete"][facts].astype(bool)
        for component in ("full", "mlp"):
            if not len(recipients):
                continue
            pred, _ = generate(
                model,
                rows[recipients],
                separator,
                repeats,
                donor_atoms=world["atomic"][facts],
                component=component,
            )
            m = prediction_metrics(pred, rows[recipients, -1])
            key = f"same_{label}_{component}"
            for field, value in {**pred, **m}.items():
                arrays[key + "_" + field] = value
            balanced = {k: query_means(v, recipients, len(rows)) for k, v in m.items()}
            summaries[key] = {
                "common_graph_pool": mean_metrics(balanced, common),
                "all_available": mean_metrics(balanced, np.isfinite(balanced["complete"])),
            }
    known = metrics["atomic"]["baseline"]["complete"][edges].all(1)
    for label in ("experienced", "strict"):
        recipients, facts = selection[label + "_recipients"], selection[label + "_atoms"]
        for recipient, fact in zip(recipients, facts, strict=True):
            known[recipient] &= donor_atom_ok[fact]
    arrays["same_common_known_mask"] = known & common
    summaries["same_baseline"] = {
        "common_graph_pool": mean_metrics(metrics["strict_2"]["baseline"], common),
        "common_known_pool": mean_metrics(metrics["strict_2"]["baseline"], known & common),
    }
    for label in ("experienced", "strict"):
        recipients = selection[label + "_recipients"]
        for component in ("full", "mlp"):
            key = f"same_{label}_{component}"
            if key in summaries:
                m = {
                    k: arrays[key + "_" + k] for k in ("answer", "complete", "eos", "nll", "margin")
                }
                balanced = {k: query_means(v, recipients, len(rows)) for k, v in m.items()}
                summaries[key]["common_known_pool"] = mean_metrics(balanced, known & common)
    for label in ("familiar", "strict"):
        task = label + "_2"
        recipients = selection[label + "_changed_recipients"]
        facts = selection[label + "_changed_atoms"]
        targets = selection[label + "_changed_targets"]
        if not len(recipients):
            continue
        original = world[task][recipients]
        counterfactual = original.copy()
        counterfactual[:, 0] = world["atomic"][facts, 0]
        counterfactual[:, -1] = targets
        truth_path_details(world, counterfactual)
        native_cf, _ = generate(model, counterfactual, separator, repeats)
        values = {
            "baseline": {k: v[recipients] for k, v in predictions[task]["baseline"].items()},
            "counterfactual": native_cf,
        }
        for component in ("full", "mlp"):
            values[component], _ = generate(
                model,
                original,
                separator,
                repeats,
                donor_atoms=world["atomic"][facts],
                component=component,
            )
        for name, pred in values.items():
            key = f"changed_{label}_{name}"
            m = prediction_metrics(pred, targets)
            for field, vector in {**pred, **m}.items():
                arrays[key + "_" + field] = vector
            summaries[key] = mean_metrics(m)
    arrays.update({"selection_" + k: v for k, v in selection.items()})
    if model_digest(model) != before:
        raise AssertionError("Intervention changed model parameters")
    return arrays, {
        "repeats": repeats,
        "seconds": time.perf_counter() - started,
        "weights_unchanged": True,
        "model_sha256": before,
        "engineering": engineering,
        "conditions": summaries,
    }
