"""Fixed-vocabulary random worlds for the capacity development experiment.

Knowledge is in independent A->B and B->C maps. Composition labels are derived,
never counted as independent information. This is a QA adaptation, not bioS.
"""

from __future__ import annotations

import hashlib
import math

import numpy as np

PAD, BOS, EOS, ANSWER = 0, 1, 2, 3
TYPE_A, TYPE_B, TYPE_C = 4, 5, 6
DIGIT, REL_A, REL_B = 7, 23, 27
VOCAB, CONTEXT, ANSWER_LENGTH = 31, 14, 6


def digest_arrays(arrays):
    h = hashlib.sha256()
    for key, value in sorted(arrays.items()):
        value = np.ascontiguousarray(value)
        h.update(key.encode())
        h.update(str(value.shape).encode())
        h.update(str(value.dtype).encode())
        h.update(value.tobytes())
    return h.hexdigest()


def generate_world(spec):
    """Rows: kind, subject, r1, r2, answer, necessary_atom_1, necessary_atom_2.

    The maximum address universe and all maps are generated before choosing a
    load. Prefix worlds are nested. Background controls keep the entire low-load
    composition training/evaluation sets, including necessary facts, unchanged.
    """
    seed, heads = spec["world_seed"], spec["heads_n"]
    maximum, values, relations = spec["max_heads"], spec["values_n"], spec["relations"]
    if not (1 <= heads <= maximum <= 65536 and values == 256 and relations == 4):
        raise ValueError("This version requires 256 values, four relations, <=65536 heads")
    support_heads = spec.get("support_heads_n", heads)
    if not 1 <= support_heads <= heads:
        raise ValueError("Invalid support prefix")

    def rng(part):
        return np.random.default_rng(np.random.SeedSequence([seed, part]))

    first = rng(0).integers(values, size=(maximum, relations), dtype=np.int64)[:heads]
    second = rng(1).integers(values, size=(values, relations), dtype=np.int64)
    eligible_first = rng(2).random((maximum, relations))[:support_heads] < 0.75
    eligible_second = rng(3).random((values, relations)) < 0.75
    first_n = heads * relations
    h, r = np.indices(first.shape).reshape(2, -1)
    atoms_a = np.column_stack(
        (
            np.zeros(first_n, dtype=int),
            h,
            r,
            -np.ones(first_n, dtype=int),
            first.ravel(),
            np.arange(first_n),
            -np.ones(first_n, dtype=int),
        )
    )
    b, r2 = np.indices(second.shape).reshape(2, -1)
    second_n = second.size
    atoms_b = np.column_stack(
        (
            np.ones(second_n, dtype=int),
            b,
            r2,
            -np.ones(second_n, dtype=int),
            second.ravel(),
            first_n + np.arange(second_n),
            -np.ones(second_n, dtype=int),
        )
    )
    atoms = np.concatenate((atoms_a, atoms_b)).astype(np.int64)
    h, r1, r2 = np.indices((support_heads, relations, relations)).reshape(3, -1)
    bridge = first[h, r1]
    a_id, b_id = h * relations + r1, first_n + bridge * relations + r2
    comps = np.column_stack((np.full(len(h), 2), h, r1, r2, second[bridge, r2], a_id, b_id)).astype(
        np.int64
    )
    train_mask = (
        eligible_first[h, r1] & eligible_second[bridge, r2] & (((r2 - h - r1) % relations) < 2)
    )
    trained = np.zeros(len(atoms), dtype=bool)
    trained[comps[train_mask, 5:].ravel()] = True
    seen1, seen2 = trained[a_id], trained[b_id]
    world = {
        "first": first,
        "second": second,
        "atomic": atoms,
        "train_composition": comps[train_mask],
    }
    for name, mask in {
        "II": seen1 & seen2,
        "IO": seen1 & ~seen2,
        "OI": ~seen1 & seen2,
        "OO": ~seen1 & ~seen2,
    }.items():
        ids = np.flatnonzero(mask & ~train_mask)
        # Selection stream does not depend on load in fixed-target controls.
        take = rng(20 + ["II", "IO", "OI", "OO"].index(name)).permutation(ids)
        world[name] = comps[take[: spec["evaluation_per_pool"]]]
        world[name + "_population"] = np.array([len(ids)], dtype=np.int64)
    panel_rng = rng(30)
    for name in ("atomic", "train_composition"):
        ids = panel_rng.permutation(len(world[name]))[: spec["evaluation_per_pool"]]
        world[name + "_panel"] = world[name][ids]
    world["role_counts"] = np.stack(
        [np.bincount(world["train_composition"][:, col], minlength=len(atoms)) for col in (5, 6)]
    )
    return world


def digits(values):
    values = np.asarray(values, dtype=np.int64)
    return DIGIT + ((values[:, None] >> np.array([12, 8, 4, 0])) & 15)


def pack(rows):
    """Right padding is causally downstream; only answer+EOS is supervised."""
    rows = np.asarray(rows, dtype=np.int64)
    x = np.zeros((len(rows), CONTEXT), dtype=np.int64)
    composition = rows[:, 0] == 2
    prefix_length = np.where(composition, 9, 8)
    x[:, 0] = BOS
    x[:, 1] = np.where(rows[:, 0] == 1, TYPE_B, TYPE_A)
    x[:, 2:6] = digits(rows[:, 1])
    x[:, 6] = np.where(rows[:, 0] == 1, REL_B, REL_A) + rows[:, 2]
    x[:, 7] = np.where(composition, REL_B + rows[:, 3], ANSWER)
    x[composition, 8] = ANSWER
    labels = np.column_stack(
        (np.where(rows[:, 0] == 0, TYPE_B, TYPE_C), digits(rows[:, 4]), np.full(len(rows), EOS))
    )
    positions = prefix_length[:, None] - 1 + np.arange(ANSWER_LENGTH)
    x[np.arange(len(rows))[:, None], positions[:, 1:]] = labels[:, :-1]
    return x, positions, labels


def information_ledger(world, parameters, knowledge_nll=None):
    facts = len(world["atomic"])
    bits = facts * math.log2(world["second"].shape[0])
    result = {
        "independent_facts": facts,
        "world_bits": bits,
        "parameters": parameters,
        "adjusted_parameters": parameters,
        "data_bits_per_parameter": bits / parameters,
        "composition_independent_bits": 0,
    }
    if knowledge_nll is not None:
        learned = bits - facts * knowledge_nll / math.log(2)
        result.update(learned_bits_raw=learned, learned_bits_per_parameter=learned / parameters)
    return result


def model_parameter_count(width, layers=2):
    # Tied token/output weights, learned positions, two biased projections each
    # in attention/MLP, per-block and final LayerNorms.
    return (VOCAB + CONTEXT) * width + layers * (12 * width**2 + 13 * width) + 2 * width


def prerequisite_pass(metrics, thresholds):
    return all(
        metrics[key]["accuracy"] >= thresholds[key] for key in ("atomic", "train_composition", "II")
    )


def grid_capacity(points, metric, threshold, atomic_threshold):
    """Report sampled successes and censoring, never silently assume monotonicity."""
    points = sorted(points, key=lambda p: p["bits"])
    passed = [p["atomic"] >= atomic_threshold and p[metric] >= threshold for p in points]
    reversals = any(passed[i] and not passed[i - 1] for i in range(1, len(passed)))
    successes = [p["bits"] for p, ok in zip(points, passed, strict=True) if ok]
    lower = max(successes) if successes else None
    failures_above = [
        p["bits"]
        for p, ok in zip(points, passed, strict=True)
        if not ok and (lower is None or p["bits"] > lower)
    ]
    return {
        "largest_observed_passing_bits": lower,
        "next_observed_failing_bits": min(failures_above) if failures_above else None,
        "nonmonotonic_grid": reversals,
        "interpretation": "sampled_grid_only; bracket requires monotone interpolation",
        "censoring": "below_grid"
        if lower is None
        else ("above_grid" if not failures_above else "bracketed_on_grid"),
    }
