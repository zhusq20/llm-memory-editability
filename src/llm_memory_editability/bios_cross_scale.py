"""Contracts and denominator-aware summaries for the frozen crossover size extension."""

from itertools import product

import numpy as np


def validate_size_study(reference, study, width, heads):
    expected = {**reference, "width": width, "heads": heads}
    if study != expected:
        changed = sorted(k for k in set(expected) | set(study) if expected.get(k) != study.get(k))
        raise ValueError(f"Non-size study changes: {changed}")
    if width // heads != 64 or study["layers"] != 8:
        raise ValueError("Expected eight layers and head dimension 64")


def validate_run_grid(configs, study, require_complete):
    expected = set(product(study["worlds"], study["seeds"], study["conditions"]))
    keys = [(c["world"], c["seed"], c["condition"]) for c in configs]
    if len(keys) != len(set(keys)) or set(keys) - expected:
        raise ValueError("Duplicate or unexpected experimental units")
    if require_complete and set(keys) != expected:
        raise ValueError("Missing experimental units")


def retention_coverage(pair, old_correct, row):
    """Keep unknown facts outside the damage denominator and expose their coverage."""
    result = {}
    for pool_name, pool in (
        ("full", np.arange(len(old_correct))),
        ("heldout", pair["heldout"]),
    ):
        totals, knowns, brokens = [], [], []
        for group in range(4):
            ids = np.intersect1d(pool, np.flatnonzero(pair["strata"] == group))
            prefix = f"U_{pool_name}_{group}"
            total, known, broken = len(ids), int(old_correct[ids].sum()), row[f"{prefix}_broken"]
            if known != row[f"{prefix}_known"] or not 0 <= broken <= known:
                raise ValueError("Retention denominator or broken count mismatch")
            totals.append(total)
            knowns.append(known)
            brokens.append(broken)
            result[f"{prefix}_total"] = total
            result[f"{prefix}_coverage"] = known / total if total else None
        total, known, broken = sum(totals), sum(knowns), sum(brokens)
        result[f"U_{pool_name}_total"] = total
        result[f"U_{pool_name}_coverage"] = known / total if total else None
        result[f"U_{pool_name}_micro_damage"] = broken / known if known else None
    result["E_old_accuracy"] = float(old_correct[pair["E"]].mean())
    result["D_old_accuracy"] = float(old_correct[pair["D"]].mean())
    return result
