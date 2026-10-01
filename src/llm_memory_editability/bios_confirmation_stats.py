"""Locked world-level inference for the narrower shortcut confirmation.

This module consumes independently audited endpoint counts, never query outcomes
for selecting hypotheses. Nested seeds, organizations, chains and supports are
averaged within a world before any inferential calculation.
"""

import itertools
import math

DEFAULT_STATISTICS = {
    "revision": "shortcut-confirmation-statistics-v1",
    "scope": "training-correlation behavioral confirmation, not organization-path mediation",
    "worlds": list(range(100, 108)),
    "seeds": [0, 1],
    "conditions": ["company", "project", "neither"],
    "chains": ["company", "project"],
    "prevalences": ["low", "high"],
    "supports": [0, 1],
    "kinds": ["coherent", "exception"],
    "width": 256,
    "learning_step": 15360,
    "editing_step": 512,
    "learning_denominator": 64,
    "editing_denominator": 9,
    "primary_endpoint": "editing_exception_fixed9",
    "confirmatory_family": [
        "editing_exception_fixed9",
        "learning_original_exception",
        "editing_coherent_fixed9",
    ],
    "contrast": "high-minus-low",
    "unit": "world; all nested pairs equally weighted within each world",
    "test": "two-sided one-sample t on eight world-level paired differences, df=7",
    "assumption": "independent generated worlds; approximately normal world-level differences",
    "multiplicity": "Holm across the three confirmatory endpoints, family alpha=0.05",
    "intervals": ["95% marginal t", "98.33333333333333% Bonferroni simultaneous t"],
    "sensitivity": "all 256 world sign flips; valid under null sign symmetry, not an "
    "unconditional randomized-treatment test",
    "zero_sd": "all-zero differences: t=0,p=1,CI=[0,0], degenerate/no equivalence; "
    "constant nonzero differences: t/p/CI=null, Holm input=1, adjusted p=1",
    "stopping": "fixed eight worlds; no interim outcome inspection or optional extension",
    "publication": "all three endpoints, signs, all E/U/global-D/base checks published "
    "regardless of significance; no automatic natural-language or editor-method trigger",
}


def _beta_fraction(a, b, x):
    tiny, tolerance = 1e-300, 3e-14
    qab, qap, qam = a + b, a + 1, a - 1
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1 / (d if abs(d) >= tiny else tiny)
    h = d
    for m in range(1, 401):
        double = 2 * m
        aa = m * (b - m) * x / ((qam + double) * (a + double))
        d, c = 1 + aa * d, 1 + aa / c
        d = 1 / (d if abs(d) >= tiny else tiny)
        c = c if abs(c) >= tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + double) * (qap + double))
        d, c = 1 + aa * d, 1 + aa / c
        d = 1 / (d if abs(d) >= tiny else tiny)
        c = c if abs(c) >= tiny else tiny
        factor = d * c
        h *= factor
        if abs(factor - 1) <= tolerance:
            return h
    raise ArithmeticError("Incomplete beta continued fraction did not converge")


def regularized_beta(x, a, b):
    if not (0 <= x <= 1 and a > 0 and b > 0):
        raise ValueError("Invalid beta arguments")
    if x in (0, 1):
        return float(x)
    scale = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    )
    if x < (a + 1) / (a + b + 2):
        result = scale * _beta_fraction(a, b, x) / a
    else:
        result = 1 - scale * _beta_fraction(b, a, 1 - x) / b
    return min(1.0, max(0.0, result))


def t_two_sided_p(statistic, degrees):
    if degrees <= 0 or not math.isfinite(statistic):
        raise ValueError("Finite t and positive degrees of freedom required")
    return regularized_beta(degrees / (degrees + statistic**2), degrees / 2, 0.5)


def t_critical(alpha, degrees):
    if not (0 < alpha < 1 and degrees > 0):
        raise ValueError("Invalid t quantile arguments")
    lower, upper = 0.0, 1.0
    while t_two_sided_p(upper, degrees) > alpha:
        upper *= 2
    for _ in range(100):
        middle = (lower + upper) / 2
        if t_two_sided_p(middle, degrees) > alpha:
            lower = middle
        else:
            upper = middle
    return (lower + upper) / 2


def holm(p_values):
    """Unavailable p-values conservatively occupy a place in the family as 1."""
    values = [1.0 if p is None else float(p) for p in p_values]
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in values):
        raise ValueError("p-values must be finite and in [0,1], or None")
    order = sorted(range(len(values)), key=lambda i: (values[i], i))
    result, previous = [None] * len(values), 0.0
    for rank, index in enumerate(order):
        previous = max(previous, min(1.0, (len(values) - rank) * values[index]))
        result[index] = previous
    return result


def sign_flip_sensitivity(values):
    if len(values) != 8 or not all(math.isfinite(x) for x in values):
        raise ValueError("Exactly eight finite world differences required")
    observed = abs(math.fsum(values))
    tolerance = 1e-12 * max(1.0, math.fsum(abs(x) for x in values))
    extreme = sum(
        abs(math.fsum(s * x for s, x in zip(signs, values, strict=True))) >= observed - tolerance
        for signs in itertools.product((-1, 1), repeat=8)
    )
    return {"p": extreme / 256, "extreme": extreme, "enumerated": 256}


def world_t_statistics(values):
    values = [float(x) for x in values]
    if len(values) != 8 or not all(math.isfinite(x) for x in values):
        raise ValueError("Exactly eight finite world differences required")
    mean = math.fsum(values) / 8
    variance = math.fsum((value - mean) ** 2 for value in values) / 7
    sd = math.sqrt(variance)
    result = {
        "worlds": 8,
        "df": 7,
        "mean": mean,
        "sd": sd,
        "sign_flip_sensitivity": sign_flip_sensitivity(values),
    }
    if max(values) == min(values):
        zero = values[0] == 0
        result.update(
            status="degenerate_zero" if zero else "degenerate_nonzero",
            t=0.0 if zero else None,
            p=1.0 if zero else None,
            ci95=[0.0, 0.0] if zero else None,
            ci_bonferroni=[0.0, 0.0] if zero else None,
        )
        return result
    standard_error = sd / math.sqrt(8)
    statistic = mean / standard_error
    result.update(status="estimable", t=statistic, p=t_two_sided_p(statistic, 7))
    for key, alpha in [("ci95", 0.05), ("ci_bonferroni", 0.05 / 3)]:
        half = t_critical(alpha, 7) * standard_error
        result[key] = [mean - half, mean + half]
    return result


def _integer(value, label):
    if isinstance(value, bool):
        raise ValueError(f"Boolean integer for {label}")
    parsed = int(value)
    if isinstance(value, str) and str(parsed) != value:
        raise ValueError(f"Noncanonical integer for {label}: {value!r}")
    if not isinstance(value, str) and parsed != value:
        raise ValueError(f"Noninteger value for {label}: {value!r}")
    return parsed


def validate_endpoints(rows, editing=False):
    keys = ["world", "seed", "condition", "chain", "prevalence"]
    if editing:
        keys += ["support", "kind"]
    indexed = {}
    for raw in rows:
        row = dict(raw)
        if not editing and "support" in row:
            raise ValueError("Learning endpoints must not be repeated or labeled by support")
        for name in ["world", "seed", "step", "n", "correct"] + (["support"] if editing else []):
            row[name] = _integer(row[name], name)
        if "width" in row and _integer(row["width"], "width") != 256:
            raise ValueError("Unexpected model width")
        if row["step"] != (512 if editing else 15360):
            raise ValueError("Endpoint step differs from the fixed budget")
        if row["n"] != (9 if editing else 64) or not 0 <= row["correct"] <= row["n"]:
            raise ValueError("Fixed cohort count is invalid")
        if not editing and (row["cohort"] != "original_exception" or row["split"] != "heldout"):
            raise ValueError("Wrong learning cohort or split")
        key = tuple(row[k] for k in keys)
        if key in indexed:
            raise ValueError(f"Duplicate endpoint: {key}")
        indexed[key] = row
    dimensions = [
        range(100, 108),
        (0, 1),
        ("company", "project", "neither"),
        ("company", "project"),
        ("low", "high"),
    ]
    if editing:
        dimensions += [(0, 1), ("coherent", "exception")]
    expected = set(itertools.product(*dimensions))
    if set(indexed) != expected:
        raise ValueError(
            f"Endpoint matrix mismatch: missing {len(expected - set(indexed))}, "
            f"unexpected {len(set(indexed) - expected)}"
        )
    return indexed


def analyze_endpoints(learning, editing):
    learning = validate_endpoints(learning)
    editing = validate_endpoints(editing, editing=True)
    world_rows, case_rows = [], []
    for endpoint in DEFAULT_STATISTICS["confirmatory_family"]:
        kind = None if endpoint == "learning_original_exception" else endpoint.split("_")[1]
        source, denominator = (learning, 64) if kind is None else (editing, 9)
        for world in range(100, 108):
            pairs = []
            for seed, condition, chain in itertools.product(
                (0, 1), ("company", "project", "neither"), ("company", "project")
            ):
                for support in [None] if kind is None else (0, 1):
                    base = (world, seed, condition, chain)
                    suffix = () if kind is None else (support, kind)
                    low = source[(*base, "low", *suffix)]
                    high = source[(*base, "high", *suffix)]
                    row = {
                        "endpoint": endpoint,
                        "world": world,
                        "seed": seed,
                        "condition": condition,
                        "chain": chain,
                        "support": support,
                        "kind": kind,
                        "n": denominator,
                        "low_correct": low["correct"],
                        "high_correct": high["correct"],
                        "correct_difference": high["correct"] - low["correct"],
                    }
                    row["accuracy_difference"] = row["correct_difference"] / denominator
                    pairs.append(row)
                    case_rows.append(row)
            n = len(pairs) * denominator
            world_rows.append(
                {
                    "endpoint": endpoint,
                    "world": world,
                    "paired_cases": len(pairs),
                    "nested_query_count_descriptive": n,
                    "low_accuracy": sum(r["low_correct"] for r in pairs) / n,
                    "high_accuracy": sum(r["high_correct"] for r in pairs) / n,
                    "difference": sum(r["correct_difference"] for r in pairs) / n,
                }
            )
    results = []
    for endpoint in DEFAULT_STATISTICS["confirmatory_family"]:
        values = [row["difference"] for row in world_rows if row["endpoint"] == endpoint]
        results.append({"endpoint": endpoint, **world_t_statistics(values)})
    adjusted = holm([result["p"] for result in results])
    for result, value in zip(results, adjusted, strict=True):
        result.update(
            holm_input=1.0 if result["p"] is None else result["p"],
            holm_adjusted_p=value,
            reject_two_sided_family_05=result["p"] is not None and value <= 0.05,
        )
    return {
        "contract": DEFAULT_STATISTICS,
        "results": results,
        "world_effects": world_rows,
        "paired_cases": case_rows,
        "input_rows": {"learning": len(learning), "editing": len(editing)},
    }
