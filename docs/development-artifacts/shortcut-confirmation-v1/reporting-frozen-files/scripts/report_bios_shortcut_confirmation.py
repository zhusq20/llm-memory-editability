"""Publication figures and a descriptive report after the locked full-matrix audit.

--demo constructs fictitious counts in memory; it never creates or reads a world.
The formal path only consumes complete, hash-verified summary/statistics outputs.
No inferential procedure is introduced here: primary results reproduce the frozen
statistics module, and all additional displays are labeled descriptive checks.
"""

import argparse
import csv
import hashlib
import itertools
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from llm_memory_editability import bios_confirmation_stats as locked_statistics

WORLDS = tuple(locked_statistics.DEFAULT_STATISTICS["worlds"])
PHASES = ("low", "high")
KINDS = ("coherent", "exception")
CONDITIONS = ("company", "project", "neither")
CHAINS = ("company", "project")
STEPS = (0, 1280, 2560, 5120, 10240, 15360)
COLORS = ("#285A8E", "#C66A0A")
TABLES = (
    "learning-strata.csv",
    "learning-facts.csv",
    "learning-metrics.csv",
    "two-step-strata.csv",
    "editing-all-nodes.csv",
    "learning-endpoint.csv",
    "editing-endpoint.csv",
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def number(value):
    return None if value in (None, "") else float(value)


def load_audited(summary, statistics):
    summary, statistics = Path(summary).resolve(), Path(statistics).resolve()
    audit = json.loads((summary / "audit.json").read_text())
    stat_audit = json.loads((statistics / "statistics-audit.json").read_text())
    if any(
        a.get("complete") is not True or a.get("weight_archive_verified") is not True
        for a in (audit, stat_audit)
    ):
        raise ValueError("Figures require complete summary/statistics and verified weight archives")
    if audit["archive_index_sha256"] != stat_audit["archive_index_sha256"]:
        raise ValueError("Summary and statistics refer to different verified archives")
    for path, expected in stat_audit["statistics_source_hashes"].items():
        if sha(path) != expected:
            raise ValueError("Frozen statistics implementation changed")
    tables, sources = {}, {}
    for name in TABLES:
        path = summary / name
        digest = sha(path)
        if audit["outputs_sha256"].get(name) != digest:
            raise ValueError(f"Audited summary table changed: {name}")
        tables[name] = read_csv(path)
        sources[str(path)] = digest
    for name in ("audit.json", "learning-endpoint.csv", "editing-endpoint.csv"):
        expected = [
            value for path, value in stat_audit["source_hashes"].items() if Path(path).name == name
        ]
        if expected != [sha(summary / name)]:
            raise ValueError(f"Statistics input identity mismatch: {name}")
    results = json.loads((statistics / "confirmation-statistics.json").read_text())
    reproduced = locked_statistics.analyze_endpoints(
        tables["learning-endpoint.csv"], tables["editing-endpoint.csv"]
    )
    if results != reproduced:
        raise ValueError("Published statistics do not reproduce from the audited endpoints")
    for path in (
        summary / "audit.json",
        statistics / "statistics-audit.json",
        statistics / "confirmation-statistics.json",
    ):
        sources[str(path)] = sha(path)
    return tables, results, sources


COMMON_POOLS = ("U_full", "U_unseen", "U_heldout")
COMMON_CONTROLS = ("common_known", "same_truth_common_known")
EDIT_STEPS = (0, 32, 128, 512)


def load_common_u(directory, summary, sources):
    if directory is None:
        raise ValueError("Formal report requires audited --common-u controls")
    directory, summary = Path(directory).resolve(), Path(summary).resolve()
    path = directory / "audit.json"
    audit = json.loads(path.read_text())
    primary = json.loads((summary / "audit.json").read_text())
    if audit.get("complete") is not True or audit.get("weight_archive_verified") is not True:
        raise ValueError("Common-U controls require a complete verified audit")
    if (
        audit.get("primary_audit_sha256") != sha(summary / "audit.json")
        or audit.get("archive_index_sha256") != primary["archive_index_sha256"]
    ):
        raise ValueError("Common-U audit has a different primary or archive identity")
    for key in ("receipt_sha256", "sources_sha256"):
        value = audit.get(key, "")
        if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("Common-U audit lacks a sealed source identity")
    if not audit.get("supplementary_sources"):
        raise ValueError("Common-U audit lacks supplementary source identities")
    for source, digest in audit["supplementary_sources"].items():
        source = Path(source)
        if not source.is_absolute():
            source = Path(__file__).resolve().parents[1] / source
        if sha(source) != digest:
            raise ValueError("Common-U supplementary implementation changed")
        sources[str(Path(source).resolve())] = digest
    source_manifest = directory / "sources.json"
    if sha(source_manifest) != audit["sources_sha256"]:
        raise ValueError("Common-U source ledger changed")
    expected_audit = dict(
        protocol="confirmation-common-U-supplement-v1",
        parent_models=96,
        edit_cases=768,
        paired_cases=384,
        paired_checkpoints=1536,
        rows=55296,
        declared_weights=2208,
        lock_sha256=primary["lock_sha256"],
    )
    if any(audit.get(key) != value for key, value in expected_audit.items()):
        raise ValueError("Common-U audited matrix or protocol differs")
    ledger = json.loads(source_manifest.read_text())
    receipts = [Path(path) for path, digest in ledger.items() if digest == audit["receipt_sha256"]]
    if len(receipts) != 1 or sha(receipts[0]) != audit["receipt_sha256"]:
        raise ValueError("Common-U receipt is absent, ambiguous, or changed")
    receipt_path = receipts[0]
    receipt = json.loads(receipt_path.read_text())
    expected_receipt = dict(
        protocol=expected_audit["protocol"],
        status="frozen_before_confirmation_effect_inspection",
        config_sha256=primary["config_sha256"],
        lock_sha256=primary["lock_sha256"],
        supplemental_sources=audit["supplementary_sources"],
        checkpoints=list(EDIT_STEPS),
        controls=list(COMMON_CONTROLS),
        pools=list(COMMON_POOLS),
    )
    if any(receipt.get(key) != value for key, value in expected_receipt.items()):
        raise ValueError("Common-U pre-inspection receipt identity differs")
    if datetime.fromisoformat(receipt.get("frozen_at_utc", "")).tzinfo is None:
        raise ValueError("Common-U receipt needs an explicit time-zone timestamp")
    sources[str(receipt_path.resolve())] = audit["receipt_sha256"]
    rows_path = directory / "paired-common-u.csv"
    if sha(rows_path) != audit["outputs_sha256"].get(rows_path.name):
        raise ValueError("Common-U outcome table changed")
    for source in (path, source_manifest, rows_path):
        sources[str(source)] = sha(source)
    return read_csv(rows_path)


def common_u_checks(rows):
    """Validate the entire paired matrix and independently aggregate the sealed counts."""
    keys = (
        "world",
        "seed",
        "condition",
        "chain",
        "support",
        "kind",
        "step",
        "pool",
        "stratum",
        "control",
    )
    integer_keys = {"world", "seed", "support", "step", "stratum"}
    expected = set(
        itertools.product(
            WORLDS,
            (0, 1),
            CONDITIONS,
            CHAINS,
            (0, 1),
            KINDS,
            EDIT_STEPS,
            COMMON_POOLS,
            range(-1, 5),
            COMMON_CONTROLS,
        )
    )
    seen, groups = set(), defaultdict(list)
    counts = (
        "pool_n",
        "eligible_n",
        "known",
        "low_baseline_known",
        "high_baseline_known",
        "low_broken",
        "high_broken",
    )

    def rate(n, d):
        return n / d if d else None

    for original in rows:
        identity = tuple(int(original[k]) if k in integer_keys else original[k] for k in keys)
        if identity in seen or identity not in expected:
            raise ValueError("Common-U duplicate or unexpected nested identity")
        seen.add(identity)
        row = {
            **dict(zip(keys, identity, strict=True)),
            **{key: int(original[key]) for key in counts},
            "stratum_name": original["stratum_name"],
        }
        n, eligible, known = (row[k] for k in ("pool_n", "eligible_n", "known"))
        if not 0 <= known <= eligible <= n:
            raise ValueError("Common-U denominator ordering is invalid")
        rates = {
            "joint_known_coverage": rate(known, eligible),
            "joint_known_fraction_of_pool": rate(known, n),
        }
        for phase in PHASES:
            if not known <= row[f"{phase}_baseline_known"] <= eligible:
                raise ValueError("Common-U baseline overlap is invalid")
            if not 0 <= row[f"{phase}_broken"] <= known:
                raise ValueError("Common-U broken count exceeds the shared denominator")
            rates[f"{phase}_damage"] = rate(row[f"{phase}_broken"], known)
        rates["high_minus_low_damage"] = rate(row["high_broken"] - row["low_broken"], known)
        for name, value in rates.items():
            saved = number(original[name])
            if (saved is None) != (value is None) or (
                value is not None and not np.isclose(saved, value, rtol=0, atol=1e-12)
            ):
                raise ValueError(f"Common-U stored rate disagrees with counts: {name}")
            row[name] = value
        group = tuple(row[k] for k in ("world", "kind", "step", "pool", "stratum", "control"))
        groups[group].append(row)
    if seen != expected:
        raise ValueError("Common-U requires the complete paired matrix before reporting")
    result = []
    for identity, group in sorted(groups.items()):
        row = dict(
            zip(("world", "kind", "step", "pool", "stratum", "control"), identity, strict=True)
        )
        if len(group) != 24 or len({r["stratum_name"] for r in group}) != 1:
            raise ValueError("Common-U world block count or stratum label changed")
        row.update(stratum_name=group[0]["stratum_name"], paired_cases=24)
        row.update({key: sum(r[key] for r in group) for key in counts})
        for name in (
            "low_damage",
            "high_damage",
            "high_minus_low_damage",
            "joint_known_coverage",
            "joint_known_fraction_of_pool",
        ):
            valid = [r[name] for r in group if r[name] is not None]
            row[f"{name}_case_macro"] = float(np.mean(valid)) if valid else None
            row[f"{name}_valid_cases"] = len(valid)
        for phase in PHASES:
            row[f"{phase}_damage_pooled"] = rate(row[f"{phase}_broken"], row["known"])
        row["high_minus_low_damage_pooled"] = rate(
            row["high_broken"] - row["low_broken"], row["known"]
        )
        row["joint_known_coverage_pooled"] = rate(row["known"], row["eligible_n"])
        result.append(row)
    return result


def matches(row, **filters):
    return all(str(row[key]) == str(value) for key, value in filters.items())


def collect_checks(tables):
    output = []

    def add(
        rows,
        metric,
        value,
        expected_cases,
        filters=None,
        known=None,
        total=None,
        step=15360,
        numerator=None,
        pooled_denominator=None,
    ):
        selected = [r for r in rows if matches(r, **(filters or {}))]
        groups = defaultdict(list)
        for row in selected:
            groups[int(row["world"]), row.get("phase", row.get("prevalence"))].append(row)
        if set(groups) != set(itertools.product(WORLDS, PHASES)):
            raise ValueError(f"Descriptive check omits worlds/phases: {metric}")
        for (world, phase), group in sorted(groups.items()):
            if len(group) != expected_cases:
                raise ValueError(f"Descriptive case count changed: {metric}/{world}/{phase}")
            values = [value(r) for r in group]
            finite = [v for v in values if v is not None]
            if not all(np.isfinite(v) for v in finite):
                raise ValueError("Nonfinite descriptive metric")
            numerator_total = sum(int(r[numerator]) for r in group) if numerator else None
            pooled_total = (
                sum(int(r[pooled_denominator]) for r in group) if pooled_denominator else None
            )
            output.append(
                dict(
                    metric=metric,
                    world=world,
                    phase=phase,
                    step=step,
                    value=float(np.mean(finite)) if finite else None,
                    cases=len(group),
                    available_cases=len(finite),
                    undefined_cases=len(group) - len(finite),
                    known_total=sum(int(r[known]) for r in group) if known else None,
                    denominator_total=sum(int(r[total]) for r in group) if total else None,
                    numerator_total=numerator_total,
                    pooled_denominator_total=pooled_total,
                    pooled_value=numerator_total / pooled_total if pooled_total else None,
                    aggregation="case_macro_within_world",
                )
            )

    direct = tables["learning-strata.csv"]
    for cohort in ("all", "original_exception", "newly_exception", "remaining_ordinary"):
        add(
            direct,
            f"learning_direct_{cohort}",
            lambda r: number(r["accuracy"]),
            12,
            dict(step=15360, method="direct", split="heldout", cohort_kind="fixed", cohort=cohort),
            total="n",
        )
    old = dict(
        step=15360,
        method="direct",
        split="heldout",
        cohort_kind="fixed",
        cohort="original_exception",
    )
    add(
        direct,
        "learning_original_actual_error",
        lambda r: int(r["actual_on_conflict"]) / int(r["n"]),
        12,
        old,
        total="n",
    )
    add(
        direct,
        "learning_original_other_error",
        lambda r: (int(r["n"]) - int(r["correct"]) - int(r["actual_on_conflict"])) / int(r["n"]),
        12,
        old,
        total="n",
    )
    add(
        tables["two-step-strata.csv"],
        "learning_original_two_step",
        lambda r: number(r["accuracy"]),
        12,
        dict(step=15360, split="heldout", cohort_kind="fixed", cohort="original_exception"),
        total="n",
    )
    for field in ("bridge_correct", "bridge_valid"):
        add(
            tables["two-step-strata.csv"],
            f"learning_original_{field}",
            lambda r, f=field: int(r[f]) / int(r["n"]),
            12,
            dict(step=15360, split="heldout", cohort_kind="fixed", cohort="original_exception"),
            total="n",
        )
    for fact in ("actual", "membership", "root"):
        add(
            tables["learning-facts.csv"],
            f"learning_original_{fact}_fact",
            lambda r, f=fact: number(r[f"{f}_accuracy"]),
            12,
            dict(step=15360, split="heldout", cohort="original_exception"),
            total="n",
        )
    for step in STEPS:
        for metric in ("base_accuracy", "base_nll", "mean_heldout"):
            add(
                tables["learning-metrics.csv"],
                metric,
                lambda r, m=metric: number(r[m]),
                6,
                dict(step=step),
                step=step,
            )
    edits = tables["editing-all-nodes.csv"]
    for kind in KINDS:
        filters = dict(step=512, kind=kind)
        for name in ("E", "E_roots", "E_actual", "D", "D_heldout", "paired_reference_D_heldout"):
            add(
                edits,
                f"{kind}_{name}",
                lambda r, n=name: number(r[f"{n}_accuracy"]),
                24,
                filters,
                total=f"{name}_n",
            )
        for pool in ("U_full", "U_heldout"):
            for field in ("damage", "coverage"):
                add(
                    edits,
                    f"{kind}_{pool}_{field}",
                    lambda r, p=pool, f=field: number(r[f"{p}_{f}"]),
                    24,
                    filters,
                    known=f"{pool}_known",
                    total=f"{pool}_n",
                    numerator=f"{pool}_{'broken' if field == 'damage' else 'known'}",
                    pooled_denominator=f"{pool}_{'known' if field == 'damage' else 'n'}",
                )
        for stratum in range(5):
            for field in ("damage", "coverage"):
                prefix = f"U_full_strata_{stratum}"
                add(
                    edits,
                    f"{kind}_local{stratum}_{field}",
                    lambda r, p=prefix, f=field: number(r[f"{p}_{f}"]),
                    24,
                    filters,
                    known=f"{prefix}_known",
                    total=f"{prefix}_n",
                    numerator=f"{prefix}_{'broken' if field == 'damage' else 'known'}",
                    pooled_denominator=f"{prefix}_{'known' if field == 'damage' else 'n'}",
                )
    return output


def points(checks, metric, phase, step=15360):
    selected = {
        r["world"]: r
        for r in checks
        if r["metric"] == metric and r["phase"] == phase and r["step"] == step
    }
    if set(selected) != set(WORLDS):
        raise ValueError(f"Plot requires all eight worlds: {metric}/{phase}/{step}")
    return np.array(
        [np.nan if selected[w]["value"] is None else selected[w]["value"] for w in WORLDS]
    )


ORG_CONTRASTS = (
    "matching_interaction",
    "company_matched_vs_neither",
    "company_unmatched_vs_neither",
    "company_both_chains_vs_neither",
    "project_matched_vs_neither",
    "project_unmatched_vs_neither",
    "project_both_chains_vs_neither",
)


def org_block(values):
    """Paired descriptive contrasts; first key is training organization, second is QA chain."""
    cc, cp = values["company", "company"], values["company", "project"]
    pc, pp = values["project", "company"], values["project", "project"]
    nc, np_ = values["neither", "company"], values["neither", "project"]
    return dict(
        matching_interaction=0.5 * (cc - pc + pp - cp),
        company_matched_vs_neither=cc - nc,
        company_unmatched_vs_neither=cp - np_,
        company_both_chains_vs_neither=0.5 * (cc - nc + cp - np_),
        project_matched_vs_neither=pp - np_,
        project_unmatched_vs_neither=pc - nc,
        project_both_chains_vs_neither=0.5 * (pp - np_ + pc - nc),
    )


def org_effects(tables):
    """Every world and phase, without outcome-based selection or extra tests.

    Each seed (and editing support) contributes its complete organization × chain
    block. Supports never replicate the learning endpoint. High-minus-low is a
    paired change in the same descriptive contrast, not a new hypothesis test.
    """
    specifications = []
    for cohort in ("all", "original_exception", "newly_exception", "remaining_ordinary"):
        rows = [
            r
            for r in tables["learning-strata.csv"]
            if matches(
                r, step=15360, method="direct", split="heldout", cohort_kind="fixed", cohort=cohort
            )
        ]
        specifications.append((f"learning_{cohort}", rows, "correct", "n", False))
    for kind, metric in itertools.product(
        KINDS, ("paired_reference_D_heldout", "D", "D_heldout", "E", "E_roots", "E_actual")
    ):
        rows = [r for r in tables["editing-all-nodes.csv"] if matches(r, step=512, kind=kind)]
        name = "fixed9" if metric == "paired_reference_D_heldout" else metric
        specifications.append(
            (f"editing_{kind}_{name}", rows, f"{metric}_correct", f"{metric}_n", True)
        )
    result = []
    expected_cells = set(itertools.product(CONDITIONS, CHAINS))
    for endpoint, rows, correct, denominator, editing in specifications:
        grouped = defaultdict(list)
        for row in rows:
            identity = (
                int(row["world"]),
                row["phase"],
                int(row["seed"]),
                int(row["support"]) if editing else -1,
            )
            grouped[identity].append(row)
        expected_groups = set(
            itertools.product(WORLDS, PHASES, (0, 1), (0, 1) if editing else (-1,))
        )
        if set(grouped) != expected_groups:
            raise ValueError(f"Organization contrast requires complete nested matrix: {endpoint}")
        block_values = defaultdict(list)
        for (world, phase, seed, support), group in sorted(grouped.items()):
            values = {}
            for row in group:
                cell = (row["condition"], row["chain"])
                if cell in values:
                    raise ValueError(
                        f"Duplicate organization cell: {endpoint}/{world}/{phase}/{seed}/{support}"
                    )
                n, c = int(row[denominator]), int(row[correct])
                if not 0 <= c <= n or n <= 0:
                    raise ValueError("Organization accuracy requires a positive count denominator")
                values[cell] = c / n
            if set(values) != expected_cells:
                raise ValueError("Organization contrast requires each complete six-cell block")
            combined = {f"cell_{org}_{chain}": value for (org, chain), value in values.items()}
            combined.update(org_block(values))
            block_values[world, phase].append(combined)
        for world in WORLDS:
            phase_values = {}
            for phase in PHASES:
                blocks = block_values[world, phase]
                means = {key: float(np.mean([b[key] for b in blocks])) for key in blocks[0]}
                phase_values[phase] = means
                result.append(
                    dict(
                        endpoint=endpoint,
                        world=world,
                        phase=phase,
                        nested_blocks=len(blocks),
                        **means,
                    )
                )
            result.append(
                dict(
                    endpoint=endpoint,
                    world=world,
                    phase="high_minus_low",
                    nested_blocks=4 if editing else 2,
                    **{
                        key: phase_values["high"][key] - phase_values["low"][key]
                        for key in phase_values["low"]
                    },
                )
            )
    return result


def style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 11,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def finish(fig, output, stem, synthetic, footer):
    if synthetic:
        fig.text(
            0.5,
            0.985,
            "SYNTHETIC LAYOUT CHECK — NOT EXPERIMENTAL RESULTS",
            ha="center",
            color="#a02020",
            fontsize=13,
            weight="bold",
            va="top",
        )
    fig.text(0.5, 0.015, footer, ha="center", fontsize=8.5, color="#444444")
    fig.savefig(output / f"{stem}.png", dpi=300)
    fig.savefig(output / f"{stem}.pdf")
    plt.close(fig)


def clean(ax, percent=True):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#888888")
    ax.grid(axis="y", color="#dddddd", linewidth=0.7)
    ax.set_axisbelow(True)
    if percent:
        ax.set_ylim(0, 110)


def paired_bars(ax, checks, metrics, labels, percent=True, show_labels=True):
    x, width = np.arange(len(metrics)), 0.36
    for idx, (phase, color) in enumerate(zip(PHASES, COLORS, strict=True)):
        groups = [points(checks, metric, phase) * 100 for metric in metrics]
        means = [float(np.nanmean(g)) if np.isfinite(g).any() else np.nan for g in groups]
        bars = ax.bar(x + (idx - 0.5) * width, means, width, color=color, alpha=0.85)
        for j, value in enumerate(means):
            if not np.isfinite(value):
                ax.text(
                    x[j] + (idx - 0.5) * width,
                    0.03,
                    "N/A",
                    ha="center",
                    transform=ax.get_xaxis_transform(),
                    fontsize=8,
                )
        for j, values in enumerate(groups):
            ax.scatter(
                np.full(8, x[j] + (idx - 0.5) * width) + np.linspace(-0.11, 0.11, 8),
                values,
                s=11,
                facecolors="white",
                edgecolors=color,
                linewidths=0.65,
                zorder=3,
            )
        if show_labels:
            ax.bar_label(
                bars,
                labels=[f"{m:.1f}" if np.isfinite(m) else "N/A" for m in means],
                padding=3,
                fontsize=8,
            )
    ax.set_xticks(x, labels)
    clean(ax, percent)


def endpoint_figure(results, output, synthetic):
    order = ("editing_exception_fixed9", "learning_original_exception", "editing_coherent_fixed9")
    titles = (
        "(a) Primary: exception-edit conflicts",
        "(b) Learning: original exceptions",
        "(c) Coherent edits: paired reference",
    )
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 5.6), sharey=True)
    limits = [0.0, *[100 * r["difference"] for r in results["world_effects"]]]
    for row in results["results"]:
        if row["ci_bonferroni"] is not None:
            limits.extend(100 * v for v in row["ci_bonferroni"])
    span = max(max(limits) - min(limits), 1)
    bounds = (min(limits) - span * 0.08, max(limits) + span * 0.08)
    for ax, endpoint, title in zip(axes, order, titles, strict=True):
        row = next(r for r in results["results"] if r["endpoint"] == endpoint)
        values = [r for r in results["world_effects"] if r["endpoint"] == endpoint]
        values.sort(key=lambda r: r["world"])
        ax.scatter([100 * r["difference"] for r in values], np.arange(8), color="#527B91", s=33)
        mean = 100 * row["mean"]
        interval = row["ci_bonferroni"]
        if interval is not None:
            ax.plot(100 * np.array(interval), [8.5, 8.5], color="#222222", linewidth=2)
        ax.scatter([mean], [8.5], marker="D", color="#222222", s=38)
        ax.axvline(0, color="#aaaaaa", linewidth=1, linestyle="--")
        ax.axhline(7.75, color="#dddddd", linewidth=0.8)
        ax.set_title(title, loc="left", pad=12)
        ax.set_yticks([*range(8), 8.5], [*(f"World {w}" for w in WORLDS), "Mean"])
        ax.set_ylim(9.2, -0.7)
        ax.set_xlim(*bounds)
        ax.set_xlabel("High minus low prevalence (pp)")
        ax.spines[["top", "right"]].set_visible(False)
        text = f"Mean {mean:+.2f} pp; Holm p={row['holm_adjusted_p']:.3g}"
        if interval is None:
            text += "\nInterval not estimable under locked policy"
        ax.text(0.02, -0.18, text, transform=ax.transAxes, fontsize=8.5)
    fig.subplots_adjust(left=0.078, right=0.985, bottom=0.23, top=0.87, wspace=0.22)
    finish(
        fig,
        output,
        "confirmation-endpoints",
        synthetic,
        "Eight world-level paired effects; mean intervals are 98.333% simultaneous t intervals. "
        "All three endpoints shown; no query-level inference.",
    )


def learning_figure(checks, output, synthetic):
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.2))
    ax = axes[0, 0]
    paired_bars(
        ax,
        checks,
        [
            f"learning_direct_{c}"
            for c in ("original_exception", "newly_exception", "remaining_ordinary", "all")
        ],
        ("Original\nexceptions", "Newly\nexceptions", "Remaining\nordinary", "All\nheld-out"),
    )
    ax.set_title("(a) Direct queries: fixed people and overall cost", loc="left", pad=10)
    ax.set_ylabel("Held-out accuracy (%)")
    ax = axes[0, 1]
    for phase, color in zip(PHASES, COLORS, strict=True):
        for metric, line, marker, label in (
            ("learning_direct_original_exception", "-", "o", "Direct"),
            ("learning_original_two_step", "--", "s", "Two-step final"),
        ):
            ax.plot(
                range(8),
                100 * points(checks, metric, phase),
                color=color,
                linestyle=line,
                marker=marker,
                markersize=4,
                label=f"{phase.title()} prevalence: {label}",
            )
    clean(ax)
    ax.set_title("(b) Original exceptions: autonomous two-step", loc="left", pad=10)
    ax.set_xticks(range(8), [str(w) for w in WORLDS])
    ax.set_xlabel("World")
    ax.set_ylabel("Accuracy (%)")
    ax.legend(fontsize=8, frameon=False, loc="best")
    ax = axes[1, 0]
    paired_bars(
        ax,
        checks,
        (
            "learning_original_actual_fact",
            "learning_original_membership_fact",
            "learning_original_root_fact",
            "base_accuracy",
        ),
        ("Actual\nfact", "Membership\nfact", "Person's\nroot", "All base\nfacts"),
    )
    ax.set_title("(c) Basic knowledge remains a separate check", loc="left", pad=10)
    ax.set_ylabel("Accuracy (%)")
    ax = axes[1, 1]
    steps = (5120, 10240, 15360)
    for phase, color in zip(PHASES, COLORS, strict=True):
        values = np.stack([points(checks, "base_nll", phase, step) for step in steps])
        for column in range(8):
            ax.plot(range(3), values[:, column], color=color, alpha=0.25, linewidth=0.7)
        ax.plot(range(3), values.mean(axis=1), color=color, marker="o", linewidth=2)
    clean(ax, False)
    ax.set_ylim(bottom=0)
    ax.set_title("(d) Foundation-fact learning trajectories", loc="left", pad=10)
    ax.set_xticks(range(3), ("5,120", "10,240", "15,360"))
    ax.set_xlabel("Training steps; faint lines: all eight worlds")
    ax.set_ylabel("Mean value-token NLL")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in COLORS]
    fig.legend(
        handles,
        ("Low prevalence: 2/32", "High prevalence: 16/32"),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.064),
        ncol=2,
        frameon=False,
    )
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.17, top=0.91, wspace=0.23, hspace=0.35)
    finish(
        fig,
        output,
        "confirmation-learning-checks",
        synthetic,
        "Descriptive case means; open dots are worlds, not confidence intervals. Root facts are "
        "person-weighted. Two-step means final-city success; first-hop truth accuracy is separate.",
    )


def editing_figure(checks, organization, output, synthetic):
    fig, axes = plt.subplots(3, 2, figsize=(12.5, 12.9))
    ax = axes[0, 0]
    paired_bars(
        ax, checks, ("exception_E", "coherent_E"), ("Exception updates", "Coherent updates")
    )
    ax.set_title("(a) Immediate update fit: all 39 edited facts", loc="left", pad=10)
    ax.set_ylabel("E accuracy (%)")
    ax = axes[0, 1]
    paired_bars(
        ax,
        checks,
        ("exception_D", "exception_D_heldout", "coherent_D", "coherent_D_heldout"),
        (
            "Exception\nall D96",
            "Exception\nheld-out D48",
            "Coherent\nall D96",
            "Coherent\nheld-out D48",
        ),
    )
    ax.set_title("(b) Propagation beyond the fixed nine people", loc="left", pad=10)
    ax.set_ylabel("Propagation accuracy (%)")
    ax = axes[1, 0]
    paired_bars(
        ax,
        checks,
        (
            "exception_U_full_damage",
            "exception_U_heldout_damage",
            "coherent_U_full_damage",
            "coherent_U_heldout_damage",
        ),
        ("Exception\nall U", "Exception\nheld-out U", "Coherent\nall U", "Coherent\nheld-out U"),
        percent=False,
    )
    ax.set_title("(c) Global damage among previously correct facts", loc="left", pad=10)
    ax.set_ylabel("Conditional damage (%)")
    ax.set_ylim(bottom=0)
    local_metrics = [
        f"{kind}_local{s}_damage" for kind in ("exception", "coherent") for s in (0, 1, 2)
    ]
    local_labels = (
        "Exc.\nold actual",
        "Exc.\nnew actual",
        "Exc.\nother local",
        "Coh.\nold actual",
        "Coh.\nnew actual",
        "Coh.\nother local",
    )
    ax = axes[1, 1]
    paired_bars(ax, checks, local_metrics, local_labels, percent=False, show_labels=False)
    ax.set_title("(d) Local damage in affected groups", loc="left", pad=10)
    ax.set_ylabel("Conditional damage (%)")
    ax.set_ylim(bottom=0)
    ax = axes[2, 0]
    paired_bars(
        ax,
        checks,
        [m.replace("_damage", "_coverage") for m in local_metrics],
        local_labels,
        show_labels=False,
    )
    ax.set_title("(e) The previously correct denominator is visible", loc="left", pad=10)
    ax.set_ylabel("Old-correct coverage (%)")
    ax = axes[2, 1]
    for phase, color in zip(PHASES, COLORS, strict=True):
        rows = sorted(
            [
                r
                for r in organization
                if r["phase"] == phase and r["endpoint"] == "editing_exception_fixed9"
            ],
            key=lambda r: r["world"],
        )
        ax.plot(
            range(8),
            [100 * r["matching_interaction"] for r in rows],
            color=color,
            marker="o",
            markersize=4,
            label=phase.title(),
        )
    clean(ax, False)
    ax.axhline(0, color="#888888", linewidth=1)
    ax.set_title("(f) Organization interaction: descriptive only", loc="left", pad=10)
    ax.set_xticks(range(8), [str(w) for w in WORLDS])
    ax.set_xlabel("World; exception-update fixed-nine endpoint")
    ax.set_ylabel("Matching interaction (pp)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in COLORS]
    fig.legend(
        handles,
        ("Low prevalence: 2/32", "High prevalence: 16/32"),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.05),
        ncol=2,
        frameon=False,
    )
    fig.subplots_adjust(left=0.067, right=0.985, bottom=0.12, top=0.94, wspace=0.27, hspace=0.4)
    finish(
        fig,
        output,
        "confirmation-editing-checks",
        synthetic,
        "Descriptive case means and all eight worlds. Conditional U damage uses each phase's own "
        "old-correct facts; coverage and undefined cases are retained in the accompanying CSV.",
    )


def organization_figure(organization, output, synthetic):
    endpoints = (
        ("learning_all", "All held-out learning"),
        ("learning_original_exception", "Original-exception learning"),
        ("editing_exception_fixed9", "Exception edit: fixed nine"),
        ("editing_coherent_fixed9", "Coherent edit: paired nine"),
    )
    metrics = (
        ("matching_interaction", "Matching interaction"),
        ("company_unmatched_vs_neither", "Company org: unmatched − neither"),
        ("project_unmatched_vs_neither", "Project org: unmatched − neither"),
    )
    fig, axes = plt.subplots(4, 3, figsize=(13.3, 13.5))
    for row_index, (endpoint, label) in enumerate(endpoints):
        for column, (metric, title) in enumerate(metrics):
            ax = axes[row_index, column]
            for phase, color, marker in zip(
                (*PHASES, "high_minus_low"), (*COLORS, "#555555"), ("o", "s", "D"), strict=True
            ):
                selected = {
                    r["world"]: r
                    for r in organization
                    if r["endpoint"] == endpoint and r["phase"] == phase
                }
                if set(selected) != set(WORLDS):
                    raise ValueError("Organization figure requires every world")
                ax.plot(
                    range(8),
                    [100 * selected[w][metric] for w in WORLDS],
                    color=color,
                    marker=marker,
                    markersize=3.5,
                    linewidth=1,
                    linestyle="--" if phase == "high_minus_low" else "-",
                )
            clean(ax, False)
            ax.axhline(0, color="#aaaaaa", linewidth=0.8)
            ax.set_xticks(range(8), [str(w) for w in WORLDS], fontsize=8)
            if row_index == 0:
                ax.set_title(title, pad=12)
            if column == 0:
                ax.set_ylabel(label + "\nEffect (pp)")
            if row_index == 3:
                ax.set_xlabel("World")
    handles = [
        plt.Line2D([], [], color=c, marker=m, linestyle=ls)
        for c, m, ls in zip((*COLORS, "#555555"), ("o", "s", "D"), ("-", "-", "--"), strict=True)
    ]
    fig.legend(
        handles,
        ("Low prevalence", "High prevalence", "High minus low interaction"),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.049),
        ncol=3,
        frameon=False,
    )
    fig.subplots_adjust(left=0.085, right=0.99, top=0.94, bottom=0.12, hspace=0.25, wspace=0.3)
    finish(
        fig,
        output,
        "confirmation-organization-interactions",
        synthetic,
        "Descriptive paired contrasts; all eight worlds, no secondary tests or intervals. "
        "Matched, unmatched, both-chain means and all six cells accompany the figure.",
    )


def organization_report(organization, output, synthetic):
    lines = [
        "# " + ("Synthetic layout only: " if synthetic else "") + "Organization × prevalence",
        "",
        "All entries are descriptive percentage-point contrasts; no added tests or intervals.",
        "A cell x[o,q] is accuracy under training organization o and query chain q. "
        "Within each world, contrasts are first paired within seed (and editing support), "
        "then averaged over 2 learning or 4 editing blocks. Learning is not repeated by support.",
        "",
        "Matching = ((x[company,company] − x[project,company]) + "
        "(x[project,project] − x[company,project])) / 2. "
        "For each organization, matched/unmatched minus neither compares the same query chain; "
        "the both-chain benefit is their equally weighted mean. High-minus-low subtracts "
        "the same contrast at low prevalence from high prevalence.",
        "",
        "The complete world-wise low/high/change values and six original accuracy cells "
        "are in [organization-world-checks.csv](organization-world-checks.csv). "
        "A matched benefit alone does not establish an unmatched-query benefit. "
        "An unmatched benefit can reject an explanation limited to matched queries in this "
        "setting, but does not uniquely identify a training mechanism.",
        "",
    ]
    for endpoint in dict.fromkeys(r["endpoint"] for r in organization):
        lines += [
            f"## {endpoint}",
            "",
            "All eight worlds are equally weighted below.",
            "",
            "| Contrast | Low (pp) | High (pp) | High − low (pp) |",
            "|---|---:|---:|---:|",
        ]
        for metric in ORG_CONTRASTS:
            means = [
                100
                * np.mean(
                    [
                        r[metric]
                        for r in organization
                        if r["endpoint"] == endpoint and r["phase"] == phase
                    ]
                )
                for phase in (*PHASES, "high_minus_low")
            ]
            lines.append(f"| {metric} | {means[0]:+.4f} | {means[1]:+.4f} | {means[2]:+.4f} |")
        lines += [
            "",
            "Each world: cells show low / high / high−low, in pp.",
            "",
            "| World | Matching | Company matched | Company unmatched | Company both | "
            "Project matched | Project unmatched | Project both |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for world in WORLDS:
            selected = {
                r["phase"]: r
                for r in organization
                if r["endpoint"] == endpoint and r["world"] == world
            }
            values = [
                " / ".join(
                    f"{100 * selected[phase][metric]:+.3f}" for phase in (*PHASES, "high_minus_low")
                )
                for metric in ORG_CONTRASTS
            ]
            lines.append(f"| {world} | " + " | ".join(values) + " |")
        lines.append("")
    (output / "organization-report.md").write_text("\n".join(lines) + "\n")


def common_u_figure(checks, output, synthetic):
    columns = (
        ("U_full", -1, "All common U"),
        ("U_unseen", -1, "Unseen common U"),
        ("U_heldout", -1, "Held-out common U"),
        ("U_full", 0, "Local original-exception actual"),
    )
    fig, axes = plt.subplots(2, 4, figsize=(14.2, 7.7))
    for index, kind in enumerate(("exception", "coherent")):
        for column, (pool, stratum, title) in enumerate(columns):
            ax = axes[index, column]
            selected = {
                r["world"]: r
                for r in checks
                if matches(
                    r,
                    kind=kind,
                    step=512,
                    pool=pool,
                    stratum=stratum,
                    control="same_truth_common_known",
                )
            }
            if set(selected) != set(WORLDS):
                raise ValueError("Common-U figure requires all worlds")
            finite = False
            for phase, color, marker in zip(PHASES, COLORS, ("o", "s"), strict=True):
                values = [selected[w][f"{phase}_damage_case_macro"] for w in WORLDS]
                finite |= any(v is not None for v in values)
                ax.plot(
                    range(8),
                    [100 * v if v is not None else np.nan for v in values],
                    color=color,
                    marker=marker,
                    markersize=3.5,
                    linewidth=1,
                )
            if not finite:
                ax.text(
                    0.5,
                    0.5,
                    "Undefined: no shared known facts",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=8,
                )
            clean(ax, False)
            ax.set_ylim(bottom=0)
            ax.set_xticks(range(8), [str(w) for w in WORLDS], fontsize=8)
            if index == 0:
                ax.set_title(title, fontsize=10, pad=10)
            if column == 0:
                ax.set_ylabel(f"{kind.title()} edits\nConditional damage (%)")
            if index == 1:
                ax.set_xlabel("World")
    handles = [
        plt.Line2D([], [], color=c, marker=m) for c, m in zip(COLORS, ("o", "s"), strict=True)
    ]
    fig.legend(
        handles,
        ("Low prevalence", "High prevalence"),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.071),
        ncol=2,
        frameon=False,
    )
    fig.subplots_adjust(left=0.062, right=0.99, top=0.9, bottom=0.17, wspace=0.35, hspace=0.23)
    finish(
        fig,
        output,
        "confirmation-common-knowledge-retention",
        synthetic,
        "Same truth, unchanged in both edits, correct before both edits. "
        "Case-macro within each world; "
        "shared-known coverage, pooled damage and undefined cases are reported separately.",
    )


def common_u_report(checks, output, synthetic):
    lines = [
        "# "
        + ("Synthetic layout only: " if synthetic else "")
        + "Paired common-knowledge retention",
        "",
        "These are descriptive controls, not additional confirmatory tests. Both phases use "
        "the intersection of unchanged U query IDs and both baselines must answer correctly. "
        "same_truth_common_known additionally requires identical truth in low/high worlds; "
        "common_known retains phase-specific truths and is reported separately. U_unseen "
        "excludes the union of both replay pools; U_heldout intersects the independently selected "
        "heldout pools. None replaces the phase-own U results. Both-old-correct is a "
        "post-training subset affected by the prevalence intervention; this conditional control "
        "is not the primary causal effect of prevalence on retention.",
        "",
        "Case-macro averages defined case ratios within each world, then equally weights "
        "worlds with defined values. Pooled first divides summed broken by shared-known counts "
        "within each world, then equally weights defined worlds. Counts below are repeated "
        "case/query instances across worlds, never independent sample sizes. Undefined "
        "zero-known cases remain missing, not zero. Learning is not repeated by edit supports.",
        "",
        "All four editing nodes and all eight worlds, with coverage and numerator/denominator "
        "counts, are preserved in [common-u-world-checks.csv](common-u-world-checks.csv). "
        "The table below shows step 512 for every control, kind, pool, and stratum.",
        "",
        "| Control / kind / pool / stratum | Low macro % | High macro % | Change pp | "
        "Low pooled % | High pooled % | Joint-known / eligible / pool instances | "
        "Valid cases low/high | Valid worlds low/high |",
        "|---|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for control, kind, pool, stratum in itertools.product(
        COMMON_CONTROLS, KINDS, COMMON_POOLS, range(-1, 5)
    ):
        selected = [
            r
            for r in checks
            if matches(r, control=control, kind=kind, pool=pool, stratum=stratum, step=512)
        ]
        values = []
        for field in (
            "low_damage_case_macro",
            "high_damage_case_macro",
            "high_minus_low_damage_case_macro",
            "low_damage_pooled",
            "high_damage_pooled",
        ):
            valid = [r[field] for r in selected if r[field] is not None]
            values.append(f"{100 * np.mean(valid):.4f}" if valid else "Undefined")
        count = "/".join(
            str(sum(r[k] for r in selected)) for k in ("known", "eligible_n", "pool_n")
        )
        valid_counts = "/".join(
            str(sum(r[f"{phase}_damage_valid_cases"] for r in selected)) for phase in PHASES
        )
        lines.append(
            f"| {control} / {kind} / {pool} / {stratum} | "
            + " | ".join(
                values
                + [
                    count,
                    valid_counts,
                    "/".join(
                        str(sum(r[f"{phase}_damage_case_macro"] is not None for r in selected))
                        for phase in PHASES
                    ),
                ]
            )
            + " |"
        )
    (output / "common-u-report.md").write_text("\n".join(lines) + "\n")


def report(results, checks, output, synthetic):
    title = "合成数据版式预览（不是实验结果）" if synthetic else "独立世界确认：行为效应及必要代价"
    lines = [
        f"# {title}",
        "",
        "以下三个确认终点完整呈现；高比例减低比例，以世界为独立推断单位。"
        "区间、Holm 校正与符号翻转敏感性均直接使用冻结统计实现。",
        "",
        "| 确认终点 | 均值变化 pp | 98.333% 同时区间 pp | Holm p | 状态 |",
        "|---|---:|---|---:|---|",
    ]
    for row in results["results"]:
        interval = row["ci_bonferroni"]
        formatted = (
            "不可估计"
            if interval is None
            else f"[{100 * interval[0]:+.3f}, {100 * interval[1]:+.3f}]"
        )
        lines.append(
            f"| {row['endpoint']} | {100 * row['mean']:+.3f} | {formatted} | "
            f"{row['holm_adjusted_p']:.6g} | {row['status']} |"
        )
    lines += [
        "",
        "## 同时呈现的描述性检查",
        "",
        "表中准确率与损伤单位为 %，NLL 保持原单位。每个世界先平均内部配对单元，"
        "再对有定义的世界等权平均；每项同时列出低/高比例的可用世界数。"
        "未额外检验或选择有利的次级结果。",
        "",
        "| 检查 | 低比例 | 高比例 | 高 − 低 | 可用世界数 低/高 |",
        "|---|---:|---:|---:|---:|",
    ]
    for metric in dict.fromkeys(r["metric"] for r in checks):
        scale = 1 if metric == "base_nll" else 100
        values = [points(checks, metric, phase) for phase in PHASES]
        means = [float(np.nanmean(v)) * scale if np.isfinite(v).any() else None for v in values]

        def fmt(value):
            return "未定义" if value is None else f"{value:.4f}"

        delta = None if any(v is None for v in means) else means[1] - means[0]
        available = [int(np.isfinite(v).sum()) for v in values]
        lines.append(
            f"| {metric} | {fmt(means[0])} | {fmt(means[1])} | {fmt(delta)} | "
            f"{available[0]}/{available[1]} |"
        )
    missing = [r for r in checks if r["undefined_cases"]]
    lines += [
        "",
        f"条件指标出现零已知分母的世界/条件/指标单元数：{len(missing)}。"
        "这些单元及各单元 available_cases、known_total、denominator_total 完整保留；"
        "未将无法定义的损伤率填为零。不同训练比例的已知覆盖可能不同。",
        "",
        "## Case-macro 与 pooled 保持率分母",
        "",
        "上表与图中 U 指标先平均每个编辑 case 的条件损伤 broken/known；known=0 的 "
        "case 保留为未定义。下表另列每世界内 sum(broken)/sum(known) 的 pooled 损伤，"
        "然后对八世界等权平均。二者不是同一个估计量。broken、known 和 n 总数是重复"
        "编辑/支持下的查询实例数，不是独立人员或独立世界样本量；不据此计算置信区间。"
        "每个 world 的两种估计及其分子分母完整保存在 descriptive-world-checks.csv。",
        "",
        "| 损伤检查 | 低 case-macro % | 低 pooled % | 高 case-macro % | 高 pooled % | "
        "低 broken/known | 高 broken/known |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for metric in dict.fromkeys(r["metric"] for r in checks if r["metric"].endswith("_damage")):
        cells, counts = [], []
        for phase in PHASES:
            selected = [r for r in checks if r["metric"] == metric and r["phase"] == phase]
            for field in ("value", "pooled_value"):
                valid = [r[field] for r in selected if r[field] is not None]
                cells.append(f"{100 * np.mean(valid):.4f}" if valid else "未定义")
            counts.append(
                f"{sum(r['numerator_total'] for r in selected)}/"
                f"{sum(r['pooled_denominator_total'] for r in selected)}"
            )
        lines.append(f"| {metric} | " + " | ".join(cells + counts) + " |")
    lines += [
        "",
        "## 组织与测试关系的交叉结果",
        "",
        "[完整组织报告](organization-report.md)无条件呈现所有固定学习 cohort、两种编辑"
        "的固定九人、全 D、heldout D、E 全体与 root/actual 分项。每项包含三组织×两测试链"
        "六个原始单元、匹配交互、company/project 各自匹配和不匹配查询相对 neither 的差异"
        "及双链平均收益；八个世界分别列低比例、高比例和高−低变化，不做次级显著性检验。",
        "",
        "## 共同已知且同真值的旧知识保持",
        "",
        "[配对保持控制报告](common-u-report.md)同时报告 unchanged U 交集、双侧原先均答对的"
        "共同已知子集，以及进一步要求同真值的预注册控制。两侧使用完全相同的已知分母，"
        "保留 full、unseen、heldout 和所有局部分层、覆盖与零分母。这一控制不能替代"
        "前述各 phase 自身旧知识上的破坏率。共同旧正确是例外比例干预后的条件子集，"
        "不代表例外比例对保持的主因果效应；均不增加显著性检验。",
        "",
        "## 解释范围",
        "",
        "该确认针对 758 万参数模型、固定合成数据生成器与训练/编辑预算下，训练相关性"
        "干预对原例外学习和编辑传播的行为效应。主编辑终点是 exception 更新的固定九名"
        "heldout 人员；coherent 更新的同九人是配对参照。学习没有按支持集重复；"
        "两初始化、三组织、两查询链与两编辑支持均为世界内因素。",
        "",
        "直接准确率要求目标值与 EOS 正确。自主两步最终成功要求第一步输出合法组织并"
        "以 EOS 终止，第二步输出目标城市并以 EOS 终止；第一步不必等于真值组织，"
        "因为错误组织也可能共享城市。第一步真值准确率 bridge_correct 与合法率 "
        "bridge_valid 在表中分别报告。两步使用模型预测组织，不输入真值桥接；"
        "它增加推理调用，不能直接等同于单次前向内部算法。对应根事实按人员加权，"
        "同一 root 可被多名人员重复引用。基础知识、全体查询、全 D、E 拟合、局部与"
        "全局 U 损伤及覆盖必须共同解释，不能只报告固定九人的收益。",
        "",
        "组织交互图只作描述，未追加显著性或等效性结论。世界级 t 推断有分布假设；"
        "符号翻转是依赖零假设符号对称性的敏感性分析。无显著差异不证明等效，"
        "行为确认也不唯一识别组织内部中介、电路或自然语言任务上的效果。",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n")


def demo_tables():
    """Fictitious arithmetic counts only; no dataset generator or outcome reader."""
    tables = {name: [] for name in TABLES}
    cohort_n = {
        "original_exception": 64,
        "newly_exception": 448,
        "remaining_ordinary": 512,
        "all": 1024,
    }
    for world, seed, condition, phase in itertools.product(WORLDS, (0, 1), CONDITIONS, PHASES):
        high = phase == "high"
        wobble = (world - 103.5) * 0.007 + (seed - 0.5) * 0.015
        base = dict(world=world, seed=seed, condition=condition, phase=phase)
        for step in STEPS:
            progress = step / 15360
            tables["learning-metrics.csv"].append(
                {
                    **base,
                    "step": step,
                    "base_accuracy": 0.03 + 0.95 * progress,
                    "base_nll": 7.7 * (1 - progress) ** 3 + 0.035 + abs(wobble),
                    "mean_heldout": (0.02 + 0.93 * progress) - (0.06 if high else 0),
                }
            )
        for chain in CHAINS:
            organization_bias = (
                0.0
                if condition == "neither"
                else (0.035 if high else 0.12)
                if condition == chain
                else (0.075 if high else 0.04)
            )
            identity = {**base, "chain": chain, "step": 15360}
            for cohort, n in cohort_n.items():
                rate = (
                    {
                        "original_exception": 0.43 + (0.31 if high else 0),
                        "newly_exception": 0.96 - (0.17 if high else 0),
                        "remaining_ordinary": 0.97 - (0.02 if high else 0),
                        "all": 0.94 - (0.05 if high else 0),
                    }[cohort]
                    + wobble
                    + organization_bias * (1 if cohort == "original_exception" else 0.1)
                )
                correct = int(np.clip(round(n * rate), 0, n))
                actual = int(n * (0.08 if high else 0.42)) if cohort == "original_exception" else 0
                row = {
                    **identity,
                    "method": "direct",
                    "split": "heldout",
                    "cohort_kind": "fixed",
                    "cohort": cohort,
                    "n": n,
                    "correct": correct,
                    "accuracy": correct / n,
                    "actual_on_conflict": min(actual, n - correct),
                }
                tables["learning-strata.csv"].append(row)
                if cohort == "original_exception":
                    tables["two-step-strata.csv"].append(
                        {
                            **row,
                            "method": "two_step",
                            "accuracy": 0.95 + abs(wobble) / 2,
                            "bridge_valid": n,
                            "bridge_correct": round(n * 0.95),
                        }
                    )
                    tables["learning-facts.csv"].append(
                        {
                            **identity,
                            "split": "heldout",
                            "cohort": cohort,
                            "n": n,
                            "actual_accuracy": 0.94 + 0.02 * high,
                            "membership_accuracy": 0.96,
                            "root_accuracy": 1.0,
                        }
                    )
                    tables["learning-endpoint.csv"].append(
                        {
                            **{
                                k: identity[k]
                                for k in ("world", "seed", "condition", "chain", "step")
                            },
                            "prevalence": phase,
                            "cohort": cohort,
                            "split": "heldout",
                            "n": n,
                            "correct": correct,
                        }
                    )
            for support, kind in itertools.product((0, 1), KINDS):
                delta = 0.14 if kind == "exception" else -0.12
                correct = int(
                    np.clip(
                        round(
                            9 * (0.4 + high * delta + wobble + organization_bias + 0.04 * support)
                        ),
                        0,
                        9,
                    )
                )
                row = {**base, "chain": chain, "support": support, "kind": kind, "step": 512}
                for name, n, rate in (
                    ("E", 39, 0.96),
                    ("E_roots", 3, 1.0),
                    ("E_actual", 36, 0.95),
                    ("D", 96, 0.68 - 0.06 * high),
                    ("D_heldout", 48, 0.55 - 0.04 * high),
                    ("paired_reference_D_heldout", 9, correct / 9),
                ):
                    row.update(
                        {
                            f"{name}_n": n,
                            f"{name}_accuracy": round(n * rate) / n,
                            f"{name}_correct": round(n * rate),
                        }
                    )
                for prefix, n in (
                    ("U_full", 20473),
                    ("U_heldout", 5000),
                    ("U_full_strata_0", 6),
                    ("U_full_strata_1", 42),
                    ("U_full_strata_2", 640),
                    ("U_full_strata_3", 6000),
                    ("U_full_strata_4", 13785),
                ):
                    coverage = 0.85 + 0.05 * high
                    damage = 0.008 + 0.004 * high + abs(wobble) / 10
                    if "strata" in prefix:
                        damage += 0.1 if kind == "exception" else 0.04
                    known = round(n * coverage)
                    broken = round(known * damage)
                    row.update(
                        {
                            f"{prefix}_n": n,
                            f"{prefix}_known": known,
                            f"{prefix}_coverage": known / n,
                            f"{prefix}_broken": broken,
                            f"{prefix}_damage": broken / known if known else None,
                        }
                    )
                tables["editing-all-nodes.csv"].append(row)
                tables["editing-endpoint.csv"].append(
                    {**row, "prevalence": phase, "n": 9, "correct": correct}
                )
    return tables


def demo_common_u():
    """Arithmetic matrix fixture only; no world generation or outcome reads."""
    rows = []
    for identity in itertools.product(
        WORLDS,
        (0, 1),
        CONDITIONS,
        CHAINS,
        (0, 1),
        KINDS,
        EDIT_STEPS,
        COMMON_POOLS,
        range(-1, 5),
        COMMON_CONTROLS,
    ):
        keys = (
            "world",
            "seed",
            "condition",
            "chain",
            "support",
            "kind",
            "step",
            "pool",
            "stratum",
            "control",
        )
        row = dict(zip(keys, identity, strict=True))
        n = 60 if row["stratum"] == -1 else 12
        eligible = n if row["control"] == "common_known" else n - 2
        known = eligible - 2
        low_broken = int(row["step"] > 0) * (1 + row["seed"])
        high_broken = int(row["step"] > 0) * (2 + row["seed"])
        row.update(
            stratum_name="all" if row["stratum"] == -1 else f"synthetic_stratum_{row['stratum']}",
            pool_n=n,
            eligible_n=eligible,
            known=known,
            low_baseline_known=known + 1,
            high_baseline_known=known,
            low_broken=low_broken,
            high_broken=high_broken,
            low_damage=low_broken / known,
            high_damage=high_broken / known,
            high_minus_low_damage=(high_broken - low_broken) / known,
            joint_known_coverage=known / eligible,
            joint_known_fraction_of_pool=known / n,
        )
        rows.append(row)
    return rows


def build(output, summary=None, statistics=None, synthetic=False, common_u=None):
    output = Path(output).resolve()
    if synthetic:
        tables, sources = demo_tables(), {}
        common_rows = demo_common_u()
        results = locked_statistics.analyze_endpoints(
            tables["learning-endpoint.csv"], tables["editing-endpoint.csv"]
        )
    else:
        if summary is None or statistics is None:
            raise ValueError("Formal reporting requires summary and statistics directories")
        if output.exists() and any(output.iterdir()):
            raise FileExistsError("Refusing to overwrite an existing formal report")
        if common_u is None:
            raise ValueError("Formal report requires audited --common-u controls")
        for source in (summary, statistics, common_u):
            source = Path(source).resolve()
            if output == source or output in source.parents or source in output.parents:
                raise ValueError("Report output must not overlap its audited inputs")
        tables, results, sources = load_audited(summary, statistics)
        common_rows = load_common_u(common_u, summary, sources)
    common_checks = common_u_checks(common_rows)
    checks, organization = collect_checks(tables), org_effects(tables)
    output.mkdir(parents=True, exist_ok=True)
    style()
    endpoint_figure(results, output, synthetic)
    learning_figure(checks, output, synthetic)
    editing_figure(checks, organization, output, synthetic)
    organization_figure(organization, output, synthetic)
    organization_report(organization, output, synthetic)
    common_u_figure(common_checks, output, synthetic)
    common_u_report(common_checks, output, synthetic)
    report(results, checks, output, synthetic)
    write_csv(output / "descriptive-world-checks.csv", checks)
    write_csv(output / "organization-world-checks.csv", organization)
    write_csv(output / "common-u-world-checks.csv", common_checks)
    for path, expected in sources.items():
        if sha(path) != expected:
            raise ValueError("Audited report input changed during rendering")
    metadata = dict(
        complete=True,
        synthetic_only=synthetic,
        sources_hashes=sources,
        script_sha256=sha(Path(__file__)),
        no_new_world_generation=True,
        inference="Frozen three-endpoint statistics only; other checks descriptive",
        files={
            p.name: sha(p)
            for p in output.iterdir()
            if p.is_file() and p.name != "report-audit.json"
        },
    )
    (output / "report-audit.json").write_text(json.dumps(metadata, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--statistics", type=Path)
    parser.add_argument("--common-u", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()
    build(args.output, args.summary, args.statistics, args.demo, args.common_u)


if __name__ == "__main__":
    main()
