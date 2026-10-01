"""Publish locked world-level confirmation statistics, or its outcome-free design."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from llm_memory_editability import bios_confirmation_stats as statistics


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def power_sensitivity():
    """Conditional model-based sensitivity; these are not experimental outcomes."""
    draws, seed = 2_000_000, 92027
    rng = np.random.default_rng(seed)
    normal = rng.normal(size=draws)
    scale = np.sqrt(rng.chisquare(7, size=draws) / 7)
    rows = []
    for alpha in (0.05, 0.05 / 3):
        critical = statistics.t_critical(alpha, 7)
        for sd_pp in (5, 10, 15, 20):
            for effect_pp in (5, 100 / 9):
                power = float(
                    np.mean(np.abs((normal + effect_pp * np.sqrt(8) / sd_pp) / scale) > critical)
                )
                rows.append(
                    {
                        "alpha": alpha,
                        "critical_t_df7": critical,
                        "world_sd_pp_assumed": sd_pp,
                        "effect_pp_assumed": effect_pp,
                        "power": power,
                        "monte_carlo_se": float(np.sqrt(power * (1 - power) / draws)),
                        "interval_halfwidth_per_observed_sd": critical / np.sqrt(8),
                    }
                )
    return {
        "purpose": "Outcome-free sensitivity, not a guarantee that eight worlds are powered",
        "assumption": "Eight independent normal world-level paired differences; "
        "two-sided t test; .05/3 is the most stringent Holm3 threshold, "
        "not the joint probability of rejecting all three endpoints",
        "simulation_seed": seed,
        "simulation_draws": draws,
        "development_worlds": 2,
        "development_variance_degrees_of_freedom": 1,
        "development_world_edit_gain_pp": [13.88888888888889, 8.333333333333333],
        "development_world_coherent_change_pp": [-9.25925925925926, -26.85185185185185],
        "development_world_learning_gain_pp": [39.97395833333333, 35.28645833333333],
        "limitation": "Development has one support and two reused worlds; its variance "
        "does not determine the variance of two-support confirmation. Fixed N=8; "
        "no optional stopping or extension based on outcomes.",
        "rows": rows,
    }


def write_design(output):
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "statistics-contract.json", statistics.DEFAULT_STATISTICS)
    write_json(output / "power-sensitivity.json", power_sensitivity())
    sources = {
        str(Path(__file__).resolve()): digest(__file__),
        str(Path(statistics.__file__).resolve()): digest(statistics.__file__),
    }
    write_json(
        output / "statistics-design-manifest.json",
        {
            "source_hashes": sources,
            "files": {
                name: digest(output / name)
                for name in ("statistics-contract.json", "power-sensitivity.json")
            },
            "new_worlds_generated_or_read": False,
        },
    )


def analyze(summary, output, locked_design):
    summary, output, locked_design = map(
        lambda p: Path(p).resolve(), (summary, output, locked_design)
    )
    if output == summary or output == locked_design:
        raise ValueError("Statistics output must not overwrite its input or design directory")
    manifest = json.loads((locked_design / "statistics-design-manifest.json").read_text())
    for path, expected in manifest["source_hashes"].items():
        if digest(path) != expected:
            raise ValueError(f"Frozen statistics source changed: {path}")
    for name, expected in manifest["files"].items():
        if digest(locked_design / name) != expected:
            raise ValueError(f"Frozen statistics design changed: {name}")
    contract = json.loads((locked_design / "statistics-contract.json").read_text())
    if contract != statistics.DEFAULT_STATISTICS:
        raise ValueError("Statistics contract differs from frozen defaults")
    audit_path = summary / "audit.json"
    audit = json.loads(audit_path.read_text())
    if audit.get("complete") is not True:
        raise ValueError("Endpoint source audit is incomplete")
    archive_hash = audit.get("archive_index_sha256", "")
    if audit.get("weight_archive_verified") is not True or (
        len(archive_hash) != 64 or any(c not in "0123456789abcdef" for c in archive_hash)
    ):
        raise ValueError("Endpoint source lacks a verified complete weight-archive identity")
    inputs = {}
    sources = {
        str(audit_path): digest(audit_path),
        str(locked_design / "statistics-design-manifest.json"): digest(
            locked_design / "statistics-design-manifest.json"
        ),
    }
    for name in ("learning-endpoint.csv", "editing-endpoint.csv"):
        path = summary / name
        observed = digest(path)
        if audit.get("outputs_sha256", {}).get(name) != observed:
            raise ValueError(f"Audited endpoint hash mismatch: {name}")
        sources[str(path)] = observed
        with path.open(newline="") as handle:
            inputs[name] = list(csv.DictReader(handle))
    result = statistics.analyze_endpoints(
        inputs["learning-endpoint.csv"], inputs["editing-endpoint.csv"]
    )
    for path, expected in sources.items():
        if digest(path) != expected:
            raise ValueError(f"Statistics input changed during analysis: {path}")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "confirmation-statistics.json", result)
    write_csv(output / "world-effects.csv", result["world_effects"])
    write_csv(output / "paired-cases.csv", result["paired_cases"])
    write_json(
        output / "statistics-audit.json",
        {
            "complete": True,
            "world_count": 8,
            "input_rows": result["input_rows"],
            "source_hashes": sources,
            "statistics_source_hashes": manifest["source_hashes"],
            "frozen_design": str(locked_design),
            "weight_archive_verified": True,
            "archive_index_sha256": archive_hash,
            "no_query_level_inference": True,
            "learning_has_no_support_replication": True,
            "all_endpoints_published_regardless_of_significance": True,
        },
    )
    lines = [
        "# Independent world confirmation: narrower behavioral contrast",
        "",
        "High minus low exception prevalence; eight world-level paired differences. "
        "All three endpoints are published. E, global D, U and foundation-fact checks "
        "remain mandatory in the independent endpoint summary.",
        "",
        "| Endpoint | Mean (pp) | 95% marginal CI (pp) | 98.333% simultaneous CI (pp) "
        "| Raw two-sided p | Holm3 p | Symmetry sensitivity p | Status |",
        "|---|---:|---|---|---:|---:|---:|---|",
    ]
    for row in result["results"]:

        def interval(key, record=row):
            value = record[key]
            return (
                "not estimable"
                if value is None
                else f"[{100 * value[0]:+.3f}, {100 * value[1]:+.3f}]"
            )

        p = "not estimable" if row["p"] is None else f"{row['p']:.6g}"
        lines.append(
            f"| {row['endpoint']} | {100 * row['mean']:+.3f} | {interval('ci95')} | "
            f"{interval('ci_bonferroni')} | {p} | {row['holm_adjusted_p']:.6g} | "
            f"{row['sign_flip_sensitivity']['p']:.6g} | {row['status']} |"
        )
    lines += [
        "",
        "The primary editing endpoint is editing_exception_fixed9. The coherent "
        "endpoint uses the same nine people as a paired reference, not a conflict cohort. "
        "Learning averages 12 nested pairs/world; each editing endpoint averages 24 "
        "nested pairs/world, including both disjoint supports. These are not extra worlds.",
        "",
        "The t analysis assumes independent worlds and approximately normal world "
        "differences. Sign flips enumerate all 256 signs and require null sign symmetry; "
        "they are not an unconditional randomized-treatment test. Degenerate results "
        "and nonsignificant organization interactions do not establish equivalence. "
        "The study does not establish organization-path mediation or automatically "
        "trigger a new editor or natural-language experiment.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    return {"complete": True, "worlds": 8, "output": str(output)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design-only", action="store_true")
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--locked-design", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.design_only:
        if args.summary or args.locked_design:
            parser.error("Design-only mode cannot read endpoint summaries")
        write_design(args.output)
        print(json.dumps({"design_only": True, "output": str(args.output)}))
    else:
        if not args.summary or not args.locked_design:
            parser.error("Analysis requires an audited summary and a frozen design")
        print(json.dumps(analyze(args.summary, args.output, args.locked_design)))


if __name__ == "__main__":
    main()
