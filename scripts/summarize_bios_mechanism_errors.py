"""Reproducible P0 error attribution with a frozen source-CSV identity.

Pending analysis is allowed, but --require-complete fails closed until the original
P0 source audit is complete. Previously recorded CSV hashes cannot silently change.
"""

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/bios-mechanism-dev-v1/p0"
CSV_INPUTS = ("learning-strata.csv", "editing-strata.csv", "propagation-queries.csv")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked_source_identity(source, output, require_complete=False):
    audit_path = source / "audit.json"
    audit = json.loads(audit_path.read_text()) if audit_path.exists() else {"complete": False}
    if require_complete and not audit.get("complete"):
        raise ValueError("Formal H2 analysis requires a complete original source audit")
    if audit.get("complete"):
        expected = {
            "models": 48,
            "learning_checkpoints": 288,
            "edit_cases": 384,
            "edit_checkpoints": 1536,
            "scored_query_predictions": 37588992,
            "classified_derived_predictions": 7471104,
            "source_hashes_rechecked_after_analysis": True,
            "source_data_modified": False,
            "errors": [],
        }
        if any(audit.get(k) != v for k, v in expected.items()):
            raise ValueError("Original source audit is incomplete or has errors")
        if sha(source / "sources.json") != audit["sources_sha256"]:
            raise ValueError("Original source audit manifest identity changed")
    hashes = {name: sha(source / name) for name in CSV_INPUTS}
    previous_path = output / "h2-analysis.json"
    if previous_path.exists():
        previous = json.loads(previous_path.read_text())
        for name, digest in hashes.items():
            if previous["inputs"].get(name) != digest:
                raise ValueError(f"Source CSV identity changed: {name}")
        if previous.get("complete"):
            if not audit.get("complete") or previous["inputs"].get("audit.json") != sha(audit_path):
                raise ValueError("Previously completed source audit identity changed")
    if audit.get("complete"):
        hashes["audit.json"] = sha(audit_path)
        hashes["sources.json"] = sha(source / "sources.json")
    return audit, hashes


def read(source, name):
    with (source / name).open() as stream:
        yield from csv.DictReader(stream)


def save(output, name, rows):
    with (output / name).open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def alignment(condition, chain):
    return (
        "aligned"
        if condition == chain
        else ("neither" if condition == "neither" else "other_chain")
    )


def mean(rows, key):
    return sum(float(row[key]) / int(row["n"]) for row in rows) / len(rows)


def analyze(source=BASE, output=BASE, require_complete=False):
    source, output = Path(source), Path(output)
    audit, inputs_hashes = checked_source_identity(source, output, require_complete)
    outputs = {}
    groups = defaultdict(list)
    learning_count = 0
    for row in read(source, "learning-strata.csv"):
        learning_count += 1
        parts = row["subset"].split("/")
        if row["step"] != "15360" or len(parts) != 4 or parts[2] != "heldout" or parts[3] == "all":
            continue
        for aligned in ("all", alignment(row["condition"], parts[0])):
            groups[int(row["width"]), aligned, parts[3]].append(row)
    if learning_count != 7776:
        raise ValueError("Incomplete learning summary grid")
    learning = []
    for (width, aligned, population), rows in sorted(groups.items()):
        errors = sum(int(r["n"]) - int(r["correct"]) for r in rows)
        out = dict(
            width=width,
            alignment=aligned,
            population=population,
            cases=len(rows),
            n=sum(int(r["n"]) for r in rows),
            errors=errors,
            accuracy=mean(rows, "correct"),
        )
        for key in (
            "matches_actual",
            "unique_actual",
            "other_answer",
            "termination_error",
            "indistinguishable_wrong",
        ):
            out[key + "_per_query"] = mean(rows, key)
            out[key + "_among_errors"] = (
                sum(int(r[key]) for r in rows) / errors
                if errors and population == "old_exception"
                else None
            )
        learning.append(out)
    outputs["h2-learning.csv"] = learning

    groups = defaultdict(list)
    wanted = (
        "E/default",
        "E/actual",
        "D/heldout/all",
        "D/heldout/manipulated_conflict_cohort",
        "D/heldout/old_exception",
        "D/heldout/no_factual_conflict",
        "U/full/all",
        "U/full/stratum_0",
    )
    editing_count = 0
    for row in read(source, "editing-strata.csv"):
        editing_count += 1
        if row["step"] != "512" or row["subset"] not in wanted or not int(row["n"]):
            continue
        for aligned in ("all", alignment(row["condition"], row["chain"])):
            groups[int(row["width"]), row["scope"], row["kind"], aligned, row["subset"]].append(row)
    if editing_count != 47616:
        raise ValueError("Incomplete editing summary grid")
    editing = []
    for (width, scope, kind, aligned, subset), rows in sorted(groups.items()):
        out = dict(
            width=width,
            scope=scope,
            kind=kind,
            alignment=aligned,
            subset=subset,
            cases=len(rows),
            n=sum(int(r["n"]) for r in rows),
            accuracy=mean(rows, "correct"),
        )
        for key in (
            "matches_new_actual",
            "unique_new_actual",
            "matches_old_default",
            "matches_old_actual",
            "other_answer",
            "indistinguishable_wrong",
            "termination_error",
        ):
            out[key + "_per_query"] = mean(rows, key)
        old_known = sum(int(r["old_known"]) for r in rows)
        out["old_known"] = old_known
        out["pooled_damage_rate"] = (
            sum(int(r["old_known_now_wrong"]) for r in rows) / old_known
            if subset.startswith("U/") and old_known
            else None
        )
        editing.append(out)
    outputs["h2-editing.csv"] = editing

    signatures, denominators, errors = Counter(), Counter(), Counter()
    prediction_count = 0
    for row in read(source, "propagation-queries.csv"):
        prediction_count += 1
        if not (
            row["step"] == "512"
            and row["kind"] == "exception"
            and row["heldout"] == "True"
            and row["manipulated_conflict_cohort"] == "True"
        ):
            continue
        key = (int(row["width"]), row["scope"])
        denominators[key] += 1
        if row["correct"] == "True":
            continue
        errors[key] += 1
        signature = "EOS_FAILURE" if row["ended"] != "True" else (row["matched_roles"] or "OTHER")
        signatures[(*key, signature)] += 1
    if prediction_count != 147456:
        raise ValueError("Incomplete propagation query grid")
    signature_rows = []
    for (width, scope, signature), count in sorted(signatures.items()):
        key = width, scope
        signature_rows.append(
            dict(
                width=width,
                scope=scope,
                matched_roles=signature,
                count=count,
                total_queries=denominators[key],
                errors=errors[key],
                fraction_queries=count / denominators[key],
                fraction_errors=count / errors[key],
            )
        )
    outputs["h2-editing-error-signatures.csv"] = signature_rows

    for name, digest in inputs_hashes.items():
        if sha(source / name) != digest:
            raise ValueError(f"Source changed during H2 analysis: {name}")
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in outputs.items():
        save(output, name, rows)
    record = dict(
        complete=audit["complete"],
        status="audited" if audit["complete"] else "SOURCE_AUDIT_PENDING",
        models=48,
        edit_cases=384,
        analysis="Descriptive answer matching; no internal-route causal claim.",
        aggregate="Model/chain/edit cases equal weighted; pooled error fractions also labelled.",
        inputs=inputs_hashes,
        sources_hashes=inputs_hashes,
        source=str(source.resolve()),
        script_sha256=sha(__file__),
    )
    (output / "h2-analysis.json").write_text(json.dumps(record, indent=2) + "\n")
    print(
        json.dumps(
            {
                "learning_rows": len(learning),
                "editing_rows": len(editing),
                "error_signatures": len(signature_rows),
                "complete": audit["complete"],
            }
        )
    )


def analyze_two_step(source=BASE, output=BASE):
    source, output = Path(source), Path(output)
    checked_source_identity(source, output, require_complete=True)
    audit = json.loads((source / "summary/two-step-audit.json").read_text())
    if (
        not audit["complete"]
        or audit["states"] != 528
        or audit["chain_files"] != 1056
        or audit["models"] != 48
    ):
        raise ValueError("Two-step matrix audit is incomplete")
    inputs = {
        name: sha(source / name)
        for name in (
            "summary/two-step-audit.json",
            "summary/two-step-strata.csv",
            "summary/two-step-sources.json",
        )
    }
    previous = output / "h2-two-step-analysis.json"
    if previous.exists() and json.loads(previous.read_text())["sources_hashes"] != inputs:
        raise ValueError("Two-step summary identity changed")
    groups = defaultdict(list)
    with (source / "summary/two-step-strata.csv").open() as stream:
        for row in csv.DictReader(stream):
            phase = row["phase"]
            if phase == "editing" and not row["subset"].startswith("D/"):
                continue
            if row["subset"] not in (
                "QA/heldout/ordinary",
                "QA/heldout/old_exception",
                "D/heldout/all",
                "D/heldout/manipulated_conflict_cohort",
                "D/heldout/old_exception",
            ):
                continue
            chain = row["test_chain"]
            condition = row["condition"]
            relation = (
                "aligned"
                if condition == chain
                else ("neither" if condition == "neither" else "other_chain")
            )
            for align in ("all", relation):
                groups[
                    int(row["width"]),
                    phase,
                    int(row["step"]),
                    row["kind"],
                    row["scope"],
                    align,
                    row["subset"],
                ].append(row)
    out = []
    for (width, phase, step, kind, scope, align, subset), rows in sorted(groups.items()):
        data = dict(
            width=width,
            phase=phase,
            step=step,
            kind=kind,
            scope=scope,
            alignment=align,
            subset=subset,
            cases=len(rows),
            n=sum(int(r["n"]) for r in rows),
        )
        for key in (
            "direct_correct",
            "two_step_correct",
            "bridge_correct",
            "bridge_valid",
            "direct_only",
            "two_step_only",
            "both_wrong",
            "second_termination_error",
            "invalid_bridge",
        ):
            data[key + "_per_query"] = sum(int(r[key]) / int(r["n"]) for r in rows) / len(rows)
        errors = sum(int(r["n"]) - int(r["direct_correct"]) for r in rows)
        correct = sum(int(r["direct_correct"]) for r in rows)
        data["pooled_rescue_of_direct_errors"] = (
            sum(int(r["two_step_only"]) for r in rows) / errors if errors else None
        )
        data["pooled_harm_of_direct_correct"] = (
            sum(int(r["direct_only"]) for r in rows) / correct if correct else None
        )
        out.append(data)
    for name, digest in inputs.items():
        if sha(source / name) != digest:
            raise ValueError("Two-step source changed during analysis")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "h2-two-step.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(out[0]))
        writer.writeheader()
        writer.writerows(out)
    record = dict(
        complete=True,
        states=528,
        chain_files=1056,
        sources_hashes=inputs,
        source_sha256=hashlib.sha256(
            (source / "summary/two-step-strata.csv").read_bytes()
        ).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        interpretation="External two-call retrieval with model-predicted bridge; "
        "not evidence of identical internal computation.",
    )
    (output / "h2-two-step-analysis.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"complete": True, "rows": len(out)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=BASE)
    parser.add_argument("--output", type=Path, default=BASE)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument(
        "--two-step",
        action="store_true",
        help="Summarize autonomous two-step outputs; both source audits must be complete",
    )
    args = parser.parse_args()
    if args.two_step:
        analyze_two_step(args.source, args.output)
    else:
        analyze(args.source, args.output, args.require_complete)


if __name__ == "__main__":
    main()
