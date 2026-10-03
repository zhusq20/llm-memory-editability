"""Recount saved predictions for all exploratory diagnosis contrasts."""

from __future__ import annotations

import csv
import json

import diagnose_realworld_loop as common


def main():
    rows, differences = [], {}
    comparisons = {
        "realworld-loop-diagnosis-v1": [("canonical", "native"), ("augmented", "canonical")],
        "realworld-loop-prefix-v1": [("compatible", "displaced")],
        "realworld-loop-entity-v1": [("entity", "natural")],
        "realworld-loop-entity-replica-v1": [("entity", "natural")],
    }
    for batch, pairs in comparisons.items():
        root = common.PROJECT / "results" / batch / "development"
        by_name = {}
        for path in sorted(root.glob("*/endpoint-predictions.json")):
            pred = json.loads(path.read_text())
            stored = json.loads((path.parent / "endpoint.json").read_text())
            for role in ["II", "IO", "OI", "OO"]:
                actual = common.aggregate([r for r in pred["test_all"] if r["role"] == role])
                assert actual == stored["test_" + role.lower()]
            assert common.aggregate(pred["test_all"]) == stored["test_all"]
            audit = path.parent / "audit.json"
            row = {
                "batch": batch,
                "name": path.parent.name,
                "audited": audit.exists(),
                "n_all": len(pred["test_all"]),
                "n_oo": stored["test_oo"]["n"],
            }
            for split in [
                "atomic",
                "train_composition",
                "test_all",
                "test_ii",
                "test_io",
                "test_oi",
                "test_oo",
            ]:
                row[split] = stored[split]["alias_em"] * 100
            rows.append(row)
            by_name[path.parent.name] = row
        for treatment, control in pairs:
            needed = [
                f"{arch}-{condition}"
                for arch in ["standard8", "loop4x2"]
                for condition in [treatment, control]
            ]
            if not all(name in by_name for name in needed):
                continue
            contrast = {}
            for metric in ["test_all", "test_ii", "test_oo"]:
                effects = {
                    arch: by_name[f"{arch}-{treatment}"][metric]
                    - by_name[f"{arch}-{control}"][metric]
                    for arch in ["standard8", "loop4x2"]
                }
                contrast[metric] = {
                    **effects,
                    "loop_minus_standard_interaction_pp": effects["loop4x2"] - effects["standard8"],
                }
            differences[f"{batch}:{treatment}-{control}"] = contrast
    art = common.ART
    common.write_json(
        art / "summary.json",
        {
            "created_utc": common.utc(),
            "runs": rows,
            "contrasts": differences,
            "scope": (
                "Exploratory single-initialization contrasts on an already-inspected split; "
                "entity batch uses width256 fresh models, other batches width768 continuations. "
                "Entity scoring requires node identity; natural scoring accepts aliases. "
                "Do not pool across batches or infer population significance."
            ),
        },
    )
    if rows:
        with (art / "endpoints.csv").open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps({"runs": rows, "contrasts": differences}, indent=2))


if __name__ == "__main__":
    main()
