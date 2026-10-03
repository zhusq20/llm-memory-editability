"""Recount fixed unique-surface queries and shared prerequisite diagnostics."""

import json
from pathlib import Path

import diagnose_realworld_loop as common

from llm_memory_editability.realworld_composition import answer_scores


def main():
    raw = json.loads(Path(common.BASE["data_file"]).read_text())
    tests = raw["evaluation_compositions"]
    unique = set(
        json.loads((common.ART / "entity-surface-audit.json").read_text())[
            "unique_surface_evaluation_ids"
        ]
    )
    reports = {}
    encoded_data = json.loads(
        (common.PROJECT / "data/realworld-loop-entity-v1/prepared.json").read_text()
    )
    lookup = {
        condition: {
            row["id"]: row
            for split in ["atoms", "train_compositions", "evaluation_compositions"]
            for row in data[split]
        }
        for condition, data in encoded_data.items()
    }
    rescored = 0
    for batch in ["realworld-loop-entity-v1", "realworld-loop-entity-replica-v1"]:
        root = common.PROJECT / "results" / batch / "development"
        names = [f"{a}-{c}" for a in ["standard8", "loop4x2"] for c in ["natural", "entity"]]
        files = [root / n / "endpoint-predictions.json" for n in names]
        if not all(p.exists() for p in files):
            continue
        predictions = {n: json.loads(p.read_text()) for n, p in zip(names, files, strict=True)}
        for name, pred in predictions.items():
            condition = name.split("-")[-1]
            for rows in pred.values():
                for row in rows:
                    target = lookup[condition][row["id"]]
                    assert row["gold"] == target["answer"]
                    actual = answer_scores(
                        row["prediction"], [target["answer"], *target.get("aliases", [])]
                    )[0]
                    assert actual == row["alias_em"]
                    rescored += 1
        knowledge = {
            n: {r["id"] for r in pred["atomic"] if r["alias_em"] == 1}
            for n, pred in predictions.items()
        }
        jointly_known = set.intersection(*knowledge.values())
        common_ids = {r["id"] for r in tests if all(a in jointly_known for a in r["atom_ids"])}
        sets = {"unique_surface_identity": unique, "four_model_facts_correct": common_ids}
        report = {}
        for subset, ids in sets.items():
            result = {}
            for role in ["all", "II", "IO", "OI", "OO"]:
                eligible = {
                    r["id"]
                    for r in tests
                    if r["id"] in ids and (role == "all" or r["role"] == role)
                }
                scores = {
                    n: common.aggregate([r for r in pred["test_all"] if r["id"] in eligible])
                    for n, pred in predictions.items()
                }
                result[role] = {
                    "n": len(eligible),
                    "coverage": len(eligible)
                    / sum(role == "all" or r["role"] == role for r in tests),
                    "accuracy_percent": {n: s["alias_em"] * 100 for n, s in scores.items()},
                }
            report[subset] = result
        reports[batch] = report
    common.write_json(
        common.ART / "entity-subsets.json",
        {
            "batches": reports,
            "independently_rescored_prediction_texts": rescored,
            "caveat": (
                "Unique-surface membership is model-independent. Common-knowledge membership "
                "is post-treatment and descriptive, not a separately identified causal effect."
            ),
        },
    )
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
