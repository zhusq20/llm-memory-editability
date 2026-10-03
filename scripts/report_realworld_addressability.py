"""Re-score graph addressability predictions and compute matched interactions."""

import json

import diagnose_realworld_addressability as experiment

from llm_memory_editability.realworld_composition import answer_scores

common = experiment.common


def main():
    data = json.loads(experiment.DATA.read_text())
    targets = {
        condition: {
            r["id"]: r
            for key in ["atoms", "train_compositions", "evaluation_compositions"]
            for r in value[key]
        }
        for condition, value in data.items()
    }
    reports = {}
    for step, root_name in [
        (32000, "realworld-loop-addressability-midpoint-v1"),
        (128000, "realworld-loop-addressability-v1"),
    ]:
        root = common.PROJECT / "results" / root_name / "development"
        rows = {}
        for path in sorted(root.glob("*/endpoint.json")):
            condition = path.parent.name.split("-")[-1]
            saved = json.loads(path.read_text())
            pred = json.loads((path.parent / "endpoint-predictions.json").read_text())
            count = 0
            for group, predictions in pred.items():
                for row in predictions:
                    target = targets[condition][row["id"]]
                    assert row["gold"] == target["answer"]
                    assert (
                        row["alias_em"] == answer_scores(row["prediction"], [target["answer"]])[0]
                    )
                    count += 1
                assert common.aggregate(predictions) == saved[group]
            for role in ["II", "IO", "OI", "OO"]:
                assert (
                    common.aggregate([r for r in pred["test_all"] if r["role"] == role])
                    == saved["test_" + role.lower()]
                )
            rows[path.parent.name] = {
                "completed": (path.parent / "complete.json").exists(),
                "rescored_prediction_texts": count,
                **{
                    key: saved[key]["alias_em"] * 100
                    for key in ["atomic", "train_composition", "test_all", "test_ii", "test_oo"]
                },
            }
        gaps = {}
        if len(rows) == 4:
            gaps = {
                condition: rows[f"loop4x2-{condition}"]["test_oo"]
                - rows[f"standard8-{condition}"]["test_oo"]
                for condition in ["shared", "unique"]
            }
            gaps["shared_minus_unique_interaction_pp"] = gaps["shared"] - gaps["unique"]
        reports[str(step)] = {"runs": rows, "loop_minus_standard_oo_pp": gaps}
    common.write_json(experiment.ART / "summary.json", reports)
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
