#!/usr/bin/env python3
"""Audit and summarize all fixed evaluation interventions repeated in FP32."""

from collections import defaultdict

import numpy as np
from report_twohop_frozen import ART, RESULTS, ci, digest, mean, read, rows, table, write


def main():
    checks, prediction, behavior = [], [], []

    def check(name, passed):
        checks.append(dict(name=name, passed=bool(passed)))
        if not passed:
            raise AssertionError(name)

    for model in ["small", "main"]:
        name = model + "-fp32-all-evaluation"
        done = read(ART / f"precision-completion-{model}-all-evaluation.json")
        check(name + ":complete", done["state"] == "complete" and done["cases"] == 48)
        for file, expected in done["files"].items():
            check(name + ":hash:" + file, digest(RESULTS / name / file) == expected)
        records = rows(RESULTS / name / "mechanism.jsonl")
        check(name + ":count", len(records) == 1296)
        check(
            name + ":unique",
            len({(r["dataset"], r["id"], r["layer"], r["condition"]) for r in records}) == 1296,
        )
        grouped = defaultdict(dict)
        for row in records:
            grouped[(row["dataset"], row["id"], row["layer"])][row["condition"]] = row
        for path in (RESULTS / name / "vectors").glob("*.npz"):
            v = np.load(path)
            prefix, layer = path.stem.rsplit("-layer-", 1)
            dataset, case_id = prefix.split("-", 1)
            selected = grouped[(dataset, case_id, int(layer))]
            names = list(read(ART / "data-lock.json")["config"]["patches"])
            px, pp = v["deltas"] @ v["grad_x"], v["delta_phi"] @ v["grad_phi"]
            gap = v["scores"][:, 0] - v["scores"][:, 1]
            for i, condition in enumerate(names):
                r = selected[condition]
                check(
                    f"{name}:recompute:{path.name}:{condition}",
                    np.isclose(px[i], r["predicted_input_change"], atol=1e-4, rtol=1e-4)
                    and np.isclose(pp[i], r["predicted_feature_change"], atol=1e-4, rtol=1e-4)
                    and np.isclose(gap[i] - gap[0], r["gap_change"], atol=1e-6),
                )
        for dataset in ["mquake", "2wiki"]:
            group = [r for r in records if r["dataset"] == dataset]
            for condition in ["all_nonzero", *names[1:]]:
                part = [
                    r
                    for r in group
                    if r["condition"] != "zero"
                    and (condition == "all_nonzero" or r["condition"] == condition)
                ]
                y = np.array([r["gap_change"] for r in part])
                px = np.array([r["predicted_input_change"] for r in part])
                pp = np.array([r["predicted_feature_change"] for r in part])
                differences = defaultdict(list)
                for r, value in zip(part, abs(y - px) - abs(y - pp), strict=True):
                    differences[r["id"]].append(value)
                interval = ci([np.mean(v) for v in differences.values()])
                prediction.append(
                    dict(
                        model=model,
                        dataset=dataset,
                        condition=condition,
                        cases=len(differences),
                        effects=len(part),
                        zero_mae=float(abs(y).mean()),
                        input_mae=float(abs(y - px).mean()),
                        feature_mae=float(abs(y - pp).mean()),
                        feature_advantage=float((abs(y - px) - abs(y - pp)).mean()),
                        lower=interval[0],
                        upper=interval[1],
                        max_batch_gap_difference=max(
                            abs(r["probe_batch_gap_difference"]) for r in part
                        ),
                    )
                )
            for layer in sorted({r["layer"] for r in group}):
                for condition in names:
                    part = [r for r in group if r["layer"] == layer and r["condition"] == condition]
                    changes = [
                        int(r["metrics"]["em"])
                        - int(grouped[(dataset, r["id"], layer)]["zero"]["metrics"]["em"])
                        for r in part
                    ]
                    interval = ci(changes)
                    behavior.append(
                        dict(
                            model=model,
                            dataset=dataset,
                            layer=layer,
                            condition=condition,
                            n=len(part),
                            em=mean([r["metrics"]["em"] for r in part]),
                            delta_em=mean(changes),
                            lower=interval[0],
                            upper=interval[1],
                            rescued=sum(c == 1 for c in changes),
                            harmed=sum(c == -1 for c in changes),
                            wrong_successor_em=mean([r["wrong_successor_em"] for r in part]),
                        )
                    )
    table(ART / "full-precision-prediction.csv", prediction)
    table(ART / "full-precision-behavior.csv", behavior)
    write(ART / "full-precision-summary.json", dict(prediction=prediction, behavior=behavior))
    write(
        ART / "full-precision-audit.json",
        dict(
            summary=dict(total=len(checks), passed=sum(c["passed"] for c in checks)), checks=checks
        ),
    )
    print({"checks": len(checks), "predictions": len(prediction), "behavior": len(behavior)})


if __name__ == "__main__":
    main()
