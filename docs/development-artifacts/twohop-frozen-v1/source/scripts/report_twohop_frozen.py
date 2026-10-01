#!/usr/bin/env python3
"""Summarize every frozen condition and audit saved finite-perturbation predictions."""

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from prepare_twohop_frozen import ART, DATA, ROOT, digest, read, write

RESULTS = ROOT / "results/twohop-frozen-v1"


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def mean(values):
    return float(np.mean(values)) if len(values) else None


def ci(values):
    if not len(values):
        return [None, None]
    x = np.asarray(values, dtype=float)
    rng = np.random.default_rng(142913)
    samples = x[rng.integers(0, len(x), (2000, len(x)))].mean(1)
    return list(map(float, np.quantile(samples, [0.025, 0.975])))


def table(path, data):
    if not data:
        return
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)


def main():
    cfg = read(ROOT / "configs/twohop-frozen-v1.json")
    cases = read(DATA / "cases.json")
    behavior_summary, effects, mechanism_summary, prediction_summary = [], [], [], []
    checks, examples, precision_summary = [], [], []

    def check(name, passed):
        checks.append(dict(name=name, passed=bool(passed)))
        if not passed:
            raise AssertionError(name)

    lock = read(ART / "data-lock.json")
    check("data_hash", digest(DATA / "cases.json") == lock["cases_sha256"])
    check("config_hash", digest(ROOT / "configs/twohop-frozen-v1.json") == lock["config_sha256"])
    for model in cfg["models"]:
        done = read(ART / f"completion-{model}.json")
        execution = read(ART / f"execution-lock-{model}.json")
        check(
            f"{model}:finished_frozen", done["state"] == "complete" and done["unchanged_parameters"]
        )
        check(
            f"{model}:parameter_hash",
            execution["parameter_sha256"] == done["parameter_sha256_after"],
        )
        check(
            f"{model}:runner_hash",
            execution["script_sha256"] == digest(ROOT / "scripts/run_twohop_frozen.py"),
        )
        check(
            f"{model}:config_hash",
            execution["config_sha256"] == digest(ROOT / "configs/twohop-frozen-v1.json"),
        )
        for file, expected in done["files"].items():
            check(f"{model}:hash:{file}", digest(RESULTS / model / file) == expected)
        behavior = rows(RESULTS / model / "behavior.jsonl")
        mechanism = rows(RESULTS / model / "mechanism.jsonl")
        check(f"{model}:behavior_count", len(behavior) == 288)
        check(f"{model}:behavior_unique", len({(r["dataset"], r["id"]) for r in behavior}) == 288)
        expected = {
            (r["dataset"], r["id"], layer, condition)
            for r in cases
            if r["mechanism"] and r["wrong_donor"]
            for layer in execution["layers"]
            for condition in cfg["patches"]
        }
        actual = [
            (r["dataset"], r["id"], r["layer"], r["condition"])
            for r in mechanism
            if r["status"] == "complete"
        ]
        check(
            f"{model}:mechanism_complete", set(actual) == expected and len(actual) == len(expected)
        )
        for path in sorted((RESULTS / model / "vectors").glob("*.npz")):
            v = np.load(path)
            parts = path.stem.rsplit("-layer-", 1)
            dataset, case_id = parts[0].split("-", 1)
            selected = {
                r["condition"]: r
                for r in mechanism
                if r["status"] == "complete"
                and r["dataset"] == dataset
                and r["id"] == case_id
                and r["layer"] == int(parts[1])
            }
            px = v["deltas"] @ v["grad_x"]
            pp = v["delta_phi"] @ v["grad_phi"]
            gap = v["scores"][:, 0] - v["scores"][:, 1]
            for index, condition in enumerate(cfg["patches"]):
                r = selected[condition]
                check(
                    f"{model}:recompute:{path.name}:{condition}",
                    np.isclose(px[index], r["predicted_input_change"], atol=1e-3, rtol=1e-4)
                    and np.isclose(pp[index], r["predicted_feature_change"], atol=1e-3, rtol=1e-4)
                    and np.isclose(gap[index] - gap[0], r["gap_change"], atol=1e-6),
                )
            norms = np.linalg.norm(v["deltas"], axis=1)
            names = list(cfg["patches"])
            check(
                f"{model}:norm_controls:{path.name}",
                np.allclose(
                    norms[[names.index(n) for n in ["wrong_oracle", "random", "reverse"]]],
                    norms[names.index("oracle_050")],
                    atol=1e-4,
                    rtol=1e-5,
                ),
            )
        for dataset in ["mquake", "2wiki"]:
            for split in ["development", "evaluation"]:
                group = [r for r in behavior if r["dataset"] == dataset and r["split"] == split]
                mastered = [
                    r
                    for r in group
                    if r["metrics"]["first"]["em"] and r["metrics"]["second_oracle"]["em"]
                ]
                alias_mastered = [
                    r
                    for r in group
                    if r["metrics"]["first"]["alias_em"]
                    and r["metrics"]["second_oracle"]["alias_em"]
                ]
                for condition in group[0]["metrics"]:
                    metric = [r["metrics"][condition] for r in group]
                    behavior_summary.append(
                        dict(
                            model=model,
                            dataset=dataset,
                            split=split,
                            condition=condition,
                            n=len(group),
                            em=mean([m["em"] for m in metric]),
                            f1=mean([m["f1"] for m in metric]),
                            alias_em=mean([m["alias_em"] for m in metric]),
                            filtered_alias_em=mean(
                                [m["filtered_alias_em"] for m in metric if "filtered_alias_em" in m]
                            ),
                            mastered_n=len(mastered),
                            conditional_em=mean([r["metrics"][condition]["em"] for r in mastered]),
                            alias_mastered_n=len(alias_mastered),
                            conditional_alias_em=mean(
                                [r["metrics"][condition]["alias_em"] for r in alias_mastered]
                            ),
                            cot_missing_final_marker=sum(
                                "Final answer:" not in r["outputs"]["cot"] for r in group
                            )
                            if condition == "cot"
                            else None,
                        )
                    )
                for condition in ["self_bridge", "oracle_bridge", "scaffold_chain", "cot"]:
                    differences = [
                        int(r["metrics"][condition]["em"]) - int(r["metrics"]["direct"]["em"])
                        for r in group
                    ]
                    interval = ci(differences)
                    effects.append(
                        dict(
                            model=model,
                            dataset=dataset,
                            split=split,
                            condition=condition,
                            n=len(group),
                            delta_em=mean(differences),
                            lower=interval[0],
                            upper=interval[1],
                        )
                    )
                complete = [
                    r
                    for r in mechanism
                    if r["status"] == "complete" and r["dataset"] == dataset and r["split"] == split
                ]
                for layer in execution["layers"]:
                    baseline = {
                        r["id"]: r
                        for r in complete
                        if r["layer"] == layer and r["condition"] == "zero"
                    }
                    for condition in cfg["patches"]:
                        part = [
                            r
                            for r in complete
                            if r["layer"] == layer and r["condition"] == condition
                        ]
                        if not part:
                            continue
                        differences = [
                            int(r["metrics"]["em"]) - int(baseline[r["id"]]["metrics"]["em"])
                            for r in part
                        ]
                        interval = ci(differences)
                        mechanism_summary.append(
                            dict(
                                model=model,
                                dataset=dataset,
                                split=split,
                                layer=layer,
                                condition=condition,
                                n=len(part),
                                em=mean([r["metrics"]["em"] for r in part]),
                                alias_em=mean([r["metrics"]["alias_em"] for r in part]),
                                delta_em=mean(differences),
                                lower=interval[0],
                                upper=interval[1],
                                rescued=sum(d == 1 for d in differences),
                                harmed=sum(d == -1 for d in differences),
                                wrong_successor_em=mean([r["wrong_successor_em"] for r in part]),
                                gap_change=mean([r["gap_change"] for r in part]),
                            )
                        )
                nonzero = [r for r in complete if r["condition"] != "zero"]
                for condition in ["all_nonzero", *[n for n in cfg["patches"] if n != "zero"]]:
                    part = (
                        nonzero
                        if condition == "all_nonzero"
                        else [r for r in nonzero if r["condition"] == condition]
                    )
                    if not part:
                        continue
                    measured = np.array([r["gap_change"] for r in part])
                    px = np.array([r["predicted_input_change"] for r in part])
                    pp = np.array([r["predicted_feature_change"] for r in part])
                    per_case = defaultdict(list)
                    for r, advantage in zip(part, abs(measured) - abs(measured - pp), strict=True):
                        per_case[r["id"]].append(float(advantage))
                    vs_zero = ci([np.mean(v) for v in per_case.values()])
                    per_case = defaultdict(list)
                    for r, advantage in zip(
                        part, abs(measured - px) - abs(measured - pp), strict=True
                    ):
                        per_case[r["id"]].append(float(advantage))
                    interval = ci([np.mean(v) for v in per_case.values()])
                    selected = abs(measured) > 0.05
                    prediction_summary.append(
                        dict(
                            model=model,
                            dataset=dataset,
                            split=split,
                            condition=condition,
                            n=len(part),
                            cases=len(per_case),
                            zero_mae=float(np.abs(measured).mean()),
                            input_mae=float(np.abs(measured - px).mean()),
                            feature_mae=float(np.abs(measured - pp).mean()),
                            feature_advantage=float(
                                np.mean(abs(measured - px) - abs(measured - pp))
                            ),
                            advantage_lower=interval[0],
                            advantage_upper=interval[1],
                            feature_vs_zero_lower=vs_zero[0],
                            feature_vs_zero_upper=vs_zero[1],
                            input_sign=float(np.mean(np.sign(px) == np.sign(measured))),
                            feature_sign=float(np.mean(np.sign(pp) == np.sign(measured))),
                            nontrivial_n=int(selected.sum()),
                            input_sign_above_005=mean(
                                (np.sign(px[selected]) == np.sign(measured[selected])).tolist()
                            ),
                            feature_sign_above_005=mean(
                                (np.sign(pp[selected]) == np.sign(measured[selected])).tolist()
                            ),
                            max_probe_batch_gap_difference=max(
                                abs(r["probe_batch_gap_difference"]) for r in part
                            ),
                        )
                    )
                for row in group[:8]:
                    source = next(
                        r for r in cases if r["id"] == row["id"] and r["dataset"] == dataset
                    )
                    examples.append(
                        dict(
                            model=model,
                            **row,
                            gold_answer=source["answer"],
                            gold_bridge=source["bridge"],
                            question=source["question"],
                            aliases=source["aliases"],
                        )
                    )
    for model in cfg["models"]:
        fp_dir = RESULTS / (model + "-fp32")
        done = read(ART / f"precision-completion-{model}.json")
        check(f"{model}:precision_complete", done["state"] == "complete" and done["cases"] == 8)
        for file, expected in done["files"].items():
            check(f"{model}:precision_hash:{file}", digest(fp_dir / file) == expected)
        fp_rows = rows(fp_dir / "mechanism.jsonl")
        check(f"{model}:precision_rows", len(fp_rows) == 216)
        bf_rows = {
            (r["dataset"], r["id"], r["layer"], r["condition"]): r
            for r in rows(RESULTS / model / "mechanism.jsonl")
            if r["status"] == "complete"
        }
        for dataset in ["mquake", "2wiki"]:
            group = [r for r in fp_rows if r["dataset"] == dataset]
            for precision in ["bfloat16", "float32"]:
                part = (
                    group
                    if precision == "float32"
                    else [
                        bf_rows[(r["dataset"], r["id"], r["layer"], r["condition"])] for r in group
                    ]
                )
                nonzero = [r for r in part if r["condition"] != "zero"]
                measured = np.array([r["gap_change"] for r in nonzero])
                px = np.array([r["predicted_input_change"] for r in nonzero])
                pp = np.array([r["predicted_feature_change"] for r in nonzero])
                precision_summary.append(
                    dict(
                        model=model,
                        dataset=dataset,
                        precision=precision,
                        cases=4,
                        effects=len(nonzero),
                        zero_mae=float(abs(measured).mean()),
                        input_mae=float(abs(measured - px).mean()),
                        feature_mae=float(abs(measured - pp).mean()),
                        max_batch_gap_difference=max(
                            abs(r["probe_batch_gap_difference"]) for r in part
                        ),
                        same_generated_string_as_bf16=mean(
                            [
                                r["output"]
                                == bf_rows[(r["dataset"], r["id"], r["layer"], r["condition"])][
                                    "output"
                                ]
                                for r in part
                            ]
                        ),
                    )
                )
    cot_done = read(ART / "cot256-completion.json")
    check(
        "cot256:complete_frozen",
        cot_done["state"] == "complete"
        and cot_done["unchanged_parameters"]
        and cot_done["n"] == 256,
    )
    check("cot256:hash", digest(RESULTS / "main-cot256.jsonl") == cot_done["output_sha256"])
    longer = rows(RESULTS / "main-cot256.jsonl")
    primary = {(r["dataset"], r["id"]): r for r in rows(RESULTS / "main/behavior.jsonl")}
    long_cot_summary = []
    for dataset in ["mquake", "2wiki"]:
        part = [r for r in longer if r["dataset"] == dataset]
        mastered = [
            r
            for r in part
            if primary[(dataset, r["id"])]["metrics"]["first"]["em"]
            and primary[(dataset, r["id"])]["metrics"]["second_oracle"]["em"]
        ]
        changes = [
            int(r["metrics"]["em"]) - int(primary[(dataset, r["id"])]["metrics"]["direct"]["em"])
            for r in part
        ]
        interval = ci(changes)
        long_cot_summary.append(
            dict(
                dataset=dataset,
                n=len(part),
                max_tokens=256,
                em=mean([r["metrics"]["em"] for r in part]),
                alias_em=mean([r["metrics"]["alias_em"] for r in part]),
                final_marker_count=sum(r["has_final_marker"] for r in part),
                mastered_n=len(mastered),
                conditional_em=mean([r["metrics"]["em"] for r in mastered]),
                delta_direct_em=mean(changes),
                lower=interval[0],
                upper=interval[1],
            )
        )
    table(ART / "cot256-summary.csv", long_cot_summary)
    table(ART / "precision-summary.csv", precision_summary)
    table(ART / "behavior-summary.csv", behavior_summary)
    table(ART / "paired-effects.csv", effects)
    table(ART / "mechanism-summary.csv", mechanism_summary)
    table(ART / "prediction-summary.csv", prediction_summary)
    write(ART / "generation-audit-sample.json", examples)
    write(
        ART / "summary.json",
        dict(
            behavior=behavior_summary,
            effects=effects,
            mechanism=mechanism_summary,
            predictions=prediction_summary,
            precision=precision_summary,
            long_cot=long_cot_summary,
        ),
    )
    write(
        ART / "completion-audit.json",
        dict(
            summary=dict(total=len(checks), passed=sum(c["passed"] for c in checks)),
            checks=checks,
            source_sha256=digest(Path(__file__)),
        ),
    )
    print(
        json.dumps(
            dict(
                checks=len(checks),
                behavior_rows=len(behavior_summary),
                mechanism_rows=len(mechanism_summary),
                prediction_rows=len(prediction_summary),
            )
        )
    )


if __name__ == "__main__":
    main()
