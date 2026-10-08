"""Full residual/KV matrix, historical references and fixed paired contrasts."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from llm_memory_editability.grok_depth import write_json

ROOT = Path("/ossfs/workspace/llm-memory-editability")


def read(path):
    return json.loads(Path(path).read_text())


def report(config):
    results = ROOT / "results" / config["batch"]
    output = results / "report"
    output.mkdir(parents=True, exist_ok=True)
    reused = {s["name"]: s for s in config["reuse_specs"]}
    records, missing, editing, compute = [], [], [], []
    exposures = {}
    for spec in config["all_training_cells"]:
        reference = reused.get(spec["name"])
        folder = Path(reference["reuse_dir"]) if reference else results / "runs" / spec["name"]
        if reference:
            ref_path = results / "reused" / spec["name"] / "reference.json"
            if not ref_path.exists():
                missing.append(spec["name"])
                continue
            audit_dir = Path(read(ref_path)["independent_audit_dir"])
        else:
            audit_dir = folder
        if not (audit_dir / "audit.json").exists() or not read(audit_dir / "audit.json")["passed"]:
            missing.append(spec["name"])
            continue
        complete = read(folder / "complete.json")
        assert complete["data_sha256"] == spec["data_sha256"]
        assert complete["initial_model_sha256"] == spec["initial_model_sha256"]
        assert complete["parameters"] == spec["parameters"]
        with np.load(folder / "exposures.npz") as raw:
            current = {k: raw[k].copy() for k in raw.files}
        if spec["world"] in exposures:
            for key in current:
                np.testing.assert_array_equal(current[key], exposures[spec["world"]][key])
        exposures[spec["world"]] = current
        metrics = complete["endpoint"]["metrics"]
        row = dict(
            name=spec["name"],
            phase=spec["phase"],
            role=spec["role"],
            world=spec["world"],
            residual_kind=spec["residual_kind"],
            memory_arm=spec["memory_arm"],
            train_R=spec["repeats"],
            gamma=spec["gamma"],
            parameters=spec["parameters"],
            reused=reference is not None,
            training_seconds=complete["training_seconds"],
            flops=complete["endpoint"]["estimated_training_flops"],
            atomic=metrics["common_atomic"]["accuracy"],
            train=metrics["train_composite"]["accuracy"],
            familiar=metrics["familiar_test"]["accuracy"],
            strict=metrics["strict_test"]["accuracy"],
            familiar_nll=metrics["familiar_test"]["answer_nll"],
            strict_nll=metrics["strict_test"]["answer_nll"],
            coverage=metrics["familiar_test"]["atomic_correct_coverage"],
        )
        records.append(row)
        history = {r["step"]: r for r in read(folder / "learning.json")}
        for kind, step_key, budget_key in [
            ("same_R", "matched_compute_step", "reference_compute_budget"),
            ("R4_reference", "cross_R_compute_step", "cross_R_compute_budget"),
        ]:
            node = history[spec[step_key]]
            compute.append(
                dict(
                    name=spec["name"],
                    comparison=kind,
                    step=node["step"],
                    budget=spec[budget_key],
                    flops=node["estimated_training_flops"],
                    gap=spec[budget_key] - node["estimated_training_flops"],
                    metrics=node["metrics"],
                )
            )
    for spec in config["editing_specs"]:
        folder = results / "runs" / spec["name"]
        if not (folder / "audit.json").exists() or not read(folder / "audit.json")["passed"]:
            missing.append(spec["name"])
            continue
        assert read(folder / "run.json")["case_sha256"] == spec["case_sha256"]
        editing.append(dict(spec=spec, result=read(folder / "complete.json")))
    # Ordinary R4 edits are existing, independently audited observations.
    historical_edits = []
    parent = read(ROOT / "docs/development-artifacts/shared-cache-branch-v1/frozen-config.json")
    for spec in parent["editing_specs"]:
        if spec["memory_arm"] not in {"local", "shared_full"}:
            continue
        folder = ROOT / "results/shared-cache-branch-v1/runs" / spec["name"]
        assert read(folder / "audit.json")["passed"]
        historical_edits.append(
            dict(
                source=str(folder),
                world=spec["world"],
                memory_arm=spec["memory_arm"],
                result=read(folder / "complete.json"),
            )
        )
    paired = []
    index = {
        (r["world"], r["train_R"], r["residual_kind"], r["memory_arm"]): r
        for r in records
        if r["role"] == "primary"
    }
    for world in sorted({r["world"] for r in records}):
        for repeat in [4, 8]:
            for memory in ["local", "shared_full"]:
                for new_kind, old_kind in [("identity_mhc", "single"), ("mhc", "identity_mhc")]:
                    a, b = (
                        index.get((world, repeat, new_kind, memory)),
                        index.get((world, repeat, old_kind, memory)),
                    )
                    if a and b:
                        paired.append(
                            dict(
                                world=world,
                                train_R=repeat,
                                memory_arm=memory,
                                contrast=f"{new_kind}-{old_kind}",
                                **{
                                    k + "_delta_pp": 100 * (a[k] - b[k])
                                    for k in ["atomic", "train", "familiar", "strict"]
                                },
                            )
                        )
            for kind in ["single", "identity_mhc", "mhc"]:
                a, b = (
                    index.get((world, repeat, kind, "shared_full")),
                    index.get((world, repeat, kind, "local")),
                )
                if a and b:
                    paired.append(
                        dict(
                            world=world,
                            train_R=repeat,
                            residual_kind=kind,
                            contrast="shared_full-local",
                            **{
                                k + "_delta_pp": 100 * (a[k] - b[k])
                                for k in ["atomic", "train", "familiar", "strict"]
                            },
                        )
                    )
            cells = [
                index.get((world, repeat, kind, memory))
                for kind, memory in [
                    ("mhc", "shared_full"),
                    ("mhc", "local"),
                    ("single", "shared_full"),
                    ("single", "local"),
                ]
            ]
            if all(cells):
                a, b, c, d = cells
                paired.append(
                    dict(
                        world=world,
                        train_R=repeat,
                        contrast="mhc_by_shared_interaction",
                        **{
                            key + "_delta_pp": 100 * ((a[key] - b[key]) - (c[key] - d[key]))
                            for key in ["atomic", "train", "familiar", "strict"]
                        },
                    )
                )
        for kind in ["single", "identity_mhc", "mhc"]:
            for memory in ["local", "shared_full"]:
                a, b = index.get((world, 8, kind, memory)), index.get((world, 4, kind, memory))
                if a and b:
                    paired.append(
                        dict(
                            world=world,
                            residual_kind=kind,
                            memory_arm=memory,
                            contrast="native_R8-R4",
                            **{
                                key + "_delta_pp": 100 * (a[key] - b[key])
                                for key in ["atomic", "train", "familiar", "strict"]
                            },
                        )
                    )
    result = dict(
        passed=not missing,
        endpoints=records,
        paired_differences=paired,
        matched_compute=compute,
        new_edit_parents=editing,
        historical_edits=historical_edits,
        missing=missing,
        independent_confirmation_worlds=0,
        known_paired_worlds=3,
        exposures_identical_within_world=True,
    )
    write_json(output / "summary.json", result)
    if records:
        with (output / "endpoints.csv").open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    lines = [
        "# Residual/cache comparison",
        "",
        f"Audited training cells: {len(records)}/56. Missing jobs: {len(missing)}.",
        "",
        "The three paired worlds have already been observed in earlier experiments; "
        "they are not independent confirmation. Native train-R scores are primary. "
        "Unequal controller parameters, elementwise Sinkhorn work, strict-task floors "
        "and editing prerequisites are retained.",
        "",
        "| World | Kind | Memory | R | gamma | Familiar % | Strict % |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    lines += [
        f"| {r['world']} | {r['residual_kind']} | {r['memory_arm']} | {r['train_R']} "
        f"| {r['gamma']} | {100 * r['familiar']:.2f} | {100 * r['strict']:.2f} |"
        for r in records
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    result = report(read(args.config))
    print(json.dumps({"passed": result["passed"], "missing": result["missing"]}))
    raise SystemExit(0 if result["passed"] else 1)
