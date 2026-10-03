"""Independently recount raw bridge intervention predictions and summarize runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from llm_memory_editability.grok_depth import write_json


def report(root, out):
    root, out = Path(root), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    records, audits = [], []
    for path in sorted((root / "runs").glob("*")):
        if not (path / "complete.json").exists():
            continue
        run = json.loads((path / "run.json").read_text())
        spec = run["spec"]
        selection = dict(np.load(path / "selection.npz"))
        summary = json.loads((path / "summary.json").read_text())
        large = spec["family"] == "reproduction"
        metadata = (
            json.loads((Path(spec["data"]) / "complete.json").read_text())
            if large
            else {"entity_offset": 0}
        )
        offset = metadata["entity_offset"]
        for key, strata in summary.items():
            family, condition, layer = key.split("/")
            layer = int(layer)
            raw = dict(np.load(path / f"{family}-{condition}-l{layer:02d}.npz"))
            if family == "swap":
                targets = selection["queries"][:, 2]
                cf = selection["queries"][:, 3]
                labels = selection["strata"]
            elif family == "native":
                targets = cf = selection["native_rows"][:, -1]
                labels = selection["native_strata"]
            else:
                # Recount against graph truth, including atoms the model got wrong.
                world_path = (
                    Path(spec["data"]) / "world.npz"
                    if large
                    else Path(spec["checkpoint"]).parent / "world.npz"
                )
                world = dict(np.load(world_path))
                atoms = (
                    world["atoms"]
                    if large
                    else np.concatenate((world["common_atomic"], world["anchor_atomic"]))
                )
                targets = cf = atoms[selection["needed_atomic_indices"], -1]
                labels = np.full(len(targets), "needed_atoms")
            original_ok = raw["generated"][:, 0] == targets + offset
            cf_ok = raw["generated"][:, 0] == cf + offset
            format_ok = (
                raw["generated"][:, 1] == metadata["end_marker"]
                if large
                else (raw["generated"][:, 1] == 5) & (raw["generated"][:, 2] == 1)
            )
            for name, metrics in strata.items():
                if name == "prerequisites_correct":
                    mask = selection["prerequisite_correct"]
                elif name in {"both_queries_untrained", "fixed_entity_target"}:
                    mask = selection[name]
                elif name == "fixed_entity_unrelated":
                    mask = ~selection["fixed_entity_target"]
                else:
                    mask = np.ones(len(targets), dtype=bool) if name == "all" else labels == name
                assert int(mask.sum()) == metrics["n_queries"]
                for metric, values in (
                    ("answer_accuracy", original_ok),
                    ("cf_answer_accuracy", cf_ok),
                    ("accuracy", original_ok & format_ok),
                    ("cf_accuracy", cf_ok & format_ok),
                ):
                    assert abs(float(values[mask].mean()) - metrics[metric]) < 1e-12, (
                        path.name,
                        key,
                        name,
                        metric,
                    )
                records.append(
                    {
                        "run": path.name,
                        "family": spec["family"],
                        "phase": spec["phase"],
                        "checkpoint_step": spec["checkpoint_step"],
                        "intervention": family,
                        "condition": condition,
                        "layer": layer,
                        "stratum": name,
                        **metrics,
                    }
                )
        audits.append(
            {"run": path.name, "raw_predictions_recounted": True, "evaluations": len(summary)}
        )
    if records:
        fields = list(dict.fromkeys(k for row in records for k in row))
        with (out / "endpoints.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)
    write_json(out / "recount-audit.json", audits)
    write_json(
        out / "summary.json",
        {"runs": len(audits), "records": len(records), "new_training_updates": 0},
    )
    print(json.dumps({"runs": len(audits), "records": len(records), "report": str(out)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report(args.root, args.out)
