"""Combine completed Loop batches without merging seeds into new worlds."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


def summarize(roots, out):
    groups, sources, runs, manifests = defaultdict(list), {}, [], []
    for root in map(Path, roots):
        path = root / "report/summary.json"
        report = json.loads(path.read_text())
        cloud_path = root / "tracking-cloud-audit.json"
        cloud = json.loads(cloud_path.read_text())
        assert report["state"] == "complete" and cloud["passed"]
        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        sources[str(cloud_path)] = hashlib.sha256(cloud_path.read_bytes()).hexdigest()
        manifests.append(
            {"batch": report["batch"], "runs": report["completed_runs"], "cloud_passed": True}
        )
        for run in report["runs"]:
            assert run["complete"] and run["independent_audit_passed"]
            groups[(run["hidden_size"], run["repeats"], run["arm"])].append(run)
            runs.append(run)
    rows = []
    for (width, repeats, arm), group in sorted(groups.items()):
        row = {"width": width, "repeats": repeats, "arm": arm, "initializations": len(group)}
        for stage in ("stage_a", "endpoint"):
            for pool in ("atomic_A", "atomic_B", "AA", "BA", "AB", "BB"):
                scores = [r[stage][pool]["answer_accuracy"] for r in group]
                row[f"{stage}_{pool}_percent"] = 100 * mean(scores)
                row[f"{stage}_{pool}_min_percent"] = 100 * min(scores)
                row[f"{stage}_{pool}_max_percent"] = 100 * max(scores)
        row["BB_change_pp"] = 100 * mean(r["change_during_B"]["BB"] for r in group)
        row["AA_retention_percent"] = 100 * mean(
            r["transitions"]["AA"]["retention_of_stage_a_correct"] for r in group
        )
        row["parameters"] = mean(r["parameters"] for r in group)
        for key in (
            "estimated_matmul_training_flops",
            "effective_input_tokens",
            "supervised_tokens",
            "training_seconds",
        ):
            row[key] = mean(r["cost_at_endpoint"][key] for r in group)
        rows.append(row)
    result = {
        "complete": True,
        "runs": len(runs),
        "independent_data_graphs": 1,
        "initializations": sorted({r["initialization"] for r in runs}),
        "optimizer_updates": sum(r["expected_final_step"] for r in runs),
        "groups": rows,
        "sources": sources,
        "batches": manifests,
        "limits": [
            "One development split of one real graph; seed ranges are not confidence intervals.",
            "BB change includes old composition replay and does not isolate B's causal effect.",
            "Two widths and two recurrence settings do not establish a scaling law.",
            "Training time includes diagnostic overhead; FLOPs are matmul estimates.",
        ],
    }
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    with (out / "group-means.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / "completion-manifest.json").write_text(
        json.dumps(
            {k: result[k] for k in ("complete", "runs", "optimizer_updates", "batches", "sources")},
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"runs": result["runs"], "groups": len(rows), "out": str(out)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    summarize(args.roots, args.out)


if __name__ == "__main__":
    main()
