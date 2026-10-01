"""Export the completed grok depth/hop studies without rerunning training.

The exporter preserves individual runs and every recorded evaluation node. It
does not pool development and confirmation, or count reused baselines twice.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = Path("docs/development-artifacts")
FAMILIES = {"grok-depth-v1": 20, "grok-depth-extension-v1": 12, "grok-multihop-v1": 11}
IDENTITY = [
    "family",
    "phase",
    "run_id",
    "world",
    "initialization",
    "hops",
    "layers",
    "width",
    "phi",
]
ENDPOINT_COLUMNS = IDENTITY + [
    "steps",
    "parameters",
    "atomic_accuracy",
    "train_accuracy",
    "heldout_full_accuracy",
    "heldout_full_n",
    "heldout_probe_accuracy",
    "heldout_probe_n",
    "calls_accuracy",
    "calls_n",
    "ood_accuracy",
    "ood_n",
    "t90_step",
    "t90_confirmed_step",
    "t90_basis",
    "estimated_training_flops",
    "atomic_exposures",
    "train_n",
    "source_directory",
    "atomic_n",
    "id_atomic_accuracy",
    "id_atomic_n",
    "ood_atomic_accuracy",
    "ood_atomic_n",
    "calls_basis",
    "ood_basis",
    "dataset_sha256",
    "training_fraction_of_id_paths",
    "examples",
    "effective_input_tokens",
    "supervised_tokens",
    "composite_exposures",
    "training_seconds",
    "evaluation_seconds",
    "t90_estimated_training_flops",
    "t90_atomic_exposures",
]
LEARNING_COLUMNS = IDENTITY + [
    "parameters",
    "step",
    "atomic_accuracy",
    "train_accuracy",
    "heldout_accuracy",
    "heldout_basis",
    "heldout_n",
    "calls_accuracy",
    "ood_accuracy",
    "estimated_training_flops",
    "atomic_exposures",
    "atomic_n",
    "train_n",
    "calls_n",
    "ood_n",
    "examples",
    "effective_input_tokens",
    "supervised_tokens",
    "composite_exposures",
    "training_seconds",
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Sources:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.hashes: dict[str, str] = {}

    def read(self, path: Path) -> Any:
        self.hashes[str(path)] = sha256(self.root / path)
        return json.loads((self.root / path).read_text())

    def csv(self, path: Path) -> list[dict[str, str]]:
        self.hashes[str(path)] = sha256(self.root / path)
        with (self.root / path).open(newline="") as handle:
            return list(csv.DictReader(handle))


def threshold_time(nodes: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, int | None]:
    for i in range(len(nodes) - 2):
        if all(node["test_composite"]["accuracy"] >= 0.9 for node in nodes[i : i + 3]):
            return nodes[i], nodes[i + 2]["step"]
    return None, None


def observations(identity: dict[str, Any], node: dict[str, Any]) -> dict[str, Any]:
    calls = "two_calls" if identity["hops"] == 2 else "autonomous_calls"
    return {
        **identity,
        "parameters": node["parameters"],
        "step": node["step"],
        "atomic_accuracy": node["atomic"]["accuracy"],
        "train_accuracy": node["train_composite"]["accuracy"],
        "heldout_accuracy": node["test_composite"]["accuracy"],
        "heldout_basis": "full" if identity["hops"] == 2 else "probe",
        "heldout_n": node["test_composite"]["n"],
        "calls_accuracy": node[calls]["accuracy"],
        "ood_accuracy": node["ood_composite"]["accuracy"],
        "estimated_training_flops": node["estimated_training_flops"],
        "atomic_exposures": node["counts"]["atomic"] / node["atomic"]["n"],
        "atomic_n": node["atomic"]["n"],
        "train_n": node["train_composite"]["n"],
        "calls_n": node[calls]["n"],
        "ood_n": node["ood_composite"]["n"],
        "examples": node["examples"],
        "effective_input_tokens": node["effective_input_tokens"],
        "supervised_tokens": node["supervised_tokens"],
        "composite_exposures": node["counts"]["composite"] / node["train_composite"]["n"],
        "training_seconds": node["training_seconds"],
    }


def compare_historical(row: dict[str, Any], historical: dict[str, str], family: str) -> None:
    if family == "grok-multihop-v1":
        mapping = {
            "atomic_accuracy": "atomic",
            "train_accuracy": "train_composite",
            "heldout_full_accuracy": "test_full",
            "heldout_full_n": "test_full_n",
            "heldout_probe_accuracy": "test_probe",
            "calls_accuracy": "autonomous_calls_probe",
            "ood_accuracy": "ood_composite",
            "ood_n": "ood_composite_n",
            "t90_step": "T90_probe",
            "atomic_exposures": "atomic_mean_exposure",
        }
    else:
        mapping = {
            "atomic_accuracy": "atomic_accuracy",
            "train_accuracy": "train_composite_accuracy",
            "heldout_full_accuracy": "test_composite_accuracy",
            "ood_accuracy": "ood_composite_accuracy",
            "calls_accuracy": "two_calls_accuracy",
            "atomic_exposures": "atomic_exposures",
        }
        if family == "grok-depth-v1":
            mapping.update({"t90_step": "t90_step", "t90_confirmed_step": "t90_confirmed_at_step"})
    mapping.update(
        {"parameters": "parameters", "estimated_training_flops": "estimated_training_flops"}
    )
    for canonical, old in mapping.items():
        value = float(historical[old]) if historical[old] else None
        require(row[canonical] == value, f"Historical mismatch: {row['run_id']} {canonical}")


def export(root: Path, output: Path) -> dict[str, Any]:
    sources = Sources(root)
    endpoints: list[dict[str, Any]] = []
    learning: list[dict[str, Any]] = []
    run_checks: list[dict[str, Any]] = []
    audits = [
        "grok-depth-v1/endpoint-audit-development.json",
        "grok-depth-v1/endpoint-audit-confirmation.json",
        "grok-depth-extension-v1/endpoint-audit.json",
        "grok-multihop-v1/all-development-endpoint-audit.json",
        "grok-multihop-v1/confirmation-endpoint-audit.json",
    ]
    audited_ids = set()
    for audit_path in audits:
        audit = sources.read(ARTIFACTS / audit_path)
        require(audit["passed"] is True, f"Historical audit failed: {audit_path}")
        audited_ids.update(run.get("run_id", run.get("run")) for run in audit["runs"])

    for family, expected_count in FAMILIES.items():
        artifact_dir = ARTIFACTS / family
        summary = sources.read(artifact_dir / "summary.json")
        completion = sources.read(artifact_dir / "completion-manifest.json")
        require(completion.get("status", completion.get("state")) == "complete", family)
        old_rows = sources.csv(artifact_dir / "endpoint-table.csv")
        if family == "grok-multihop-v1":
            confirmation = sources.read(artifact_dir / "confirmation/summary.json")
            old_rows += sources.csv(artifact_dir / "confirmation/endpoint-table.csv")
            require(len(confirmation["endpoints"]) == 4, "Missing multihop confirmation")
        historical = {row["run_id"]: row for row in old_rows}
        summary_runs = {run["run_id"]: run for run in summary.get("runs", [])}
        raw_files = sorted((root / "results" / family).glob("*/*/metadata.json"))
        require(len(raw_files) == expected_count, f"Unexpected raw run count: {family}")
        for metadata_path in raw_files:
            run_dir = metadata_path.parent.relative_to(root)
            run_id = run_dir.name
            metadata = sources.read(run_dir / "metadata.json")
            nodes = sources.read(run_dir / "learning.json")
            receipt = sources.read(run_dir / "complete.json")
            data = sources.read(run_dir / "data-audit.json")
            spec = metadata["spec"]
            require(run_id in historical and run_id in audited_ids, f"Unaudited run: {run_id}")
            require(spec == receipt["spec"], f"Spec mismatch: {run_id}")
            require(
                [n["step"] for n in nodes] == spec["nodes"], f"Missing/duplicate node: {run_id}"
            )
            require(len(nodes) == 69 and nodes[0]["step"] == 0, f"Node count: {run_id}")
            require(nodes[-1] == receipt["endpoint"], f"Receipt mismatch: {run_id}")
            require(nodes[-1]["step"] == spec["steps"] == 128000, f"Incomplete budget: {run_id}")
            for node in nodes:
                for metric in ("atomic", "train_composite", "test_composite", "ood_composite"):
                    require(node[metric]["n"] == data["counts"][metric], f"N mismatch: {run_id}")
                    require(0 <= node[metric]["accuracy"] <= 1, f"Invalid accuracy: {run_id}")
            if run_id in summary_runs:
                saved = summary_runs[run_id]
                require(saved["learning"] == nodes, f"Summary trajectory mismatch: {run_id}")
                require(saved["fixed_budget_endpoint"] == nodes[-1], f"Summary endpoint: {run_id}")
            identity = {
                "family": family,
                "phase": spec["phase"],
                "run_id": run_id,
                "world": spec["world_seed"],
                "initialization": spec["initialization"],
                "hops": spec.get("hops", 2),
                "layers": spec["layers"],
                "width": spec["width"],
                "phi": spec["phi"],
            }
            trajectory = [observations(identity, node) for node in nodes]
            learning.extend(trajectory)
            endpoint = nodes[-1]
            is_twohop = identity["hops"] == 2
            full = endpoint["test_composite" if is_twohop else "test_full_composite"]
            probe = {} if is_twohop else endpoint["test_composite"]
            first, confirmed = threshold_time(nodes)
            observed = trajectory[-1]
            row = {name: observed[name] for name in ENDPOINT_COLUMNS if name in observed}
            row.update(
                {
                    "steps": endpoint["step"],
                    "heldout_full_accuracy": full["accuracy"],
                    "heldout_full_n": full["n"],
                    "heldout_probe_accuracy": probe.get("accuracy"),
                    "heldout_probe_n": probe.get("n"),
                    "t90_step": first["step"] if first else None,
                    "t90_confirmed_step": confirmed,
                    "t90_basis": "full" if is_twohop else "probe",
                    "source_directory": str(run_dir),
                    "dataset_sha256": data["dataset_sha256"],
                    "training_fraction_of_id_paths": endpoint["train_composite"]["n"]
                    / data.get("id_paths_independently_counted", data.get("id_composite_total")),
                    "calls_basis": "full" if is_twohop else "probe",
                    "ood_basis": "full",
                    "evaluation_seconds": endpoint["evaluation_seconds"],
                    "t90_estimated_training_flops": first["estimated_training_flops"]
                    if first
                    else None,
                    "t90_atomic_exposures": first["counts"]["atomic"] / first["atomic"]["n"]
                    if first
                    else None,
                }
            )
            atomic_split = summary_runs.get(run_id, {}).get("atomic_split_endpoint", {})
            for metric in ("id_atomic", "ood_atomic"):
                split = endpoint.get(metric, atomic_split.get(metric, {}))
                row[f"{metric}_accuracy"] = split.get("accuracy")
                row[f"{metric}_n"] = data["counts"][metric]
            compare_historical(row, historical[run_id], family)
            if family == "grok-depth-extension-v1":
                saved_t90 = summary_runs[run_id]["thresholds"]["t90"]
                require(saved_t90["step"] == row["t90_step"], f"Extension T90: {run_id}")
                require(
                    saved_t90["confirmed_at_step"] == confirmed, f"Extension T90 confirm: {run_id}"
                )
            endpoints.append(row)
            run_checks.append({"run_id": run_id, "nodes": len(nodes), "passed": True})
        expected_ids = set(historical)
        if family == "grok-depth-extension-v1":
            reused = {run["run_id"] for run in summary["runs"] if run["reused_baseline"]}
            require(len(reused) == 6, "Reused baseline count")
            expected_ids -= reused
        require(
            {p.parent.name for p in raw_files} == expected_ids, f"Run matrix mismatch: {family}"
        )

    require(len(endpoints) == len({row["run_id"] for row in endpoints}) == 43, "Unique runs")
    require(len(learning) == 2967, "Expected all 43 x 69 nodes")
    output.mkdir(parents=True, exist_ok=True)
    for name, columns, rows in (
        ("endpoints.csv", ENDPOINT_COLUMNS, endpoints),
        ("learning.csv", LEARNING_COLUMNS, learning),
    ):
        with (output / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
    manifest = {
        "scope": "43 unique completed grok-depth, paired extension, and grok-multihop runs",
        "run_count": len(endpoints),
        "learning_row_count": len(learning),
        "family_counts": dict(Counter(row["family"] for row in endpoints)),
        "phase_counts": dict(Counter(row["phase"] for row in endpoints)),
        "metric_definition": "Complete answer plus EOS exact accuracy, in [0, 1].",
        "missing_values": "Empty CSV cell means unavailable/not measured, or T90 not established.",
        "t90_definition": "First of three consecutive nodes >= 0.9; confirmed at third node. "
        "Full two-hop ID test; fixed 1024-example ID probe for three/four-hop. No interpolation.",
        "heldout_definition": "Full reserved ID test pool at endpoint; curves use full two-hop "
        "test or fixed multihop probe. Do not substitute full and probe values for one another.",
        "atomic_partition_definition": "All atomic facts are trained. ID/OOD atomic denotes the "
        "partition used to construct composition splits; OOD atomic is not untrained knowledge.",
        "aggregation": "Keep phases separate. Average initialization replicates within a world "
        "before averaging worlds. Extension uses previously observed two-hop confirmation worlds.",
        "deduplication": "Six reused two-layer baselines occur only under grok-depth-v1. "
        "The threehop-depth report duplicates development trajectories and is not loaded.",
        "flops_definition": "Executed-shape matrix/attention FLOPs estimated by original training "
        "logs; excludes elementwise operations and evaluation.",
        "validation": {
            "passed": True,
            "checks": [
                "43 unique audited completed runs",
                "69 registered nodes per run including step zero",
                "exact receipt/trajectory endpoint identity",
                "raw metrics equal historical endpoint CSVs",
                "T90 recomputed and checked against historical values",
                "no duplicate baseline runs",
                "evaluation sample counts equal data audits",
            ],
            "runs": run_checks,
        },
        "source_sha256": dict(sorted(sources.hashes.items())),
        "exporter_sha256": sha256(Path(__file__)),
        "output_sha256": {
            name: sha256(output / name) for name in ("endpoints.csv", "learning.csv")
        },
        "columns": {"endpoints.csv": ENDPOINT_COLUMNS, "learning.csv": LEARNING_COLUMNS},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / ARTIFACTS / "hop-depth-comparison-v1")
    args = parser.parse_args()
    manifest = export(args.root.resolve(), args.output.resolve())
    print(
        json.dumps(
            {key: manifest[key] for key in ("run_count", "learning_row_count", "family_counts")}
        )
    )


if __name__ == "__main__":
    main()
