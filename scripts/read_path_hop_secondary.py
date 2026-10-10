"""Read out S1/S2 of the frozen path-hop-analysis-v1 MQuAKE outputs.

S1 (prerequisite subset) and S2 (old-answer retention) were computed by
scripts/analyze_path_hop_position.py under configs/path-hop-analysis-v1.json.
This script copies those numbers and applies the reading rules frozen in
configs/zero-training-path-overlap-v1.json. It does not re-score, re-group or
re-bootstrap anything.

Usage: python scripts/read_path_hop_secondary.py \
    [--analysis-dir results/path-hop-analysis-v1] \
    [--output results/zero-training-path-overlap-v1/s1s2-readout.json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
CONTRACT = PROJECT / "configs/zero-training-path-overlap-v1.json"
SOURCE_CONTRACT = PROJECT / "configs/path-hop-analysis-v1.json"
EDITORS = {"ROME": "gpt-j-6B-rome.json", "MEMIT": "gpt-j-6B-memit.json"}
MIN_LATER_CASES = 30


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def manifest_hashes(manifest):
    """Collect every {path-like: sha256} pair the manifest records, whatever its layout."""
    found = {}

    def walk(node):
        if isinstance(node, dict):
            name = next((node[k] for k in ("path", "file", "output") if k in node), None)
            if isinstance(name, str) and isinstance(node.get("sha256"), str):
                found[Path(name).name] = node["sha256"]
            for key, value in node.items():
                if isinstance(value, str) and len(value) == 64 and key.endswith(".json"):
                    found[Path(key).name] = value
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(manifest)
    return found


def read_editor(result):
    assert result["contract"] == "path-hop-analysis-v1", result["contract"]
    s1 = result["secondary_prerequisite_subset"]
    full = result["primary_full_pool"]
    cells = s1["cells"]
    later_cases = {h: cells.get(f"{h}-hop/later", {}).get("cases", 0) for h in (2, 3)}
    interaction = s1["interval"].get("interaction")
    if min(later_cases.values()) < MIN_LATER_CASES or interaction is None:
        s1_reading = "underpowered"
    elif interaction["low"] > 0:
        s1_reading = "consistent with prerequisite failure"
    else:
        s1_reading = "against pattern persists within the prerequisite subset"
    s2 = {}
    for hops in (2, 3):
        first, later = full["cells"][f"{hops}-hop/first"], full["cells"][f"{hops}-hop/later"]
        s2[f"{hops}-hop"] = {
            "first": first["old_answer_retained"],
            "first_cases": first["old_answer_cases"],
            "later": later["old_answer_retained"],
            "later_cases": later["old_answer_cases"],
            "later_minus_first": later["old_answer_retained"] - first["old_answer_retained"],
            "P4_holds": later["old_answer_retained"] > first["old_answer_retained"],
        }
    return {
        "full_pool_estimate": full["estimate"],
        "full_pool_interval": full["interval"],
        "S1": {
            "coverage": s1["coverage"],
            "estimate": s1["estimate"],
            "interval": s1["interval"],
            "cells": cells,
            "later_cases_by_stratum": later_cases,
            "reading": s1_reading,
        },
        "S2": {
            "by_stratum": s2,
            "equal_weight_later_minus_first": sum(v["later_minus_first"] for v in s2.values())
            / len(s2),
        },
    }


def pp(value):
    return "—" if value is None else f"{100 * value:.2f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, default=Path("results/path-hop-analysis-v1"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/zero-training-path-overlap-v1/s1s2-readout.json"),
    )
    args = parser.parse_args()
    manifest_path = args.analysis_dir / "completion-manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"{manifest_path} is required: S1/S2 are read only from completed runs")
    recorded = manifest_hashes(json.loads(manifest_path.read_text()))
    out = {
        "contract": "zero-training-path-overlap-v1",
        "contract_sha256": sha(CONTRACT),
        "source_contract_sha256": sha(SOURCE_CONTRACT),
        "manifest_sha256": sha(manifest_path),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "editors": {},
        "files": {},
    }
    for editor, name in EDITORS.items():
        path = args.analysis_dir / name
        digest = sha(path)
        if name in recorded and recorded[name] != digest:
            raise SystemExit(f"{path}: sha256 differs from the completion manifest")
        out["files"][name] = {"sha256": digest, "in_manifest": name in recorded}
        out["editors"][editor] = read_editor(json.loads(path.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    for editor, row in out["editors"].items():
        s1, ci = row["S1"], row["S1"]["interval"]
        later = s1["later_cases_by_stratum"]
        print(f"{editor} S1 coverage {pp(s1['coverage'])}%, later cases {later}")
        for key in ("direct", "cot", "interaction"):
            interval = ci.get(key, {})
            print(
                f"  {key:11s} {pp(s1['estimate'] and s1['estimate'][key])} pp "
                f"({pp(interval.get('low'))}–{pp(interval.get('high'))})"
            )
        print(f"  reading: {s1['reading']}")
        for stratum, cell in row["S2"]["by_stratum"].items():
            print(
                f"  S2 {stratum}: first {pp(cell['first'])}% (n={cell['first_cases']}), "
                f"later {pp(cell['later'])}% (n={cell['later_cases']}), P4 {cell['P4_holds']}"
            )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
