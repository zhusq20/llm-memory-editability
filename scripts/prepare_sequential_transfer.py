"""Create fixed real-fact sequential-transfer datasets; never access a model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm_memory_editability.realworld_composition_data import sha256, write_json
from llm_memory_editability.sequential_data import SplitSettings, build_split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--only", choices=["development", "confirmation", "all"], default="all")
    parser.add_argument("--bb-count", type=int, default=64)
    parser.add_argument("--train-count", type=int, default=512)
    parser.add_argument("--evaluation-count", type=int, default=64)
    args = parser.parse_args()
    base = args.project / "data/realworld-composition-v1"
    output = args.project / "data/sequential-transfer-v1"
    labels = json.loads((base / "canonical-labels.json").read_text())
    splits = []
    if args.only in {"development", "all"}:
        splits.append(("development", "development.json"))
    if args.only in {"confirmation", "all"}:
        splits.extend((f"split-{index}", "confirmation-v1.json") for index in range(3))
    sources = {}
    manifest = {
        "schema": "sequential-transfer-v1",
        "statistical_unit": "Three fixed partitions of one real source graph, not new worlds.",
        "tokenizer": "data/realworld-composition-v1/tokenizer",
        "datasets": [],
    }
    for split, filename in splits:
        if filename not in sources:
            sources[filename] = json.loads((base / filename).read_text())
        data = build_split(
            sources[filename],
            SplitSettings(split, args.bb_count, args.train_count, args.evaluation_count),
            labels,
        )
        data["provenance"] = {
            "source_file": str((base / filename).relative_to(args.project)),
            "source_sha256": sha256(base / filename),
            "canonical_labels_sha256": sha256(base / "canonical-labels.json"),
            "original_archive": "data/realworld-composition-v1/raw/data_ids_april7.zip",
            "source_tokenizer": manifest["tokenizer"],
            "reencoded": False,
            "new_synthetic_facts": 0,
        }
        path = output / f"{split}.json"
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite frozen dataset: {path}")
        write_json(path, data)
        audit = {
            "path": str(path.relative_to(args.project)),
            "sha256": sha256(path),
            **data["audit"],
        }
        write_json(output / f"{split}-audit.json", audit)
        manifest["datasets"].append(audit)
        print(json.dumps({"split": split, **data["audit"]}, ensure_ascii=False), flush=True)
    manifest_path = output / (
        "data-lock.json" if args.only == "all" else f"{args.only}-data-lock.json"
    )
    write_json(manifest_path, manifest)


if __name__ == "__main__":
    main()
