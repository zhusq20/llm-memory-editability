"""Freeze name-isolated confirmation data without consulting model results."""

import argparse
import json
from pathlib import Path

from llm_memory_editability.realworld_composition_data import sha256, write_json
from llm_memory_editability.realworld_confirmation import audit_confirmation, isolate_confirmation

PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / "data/realworld-composition-v1"
ART = PROJECT / "docs/development-artifacts/realworld-composition-v1/confirmation"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "audit"])
    args = parser.parse_args()
    original_lock = json.loads((DATA / "data-lock.json").read_text())
    for phase in ("development", "confirmation-candidate"):
        assert sha256(DATA / (phase + ".json")) == original_lock["datasets"][phase]["data_sha256"]
    development = json.loads((DATA / "development.json").read_text())
    canonical = json.loads((DATA / "canonical-labels.json").read_text())
    aliases = json.loads((DATA / "aliases.json").read_text())
    path = DATA / "confirmation-v1.json"
    if args.command == "prepare":
        if path.exists():
            raise FileExistsError("Preserve previously frozen confirmation data")
        candidate = json.loads((DATA / "confirmation-candidate.json").read_text())
        data, exclusions = isolate_confirmation(candidate, development, canonical, aliases)
        write_json(path, data)
        write_json(ART / "name-exclusions.json", exclusions)
        lock = {
            "data_file": str(path),
            "data_sha256": sha256(path),
            "source_files": {
                str(p): sha256(p)
                for p in [
                    DATA / "confirmation-candidate.json",
                    DATA / "development.json",
                    DATA / "canonical-labels.json",
                    DATA / "aliases.json",
                    DATA / "data-lock.json",
                    Path(__file__),
                    PROJECT / "src/llm_memory_editability/realworld_confirmation.py",
                ]
            },
            "name_exclusions_sha256": sha256(ART / "name-exclusions.json"),
            "inherited_source_review_sha256": original_lock["semantic_review_sha256"],
            "selection_depends_on_model_results": False,
        }
        write_json(ART / "confirmation-data-lock.json", lock)
    else:
        lock = json.loads((ART / "confirmation-data-lock.json").read_text())
        assert sha256(path) == lock["data_sha256"]
        for p, digest in lock["source_files"].items():
            assert sha256(p) == digest
        data = json.loads(path.read_text())
        result = audit_confirmation(data, development, canonical, aliases)
        result.update(data_sha256=sha256(path), audit_source_sha256=sha256(__file__))
        write_json(ART / "data-contract-audit.json", result)
        print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
