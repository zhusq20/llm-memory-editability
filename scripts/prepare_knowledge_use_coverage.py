#!/usr/bin/env python3
"""Materialize controlled Knowledge coverage data on CPU, without training or sampling."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from llm_memory_editability import knowledge_use_coverage_data as D

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/knowledge-use-coverage-v1.json"
DEFAULT_FACTORIAL = ROOT / "results/knowledge-method-batchsize-20261022/runtime/.."
NATURAL_RETENTION_COUNT = 317
REQUIRED_ROW_KEYS = {
    "id",
    "fact_ids",
    "question",
    "context",
    "gold",
    "aliases",
    "category",
    "split",
    "variant",
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def user_question(prompt):
    """Extract exactly one frozen Qwen user turn, excluding system/assistant content."""
    turns = re.findall(r"<\|im_start\|>user\r?\n(.*?)<\|im_end\|>", prompt, flags=re.S)
    if len(turns) != 1 or not turns[0].strip():
        raise ValueError("natural retention source must contain exactly one nonempty user turn")
    return turns[0].strip()


def natural_retention_rows(source, cohort):
    """Read only source question/gold/type/exclude; never reuse old predictions or scores."""
    before = json.loads(Path(source).read_text())["retention"]["before"]
    if len(before) != NATURAL_RETENTION_COUNT:
        raise ValueError(f"expected {NATURAL_RETENTION_COUNT} original natural retention questions")
    clusters = D.natural_retention_clusters(before)
    rows = []
    for i, old in enumerate(before):
        if not isinstance(old["gold"], str) or not old["gold"].strip():
            raise ValueError("natural retention gold must be a nonempty string")
        rows.append(
            {
                "id": f"{cohort}/natural-old-retention-{i:03d}",
                "original_source_id": old["id"],
                "fact_ids": [f"natural-old-retention/{i:03d}"],
                "question": user_question(old["prompt"]),
                "context": "",
                "gold": old["gold"],
                "aliases": [old["gold"]],
                "type": old["type"],
                "exclude": list(old["exclude"]),
                "category": "natural_old_retention",
                "split": "retention",
                "variant": "direct",
                "mode": "direct",
                "cohort": cohort,
                "chain": [],
                "pair_id": None,
                "cluster_id": clusters[old["id"]],
                "cluster_status": "country_identified" if clusters[old["id"]] else "unknown",
                "gold_source": "frozen_factorial_retention_question_metadata",
                "knowledge_status": "preexisting_natural_knowledge",
                "retention_reference": "released_base",
                "is_old_retention": True,
            }
        )
    if len({row["question"] for row in rows}) != len(rows):
        raise ValueError("natural retention source has duplicate user questions")
    return rows


def _location_retention(row, stage):
    row = dict(row)
    suffix = row["id"].split(f"/{stage}/", 1)[1].split("/", 1)[1]
    row["id"] = f"{row['pair_id']}/location-preservation/{suffix}"
    row["split"] = "retention"
    row["category"] = (
        "learned_location_baseline" if stage == "addition" else "retention_unchanged_new_location"
    )
    row["knowledge_status"] = "newly_learned" if stage == "addition" else "unchanged_after_rewrite"
    row["retention_reference"] = "addition_endpoint"
    row["is_old_retention"] = False
    row["is_rewrite_preservation"] = stage == "rewrite"
    return row


def retention_rows(world, stage, natural):
    rows = []
    for row in D.retention_rows(world, stage):
        row = dict(row)
        if row["category"] == "retention_unchanged_new_location":
            row = _location_retention(row, stage)
        else:
            row.update(
                knowledge_status="preexisting_background_knowledge",
                retention_reference="released_base",
                is_old_retention=True,
            )
        rows.append(row)
    if stage == "addition":
        rows.extend(
            _location_retention(row, stage)
            for row in D.evaluation_rows(world, stage)
            if row["category"] == "atomic_unseen" and row["fact_ids"][0].endswith("/location")
        )
    for old in D.general_retention_rows():
        row = dict(old)
        row["id"] = f"{world.cohort}/{row['id']}"
        row.update(
            knowledge_status="preexisting_arithmetic_skill",
            retention_reference="released_base",
            is_old_retention=True,
        )
        rows.append(row)
    rows.extend(dict(row) for row in natural)
    return rows


def validate_rows(name, rows):
    if not rows:
        raise ValueError(f"empty data file: {name}")
    ids = [row["id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate IDs in {name}")
    for row in rows:
        if not REQUIRED_ROW_KEYS <= row.keys():
            raise ValueError(f"missing required row fields in {name}")
        if not row["question"] or not row["gold"] or row["gold"] not in row["aliases"]:
            raise ValueError(f"missing question or gold alias in {name}")
    return len(rows)


def validate_cohort_isolation(worlds, datasets):
    names = {
        cohort: {name for p in world.pairs for name in (p.organization, p.office)}
        for cohort, world in worlds.items()
    }
    if names["development"] & names["formal"]:
        raise ValueError("development and formal new entities overlap")
    for cohort, files in datasets.items():
        other = "formal" if cohort == "development" else "development"
        other_names = names[other]
        for filename, rows in files.items():
            for row in rows:
                text = json.dumps(row, ensure_ascii=False)
                if any(name in text for name in other_names):
                    raise ValueError(f"cross-cohort new entity in {cohort}/{filename}")
    return {
        "new_entities_disjoint": True,
        "new_entity_count": {cohort: len(pool) for cohort, pool in names.items()},
        "background_entities_may_overlap": True,
    }


def _write_json(path, data):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")


def materialize(config_path, output, factorial_batch):
    config_path, output, factorial_batch = map(Path, (config_path, output, factorial_batch))
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    config = json.loads(config_path.read_text())
    source = factorial_batch / "runs/sft_full/paper-eval/auxiliary.json"
    worlds = {
        cohort: D.make_world(
            cohort=cohort, pair_count=config[f"{cohort}_pairs"], seed=config["seed"]
        )
        for cohort in ("development", "formal")
    }
    datasets, audits = {}, {}
    for cohort, world in worlds.items():
        audits[cohort] = D.audit_world(world)
        natural = natural_retention_rows(source, cohort)
        files = {f"train-{arm}.jsonl": D.training_rows(world, arm) for arm in "ABC"}
        files["train-rewrite.jsonl"] = D.rewrite_training_rows(world)
        for stage in ("addition", "rewrite"):
            files[f"evaluation-{stage}.jsonl"] = D.evaluation_rows(world, stage)
            files[f"retention-{stage}.jsonl"] = retention_rows(world, stage, natural)
        files["replay.jsonl"] = D.replay_rows()
        # Kept separately to assess the old-knowledge prerequisite before any write.
        files["background.jsonl"] = D.background_rows(world)
        for stage in ("addition", "rewrite"):
            D.validate_retention_split(
                files["replay.jsonl"],
                files[f"retention-{stage}.jsonl"],
                target_fact_ids=[p.assignment_id for p in world.pairs],
            )
        audits[cohort]["file_counts"] = {
            name: validate_rows(name, rows) for name, rows in files.items()
        }
        audits[cohort]["natural_old_retention_questions"] = len(natural)
        audits[cohort]["natural_old_retention_clusters"] = {
            "unit": "country",
            "identified_clusters": len({r["cluster_id"] for r in natural if r["cluster_id"]}),
            "unidentifiable_rows": sum(r["cluster_id"] is None for r in natural),
        }
        datasets[cohort] = files
    isolation = validate_cohort_isolation(worlds, datasets)
    # Everything is checked in memory before creating a new destination.
    output.mkdir(parents=True, exist_ok=False)
    file_audit = {}
    for cohort, world in worlds.items():
        destination = output / "data" / cohort
        destination.mkdir(parents=True)
        file_audit[cohort] = {}
        for name, rows in datasets[cohort].items():
            path = destination / name
            with path.open("x", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            file_audit[cohort][name] = {"rows": len(rows), "sha256": sha256(path)}
        manifest = world.manifest()
        manifest.update(
            files=file_audit[cohort],
            config_sha256=sha256(config_path),
            natural_retention_source=str(source.resolve()),
            natural_retention_source_sha256=sha256(source),
        )
        _write_json(destination / "manifest.json", manifest)
        file_audit[cohort]["manifest.json"] = {"sha256": sha256(destination / "manifest.json")}
    audit = {
        "schema_version": D.SCHEMA_VERSION,
        "status": "complete_cpu_materialization",
        "config": str(config_path.resolve()),
        "config_sha256": sha256(config_path),
        "natural_retention_source": str(source.resolve()),
        "natural_retention_source_sha256": sha256(source),
        "natural_old_prediction_fields_reused": False,
        "teacher_sampling_performed": False,
        "training_performed": False,
        "cohort_isolation": isolation,
        "world_audits": audits,
        "files": file_audit,
        "row_id_contract": "unique within each file; shared questions retain IDs across arm files",
        "location_retention_contract": "new at addition; rewrite retention uses addition endpoint",
        "data_module_sha256": sha256(D.__file__),
        "materializer_sha256": sha256(__file__),
    }
    _write_json(output / "audit.json", audit)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--factorial-batch", type=Path, default=DEFAULT_FACTORIAL)
    args = parser.parse_args()
    audit = materialize(args.config, args.output, args.factorial_batch)
    print(
        json.dumps(
            {
                "status": audit["status"],
                "output": str(args.output.resolve()),
                "counts": {c: a["file_counts"] for c, a in audit["world_audits"].items()},
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
