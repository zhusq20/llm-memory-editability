#!/usr/bin/env python3
"""Snapshot public benchmark sources and audit two-hop candidates without model selection."""

from __future__ import annotations

import hashlib
import json
import re
import urllib.request
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/twohop-hebbian-v1"
ARTIFACTS = ROOT / "docs/development-artifacts/twohop-hebbian-v1"
MQUAKE_REVISION = "fb43dadc2d8cd19d08ce81c63d957b59deb3f3cd"
MQUAKE_URL = (
    f"https://raw.githubusercontent.com/princeton-nlp/MQuAKE/{MQUAKE_REVISION}/"
    "datasets/MQuAKE-CF-3k-v2.json"
)
WIKI_URL = "https://www.dropbox.com/s/ms2m13252h6xubs/data_ids_april7.zip?dl=1"


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def download(url, filename):
    path = DATA / "raw" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temporary = path.with_suffix(path.suffix + ".part")
        request = urllib.request.Request(url, headers={"User-Agent": "llm-memory-research"})
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as out:
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                out.write(chunk)
        temporary.replace(path)
    return path, dict(url=url, bytes=path.stat().st_size, sha256=digest(path))


def audit_mquake(path):
    rows = json.loads(path.read_text())
    candidates, excluded = [], Counter()
    for row in rows:
        original = row["orig"]["triples"]
        updated = row["orig"]["new_triples"]
        if len(original) != 2 or len(updated) != 2:
            excluded["not_two_hops_before_and_after"] += 1
            continue
        if len(row["requested_rewrite"]) != 1:
            excluded["not_one_edit"] += 1
            continue
        if original[0][2] != original[1][0] or updated[0][2] != updated[1][0]:
            excluded["disconnected_chain"] += 1
            continue
        edit_index = row["orig"]["edit_triples_idx"]
        if len(edit_index) != 1 or edit_index[0] not in [0, 1]:
            excluded["invalid_edit_index"] += 1
            continue
        if original[-1][-1] == updated[-1][-1]:
            excluded["unchanged_final_entity"] += 1
            continue
        edited_subject = row["orig"]["edit_triples"][0][0]
        candidates.append(
            dict(
                case_id=row["case_id"],
                source_entity=original[0][0],
                edited_subject=edited_subject,
                edited_hop=edit_index[0] + 1,
                original_triples=original,
                new_triples=updated,
                relation_pair=[original[0][1], original[1][1]],
                question_count=len(row["questions"]),
            )
        )
    # Union source and edited-subject identities, avoiding case/paraphrase leakage.
    parent = {}

    def find(value):
        parent.setdefault(value, value)
        if parent[value] != value:
            parent[value] = find(parent[value])
        return parent[value]

    for row in candidates:
        a, b = find(row["source_entity"]), find(row["edited_subject"])
        parent[max(a, b)] = min(a, b)
    for row in candidates:
        group = find(row["source_entity"])
        bucket = int(hashlib.sha256(("142901:" + group).encode()).hexdigest()[:8], 16) % 5
        row.update(group=group, split="development" if bucket == 0 else "evaluation")
    write(DATA / "mquake-candidates.json", candidates)
    write(ARTIFACTS / "mquake-split-manifest.json", candidates)
    return dict(
        total=len(rows),
        candidates=len(candidates),
        excluded=dict(excluded),
        splits=dict(Counter(row["split"] for row in candidates)),
        edited_hops=dict(Counter(row["edited_hop"] for row in candidates)),
        unique_edited_subjects=len({row["edited_subject"] for row in candidates}),
        source_edit_components=len({row["group"] for row in candidates}),
        largest_component=max(Counter(row["group"] for row in candidates).values(), default=0),
        candidate_manifest_sha256=digest(DATA / "mquake-candidates.json"),
        scope="Structural candidates, not independently verified current-world facts. "
        "Dataset was accessed by earlier project work; "
        "these are new fixed groups, not pristine data.",
    )


def wiki_chain(row):
    # ID annotations identify entities, and the ID/label relation names need not be Q/P codes.
    edges = row.get("evidences_id", [])
    if row.get("type") != "compositional" or len(edges) != 2:
        return None
    if any(not isinstance(edge, list) or len(edge) != 3 for edge in edges):
        return None
    # date_information/number_information are placeholders, not shared entity IDs.
    labels = row.get("evidences", [])
    if len(labels) != len(edges):
        return None
    edges = [list(edge) for edge in edges]
    for edge, label in zip(edges, labels, strict=True):
        for position in [0, 2]:
            if not re.fullmatch(r"Q\d+", str(edge[position])):
                edge[position] = "literal:" + " ".join(str(label[position]).casefold().split())
    for first, second in [(edges[0], edges[1]), (edges[1], edges[0])]:
        if first[2] == second[0] and first[0] != second[2]:
            return [first, second]
    return None


def audit_wiki(path):
    audit, chains, samples = {}, {}, {}
    with zipfile.ZipFile(path) as archive:
        members = archive.namelist()
        audit["members"] = members
        for split in ["train", "dev"]:
            files = [name for name in members if Path(name).name == split + ".json"]
            if len(files) != 1:
                raise ValueError(f"Expected one {split}.json, found {files}")
            with archive.open(files[0]) as stream:
                rows = json.load(stream)
            selected = []
            for row in rows:
                chain = wiki_chain(row)
                if chain is not None:
                    selected.append(
                        dict(id=row["_id"], chain=chain, answer_id=row.get("answer_id"))
                    )
                    samples.setdefault(split, row)
            chains[split] = {json.dumps(row["chain"], sort_keys=True) for row in selected}
            audit[split] = dict(
                total=len(rows),
                types=dict(Counter(r["type"] for r in rows)),
                connected_two_edge_compositional=len(selected),
                literal_answer_cases=sum(row["answer_id"] is None for row in selected),
            )
            write(DATA / f"2wiki-{split}-candidates.json", selected)
            # Keep only annotated metadata, not the full copyrighted contexts, in artifacts.
            audit[split]["manifest_sha256"] = digest(DATA / f"2wiki-{split}-candidates.json")
        overlap = chains["train"] & chains["dev"]
        audit["exact_chain_overlap_train_dev"] = len(overlap)
        audit["policy"] = (
            "Official train supplies training/development; official dev is final evaluation. "
            "Exact chain overlap is excluded from evaluation before model access. "
            "Candidate counts are structural, not model-mastery filtered."
        )
        write(DATA / "2wiki-schema-examples.json", samples)
    return audit


def main():
    mquake, source_mquake = download(MQUAKE_URL, "MQuAKE-CF-3k-v2.json")
    first = audit_mquake(mquake)
    print("MQuAKE " + json.dumps(first), flush=True)
    wiki, source_wiki = download(WIKI_URL, "data_ids_april7.zip")
    second = audit_wiki(wiki)
    sources = dict(
        mquake=dict(**source_mquake, repository_revision=MQUAKE_REVISION),
        two_wiki=dict(
            **source_wiki,
            repository_revision="13800e5be57df1b4040b9b1588c6c811779e69e9",
            note="Official README links this archive; SHA256 pins downloaded bytes.",
        ),
    )
    result = dict(
        generated_utc=datetime.now(timezone.utc).isoformat(),
        sources=sources,
        mquake=first,
        two_wiki=second,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    write(ARTIFACTS / "dataset-audit.json", result)
    print("2Wiki " + json.dumps(second), flush=True)


if __name__ == "__main__":
    main()
