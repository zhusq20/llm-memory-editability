#!/usr/bin/env python3
"""Freeze source-grouped benchmark subsets and natural counterfactual donors."""

import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/twohop-frozen-v1"
ART = ROOT / "docs/development-artifacts/twohop-frozen-v1"
OLD = ROOT / "data/twohop-hebbian-v1"


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def order(value):
    return hashlib.sha256(f"142913:{value}".encode()).hexdigest()


def normalize(value):
    import re
    import string

    value = value.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", value).split())


def prepare():
    if (ART / "data-lock.json").exists():
        raise FileExistsError("Data lock exists; do not silently resample")
    cfg = read(ROOT / "configs/twohop-frozen-v1.json")
    pools = {("mquake", s): [] for s in ["development", "evaluation"]}
    source = {str(r["case_id"]): r for r in read(OLD / "raw/MQuAKE-CF-3k-v2.json")}
    for row in read(OLD / "mquake-candidates.json"):
        r = source[str(row["case_id"])]
        s, b, c = r["orig"]["triples_labeled"][0][0], r["single_hops"][0]["answer"], r["answer"]
        q2 = r["single_hops"][1]["question"]
        pools[("mquake", row["split"])].append(
            dict(
                dataset="mquake",
                id=str(r["case_id"]),
                split=row["split"],
                group=row["group"],
                question=r["questions"][0],
                paraphrases=r["questions"][1:],
                q1=r["single_hops"][0]["question"],
                q2=q2,
                q2_template=q2.replace(b, "{bridge}"),
                bridge=b,
                answer=c,
                aliases=r.get("answer_alias", []),
                bridge_aliases=r["single_hops"][0].get("answer_alias", []),
                relation=r["orig"]["triples"][1][1],
                subject=s,
                context="",
                support_context="",
                first_context="",
                source_id=row["source_entity"],
                bridge_id=row["original_triples"][0][2],
                answer_id=row["original_triples"][1][2],
                scaffold_template_valid=b in q2,
            )
        )
    with zipfile.ZipFile(OLD / "raw/data_ids_april7.zip") as z:
        alias_rows = [json.loads(line) for line in z.read("id_aliases.json").decode().splitlines()]
        alias_map = {r["Q_id"]: r["aliases"] + r.get("demonyms", []) for r in alias_rows}
        alias_owners = {}
        for r in alias_rows:
            for alias in r["aliases"] + r.get("demonyms", []):
                alias_owners.setdefault(normalize(alias), set()).add(r["Q_id"])
        eval_sources = {r["chain"][0][0] for r in read(OLD / "2wiki-dev-candidates.json")}
        for part, split in [("train", "development"), ("dev", "evaluation")]:
            candidates = {r["id"]: r for r in read(OLD / f"2wiki-{part}-candidates.json")}
            pool = []
            for r in json.loads(z.read(f"{part}.json")):
                if r["_id"] not in candidates:
                    continue
                chain = candidates[r["_id"]]["chain"]
                if split == "development" and chain[0][0] in eval_sources:
                    continue
                edges = r["evidences"]
                if r["evidences_id"][0][0] != chain[0][0]:
                    edges = edges[::-1]
                # Entity IDs establish connectivity; surface aliases may differ.
                surface_mismatch = edges[0][2] != edges[1][0]
                s, rel1, b = edges[0]
                _, rel2, c = edges[1]
                titles = {t for t, _ in r["supporting_facts"]}

                def paragraphs(items):
                    return "\n\n".join(f"{t}: {' '.join(sentences)}" for t, sentences in items)

                a_id = r.get("answer_id")
                # Official aliases include cross-ID collisions; canonical scoring remains primary.
                aliases = alias_map.get(a_id, [])
                pool.append(
                    dict(
                        dataset="2wiki",
                        id=r["_id"],
                        split=split,
                        group=chain[0][0],
                        question=r["question"],
                        paraphrases=[],
                        q1=f"What is the {rel1} of {s}?",
                        q2=f"What is the {rel2} of {b}?",
                        q2_template=f"What is the {rel2} of {{bridge}}?",
                        bridge=b,
                        answer=r["answer"],
                        aliases=aliases,
                        bridge_aliases=alias_map.get(chain[0][2], []),
                        unambiguous_aliases=[
                            a for a in aliases if len(alias_owners[normalize(a)]) == 1
                        ],
                        relation=rel2,
                        subject=s,
                        context=paragraphs(r["context"]),
                        support_context=paragraphs(
                            [(t, sentences) for t, sentences in r["context"] if t in titles]
                        ),
                        first_context=f"{s} — {rel1} — {b}.",
                        source_id=chain[0][0],
                        bridge_id=chain[0][2],
                        answer_id=chain[1][2],
                        scaffold_template_valid=True,
                        bridge_surface_mismatch=surface_mismatch,
                    )
                )
            pools[("2wiki", split)] = pool
    selected = []
    audit = {}
    for (dataset, split), pool in pools.items():
        pool.sort(key=lambda r: order(r["id"]))
        used_groups = set()
        chosen = []
        for r in pool:
            if r["group"] in used_groups:
                continue
            used_groups.add(r["group"])
            chosen.append(r)
            if len(chosen) == cfg["behavior_per_dataset"][split]:
                break
        assert len(chosen) == cfg["behavior_per_dataset"][split]
        for index, r in enumerate(chosen):
            donors = [
                d
                for d in pool
                if d["relation"] == r["relation"]
                and d["bridge_id"] != r["bridge_id"]
                and d["answer_id"] != r["answer_id"]
                and normalize(d["answer"]) != normalize(r["answer"])
            ]
            donor = donors[0] if donors else None
            r["wrong_donor"] = (
                {k: donor[k] for k in ["id", "q2", "answer", "bridge", "support_context"]}
                if donor
                else None
            )
            r["mechanism"] = index < cfg["mechanism_per_dataset"][split]
            selected.append(r)
        audit[f"{dataset}:{split}"] = dict(
            pool=len(pool),
            selected=len(chosen),
            mechanism=sum(r["mechanism"] for r in chosen),
            missing_donors=sum(r["wrong_donor"] is None for r in chosen),
        )
    write(DATA / "cases.json", selected)
    write(
        ART / "case-manifest.json",
        [
            {
                k: r[k]
                for k in [
                    "dataset",
                    "id",
                    "split",
                    "group",
                    "mechanism",
                    "source_id",
                    "bridge_id",
                    "answer_id",
                ]
            }
            for r in selected
        ],
    )
    write(
        ART / "data-lock.json",
        dict(
            created_utc=datetime.now(timezone.utc).isoformat(),
            config=cfg,
            config_sha256=digest(ROOT / "configs/twohop-frozen-v1.json"),
            cases_sha256=digest(DATA / "cases.json"),
            script_sha256=digest(Path(__file__)),
            sources={
                str(p.relative_to(ROOT)): digest(p)
                for p in [
                    OLD / "raw/MQuAKE-CF-3k-v2.json",
                    OLD / "raw/data_ids_april7.zip",
                    OLD / "mquake-candidates.json",
                ]
            },
            audit=audit,
            two_wiki_colliding_alias_strings=sum(len(v) > 1 for v in alias_owners.values()),
            scoring=(
                "Canonical EM/F1 primary; supplied alias EM secondary; "
                "2Wiki collision-filtered alias EM also reported. No outcome-based selection."
            ),
            total=len(selected),
        ),
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    prepare()
