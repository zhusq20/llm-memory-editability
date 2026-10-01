#!/usr/bin/env python3
"""Audited stages for the real-model cross-position learning experiment."""

from __future__ import annotations

import argparse
import collections
import json
import shutil
import unicodedata

from transformers import AutoTokenizer

from llm_memory_editability.hebbian_data import normalize
from llm_memory_editability.qwen_path_learning import (
    ART,
    CONFIG,
    DATA,
    RELATIONS,
    ROOT,
    PathEngine,
    articles_from_lines,
    digest,
    encode_fact,
    now,
    read,
    stable,
    valid_aliases,
    write,
)


def prepare():
    if (ART / "candidate-lock.json").exists():
        raise RuntimeError("Candidate lock exists; do not overwrite")
    cfg = read(CONFIG)
    old = ROOT / "data/hebbian-learning-v1"
    raw = read(old / "source/counterfact.json")
    pools = read(old / "pools.json")
    used = {normalize(r["subject"]) for records in pools.values() for r in records}
    previous_baselines = {r["case_id"] for r in read(old / "baseline.json")}
    tokenizer = AutoTokenizer.from_pretrained(ROOT / cfg["model_source"], local_files_only=True)
    aliases = valid_aliases(raw)
    names = collections.defaultdict(set)
    targets = collections.defaultdict(set)
    for row in raw:
        r = row["requested_rewrite"]
        normalized = normalize(r["subject"])
        near = "".join(c for c in unicodedata.normalize("NFKD", normalized) if c.isalnum())
        names[near].add(normalized)
        targets[(normalized, r["relation_id"])].add(r["target_true"]["id"])
    ambiguous = set().union(*(v for v in names.values() if len(v) > 1))
    candidates, audit, seen = [], [], set()
    ordered = sorted(raw, key=lambda r: stable(f"{cfg['seed']}:case:{r['case_id']}"))
    for row in ordered:
        r = row["requested_rewrite"]
        subject, answer = r["subject"].strip(), r["target_true"]["str"].strip()
        group, relation = normalize(subject), r["relation_id"]
        reason = None
        if relation not in RELATIONS:
            reason = "outside_locked_relations"
        elif group in used:
            reason = "previous_experiment_subject"
        elif group in ambiguous:
            reason = "ambiguous_name_variant"
        elif len(targets[(group, relation)]) > 1:
            reason = "conflicting_source_targets"
        elif group in seen:
            reason = "one_fact_per_subject"
        elif not subject or not answer or any(c in subject + answer for c in "\n\r{}"):
            reason = "invalid_subject_or_answer"
        elif relation in {"P176", "P178"} and answer in {"Iran", "Douglas", "Square"}:
            reason = "ambiguous_source_answer"
        else:
            statements = [template.format(subject) for template in RELATIONS[relation]]
            enc = [encode_fact(tokenizer, subject, s, answer, v) for v, s in enumerate(statements)]
            enc.append(encode_fact(tokenizer, subject, statements[0], answer, 3, plain=True))
            if any(e["prompt_tokens"] > 192 or e["answer_tokens"] > 8 for e in enc):
                reason = "token_limit"
        if reason:
            audit.append({"case_id": row["case_id"], "decision": "exclude", "reason": reason})
            continue
        seen.add(group)
        fraction = int(stable(f"{cfg['seed']}:role:{group}")[:16], 16) / 2**64
        role = next(
            role
            for threshold, role in [
                (0.08, "E_dev"),
                (0.20, "E_confirm"),
                (0.40, "R"),
                (0.60, "V"),
                (1.0, "U"),
            ]
            if fraction < threshold
        )
        for e in enc:
            e["key"] = f"cf-{row['case_id']}-v{e['view']}"
        record = {
            "case_id": row["case_id"],
            "subject": subject,
            "subject_group": group,
            "relation_id": relation,
            "answer": answer,
            "aliases": aliases[r["target_true"]["id"]],
            "target_id": r["target_true"]["id"],
            "role": role,
            "statements": statements,
            "encoded": enc,
            "historical_baseline_access": row["case_id"] in previous_baselines,
        }
        candidates.append(record)
        audit.append(
            {
                "case_id": row["case_id"],
                "decision": "include",
                "role": role,
                "truth_status": "CounterFact original target; not independently fact-checked",
                "semantic_audit": "explicit relation templates; subject spans; original targets",
                "historical_baseline_access": record["historical_baseline_access"],
            }
        )
    write(DATA / "candidates.json", candidates)
    write(ART / "candidate-audit.json", audit)
    # Article-separated WikiText windows; earlier experiments accessed this corpus.
    import pyarrow.parquet as pq

    article_rows = []
    for split in ("train", "test"):
        path = old / f"source/wikitext-2-raw-v1/{split}-00000-of-00001.parquet"
        lines = pq.read_table(path).column("text").to_pylist()
        for title, text in articles_from_lines(lines):
            ids = tokenizer.encode(text, add_special_tokens=False)
            if len(ids) >= cfg["text_tokens"]:
                article_rows.append(
                    {"split": split, "title": title, "ids": ids[: cfg["text_tokens"]]}
                )
    article_rows.sort(key=lambda r: stable(f"{cfg['seed']}:article:{normalize(r['title'])}"))
    texts, assigned = {}, set()
    # All three sets may draw from both official splits, but never share an article.
    cursor = 0
    for role, count in cfg["text_counts"].items():
        texts[role] = []
        while len(texts[role]) < count:
            row = article_rows[cursor]
            cursor += 1
            title = normalize(row["title"])
            if title in assigned:
                continue
            assigned.add(title)
            ids = row["ids"]
            texts[role].append(
                {
                    "key": f"wiki-{stable(title)[:16]}",
                    "article": row["title"],
                    "source_split": row["split"],
                    "input_ids": ids,
                    "loss_mask": [0] + [1] * (len(ids) - 1),
                    "roles": ["text"] * len(ids),
                    "answer_start": 1,
                }
            )
    write(DATA / "text.json", texts)
    source_files = [
        CONFIG,
        ROOT / "scripts/run_qwen_path_learning.py",
        ROOT / "src/llm_memory_editability/qwen_path_learning.py",
        old / "source/counterfact.json",
        old / "pools.json",
        old / "baseline.json",
        DATA / "candidates.json",
        DATA / "text.json",
    ]
    write(
        ART / "candidate-lock.json",
        {
            "time": now(),
            "status": "roles_and_templates_locked_before_new_baseline",
            "counts": dict(collections.Counter(r["role"] for r in candidates)),
            "historical_access": sum(r["historical_baseline_access"] for r in candidates),
            "files": {str(p.relative_to(ROOT)): digest(p) for p in source_files},
            "text_articles_disjoint": True,
            "limitations": [
                "Source facts are benchmark labels, not independently verified current truths",
                "Canonical relation templates are author-written; labels are original",
                "Corpus and some entities had historical baseline access; "
                "newly selected updates are disjoint from old experiments",
            ],
        },
    )
    print(
        json.dumps(
            {
                "prepared": len(candidates),
                "roles": dict(collections.Counter(r["role"] for r in candidates)),
            }
        ),
        flush=True,
    )


def baseline(args):
    out = DATA / f"baseline-{args.shard}.jsonl"
    completed = set()
    if out.exists():
        completed = {json.loads(line)["case_id"] for line in out.read_text().splitlines()}
    rows = [
        r
        for i, r in enumerate(read(DATA / "candidates.json"))
        if i % args.shards == args.shard and r["case_id"] not in completed
    ]
    engine = PathEngine(args.device)
    with out.open("a") as stream:
        for start in range(0, len(rows), 24):
            chunk = rows[start : start + 24]
            for row in engine.evaluate(chunk, with_kl=False, batch_size=24):
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            write(
                ART / f"baseline-status-{args.shard}.json",
                {
                    "time": now(),
                    "done": len(completed) + min(start + 24, len(rows)),
                    "total": len(completed) + len(rows),
                    "device": args.device,
                },
            )
    write(
        ART / f"baseline-complete-{args.shard}.json",
        {"time": now(), "sha256": digest(out), "rows": len(completed) + len(rows)},
    )


def select():
    cfg = read(CONFIG)
    baseline_rows = {}
    for path in sorted(DATA.glob("baseline-*.jsonl")):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            assert row["case_id"] not in baseline_rows
            baseline_rows[row["case_id"]] = row
    candidates = read(DATA / "candidates.json")
    assert len(baseline_rows) == len(candidates), (len(baseline_rows), len(candidates))
    selected, coverage = {}, {}
    quarantine = (
        read(ART / "semantic-revision.json")["quarantined_case_ids"]
        if (ART / "semantic-revision.json").exists()
        else []
    )
    for role, count in cfg["counts"].items():
        rows = [
            r
            for r in candidates
            if r["case_id"] not in quarantine
            and r["role"] == role
            and bool(baseline_rows[r["case_id"]]["correct"]) == (not role.startswith("E"))
        ]
        rows.sort(key=lambda r: stable(f"{cfg['seed']}:select:{role}:{r['case_id']}"))
        coverage[role] = {"available": len(rows), "requested": count}
        if len(rows) < count:
            write(ART / "selection-coverage.json", coverage)
            raise RuntimeError(f"Insufficient baseline-qualified {role}: {len(rows)} < {count}")
        selected[role] = rows[:count]
    all_subjects = [r["subject_group"] for rows in selected.values() for r in rows]
    assert len(all_subjects) == len(set(all_subjects))
    write(DATA / "selection.json", selected)
    write(ART / "selection-coverage.json", coverage)
    write(
        ART / "selection-lock.json",
        {
            "time": now(),
            "selection_sha256": digest(DATA / "selection.json"),
            "candidate_lock_sha256": digest(ART / "candidate-lock.json"),
            "coverage": coverage,
            "baseline_files": {p.name: digest(p) for p in sorted(DATA.glob("baseline-*.jsonl"))},
            "subject_disjoint": True,
        },
    )
    print(json.dumps(coverage), flush=True)


def semantic_revision():
    """Preserve initial screening; repair untrained views and quarantine source ambiguities."""
    assert not (ART / "execution-lock.json").exists()
    backup = ART / "pre-semantic-revision"
    backup.mkdir(exist_ok=False)
    for path in [
        DATA / "candidates.json",
        DATA / "selection.json",
        ART / "selection-lock.json",
        ART / "selection-coverage.json",
    ]:
        shutil.copyfile(path, backup / path.name)
    cfg = read(CONFIG)
    tokenizer = AutoTokenizer.from_pretrained(ROOT / cfg["model_source"], local_files_only=True)
    records = read(DATA / "candidates.json")
    changed = []
    for r in records:
        if r["relation_id"] not in {"P19", "P20"}:
            continue
        statement = RELATIONS[r["relation_id"]][2].format(r["subject"])
        enc = encode_fact(tokenizer, r["subject"], statement, r["answer"], 2)
        enc["key"] = r["encoded"][2]["key"]
        r["statements"][2], r["encoded"][2] = statement, enc
        changed.append(r["case_id"])
    write(DATA / "candidates.json", records)
    write(
        ART / "semantic-revision.json",
        {
            "time": now(),
            "before_training": True,
            "screening_view_0_unchanged": True,
            "changed_view": 2,
            "reason": "Birth/death places need not be cities; use location-neutral paraphrases",
            "changed_case_ids": changed,
            "quarantined_case_ids": [36, 5698, 17421, 11928],
            "quarantine_reasons": {
                "36": "Native-language historical attribution unresolved; no verified replacement",
                "5698": "Ronan Keating with Australia origin: unresolved entity/relation ambiguity",
                "17421": "Developer versus parent-company ambiguity: source Microsoft, "
                "official acquisition identifies developer Mojang",
                "11928": "Source Finland conflicts with the federation's Sweden origin account",
            },
            "sources": [
                "https://blogs.microsoft.com/blog/2014/09/15/minecraft-join-microsoft/",
                "https://archive.floorball.sport/this-is-floorball/history-in-short/",
            ],
            "limitations": "Author semantic screening with targeted primary-source checks; "
            "not exhaustive independent verification of every benchmark fact",
            "candidate_sha256": digest(DATA / "candidates.json"),
        },
    )
    select()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=[
            "prepare",
            "baseline",
            "select",
            "semantic-revision",
            "preflight",
            "freeze",
            "train",
            "select-lr",
            "diagnose",
            "decide",
        ],
    )
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=8)
    parser.add_argument("--episode", type=int, default=0)
    args = parser.parse_args()
    from llm_memory_editability.qwen_path_train import (
        candidate_decision,
        diagnose,
        freeze,
        preflight,
        select_learning_rates,
        train_shard,
    )

    {
        "prepare": prepare,
        "baseline": lambda: baseline(args),
        "select": select,
        "semantic-revision": semantic_revision,
        "preflight": lambda: preflight(args.device),
        "freeze": freeze,
        "train": lambda: train_shard(args.device, args.shard, args.shards),
        "select-lr": select_learning_rates,
        "diagnose": lambda: diagnose(args.device, args.episode),
        "decide": candidate_decision,
    }[args.stage]()


if __name__ == "__main__":
    main()
