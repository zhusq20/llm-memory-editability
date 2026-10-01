"""CounterFact true-label data, explicit audit decisions and indivisible subject pools."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter, defaultdict

import numpy as np

from .hebbian_learning import ARTIFACTS, DATA, config, now, read_json, sha256, write_json

TEMPLATE = "Complete the statement with the missing answer only.\nStatement: {}\nAnswer:"


def normalize(text):
    return " ".join(unicodedata.normalize("NFKC", text).split()).casefold()


def normalize_answer(text):
    text = normalize(text)
    while text and unicodedata.category(text[-1]).startswith("P"):
        text = text[:-1].rstrip()
    return text


def score_answer(prediction, aliases):
    lines = [x.strip() for x in prediction.splitlines() if x.strip()]
    first = lines[0] if lines else ""
    return int(normalize_answer(first) in {normalize_answer(x) for x in aliases})


def encode_answer(tokenizer, prompt, answer):
    """Tokenize the complete string once; offset overlap includes leading answer space."""
    suffix = " " + answer
    text = prompt + suffix
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
    start = len(prompt)
    mask = [int(end > start) for begin, end in offsets]
    if any(begin < start < end for begin, end in offsets):
        raise ValueError("A token straddles the prompt/answer boundary")
    answer_start = mask.index(1)
    if ids[:answer_start] != tokenizer(prompt, add_special_tokens=False)["input_ids"]:
        raise ValueError("Prompt tokenization changes at the answer boundary")
    assert not any(mask[:answer_start]) and all(mask[answer_start:])
    return {
        "input_ids": ids + [tokenizer.eos_token_id],
        "loss_mask": mask + [1],
        "answer_start": answer_start,
        "answer_tokens": sum(mask),
        "prompt_tokens": answer_start,
    }


def stable_hash(text):
    return hashlib.sha256(text.encode()).hexdigest()


def clean_candidates(tokenizer):
    cfg = config()["data"]
    raw_path = DATA / "source/counterfact.json"
    raw = read_json(raw_path)
    audit, candidates, seen = [], [], {}
    conflicting = set()
    # Near-identical punctuation/accent variants are conservatively quarantined;
    # normalized identical subjects already share a group.
    subjects = defaultdict(set)
    for row in raw:
        name = normalize(row["requested_rewrite"]["subject"])
        key = "".join(c for c in unicodedata.normalize("NFKD", name) if c.isalnum())
        subjects[key].add(name)
    ambiguous = set().union(*(v for v in subjects.values() if len(v) > 1))
    for row in raw:
        r = row["requested_rewrite"]
        if r["relation_id"] not in cfg["relations"]:
            continue
        subject = " ".join(unicodedata.normalize("NFKC", r["subject"]).split())
        group = normalize(subject)
        answer = r["target_true"]["str"].strip()
        views = [r["prompt"].format(r["subject"])] + row["paraphrase_prompts"][:2]
        views = [" ".join(unicodedata.normalize("NFKC", p).split()) for p in views]
        reasons = []
        decision = "include"
        if len(views) != 3 or len(set(map(normalize, views))) != 3:
            reasons.append("missing_or_duplicate_views")
        if any(
            re.search(r"(?<!\w)" + re.escape(normalize(answer)) + r"(?!\w)", normalize(p))
            for p in views
        ):
            reasons.append("answer_leakage_in_prompt_or_subject")
        if any(normalize(subject) not in normalize(p) for p in views):
            reasons.append("subject_missing_from_view")
        if group in ambiguous:
            reasons.append("possible_subject_alias")
            decision = "review"
        # Some source paraphrases ask nationality, not native language.
        if r["relation_id"] == "P103" and any(p.endswith(", a native") for p in views):
            reasons.append("native_is_ambiguous_between_language_and_origin")
            decision = "review"
        # Manufacturer names truncated to a country or a non-unique first word.
        if r["relation_id"] in {"P176", "P178"} and answer in {"Iran", "Douglas", "Square"}:
            reasons.append("source_answer_name_requires_entity_review")
            decision = "review"
        encoded = []
        try:
            encoded = [encode_answer(tokenizer, TEMPLATE.format(p), answer) for p in views]
            if any(x["prompt_tokens"] > cfg["prompt_max_tokens"] for x in encoded):
                reasons.append("prompt_over_limit")
            if any(x["answer_tokens"] > cfg["answer_max_tokens"] for x in encoded):
                reasons.append("answer_over_limit")
        except ValueError as e:
            reasons.append(str(e))
        key = (group, r["relation_id"])
        if key in seen:
            previous = seen[key]
            if previous["target_id"] == r["target_true"]["id"]:
                previous["source_ids"].append(row["case_id"])
                reasons.append("merged_subject_relation_duplicate")
            else:
                reasons.append("conflicting_subject_relation_targets")
                decision = "review"
                conflicting.add(key)
        if reasons and decision == "include":
            decision = "exclude"
        record = {
            "case_id": row["case_id"],
            "source_ids": [row["case_id"]],
            "subject": subject,
            "subject_group": group,
            "relation_id": r["relation_id"],
            "target_id": r["target_true"]["id"],
            "answer": answer,
            "target_new_source_only": r["target_new"],
            "views": views,
            "encoded": encoded,
        }
        audit.append(
            {
                "case_id": row["case_id"],
                "decision": decision,
                "reasons": reasons,
                "audit_type": "programmatic",
                "subject_group": group,
            }
        )
        if decision == "include":
            candidates.append(record)
            seen[key] = record
    # Conflicting labels disqualify every side of a conflict, including the first row.
    conflict_ids = {
        r["case_id"] for r in candidates if (r["subject_group"], r["relation_id"]) in conflicting
    }
    for entry in audit:
        if entry["case_id"] in conflict_ids:
            entry["decision"] = "review"
            entry["reasons"].append("conflicting_subject_relation_targets")
    candidates = [r for r in candidates if r["case_id"] not in conflict_ids]
    prompt_owners = defaultdict(set)
    for record in candidates:
        for prompt in record["views"]:
            prompt_owners[normalize(prompt)].add(record["subject_group"])
    collision_ids = {
        r["case_id"]
        for r in candidates
        if any(len(prompt_owners[normalize(p)]) > 1 for p in r["views"])
    }
    for entry in audit:
        if entry["case_id"] in collision_ids:
            entry["decision"] = "review"
            entry["reasons"].append("cross_subject_prompt_collision")
    candidates = [r for r in candidates if r["case_id"] not in collision_ids]
    # Each pool gets a disjoint candidate reserve before any baseline outcome is observed.
    # Extra keep reserve accounts for its required pre-existing exact generation success.
    pool_weights = {k: n * (3 if k.endswith("keep") else 1) for k, n in cfg["pools"].items()}
    total_weight = sum(pool_weights.values())
    by_subject = defaultdict(list)
    for r in candidates:
        by_subject[r["subject_group"]].append(r)
    stratified = defaultdict(list)
    for group, rows in by_subject.items():
        stratified[min(r["relation_id"] for r in rows)].append(group)
    rng = np.random.default_rng(cfg["split_seed"])
    for _relation, groups in sorted(stratified.items()):
        groups.sort(key=stable_hash)
        rng.shuffle(groups)
        cumulative = 0.0
        start = 0
        for pool, weight in pool_weights.items():
            cumulative += weight / total_weight
            end = round(cumulative * len(groups))
            for group in groups[start:end]:
                for record in by_subject[group]:
                    record["candidate_pool"] = pool
            start = end
    write_json(DATA / "candidates.json", candidates)
    write_json(ARTIFACTS / "programmatic-audit.json", audit)
    write_json(
        ARTIFACTS / "subject-alias-audit.json",
        {
            "method": "NFKC/casefold grouping; punctuation/accent variants quarantined",
            "limitations": "No claim of exhaustive real-world entity resolution",
            "review_groups": [sorted(v) for v in subjects.values() if len(v) > 1],
        },
    )
    write_json(
        ARTIFACTS / "preparation.json",
        {
            "time": now(),
            "raw_count": len(raw),
            "source_sha256": sha256(raw_path),
            "candidate_count": len(candidates),
            "decisions": dict(Counter(x["decision"] for x in audit)),
            "reasons": dict(Counter(y for x in audit for y in x["reasons"])),
            "candidate_pools": dict(Counter(x["candidate_pool"] for x in candidates)),
            "candidate_sha256": sha256(DATA / "candidates.json"),
            "status": "awaiting_individual_semantic_audit_and_baseline",
            "source_license": (
                "CounterFact source snapshot; upstream ROME repository MIT; "
                "dataset-specific license not asserted"
            ),
        },
    )
    return candidates


def balanced_order(records, seed):
    rng = np.random.default_rng(seed)
    groups = defaultdict(list)
    for record in sorted(records, key=lambda r: r["case_id"]):
        groups[record["relation_id"]].append(record)
    for items in groups.values():
        rng.shuffle(items)
    result = []
    while any(groups.values()):
        for relation in sorted(groups):
            if groups[relation]:
                result.append(groups[relation].pop())
    return result


def semantic_audit():
    """Apply explicitly reviewed predicate rules; retain per-row evidence and limitations.

    This is a reproducible semantic rule audit, not a claim that a human or another
    model independently reviewed every source fact. Benchmark truth is assumed.
    """
    candidates = read_json(DATA / "candidates.json")
    result = []
    for record in candidates:
        issues, evidence = [], []
        subject = normalize(record["subject"])
        relation = record["relation_id"]
        for view in record["views"]:
            text = normalize(view)
            start = text.rfind(subject)
            suffix = text[start + len(subject) :].strip()
            prefix = text[:start]
            evidence.append({"predicate_suffix": suffix, "relation_prefix": prefix[-64:]})
            if relation == "P19" and suffix != "was born in":
                issues.append("origin_or_native_place_does_not_uniquely_mean_birthplace")
            if relation == "P103" and (suffix in {"spoke the language", ", speaker of"}):
                issues.append("spoken_language_does_not_uniquely_mean_mother_tongue")
            if (
                relation == "P103"
                and suffix == "is"
                and not any(
                    prefix.endswith(p) for p in ["the mother tongue of ", "the native language of "]
                )
            ):
                issues.append("missing_native_language_relation")
            if relation == "P176" and "developed by" in suffix:
                issues.append("developer_is_not_necessarily_manufacturer")
            if relation == "P178" and "manufactured by" in suffix:
                issues.append("manufacturer_is_not_necessarily_developer")
            if (
                relation in {"P364", "P407"}
                and suffix in {"is", "was"}
                and not any(
                    prefix.endswith(p) for p in ["the language of ", "the original language of "]
                )
            ):
                issues.append("missing_language_relation")
        result.append(
            {
                "case_id": record["case_id"],
                "decision": "review" if issues else "include",
                "reasons": sorted(set(issues)),
                "view_evidence": evidence,
                "audit_type": "model_authored_semantic_rules_applied_programmatically",
                "truth_status": "CounterFact original label; no independent fact verification",
            }
        )
    write_json(ARTIFACTS / "semantic-audit.json", result)
    write_json(
        ARTIFACTS / "semantic-audit-method.json",
        {
            "time": now(),
            "status": "conservative_template_semantics_audit",
            "decisions": dict(Counter(r["decision"] for r in result)),
            "limitations": [
                "Real-world truths and all entity aliases are not independently verified",
                "Creator/product/origin formulations use source benchmark semantics",
                "Model-authored rule audit; independent human review not performed",
            ],
            "primary_subset": "Only all-three-view records passing these conservative rules",
            "review_pool": "Excluded from all fitting and evaluation; never auto-promoted",
        },
    )
    return result


def lock_data():
    """No training may consume unaudited rows or an outcome-adapted subject split."""
    cfg = config()["data"]
    candidates = read_json(DATA / "candidates.json")
    semantic = {r["case_id"]: r for r in read_json(ARTIFACTS / "semantic-audit.json")}
    baseline = {r["case_id"]: r for r in read_json(DATA / "baseline.json")}
    pools, coverage = {}, {}
    for name, count in cfg["pools"].items():
        available = [
            r
            for r in candidates
            if r["candidate_pool"] == name
            and semantic.get(r["case_id"], {}).get("decision") == "include"
            and r["case_id"] in baseline
        ]
        if name.endswith("keep"):
            available = [r for r in available if baseline[r["case_id"]]["views"][0]["answer_em"]]
        ordered = balanced_order(available, cfg["split_seed"])
        if name.startswith(("B_", "C_")):

            def needs_learning(r):
                v = baseline[r["case_id"]]["views"]
                return not v[0]["answer_em"] and not all(x["answer_em"] for x in v[1:])

            ordered = [r for r in ordered if needs_learning(r)] + [
                r for r in ordered if not needs_learning(r)
            ]
        unit = cfg["episode_size"].get(name, 8)
        actual = min(count, len(ordered) // unit * unit)
        if actual < unit:
            raise ValueError(f"{name} has fewer than one complete batch/episode: {len(ordered)}")
        pools[name] = ordered[:actual]
        coverage[name] = {
            "planned": count,
            "available": len(ordered),
            "actual": actual,
            "relations": dict(Counter(r["relation_id"] for r in ordered[:actual])),
        }
    group_sets = {k: {r["subject_group"] for r in rows} for k, rows in pools.items()}
    names = list(pools)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            assert not group_sets[a] & group_sets[b], (a, b)
    aliases = defaultdict(set)
    for rows in pools.values():
        for r in rows:
            aliases[r["target_id"]].add(r["answer"])
    for rows in pools.values():
        for r in rows:
            r["aliases"] = sorted(aliases[r["target_id"]])
            r["baseline"] = baseline[r["case_id"]]
    episodes = {
        name: [
            [r["case_id"] for r in rows[i : i + cfg["episode_size"][name]]]
            for i in range(0, len(rows), cfg["episode_size"][name])
        ]
        for name, rows in pools.items()
        if name in cfg["episode_size"]
    }
    write_json(DATA / "pools.json", pools)
    write_json(DATA / "episodes.json", episodes)
    write_json(
        ARTIFACTS / "data-lock.json",
        {
            "time": now(),
            "coverage": coverage,
            "subject_disjoint": True,
            "files": {
                p: sha256(DATA / p)
                for p in [
                    "candidates.json",
                    "pools.json",
                    "episodes.json",
                    "baseline.json",
                    "text.json",
                ]
            },
            "semantic_audit_sha256": sha256(ARTIFACTS / "semantic-audit.json"),
            "revision": "Full-episode coverage reduction before model learning, where needed",
            "status": "locked",
        },
    )
    return coverage
