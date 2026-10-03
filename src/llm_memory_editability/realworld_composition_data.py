"""Source-grounded 2Wiki closed-book data preparation, without model access.

Answer normalization and alias scoring follow the pinned author's v1.1 scorer.
The original questions and facts are retained; this is an adapted task in which
evaluation facts may be learned separately, never with evaluation chain labels.
"""

from __future__ import annotations

import hashlib
import json
import re
import string
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

ARCHIVE_SHA = "95df2bf56fdabe034e27aebc580e02264232203cf52552f9efe8a919e5529eef"
SPLIT_SALT = "2wiki-realworld-v1:"
QUESTION_TEMPLATES = {
    "father": "Who is the father of {subject}?",
    "mother": "Who is the mother of {subject}?",
    "child": "Who is the child of {subject}?",
    "spouse": "Who is the spouse of {subject}?",
    "director": "Who directed {subject}?",
    "performer": "Who performed {subject}?",
    "composer": "Who composed {subject}?",
    "creator": "Who created {subject}?",
    "editor": "Who edited {subject}?",
    "presenter": "Who presented {subject}?",
    "manufacturer": "Who manufactured {subject}?",
    "publisher": "Who published {subject}?",
    "founded by": "Who founded {subject}?",
    "student of": "Who was the teacher of {subject}?",
    "educated at": "Where was {subject} educated?",
    "employer": "Who employed {subject}?",
    "date of birth": "On what date was {subject} born?",
    "date of death": "On what date did {subject} die?",
    "place of birth": "Where was {subject} born?",
    "place of death": "Where did {subject} die?",
    "place of burial": "Where was {subject} buried?",
    "place of detention": "Where was {subject} detained?",
    "cause of death": "What was the cause of death of {subject}?",
    "country of citizenship": "What is the country of citizenship of {subject}?",
    "country": "In which country is {subject}?",
    "inception": "On what date was {subject} established?",
    "award received": "What award did {subject} receive?",
    "has part": "What is a part of {subject}?",
}


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def order(value, salt=SPLIT_SALT):
    return hashlib.sha256((salt + str(value)).encode()).hexdigest()


def normalize_answer(value):
    value = "".join(c for c in value.lower() if c not in string.punctuation)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", value).split())


def answer_scores(prediction, answers):
    """Max EM/F1 across the author's canonical answer and entity aliases."""
    p = normalize_answer(prediction)
    scores = []
    for answer in answers:
        a = normalize_answer(answer)
        if (p in {"yes", "no", "noanswer"} or a in {"yes", "no", "noanswer"}) and p != a:
            f1 = 0.0
        else:
            common = sum((Counter(p.split()) & Counter(a.split())).values())
            f1 = 2 * common / (len(p.split()) + len(a.split())) if common else 0.0
        scores.append((float(p == a), f1))
    return max((s[0] for s in scores), default=0.0), max((s[1] for s in scores), default=0.0)


def entity_id(identifier, surface):
    if re.fullmatch(r"Q\d+", str(identifier)):
        return identifier
    return "literal:" + " ".join(str(surface).casefold().split())


def extract_chain(row):
    edges, surfaces = row.get("evidences_id", []), row.get("evidences", [])
    if row.get("type") != "compositional" or len(edges) != 2 or len(surfaces) != 2:
        return None
    if any(len(e) != 3 for e in edges + surfaces):
        return None
    converted = [
        (entity_id(e[0], s[0]), str(e[1]), entity_id(e[2], s[2]))
        for e, s in zip(edges, surfaces, strict=True)
    ]
    for indices in [(0, 1), (1, 0)]:
        first, second = [converted[i] for i in indices]
        if first[2] == second[0] and first[0] != second[2]:
            return (first, second), [surfaces[i] for i in indices]
    return None


class Components:
    def __init__(self):
        self.parent = {}

    def find(self, node):
        self.parent.setdefault(node, node)
        while self.parent[node] != node:
            self.parent[node] = self.parent[self.parent[node]]
            node = self.parent[node]
        return node

    def union(self, first, second):
        first, second = self.find(first), self.find(second)
        if first != second:
            self.parent[max(first, second)] = min(first, second)


def support_sentences(row):
    context = {title: sentences for title, sentences in row["context"]}
    result = []
    for title, index in row["supporting_facts"]:
        if title not in context or not 0 <= index < len(context[title]):
            result.append(
                {"title": title, "index": index, "sentence": "", "invalid_support_reference": True}
            )
            continue
        result.append({"title": title, "index": index, "sentence": context[title][index]})
    return result


def prior_cases(project):
    path = Path(project) / "docs/development-artifacts/twohop-frozen-v1/case-manifest.json"
    if not path.exists():
        return set()
    return {r["id"] for r in json.loads(path.read_text()) if r["dataset"] == "2wiki"}


def extract_archive(archive, project):
    if sha256(archive) != ARCHIVE_SHA:
        raise ValueError("2Wiki archive differs from the pinned author data")
    records, aliases, labels = [], {}, defaultdict(Counter)
    groups = Components()
    with zipfile.ZipFile(archive) as source:
        for line in source.read("id_aliases.json").decode().splitlines():
            entry = json.loads(line)
            aliases[entry["Q_id"]] = entry["aliases"] + entry.get("demonyms", [])
        for split in ("train", "dev"):
            for row in json.loads(source.read(split + ".json")):
                parsed = extract_chain(row)
                if parsed is None:
                    continue
                edges, surfaces = parsed
                groups.union(edges[0][0], edges[0][2])
                for edge, surface in zip(edges, surfaces, strict=True):
                    labels[edge[0]][surface[0]] += 1
                    labels[edge[2]][surface[2]] += 1
                records.append(
                    {
                        "id": row["_id"],
                        "source_split": split,
                        "question": row["question"],
                        "answer": row["answer"],
                        "answer_id": row.get("answer_id"),
                        "edges": edges,
                        "surfaces": surfaces,
                        "support": support_sentences(row),
                    }
                )
    canonical = {
        key: sorted(counts, key=lambda s: (-counts[s], s))[0] for key, counts in labels.items()
    }
    alias_owners = defaultdict(set)
    for identifier, names in aliases.items():
        for name in names:
            alias_owners[normalize_answer(name)].add(identifier)
    old = prior_cases(project)
    subject_relation = defaultdict(set)
    for row in records:
        for edge in row["edges"]:
            subject_relation[edge[:2]].add(edge[2])
    ambiguous = {key for key, values in subject_relation.items() if len(values) > 1}
    for row in records:
        row["group"] = groups.find(row["edges"][0][0])
        row["development_group"] = int(order(row["group"])[:8], 16) % 5 == 0
        row["known_prior_case"] = row["id"] in old
        row["aliases"] = aliases.get(row["answer_id"], [])
        row["unambiguous_aliases"] = [
            a for a in row["aliases"] if len(alias_owners[normalize_answer(a)]) == 1
        ]
        row["structural_errors"] = []
        if any(s.get("invalid_support_reference") for s in row["support"]):
            row["structural_errors"].append("invalid_support_sentence_reference")
        if any(edge[:2] in ambiguous for edge in row["edges"]):
            row["structural_errors"].append("multiple_objects_for_subject_relation")
        if row["answer_id"] and row["answer_id"] != row["edges"][1][2]:
            row["structural_errors"].append("answer_id_does_not_match_chain")
        if not row["answer_id"] and normalize_answer(row["answer"]) != normalize_answer(
            row["surfaces"][1][2]
        ):
            row["structural_errors"].append("literal_answer_does_not_match_chain")
        if any(edge[1] not in QUESTION_TEMPLATES for edge in row["edges"]):
            row["structural_errors"].append("unreviewed_relation_template")
        row["support_surface_flags"] = []
        text = normalize_answer(" ".join(s["title"] + " " + s["sentence"] for s in row["support"]))
        for i, (edge, surface) in enumerate(zip(row["edges"], row["surfaces"], strict=True)):
            for j in [0, 2]:
                choices = [surface[j], *aliases.get(edge[j], [])]
                if not any(normalize_answer(c) in text for c in choices if normalize_answer(c)):
                    row["support_surface_flags"].append(
                        f"edge{i}_position{j}_not_literal_in_support"
                    )
    return records, aliases, canonical


def review_sample(records, size=200):
    """Round-robin relation pairs, then fixed hashes; independent of model scores."""
    result = []
    for split in ("train", "dev"):
        by_pair = defaultdict(list)
        for row in records:
            if row["source_split"] == split:
                by_pair[tuple(e[1] for e in row["edges"])].append(row)
        for rows in by_pair.values():
            rows.sort(key=lambda r: order(r["id"], "2wiki-semantic-review-v1:"))
        chosen = []
        while len(chosen) < size:
            added = False
            for pair in sorted(by_pair, key=lambda p: order(p, "relation-review:")):
                if by_pair[pair]:
                    chosen.append(by_pair[pair].pop(0))
                    added = True
                    if len(chosen) == size:
                        break
            if not added:
                break
        result.extend(chosen)
    return result


def atomic_question(subject, relation):
    return QUESTION_TEMPLATES[relation].format(subject=subject)


def prompt(question):
    return "Question: " + question + "\nAnswer:"


def encode_example(tokenizer, question, answer, limit=256, generation_limit=64):
    prefix = tokenizer.encode(prompt(question), add_special_tokens=False)
    target = tokenizer.encode(" " + answer, add_special_tokens=False) + [tokenizer.eos_token_id]
    joint = tokenizer.encode(prompt(question) + " " + answer, add_special_tokens=False)
    if joint != prefix + target[:-1]:
        raise ValueError("Answer boundary changes joint tokenization")
    if len(prefix) + len(target) > limit or len(target) > generation_limit:
        raise ValueError("Example exceeds the frozen length limits; never truncate an answer")
    return {"prefix": prefix, "target": target, "input": (prefix + target)[:-1]}
