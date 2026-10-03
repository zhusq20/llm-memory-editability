"""Extract the author's data, then freeze reviewed, entity-grouped 2Wiki splits."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from transformers import GPT2TokenizerFast

from llm_memory_editability.realworld_composition_data import (
    ARCHIVE_SHA,
    atomic_question,
    encode_example,
    extract_archive,
    normalize_answer,
    order,
    review_sample,
    sha256,
    write_json,
)

PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / "data/realworld-composition-v1"
ART = PROJECT / "docs/development-artifacts/realworld-composition-v1"


def fact_key(edge):
    return json.dumps(edge, ensure_ascii=False, separators=(",", ":"))


def extract():
    if (DATA / "data-lock.json").exists():
        raise FileExistsError("Prepared data is frozen; preserve its bytes")
    rows, aliases, canonical = extract_archive(DATA / "raw/data_ids_april7.zip", PROJECT)
    DATA.mkdir(exist_ok=True)
    with (DATA / "candidates.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_json(DATA / "aliases.json", aliases)
    write_json(DATA / "canonical-labels.json", canonical)
    review = review_sample(rows)
    write_json(DATA / "semantic-review-packet.json", review)
    summary = {
        "source_sha256": ARCHIVE_SHA,
        "candidate_sha256": sha256(DATA / "candidates.jsonl"),
        "semantic_packet_sha256": sha256(DATA / "semantic-review-packet.json"),
        "semantic_review_complete": False,
        "relations": dict(Counter(e[1] for r in rows for e in r["edges"])),
        "splits": {},
    }
    for split in ("train", "dev"):
        selected = [r for r in rows if r["source_split"] == split]
        groups = Counter(r["group"] for r in selected)
        summary["splits"][split] = {
            "candidates": len(selected),
            "components": len(groups),
            "largest_components": groups.most_common(5),
            "development_group_rows": sum(r["development_group"] for r in selected),
            "structural_error_rows": sum(bool(r["structural_errors"]) for r in selected),
            "literal_support_flag_rows": sum(bool(r["support_surface_flags"]) for r in selected),
            "prior_cases": sum(r["known_prior_case"] for r in selected),
            "review_packet_rows": sum(r["source_split"] == split for r in review),
        }
    write_json(ART / "preparation-audit.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def freeze():
    if (DATA / "data-lock.json").exists():
        raise FileExistsError("Prepared data is frozen; do not silently resample")
    review_path = ART / "semantic-review.json"
    review = json.loads(review_path.read_text())
    packet = json.loads((DATA / "semantic-review-packet.json").read_text())
    if review["packet_sha256"] != sha256(DATA / "semantic-review-packet.json"):
        raise ValueError("Semantic review applies to a different source packet")
    verdicts = {r["id"]: r for r in review["cases"]}
    if set(verdicts) != {r["id"] for r in packet} or len(verdicts) != len(packet):
        raise ValueError("Every prespecified sample must have an explicit review result")
    if not all(r.get("reviewed") for r in verdicts.values()):
        raise ValueError("Semantic review contains unfinished cases")
    if any(
        r["verdict"] not in {"consistent", "ambiguous", "unsupported", "conflict", "uncertain"}
        for r in verdicts.values()
    ):
        raise ValueError("Invalid semantic review result")
    bad_cases = {k for k, v in verdicts.items() if v["verdict"] in {"unsupported", "conflict"}}
    bad_facts = {
        fact_key(row["edges"][index])
        for row in packet
        for index in verdicts[row["id"]].get("invalid_edges", [])
    }
    rows = [json.loads(line) for line in (DATA / "candidates.jsonl").open()]
    aliases = json.loads((DATA / "aliases.json").read_text())
    canonical = json.loads((DATA / "canonical-labels.json").read_text())
    alias_owners = defaultdict(set)
    for identifier, names in aliases.items():
        for name in names:
            alias_owners[normalize_answer(name)].add(identifier)
    for identifier, name in canonical.items():
        alias_owners[normalize_answer(name)].add(identifier)

    def unique_answers(identifier, answer):
        return [
            name
            for name in [answer, *aliases.get(identifier, [])]
            if alias_owners[normalize_answer(name)] == {identifier}
        ]

    reference = PROJECT / "results/grokking-reproduction-v1/reference/gpt2"
    tokenizer = GPT2TokenizerFast.from_pretrained(reference, local_files_only=True)
    assert len(tokenizer) == 50257, "Keep the author's base tokenizer, without entity tokens"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained(DATA / "tokenizer")
    atom_by_key = {}
    for row in rows:
        for edge in row["edges"]:
            key = fact_key(edge)
            if key not in atom_by_key:
                atom_by_key[key] = {
                    "id": order(key, "fact:")[:24],
                    "edge": edge,
                    "question": atomic_question(canonical[edge[0]], edge[1]),
                    "answer": canonical[edge[2]],
                    "aliases": aliases.get(edge[2], []),
                    "unambiguous_answers": unique_answers(edge[2], canonical[edge[2]]),
                    "source_ids": [],
                }
            atom_by_key[key]["source_ids"].append(row["id"])
    question_answers = defaultdict(set)
    for atom in atom_by_key.values():
        question_answers[normalize_answer(atom["question"])].add(atom["edge"][2])
        try:
            atom["encoded"] = encode_example(tokenizer, atom["question"], atom["answer"])
        except ValueError:
            bad_facts.add(fact_key(atom["edge"]))
    ambiguous_questions = {q for q, answers in question_answers.items() if len(answers) > 1}
    for atom in atom_by_key.values():
        if normalize_answer(atom["question"]) in ambiguous_questions:
            bad_facts.add(fact_key(atom["edge"]))
    exclusions = Counter()
    accepted = []
    for row in rows:
        reason = None
        if row["structural_errors"]:
            reason = "structural_error"
        elif row["edges"][0][0] == row["edges"][0][2]:
            reason = "head_equals_bridge"
        elif row["id"] in bad_cases:
            reason = "reviewed_source_or_question_conflict"
        elif any(fact_key(e) in bad_facts for e in row["edges"]):
            reason = "invalid_or_ambiguous_or_overlength_atomic_fact"
        else:
            try:
                row["encoded"] = encode_example(tokenizer, row["question"], row["answer"])
            except ValueError:
                reason = "combination_length_or_token_boundary"
        if reason:
            exclusions[row["source_split"] + ":" + reason] += 1
            continue
        row["atom_ids"] = [atom_by_key[fact_key(e)]["id"] for e in row["edges"]]
        row["unambiguous_answers"] = unique_answers(row["edges"][1][2], row["answer"])
        row["review_verdict"] = verdicts.get(row["id"], {}).get(
            "verdict", "not_individually_reviewed"
        )
        accepted.append(row)
    development = [r for r in accepted if r["source_split"] == "train" and r["development_group"]]
    formal_train = [
        r for r in accepted if r["source_split"] == "train" and not r["development_group"]
    ]
    formal_eval = [
        r
        for r in accepted
        if r["source_split"] == "dev" and not r["development_group"] and not r["known_prior_case"]
    ]
    # Same chain under different questions always receives the same split.
    for row in development:
        row["split"] = (
            "heldout"
            if int(order(fact_key(row["edges"]), "development-query:")[:8], 16) % 5 == 0
            else "train"
        )
    held_questions = {
        normalize_answer(r["question"]) for r in development if r["split"] == "heldout"
    }
    development = [
        r
        for r in development
        if r["split"] == "heldout" or normalize_answer(r["question"]) not in held_questions
    ]
    training_questions = {normalize_answer(r["question"]) for r in formal_train}
    training_chains = {fact_key(r["edges"]) for r in formal_train}
    formal_eval = [
        r
        for r in formal_eval
        if normalize_answer(r["question"]) not in training_questions
        and fact_key(r["edges"]) not in training_chains
    ]
    dev_entities = {e for r in development for e in [r["edges"][0][0], r["edges"][0][2]]}
    formal_entities = {
        e for r in formal_train + formal_eval for e in [r["edges"][0][0], r["edges"][0][2]]
    }
    assert not dev_entities & formal_entities, (
        "Development and confirmation head/bridge entities overlap"
    )
    results = {}
    for phase, all_chains, train_chains, eval_chains in [
        (
            "development",
            development,
            [r for r in development if r["split"] == "train"],
            [r for r in development if r["split"] == "heldout"],
        ),
        ("confirmation-candidate", formal_train + formal_eval, formal_train, formal_eval),
    ]:
        first = {r["atom_ids"][0] for r in train_chains}
        second = {r["atom_ids"][1] for r in train_chains}
        any_role = first | second
        used = {key for r in all_chains for key in r["atom_ids"]}
        atoms = sorted([a for a in atom_by_key.values() if a["id"] in used], key=lambda a: a["id"])
        direct = {(a["edge"][0], a["edge"][2]) for a in atoms}
        seen_pairs = {(r["edges"][0][0], r["edges"][1][2]) for r in train_chains}
        operations = {tuple(e[1] for e in r["edges"]) for r in train_chains}
        for row in all_chains:
            a, b = row["atom_ids"]
            row["role"] = ("I" if a in any_role else "O") + ("I" if b in any_role else "O")
            row["required_role"] = ("I" if a in first else "O") + ("I" if b in second else "O")
            pair = row["edges"][0][0], row["edges"][1][2]
            row["known_direct_shortcut"] = pair in direct
            row["head_tail_seen_in_combination_training"] = pair in seen_pairs
            row["operation_seen"] = tuple(e[1] for e in row["edges"]) in operations
            row.pop("support", None)
        ordered_train = sorted(train_chains, key=lambda r: r["id"])
        ordered_eval = sorted(eval_chains, key=lambda r: r["id"])
        panels = {}
        for role in ["II", "IO", "OI", "OO"]:
            pool = [r for r in ordered_eval if r["role"] == role]
            panels["test_" + role.lower()] = [
                r["id"] for r in sorted(pool, key=lambda r: order(r["id"], "panel:"))[:64]
            ]
        panels["train_composition"] = [
            r["id"] for r in sorted(ordered_train, key=lambda r: order(r["id"], "panel:"))[:128]
        ]
        required = {
            a
            for r in all_chains
            if r["id"] in {c for ids in panels.values() for c in ids}
            for a in r["atom_ids"]
        }
        atom_panel = required | {
            a["id"] for a in sorted(atoms, key=lambda a: order(a["id"], "atom-panel:"))[:128]
        }
        panels["atomic"] = sorted(atom_panel)
        dataset = {
            "atoms": atoms,
            "train_compositions": ordered_train,
            "evaluation_compositions": ordered_eval,
            "panels": panels,
        }
        write_json(DATA / (phase + ".json"), dataset)
        results[phase] = {
            "atoms": len(atoms),
            "train_compositions": len(ordered_train),
            "evaluation_compositions": len(ordered_eval),
            "evaluation_roles": dict(Counter(r["role"] for r in ordered_eval)),
            "required_role_counts": dict(Counter(r["required_role"] for r in ordered_eval)),
            "known_direct_shortcuts": sum(r["known_direct_shortcut"] for r in ordered_eval),
            "unseen_operations": sum(not r["operation_seen"] for r in ordered_eval),
            "panels": {name: len(ids) for name, ids in panels.items()},
            "max_input_tokens": max(len(r["encoded"]["input"]) for r in atoms + all_chains),
            "max_answer_tokens_including_eos": max(
                len(r["encoded"]["target"]) for r in atoms + all_chains
            ),
            "data_sha256": sha256(DATA / (phase + ".json")),
        }
    lock = {
        "source_archive_sha256": ARCHIVE_SHA,
        "semantic_review_sha256": sha256(review_path),
        "semantic_packet_sha256": sha256(DATA / "semantic-review-packet.json"),
        "preparation_source_sha256": sha256(Path(__file__)),
        "data_module_sha256": sha256(
            PROJECT / "src/llm_memory_editability/realworld_composition_data.py"
        ),
        "tokenizer_files": {
            p.name: sha256(p) for p in (DATA / "tokenizer").iterdir() if p.is_file()
        },
        "split": (
            "global head/bridge components; fixed hash reserves approximately 20%; "
            "development chain hash holds out approximately 20%"
        ),
        "development_confirmation_entity_overlap": 0,
        "excluded_known_prior_formal_cases": sum(
            r["source_split"] == "dev" and not r["development_group"] and r["known_prior_case"]
            for r in rows
        ),
        "exclusions": dict(exclusions),
        "datasets": results,
        "formal_budget_frozen": False,
        "scope": (
            "Source-consistency sample review by assistant; no independent human or "
            "current-world fact verification. Unsampled cases retain official annotations."
        ),
    }
    write_json(DATA / "data-lock.json", lock)
    write_json(ART / "data-lock.json", lock)
    print(json.dumps(lock, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["extract", "freeze"])
    args = parser.parse_args()
    {"extract": extract, "freeze": freeze}[args.command]()
