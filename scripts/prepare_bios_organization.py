"""Materialize and audit matched A/B/C document corpora from shared bioS worlds."""

import argparse
import json
from pathlib import Path

import numpy as np

from llm_memory_editability.bios_data import (
    N_BASE,
    N_QUERIES,
    RELATIONS,
    array_hash,
    load_world,
    make_world,
    paired_edit,
    write_json,
)
from llm_memory_editability.bios_organization import (
    CONDITIONS,
    N_PRESENTATIONS,
    audit_organizations,
    english_fact,
    make_documents,
    presentation_ids,
    render_documents,
)


def write_jsonl(path, records):
    with Path(path).open("w") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def prepare(args):
    output = Path(args.output)
    audits = []
    for seed in args.worlds:
        existing = Path(args.world_root) / f"world-{seed}"
        world = (
            load_world(existing)
            if (existing / "world.npz").exists()
            else (make_world(seed, args.source))
        )
        if world.seed != seed:
            raise ValueError("World directory and metadata seed disagree")
        destination = output / f"world-{seed}"
        destination.mkdir(parents=True, exist_ok=True)
        documents = {
            condition: make_documents(world, condition, args.organization_seed)
            for condition in CONDITIONS
        }
        audit = audit_organizations(world, documents)
        audit["organization_seed"] = args.organization_seed
        audit["origin_world_directory"] = str(existing)
        world.save(destination)

        # Master presentations are assigned text before condition-specific grouping.
        master_ids = presentation_ids(world, documents["A"])
        master_facts = np.empty(N_PRESENTATIONS, dtype=np.int64)
        master_facts[master_ids.ravel()] = documents["A"].ravel()
        variants = np.arange(N_PRESENTATIONS) % 3
        sentences = [
            english_fact(world, int(fact), int(variant))
            for fact, variant in zip(master_facts, variants, strict=True)
        ]
        write_jsonl(
            destination / "truth.jsonl",
            (
                {
                    "fact_id": fact,
                    "relation": RELATIONS[int(world.relation[fact])],
                    "person_id": int(world.person[fact]),
                    "company_origin_id": int(world.company[fact]),
                    "prompt_tokens": world.prompts[fact, : world.lengths[fact]].tolist(),
                    "answer_token": int(world.answers[fact]),
                    "canonical_sentence": english_fact(world, fact),
                }
                for fact in range(N_BASE)
            ),
        )
        write_jsonl(
            destination / "presentations.jsonl",
            (
                {
                    "presentation_id": pid,
                    "fact_id": int(master_facts[pid]),
                    "template_id": int(variants[pid]),
                    "alias_policy": "canonical names only",
                    "sentence": sentences[pid],
                }
                for pid in range(N_PRESENTATIONS)
            ),
        )
        np.savez_compressed(
            destination / "presentations.npz",
            fact_ids=master_facts,
            template_ids=variants,
            presentation_ids=np.arange(N_PRESENTATIONS),
        )
        for condition, facts in documents.items():
            pids = presentation_ids(world, facts)
            if not np.array_equal(master_facts[pids], facts):
                raise ValueError("Presentation identity no longer matches its fact")
            rendered = render_documents(world, facts)
            np.savez_compressed(
                destination / f"documents-{condition}.npz",
                **rendered,
                presentation_ids=pids,
            )
            write_jsonl(
                destination / f"documents-{condition}.jsonl",
                (
                    {
                        "document_id": doc,
                        "condition": condition,
                        "slots": [
                            {
                                "slot": slot,
                                "fact_id": int(facts[doc, slot]),
                                "presentation_id": int(pids[doc, slot]),
                            }
                            for slot in range(7)
                        ],
                        "text": "\n".join(sentences[pid] for pid in pids[doc]),
                    }
                    for doc in range(len(facts))
                ),
            )

        # Shared independent queries contain no support document. Derived queries
        # are trained equally in all conditions by the separate training schedule.
        np.savez_compressed(
            destination / "evaluation.npz",
            query_ids=np.arange(N_QUERIES),
            prompts=world.prompts,
            lengths=world.lengths,
            answers=world.answers,
            relations=world.relation,
        )
        edits = []
        for support in args.supports:
            pair = paired_edit(world, support, k=1)
            np.savez_compressed(
                destination / f"edit-k1-support{support}.npz",
                **{key: value for key, value in pair.items() if isinstance(value, np.ndarray)},
            )
            if (
                np.intersect1d(pair["D"], pair["replay"]).size
                or np.intersect1d(pair["heldout"], pair["replay"]).size
            ):
                raise ValueError("Edit replay contaminates held-out evaluation")
            edits.append(
                {
                    "support": support,
                    "k": 1,
                    "E": len(pair["E"]),
                    "D": len(pair["D"]),
                    "coherent_sha256": array_hash(pair["coherent"]),
                    "exception_sha256": array_hash(pair["exception"]),
                    "replay_excludes_D_and_heldout": True,
                }
            )
        audit["paired_edits"] = edits
        audit["evaluation"] = {
            "isolated_base_queries": N_BASE,
            "isolated_derived_queries": N_QUERIES - N_BASE,
            "derived_query_policy": "same old-world derived QA trained in every condition",
            "paraphrase_and_alias_evaluation": "not applicable to symbolic training",
            "english_corpora": "supplemental readable views only; not trained or evaluated",
        }
        write_json(destination / "organization-audit.json", audit)
        audits.append(audit)
        print(
            json.dumps(
                {
                    "world": seed,
                    "status": "organization_contract_passed",
                    "output": str(destination),
                }
            ),
            flush=True,
        )
    write_json(output / "audit.json", {"schema": "bios-organization-v1", "worlds": audits})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="data/bios-source")
    parser.add_argument("--world-root", default="data/bios-work-v1")
    parser.add_argument("--output", default="data/bios-organization-v1")
    parser.add_argument("--worlds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--organization-seed", type=int, default=0)
    parser.add_argument("--supports", type=int, nargs="+", default=[0, 1])
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
