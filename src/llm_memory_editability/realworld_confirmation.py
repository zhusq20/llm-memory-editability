"""Data-only confirmation isolation and independent endpoint recounting."""

from __future__ import annotations

import ast
import copy
import json
import re
import statistics
import string
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .realworld_composition_data import normalize_answer, order, sha256, write_json


def head_bridge_names(data, canonical, aliases):
    names = defaultdict(set)
    for row in data["train_compositions"] + data["evaluation_compositions"]:
        for index in (0, 2):
            identifier = row["edges"][0][index]
            labels = [canonical[identifier], row["surfaces"][0][index]]
            for label in labels + aliases.get(identifier, []):
                normalized = normalize_answer(label)
                if normalized:
                    names[identifier].add(normalized)
    return names


def answer_owners(canonical, aliases):
    owners = defaultdict(set)
    for identifier in set(canonical) | set(aliases):
        for name in [canonical.get(identifier, ""), *aliases.get(identifier, [])]:
            normalized = normalize_answer(name)
            if normalized:
                owners[normalized].add(identifier)
    return owners


def isolate_confirmation(candidate, development, canonical, aliases):
    """Exclude every chain touching a development head/bridge name, then rebuild."""
    dev_names = set().union(*head_bridge_names(development, canonical, aliases).values())
    formal_names = head_bridge_names(candidate, canonical, aliases)
    excluded_entities = {e for e, names in formal_names.items() if names & dev_names}
    excluded = {}
    parts = {}
    for key in ("train_compositions", "evaluation_compositions"):
        excluded[key] = [
            r["id"]
            for r in candidate[key]
            if {r["edges"][0][0], r["edges"][0][2]} & excluded_entities
        ]
        excluded_ids = set(excluded[key])
        parts[key] = [copy.deepcopy(r) for r in candidate[key] if r["id"] not in excluded_ids]
    all_chains = parts["train_compositions"] + parts["evaluation_compositions"]
    used = {a for r in all_chains for a in r["atom_ids"]}
    atoms = [copy.deepcopy(r) for r in candidate["atoms"] if r["id"] in used]
    first, second = ({r["atom_ids"][i] for r in parts["train_compositions"]} for i in (0, 1))
    seen = first | second
    direct = {(r["edge"][0], r["edge"][2]) for r in atoms}
    pairs = {(r["edges"][0][0], r["edges"][1][2]) for r in parts["train_compositions"]}
    operations = {tuple(e[1] for e in r["edges"]) for r in parts["train_compositions"]}
    owners = answer_owners(canonical, aliases)
    for row in atoms + all_chains:
        identifier = row["edge"][2] if "edge" in row else row["edges"][1][2]
        row["canonical_answer_globally_unambiguous"] = owners[normalize_answer(row["answer"])] == {
            identifier
        }
    for row in all_chains:
        a, b = row["atom_ids"]
        row["role"] = ("I" if a in seen else "O") + ("I" if b in seen else "O")
        row["required_role"] = ("I" if a in first else "O") + ("I" if b in second else "O")
        pair = row["edges"][0][0], row["edges"][1][2]
        row["known_direct_shortcut"] = pair in direct
        row["head_tail_seen_in_combination_training"] = pair in pairs
        row["operation_seen"] = tuple(e[1] for e in row["edges"]) in operations
    panels = {
        "test_" + role.lower(): [
            r["id"]
            for r in sorted(
                [r for r in parts["evaluation_compositions"] if r["role"] == role],
                key=lambda r: order(r["id"], "panel:"),
            )[:64]
        ]
        for role in ("II", "IO", "OI", "OO")
    }
    panels["train_composition"] = [
        r["id"]
        for r in sorted(parts["train_compositions"], key=lambda r: order(r["id"], "panel:"))[:128]
    ]
    selected = {i for ids in panels.values() for i in ids}
    required = {a for r in all_chains if r["id"] in selected for a in r["atom_ids"]}
    panels["atomic"] = sorted(
        required
        | {r["id"] for r in sorted(atoms, key=lambda r: order(r["id"], "atom-panel:"))[:128]}
    )
    data = {"atoms": atoms, **parts, "panels": panels}
    report = {
        "rule": (
            "Exclude chains whose head/bridge canonical, annotated surface or official alias "
            "name overlaps development head/bridge names after author normalization; "
            "retain shared tails."
        ),
        "excluded_entities": sorted(excluded_entities),
        "excluded_case_ids": excluded,
        "excluded_counts": {key: len(ids) for key, ids in excluded.items()},
        "colliding_names": sorted(dev_names & set().union(*formal_names.values())),
        "model_results_used": False,
    }
    return data, report


def audit_confirmation(data, development, canonical, aliases):
    """Recount from the selected records rather than selection bookkeeping."""
    atoms = {r["id"]: r for r in data["atoms"]}
    train, held = data["train_compositions"], data["evaluation_compositions"]
    rows = train + held
    dev_names = head_bridge_names(development, canonical, aliases)
    formal_names = head_bridge_names(data, canonical, aliases)
    assert not set(dev_names) & set(formal_names)
    assert not set().union(*dev_names.values()) & set().union(*formal_names.values())
    assert len(atoms) == len(data["atoms"])
    assert len({r["id"] for r in rows}) == len(rows)

    def chain(row):
        return tuple(tuple(e) for e in row["edges"])

    assert not {chain(r) for r in train} & {chain(r) for r in held}
    held_questions = {normalize_answer(r["question"]) for r in held}
    assert not {normalize_answer(r["question"]) for r in train} & held_questions
    assert not {normalize_answer(r["question"]) for r in atoms.values()} & held_questions
    assert not {normalize_answer(r["question"]) for r in atoms.values()} & {
        normalize_answer(r["question"]) for r in development["atoms"]
    }
    seen = {a for r in train for a in r["atom_ids"]}
    first, second = ({r["atom_ids"][i] for r in train} for i in (0, 1))
    owners = answer_owners(canonical, aliases)
    for row in rows:
        a, b = row["atom_ids"]
        assert [atoms[a]["edge"], atoms[b]["edge"]] == row["edges"]
        assert row["edges"][0][2] == row["edges"][1][0]
        assert row["role"] == ("I" if a in seen else "O") + ("I" if b in seen else "O")
        assert row["required_role"] == ("I" if a in first else "O") + ("I" if b in second else "O")
        assert not row["structural_errors"]
    assert not any(r["known_prior_case"] for r in held)
    for pool in (list(atoms.values()), train, held):
        prompt_answers = defaultdict(set)
        for row in pool:
            prompt_answers[normalize_answer(row["question"])].add(normalize_answer(row["answer"]))
            identifier = row["edge"][2] if "edge" in row else row["edges"][1][2]
            assert row["canonical_answer_globally_unambiguous"] == (
                owners[normalize_answer(row["answer"])] == {identifier}
            )
            e = row["encoded"]
            assert e["target"][-1] == 50256 and len(e["target"]) <= 64
            assert e["input"] == (e["prefix"] + e["target"])[:-1]
            assert len(e["prefix"]) + len(e["target"]) <= 256
        assert all(len(answers) == 1 for answers in prompt_answers.values())
    panel_ids = {i for name, ids in data["panels"].items() if name != "atomic" for i in ids}
    assert {a for r in rows if r["id"] in panel_ids for a in r["atom_ids"]} <= set(
        data["panels"]["atomic"]
    )
    return {
        "passed": True,
        "atoms": len(atoms),
        "train_compositions": len(train),
        "evaluation_compositions": len(held),
        "evaluation_roles": dict(Counter(r["role"] for r in held)),
        "required_role_counts": dict(Counter(r["required_role"] for r in held)),
        "canonical_unambiguous_evaluation_count": sum(
            r["canonical_answer_globally_unambiguous"] for r in held
        ),
        "panels": {name: len(ids) for name, ids in data["panels"].items()},
        "head_bridge_id_overlap": 0,
        "head_bridge_normalized_name_overlap": 0,
        "necessary_atomic_coverage": 1.0,
        "known_direct_shortcuts": sum(r["known_direct_shortcut"] for r in held),
        "unseen_operations": sum(not r["operation_seen"] for r in held),
        "max_input_tokens": max(len(r["encoded"]["input"]) for r in list(atoms.values()) + rows),
        "atomic_relations": dict(Counter(r["edge"][1] for r in atoms.values())),
        "train_relation_pairs": dict(Counter(" -> ".join(e[1] for e in r["edges"]) for r in train)),
    }


def finalize_confirmation(config):
    """Independently score raw outputs using the pinned author's scorer functions."""
    root, art = Path(config["results_root"]), Path(config["artifact_root"])
    data = json.loads(Path(config["data_file"]).read_text())
    scorer = Path(config["official_scorer"])
    scope = dict(re=re, string=string, Counter=Counter)
    tree = ast.parse(scorer.read_text())
    functions = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name in {"normalize_answer", "exact_match_score"}
    ]
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(scorer), "exec"), scope)

    def score(prediction, row):
        return float(
            any(
                scope["exact_match_score"](prediction, g)
                for g in [row["answer"], *row.get("aliases", [])]
            )
        )

    held = data["evaluation_compositions"]
    summary, scores, atom_scores, exposures, metadata = {}, {}, {}, {}, {}
    for spec in config["runs"]:
        name = spec["name"]
        out = root / config["phase"] / name
        complete = json.loads((out / "complete.json").read_text())
        assert complete["step"] == spec["steps"] and complete["independently_reloaded"]
        raw = json.loads((out / "endpoint-predictions.json").read_text())
        metrics = json.loads((out / "endpoint.json").read_text())
        for group, records in [
            ("atomic", data["atoms"]),
            ("train_composition", data["train_compositions"]),
            ("test_all", held),
        ]:
            by_id = {p["id"]: p for p in raw[group]}
            assert len(by_id) == len(records) == len(raw[group])
            assert set(by_id) == {r["id"] for r in records}
            recomputed = {r["id"]: score(by_id[r["id"]]["prediction"], r) for r in records}
            assert all(recomputed[i] == by_id[i]["alias_em"] for i in recomputed)
            assert abs(sum(recomputed.values()) / len(records) - metrics[group]["alias_em"]) < 1e-12
            if group == "test_all":
                scores[name] = recomputed
                for role in ("II", "IO", "OI", "OO"):
                    selected = [r for r in held if r["role"] == role]
                    assert (
                        abs(
                            sum(recomputed[r["id"]] for r in selected) / len(selected)
                            - metrics["test_" + role.lower()]["alias_em"]
                        )
                        < 1e-12
                    )
            if group == "atomic":
                atom_scores[name] = recomputed
        summary[name] = metrics
        exposures[name] = {
            step: np.load(out / f"exposure-{step:07d}.npz")["counts"]
            for step in config["evaluation_nodes"]
        }
        metadata[name] = json.loads((out / "run.json").read_text())
        assert complete["checkpoint_sha256"] == sha256(out / "latest.pt")
    comparisons = []
    for seed in sorted({s["initialization"] for s in config["runs"]}):
        specs = {s["architecture"]: s for s in config["runs"] if s["initialization"] == seed}
        assert set(specs) == {"standard8", "loop4x2", "standard4"}
        names = [s["name"] for s in specs.values()]
        assert len({metadata[n]["common_four_block_initial_sha256"] for n in names}) == 1
        for step in config["evaluation_nodes"]:
            assert all(
                np.array_equal(exposures[names[0]][step], exposures[n][step]) for n in names[1:]
            )
        for baseline in ("standard8", "standard4"):
            loop, base = specs["loop4x2"]["name"], specs[baseline]["name"]
            for role in ("all", "OO"):
                selected = held if role == "all" else [r for r in held if r["role"] == role]
                common = [
                    r
                    for r in selected
                    if all(atom_scores[n][a] for n in (loop, base) for a in r["atom_ids"])
                ]
                comparisons.append(
                    {
                        "initialization": seed,
                        "baseline": baseline,
                        "pool": role,
                        "n": len(selected),
                        "paired_difference": sum(
                            scores[loop][r["id"]] - scores[base][r["id"]] for r in selected
                        )
                        / len(selected),
                        "both_models_necessary_atoms_correct_n": len(common),
                        "both_models_necessary_atoms_correct_coverage": len(common) / len(selected),
                        "conditional_paired_difference": sum(
                            scores[loop][r["id"]] - scores[base][r["id"]] for r in common
                        )
                        / len(common)
                        if common
                        else None,
                    }
                )
    means = {}
    for architecture in ("standard8", "loop4x2", "standard4"):
        names = [s["name"] for s in config["runs"] if s["architecture"] == architecture]
        means[architecture] = {}
        for group in (
            "atomic",
            "train_composition",
            "test_all",
            "test_ii",
            "test_io",
            "test_oi",
            "test_oo",
            "autonomous",
        ):
            values = [summary[n][group]["alias_em"] for n in names]
            means[architecture][group] = {
                "mean": statistics.mean(values),
                "sample_sd": statistics.stdev(values),
                "per_initialization": values,
            }
    result = {
        "passed": True,
        "runs": len(config["runs"]),
        "scientific_updates": sum(s["steps"] for s in config["runs"]),
        "independent_dataset_splits": 1,
        "paired_initializations": 3,
        "paired_exposures_equal_at_all_nodes": True,
        "architecture_scores": means,
        "paired_comparisons": comparisons,
        "source_data_sha256": sha256(config["data_file"]),
        "official_scorer_sha256": sha256(scorer),
    }
    write_json(art / "confirmation-summary.json", summary)
    write_json(art / "confirmation-recount-audit.json", result)
    return result
