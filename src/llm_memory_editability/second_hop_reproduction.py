"""Ye et al. §2.4 / Appendix D.1: exclude selected facts from second-hop roles."""

from __future__ import annotations

import ast
import os
import random
from pathlib import Path

from .paper_reproduction import TOKEN, read_json, sha, write_json
from .paper_reproduction import audit as audit
from .paper_reproduction import run as run


def validate_roles(id_rows, ood_rows, restricted, training, evaluation):
    atomic = {}
    id_keys = set()
    for name, rows in (("id", id_rows), ("ood", ood_rows)):
        for row in rows:
            head, relation, tail, eos = TOKEN.findall(row["target_text"])
            key = (head, relation)
            assert eos == "</a>" and key not in atomic
            assert row["input_text"] == head + relation
            atomic[key] = tail
            if name == "id":
                id_keys.add(key)
    assert restricted <= id_keys
    train_queries = {row["input_text"] for row in training}
    assert len(train_queries) == len(training)
    first_role = set()
    for name, rows in [("Train-II", training), *evaluation.items()]:
        if name.endswith("Triples"):
            continue
        seen = set()
        for row in rows:
            head, r1, r2 = TOKEN.findall(row["input_text"])
            bridge = atomic[(head, r1)]
            second = (bridge, r2)
            assert (head, r1) in id_keys and second in id_keys
            assert row["target_text"] == head + r1 + r2 + atomic[second] + "</a>"
            assert (second in restricted) == (name == "Test-II-SR")
            assert (row["input_text"] in train_queries) == (name == "Train-II")
            assert row["input_text"] not in seen
            seen.add(row["input_text"])
            if name == "Train-II" and (head, r1) in restricted:
                first_role.add((head, r1))
    return {
        "passed": True,
        "atomic_truths": len(atomic),
        "restricted_facts": len(restricted),
        "restricted_facts_used_as_first_hop": len(first_role),
        "restricted_second_hop_training_queries": 0,
    }


def prepare(notebook, destination, *, seed=42, entities=2000, relations=200, degree=20):
    import numpy as np
    from tqdm.auto import tqdm

    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    assert os.environ.get("PYTHONHASHSEED") == "0"
    random.seed(seed)
    np.random.seed(seed)
    namespace = {"np": np, "tqdm": tqdm}
    cells = read_json(notebook)["cells"]
    for index in (1, 11):
        parsed = ast.parse("".join(cells[index]["source"]))
        definitions = ast.Module(
            body=[node for node in parsed.body if isinstance(node, ast.FunctionDef)],
            type_ignores=[],
        )
        exec(compile(definitions, f"{notebook}:cell-{index}", "exec"), namespace)
    original_split = namespace["split"]
    splits = []

    def capture_split(*args, **kwargs):
        result = original_split(*args, **kwargs)
        splits.append(result)
        return result

    namespace["split"] = capture_split
    et, rt, id_rows, ood_rows, ii, test_ii, test_sr = namespace["build_second_hop_ablation"](
        entities, relations, out_degree=degree
    )
    assert len(splits) == 2
    restricted = {(head, relation) for head, relation, _ in splits[1][0]}
    choose = namespace["choose"]
    evaluation = {
        "ID Triples": choose(id_rows, 3000),
        "OOD Triples": choose(ood_rows, 3000),
        "Test-II": choose(test_ii, 3000),
        "Test-II-SR": choose(test_sr, 3000),
    }
    candidates = len(ii)
    ii = choose(ii, round(7.2 * len(id_rows)))
    evaluation["Train-II"] = choose(ii, 3000)
    audit_result = validate_roles(id_rows, ood_rows, restricted, ii, evaluation)
    probes = [dict(row, type=name) for name, rows in evaluation.items() for row in rows]
    vocab = et + rt + ["<mask>", "<sep>", "<a>", "</a>", "<q>", "</q>"]
    training = {"only_ii": ii, "full_atomic": id_rows + ood_rows + ii}
    for arm, rows in training.items():
        for filename, values in {
            "train.json": rows,
            "valid.json": evaluation["Test-II"],
            "test.json": probes,
            "vocab.json": vocab,
        }.items():
            write_json(destination / arm / filename, values)
    write_json(destination / "atomic-truth.json", id_rows + ood_rows)
    write_json(destination / "restricted-facts.json", sorted(restricted))
    audit_result.update(
        seed=seed,
        python_hash_seed=0,
        notebook_sha256=sha(notebook),
        original_cells=[1, 11, 12, 13],
        candidate_train_ii=candidates,
        selected_train_ii=len(ii),
        training_counts={arm: len(rows) for arm, rows in training.items()},
        evaluation_counts={name: len(rows) for name, rows in evaluation.items()},
        same_ii_and_probes=True,
        ordinary_ii_holdout_probability=0.005,
        integer_count_fix="7.2 * ID count is an integer count, as in paper Table 2",
        validation_fix=(
            "Notebook cell 13 refers to undefined stale test_2hop_oo; "
            "use Test-II, no early stopping/config selection"
        ),
        files={
            str(path.relative_to(destination)): sha(path) for path in destination.rglob("*.json")
        },
    )
    write_json(destination / "data-audit.json", audit_result)
    return audit_result
