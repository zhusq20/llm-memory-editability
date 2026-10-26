"""Compare the vendored generator with the frozen authors' exact methods."""

import ast
import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from llm_memory_editability import dependency_data as data

UPSTREAM = (
    Path(__file__).resolve().parents[1]
    / "docs/development-artifacts/sequence-dependency-gate-v1/upstream"
)


def author_generator(task):
    filename = "listops.py" if task == "listops" else "simple_arithmetic.py"
    classname = "ListOps" if task == "listops" else "SimpleArithmetics"
    tree = ast.parse((UPSTREAM / "dataset" / filename).read_text())
    names = {"Sample", "get_selector_op", "sum_op", "median_op"}
    methods = {"get_one", "format_op", "generate_basic_set"}
    nodes = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            nodes.append(node)
        if isinstance(node, ast.ClassDef) and node.name == classname:
            node.bases, node.decorator_list = [], []
            node.body = [
                member
                for member in node.body
                if (isinstance(member, ast.FunctionDef) and member.name in methods)
                or (
                    isinstance(member, ast.Assign)
                    and any(
                        isinstance(target, ast.Name) and target.id == "OPS"
                        for target in member.targets
                    )
                )
            ]
            nodes.append(node)
    namespace = {"np": np, "dataclasses": dataclasses, "dataclass": dataclasses.dataclass}
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
            str(UPSTREAM / "dataset" / filename),
            "exec",
        ),
        namespace,
    )
    return namespace[classname]()


def test_archived_source_hashes():
    manifest = json.loads((UPSTREAM / "manifest.json").read_text())
    assert manifest["commit"] == data.SOURCE_COMMIT
    for path, expected in manifest["files"].items():
        assert hashlib.sha256((UPSTREAM / path).read_bytes()).hexdigest() == expected


@pytest.mark.parametrize("task", ["listops", "arithmetic"])
@pytest.mark.parametrize("depth", [(2, 6), (7, 7), (8, 9)])
def test_same_rng_same_author_samples(task, depth):
    probability = 0.2 if task == "arithmetic" and depth == (2, 6) else 0.3
    vendored = data._generator(task, depth, probability)
    reference = author_generator(task)
    reference.__dict__.update(vendored.__dict__)
    first, second = np.random.RandomState(9123), np.random.RandomState(9123)
    for _ in range(1000):
        a = vendored.get_one(first) if task == "listops" else vendored.get_one(first, *depth)
        b = reference.get_one(second) if task == "listops" else reference.get_one(second, *depth)
        assert dataclasses.asdict(a) == dataclasses.asdict(b)
    np.testing.assert_array_equal(first.get_state()[1], second.get_state()[1])
