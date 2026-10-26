"""Causal answer-only adaptation of the NDR authors' dependency datasets.

Expression generation is vendored from the MIT-licensed NDR repository. The
adaptation adds deterministic split seeds, explicit split disjointness, an
independent expression interpreter, and a common causal language-model format.
Structural annotations are evaluation metadata, never model input or targets.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ._ndr_dependency_generators import ListOps, SimpleArithmetics

SOURCE_COMMIT = "da20530dfb4336deddfbe5e79d62e72d1dc2580e"
SOURCE_URL = "https://github.com/RobertCsordas/ndr"
PANELS = ("train_sample", "valid_iid", "valid_depth", "test_iid", "test_depth")
FIELDS = (
    "tokens",
    "positions",
    "labels",
    "length",
    "depth",
    "tree_depth",
    "branch_imbalance",
    "computed_branch_imbalance",
    "branch_nodes",
    "join_count",
    "operator_count",
    "author_length",
    "expression_length",
    "author_depth",
    "ids",
)
BLOCK_SIZE = 1000


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stable_seed(master, task, split, block=0):
    value = f"ndr-dependency-v1:{master}:{task}:{split}:{block}".encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:4], "little")


def panel_names():
    return PANELS


@dataclass(frozen=True)
class Analysis:
    answer: int
    tree_depth: int
    depth: int
    author_length: int
    branch_imbalance: int
    branch_nodes: int
    operator_count: int
    computed_branch_imbalance: int = 0

    @property
    def join_count(self):
        return self.branch_nodes


def _combine(operation, children, task):
    values = [child.answer for child in children]
    depths = [child.tree_depth for child in children]
    if operation == "MIN":
        chosen = [values.index(min(values))]
        answer = min(values)
    elif operation == "MAX":
        chosen = [values.index(max(values))]
        answer = max(values)
    elif operation == "MED":
        order = np.argsort(values)
        middle = len(order) // 2
        chosen = (
            [int(order[middle])] if len(order) % 2 else [int(order[middle - 1]), int(order[middle])]
        )
        answer = sum(values[index] for index in chosen) // len(chosen)
    elif operation in ("SM", "+"):
        answer, chosen = sum(values) % 10, list(range(len(values)))
    elif operation == "*":
        answer, chosen = math.prod(values) % 10, list(range(len(values)))
    else:
        raise ValueError(f"Unknown operation {operation}")
    tree_depth = max(depths) + 1
    computed_depths = [child.tree_depth for child in children if child.operator_count]
    computed_imbalance = (
        max(computed_depths) - min(computed_depths) if len(computed_depths) >= 2 else 0
    )
    return Analysis(
        answer=answer,
        tree_depth=tree_depth,
        depth=max(children[index].depth for index in chosen) + 1,
        author_length=sum(child.author_length for child in children) + 3,
        branch_imbalance=max(
            max(depths) - min(depths), *(child.branch_imbalance for child in children)
        ),
        branch_nodes=sum(child.branch_nodes for child in children)
        + int(sum(child.operator_count > 0 for child in children) >= 2),
        operator_count=1 + sum(child.operator_count for child in children),
        computed_branch_imbalance=max(
            computed_imbalance, *(child.computed_branch_imbalance for child in children)
        ),
    )


def analyze_expression(expression, task):
    """Parse independently of the generator and recompute truth and structure."""
    units = expression.split() if task == "listops" else list(expression)
    cursor = 0

    def parse():
        nonlocal cursor
        if cursor >= len(units):
            raise ValueError("Truncated expression")
        token = units[cursor]
        cursor += 1
        if token in tuple(str(number) for number in range(10)):
            return Analysis(int(token), 1, 1, 1, 0, 0, 0)
        if task == "listops" and token in ("[MIN", "[MAX", "[MED", "[SM"):
            children = []
            while cursor < len(units) and units[cursor] != "]":
                children.append(parse())
            if cursor == len(units) or not 2 <= len(children) <= 5:
                raise ValueError("Invalid ListOps argument list")
            cursor += 1
            return _combine(token[1:], children, task)
        if task == "arithmetic" and token == "(":
            left = parse()
            if cursor >= len(units) or units[cursor] not in ("+", "*"):
                raise ValueError("Missing arithmetic operator")
            operation = units[cursor]
            cursor += 1
            right = parse()
            if cursor >= len(units) or units[cursor] != ")":
                raise ValueError("Missing arithmetic closing parenthesis")
            cursor += 1
            return _combine(operation, [left, right], task)
        raise ValueError(f"Unexpected expression token {token!r}")

    result = parse()
    if cursor != len(units) or result.operator_count == 0:
        raise ValueError("Expression must contain exactly one complete operation tree")
    return result


def _generator(task, depth, probability):
    if task == "listops":
        generator = ListOps()
        generator.depth, generator.p_op, generator.max_args = depth, probability, 5
    elif task == "arithmetic":
        generator = SimpleArithmetics()
        generator.rpn, generator.maxval, generator.p_subop = False, 10, probability
    else:
        raise ValueError(f"Unsupported dependency task: {task}")
    return generator


def _candidate_block(arguments):
    task, depth, probability, seed = arguments
    generator = _generator(task, depth, probability)
    rng = np.random.RandomState(seed)
    candidates = []
    for _ in range(BLOCK_SIZE):
        sample = generator.get_one(rng) if task == "listops" else generator.get_one(rng, *depth)
        length = sample.length if task == "listops" else len(sample.input)
        if length <= 50:
            bucket = sample.max_dependency_depth if task == "listops" else sample.depth
            candidates.append(
                (sample.input, int(sample.output), int(sample.depth), int(bucket), int(length))
            )
    return candidates


def _candidates(task, depth, probability, master, split, workers):
    # Fixed block size, seed and ordering make output independent of worker count.
    block = 0
    pool = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        while True:
            arguments = [
                (task, depth, probability, stable_seed(master, task, split, block + index))
                for index in range(max(workers, 1))
            ]
            batches = (
                pool.map(_candidate_block, arguments) if pool else map(_candidate_block, arguments)
            )
            for batch in batches:
                yield from batch
            block += max(workers, 1)
    finally:
        if pool:
            pool.shutdown(wait=True, cancel_futures=True)


def _quotas(task, count, depth):
    if task == "listops":
        # Exact author rule: all bins share ceil(N / n_bins); stop at N.
        return {
            value: math.ceil(count / (depth[1] - depth[0] + 1))
            for value in range(depth[0], depth[1] + 1)
        }
    remaining, result = count, {}
    for value in range(depth[0], depth[1] + 1):
        result[value] = math.ceil(remaining / (depth[1] - value + 1))
        remaining -= result[value]
    return result


def _draw(task, count, depth, probability, master, split, forbidden, workers, progress=None):
    remaining = _quotas(task, count, depth)
    iterator = _candidates(task, depth, probability, master, split, workers)
    accepted, proposals = [], 0
    try:
        for row in iterator:
            proposals += 1
            expression, _, _, bucket, _ = row
            if remaining.get(bucket, 0) <= 0 or expression in forbidden:
                continue
            forbidden.add(expression)
            remaining[bucket] -= 1
            accepted.append(row)
            if progress and (len(accepted) % 10000 == 0 or len(accepted) == count):
                progress(
                    {
                        "phase": split,
                        "accepted": len(accepted),
                        "target": count,
                        "length_limited_proposals": proposals,
                        "remaining_by_depth": remaining,
                    }
                )
            if len(accepted) == count:
                break
    finally:
        iterator.close()
    return accepted, proposals


def vocabulary(task):
    symbols = ["<PAD>", "<BOS>", "<EOS>", "<SEP>"] + [str(number) for number in range(10)]
    symbols += ["[MIN", "[MAX", "[MED", "[SM", "]"] if task == "listops" else ["(", ")", "+", "*"]
    return symbols


def pack_rows(rows, metadata):
    """Pack BOS expression SEP answer; supervise only answer and following EOS."""
    task, symbols = metadata["task"], metadata["vocabulary"]
    lookup = {symbol: index for index, symbol in enumerate(symbols)}
    count, width = len(rows), metadata["sequence_length"]
    packed = {
        name: np.empty(count, dtype=np.int16)
        for name in FIELDS
        if name not in ("tokens", "positions", "labels", "ids")
    }
    packed["tokens"] = np.full((count, width), metadata["pad_token"], dtype=np.int16)
    packed["positions"] = np.empty((count, 2), dtype=np.int16)
    packed["labels"] = np.empty((count, 2), dtype=np.int16)
    packed["ids"] = np.empty(count, dtype="S64")
    for index, (expression, answer, author_depth, author_bucket, author_length) in enumerate(rows):
        analysis = analyze_expression(expression, task)
        units = expression.split() if task == "listops" else list(expression)
        if analysis.answer != answer or analysis.author_length != author_length:
            raise ValueError("Independent interpreter disagrees with generator truth/length")
        if task == "listops" and (
            analysis.depth != author_bucket or analysis.tree_depth != author_depth
        ):
            raise ValueError("Independent interpreter disagrees with ListOps dependency depth")
        if (
            task == "arithmetic"
            and author_depth != analysis.tree_depth
            and not (author_depth == 1 and analysis.tree_depth == 2)
        ):
            raise ValueError("Independent interpreter disagrees with arithmetic tree depth")
        encoded = [
            metadata["bos_token"],
            *(lookup[unit] for unit in units),
            metadata["separator_token"],
            lookup[str(answer)],
        ]
        if len(encoded) > width:
            raise ValueError("Expression exceeds the frozen sequence length")
        packed["tokens"][index, : len(encoded)] = encoded
        packed["positions"][index] = len(encoded) - 2, len(encoded) - 1
        packed["labels"][index] = lookup[str(answer)], metadata["end_marker"]
        packed["length"][index] = len(encoded)
        packed["expression_length"][index] = len(units)
        packed["author_depth"][index] = author_depth
        for name in (
            "depth",
            "tree_depth",
            "branch_imbalance",
            "computed_branch_imbalance",
            "branch_nodes",
            "join_count",
            "operator_count",
            "author_length",
        ):
            packed[name][index] = getattr(analysis, name)
        packed["ids"][index] = hashlib.sha256(expression.encode()).hexdigest().encode()
    return packed


def decode_expression(tokens, position, metadata):
    symbols = metadata["vocabulary"]
    units = [symbols[int(token)] for token in tokens[1 : int(position)]]
    return " ".join(units) if metadata["task"] == "listops" else "".join(units)


def _histogram(values):
    return {str(int(key)): int(count) for key, count in sorted(Counter(values.tolist()).items())}


def audit_data(world, panels, metadata):
    if set(panels) != set(PANELS):
        raise ValueError("Incorrect dependency evaluation panel matrix")
    known, diagnostics = set(), {}
    training_positions = set(world["positions"][:, 0].tolist())
    training_lengths = set(world["expression_length"].tolist())
    token_position_support = np.zeros(
        (metadata["vocab_size"], metadata["sequence_length"]), dtype=bool
    )
    for tokens, positions in zip(world["tokens"], world["positions"], strict=True):
        for position in range(1, int(positions[0])):
            token_position_support[int(tokens[position]), position] = True
    for name, rows in [
        ("train", world),
        *((key, panels[key]) for key in PANELS if key != "train_sample"),
    ]:
        if set(rows) != set(FIELDS):
            raise ValueError("Unexpected packed fields")
        ids = rows["ids"].tolist()
        expected_count = (
            metadata["training_examples"] if name == "train" else metadata["evaluation_samples"]
        )
        lower, upper = (
            (7, 7) if name == "valid_depth" else (8, 9) if name == "test_depth" else (2, 6)
        )
        if metadata["task"] == "arithmetic" and name in ("valid_iid", "test_iid"):
            lower = 3
        if (
            len(ids) != expected_count
            or np.any(rows["depth"] < lower)
            or np.any(rows["depth"] > upper)
        ):
            raise ValueError("Dataset size or registered depth range mismatch")
        if len(set(ids)) != len(ids) or known.intersection(ids):
            raise ValueError("Repeated expression or training/evaluation leakage")
        known.update(ids)
        for index, row_id in enumerate(ids):
            length = int(rows["length"][index])
            positions = rows["positions"][index]
            tokens, labels = rows["tokens"][index], rows["labels"][index]
            if tuple(positions) != (length - 2, length - 1) or not np.all(
                tokens[length:] == metadata["pad_token"]
            ):
                raise ValueError("Incorrect supervision positions or right padding")
            if (
                tokens[0] != metadata["bos_token"]
                or tokens[positions[0]] != metadata["separator_token"]
            ):
                raise ValueError("Missing causal prompt delimiter")
            if tokens[positions[1]] != labels[0] or labels[1] != metadata["end_marker"]:
                raise ValueError("Incorrect final answer/EOS targets")
            expression = decode_expression(tokens, positions[0], metadata)
            parsed = analyze_expression(expression, metadata["task"])
            if (
                hashlib.sha256(expression.encode()).hexdigest().encode() != row_id
                or int(metadata["vocabulary"][int(labels[0])]) != parsed.answer
            ):
                raise ValueError("Packed sample truth or identity mismatch")
            for field in (
                "depth",
                "tree_depth",
                "branch_imbalance",
                "computed_branch_imbalance",
                "branch_nodes",
                "join_count",
                "operator_count",
                "author_length",
            ):
                if int(rows[field][index]) != getattr(parsed, field):
                    raise ValueError(f"Incorrect structural annotation {field}")
        unseen_occurrences = unseen_examples = expression_occurrences = 0
        for tokens, positions in zip(rows["tokens"], rows["positions"], strict=True):
            unseen = sum(
                not token_position_support[int(tokens[position]), position]
                for position in range(1, int(positions[0]))
            )
            unseen_occurrences += unseen
            unseen_examples += int(unseen > 0)
            expression_occurrences += int(positions[0]) - 1
        prediction_positions = set(rows["positions"][:, 0].tolist())
        diagnostics[name] = {
            "n": len(ids),
            "depth": _histogram(rows["depth"]),
            "tree_depth": _histogram(rows["tree_depth"]),
            "branch_imbalance": _histogram(rows["branch_imbalance"]),
            "computed_branch_imbalance": _histogram(rows["computed_branch_imbalance"]),
            "syntactic_branch_join_examples": int((rows["join_count"] > 0).sum()),
            "answer_counts": _histogram(rows["labels"][:, 0] - 4),
            "expression_length": _histogram(rows["expression_length"]),
            "answer_prediction_position": _histogram(rows["positions"][:, 0]),
            "answer_positions_unseen_in_training": sorted(
                prediction_positions - training_positions
            ),
            "answer_position_covered_examples": int(
                np.isin(rows["positions"][:, 0], list(training_positions)).sum()
            ),
            "expression_lengths_shared_with_training": sorted(
                set(rows["expression_length"].tolist()) & training_lengths
            ),
            "expression_length_covered_examples": int(
                np.isin(rows["expression_length"], list(training_lengths)).sum()
            ),
            "expression_token_position_unseen_occurrences": unseen_occurrences,
            "expression_token_position_unseen_examples": unseen_examples,
            "expression_token_occurrences": expression_occurrences,
        }
    training = {row_id: index for index, row_id in enumerate(world["ids"].tolist())}
    sample = panels["train_sample"]
    if len(set(sample["ids"].tolist())) != len(sample["ids"]):
        raise ValueError("Repeated training diagnostic sample")
    for index, row_id in enumerate(sample["ids"].tolist()):
        if row_id not in training or any(
            not np.array_equal(sample[field][index], world[field][training[row_id]])
            for field in FIELDS
        ):
            raise ValueError("Training diagnostic is not an exact in-sample subset")
    if metadata["task"] == "arithmetic":
        if int((world["tree_depth"] == 2).sum()) != 200:
            raise ValueError("Arithmetic training must contain all 200 atomic operations")
        if any(np.any(panels[name]["tree_depth"] == 2) for name in ("valid_iid", "test_iid")):
            raise ValueError("IID evaluation includes already trained atomic expressions")
    return {
        "passed": True,
        "independent_interpreter_examples": sum(value["n"] for value in diagnostics.values()),
        "training_evaluation_overlap": 0,
        "evaluation_evaluation_overlap": 0,
        "train_sample_is_in_sample": True,
        "join_annotation_note": (
            "syntactic joins do not guarantee every child changes the final answer"
        ),
        "position_coverage_note": (
            "equal maximum length does not imply equal length distributions "
            "or trained token-position coverage"
        ),
        "panels": diagnostics,
    }


def prepare(config, destination, workers=1):
    spec = config.get("data", config)
    task = spec["task"]
    if task not in ("listops", "arithmetic"):
        raise ValueError("task must be listops or arithmetic")
    train_count = int(spec.get("train_samples", 1_000_000 if task == "listops" else 100_000))
    evaluation_count, master = (
        int(spec.get("evaluation_samples", 1000)),
        int(spec.get("seed", 20261018)),
    )
    if min(train_count, evaluation_count, workers) < 1:
        raise ValueError("Dataset sizes and worker count must be positive")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)

    def progress(status):
        (destination / "progress.json").write_text(json.dumps(status, indent=2) + "\n")

    metadata = {
        "schema": "ndr-causal-dependency-v1",
        "task": task,
        "source": SOURCE_URL,
        "source_commit": SOURCE_COMMIT,
        "source_license": "MIT",
        "vendored_generator_sha256": sha256(
            Path(__file__).with_name("_ndr_dependency_generators.py")
        ),
        "seed": master,
        "train_samples": train_count,
        "evaluation_samples": evaluation_count,
        "vocabulary": vocabulary(task),
        "pad_token": 0,
        "bos_token": 1,
        "end_marker": 2,
        "separator_token": 3,
        "vocab_size": len(vocabulary(task)),
        "sequence_length": 53,
        "length_definition": (
            "BOS + expression tokens + SEP + teacher-forced answer; excludes EOS and right padding"
        ),
        "primary_score": "answer_accuracy",
        "training_objective": "ordinary answer plus EOS cross-entropy; no intermediate labels",
        "structural_metadata_used_as_model_input": False,
        "depth_definition": (
            "author effective dependency depth for ListOps; true tree depth for arithmetic, "
            "counting leaves as one"
        ),
        "branch_imbalance_definition": (
            "maximum sibling tree-depth difference over all operation nodes"
        ),
        "computed_branch_imbalance_definition": (
            "maximum depth difference among non-leaf siblings, "
            "at nodes with at least two such children"
        ),
        "branch_nodes_definition": "number of nodes with at least two non-leaf children",
        "join_count_definition": (
            "same as branch_nodes; syntactic rather than answer-sensitivity count"
        ),
        "adaptations": [
            "causal BOS expression SEP answer format replaces bidirectional encoder classification",
            "stable split and block seeds replace host-dependent author seeding",
            "explicit disjointness across training and all held-out pools",
            "separate IID validation and test pools",
            "arithmetic IID pools exclude the 200 atomic expressions already exhaustively trained",
        ],
        "generator_block_size": BLOCK_SIZE,
        "numpy_version": np.__version__,
    }
    forbidden, proposals = set(), {}
    depth = (2, 6) if task == "listops" else (3, 6)
    rows, proposals["train"] = _draw(
        task,
        train_count,
        depth,
        0.3 if task == "listops" else 0.2,
        master,
        "train",
        forbidden,
        workers,
        progress,
    )
    if task == "arithmetic":
        atoms = _generator(task, (2, 6), 0.2).generate_basic_set()
        atom_rows = [
            (sample.input, int(sample.output), int(sample.depth), 2, len(sample.input))
            for sample in atoms
        ]
        forbidden.update(sample.input for sample in atoms)
        rows = atom_rows + rows
    world = pack_rows(rows, metadata)
    del rows
    rng = np.random.RandomState(stable_seed(master, task, "train_sample"))
    indices = rng.choice(
        len(world["ids"]),
        min(int(spec.get("training_panel_size", 1000)), len(world["ids"])),
        replace=False,
    )
    panels = {"train_sample": {name: values[indices].copy() for name, values in world.items()}}
    for split in ("valid_iid", "valid_depth", "test_iid", "test_depth"):
        split_depth = (
            depth if split.endswith("iid") else (7, 7) if split == "valid_depth" else (8, 9)
        )
        probability = 0.2 if task == "arithmetic" and split.endswith("iid") else 0.3
        rows, proposals[split] = _draw(
            task,
            evaluation_count,
            split_depth,
            probability,
            master,
            split,
            forbidden,
            workers,
            progress,
        )
        panels[split] = pack_rows(rows, metadata)
    metadata.update(
        training_examples=len(world["ids"]),
        panels={name: len(rows["ids"]) for name, rows in panels.items()},
        accepted_length_limited_proposals=proposals,
        source_basic_arithmetic_depth_note=(
            "author generate_basic_set labels atoms depth=1; author_depth preserves it, "
            "structural tree_depth/depth correctly count leaves as 1 and atoms as 2"
        ),
    )
    progress({"phase": "independent_audit", "training_examples": len(world["ids"])})
    audit = audit_data(world, panels, metadata)
    np.savez_compressed(destination / "world.npz", **world)
    np.savez_compressed(
        destination / "panels.npz",
        **{
            f"{name}/{field}": values
            for name, rows in panels.items()
            for field, values in rows.items()
        },
    )
    metadata.update(
        world_sha256=sha256(destination / "world.npz"),
        panels_sha256=sha256(destination / "panels.npz"),
    )
    (destination / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    metadata["audit_sha256"] = sha256(destination / "audit.json")
    (destination / "complete.json").write_text(json.dumps(metadata, indent=2) + "\n")
    progress({"phase": "complete", "training_examples": len(world["ids"]), "passed": True})
    return metadata


def load(destination):
    root = Path(destination)
    metadata = json.loads((root / "complete.json").read_text())
    for name in ("world.npz", "panels.npz", "audit.json"):
        if sha256(root / name) != metadata[f"{Path(name).stem}_sha256"]:
            raise ValueError(f"Frozen dependency data hash mismatch: {name}")
    with np.load(root / "world.npz", allow_pickle=False) as data:
        world = dict(data)
    with np.load(root / "panels.npz", allow_pickle=False) as data:
        panels = {name: {field: data[f"{name}/{field}"] for field in FIELDS} for name in PANELS}
    return world, panels, metadata


def make_table(world, metadata, device):
    import torch

    padding = {
        "tokens": np.full((1, metadata["sequence_length"]), metadata["pad_token"], np.int16),
        "positions": np.array([[0, 1]], np.int16),
        "labels": np.array([[-100, -100]], np.int16),
    }
    return tuple(
        torch.as_tensor(
            np.concatenate((world[name], padding[name])), dtype=torch.long, device=device
        )
        for name in ("tokens", "positions", "labels")
    )


def exposure_summary(counts, world, metadata, steps, batch_size):
    counts = np.asarray(counts, dtype=np.int64)
    if len(counts) != len(world["ids"]):
        raise AssertionError("Exposure vector disagrees with training support")
    epochs, remaining = divmod(steps, math.ceil(len(counts) / batch_size))
    if (
        int(counts.sum()) != epochs * len(counts) + remaining * batch_size
        or counts.max() - counts.min() > 1
    ):
        raise AssertionError("Exposure does not match the merged shuffled sample stream")
    actual = int(counts.sum())
    return {
        "examples": actual,
        "epochs": actual / len(counts),
        "effective_input_tokens": int(np.dot(counts, world["length"].astype(np.int64))),
        "supervised_tokens": 2 * actual,
        "executed_input_tokens": steps * batch_size * metadata["sequence_length"],
        "masked_batch_slots": steps * batch_size - actual,
        "by_depth": {
            str(int(depth)): int(counts[world["depth"] == depth].sum())
            for depth in np.unique(world["depth"])
        },
    }
