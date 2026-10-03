"""Paired synthetic control of first-hop fact addressability."""

import argparse
import collections
import json
import subprocess
import sys
from pathlib import Path

import diagnose_realworld_entities as experiment
import numpy as np

common = experiment.common
ROOT = common.PROJECT / "results/realworld-loop-addressability-v1"
ART = common.PROJECT / "docs/development-artifacts/realworld-loop-addressability-v1"
DATA = common.PROJECT / "data/realworld-loop-addressability-v1/prepared.json"
SEED = 917001
NODES = (0, 8000, 32000, 128000)


class Codec:
    eos_token_id = 0

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(f"node{i}" for i in ids if i != 0 or not skip_special_tokens)


def spec_for(arch):
    spec = common.spec_for(arch)
    spec.update(
        initialization=SEED,
        sampling_seed=SEED,
        dropout_seed=SEED,
        learning_rate=1e-3,
        weight_decay=0.1,
        steps=NODES[-1],
        microbatch_size=512,
    )
    return spec


def record(identifier, prefix, tail, **metadata):
    return {
        "id": identifier,
        "question": " ".join(map(str, prefix)),
        "answer": f"node{tail}",
        "aliases": [],
        "unambiguous_answers": [f"node{tail}"],
        "canonical_answer_globally_unambiguous": True,
        "encoded": {"prefix": prefix, "target": [tail, 0], "input": prefix + [tail]},
        **metadata,
    }


def prepare():
    assert not DATA.exists()
    rng = np.random.default_rng(SEED)
    n, relations, facts = 128, 16, 1024
    keys = rng.choice(n * relations, facts, replace=False)
    edges = [
        (int(k // relations) + 1, int(k % relations) + 1153, int(rng.integers(1, n + 1)))
        for k in keys
    ]
    first_id = set(map(int, rng.choice(facts, 768, replace=False)))
    second_id = set(map(int, rng.choice(facts, 768, replace=False)))
    outgoing = collections.defaultdict(list)
    for j, (head, _, _) in enumerate(edges):
        outgoing[head].append(j)
    pairs = [(i, j) for i, (_, _, tail) in enumerate(edges) for j in outgoing[tail]]
    candidates = [(i, j) for i, j in pairs if i in first_id and j in second_id]
    order = rng.permutation(len(candidates))
    training = {candidates[int(k)] for k in order[: int(len(candidates) * 0.8)]}
    exposed = {("first", i) for i, _ in training} | {("second", j) for _, j in training}
    payload = {}
    for condition in ["shared", "unique"]:
        atoms, train, test = [], [], []
        for i, (head, rel, tail) in enumerate(edges):
            alias = head + 128 if condition == "shared" else 129 + i
            atoms.append(record(f"first-{i}", [alias, rel], tail, edge=[alias, rel, tail]))
            atoms.append(record(f"second-{i}", [head, rel], tail, edge=[head, rel, tail]))
        atom_map = {r["id"]: r for r in atoms}
        for i, j in pairs:
            a, b = atom_map[f"first-{i}"], atom_map[f"second-{j}"]
            role = ("I" if ("first", i) in exposed else "O") + (
                "I" if ("second", j) in exposed else "O"
            )
            row = record(
                f"chain-{i}-{j}",
                [a["edge"][0], a["edge"][1], b["edge"][1]],
                b["edge"][2],
                atom_ids=[a["id"], b["id"]],
                edges=[a["edge"], b["edge"]],
                role=role,
                required_role=role,
            )
            (train if (i, j) in training else test).append(row)
        panels = {
            "atomic": [r["id"] for r in atoms[:128]],
            "train_composition": [r["id"] for r in train[:128]],
        }
        for role in ["II", "IO", "OI", "OO"]:
            panels["test_" + role.lower()] = [r["id"] for r in test if r["role"] == role][:64]
        payload[condition] = {
            "atoms": atoms,
            "train_compositions": train,
            "evaluation_compositions": test,
            "panels": panels,
        }
        assert len({tuple(r["encoded"]["prefix"]) for r in train + test}) == len(pairs)
        assert {r["id"] for r in train}.isdisjoint(r["id"] for r in test)
        assert all(r["role"] == "II" for r in train)
    for split in ["atoms", "train_compositions", "evaluation_compositions"]:
        for a, b in zip(payload["shared"][split], payload["unique"][split], strict=True):
            assert a["id"] == b["id"] and a["encoded"]["target"] == b["encoded"]["target"]
            assert a["encoded"]["prefix"][1:] == b["encoded"]["prefix"][1:]
            assert len(a["encoded"]["prefix"]) == len(b["encoded"]["prefix"])
            assert a.get("atom_ids") == b.get("atom_ids") and a.get("role") == b.get("role")
    DATA.parent.mkdir(parents=True, exist_ok=True)
    DATA.write_text(json.dumps(payload, separators=(",", ":")))
    model = {**common.BASE["model"], "vocab_size": 1169, "hidden_size": 128, "attention_heads": 4}
    manifest = {
        "created_utc": common.utc(),
        "seed": SEED,
        "nodes": NODES,
        "model": model,
        "prepared_sha256": common.sha256(DATA),
        "atomic": len(atoms),
        "train": len(train),
        "test_roles": dict(collections.Counter(r["role"] for r in test)),
        "source_sha256": {
            str(p): common.sha256(p)
            for p in [Path(__file__), ART / "design.md", Path(experiment.__file__)]
        },
    }
    common.write_json(ART / "manifest.json", manifest)
    print(json.dumps(manifest), flush=True)


def load(condition):
    manifest = json.loads((ART / "manifest.json").read_text())
    assert common.sha256(DATA) == manifest["prepared_sha256"]
    name = {"natural": "shared", "entity": "unique"}.get(condition, condition)
    return json.loads(DATA.read_text())[name], manifest


def pipeline(arch, condition, gpu):
    out = ROOT / "development" / f"{arch}-{condition}"
    out.mkdir(parents=True, exist_ok=True)
    for mode in ["worker", "audit"]:
        with (out / f"{mode}.log").open("a") as log:
            subprocess.run(
                [
                    sys.executable,
                    "-u",
                    str(Path(__file__).resolve()),
                    mode,
                    "--arch",
                    arch,
                    "--condition",
                    condition,
                    "--gpu",
                    str(gpu),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=["prepare", "engineering", "worker", "audit", "pipeline"]
    )
    parser.add_argument("--arch", choices=["standard8", "loop4x2"])
    parser.add_argument("--condition", choices=["shared", "unique"])
    parser.add_argument("--gpu", type=int)
    args = parser.parse_args()
    experiment.SEED, experiment.NODES = SEED, NODES
    experiment.ROOT, experiment.ART = ROOT, ART
    experiment.load, experiment.NodeCodec, experiment.spec_for = load, Codec, spec_for
    if args.command == "prepare":
        prepare()
    elif args.command == "engineering":
        experiment.engineering(args.gpu)
    else:
        {"worker": experiment.worker, "audit": experiment.audit, "pipeline": pipeline}[
            args.command
        ](args.arch, args.condition, args.gpu)
