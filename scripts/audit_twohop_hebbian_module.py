#!/usr/bin/env python3
"""Reload saved E1 memories without fitting; verify predictions and source hashes."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/development-artifacts/twohop-hebbian-v1"
RAW = ROOT / "results/twohop-hebbian-v1"
DATA = ROOT / "data/twohop-hebbian-v1"


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    results = read(OUT / "module-results.json")
    lock = read(OUT / "module-lock.json")
    graph = read(RAW / "graph.json")
    math = read(OUT / "math-checks.json")
    dataset = read(OUT / "dataset-audit.json")
    checks = []

    def check(name, passed):
        checks.append({"name": name, "passed": bool(passed)})
        if not passed:
            raise AssertionError(name)

    for path, expected in [
        (ROOT / "configs/twohop-hebbian-v1.json", lock["config_sha256"]),
        (ROOT / "scripts/run_twohop_hebbian_module.py", lock["script_sha256"]),
        (DATA / "mquake-candidates.json", lock["source_manifest_sha256"]),
        (ROOT / "scripts/check_twohop_hebbian_theory.py", math["source_sha256"]),
        (ROOT / "scripts/prepare_twohop_hebbian_benchmarks.py", dataset["script_sha256"]),
        (DATA / "raw/MQuAKE-CF-3k-v2.json", dataset["sources"]["mquake"]["sha256"]),
        (DATA / "raw/data_ids_april7.zip", dataset["sources"]["two_wiki"]["sha256"]),
    ]:
        check(f"hash:{path.relative_to(ROOT)}", digest(path) == expected)
    for name, expected in results["raw_artifacts"].items():
        check(f"hash:{name}", digest(RAW / name) == expected)
    check("36_math_checks_passed", math["summary"] == {"total": 36, "passed": 36})
    splits = read(OUT / "mquake-split-manifest.json")
    dev = [r for r in splits if r["split"] == "development"]
    evaluation = [r for r in splits if r["split"] == "evaluation"]
    check("mquake_split_counts", (len(dev), len(evaluation)) == (143, 456))
    for field in ["group", "source_entity", "edited_subject"]:
        check(
            f"split_disjoint:{field}", not {r[field] for r in dev} & {r[field] for r in evaluation}
        )
    dev_ids = {r[field] for r in dev for field in ["source_entity", "edited_subject"]}
    eval_ids = {r[field] for r in evaluation for field in ["source_entity", "edited_subject"]}
    check("split_disjoint:source_edit_union", not dev_ids & eval_ids)

    cfg = lock["config"]
    expected = {
        (s, w, i) for s in cfg["seeds"] for w in cfg["hidden_widths"] for i in cfg["interfaces"]
    }
    rows = {(r["seed"], r["hidden_width"], r["interface"]): r for r in results["rows"]}
    check("all_24_rows_once", len(results["rows"]) == len(rows) == 24 and set(rows) == expected)
    eids = {e: i for i, e in enumerate(graph["entities"])}
    rids = {r: i for i, r in enumerate(graph["relations"])}
    facts, chains = graph["facts"], graph["chains"]
    first = np.array([eids[c[0][0]] for c in chains])
    bridge = np.array([eids[c[0][2]] for c in chains])
    target = np.array([eids[c[1][2]] for c in chains])
    r1 = np.array([rids[c[0][1]] for c in chains])
    r2 = np.array([rids[c[1][1]] for c in chains])
    reloaded = 0
    for seed in cfg["seeds"]:
        for width in cfg["hidden_widths"]:
            memory = np.load(RAW / f"seed-{seed}-width-{width}-memory.npz")
            e, r = memory["entity_codes"], memory["relation_codes"]
            up, gate, weight = memory["up"], memory["gate"], memory["weight"]

            def feature(x, relation, r=r, up=up, gate=gate, width=width):
                q = np.concatenate((x, r[relation]), axis=1) / np.sqrt(2)
                return (q @ up.T) * (q @ gate.T) / np.sqrt(width)

            atom_output = (
                feature(e[[eids[f[0]] for f in facts]], [rids[f[1]] for f in facts]) @ weight
            )
            atom_acc = np.mean((atom_output @ e.T).argmax(1) == [eids[f[2]] for f in facts])
            u = feature(e[first], r1) @ weight
            first_ok = (u @ e.T).argmax(1) == bridge
            clean_phi = feature(e[bridge], r2)
            clean_scores = clean_phi @ weight @ e.T
            second_ok = clean_scores.argmax(1) == target
            readouts = weight @ e.T
            # Candidate loop independently avoids the runner's Gram norm implementation.
            radii = np.full(len(chains), np.inf)
            for j in range(len(e)):
                gaps = clean_scores[np.arange(len(chains)), target] - clean_scores[:, j]
                norms = np.linalg.norm(readouts[:, target] - readouts[:, j, None], axis=0)
                ratio = np.divide(gaps, norms, out=np.full_like(gaps, np.inf), where=norms > 0)
                ratio[target == j] = np.inf
                radii = np.minimum(radii, ratio)
            for interface in cfg["interfaces"]:
                row = rows[(seed, width, interface)]
                prefix = f"{seed}:{width}:{interface}"
                saved = np.load(RAW / f"seed-{seed}-width-{width}-{interface}.npz")
                x = (
                    u
                    if interface == "raw"
                    else u / np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-12)
                )
                phi = feature(x, r2)
                predicted = (phi @ weight @ e.T).argmax(1)
                correct = predicted == target
                mastered = first_ok & second_ok
                certified = np.linalg.norm(phi - clean_phi, axis=1) < radii
                for name, value in [
                    ("first", first),
                    ("bridge", bridge),
                    ("target", target),
                    ("predicted", predicted),
                    ("correct", correct),
                    ("mastered", mastered),
                    ("certified", certified),
                ]:
                    check(f"reload:{prefix}:{name}", np.array_equal(saved[name], value))
                for name, value in [
                    ("atomic_accuracy", atom_acc),
                    ("first_hop_accuracy", first_ok.mean()),
                    ("clean_second_hop_accuracy", second_ok.mean()),
                    ("twohop_accuracy", correct.mean()),
                    ("twohop_given_both_atoms", correct[mastered].mean()),
                    ("certified_chains", certified.sum()),
                    ("false_certificates", (certified & ~correct).sum()),
                ]:
                    check(
                        f"metric:{prefix}:{name}", np.isclose(row[name], value, atol=1e-12, rtol=0)
                    )
            reloaded += 1
    output = {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "memories_reloaded_without_refit": reloaded,
        "prediction_arrays_reloaded": len(rows),
        "checks": checks,
        "summary": {"total": len(checks), "passed": sum(c["passed"] for c in checks)},
        "script_sha256": digest(Path(__file__)),
    }
    (OUT / "completion-audit.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({k: v for k, v in output.items() if k != "checks"}, indent=2))


if __name__ == "__main__":
    main()
