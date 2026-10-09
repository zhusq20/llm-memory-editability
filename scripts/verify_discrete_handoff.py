"""CPU/FP64 checks for theory §13.6 (decode-then-encode handoff); no model training.

1. Lemma (28): ||Phi(h) - e_b|| <= eps * max_v ||e_v - e_b||,
   ||dPhi/dh|| <= (2 eps / tau) ||E||^2 ||dN/dh||, and eps <= (V - 1) exp(-margin),
   with and without a LayerNorm readout.
2. Linear key-value example (29)-(30): how often a slot state decodes the bridge
   but the second-hop lookup fails, for orthonormal/bijective versus superposed memories.
3. For (1 - a) h + a * lam * e_b, correctness switches at most once in a.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def layernorm(h):
    centered = h - h.mean()
    return centered / np.sqrt((centered**2).mean() + 1e-12)


def phi(E, h, tau, use_ln):
    z = E @ (layernorm(h) if use_ln else h) / tau
    p = np.exp(z - z.max())
    p /= p.sum()
    return E.T @ p, p, z


def jacobian(fn, h, step=1e-6):
    columns = []
    for j in range(len(h)):
        delta = np.zeros_like(h)
        delta[j] = step
        columns.append((fn(h + delta) - fn(h - delta)) / (2 * step))
    return np.stack(columns, axis=1)


def check_lemma(rng, cases):
    worst = {"deviation": 0.0, "jacobian": 0.0, "epsilon": 0.0}
    for trial in range(cases):
        vocab, width = int(rng.integers(20, 80)), int(rng.integers(8, 24))
        E = rng.normal(size=(vocab, width)) / np.sqrt(width)
        tau, use_ln = rng.uniform(0.2, 2.0), bool(trial % 2)
        h = 0.3 * rng.normal(size=width) + rng.uniform(0.5, 30) * E[rng.integers(vocab)]
        value, p, z = phi(E, h, tau, use_ln)
        b = int(np.argmax(p))
        eps = 1.0 - p[b]
        jac = jacobian(lambda x, E=E, tau=tau, use_ln=use_ln: phi(E, x, tau, use_ln)[0], h)
        norm_jac = np.linalg.norm(jacobian(layernorm, h), 2) if use_ln else 1.0
        pairs = {
            "deviation": (
                np.linalg.norm(value - E[b]),
                eps * np.max(np.linalg.norm(E - E[b], axis=1)),
            ),
            "jacobian": (
                np.linalg.norm(jac, 2),
                2 * eps / tau * np.linalg.norm(E, 2) ** 2 * norm_jac,
            ),
            "epsilon": (eps, (vocab - 1) * np.exp(-(z[b] - np.max(np.delete(z, b))))),
        }
        for name, (lhs, rhs) in pairs.items():
            assert lhs <= rhs + 1e-6 * max(1.0, rhs), (name, lhs, rhs)
            if rhs > 1e-12:
                worst[name] = max(worst[name], float(lhs / rhs))
    return {"cases": cases, "max_lhs_over_rhs_where_rhs_gt_1e-12": worst}


def memory(rng, vocab, width, injective, orthonormal):
    if orthonormal:
        q, _ = np.linalg.qr(rng.normal(size=(width, vocab)))
        E = q.T
    else:
        E = rng.normal(size=(vocab, width))
        E /= np.linalg.norm(E, axis=1, keepdims=True)
    answer = rng.permutation(vocab) if injective else rng.integers(vocab, size=vocab)
    weights = (E @ E.T)[:, answer]  # weights[w, x] = G[w, y(x)], equation (29)
    atomic_ok = np.argmax(weights @ (E @ E.T), axis=0) == answer
    return E, answer, weights, atomic_ok


def check_memory(rng, queries):
    regimes = {
        "orthonormal_bijective": (64, 64, True, True),
        "orthonormal_random_answers": (64, 64, False, True),
        "superposed_bijective": (256, 64, True, False),
        "superposed_random_answers": (256, 64, False, False),
    }
    results = {}
    for name, (vocab, width, injective, orthonormal) in regimes.items():
        E, answer, weights, atomic_ok = memory(rng, vocab, width, injective, orthonormal)
        rows = {}
        for noise in (0.05, 0.10, 0.15):
            counts = {"atomic_correct_queries": 0, "decodable": 0, "decodable_but_unusable": 0}
            for _ in range(queries):
                b = int(rng.integers(vocab))
                if not atomic_ok[b]:
                    continue
                n = noise * rng.normal(size=width)
                n -= (n @ E[b]) * E[b]
                state = E[b] + n
                decodable = int(np.argmax(E @ state)) == b
                usable = int(np.argmax(weights @ (E @ state))) == answer[b]
                counts["atomic_correct_queries"] += 1
                counts["decodable"] += int(decodable)
                counts["decodable_but_unusable"] += int(decodable and not usable)
            rows[f"noise_{noise:.2f}"] = counts
        results[name] = rows
    return results


def check_alpha(rng, queries):
    E, answer, weights, atomic_ok = memory(rng, 256, 64, False, False)
    candidates = np.flatnonzero(atomic_ok)
    alphas = np.linspace(0, 1, 201)
    switches = []
    for _ in range(queries):
        b = int(rng.choice(candidates))
        h = rng.uniform(0.1, 0.6) * E[b] + rng.uniform(0.1, 0.4) * rng.normal(size=64)
        lam = np.linalg.norm(h)
        correct = [
            int(np.argmax(weights @ (E @ ((1 - a) * h + a * lam * E[b])))) == answer[b]
            for a in alphas
        ]
        switches.append(int(np.abs(np.diff(np.asarray(correct, dtype=int))).sum()))
    # alpha=1 is the atomic query itself, which is correct for every sampled bridge.
    return {"queries": queries, "at_most_one_switch": int((np.asarray(switches) <= 1).sum())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--out")
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    result = {
        "seed": args.seed,
        "precision": "numpy float64 on CPU",
        "lemma_28": check_lemma(rng, 300),
        "linear_memory_29_30": check_memory(rng, 400),
        "alpha_threshold": check_alpha(rng, 200),
    }
    text = json.dumps(result, indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
