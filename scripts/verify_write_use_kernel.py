"""CPU/FP64 checks for theory §14.4 (Eq. 35–36), §14.6–14.7; no model training.

1. §14.6 Eq. (31): on ReproductionGPT (GPT-2 blocks, LayerNorm, gelu_new, tied
   embedding), the kernel of each MLP output matrix equals
   sum_{e,e'} sum_{p,q} G_{e,p}(u) G_{e',q}(c)^T <phi_{e,p}(u), phi_{e',q}(c)>,
   for an untied 4-layer stack and a 2x2 Loop. For the Loop, the same-execution part
   equals the kernel of the unrolled untied copies and the rest is the cross term of
   Eq. (25).
2. §14.7 Eq. (33): with K(c_i, c_j) = A + S R_ij and K(u, c_i) = A, kernel regression
   moves a held-out form by A / (A + S / n_eff), n_eff = 1^T R^+ 1.
3. §14.7 Eq. (34): linear features, cross-entropy gradient descent from equal logits:
   the held-out/trained logit-change ratio is n A / (n A + S) for n distinct forms at
   every step, and A / (A + S) for n repeats of one form.
4. §14.4 Eq. (35)–(36), partial gradient pooling: on the 2x2 Loop, give each execution
   of one block's MLP output matrix its own copy and take one SGD step with
   q_e = alpha * sum_e' g_e' + T (1 - alpha) g_e on a cross-entropy loss at c. The change
   of u's logits must equal -eta * K_alpha(u, c) v_c up to O(eta^2), with K_alpha built
   from the same-execution and cross-execution terms of Eq. (31).

Usage: python scripts/verify_write_use_kernel.py [--output <json>]
(1) needs torch and transformers; (2)-(3) need numpy only.
"""

import argparse
import json
from pathlib import Path

import numpy as np

SEED = 20261010


# 1. Output-weight kernel factorization ------------------------------------------


def kernel_checks():
    import torch
    from torch.nn import functional as F
    from transformers import GPT2Config

    from llm_memory_editability.grokking_reproduction import ReproductionGPT

    torch.manual_seed(SEED)
    results = []
    for unique, repeats in ((4, 1), (2, 2)):
        config = GPT2Config(
            vocab_size=40,
            n_embd=16,
            n_layer=unique,
            n_head=2,
            n_positions=8,
            activation_function="gelu_new",
            use_cache=False,
        )
        model = ReproductionGPT(config, repeats).double().eval()
        with torch.no_grad():
            for p in model.parameters():
                p.add_(0.05 * torch.randn_like(p))
        blocks = list(model.transformer.h)
        executions = [blocks[e % unique] for e in range(unique * repeats)]
        prompts = {"c": torch.tensor([[7, 21]]), "u": torch.tensor([[3, 25, 21]])}

        def run(
            tokens,
            probes=None,
            record=None,
            copies=None,
            model=model,
            executions=executions,
            config=config,
        ):
            """Explicit forward equal to ReproductionGPT; probes add zero to H_{e+1}.

            copies maps an execution index to its own MLP output matrix (Conv1D layout).
            """
            t = model.transformer
            x = t.wte(tokens) + t.wpe(torch.arange(tokens.shape[1]))
            for e, block in enumerate(executions):
                z = block.ln_1(x)
                q, k, v = block.attn.c_attn(z).split(config.n_embd, dim=2)
                shape = (1, tokens.shape[1], config.n_head, config.n_embd // config.n_head)
                q, k, v = (a.view(shape).transpose(1, 2) for a in (q, k, v))
                y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
                x = x + block.attn.c_proj(y.transpose(1, 2).reshape(1, -1, config.n_embd))
                hidden = block.mlp.act(block.mlp.c_fc(block.ln_2(x)))
                if record is not None:
                    record.append(hidden[0].detach())
                if copies is not None and e in copies:
                    x = x + hidden @ copies[e] + block.mlp.c_proj.bias
                else:
                    x = x + block.mlp.c_proj(hidden)
                if probes is not None:
                    x = x + probes[e]
            return F.linear(t.ln_f(x)[0, -1], t.wte.weight)

        native = {k: model(v)[0, -1] for k, v in prompts.items()}
        forward_error = max(float((run(v) - native[k]).abs().max()) for k, v in prompts.items())
        vocab = config.vocab_size
        jac, phis, back = {}, {}, {}
        for name, tokens in prompts.items():
            params = [b.mlp.c_proj.weight for b in blocks]
            rows = [torch.autograd.grad(run(tokens)[a], params) for a in range(vocab)]
            jac[name] = [torch.stack([r[j].reshape(-1) for r in rows]) for j in range(unique)]
            record, probes = (
                [],
                [
                    torch.zeros(
                        tokens.shape[1], config.n_embd, dtype=torch.double, requires_grad=True
                    )
                    for _ in executions
                ],
            )
            logits = run(tokens, probes, record)
            phis[name] = record
            back[name] = [
                torch.stack(
                    [
                        torch.autograd.grad(logits[a], probes[e], retain_graph=True)[0]
                        for a in range(vocab)
                    ]
                )
                for e in range(len(executions))
            ]  # [vocab, position, width]
        worst, cross_norm, untied_error = 0.0, 0.0, 0.0
        terms = {}
        for j in range(unique):
            autograd = jac["u"][j] @ jac["c"][j].T
            occurrences = [e for e in range(len(executions)) if e % unique == j]
            same, cross = 0.0, 0.0
            for e in occurrences:
                for e2 in occurrences:
                    term = torch.einsum(
                        "apd,bqd,pq->ab",
                        back["u"][e],
                        back["c"][e2],
                        phis["u"][e] @ phis["c"][e2].T,
                    )
                    same, cross = (same + term, cross) if e == e2 else (same, cross + term)
            terms[j] = (same, cross, occurrences)
            worst = max(worst, float((autograd - same - cross).abs().max()))
            cross_norm = max(cross_norm, float(torch.as_tensor(cross).abs().max()))
            if repeats > 1:
                # Unrolled untied copies: one parameter per occurrence, same forward.
                untied = sum(
                    torch.einsum(
                        "apd,bqd,pq->ab", back["u"][e], back["c"][e], phis["u"][e] @ phis["c"][e].T
                    )
                    for e in occurrences
                )
                untied_error = max(untied_error, float((untied - same).abs().max()))
        pooling = (
            partial_pooling_checks(run, prompts, blocks[0], *terms[0], vocab)
            if repeats > 1
            else None
        )
        results.append(
            {
                "partial_pooling_eq35_36": pooling,
                "unique_blocks": unique,
                "repeats": repeats,
                "forward_max_error": forward_error,
                "eq31_max_abs_error": worst,
                "max_abs_kernel_entry": max(
                    float((jac["u"][j] @ jac["c"][j].T).abs().max()) for j in range(unique)
                ),
                "cross_execution_term_max_abs": cross_norm,
                "same_execution_equals_untied_copies_max_error": untied_error,
            }
        )
    return results


def partial_pooling_checks(run, prompts, block, same, cross, occurrences, vocab):
    """One SGD step with partial pooling on per-execution copies of one MLP output matrix."""
    import torch
    from torch.nn import functional as F

    T, target, rows = len(occurrences), 5, []
    with torch.no_grad():
        logits_c = run(prompts["c"])
        v_c = F.softmax(logits_c, -1) - F.one_hot(torch.tensor(target), vocab).double()
        base_u = run(prompts["u"])
    for alpha in (0.0, 0.25, 0.5, 1.0):
        kernel = (alpha + T * (1 - alpha)) * same + alpha * cross
        errors = []
        for eta in (1e-3, 5e-4):
            copies = {
                e: block.mlp.c_proj.weight.detach().clone().requires_grad_(True)
                for e in occurrences
            }
            loss = F.cross_entropy(run(prompts["c"], copies=copies)[None], torch.tensor([target]))
            grads = dict(
                zip(occurrences, torch.autograd.grad(loss, list(copies.values())), strict=True)
            )
            pooled = sum(grads.values())
            with torch.no_grad():
                stepped = {
                    e: copies[e] - eta * (alpha * pooled + T * (1 - alpha) * grads[e])
                    for e in occurrences
                }
                actual = run(prompts["u"], copies=stepped) - base_u
            predicted = -eta * kernel @ v_c
            errors.append(float((actual - predicted).abs().max()))
        rows.append(
            {
                "alpha": alpha,
                "cross_to_same_ratio": alpha / (alpha + T * (1 - alpha)),
                "first_order_residual_eta_1e-3": errors[0],
                "first_order_residual_eta_5e-4": errors[1],
                "residual_ratio_halving_eta": errors[0] / errors[1],
                "max_abs_predicted_change_eta_1e-3": float((1e-3 * kernel @ v_c).abs().max()),
            }
        )
    return rows


# 2. Held-out form under kernel regression ----------------------------------------


def effective_forms(R):
    return float(np.ones(len(R)) @ np.linalg.pinv(R) @ np.ones(len(R)))


def regression_checks(rng, cases=500):
    worst, examples = 0.0, []
    for trial in range(cases):
        n = int(rng.integers(1, 9))
        A, S = rng.uniform(0.01, 3.0), rng.uniform(0.01, 3.0)
        kind = ("independent", "repeated", "random")[trial % 3]
        if kind == "independent":
            R = np.eye(n)
        elif kind == "repeated":
            R = np.ones((n, n))
        else:
            X = rng.normal(size=(n, n + 2))
            X /= np.linalg.norm(X, axis=1, keepdims=True)
            R = X @ X.T
        K = A + S * R
        moved = A * np.ones(n) @ np.linalg.pinv(K) @ np.ones(n)
        q = effective_forms(R)
        closed = A / (A + S / q)
        worst = max(worst, abs(moved - closed))
        if trial < 6:
            examples.append(
                {
                    "forms": n,
                    "R": kind,
                    "A": A,
                    "S": S,
                    "n_eff": q,
                    "kernel_regression": moved,
                    "closed_form": closed,
                }
            )
    return {"cases": cases, "max_abs_error": worst, "examples": examples}


# 3. Cross-entropy gradient descent with linear features --------------------------


def gradient_flow(rng, n, repeated, A=0.4, S=1.0, width=64, vocab=12, steps=400, lr=0.05):
    basis = np.linalg.qr(rng.normal(size=(width, width)))[0]
    shared = np.sqrt(A) * basis[:, 0]
    specific = [np.sqrt(S) * basis[:, 1 + (0 if repeated else i)] for i in range(n)]
    trained = np.stack([shared + s for s in specific])
    heldout = shared + np.sqrt(S) * basis[:, 1 + n + 1]
    W, target, ratios = np.zeros((vocab, width)), 3, []
    for _ in range(steps):
        logits = trained @ W.T
        p = np.exp(logits - logits.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        p[:, target] -= 1.0
        W -= lr * p.T @ trained
        change_c, change_u = trained[0] @ W.T, heldout @ W.T
        ratios.append(float((change_u @ change_c) / (change_c @ change_c)))
    expected = A / (A + S) if repeated else n * A / (n * A + S)
    final = trained @ W.T
    return {
        "forms": n,
        "repeated": repeated,
        "expected_ratio": expected,
        "ratio_range": [min(ratios), max(ratios)],
        "trained_target_probability": float(np.exp(final[0, target]) / np.exp(final[0]).sum()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rng = np.random.default_rng(SEED)
    result = {"seed": SEED, "regression": regression_checks(rng)}
    result["gradient_flow"] = [
        gradient_flow(rng, n, repeated) for n in (1, 2, 4, 8) for repeated in (False, True)
    ]
    try:
        result["output_weight_kernel"] = kernel_checks()
    except ImportError as error:
        result["output_weight_kernel"] = f"skipped: {error}"
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
