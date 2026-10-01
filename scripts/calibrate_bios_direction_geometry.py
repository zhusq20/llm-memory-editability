"""Exact small-model output Jacobians and strict local feasibility diagnostics."""

import json

import numpy as np
import torch

from llm_memory_editability.bios_data import write_json
from llm_memory_editability.bios_direction import ROOT, Suffix, load_parent, minimal_task


def jacobian(outputs, w, chunk=8):
    flat = outputs.flatten()
    rows = []
    for begin in range(0, len(flat), chunk):
        end = min(begin + chunk, len(flat))
        basis = torch.zeros((end - begin, len(flat)), device=w.device)
        basis[
            torch.arange(end - begin, device=w.device), torch.arange(begin, end, device=w.device)
        ] = 1
        rows.append(
            torch.autograd.grad(flat, w, basis, is_grads_batched=True, retain_graph=True)[0]
            .flatten(1)
            .detach()
        )
    return torch.cat(rows).double()


def run(device):
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    records = []
    for seed in (0, 1):
        model = load_parent(ROOT / f"results/bios-path-minimal-v1/seed-{seed}/parent.pt", device)
        for person in (0, 1, 2):
            task = minimal_task(seed, person, "independent", device)
            ids = task.sets["E"] + task.sets["R"]
            w0 = model.blocks[0].mlp.down.weight
            w = w0[None].clone().requires_grad_()
            suffix = Suffix(model, 0, task.tokens[ids], task.positions[ids])
            logits = suffix(w)[0, :, 0]
            new = task.labels[task.sets["E"], 0]
            old = task.old_labels[task.sets["E"], 0]
            target = (
                logits[torch.arange(2, device=device), new]
                - logits[torch.arange(2, device=device), old]
            )
            a, b = task.metadata["a"], task.metadata["b"]
            pair = logits[:2, a] - logits[:2, b]
            ae = jacobian(torch.cat((target, pair)), w)
            ar = jacobian(logits[2:, :-1] - logits[2:, -1:], w)
            gram = ar @ ar.T
            evals, vecs = torch.linalg.eigh(gram)
            row = dict(
                seed=seed,
                person=person,
                parameters=w0.numel(),
                retention_rows=len(ar),
                target_margin=target.detach().tolist(),
                thresholds=[],
            )
            for cutoff in (1e-8, 1e-10):
                keep = evals > evals[-1] * cutoff
                q = (vecs[:, keep].T @ ar) / evals[keep, None].sqrt()
                projected = ae - (ae @ q.T) @ q
                k = projected[:2] @ projected[:2].T
                desired = torch.ones(2, device=device, dtype=torch.float64)
                delta = projected[:2].T @ torch.linalg.pinv(k, rtol=1e-10) @ desired
                kp = projected[2:] @ projected[2:].T
                cplus = torch.tensor([1.0, 1.0], device=device, dtype=torch.float64) / np.sqrt(2)
                cminus = torch.tensor([1.0, -1.0], device=device, dtype=torch.float64) / np.sqrt(2)
                retention = ar @ delta
                row["thresholds"].append(
                    dict(
                        relative_cutoff=cutoff,
                        rank=int(keep.sum()),
                        target_residual=float((ae[:2] @ delta - desired).norm()),
                        retention_max=float(retention.abs().max()),
                        retention_l2=float(retention.norm()),
                        minimum_norm=float(delta.norm()),
                        relative_parameter_norm=float(delta.norm() / w0.norm()),
                        pair_common=float(cplus @ kp @ cplus),
                        pair_difference=float(cminus @ kp @ cminus),
                        pair_mixing=float(cplus @ kp @ cminus),
                        pair_eigenvalues=torch.linalg.eigvalsh(kp).tolist(),
                        rowspace_orthogonality_error=float(
                            (q @ q.T - torch.eye(len(q), device=device)).norm()
                        ),
                    )
                )
            records.append(row)
            print(json.dumps(dict(seed=seed, person=person, done=True)), flush=True)
    dest = ROOT / "docs/development-artifacts/direction-v1"
    write_json(
        dest / "exact-local-geometry.json",
        dict(
            cases=records,
            scope=(
                "FP32 derivatives, FP64 algebra; two rank thresholds. Strict logit protection "
                "is stronger than correct answers. Discarded modes can cause residuals. "
                "Not a nonlinear feasibility proof."
            ),
        ),
    )


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    run(torch.device(args.device))
