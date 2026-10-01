"""Prospectively frozen context-statistic prediction of common-state gradients.

The predictor contracts supervised target/context co-occurrences with fixed
embeddings and first-layer value weights. It uses neither real gradients nor
contextual activations, and has no fitted scale. It is a new residual-path
approximation, not a theorem for the project's Transformer or AdamW updates.
"""

import argparse
import hashlib
import itertools
import json
import platform
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from types import MethodType

import numpy as np
import torch
from torch.nn import functional as F

from .bios_cross import (
    CONDITIONS,
    documents,
    epoch_documents,
    make_cross_world,
    qa_schedule,
    render,
)
from .bios_cross_train import tensor_queries
from .bios_data import array_hash, write_json
from .bios_model import CausalLM, ModelConfig

TARGET = "blocks.0.attention.proj.weight"
CONTRASTS = (("company", "neither"), ("project", "neither"), ("company", "project"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stamp():
    return datetime.now(timezone.utc).isoformat()


def allowed_mask(length, isolated, device=None):
    pos = torch.arange(length, device=device)
    mask = pos[None, :] <= pos[:, None]
    if isolated:
        mask &= pos[None, :] // 6 == pos[:, None] // 6
    return mask


def _isolated_forward(self, x):
    batch, length, width = x.shape
    q, k, v = self.qkv(x).view(batch, length, 3, self.heads, width // self.heads).unbind(2)
    output = F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        attn_mask=allowed_mask(length, True, x.device),
    )
    return self.proj(output.transpose(1, 2).reshape(batch, length, width))


@contextmanager
def attention_mode(model, isolated):
    originals = []
    if isolated:
        for block in model.blocks:
            attention = block.attention
            originals.append((attention, attention.forward))
            attention.forward = MethodType(_isolated_forward, attention)
    try:
        yield
    finally:
        for attention, forward in originals:
            attention.forward = forward


def ln_vjp(x, upstream, layer):
    """Exact input VJP of LayerNorm, without reading a real model gradient."""
    centered = x - x.mean(-1, keepdim=True)
    inverse = torch.rsqrt(centered.square().mean(-1, keepdim=True) + layer.eps)
    normalized = centered * inverse
    affine = upstream * layer.weight
    return inverse * (
        affine
        - affine.mean(-1, keepdim=True)
        - normalized * (affine * normalized).mean(-1, keepdim=True)
    )


@torch.no_grad()
def statistic_predictors(model, tokens, positions, labels, shuffled, isolated):
    """Gradients of a zero-residual baseline plus one infinitesimal OV branch.

    x_j = E[token_j] + P[j] (or E[token_j] for the token-only control).
    m_t = average_{j allowed at t} (Wv LN1(x_j) + bv).
    epsilon_t = J_LNfinal(x_t)^T E^T (softmax(E LNfinal(x_t)) - onehot(y_t)).
    Ghat = mean_t epsilon_t m_t^T.  This is a contraction of position/mask
    weighted token-target counts; all other residual branches are omitted.
    The answer and EOS means returned here are separately normalized.
    """
    width = model.config.width
    result = {}
    for position_aware in (True, False):
        x = model.token(tokens)
        if position_aware:
            x = x + model.position(torch.arange(tokens.shape[1], device=tokens.device))
        value = F.linear(
            model.blocks[0].ln1(x),
            model.blocks[0].attention.qkv.weight[2 * width :],
            model.blocks[0].attention.qkv.bias[2 * width :],
        )
        mask = allowed_mask(tokens.shape[1], isolated, tokens.device).to(x.dtype)
        mask /= mask.sum(-1, keepdim=True)
        mean_value = (mask @ value)[:, positions]
        query = x[:, positions]
        probability = F.linear(model.ln_final(query), model.token.weight).softmax(-1)
        expected = probability @ model.token.weight
        variants = (
            (("position", labels), ("shuffled", shuffled))
            if position_aware
            else (("token", labels),)
        )
        for name, targets in variants:
            epsilon = ln_vjp(query, expected - model.token(targets), model.ln_final)
            for offset, kind in enumerate(("answer", "eos")):
                left, right = epsilon[:, offset::2], mean_value[:, offset::2]
                result[f"{name}/{kind}"] = torch.einsum("btd,bte->de", left, right) / (
                    left.shape[0] * left.shape[1]
                )
    return result


def vector_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=np.float64).ravel()
    predicted = np.asarray(predicted, dtype=np.float64).ravel()
    an, pn = np.linalg.norm(actual), np.linalg.norm(predicted)
    return {
        "actual_norm": float(an),
        "predicted_norm": float(pn),
        "cosine": float(np.dot(actual, predicted) / (an * pn)) if an and pn else None,
        "relative_error": float(np.linalg.norm(actual - predicted) / an) if an else None,
        "norm_ratio": float(pn / an) if an else None,
    }


def module_name(name):
    if name.startswith("blocks."):
        parts = name.split(".")
        if parts[2] == "attention" and parts[3] == "qkv":
            return ".".join(parts[:4])
        return ".".join(parts[:3])
    return name.split(".")[0]


def module_differences(first, second):
    sums = {}
    for name in first:
        # The concatenated qkv parameter is additionally split by role.
        pieces = [(module_name(name), first[name], second[name])]
        if ".attention.qkv." in name:
            pieces = [
                (name.split(".qkv.")[0] + "." + role, a, b)
                for role, a, b in zip(
                    ("query", "key", "value"),
                    np.split(first[name], 3),
                    np.split(second[name], 3),
                    strict=True,
                )
            ]
        for group, left, right in pieces:
            values = sums.setdefault(group, np.zeros(3, dtype=np.float64))
            left, right = left.astype(np.float64), right.astype(np.float64)
            values += (np.sum((left - right) ** 2), np.sum(left**2), np.sum(right**2))
    return {
        group: {
            "difference_norm": float(np.sqrt(value[0])),
            "relative_to_mean_arm_norm": float(
                np.sqrt(value[0]) / ((np.sqrt(value[1]) + np.sqrt(value[2])) / 2)
            )
            if value[1] + value[2]
            else None,
        }
        for group, value in sums.items()
    }


def source_files(root, config):
    names = [
        "bios_context_gradient.py",
        "bios_cross.py",
        "bios_cross_train.py",
        "bios_data.py",
        "bios_model.py",
        "bios_train.py",
        "bios_organization_train.py",
        "bios_organization.py",
    ]
    paths = [root / "src/llm_memory_editability" / name for name in names]
    paths += [
        root / "scripts/run_bios_context_gradient.py",
        root / "tests/test_bios_context_gradient.py",
    ]
    paths += [root / "configs/bios-context-gradient-v1.json"]
    for world in config["worlds"]:
        paths += [
            root / config["source_world_root"] / f"world-{world}" / name
            for name in ("world.npz", "metadata.json", "audit.json")
        ]
    return paths


def freeze(root, config):
    out = root / config["artifacts"]
    lockpath = out / "preregistration-lock.json"
    if lockpath.exists():
        raise FileExistsError("An existing prospective lock must not be replaced")
    protocol = (root / "docs/experimental-protocol.md").read_text().split("## 16. ", 1)[1]
    out.mkdir(parents=True, exist_ok=True)
    (out / "preregistration.md").write_text("## 16. " + protocol)
    sources = {}
    for path in source_files(root, config):
        relative = path.relative_to(root)
        sources[str(relative)] = sha(path)
        # Data stay in their original location, hash-bound by the lock.
        if relative.parts[0] != "data":
            copy = out / "frozen-files" / relative
            copy.parent.mkdir(parents=True, exist_ok=True)
            copy.write_bytes(path.read_bytes())
    write_json(
        lockpath,
        {
            "created_at": stamp(),
            "config": config,
            "sources": sources,
            "protocol_sha256": sha(out / "preregistration.md"),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "python": platform.python_version(),
        },
    )
    return {"lock": str(lockpath), "sha256": sha(lockpath)}


def verify_lock(root, config):
    lockpath = root / config["artifacts"] / "preregistration-lock.json"
    lock = json.loads(lockpath.read_text())
    if lock["config"] != config:
        raise ValueError("Configuration differs from prospective lock")
    for path, expected in lock["sources"].items():
        if sha(root / path) != expected:
            raise ValueError("Frozen source changed: " + path)
    if sha(root / config["artifacts"] / "preregistration.md") != lock["protocol_sha256"]:
        raise ValueError("Prospective protocol changed")
    return sha(lockpath)


def context_audit(world, config):
    """Exact exposure/bigram audit plus a centered city-by-person Phi subblock."""
    people = [i for i, label in enumerate(world.token_labels) if label.startswith("person:")]
    cities = np.unique(world.city_tokens)
    pi, ci = np.full(world.vocab_size, -1), np.full(world.vocab_size, -1)
    pi[people], ci[cities] = np.arange(len(people)), np.arange(len(cities))
    histograms, contexts, receipts = {}, {}, {}
    for condition in CONDITIONS:
        docs = documents(world, condition)
        receipts[condition] = {"documents_sha256": array_hash(docs)}
        counts = np.bincount(docs.ravel(), minlength=world.n_base)
        receipts[condition]["exposure_sha256"] = array_hash(counts)
        for isolated in (False, True):
            joint = np.zeros((len(cities), len(people)))
            marginal = np.zeros(len(people))
            bigrams = np.zeros((2, world.vocab_size), dtype=np.int64)
            for epoch in range(config["rotations"]):
                batch = render(world, epoch_documents(docs, world.seed, epoch))
                for col, pos in enumerate(batch["positions"][0]):
                    targets = batch["labels"][:, col]
                    # All answer predictors use ANS; EOS predictors use answer tokens.
                    values = targets if col % 2 == 0 else batch["tokens"][:, pos]
                    bigrams[col % 2] += np.bincount(values, minlength=world.vocab_size)
                    left = int(pos) // 6 * 6 if isolated else 0
                    weight = 1 / (int(pos) - left + 1)
                    rows = ci[targets]
                    for prefix in range(left, int(pos) + 1):
                        columns = pi[batch["tokens"][:, prefix]]
                        valid = columns >= 0
                        np.add.at(marginal, columns[valid], weight)
                        valid &= rows >= 0
                        np.add.at(joint, (rows[valid], columns[valid]), weight)
            contexts[condition, isolated] = (joint - marginal[None] / world.vocab_size) / (
                config["rotations"] * len(docs) * 20
            )
            histograms[condition] = bigrams
    result = {"receipts": receipts, "comparisons": []}
    for a, b in CONTRASTS:
        equal = np.array_equal(histograms[a], histograms[b])
        if not equal or receipts[a]["exposure_sha256"] != receipts[b]["exposure_sha256"]:
            raise ValueError("Organization exposure or supervised bigrams differ")
        for isolated in (False, True):
            first, second = contexts[a, isolated], contexts[b, isolated]
            difference = np.max(np.abs(first - second))
            if isolated and difference > 1e-14:
                raise ValueError("Isolated context statistics differ")
            result["comparisons"].append(
                {
                    "first": a,
                    "second": b,
                    "isolated": isolated,
                    "supervised_bigram_equal": equal,
                    "context_max_abs_difference": float(difference),
                    "context_relative_difference": float(
                        np.linalg.norm(first - second) / np.linalg.norm(second)
                    ),
                }
            )
    result["bigram_sha256"] = array_hash(histograms["neither"])
    result["qa_ids_sha256"] = array_hash(world.train_ids)
    return result


def numpy_gradients(accumulators):
    return {name: value.cpu().numpy().astype(np.float32) for name, value in accumulators.items()}


def measure_arm(model, world, condition, isolated, config, device, out):
    docs = documents(world, condition)
    named = dict(model.named_parameters())
    params = tuple(named.values())
    accumulators = {
        kind: {name: torch.zeros_like(value, dtype=torch.float64) for name, value in named.items()}
        for kind in ("answer", "eos")
    }
    predictions = {
        f"{predictor}/{kind}": torch.zeros(
            (model.config.width, model.config.width), dtype=torch.float64, device=device
        )
        for predictor in config["predictors"]
        for kind in ("answer", "eos")
    }
    losses = {"answer": 0.0, "eos": 0.0}
    total_documents = len(docs) * config["rotations"]
    started = time.perf_counter()
    model.eval()
    with attention_mode(model, isolated):
        for epoch in range(config["rotations"]):
            batch = render(world, epoch_documents(docs, world.seed, epoch))
            shuffled = batch["labels"].copy()
            rng = np.random.default_rng(
                np.random.SeedSequence([config["shuffle_seed"], world.seed, epoch])
            )
            # Each column is one relation at one absolute position. Marginals
            # are exactly preserved, with the same permutations in every arm.
            for col in range(shuffled.shape[1]):
                shuffled[:, col] = shuffled[rng.permutation(len(docs)), col]
            for begin in range(0, len(docs), config["measurement_batch"]):
                end = min(begin + config["measurement_batch"], len(docs))
                tokens = torch.as_tensor(batch["tokens"][begin:end], device=device)
                labels = torch.as_tensor(batch["labels"][begin:end], device=device)
                positions = torch.as_tensor(batch["positions"][0], device=device)
                shuffled_batch = torch.as_tensor(shuffled[begin:end], device=device)
                weight = config["document_weight"] * 0.5 * (end - begin) / total_documents
                predictors = statistic_predictors(
                    model, tokens, positions, labels, shuffled_batch, isolated
                )
                for name, prediction in predictors.items():
                    predictions[name].add_(prediction.double(), alpha=weight)
                logits = model(tokens, positions[None].expand(len(tokens), -1))
                for offset, kind in enumerate(("answer", "eos")):
                    loss = F.cross_entropy(
                        logits[:, offset::2].flatten(0, 1), labels[:, offset::2].flatten()
                    )
                    gradients = torch.autograd.grad(loss, params, retain_graph=offset == 0)
                    for name, gradient in zip(named, gradients, strict=True):
                        accumulators[kind][name].add_(gradient.double(), alpha=weight)
                    losses[kind] += float(loss.detach()) * (end - begin) / total_documents
            print(
                json.dumps({"event": "rotation", "arm": out.name, "rotation": epoch + 1}),
                flush=True,
            )
    result = {kind: numpy_gradients(accumulators[kind]) for kind in accumulators}
    result["predictions"] = numpy_gradients(predictions)
    out.mkdir(parents=True, exist_ok=False)
    np.savez(out / "answer.npz", **result["answer"])
    np.savez(out / "eos.npz", **result["eos"])
    np.savez(out / "predictions.npz", **result["predictions"])
    metadata = {
        "condition": condition,
        "isolated": isolated,
        "documents": total_documents,
        "supervised_tokens": total_documents * 20,
        "losses": losses,
        "seconds": time.perf_counter() - started,
        "files": {name: sha(out / name) for name in ("answer.npz", "eos.npz", "predictions.npz")},
    }
    if any(not np.isfinite(value).all() for part in result.values() for value in part.values()):
        raise FloatingPointError("Nonfinite gradients or predictions")
    write_json(out / "receipt.json", metadata)
    return result


def arm_data(path):
    return {
        kind: dict(np.load(path / f"{filename}.npz"))
        for kind, filename in (("answer", "answer"), ("eos", "eos"), ("predictions", "predictions"))
    }


def common_training_step(model, optimizer, world, config, step, device, schedule, data):
    epoch, offset = divmod(step, 128)
    arranged = epoch_documents(documents(world, "neither"), world.seed, epoch)
    batch = render(world, arranged[offset * 16 : (offset + 1) * 16])
    tokens = torch.as_tensor(batch["tokens"], device=device)
    positions = torch.as_tensor(batch["positions"], device=device)
    labels = torch.as_tensor(batch["labels"], device=device)
    qa = torch.as_tensor(schedule[step], device=device)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for group in optimizer.param_groups:
        group["lr"] = config["lr"] * min((epoch + 1) / config["warmup_epochs"], 1)
    loss = F.cross_entropy(model(tokens, positions).flatten(0, 1), labels.flatten())
    (config["document_weight"] * loss).backward()
    qa_loss = F.cross_entropy(
        model(data["tokens"][qa], data["positions"][qa]).flatten(0, 1), data["labels"][qa].flatten()
    )
    (config["qa_weight"] * qa_loss).backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["clip_norm"])
    if not torch.isfinite(loss + qa_loss + norm):
        raise FloatingPointError("Nonfinite common training step")
    optimizer.step()
    return {
        "step": step + 1,
        "document_loss": float(loss.detach()),
        "qa_loss": float(qa_loss.detach()),
    }


def run(root, config, world_seed, seed, device):
    lock_hash = verify_lock(root, config)
    if world_seed not in config["worlds"] or seed not in config["seeds"]:
        raise ValueError("World/initialization is outside the frozen matrix")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(seed)
    world = make_cross_world(world_seed, root / config["source_world_root"])
    out = root / config["output"] / f"world-{world_seed}-seed-{seed}"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "complete.json").exists():
        raise FileExistsError("Completed run cannot be overwritten")
    audit_path = out / "data-audit.json"
    if not audit_path.exists():
        write_json(audit_path, context_audit(world, config))
    model = CausalLM(
        ModelConfig(
            world.vocab_size, config["width"], config["layers"], config["heads"], config["context"]
        )
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
        fused=device.type == "cuda",
    )
    schedule = qa_schedule(world, max(config["states"]))
    data = tensor_queries(world, device)
    trajectory = []
    write_json(
        out / "identity.json",
        {
            "world": world_seed,
            "seed": seed,
            "lock_sha256": lock_hash,
            "created_at": stamp(),
            "model": model.config_dict(),
            "parameters": sum(p.numel() for p in model.parameters()),
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "qa_schedule_sha256": array_hash(schedule),
        },
    )
    for step in range(max(config["states"]) + 1):
        if step in config["states"]:
            state_path = out / f"state-{step}.pt"
            if state_path.exists():
                # Resumption reconstructs the exact common trajectory and verifies
                # weights before using any existing diagnostic artifacts.
                saved = torch.load(state_path, map_location=device, weights_only=False)
                if any(
                    not torch.equal(v, saved["model"][k]) for k, v in model.state_dict().items()
                ):
                    raise ValueError("Reconstructed reference weights differ")
            else:
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "step": step,
                        "config": model.config_dict(),
                    },
                    state_path,
                )
            for mask in config["masks"]:
                for condition in config["conditions"]:
                    path = out / f"step-{step}-{mask}-{condition}"
                    if (path / "receipt.json").exists():
                        receipt = json.loads((path / "receipt.json").read_text())
                        if any(
                            sha(path / name) != digest for name, digest in receipt["files"].items()
                        ):
                            raise ValueError("Existing gradient artifact hash mismatch")
                        continue
                    if path.exists():
                        raise FileExistsError("Incomplete output must be retained and audited")
                    print(
                        json.dumps(
                            {
                                "event": "arm_start",
                                "world": world_seed,
                                "seed": seed,
                                "step": step,
                                "mask": mask,
                                "condition": condition,
                            }
                        ),
                        flush=True,
                    )
                    measure_arm(model, world, condition, mask == "isolated", config, device, path)
            write_json(out / "trajectory.json", trajectory)
        if step < max(config["states"]):
            trajectory.append(
                common_training_step(model, optimizer, world, config, step, device, schedule, data)
            )
    verify_lock(root, config)
    files = {str(path.relative_to(out)): sha(path) for path in out.rglob("*") if path.is_file()}
    write_json(
        out / "complete.json", {"completed_at": stamp(), "lock_sha256": lock_hash, "files": files}
    )


def summarize(root, config):
    verify_lock(root, config)
    rows, modules, checks = [], [], []
    for world_seed, seed in itertools.product(config["worlds"], config["seeds"]):
        runpath = root / config["output"] / f"world-{world_seed}-seed-{seed}"
        completion = json.loads((runpath / "complete.json").read_text())
        for name, digest in completion["files"].items():
            if sha(runpath / name) != digest:
                raise ValueError("Completed run artifact changed: " + name)
        for step, mask in itertools.product(config["states"], config["masks"]):
            arms = {c: arm_data(runpath / f"step-{step}-{mask}-{c}") for c in CONDITIONS}
            for a, b in CONTRASTS:
                identity = {
                    "world": world_seed,
                    "seed": seed,
                    "step": step,
                    "mask": mask,
                    "first": a,
                    "second": b,
                }
                for kind in ("answer", "eos", "combined"):
                    kinds = ("answer", "eos") if kind == "combined" else (kind,)
                    ga = {n: sum(arms[a][k][n] for k in kinds) for n in arms[a]["answer"]}
                    gb = {n: sum(arms[b][k][n] for k in kinds) for n in arms[b]["answer"]}
                    modules.append(
                        {**identity, "kind": kind, "modules": module_differences(ga, gb)}
                    )
                    for predictor in config["predictors"]:
                        pred = sum(
                            arms[a]["predictions"][f"{predictor}/{k}"]
                            - arms[b]["predictions"][f"{predictor}/{k}"]
                            for k in kinds
                        )
                        row = {
                            **identity,
                            "kind": kind,
                            "predictor": predictor,
                            **vector_metrics(ga[TARGET] - gb[TARGET], pred),
                        }
                        # Near-zero isolated contrasts are cancellation controls;
                        # cosine between rounding residues has no scientific meaning.
                        if mask == "isolated":
                            row["cosine"] = None
                        rows.append(row)
                    norm = np.linalg.norm(ga[TARGET] - gb[TARGET])
                    reference = (np.linalg.norm(ga[TARGET]) + np.linalg.norm(gb[TARGET])) / 2
                    if mask == "isolated":
                        checks.append(
                            {
                                **identity,
                                "kind": kind,
                                "relative_cancellation_residue": float(norm / reference),
                            }
                        )
    if any(c["relative_cancellation_residue"] > 1e-4 for c in checks):
        raise ValueError("Isolated common-state gradient cancellation failed")
    out = root / config["artifacts"]
    write_json(out / "metrics.json", rows)
    write_json(out / "module-metrics.json", modules)
    write_json(
        out / "audit.json",
        {"passed": True, "runs": 4, "arms": 72, "isolated_checks": checks, "completed_at": stamp()},
    )
    primary = [
        r
        for r in rows
        if r["step"] == 0
        and r["mask"] == "open"
        and r["kind"] == "answer"
        and r["second"] == "neither"
    ]
    lines = [
        "# Context statistics and common-state gradients",
        "",
        "Descriptive development experiment; no fitted scale or significance tests.",
        "",
        "| World | Seed | Contrast | Predictor | Cosine | Relative error | Norm ratio |",
        "| --- | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in primary:
        lines.append(
            f"| {row['world']} | {row['seed']} | {row['first']}−{row['second']} | "
            f"{row['predictor']} | {row['cosine']:.6f} | {row['relative_error']:.6f} | "
            f"{row['norm_ratio']:.6f} |"
        )
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            {
                "report": str(out / "report.md"),
                "rows": len(rows),
                "max_isolated_residue": max(c["relative_cancellation_residue"] for c in checks),
            }
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("freeze", "run", "summarize"))
    parser.add_argument("--world", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--config", type=Path, default=Path("configs/bios-context-gradient-v1.json")
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    config = json.loads((root / args.config).read_text())
    if args.mode == "freeze":
        print(json.dumps(freeze(root, config)))
    elif args.mode == "run":
        run(root, config, args.world, args.seed, torch.device(args.device))
    else:
        summarize(root, config)


if __name__ == "__main__":
    main()
