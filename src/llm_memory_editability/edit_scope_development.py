"""Development-only paired edits of a previously audited Wang Loop 4x2 model.

This is an extension, not an original-paper reproduction. All scopes start with
the same function; only the blocks receiving an atomic-update gradient differ.
Case selection never observes edited propagation. The bank includes failures and
pre-edit coverage, and endpoint100 selected weights support a fresh-process audit.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .grok_depth import utc, write_json
from .grokking_reproduction import encode_rows
from .path_overlap import World, load_model, sha256

ARMS = tuple(
    f"{role}-{scope}" for role in ("first", "second") for scope in ("joint", "round1", "round2")
)
CODE_FILES = (
    "src/llm_memory_editability/edit_scope_development.py",
    "scripts/run_edit_scope_development.py",
)


def scope_parameter_names(model):
    """Exact parameter names receiving gradients in an edit model."""
    return tuple(name for name, parameter in model.named_parameters() if parameter.requires_grad)


def make_edit_model(parent, scope):
    """Copy the actual Transformer and choose joint or iteration-specific writes.

    In partial scopes the eight block executions are independent copies of the
    original four blocks. Embeddings, positions and final LayerNorm are frozen.
    The joint arm still uses the same four parameter tensors in both iterations.
    """
    if scope not in ("joint", "round1", "round2"):
        raise ValueError(f"Unknown scope {scope!r}")
    if parent.repeats != 2 or len(parent.transformer.h) != 4:
        raise ValueError("U1 development requires the audited Loop 4x2 parent")
    model = copy.deepcopy(parent).eval().requires_grad_(False)
    if scope != "joint":
        original = list(model.transformer.h)
        model.transformer.h = nn.ModuleList(original + [copy.deepcopy(block) for block in original])
        model.repeats = 1
        model.config = copy.deepcopy(parent.config)
        model.config.n_layer = 8
    selected = range(4) if scope in ("joint", "round1") else range(4, 8)
    for index in selected:
        model.transformer.h[index].requires_grad_(True)
    return model


def parameter_digest(model, *, frozen=False):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if frozen and parameter.requires_grad:
            continue
        tensor = parameter.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _autocast(device, precision):
    return torch.autocast(
        device.type, dtype=torch.bfloat16, enabled=device.type == "cuda" and precision == "bf16"
    )


@torch.no_grad()
def evaluate_queries(
    model, rows, metadata, device, new_answers=None, batch_size=128, precision="bf16"
):
    """Full-vocabulary greedy answer/EOS and old/new ranks and logit margins."""
    model.eval()
    rows = np.asarray(rows, dtype=np.int64)
    if rows.ndim != 2 or rows.shape[1] not in (3, 4):
        raise ValueError("Rows must be atomic triples or two-hop quadruples")
    new_answers = rows[:, -1] if new_answers is None else np.asarray(new_answers, dtype=np.int64)
    if len(new_answers) != len(rows):
        raise ValueError("One new label is required per query")
    output = {
        key: []
        for key in (
            "answer",
            "stop",
            "old_rank",
            "new_rank",
            "new_minus_old",
            "old_margin",
            "new_margin",
        )
    }
    for start in range(0, len(rows), batch_size):
        part = rows[start : start + batch_size]
        x, positions, labels = [
            torch.as_tensor(array, device=device) for array in encode_rows(part, metadata)
        ]
        new = torch.as_tensor(
            new_answers[start : start + batch_size] + metadata["entity_offset"], device=device
        )
        with _autocast(device, precision):
            logits = model(x, positions)[:, 0].float()
            answer = logits.argmax(-1)
            old = labels[:, 0]
            row = torch.arange(len(x), device=device)
            old_score, new_score = logits[row, old], logits[row, new]
            old_rank = (logits > old_score[:, None]).sum(-1) + 1
            new_rank = (logits > new_score[:, None]).sum(-1) + 1
            top_score, top_token = logits.max(-1)
            top2 = logits.topk(2, dim=-1).values[:, 1]
            old_margin = old_score - torch.where(top_token == old, top2, top_score)
            new_margin = new_score - torch.where(top_token == new, top2, top_score)
            x[row, positions[:, 1]] = answer
            stop = model(x, positions)[:, 1].argmax(-1)
        values = (answer, stop, old_rank, new_rank, new_score - old_score, old_margin, new_margin)
        for key, value in zip(output, values, strict=True):
            output[key].append(value.cpu().numpy())
    types = {"answer": np.int64, "stop": np.int64, "old_rank": np.int64, "new_rank": np.int64}
    result = {
        key: np.concatenate(value) if value else np.empty(0, dtype=types.get(key, np.float32))
        for key, value in output.items()
    }
    result.update(rows=rows, old_answers=rows[:, -1], new_answers=new_answers)
    return result


def prediction_metrics(prediction, metadata):
    n = len(prediction["answer"])
    stop = prediction["stop"] == metadata["end_marker"]
    old = prediction["answer"] == prediction["old_answers"] + metadata["entity_offset"]
    new = prediction["answer"] == prediction["new_answers"] + metadata["entity_offset"]

    def mean(array):
        return float(np.mean(array)) if n else None

    return {
        "n": n,
        "old_answer_accuracy": mean(old),
        "old_accuracy": mean(old & stop),
        "new_answer_accuracy": mean(new),
        "new_accuracy": mean(new & stop),
        "new_minus_old": mean(prediction["new_minus_old"]),
        "old_margin": mean(prediction["old_margin"]),
        "new_margin": mean(prediction["new_margin"]),
    }


class BaselineCache:
    """Lazy full-vocabulary baseline evaluation; no post-edit information."""

    def __init__(self, world, model, device, precision):
        self.world, self.model, self.device, self.precision = world, model, device, precision
        self.atom = np.full(len(world.atoms), -1, np.int8)
        self.chain = np.full(len(world.chains), -1, np.int8)
        self.evaluated = {"atom": 0, "chain": 0}

    def correct(self, kind, indices):
        ids = np.asarray(indices, dtype=np.int64)
        cache, rows = (
            (self.atom, self.world.atoms) if kind == "atom" else (self.chain, self.world.chains)
        )
        missing = np.unique(ids[cache[ids] < 0])
        if len(missing):
            pred = evaluate_queries(
                self.model,
                rows[missing],
                self.world.metadata,
                self.device,
                precision=self.precision,
            )
            cache[missing] = (pred["answer"] == rows[missing, -1] + self.world.eo) & (
                pred["stop"] == self.world.metadata["end_marker"]
            )
            self.evaluated[kind] += len(missing)
        return cache[ids] == 1


def _choose_correct(candidates, count, kind, baseline, rng):
    candidates = np.asarray(candidates, dtype=np.int64)
    order = rng.permutation(candidates)
    selected = []
    # Evaluate enough at once without scoring the entire 800k chain universe.
    for start in range(0, len(order), max(128, count * 2)):
        chunk = order[start : start + max(128, count * 2)]
        selected.extend(chunk[baseline.correct(kind, chunk)].tolist())
        if len(selected) >= count:
            break
    return selected[:count]


def select_case_bank(world, role, baseline, settings, phi=7.2):
    """Paired bank from a fixed ordering and pre-edit prerequisite correctness.

    ``baseline.correct(kind, indices)`` may be a CPU-test fixture or BaselineCache.
    ID/OOD refers to the edited atom. Chain labels come from the unchanged world
    except the one edited edge. Cases are not selected using propagation outcomes.
    """
    if role not in ("first", "second"):
        raise ValueError(role)
    seed = int(settings.get("seed", 20261011))
    rng = np.random.RandomState(seed + (0 if role == "first" else 1))
    counts = {"id": int(settings.get("id_cases", 16)), "ood": int(settings.get("ood_cases", 16))}
    n_d = int(settings.get("affected_chains", 8))
    minimum = int(settings.get("min_affected_chains", 1))
    support = world.support(phi)
    all_atoms, all_chains = np.arange(len(world.atoms)), np.arange(len(world.chains))
    # One stable index of affected chains per target atom, avoiding repeated scans.
    fact_side = world.first if role == "first" else world.second
    ordering = np.argsort(fact_side, kind="stable")
    boundaries = np.searchsorted(fact_side[ordering], np.arange(len(world.atoms) + 1))
    cases, attempts = [], []
    for group in ("id", "ood"):
        selected = 0
        for atom_id in rng.permutation(all_atoms[world.ood == (group == "ood")]):
            if selected >= counts[group]:
                break
            atom_id = int(atom_id)
            record = {"atom": atom_id, "group": group, "status": "target_not_baseline_correct"}
            attempts.append(record)
            if not baseline.correct("atom", [atom_id])[0]:
                continue
            available = ordering[boundaries[atom_id] : boundaries[atom_id + 1]]
            available = available[(world.first[available] != world.second[available])]
            prerequisites = world.second[available] if role == "first" else world.first[available]
            available = available[baseline.correct("atom", prerequisites)]
            available = available[baseline.correct("chain", available)]
            if len(available) < minimum:
                record["status"] = "insufficient_correct_original_chains"
                continue
            available = rng.permutation(available)
            old_tail = int(world.atoms[atom_id, 2])
            chosen, new_labels, new_tail, extra_atoms = None, None, None, []
            for proposed in rng.permutation(world.entities):
                proposed = int(proposed)
                if proposed == old_tail:
                    continue
                if role == "second":
                    chosen = available[:n_d]
                    new_labels = np.full(len(chosen), proposed, dtype=np.int64)
                    new_tail = proposed
                    break
                successors = world.lookup[proposed, world.chains[available, 2]]
                valid = successors >= 0
                valid &= successors != atom_id  # The changed edge cannot occur twice.
                valid &= world.atoms[np.maximum(successors, 0), 2] != world.chains[available, 3]
                candidates, successors = available[valid], successors[valid]
                if len(candidates) < minimum:
                    continue
                valid = baseline.correct("atom", successors)
                candidates, successors = candidates[valid], successors[valid]
                if len(candidates) >= minimum:
                    chosen, extra_atoms = candidates[:n_d], successors[:n_d].tolist()
                    new_labels = world.atoms[successors[:n_d], 2]
                    new_tail = proposed
                    break
            if chosen is None:
                record["status"] = "no_valid_new_target_with_correct_successors"
                continue
            dependent_atoms = set(
                world.first[chosen].tolist()
                + world.second[chosen].tolist()
                + extra_atoms
                + [atom_id]
            )
            unaffected = (world.first != atom_id) & (world.second != atom_id)
            unrelated = unaffected & ~np.isin(world.first, list(dependent_atoms))
            unrelated &= ~np.isin(world.second, list(dependent_atoms))
            case_rng = np.random.RandomState(seed + atom_id * 13 + (0 if role == "first" else 1))
            u_atoms = _choose_correct(
                [i for i in all_atoms if i not in dependent_atoms],
                int(settings.get("unrelated_atoms", 64)),
                "atom",
                baseline,
                case_rng,
            )
            u_chains = _choose_correct(
                all_chains[unrelated],
                int(settings.get("unrelated_chains", 64)),
                "chain",
                baseline,
                case_rng,
            )
            controls = all_atoms[
                (world.atoms[:, 2] == old_tail) & (world.atoms[:, 0] != world.atoms[atom_id, 0])
            ]
            controls = np.asarray(
                [i for i in controls if i not in dependent_atoms and i not in u_atoms],
                dtype=np.int64,
            )
            control_atoms = _choose_correct(
                controls, int(settings.get("same_answer_atoms", 16)), "atom", baseline, case_rng
            )
            excluded_atoms = (
                dependent_atoms
                | set(u_atoms)
                | set(control_atoms)
                | set(world.first[u_chains].tolist())
                | set(world.second[u_chains].tolist())
            )
            replay_atoms = _choose_correct(
                [i for i in all_atoms if i not in excluded_atoms],
                int(settings.get("replay_atoms", 32)),
                "atom",
                baseline,
                case_rng,
            )
            excluded_chains = set(chosen.tolist()) | set(u_chains)
            replay_eligible = support & unaffected
            replay_eligible &= ~np.isin(world.first, list(excluded_atoms))
            replay_eligible &= ~np.isin(world.second, list(excluded_atoms))
            replay_chains = _choose_correct(
                [i for i in all_chains[replay_eligible] if i not in excluded_chains],
                int(settings.get("replay_chains", 32)),
                "chain",
                baseline,
                case_rng,
            )
            case = {
                "case_id": f"{role}-{group}-{selected:03d}-a{atom_id}",
                "role": role,
                "group": group,
                "atom": atom_id,
                "old_tail": old_tail,
                "new_tail": new_tail,
                "D": chosen.tolist(),
                "D_new_answers": np.asarray(new_labels).tolist(),
                "D_kind": world.kind[chosen].tolist(),
                "D_training_support": support[chosen].tolist(),
                "D_prerequisite_atoms": sorted(dependent_atoms - {atom_id}),
                "U_atoms": u_atoms,
                "U_chains": u_chains,
                "control_atoms": control_atoms,
                "R_atoms": replay_atoms,
                "R_chains": replay_chains,
            }
            # Controls have sparse same-answer pools; keep their coverage explicitly.
            case["coverage"] = {
                "D": len(chosen),
                "U_atoms": len(u_atoms),
                "U_chains": len(u_chains),
                "control_atoms": len(control_atoms),
                "R_atoms": len(replay_atoms),
                "R_chains": len(replay_chains),
            }
            required = {
                "U_atoms": settings.get("unrelated_atoms", 64),
                "U_chains": settings.get("unrelated_chains", 64),
                "R_atoms": settings.get("replay_atoms", 32),
                "R_chains": settings.get("replay_chains", 32),
            }
            if any(case["coverage"][key] < int(count) for key, count in required.items()):
                record["status"] = "insufficient_holdout_or_replay_coverage"
                record["coverage"] = case["coverage"]
                continue
            record["status"] = "selected"
            record["case_id"] = case["case_id"]
            cases.append(case)
            selected += 1
    return {
        "role": role,
        "seed": seed,
        "selection": settings,
        "requested": counts,
        "selected": {kind: sum(c["group"] == kind for c in cases) for kind in counts},
        "candidate_attempts": attempts,
        "cases": cases,
        "selection_uses": (
            "fixed ordering, original atomic and chain answer/EOS correctness only; "
            "no edited propagation"
        ),
    }


def case_queries(world, case):
    """Return disjoint named evaluation sets, preserving old and new truths."""
    e = world.atoms[[case["atom"]]]
    return {
        "E": (e, np.array([case["new_tail"]])),
        "D": (world.chains[case["D"]], np.array(case["D_new_answers"])),
        "U_atoms": (world.atoms[case["U_atoms"]], None),
        "U_chains": (world.chains[case["U_chains"]], None),
        "same_answer": (
            world.atoms[case["control_atoms"]],
            np.full(len(case["control_atoms"]), case["new_tail"]),
        ),
        "R_atoms": (world.atoms[case["R_atoms"]], None),
        "R_chains": (world.chains[case["R_chains"]], None),
    }


@torch.no_grad()
def first_iteration_predictions(model, rows, metadata, device, precision="bf16"):
    """Decode at r1 after the first four real block executions, with final LN."""
    rows = np.asarray(rows, dtype=np.int64)
    if not len(rows):
        return np.empty(0, dtype=np.int64)
    x, _, _ = encode_rows(rows, metadata)
    tokens = torch.as_tensor(x, device=device)
    model.eval()
    cfg, transformer = model.config, model.transformer
    with _autocast(device, precision):
        hidden = transformer.wte(tokens) + transformer.wpe(
            torch.arange(tokens.shape[1], device=device)
        )
        for block in list(transformer.h)[:4]:
            z = block.ln_1(hidden)
            q, k, v = block.attn.c_attn(z).split(cfg.n_embd, dim=2)
            shape = (len(tokens), tokens.shape[1], cfg.n_head, cfg.n_embd // cfg.n_head)
            q, k, v = (part.view(shape).transpose(1, 2) for part in (q, k, v))
            attn = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
            hidden = hidden + block.attn.c_proj(
                attn.transpose(1, 2).reshape(len(tokens), tokens.shape[1], cfg.n_embd)
            )
            hidden = hidden + block.mlp(block.ln_2(hidden))
        logits = F.linear(transformer.ln_f(hidden[:, 1]), transformer.wte.weight)
    return logits.argmax(-1).cpu().numpy()


def _write_prediction(path, model, world, case, device, precision):
    raw, metrics = {}, {}
    for group, (rows, new) in case_queries(world, case).items():
        prediction = evaluate_queries(model, rows, world.metadata, device, new, precision=precision)
        metrics[group] = prediction_metrics(prediction, world.metadata)
        for key, array in prediction.items():
            raw[f"{group}/{key}"] = array
    raw["first_iteration/E"] = first_iteration_predictions(
        model, world.atoms[[case["atom"]]], world.metadata, device, precision
    )
    raw["first_iteration/D"] = first_iteration_predictions(
        model, world.chains[case["D"]], world.metadata, device, precision
    )
    raw["first_iteration/D_old_bridge"] = world.atoms[world.first[case["D"]], 2] + world.eo
    raw["first_iteration/D_new_bridge"] = (
        np.full(len(case["D"]), case["new_tail"])
        if case["role"] == "first"
        else world.atoms[world.first[case["D"]], 2]
    ) + world.eo
    np.savez_compressed(path, **raw)
    return metrics


def _training_tables(world, case, device):
    target = world.atoms[[case["atom"]]].copy()
    target[:, 2] = case["new_tail"]
    e = [torch.as_tensor(array, device=device) for array in encode_rows(target, world.metadata)]
    atom = encode_rows(world.atoms[case["R_atoms"]], world.metadata)
    chain = encode_rows(world.chains[case["R_chains"]], world.metadata)
    replay = [
        torch.as_tensor(np.concatenate([a, b]), device=device)
        for a, b in zip(atom, chain, strict=True)
    ]
    return e, replay


def edit_loss(model, target, replay):
    """Equal target-versus-replay means; answers and EOS only, no D supervision."""
    x, positions, labels = target
    e = F.cross_entropy(model(x, positions).flatten(0, 1), labels.flatten())
    x, positions, labels = replay
    r = F.cross_entropy(model(x, positions).flatten(0, 1), labels.flatten())
    return 0.5 * (e + r), e.detach(), r.detach()


def _provenance(config, config_path, checkpoint, project):
    return {
        "config_sha256": sha256(config_path),
        "checkpoint_sha256": sha256(checkpoint),
        "world_sha256": sha256(Path(config["source"]["root"]) / "data/world.npz"),
        "code_sha256": {name: sha256(project / name) for name in CODE_FILES},
        "single_world_single_initialization": True,
        "development_only": True,
    }


def _verify_source(config, world, spec, checkpoint):
    source = config["source"]
    if (
        int(source.get("step", 1500000)) != 1500000
        or spec["phi"] != 7.2
        or spec["weight_decay"] != 0.1
    ):
        raise ValueError("Development source must be phi7.2 wd.1 step1.5M")
    if source.get("checkpoint_sha256") and sha256(checkpoint) != source["checkpoint_sha256"]:
        raise ValueError("Source checkpoint hash changed")
    if source.get("world_sha256") and world.metadata["world_sha256"] != source["world_sha256"]:
        raise ValueError("Source world metadata hash changed")
    if sha256(Path(source["root"]) / "data/world.npz") != world.metadata["world_sha256"]:
        raise ValueError("Source world hash does not match metadata")


def run_arm(config_path, arm, out, device, smoke=False):
    """Run one fixed paired development arm, retaining every selected case."""
    if arm not in ARMS:
        raise ValueError(arm)
    config_path, out = Path(config_path), Path(out)
    config = json.loads(config_path.read_text())
    out.mkdir(parents=True, exist_ok=True)
    if (out / "complete.json").exists():
        raise RuntimeError("Refusing to rerun a completed development arm")
    if (out / "run.json").exists():
        raise RuntimeError(
            "Output already contains an attempt; preserve it and choose a new output"
        )
    role, scope = arm.split("-")
    device = torch.device(device)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    settings = config.get("development", {})
    precision = settings.get("precision", "bf16")
    if precision not in ("bf16", "float32"):
        raise ValueError("Precision must be bf16 or float32")
    lr_grid = [float(value) for value in settings.get("learning_rates", [1e-5, 1e-4])]
    endpoints = [int(value) for value in settings.get("endpoints", [10, 50, 100])]
    save_endpoints = [int(value) for value in settings.get("save_endpoints", [100])]
    if not smoke and (
        lr_grid != [1e-5, 1e-4] or endpoints != [10, 50, 100] or save_endpoints != [100]
    ):
        raise ValueError(
            "Only the pre-fixed two-learning-rate 10/50/100 development grid is supported"
        )
    if smoke:
        endpoints, save_endpoints, lr_grid = [2], [2], [lr_grid[0]]
    world = World(Path(config["source"]["root"]) / "data")
    parent, spec, checkpoint = load_model(
        config["source"]["root"],
        config["source"]["run"],
        int(config["source"].get("step", 1500000)),
        world.metadata,
        device,
    )
    _verify_source(config, world, spec, checkpoint)
    project = Path(__file__).resolve().parents[2]
    provenance = _provenance(config, config_path, checkpoint, project)
    started = time.perf_counter()
    tracking = config.get("tracking", {})
    wandb_run = None
    if tracking.get("enabled", True) and not smoke:
        import wandb

        wandb_run = wandb.init(
            entity=tracking.get("entity", "zhusq20"),
            project=tracking.get("project", "llm-memory-editability"),
            name=tracking.get("run_names", {}).get(arm, f"{config['batch']}-{arm}"),
            id=tracking.get("run_ids", {}).get(
                arm, hashlib.sha256(f"{config['batch']}-{arm}".encode()).hexdigest()[:12]
            ),
            resume="never",
            config={"arm": arm, "development_only": True, "config": config, **provenance},
            mode="online",
            dir=str(out),
        )
    write_json(
        out / "run.json",
        {
            "arm": arm,
            "scope": scope,
            "role": role,
            "config": config,
            **provenance,
            "started_utc": utc(),
            "pid": os.getpid(),
            "device": str(device),
            "wandb_url": wandb_run.url if wandb_run else None,
            "wandb_run_id": wandb_run.id if wandb_run else None,
            "smoke": smoke,
        },
    )
    selection = dict(config.get("selection", {}))
    if smoke:
        selection.update(id_cases=1, ood_cases=1)
    baseline = BaselineCache(world, parent, device, precision)
    bank = select_case_bank(world, role, baseline, selection, spec["phi"])
    # Identical JSON bytes and SHA across the three scopes of each role.
    write_json(out / "case-bank.json", bank)
    bank_hash = sha256(out / "case-bank.json")
    selection_seconds = time.perf_counter() - started
    model = make_edit_model(parent, scope)
    sample = torch.as_tensor(encode_rows(world.atoms[:16], world.metadata)[0], device=device)
    with torch.no_grad(), _autocast(device, precision):
        if not torch.equal(parent(sample), model(sample)):
            raise RuntimeError("Cloning scopes changed pre-edit logits")
        chain_sample = torch.as_tensor(
            encode_rows(world.chains[:16], world.metadata)[0], device=device
        )
        if not torch.equal(parent(chain_sample), model(chain_sample)):
            raise RuntimeError("Cloning scopes changed pre-edit composition logits")
    baseline.model = None
    del parent
    base = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
    frozen_hash = parameter_digest(model, frozen=True)
    selected_names = scope_parameter_names(model)
    selected_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    records, failures, updates, loss_tokens, input_tokens, training_seconds = [], [], 0, 0, 0, 0.0
    effective_input_tokens, evaluation_queries, first_iteration_queries = 0, 0, 0
    for case_index, case in enumerate(bank["cases"]):
        case_out = out / "cases" / case["case_id"]
        case_out.mkdir(parents=True, exist_ok=True)
        target, replay = _training_tables(world, case, device)
        replay_examples = len(replay[0])
        for learning_rate in lr_grid:
            branch = case_out / f"lr-{learning_rate:.0e}"
            branch.mkdir()
            model.load_state_dict(base, strict=True)
            model.eval()
            optimizer = torch.optim.AdamW(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                lr=learning_rate,
                weight_decay=0.0,
            )
            branch_started = time.perf_counter()
            try:
                baseline_metrics = _write_prediction(
                    branch / "predictions-000.npz", model, world, case, device, precision
                )
                evaluation_queries += sum(item["n"] for item in baseline_metrics.values())
                first_iteration_queries += 1 + len(case["D"])
                if not all(
                    baseline_metrics[key]["old_accuracy"] == 1.0
                    for key in baseline_metrics
                    if baseline_metrics[key]["n"]
                ):
                    raise RuntimeError(
                        "Selected baseline-correct query no longer passes answer/EOS scoring"
                    )
                losses = []
                for step in range(1, max(endpoints) + 1):
                    tick = time.perf_counter()
                    optimizer.zero_grad(set_to_none=True)
                    with _autocast(device, precision):
                        loss, e_loss, r_loss = edit_loss(model, target, replay)
                    if not torch.isfinite(loss):
                        raise RuntimeError("Non-finite edit loss")
                    loss.backward()
                    optimizer.step()
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    training_seconds += time.perf_counter() - tick
                    updates += 1
                    loss_tokens += 2 * (1 + replay_examples)
                    input_tokens += 4 * (1 + replay_examples)
                    effective_input_tokens += (
                        3 + 3 * len(case["R_atoms"]) + 4 * len(case["R_chains"])
                    )
                    losses.append(
                        {
                            "step": step,
                            "loss": float(loss.detach()),
                            "E_loss": float(e_loss),
                            "R_loss": float(r_loss),
                        }
                    )
                    if step not in endpoints:
                        continue
                    metrics = _write_prediction(
                        branch / f"predictions-{step:03d}.npz",
                        model,
                        world,
                        case,
                        device,
                        precision,
                    )
                    evaluation_queries += sum(item["n"] for item in metrics.values())
                    first_iteration_queries += 1 + len(case["D"])
                    if parameter_digest(model, frozen=True) != frozen_hash:
                        raise RuntimeError("A frozen parameter changed")
                    state_path = None
                    if step in save_endpoints:
                        state_path = branch / f"selected-{step:03d}.pt"
                        torch.save(
                            {
                                "selected": {
                                    name: parameter.detach().cpu().clone()
                                    for name, parameter in model.named_parameters()
                                    if parameter.requires_grad
                                },
                                "scope": scope,
                                "case_id": case["case_id"],
                                "learning_rate": learning_rate,
                                "step": step,
                                "case_bank_sha256": bank_hash,
                                "frozen_parameter_sha256": frozen_hash,
                                **provenance,
                            },
                            state_path,
                        )
                    record = {
                        "case_id": case["case_id"],
                        "group": case["group"],
                        "learning_rate": learning_rate,
                        "step": step,
                        "metrics": metrics,
                        "prediction_path": str(
                            (branch / f"predictions-{step:03d}.npz").relative_to(out)
                        ),
                        "prediction_sha256": sha256(branch / f"predictions-{step:03d}.npz"),
                        "selected_state_path": str(state_path.relative_to(out))
                        if state_path
                        else None,
                        "selected_state_sha256": sha256(state_path) if state_path else None,
                        "frozen_parameters_unchanged": True,
                        "seconds": time.perf_counter() - branch_started,
                    }
                    records.append(record)
                    write_json(branch / f"endpoint-{step:03d}.json", record)
                    if wandb_run:
                        wandb_run.log(
                            {
                                "completed_endpoints": len(records),
                                "case_index": case_index,
                                "edit_step": step,
                                "learning_rate": learning_rate,
                                "group": case["group"],
                                **{
                                    f"{key}/{metric}": value
                                    for key, item in metrics.items()
                                    for metric, value in item.items()
                                    if value is not None
                                },
                            }
                        )
                write_json(branch / "losses.json", losses)
            except Exception as error:
                failure = {
                    "case_id": case["case_id"],
                    "learning_rate": learning_rate,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "utc": utc(),
                }
                failures.append(failure)
                write_json(branch / "failure.json", failure)
            finally:
                del optimizer
            write_json(
                out / "progress.json",
                {
                    "arm": arm,
                    "case_bank_sha256": bank_hash,
                    "completed_cases": case_index + 1,
                    "selected_cases": len(bank["cases"]),
                    "endpoints": len(records),
                    "failures": failures,
                    "updates": updates,
                    "wall_seconds": time.perf_counter() - started,
                    "updated_utc": utc(),
                },
            )
    expected = len(bank["cases"]) * len(lr_grid) * len(endpoints)
    summary = {
        "arm": arm,
        **provenance,
        "case_bank_sha256": bank_hash,
        "cases_requested": bank["requested"],
        "cases_selected": bank["selected"],
        "case_count": len(bank["cases"]),
        "candidate_attempts": len(bank["candidate_attempts"]),
        "baseline_evaluated": baseline.evaluated,
        "settings": {
            "learning_rates": lr_grid,
            "endpoints": endpoints,
            "save_endpoints": save_endpoints,
            "precision": precision,
            "optimizer": "AdamW",
            "weight_decay": 0.0,
            "loss": "0.5*(target answer/EOS mean + replay answer/EOS mean)",
            "composition_new_answer_supervision": False,
        },
        "endpoints_expected": expected,
        "endpoints_completed": len(records),
        "records": records,
        "failures": failures,
        "updates": updates,
        "atomic_target_exposures": updates,
        "replay_exposures": loss_tokens // 2 - updates,
        "supervised_tokens": loss_tokens,
        "executed_input_tokens": input_tokens,
        "effective_nonpadding_training_input_tokens": effective_input_tokens,
        "training_example_exposures": loss_tokens // 2,
        "evaluation_answer_eos_query_exposures": evaluation_queries,
        "evaluation_answer_eos_forward_passes": 2 * evaluation_queries,
        "evaluation_first_iteration_query_exposures": first_iteration_queries,
        "total_native_model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "executed_blocks_per_forward": len(model.transformer.h) * model.repeats,
        "selected_parameters": selected_parameters,
        "selected_parameter_names": selected_names,
        "frozen_parameter_sha256": frozen_hash,
        "selection_seconds": selection_seconds,
        "training_seconds": training_seconds,
        "wall_seconds": time.perf_counter() - started,
        "peak_gpu_bytes": int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else None,
        "smoke": smoke,
        "endpoint100_independent_audit": "pending",
        "endpoint10_50_independent_audit": "not performed; raw predictions retained",
        "completed_utc": utc(),
    }
    write_json(out / "summary.json", summary)
    passed = (
        not failures
        and len(records) == expected
        and all(bank["selected"][key] == count for key, count in bank["requested"].items())
    )
    write_json(
        out / "complete.json",
        {
            "complete": passed,
            "arm": arm,
            "summary_sha256": sha256(out / "summary.json"),
            "case_bank_sha256": bank_hash,
            "training_and_evaluation_complete": passed,
            "independent_audit_complete": False,
            "completed_utc": utc(),
        },
    )
    if wandb_run:
        wandb_run.summary.update(
            {
                "complete": passed,
                "updates": updates,
                "failures": len(failures),
                "case_bank_sha256": bank_hash,
            }
        )
        wandb_run.finish(exit_code=0 if passed else 1)
    if not passed:
        raise RuntimeError("Development arm incomplete; retained coverage/failures in summary.json")
    return summary


def audit_arm(config_path, out, device):
    """Fresh-process parent plus selected-tensor reload; no optimization steps."""
    config_path, out, device = Path(config_path), Path(out), torch.device(device)
    config = json.loads(config_path.read_text())
    summary = json.loads((out / "summary.json").read_text())
    if sha256(config_path) != summary["config_sha256"]:
        raise ValueError("Audit config differs from original")
    if sha256(out / "case-bank.json") != summary["case_bank_sha256"]:
        raise ValueError("Case bank hash changed")
    project = Path(__file__).resolve().parents[2]
    if {name: sha256(project / name) for name in CODE_FILES} != summary["code_sha256"]:
        raise ValueError("Audit source differs from execution source")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if device.type == "cuda":
        torch.cuda.set_device(device)
    world = World(Path(config["source"]["root"]) / "data")
    parent, spec, checkpoint = load_model(
        config["source"]["root"],
        config["source"]["run"],
        int(config["source"].get("step", 1500000)),
        world.metadata,
        device,
    )
    _verify_source(config, world, spec, checkpoint)
    if sha256(checkpoint) != summary["checkpoint_sha256"]:
        raise ValueError("Audit parent checkpoint differs from original")
    scope = summary["arm"].split("-")[1]
    model = make_edit_model(parent, scope)
    del parent
    base = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
    cases = {
        case["case_id"]: case for case in json.loads((out / "case-bank.json").read_text())["cases"]
    }
    started, checks = time.perf_counter(), []
    for record in summary["records"]:
        if record["selected_state_path"] is None:
            continue
        model.load_state_dict(base, strict=True)
        state_path = out / record["selected_state_path"]
        if sha256(state_path) != record["selected_state_sha256"]:
            raise ValueError("Saved selected parameter state hash changed")
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if set(state["selected"]) != set(scope_parameter_names(model)):
            raise ValueError("Saved state contains the wrong selected parameter set")
        if state["case_bank_sha256"] != summary["case_bank_sha256"]:
            raise ValueError("Selected state belongs to another case bank")
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name in state["selected"]:
                    parameter.copy_(state["selected"][name].to(device))
        if parameter_digest(model, frozen=True) != state["frozen_parameter_sha256"]:
            raise ValueError("Reload changed frozen parameters")
        case = cases[record["case_id"]]
        prediction_path = out / record["prediction_path"]
        if sha256(prediction_path) != record["prediction_sha256"]:
            raise ValueError("Recorded raw prediction hash changed")
        observed = dict(np.load(prediction_path))
        fresh_path = state_path.with_name(f"audit-predictions-{record['step']:03d}.npz")
        _write_prediction(fresh_path, model, world, case, device, summary["settings"]["precision"])
        fresh = dict(np.load(fresh_path))
        agreement, max_error = True, 0.0
        for key in observed:
            if observed[key].dtype.kind in "iub":
                agreement &= np.array_equal(observed[key], fresh[key])
            else:
                difference = (
                    float(np.max(np.abs(observed[key] - fresh[key]))) if observed[key].size else 0.0
                )
                max_error = max(max_error, difference)
                agreement &= np.allclose(observed[key], fresh[key], rtol=1e-5, atol=1e-5)
        check = {
            "case_id": record["case_id"],
            "learning_rate": record["learning_rate"],
            "step": record["step"],
            "passed": bool(agreement),
            "max_logit_margin_error": max_error,
            "frozen_parameters_unchanged": True,
            "prediction_sha256": sha256(fresh_path),
        }
        checks.append(check)
        write_json(
            out / "audit-progress.json",
            {
                "checks": len(checks),
                "passed": all(item["passed"] for item in checks),
                "updated_utc": utc(),
            },
        )
    expected = (
        len(summary["records"])
        // len(summary["settings"]["endpoints"])
        * len(summary["settings"]["save_endpoints"])
    )
    result = {
        "arm": summary["arm"],
        "independent_process_pid": os.getpid(),
        "case_bank_sha256": summary["case_bank_sha256"],
        "code_sha256": summary["code_sha256"],
        "checked": len(checks),
        "expected": expected,
        "passed": len(checks) == expected and all(item["passed"] for item in checks),
        "checks": checks,
        "seconds": time.perf_counter() - started,
        "optimization_updates": 0,
        "completed_utc": utc(),
        "scope": "saved endpoints only; other endpoints have no independent reload audit",
    }
    write_json(out / "audit.json", result)
    if not result["passed"]:
        raise RuntimeError("Independent selected-state replay failed")
    complete_path = out / "complete.json"
    complete = json.loads(complete_path.read_text())
    complete.update(
        independent_audit_complete=True,
        independent_audit_endpoints=summary["settings"]["save_endpoints"],
        audit_sha256=sha256(out / "audit.json"),
        audit_completed_utc=utc(),
    )
    write_json(complete_path, complete)
    return result
