"""Learn to use temporarily changed factual parameters, without test edit labels.

Every episode uses an atomic-only MLP update, followed by language-model training
of the remaining parameters. The edited MLP is then restored exactly. This is
episodic reader adaptation, not differentiable meta-learning or an assertion
that all facts reside in that MLP. All three arms use the same restricted scope.
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from . import interface_editing as editing
from .grok_depth import EpochStream, make_optimizer, utc, write_json
from .latent_scaling import build_world, compare_metrics, compare_predictions, model_digest
from .shared_cache import construct, executed_flops
from .storage_composition import FullTokenStep, evaluate, file_hash, generate_rows, pack_sentences

TRAINING_ARMS = ("fixed", "changed_atomic", "changed_use")
CASE_SEED = 107270041
CASE_POOL = 16
TEST_PER_CELL = 4


def panel(spec):
    """Split update addresses from graph only; reserve all test addresses first."""
    cases, qualification = editing.graph_cases(
        spec, seed=CASE_SEED, per_cell=CASE_POOL, replay_n=32
    )
    test = [c for c in cases if int(c["case_id"].rsplit("-", 1)[1]) < TEST_PER_CELL]
    reserved = {tuple(c["old_fact"][:2]) for c in test}
    world = build_world(spec)
    train = []
    for role in editing.ROLES:
        selected = [
            c
            for c in cases
            if c["role"] == role
            and c["stratum"] == "familiar"
            and tuple(c["old_fact"][:2]) not in reserved
            and editing.affected(world["train_composite"], c["old_fact"], role).any()
        ]
        if len(selected) < 4:
            raise ValueError(f"Need four train update addresses per role, got {len(selected)}")
        train.extend(selected[:4])
    train_addresses = {tuple(c["old_fact"][:2]) for c in train}
    if train_addresses & reserved or len(train_addresses) != len(train):
        raise AssertionError("Training/test update addresses overlap")
    return dict(train=train, test=test, qualification=qualification)


def episode_rows(world, case, arm):
    """No heldout query or bridge labels enter any episode's LM objective."""
    if arm not in TRAINING_ARMS:
        raise ValueError(arm)
    atoms = editing.learned_atoms(world)
    original_atoms = atoms.copy()
    lookup = {(int(h), int(r)): int(t) for h, r, t in atoms}
    target = case["old_fact"] if arm == "fixed" else case["new_fact"]
    mask = np.all(atoms[:, :2] == np.asarray(target[:2]), axis=1)
    assert mask.sum() == 1
    atoms[mask, 2] = target[2]
    queries = world["train_composite"].copy()
    affected = editing.affected(queries, case["old_fact"], case["role"])
    if not affected.any():
        raise ValueError("Training edit must affect taught composition queries")
    if arm == "changed_use":
        queries = editing.rewrite(queries, case["old_fact"], case["new_fact"], case["role"], lookup)
    if arm == "changed_atomic":
        # Remove affected queries: old answers would contradict the current fact.
        emphasized = queries[~affected]
    else:
        emphasized = queries[affected]
    unaffected = queries[~affected]
    # Four matched streams: target atomic, other atomics, emphasized composition,
    # unaffected composition. The atomic-only control differs in query exposure;
    # this limitation is recorded rather than claiming exact information matching.
    strata = [np.asarray([target]), atoms[~mask], emphasized, unaffected]
    return strata, dict(
        affected_training_n=int(affected.sum()),
        changed_training_answers=int(np.sum(queries[:, -1] != world["train_composite"][:, -1])),
        direct_atomic_changed=bool(np.any(atoms != original_atoms)),
        stream_sizes=[len(x) for x in strata],
    )


class AtomicStep:
    """Captured atomic edit with exact warmup rollback and fresh Adam state."""

    def __init__(self, model, target, replay, reference, lr):
        self.model, self.target, self.replay, self.reference = model, target, replay, reference
        self.parameters = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(self.parameters, lr=lr, capturable=True, foreach=False)
        before = [p.detach().clone() for p in model.parameters()]
        torch.cuda.synchronize()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self.eager()
        torch.cuda.current_stream().wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.loss = self.eager()
        torch.cuda.synchronize()
        with torch.no_grad():
            for parameter, old in zip(model.parameters(), before, strict=True):
                parameter.copy_(old)
                if parameter.grad is not None:
                    parameter.grad.zero_()
            for state in self.optimizer.state.values():
                for value in state.values():
                    if torch.is_tensor(value):
                        value.zero_()

    def eager(self):
        self.optimizer.zero_grad(set_to_none=True)
        loss, _, _ = editing.edit_objective(
            self.model, self.target, self.replay, self.reference, 1.0
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters, 1.0, foreach=True)
        self.optimizer.step()
        return loss

    def run(self, steps):
        for _ in range(steps):
            self.graph.replay()
        torch.cuda.synchronize()
        result = float(self.loss.detach())
        if not np.isfinite(result):
            raise FloatingPointError(result)
        return result


def set_reader_scope(model):
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(not name.startswith("blocks.0.mlp."))
        parameter.grad = None


def train(spec, out, source, engineering=False):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "run.json").exists():
        raise FileExistsError(out)
    editing._configure("cuda:0")
    parent_path = Path(spec["parent_dir"]) / "model-128000.pt"
    payload = torch.load(parent_path, map_location="cpu", weights_only=False)
    parent_spec = payload["spec"]
    model = construct(parent_spec, "cuda:0")
    model.load_state_dict(payload["model"])
    model.eval()
    world = build_world(parent_spec)
    selected_panel = panel(parent_spec)
    assert editing._hash_json(selected_panel) == spec["panel_sha256"]
    writer = {
        n: p.detach().clone() for n, p in model.named_parameters() if n.startswith("blocks.0.mlp.")
    }
    manifest = dict(
        spec=spec,
        parent_spec=parent_spec,
        parent_sha256=file_hash(parent_path),
        initial_model_sha256=model_digest(model),
        writer_names=list(writer),
        panel=selected_panel,
        source=source,
        pid=__import__("os").getpid(),
        gpu=int(__import__("os").environ.get("PHYSICAL_GPU", 0)),
        created_utc=utc(),
    )
    write_json(out / "run.json", manifest)
    np.savez_compressed(out / "world.npz", **world)
    episodes = 2 if engineering else spec["episodes"]
    inner_steps = 4 if engineering else spec["inner_steps"]
    reader_steps = 4 if engineering else spec["reader_steps"]
    history, records = [], []
    training_seconds = 0.0
    started = time.perf_counter()

    def measure(episode):
        nonlocal model
        metrics, predictions = evaluate(model, world, "low", "cuda:0")
        node = episode * (inner_steps + reader_steps)
        history.append(
            dict(
                step=node,
                episode=episode,
                metrics=metrics,
                training_seconds=training_seconds,
                wall_seconds=time.perf_counter() - started,
                reader_updates=episode * reader_steps,
                atomic_update_steps=episode * inner_steps,
                examples=episode * (inner_steps * 33 + reader_steps * spec["batch_size"]),
                target_supervised_tokens=episode * inner_steps * 3,
                replay_distillation_positions=episode * inner_steps * 32 * 3,
                reader_supervised_tokens=episode * reader_steps * spec["batch_size"] * 8,
                # Execution-shape estimate; frozen MLP and KL are not exact backward FLOPs.
                estimated_matmul_training_flops=episode
                * (
                    inner_steps
                    * executed_flops(model.config, model.repeats, 33, 9, model.memory_arm, 3)
                    + reader_steps
                    * executed_flops(
                        model.config, model.repeats, spec["batch_size"], 9, model.memory_arm
                    )
                ),
            )
        )
        write_json(out / "learning.json", history)
        np.savez_compressed(out / f"predictions-{episode:03d}.npz", **predictions)
        torch.save(
            dict(spec=parent_spec, model=model.state_dict(), episode=episode),
            out / f"model-{episode:03d}.pt",
        )
        write_json(out / "status.json", dict(state="running", step=node, episode=episode))
        print(json.dumps(dict(name=spec["name"], episode=episode, metrics=metrics)), flush=True)

    measure(0)
    for episode in range(episodes):
        case = selected_panel["train"][episode % len(selected_panel["train"])]
        before_writer = {n: p.detach().clone() for n, p in model.named_parameters() if n in writer}
        for n in writer:
            torch.testing.assert_close(before_writer[n], writer[n], rtol=0, atol=0)
        tasks, _ = editing.case_tasks(world, case)
        replay = editing.answer_batch(tasks["R_atomic"], "cuda:0")
        model.eval()
        with torch.no_grad():
            reference = model(replay[0], positions=replay[1]).log_softmax(-1).detach()
        selected = editing.editable_parameters(model)
        assert set(selected) == set(writer)
        target = case["old_fact"] if spec["training_arm"] == "fixed" else case["new_fact"]
        target_batch = editing.answer_batch(np.asarray([target]), "cuda:0")
        torch.cuda.synchronize()
        began = time.perf_counter()
        operation = AtomicStep(model, target_batch, replay, reference, spec["inner_lr"])
        if engineering:
            eager_model = copy.deepcopy(model)
            eager_parameters = [p for p in eager_model.parameters() if p.requires_grad]
            eager_optimizer = torch.optim.Adam(
                eager_parameters, lr=spec["inner_lr"], capturable=True, foreach=False
            )
            for _ in range(inner_steps):
                eager_optimizer.zero_grad(set_to_none=True)
                eager_loss, _, _ = editing.edit_objective(
                    eager_model, target_batch, replay, reference, 1.0
                )
                eager_loss.backward()
                torch.nn.utils.clip_grad_norm_(eager_parameters, 1.0, foreach=True)
                eager_optimizer.step()
        inner_loss = operation.run(inner_steps)
        if engineering:
            for actual, expected in zip(model.parameters(), eager_model.parameters(), strict=True):
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
            del eager_model, eager_optimizer
        training_seconds += time.perf_counter() - began
        inner_metrics, _ = generate_rows(model, np.asarray([target]), "cuda:0")
        del operation
        strata, info = episode_rows(world, case, spec["training_arm"])
        set_reader_scope(model)
        model.train()
        table = tuple(
            torch.as_tensor(np.concatenate(parts), device="cuda:0")
            for parts in zip(*(pack_sentences(rows) for rows in strata), strict=True)
        )
        sizes = [len(rows) for rows in strata]
        offsets = np.cumsum([0, *sizes[:-1]])
        streams = [
            EpochStream(n, spec["stream_seed"] + episode * 10 + i) for i, n in enumerate(sizes)
        ]
        batch = spec["batch_size"]
        reader_lr = torch.tensor(spec["reader_lr"], device="cuda:0")
        optimizer = make_optimizer(model, reader_lr, 0.0)
        began = time.perf_counter()
        graph = FullTokenStep(model, optimizer, table, batch)
        for start in range(0, reader_steps, 128):
            n = min(128, reader_steps - start)
            indices = np.concatenate(
                [
                    s.take(n * batch // 4).reshape(n, batch // 4) + offset
                    for s, offset in zip(streams, offsets, strict=True)
                ],
                axis=1,
            )
            indices = torch.as_tensor(indices, device="cuda:0")
            for j in range(n):
                loss = graph(indices[j])
        torch.cuda.synchronize()
        training_seconds += time.perf_counter() - began
        reader_loss = float(loss.detach())
        if not np.isfinite(reader_loss):
            raise FloatingPointError(reader_loss)
        after_metrics, _ = generate_rows(model, np.asarray([target]), "cuda:0")
        del graph, optimizer
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name in writer:
                    parameter.copy_(writer[name])
        records.append(
            dict(
                episode=episode + 1,
                case_id=case["case_id"],
                role=case["role"],
                target=target,
                inner_success=inner_metrics["accuracy"],
                after_use_atomic_success=after_metrics["accuracy"],
                inner_loss=inner_loss,
                reader_loss=reader_loss,
                **info,
            )
        )
        write_json(out / "episodes.json", records)
        if episode + 1 in {1, 8, 16, 24, episodes}:
            measure(episode + 1)
    torch.save(dict(spec=parent_spec, model=model.state_dict()), out / "model.pt")
    result = dict(
        spec=spec,
        endpoint=history[-1],
        episodes=records,
        writer_restored_exact=True,
        final_model_sha256=model_digest(model),
        checkpoint_sha256=file_hash(out / "model.pt"),
        training_seconds=training_seconds,
        process_seconds=time.perf_counter() - started,
        finished_utc=utc(),
        engineering=engineering,
    )
    write_json(out / "complete.json", result)
    return result


def audit(out):
    editing._configure("cuda:0")
    out = Path(out)
    manifest = json.loads((out / "run.json").read_text())
    complete = json.loads((out / "complete.json").read_text())
    spec = manifest["parent_spec"]
    world = build_world(spec)
    if panel(spec) != manifest["panel"]:
        raise AssertionError("Update partition changed")
    parent = torch.load(
        Path(manifest["spec"]["parent_dir"]) / "model-128000.pt",
        map_location="cpu",
        weights_only=False,
    )
    maximum = 0.0
    history = json.loads((out / "learning.json").read_text())
    for row in history:
        payload = torch.load(
            out / f"model-{row['episode']:03d}.pt", map_location="cpu", weights_only=False
        )
        model = construct(spec, "cuda:0")
        model.load_state_dict(payload["model"])
        for name in manifest["writer_names"]:
            torch.testing.assert_close(
                payload["model"][name], parent["model"][name], rtol=0, atol=0
            )
        metrics, predictions = evaluate(model, world, "low", "cuda:0")
        compare_metrics(metrics, row["metrics"])
        with np.load(out / f"predictions-{row['episode']:03d}.npz") as saved:
            maximum = max(maximum, compare_predictions(predictions, saved))
    payload = torch.load(out / "model.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])
    assert model_digest(model) == complete["final_model_sha256"]
    assert file_hash(out / "model.pt") == complete["checkpoint_sha256"]
    write_json(
        out / "audit.json",
        dict(
            passed=True,
            checked_nodes=len(history),
            max_nll_error=maximum,
            writer_exact=True,
            utc=utc(),
        ),
    )
