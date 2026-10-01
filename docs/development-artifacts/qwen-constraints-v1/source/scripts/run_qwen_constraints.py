#!/usr/bin/env python3
"""Real Qwen constraints: frozen selection, all-position gradients, forward interventions."""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_memory_editability.qwen_constraints import (
    local_gradients,
    module_projection,
    psd_solve,
    route_output,
    select_records,
    update_scale,
)

ROOT = Path.cwd()
CONFIG = ROOT / "configs/qwen-constraints-v1.json"
ART = ROOT / "docs/development-artifacts/qwen-constraints-v1"
RESULTS = ROOT / "results/qwen-constraints-v1"
SOURCE_FILES = [
    "scripts/run_qwen_constraints.py",
    "src/llm_memory_editability/qwen_constraints.py",
    "tests/test_qwen_constraints.py",
    "configs/qwen-constraints-v1.json",
]


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read(path):
    return json.loads(path.read_text())


def prepare():
    if (ART / "lock.json").exists():
        raise RuntimeError("A lock already exists; preserve it and document any new version")
    cfg = read(CONFIG)
    pools = read(ROOT / cfg["pools"])
    tokenizer = AutoTokenizer.from_pretrained(ROOT / cfg["model_source"], local_files_only=True)
    selection = {}
    for role, pool_name in cfg["pools_by_role"].items():
        selection[role] = select_records(
            pools[pool_name], cfg["counts"][role], cfg["selection_seed"], role
        )
    sets = [{r["subject_group"] for r in selection[role]} for role in ["E", "R", "U"]]
    assert all(not sets[i] & sets[j] for i in range(3) for j in range(i))
    text_rows = read(ROOT / cfg["text"])["test"]
    text_indices = sorted(
        range(len(text_rows)),
        key=lambda i: hashlib.sha256(f"{cfg['selection_seed']}:W:{i}".encode()).hexdigest(),
    )[: cfg["counts"]["W"]]
    encoded = {}
    for prompt_format in cfg["formats"]:
        rows = []
        for role in ["E", "R", "U"]:
            for item in selection[role]:
                if prompt_format == "qa":
                    enc = item["encoded"][0]
                    ids = enc["input_ids"][: enc["answer_start"]]
                    target = enc["input_ids"][enc["answer_start"]]
                else:
                    ids = tokenizer.encode(item["views"][0], add_special_tokens=False)
                    full = tokenizer.encode(
                        item["views"][0] + " " + item["answer"], add_special_tokens=False
                    )
                    assert full[: len(ids)] == ids, item["case_id"]
                    target = full[len(ids)]
                rows.append(
                    dict(
                        role=role,
                        case_id=item["case_id"],
                        relation=item["relation_id"],
                        prompt=item["views"][0],
                        answer=item["answer"],
                        input_ids=ids,
                        target=target,
                    )
                )
        for i in text_indices:
            item = text_rows[i]
            rows.append(
                dict(
                    role="W",
                    case_id=f"wiki-{i}",
                    relation="text",
                    answer="",
                    prompt="WikiText test prefix",
                    input_ids=item["input_ids"][:-1],
                    target=item["input_ids"][-1],
                )
            )
        encoded[prompt_format] = rows
    write_json(ART / "selection.json", encoded)
    paths = SOURCE_FILES + ["docs/hebbian-learning-plan-v1.md", "docs/experimental-protocol.md"]
    for name in paths:
        destination = ART / "source" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
    model_file = ROOT / cfg["model_source"] / "model.safetensors"
    lock = dict(
        time=now(),
        scope=cfg["scope"],
        config=cfg,
        sources={name: digest(ROOT / name) for name in paths},
        inputs={
            str(p.relative_to(ROOT)): digest(p)
            for p in [ROOT / cfg["pools"], ROOT / cfg["text"], model_file, ART / "selection.json"]
        },
        environment=dict(
            python=platform.python_version(),
            torch=torch.__version__,
            transformers=transformers.__version__,
            numpy=np.__version__,
        ),
        selection_counts=cfg["counts"],
        selected_text_indices=text_indices,
        status="locked_before_new_model_diagnostics_or_interventions",
    )
    write_json(ART / "lock.json", lock)
    print(json.dumps({"locked": str(ART / "lock.json"), "jobs": 6}), flush=True)


class Experiment:
    def __init__(self, layer, prompt_format, device):
        self.cfg = read(CONFIG)
        self.rows = read(ART / "selection.json")[prompt_format]
        self.device = torch.device(device)
        torch.set_num_threads(4)
        torch.manual_seed(self.cfg["selection_seed"])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                ROOT / self.cfg["model_source"],
                local_files_only=True,
                dtype=torch.float32,
                attn_implementation="eager",
            )
            .to(self.device)
            .eval()
            .requires_grad_(False)
        )
        self.mlp = self.model.model.layers[layer].mlp
        self.parameters = {
            "down": self.mlp.down_proj.weight,
            "up": self.mlp.up_proj.weight,
            "gate": self.mlp.gate_proj.weight,
        }
        self.base = {group: p.detach().clone() for group, p in self.parameters.items()}
        self.original_hash = self.local_hash()
        self.reference_logp = None
        self.competitors = None
        self.route = "all"
        self.route_handle = None
        self.attention = None
        self.out = RESULTS / f"layer-{layer}-{prompt_format}"
        self.art = ART / f"layer-{layer}-{prompt_format}"
        self.out.mkdir(parents=True, exist_ok=True)
        self.art.mkdir(parents=True, exist_ok=True)
        self.started = time.time()
        self.layer = layer
        self.prompt_format = prompt_format

    def local_hash(self):
        h = hashlib.sha256()
        for p in self.parameters.values():
            h.update(p.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()

    def status(self, phase, **extra):
        value = dict(
            time=now(),
            layer=self.layer,
            format=self.prompt_format,
            phase=phase,
            elapsed_seconds=time.time() - self.started,
            **extra,
        )
        write_json(self.art / "status.json", value)
        print(json.dumps(value), flush=True)

    def batch(self, indices):
        sequences = [self.rows[i]["input_ids"] for i in indices]
        ids = torch.full(
            (len(indices), max(map(len, sequences))), 151643, device=self.device, dtype=torch.long
        )
        attention = torch.zeros_like(ids)
        for row, sequence in enumerate(sequences):
            ids[row, : len(sequence)] = torch.tensor(sequence, device=self.device)
            attention[row, : len(sequence)] = 1
        self.attention = attention
        return ids, attention

    def hidden(self, ids, attention):
        output = self.model.model(input_ids=ids, attention_mask=attention, use_cache=False)
        return output.last_hidden_state[
            torch.arange(len(ids), device=self.device), attention.sum(1) - 1
        ]

    @torch.no_grad()
    def evaluate(self, indices):
        results = []
        for offset in range(0, len(indices), self.cfg["batch_size"]):
            chosen = indices[offset : offset + self.cfg["batch_size"]]
            ids, attention = self.batch(chosen)
            logits = self.model.lm_head(self.hidden(ids, attention))
            logp = F.log_softmax(logits, dim=-1)
            targets = torch.tensor([self.rows[i]["target"] for i in chosen], device=self.device)
            arange = torch.arange(len(chosen), device=self.device)
            if self.competitors is None:
                excluded = logits.clone()
                excluded[arange, targets] = -torch.inf
                competitors = excluded.argmax(-1)
                results.append(
                    (
                        logp,
                        competitors,
                        logits[arange, targets] - logits[arange, competitors],
                        logits.argmax(-1),
                    )
                )
            else:
                competitors = self.competitors[chosen]
                ref = self.reference_logp[chosen]
                kl = (ref.exp() * (ref - logp)).sum(-1).clamp_min(0)
                results.append(
                    torch.stack(
                        [
                            logits[arange, targets] - logits[arange, competitors],
                            kl,
                            (logits.argmax(-1) == targets).float(),
                            (logits.argmax(-1) == self.base_top[chosen]).float(),
                        ],
                        dim=1,
                    ).cpu()
                )
        if self.competitors is None:
            self.reference_logp = torch.cat([r[0] for r in results])
            self.competitors = torch.cat([r[1] for r in results])
            self.base_margin = torch.cat([r[2] for r in results]).cpu()
            self.base_top = torch.cat([r[3] for r in results])
            return None
        return torch.cat(results).numpy()

    def collect(self):
        self.evaluate(list(range(len(self.rows))))
        base_rows = []
        for i, row in enumerate(self.rows):
            base_rows.append(
                {
                    **row,
                    "margin": float(self.base_margin[i]),
                    "competitor": int(self.competitors[i]),
                    "top": int(self.base_top[i]),
                    "first_token_correct": int(self.base_top[i]) == row["target"],
                }
            )
        write_json(self.art / "baseline.json", base_rows)
        count = self.cfg["counts"]["E"] + self.cfg["counts"]["R"]
        bank = {
            group: torch.empty((count, p.numel()), device=self.device)
            for group, p in self.parameters.items()
        }
        last_bank = {
            group: torch.empty((self.cfg["counts"]["E"], p.numel()), device=self.device)
            for group, p in self.parameters.items()
        }
        features, inputs, gate_records, check_errors = [], [], [], []
        capture = {}

        def hook(module, args, output):
            capture["x"] = args[0]
            capture["h"] = output

        handle = self.mlp.register_forward_hook(hook)
        for p in self.parameters.values():
            p.requires_grad_(True)
        try:
            for i in range(count):
                ids, attention = self.batch([i])
                hidden = self.hidden(ids, attention)[0]
                readout = (
                    self.model.lm_head.weight[self.rows[i]["target"]]
                    - self.model.lm_head.weight[self.competitors[i]]
                )
                q = hidden @ readout
                gradients = torch.autograd.grad(q, [*self.parameters.values(), capture["h"]])
                x = capture["x"][0].detach()
                up = F.linear(x, self.base["up"])
                gate = F.linear(x, self.base["gate"])
                phi = up * F.silu(gate)
                d = gradients[-1][0].detach()
                explicit = local_gradients(x, up, gate, self.base["down"], d)
                last = local_gradients(x[-1:], up[-1:], gate[-1:], self.base["down"], d[-1:])
                for j, group in enumerate(self.parameters):
                    gradient = gradients[j].detach()
                    bank[group][i] = gradient.flatten()
                    error = float((explicit[group] - gradient).norm() / gradient.norm())
                    check_errors.append(error)
                    if error > 2e-5:
                        raise AssertionError(f"Full-position derivative mismatch {group}: {error}")
                    if i < self.cfg["counts"]["E"]:
                        last_bank[group][i] = last[group].flatten()
                features.append(phi.detach())
                inputs.append(x.detach())
                if i < self.cfg["counts"]["E"]:
                    gate_records.append(
                        dict(
                            case_id=self.rows[i]["case_id"],
                            silu_absolute_quantiles=torch.quantile(
                                F.silu(gate[-1]).abs(),
                                torch.tensor([0.0, 0.1, 0.5, 0.9, 1.0], device=self.device),
                            ).tolist(),
                            up_norm=float(up[-1].norm()),
                            input_norm=float(x[-1].norm()),
                            last_output_derivative_norm=float(d[-1].norm()),
                            earlier_output_derivative_norm=float(d[:-1].norm()),
                        )
                    )
                if (i + 1) % 8 == 0:
                    self.status("collect_gradients", completed=i + 1, total=count)
        finally:
            handle.remove()
            self.model.requires_grad_(False)
        write_json(
            self.art / "derivative-checks.json",
            dict(
                max_relative_error=max(check_errors),
                checks=len(check_errors),
                all_positions=True,
                gate_diagnostics=gate_records,
            ),
        )
        torch.save(
            {"features": [v.cpu() for v in features], "inputs": [v.cpu() for v in inputs]},
            self.out / "activations.pt",
        )
        return bank, last_bank, features

    def diagnose(self, bank, last_bank, features):
        ne = self.cfg["counts"]["E"]
        nr = self.cfg["counts"]["R"]
        spectrum = torch.linalg.svdvals(self.base["down"].double())
        weight_spectrum = dict(
            shape=list(self.base["down"].shape),
            sigma_min=float(spectrum[-1]),
            sigma_max=float(spectrum[0]),
            condition=float(spectrum[0] / spectrum[-1]),
            ranks={str(t): int((spectrum > spectrum[0] * t).sum()) for t in [1e-8, 1e-6, 1e-4]},
        )
        diagnostics, predictions, directions = [], [], {}
        grams = {}
        for group, gradients in bank.items():
            g64 = gradients.double()
            grams[group] = (g64 @ g64.T).cpu()
            del g64
        torch.save(grams, self.out / "gradient-grams.pt")
        # Precompute projections once; U/W are not differentiated or used to choose a direction.
        for group, gradients in bank.items():
            gram = grams[group].to(self.device)
            keep = gradients[ne:].double()
            for i in range(ne):
                g = gradients[i]
                for n in self.cfg["keep_counts"]:
                    coeff, rank = psd_solve(
                        gram[ne : ne + n, ne : ne + n],
                        gram[ne : ne + n, i],
                        self.cfg["gradient_gram_rtol"],
                    )
                    projected = (g.double() - coeff @ keep[:n]).float()
                    energy = float(projected.double().square().sum() / g.double().square().sum())
                    diagnostics.append(
                        dict(
                            kind="functional",
                            group=group,
                            case_id=self.rows[i]["case_id"],
                            keep_count=n,
                            rank=rank,
                            retained_energy=energy,
                            equal_target_cost_factor=1 / max(energy, 1e-30) ** 0.5,
                            relative_keep_residual=float(
                                (keep[:n] @ projected.double()).norm()
                                / (keep[:n].norm() * projected.double().norm()).clamp_min(1e-30)
                            ),
                        )
                    )
                    if n == nr:
                        directions[(group, i, "functional")] = projected
                full = g.double()
                last = last_bank[group][i].double()
                earlier = full - last
                diagnostics.append(
                    dict(
                        kind="positions",
                        group=group,
                        case_id=self.rows[i]["case_id"],
                        last_full_cosine=float(
                            last @ full / (last.norm() * full.norm()).clamp_min(1e-30)
                        ),
                        last_norm_over_full=float(last.norm() / full.norm()),
                        earlier_norm_over_full=float(earlier.norm() / full.norm()),
                        last_response_fraction=float(last @ full / full.square().sum()),
                        earlier_response_fraction=float(earlier @ full / full.square().sum()),
                    )
                )
            del keep
        for n in self.cfg["keep_counts"]:
            keys = torch.cat(features[ne : ne + n]).double()
            _, singular, vh = torch.linalg.svd(keys, full_matrices=False, driver="gesvd")
            for tolerance in self.cfg["feature_svd_sensitivity"]:
                basis = vh[singular > singular[0] * tolerance]
                for i in range(ne):
                    g = bank["down"][i].reshape_as(self.base["down"])
                    projected = module_projection(g, basis).flatten()
                    energy = float(projected.double().square().sum() / g.double().square().sum())
                    last_phi = features[i][-1].double()
                    z = last_phi - (last_phi @ basis.T) @ basis
                    diagnostics.append(
                        dict(
                            kind="module",
                            group="down",
                            case_id=self.rows[i]["case_id"],
                            keep_count=n,
                            token_features=len(keys),
                            rank=len(basis),
                            svd_rtol=tolerance,
                            retained_energy=energy,
                            last_feature_residual_fraction=float(z.norm() / last_phi.norm()),
                            equal_target_cost_factor=1 / max(energy, 1e-30) ** 0.5,
                        )
                    )
                    if n == nr and tolerance == self.cfg["feature_svd_rtol"]:
                        directions[("down", i, "module")] = projected
            self.status("feature_spaces", keep_count=n, features=len(keys))
        write_json(
            self.art / "geometry.json", dict(weight_spectrum=weight_spectrum, rows=diagnostics)
        )
        for i in range(ne):
            for group, gradients in bank.items():
                methods = [
                    ("gradient", "equal_norm", "all", gradients[i]),
                    ("functional", "equal_norm", "all", directions[(group, i, "functional")]),
                ]
                if group == "down":
                    methods += [
                        ("module", "equal_norm", "all", directions[(group, i, "module")]),
                        ("functional", "equal_target", "all", directions[(group, i, "functional")]),
                        ("module", "equal_target", "all", directions[(group, i, "module")]),
                        ("gradient", "equal_norm", "last", gradients[i]),
                        ("gradient", "equal_norm", "earlier", gradients[i]),
                    ]
                for method, metric, route, direction in methods:
                    for step in self.cfg["steps"]:
                        scale, capped = update_scale(
                            direction,
                            gradients[i],
                            step,
                            float(self.base[group].norm()),
                            self.cfg["relative_parameter_cap"],
                            metric,
                        )
                        delta = direction * scale
                        target_gradient = gradients[i] if route == "all" else last_bank[group][i]
                        if route == "earlier":
                            target_gradient = gradients[i] - target_gradient
                        predictions.append(
                            dict(
                                index=len(predictions),
                                case_id=self.rows[i]["case_id"],
                                target_index=i,
                                group=group,
                                method=method,
                                metric=metric,
                                route=route,
                                step=step,
                                scale=scale,
                                capped=capped,
                                delta_norm=float(delta.norm()),
                                relative_delta_norm=float(delta.norm() / self.base[group].norm()),
                                predicted_target=float(target_gradient.double() @ delta.double()),
                                predicted_R=(gradients[ne:].double() @ delta.double()).tolist()
                                if route == "all"
                                else None,
                            )
                        )
        write_json(
            self.art / "predictions.json",
            dict(
                time=now(),
                rows=predictions,
                scope="Saved before finite parameter/forward interventions; no U/W gradients",
            ),
        )
        self.status("predictions_locked", interventions=len(predictions))
        return directions, predictions

    def route_hook(self, module, args, output):
        if self.route == "all":
            return output
        x = args[0]
        original = F.linear(
            F.linear(x, self.base["up"]) * F.silu(F.linear(x, self.base["gate"])), self.base["down"]
        )
        return route_output(output, original, self.attention, self.route)

    def intervene(self, bank, directions, predictions):
        ne, nr = self.cfg["counts"]["E"], self.cfg["counts"]["R"]
        handle = self.mlp.register_forward_hook(self.route_hook)
        output_path = self.out / "interventions.jsonl"
        if output_path.exists():
            raise RuntimeError("Refusing to overwrite existing finite interventions")
        try:
            with output_path.open("w") as stream:
                for prediction in predictions:
                    i, group = prediction["target_index"], prediction["group"]
                    key = (group, i, prediction["method"])
                    direction = (
                        bank[group][i] if prediction["method"] == "gradient" else directions[key]
                    )
                    delta = (direction * prediction["scale"]).reshape_as(self.base[group])
                    with torch.no_grad():
                        self.parameters[group].copy_(self.base[group] + delta)
                    self.route = prediction["route"]
                    indices = [i, *range(ne, len(self.rows))]
                    values = self.evaluate(indices)
                    shifts = values[:, 0] - self.base_margin[indices].numpy()
                    record = {
                        **prediction,
                        "actual_target": float(shifts[0]),
                        "target_linear_error": float(shifts[0] - prediction["predicted_target"]),
                        "roles": {},
                    }
                    for role in ["R", "U", "W"]:
                        positions = [
                            j for j, k in enumerate(indices) if self.rows[k]["role"] == role
                        ]
                        drift = shifts[positions]
                        record["roles"][role] = dict(
                            count=len(positions),
                            margin_rms=float(np.sqrt(np.mean(drift**2))),
                            margin_max=float(np.max(np.abs(drift))),
                            kl_mean=float(values[positions, 1].mean()),
                            kl_max=float(values[positions, 1].max()),
                            first_token_accuracy=float(values[positions, 2].mean()),
                            top_token_retention=float(values[positions, 3].mean()),
                            margin_changes=drift.tolist(),
                        )
                    # Exact module changes on the same cached inputs; not an end-to-end proxy.
                    if group == "down" and prediction["route"] == "all":
                        module_delta = self.keep_features @ delta.T
                        record["R_module_relative_change"] = float(
                            module_delta.norm() / (self.keep_features @ self.base["down"].T).norm()
                        )
                    if prediction["predicted_R"] is not None:
                        record["R_prediction_rms_error"] = float(
                            np.sqrt(
                                np.mean(
                                    (shifts[1 : 1 + nr] - np.array(prediction["predicted_R"])) ** 2
                                )
                            )
                        )
                    stream.write(json.dumps(record, allow_nan=False) + "\n")
                    stream.flush()
                    with torch.no_grad():
                        self.parameters[group].copy_(self.base[group])
                    if (prediction["index"] + 1) % 12 == 0:
                        self.status(
                            "finite_interventions",
                            completed=prediction["index"] + 1,
                            total=len(predictions),
                        )
        finally:
            handle.remove()
            self.route = "all"
            with torch.no_grad():
                for group, parameter in self.parameters.items():
                    parameter.copy_(self.base[group])
        assert self.local_hash() == self.original_hash
        restored = self.evaluate(list(range(len(self.rows))))
        restoration_error = float(np.max(np.abs(restored[:, 0] - self.base_margin.numpy())))
        if restoration_error > 1e-5:
            raise AssertionError(f"Restoration changed model outputs: {restoration_error}")
        write_json(
            self.art / "completion.json",
            dict(
                time=now(),
                interventions=len(predictions),
                restored_parameter_hash=self.original_hash,
                restoration_max_margin_error=restoration_error,
                predictions_sha256=digest(self.art / "predictions.json"),
                interventions_sha256=digest(output_path),
                elapsed_seconds=time.time() - self.started,
            ),
        )
        self.status("complete", interventions=len(predictions))

    def run(self):
        self.status("loaded")
        bank, last_bank, features = self.collect()
        ne = self.cfg["counts"]["E"]
        self.keep_features = torch.cat(features[ne:])
        directions, predictions = self.diagnose(bank, last_bank, features)
        self.intervene(bank, directions, predictions)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "run"])
    parser.add_argument("--layer", type=int, default=14)
    parser.add_argument("--format", choices=["qa", "plain"], default="qa")
    parser.add_argument("--device", default="cuda:2")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
        return
    lock = read(ART / "lock.json")
    assert digest(CONFIG) == lock["sources"]["configs/qwen-constraints-v1.json"]
    assert (
        digest(ART / "selection.json")
        == lock["inputs"][str((ART / "selection.json").relative_to(ROOT))]
    )
    gpu = int(args.device.split(":")[-1])
    if gpu in lock["config"]["excluded_gpus"]:
        raise ValueError("Excluded GPU")
    experiment = Experiment(args.layer, args.format, args.device)
    try:
        experiment.run()
    except Exception as exc:
        experiment.status("failed", error=repr(exc))
        raise


if __name__ == "__main__":
    main()
