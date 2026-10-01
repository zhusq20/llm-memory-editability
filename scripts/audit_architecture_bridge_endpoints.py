#!/usr/bin/env python3
"""Independent, bounded replay of frozen architecture-bridge artifacts.

Predetermined before replay: development case mquake/1121 for both models;
all 22 saved conditions plus direct/one-hop/oracle behavior; one cached versus
full teacher-forced score per model, using the corrupted prefix and correct
answer including EOS; four saved learning endpoints/model, E and D only.
Absolute numeric tolerance is 1e-4. Generated and target tokens must match
exactly. Failures are recorded without changing tolerances or adding experiments.
No fitting, gradients, optimizer, parameter training, U selection or model download.

Execution requires the frozen local environment, cuda:2, and completed main
matrix. --plan-only performs no model loading or GPU work. The main worker locks
are held during replay to prevent overlapping execution on the same GPU.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import fcntl
import gc
import hashlib
import importlib.util
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = Path("docs/development-artifacts/architecture-bridge-v1")
SOURCE = "src/llm_memory_editability/architecture_bridge.py"
CONFIG = "configs/architecture-bridge-v1.json"
TOLERANCE = 1e-4
DEVELOPMENT_ID = "1121"
DEVELOPMENT_DATASET = "mquake"
DEVICE = "cuda:2"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_hash(tensor):
    value = tensor.detach().contiguous().cpu()
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


def write(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def frozen_inputs(root):
    """Validate the archived files, never substitute working-tree source."""
    root = Path(root)
    artifact = root / ARTIFACT
    lock = read(artifact / "execution-lock.json")
    snapshot = artifact / "execution-source"
    for relative, expected in lock["files"].items():
        path = snapshot / relative
        if not path.is_file() or digest(path) != expected:
            raise ValueError("Frozen execution-source hash mismatch: " + relative)
    data_lock_path = artifact / "data-lock.json"
    if digest(data_lock_path) != lock["data_lock_sha256"]:
        raise ValueError("Frozen data-lock hash mismatch")
    data_lock = read(data_lock_path)
    config = read(snapshot / CONFIG)
    if digest(snapshot / CONFIG) != data_lock["config_sha256"]:
        raise ValueError("Archived config does not match data lock")
    data = root / "data/architecture-bridge-v1"
    for name in ("cases", "learning"):
        if digest(data / (name + ".json")) != data_lock[name + "_sha256"]:
            raise ValueError("Frozen data hash mismatch: " + name)
    cases, learning = read(data / "cases.json"), read(data / "learning.json")
    development = [case for case in cases if case["split"] == "development"]
    if not development or (development[0]["dataset"], development[0]["id"]) != (
        DEVELOPMENT_DATASET,
        DEVELOPMENT_ID,
    ):
        raise ValueError("The preregistered first development case changed")
    if development[0].get("exclusion_reason") is not None:
        raise ValueError("The preregistered development case is not eligible")
    return lock, snapshot, config, cases, learning, development[0]


def load_frozen_engine(root, snapshot):
    spec = importlib.util.spec_from_file_location(
        "frozen_architecture_bridge_endpoint", snapshot / SOURCE
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # The archived module's __file__ lives below execution-source; only these
    # data-location constants are rebound. Its implementation/config stay frozen.
    module.ROOT = Path(root)
    module.ART = Path(root) / ARTIFACT
    module.DATA = Path(root) / "data/architecture-bridge-v1"
    module.RESULTS = Path(root) / "results/architecture-bridge-v1"
    module.CONFIG = snapshot / CONFIG
    return module


def completion_missing(root, config, cases):
    missing = []
    for model in config["models"]:
        folder = Path(root) / "results/architecture-bridge-v1" / model
        for case in cases:
            path = folder / "cases" / f"{case['dataset']}-{case['id']}.json"
            if not path.is_file():
                missing.append(str(path))
        for episode in range(config["local_learning"]["episodes"]):
            for name in ("complete.json", "endpoint.pt"):
                path = folder / "learning" / str(episode) / name
                if not path.is_file():
                    missing.append(str(path))
    return missing


class Audit:
    def __init__(self):
        self.checks = []
        self.max_numeric_difference = 0.0

    def exact(self, name, actual, expected):
        passed = actual == expected
        self.checks.append(
            {
                "name": name,
                "kind": "exact",
                "passed": bool(passed),
                "actual": actual,
                "expected": expected,
            }
        )
        return bool(passed)

    def close(self, name, actual, expected):
        actual, expected = (
            np.asarray(actual, dtype=np.float64),
            np.asarray(expected, dtype=np.float64),
        )
        same_shape = actual.shape == expected.shape
        finite = np.all(np.isfinite(actual)) and np.all(np.isfinite(expected))
        error = (
            float(np.max(np.abs(actual - expected), initial=0.0)) if same_shape and finite else None
        )
        if error is not None:
            self.max_numeric_difference = max(self.max_numeric_difference, error)
        passed = error is not None and error <= TOLERANCE
        self.checks.append(
            {
                "name": name,
                "kind": "absolute_numeric",
                "passed": passed,
                "maximum_absolute_difference": error,
                "tolerance": TOLERANCE,
                "same_shape": same_shape,
            }
        )
        return passed


def compare_prediction(audit, label, actual, expected):
    audit.exact(label + ":generated_tokens", actual.get("tokens"), expected.get("tokens"))
    audit.exact(label + ":raw_text", actual.get("text"), expected.get("text"))
    for name in ("em", "f1", "alias_em", "ended_eos"):
        audit.exact(label + ":" + name, actual.get(name), expected.get(name))
    for role in ("correct", "competitor"):
        a, b = actual["scores"][role], expected["scores"][role]
        audit.exact(label + f":{role}:target_tokens", a["tokens"], b["tokens"])
        for name in ("sum_logp", "mean_logp"):
            audit.close(label + f":{role}:{name}", a[name], b[name])
    audit.close(label + ":margin", actual["scores"]["margin"], expected["scores"]["margin"])


def compare_case(audit, actual, expected, model, config):
    for field in ("id", "dataset", "split", "group", "exclusion_reason"):
        audit.exact(model + ":case_metadata:" + field, actual.get(field), expected.get(field))
    for name in ("unassisted", "first_hop", "second_hop", "oracle_input"):
        compare_prediction(audit, model + ":case:" + name, actual[name], expected[name])
    indexed = {(row["layer"], row["condition"]): row for row in actual["interventions"]}
    original = {(row["layer"], row["condition"]): row for row in expected["interventions"]}
    required = sorted(
        (layer, condition)
        for layer in config["models"][model]["layers"]
        for condition in config["conditions"]
    )
    audit.exact(model + ":case:observed_condition_keys", sorted(indexed), required)
    audit.exact(model + ":case:saved_condition_keys", sorted(original), required)
    audit.exact(
        model + ":case:observed_condition_count", len(actual["interventions"]), len(required)
    )
    audit.exact(
        model + ":case:saved_condition_count", len(expected["interventions"]), len(required)
    )
    for layer, condition in required:
        if (layer, condition) not in indexed or (layer, condition) not in original:
            continue
        compare_prediction(
            audit,
            f"{model}:case:L{layer}:{condition}",
            indexed[layer, condition],
            original[layer, condition],
        )


def teacher_forcing_inputs(prefix_ids, suffix_ids, target_ids):
    """Never re-encode concatenated text: preserve all real segment boundaries."""
    if any(
        value.ndim != 2 or value.shape[0] != 1 for value in (prefix_ids, suffix_ids, target_ids)
    ):
        raise ValueError("Token segments must have batch size one")
    if not prefix_ids.shape[1] or not suffix_ids.shape[1] or not target_ids.shape[1]:
        raise ValueError("Prefix, suffix, and EOS-inclusive target cannot be empty")
    inputs = torch.cat([prefix_ids, suffix_ids, target_ids[:, :-1]], dim=1)
    return inputs, target_ids.shape[1]


@torch.no_grad()
def cached_full_comparison(audit, engine, module, case):
    prefix_text = module.prefix(case, case["corrupted_bridge"])
    prefix_ids, suffix_ids = engine.ids(prefix_text), engine.ids(engine.cfg["suffix"])
    targets = engine.tok.encode(" " + case["answer"], add_special_tokens=False) + [engine.eos]
    target_ids = torch.tensor([targets], device=engine.device, dtype=torch.long)
    inputs, target_length = teacher_forcing_inputs(prefix_ids, suffix_ids, target_ids)
    cache, _, n_prefix = engine.prefill(prefix_text)
    total = n_prefix + suffix_ids.shape[1]
    output = engine.forward(suffix_ids, cache, total)
    frozen_score = engine.continuation_score(
        output.logits, copy.deepcopy(output.past_key_values), total, case["answer"]
    )
    cached_logp, logits, cache = [], output.logits, output.past_key_values
    for index, token in enumerate(targets):
        cached_logp.append(float(logits[0, -1].log_softmax(-1)[token]))
        if index + 1 < len(targets):
            total += 1
            output = engine.forward(torch.tensor([[token]], device=engine.device), cache, total)
            logits, cache = output.logits, output.past_key_values
    full_logits = engine.model(
        input_ids=inputs,
        attention_mask=torch.ones_like(inputs),
        use_cache=False,
        logits_to_keep=target_length,
    ).logits
    full_logp = (
        full_logits.log_softmax(-1)
        .gather(-1, target_ids.unsqueeze(-1))
        .squeeze(-1)[0]
        .cpu()
        .tolist()
    )
    label = engine.key + ":cached_full"
    audit.exact(label + ":EOS_target", targets[-1], engine.eos)
    audit.exact(label + ":frozen_target_tokens", frozen_score["tokens"], targets)
    audit.close(label + ":manual_vs_frozen_sum", sum(cached_logp), frozen_score["sum_logp"])
    audit.close(label + ":manual_vs_frozen_mean", np.mean(cached_logp), frozen_score["mean_logp"])
    audit.close(label + ":per_token_logp", cached_logp, full_logp)
    audit.close(label + ":sum_logp", sum(cached_logp), sum(full_logp))
    audit.close(label + ":mean_logp", np.mean(cached_logp), np.mean(full_logp))
    return {
        "prefix_tokens": prefix_ids[0].cpu().tolist(),
        "suffix_tokens": suffix_ids[0].cpu().tolist(),
        "target_tokens_including_eos": targets,
        "input_construction": "cat(prefix_ids, suffix_ids, target_ids_without_final_eos)",
        "cached_per_token_logp": cached_logp,
        "full_per_token_logp": full_logp,
        "frozen_cached_score": frozen_score,
    }


def parameter_signature(model, excluded):
    return {
        name: {
            "shape": list(parameter.shape),
            "dtype": str(parameter.dtype),
            "requires_grad": parameter.requires_grad,
            "storage_pointer": parameter.data_ptr(),
            "version": parameter._version,
        }
        for name, parameter in model.named_parameters()
        if parameter is not excluded
    }


def compare_parameter_signatures(audit, label, after, before):
    changed = sorted(
        name for name in set(before) | set(after) if before.get(name) != after.get(name)
    )
    audit.exact(label, changed, [])


@contextlib.contextmanager
def temporary_weight(parameter, replacement):
    """Restore byte-identical original weights even if replay raises."""
    if parameter.requires_grad:
        raise ValueError("Endpoint replay must not enable training")
    if parameter.shape != replacement.shape or parameter.dtype != replacement.dtype:
        raise ValueError("Endpoint weight shape/dtype mismatch")
    if not bool(torch.isfinite(replacement).all()):
        raise ValueError("Endpoint weight contains nonfinite values")
    original = parameter.detach().clone()
    try:
        with torch.no_grad():
            parameter.copy_(replacement)
        yield original
    finally:
        with torch.no_grad():
            parameter.copy_(original)


def endpoint_replay(audit, engine, config, pools, folder, episode):
    complete = read(folder / "complete.json")
    endpoint = torch.load(folder / "endpoint.pt", map_location="cpu", weights_only=True)
    target = pools["E"][episode]
    layer, steps = config["local_learning"]["layer"], config["local_learning"]["steps"]
    label = f"{engine.key}:endpoint:{episode}"
    for name, expected in [("target_id", target["id"]), ("layer", layer), ("steps", steps)]:
        if not audit.exact(label + ":metadata:" + name, endpoint.get(name), expected):
            return {"episode": episode, "status": "invalid_endpoint_metadata"}
    nodes = [node for node in complete["nodes"] if node["step"] == steps]
    if not audit.exact(label + ":final_node_count", len(nodes), 1):
        return {"episode": episode, "status": "invalid_final_node_count"}
    node = nodes[0]
    parameter = engine.layers[layer].mlp.down_proj.weight
    original_hash = tensor_hash(parameter)
    endpoint_weight = endpoint["down_weight"].to(engine.device)
    other_before = parameter_signature(engine.model, parameter)
    audit.exact(
        label + ":all_requires_grad_disabled",
        any(value.requires_grad for value in engine.model.parameters()),
        False,
    )
    replay = {}
    try:
        with temporary_weight(parameter, endpoint_weight) as original:
            audit.exact(
                label + ":loaded_weight_hash", tensor_hash(parameter), tensor_hash(endpoint_weight)
            )
            audit.close(
                label + ":parameter_delta_norm",
                float((parameter - original).norm()),
                node["parameter_delta_norm"],
            )
            replay["E"] = engine.direct(target, target["q1"], target["bridge"])
            replay["D"] = engine.direct(target)
            for role in ("E", "D"):
                compare_prediction(audit, label + ":" + role, replay[role], node[role])
    finally:
        audit.exact(label + ":original_weight_restored", tensor_hash(parameter), original_hash)
        compare_parameter_signatures(
            audit,
            label + ":other_parameter_signatures_unchanged",
            parameter_signature(engine.model, parameter),
            other_before,
        )
        audit.exact(
            label + ":gradients_remain_disabled",
            any(value.requires_grad for value in engine.model.parameters()),
            False,
        )
    return {
        "episode": episode,
        "target_id": target["id"],
        "endpoint_sha256": digest(folder / "endpoint.pt"),
        "complete_sha256": digest(folder / "complete.json"),
        "original_down_sha256": original_hash,
        "loaded_down_sha256": tensor_hash(endpoint_weight),
        "replay": replay,
    }


def replay(root, progress=None):
    root = Path(root)
    lock, snapshot, config, cases, pools, development = frozen_inputs(root)
    missing = completion_missing(root, config, cases)
    if missing:
        raise ValueError(
            f"Main matrix is incomplete ({len(missing)} missing files); no GPU audit started"
        )
    expected_environments = {
        (root / "data/architecture-bridge-v1/environment").resolve(),
        Path("/home/siqizhu4/.local/share/llm-memory/architecture-bridge-v1/environment").resolve(),
    }
    if Path(sys.prefix).resolve() not in expected_environments:
        raise ValueError("Use the frozen architecture-bridge-v1 environment or its local copy")
    import transformers

    if torch.__version__ != lock["torch"] or transformers.__version__ != lock["transformers"]:
        raise ValueError("Audit runtime does not match the frozen execution environment")
    module = load_frozen_engine(root, snapshot)
    audit, model_results = Audit(), {}
    if progress is not None:
        progress.update({"checks": audit.checks, "models": model_results})
    with contextlib.ExitStack() as stack:
        for model in config["models"]:
            guard = stack.enter_context(
                (root / "results/architecture-bridge-v1" / model / "worker.lock").open("a")
            )
            fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for model in config["models"]:
            engine = module.Engine(model, DEVICE)
            audit.exact(model + ":device", str(engine.device), DEVICE)
            audit.exact(
                model + ":floating_parameter_dtypes",
                sorted(
                    {
                        str(parameter.dtype)
                        for parameter in engine.model.parameters()
                        if parameter.is_floating_point()
                    }
                ),
                ["torch.float32"],
            )
            audit.exact(
                model + ":all_gradients_disabled",
                any(parameter.requires_grad for parameter in engine.model.parameters()),
                False,
            )
            parameter = engine.layers[config["local_learning"]["layer"]].mlp.down_proj.weight
            original_hash = tensor_hash(parameter)
            unedited_signatures = parameter_signature(engine.model, parameter)
            expected_path = (
                root
                / "results/architecture-bridge-v1"
                / model
                / "cases"
                / f"{DEVELOPMENT_DATASET}-{DEVELOPMENT_ID}.json"
            )
            case_result = engine.case(development)
            compare_case(audit, case_result, read(expected_path), model, config)
            full_comparison = cached_full_comparison(audit, engine, module, development)
            model_results[model] = {
                "case_file_sha256": digest(expected_path),
                "case_replay": case_result,
                "cached_full_comparison": full_comparison,
                "learning_endpoints": [],
            }
            for episode in range(config["local_learning"]["episodes"]):
                folder = root / "results/architecture-bridge-v1" / model / "learning" / str(episode)
                model_results[model]["learning_endpoints"].append(
                    endpoint_replay(audit, engine, config, pools, folder, episode)
                )
            audit.exact(model + ":final_original_down_hash", tensor_hash(parameter), original_hash)
            compare_parameter_signatures(
                audit,
                model + ":final_other_parameters_unchanged",
                parameter_signature(engine.model, parameter),
                unedited_signatures,
            )
            del parameter, engine
            gc.collect()
            torch.cuda.empty_cache()
    failures = [check for check in audit.checks if not check["passed"]]
    return {
        "status": "complete" if not failures else "failed",
        "complete": not failures,
        "tolerance_absolute": TOLERANCE,
        "tokens_required_exact": True,
        "coverage": {
            "development_case": {"dataset": DEVELOPMENT_DATASET, "id": DEVELOPMENT_ID},
            "models": list(config["models"]),
            "case_condition_rows": sum(
                len(result["case_replay"]["interventions"]) for result in model_results.values()
            ),
            "cached_full_scores": len(model_results),
            "learning_endpoints": sum(
                len(result["learning_endpoints"]) for result in model_results.values()
            ),
            "learning_roles": ["E", "D"],
        },
        "maximum_numeric_difference": audit.max_numeric_difference,
        "checks": audit.checks,
        "failures": failures,
        "models": model_results,
        "parameter_scope": (
            "Only down matrix copied; original hash restored after each endpoint. "
            "Other parameters checked by requires_grad, storage identity and version "
            "counters, not full tensor hashes."
        ),
        "not_established": [
            "full model checksum of every historical training step",
            "R/U regeneration at saved endpoints",
            "generalization beyond the preregistered checks",
        ],
        "provenance": {
            "audit_source_sha256": digest(Path(__file__)),
            "execution_lock_sha256": digest(root / ARTIFACT / "execution-lock.json"),
            "data_lock_sha256": digest(root / ARTIFACT / "data-lock.json"),
            "frozen_source_sha256": digest(snapshot / SOURCE),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "python_executable": sys.executable,
            "device": DEVICE,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if args.plan_only:
        _, _, config, cases, _, _ = frozen_inputs(args.root)
        print(
            json.dumps(
                {
                    "mode": "plan_only",
                    "gpu_work": False,
                    "tolerance_absolute": TOLERANCE,
                    "case": [DEVELOPMENT_DATASET, DEVELOPMENT_ID],
                    "models": list(config["models"]),
                    "missing_main_files": len(completion_missing(args.root, config, cases)),
                }
            )
        )
        return
    started = time.monotonic()
    progress = {}
    try:
        result = replay(args.root, progress)
    except Exception:
        result = {
            **progress,
            "status": "failed",
            "complete": False,
            "tolerance_absolute": TOLERANCE,
            "tokens_required_exact": True,
            "exception": traceback.format_exc(),
            "policy": "No tolerance change or additional experiments after failure.",
        }
        numeric_checks = [
            check["maximum_absolute_difference"]
            for check in result.get("checks", [])
            if check.get("maximum_absolute_difference") is not None
        ]
        result["maximum_numeric_difference"] = max(numeric_checks, default=None)
        result["coverage"] = {
            "completed_models": list(result.get("models", {})),
            "learning_endpoints": sum(
                len(model.get("learning_endpoints", []))
                for model in result.get("models", {}).values()
            ),
            "remaining_replay_unverified": True,
        }
    result["created_utc"] = datetime.now(timezone.utc).isoformat()
    result["wall_seconds"] = time.monotonic() - started
    write(args.root / ARTIFACT / "endpoint-audit.json", result)
    print(
        json.dumps(
            {
                key: result.get(key)
                for key in (
                    "status",
                    "complete",
                    "coverage",
                    "maximum_numeric_difference",
                    "wall_seconds",
                )
            }
        )
    )
    if not result["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
