"""Controlled nonlinear memory editing with full old-fact replay on CPU.

This measures finite-budget optimization, not minimum representational capacity.
The data generator's shared rule does not establish the model's internal mechanism.
"""

import argparse
import copy
import json
import math
import platform
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ModuleNotFoundError as exc:
    raise SystemExit("Neural experiments require PyTorch: pip install '.[neural]'") from exc

from .neural_data import make_world
from .neural_models import build_model, edit_parameters


@dataclass(frozen=True)
class NeuralConfig:
    model: str = "mlp"
    seeds: tuple[int, ...] = (0, 1, 2)
    data_seed: int = 0
    entities: int = 24
    attributes: int = 8
    groups: int = 3
    answers: int = 4
    width: int = 32
    hidden: int = 64
    layers: int = 2
    heads: int = 4
    pretrain_steps: int = 400
    edit_steps: int = 100
    scratch_steps: int = 400
    lr: float = 0.01
    retention_weight: float = 1.0
    edit_scope: str = "ffn"
    eval_every: int = 10
    success_threshold: float = 0.95
    threads: int = 1

    def validate(self) -> None:
        if self.model not in {"mlp", "transformer"} or self.edit_scope not in {"ffn", "all"}:
            raise ValueError("unsupported model or edit scope")
        for name in ("pretrain_steps", "edit_steps", "scratch_steps", "eval_every", "threads"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be nonempty and unique")
        if any(isinstance(s, bool) or not isinstance(s, int) or s < 0 for s in self.seeds):
            raise ValueError("seeds must be nonnegative integers")
        if not math.isfinite(self.lr) or self.lr <= 0:
            raise ValueError("lr must be finite and positive")
        if not math.isfinite(self.retention_weight) or self.retention_weight < 0:
            raise ValueError("retention_weight must be finite and nonnegative")
        if not 0 < self.success_threshold <= 1:
            raise ValueError("success_threshold must be in (0, 1]")


@torch.no_grad()
def evaluate(model: nn.Module, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> dict:
    """Evaluate answers against a target world, separating edits and retained facts."""
    model.eval()
    logits = model(x)
    if not torch.isfinite(logits).all():
        raise ValueError("non-finite model outputs")
    correct = logits.argmax(dim=-1).eq(y).float()
    target_logits = logits.gather(1, y[:, None]).squeeze(1)
    competitors = logits.clone().scatter_(1, y[:, None], float("-inf")).max(dim=1).values
    margins = target_logits - competitors
    return {
        "all_accuracy": correct.mean().item(),
        "edit_accuracy": correct[mask].mean().item(),
        "retain_accuracy": correct[~mask].mean().item(),
        "edit_target_nll": F.cross_entropy(logits[mask], y[mask]).item(),
        "edit_margin": margins[mask].mean().item(),
        "retain_margin": margins[~mask].mean().item(),
    }


def fit(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
    *,
    steps: int,
    config: NeuralConfig,
    is_edit: bool = False,
) -> dict:
    """Use a fresh optimizer, fixed steps, and a documented full-batch objective."""
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=config.lr)
    history = []
    first_success = None
    started = time.perf_counter()
    for step in range(steps + 1):
        if step % config.eval_every == 0 or step == steps:
            metrics = evaluate(model, x, y, mask)
            history.append({"step": step, **metrics})
            if (
                first_success is None
                and min(metrics["edit_accuracy"], metrics["retain_accuracy"])
                >= config.success_threshold
            ):
                first_success = step
        if step == steps:
            break
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        if is_edit:
            loss = F.cross_entropy(logits[mask], y[mask])
            if config.retention_weight:
                loss = loss + config.retention_weight * F.cross_entropy(logits[~mask], y[~mask])
        else:
            loss = F.cross_entropy(logits, y)
        if not torch.isfinite(loss):
            raise ValueError("training diverged to a non-finite loss")
        loss.backward()
        optimizer.step()
    return {
        "steps": steps,
        "trainable_parameters": sum(p.numel() for p in parameters),
        "first_success_step": first_success,
        "elapsed_seconds": time.perf_counter() - started,
        "history": history,
        "objective": "edit_ce + retention_weight * retained_ce" if is_edit else "full_world_ce",
    }


def _summary(runs: list[dict]) -> dict:
    eligible = [run for run in runs if run["baseline_quality_passed"]]
    result = {"eligible_seeds": [run["seed"] for run in eligible], "cases": {}}
    for name in ("rule", "exception"):
        result["cases"][name] = {}
        for metric in ("edit_accuracy", "retain_accuracy"):
            values = [run["cases"][name]["after"][metric] for run in eligible]
            result["cases"][name][metric] = {
                "mean": statistics.mean(values) if values else None,
                "sample_std": statistics.stdev(values) if len(values) > 1 else None,
                "n": len(values),
            }
    return result


def _revision() -> str | None:
    """Best-effort source revision; an installed wheel may not have a checkout."""
    source_root = Path(__file__).resolve().parents[2]
    if not (source_root / ".git").exists():
        return None
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=source_root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=source_root, text=True
        ).strip()
        return revision + ("+dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return None


def run_experiment(config: NeuralConfig) -> dict:
    config.validate()
    world = make_world(
        n_entities=config.entities,
        n_attributes=config.attributes,
        n_groups=config.groups,
        n_answers=config.answers,
        seed=config.data_seed,
    )
    torch.set_num_threads(config.threads)
    torch.use_deterministic_algorithms(True)
    x = torch.as_tensor(world.x, dtype=torch.long)
    original = torch.as_tensor(world.original, dtype=torch.long)
    mask = torch.as_tensor(world.edit_mask, dtype=torch.bool)
    runs = []
    for seed in config.seeds:
        torch.manual_seed(seed)
        initial = build_model(
            config.model,
            config.entities,
            config.attributes,
            config.answers,
            width=config.width,
            hidden=config.hidden,
            layers=config.layers,
            heads=config.heads,
        )
        base = copy.deepcopy(initial)
        baseline_training = fit(
            base,
            x,
            original,
            mask,
            steps=config.pretrain_steps,
            config=config,
        )
        baseline = evaluate(base, x, original, mask)
        run = {
            "seed": seed,
            "parameter_count": sum(p.numel() for p in base.parameters()),
            "baseline": baseline,
            "baseline_quality_passed": min(baseline["edit_accuracy"], baseline["retain_accuracy"])
            >= config.success_threshold,
            "baseline_training": baseline_training,
            "cases": {},
        }
        for name in ("rule", "exception"):
            target = torch.as_tensor(getattr(world, name), dtype=torch.long)
            edited = copy.deepcopy(base)
            edit_parameters(edited, scope=config.edit_scope)
            before = evaluate(edited, x, target, mask)
            edit_training = fit(
                edited,
                x,
                target,
                mask,
                steps=config.edit_steps,
                config=config,
                is_edit=True,
            )
            scratch = copy.deepcopy(initial)
            scratch_training = fit(
                scratch,
                x,
                target,
                mask,
                steps=config.scratch_steps,
                config=config,
            )
            run["cases"][name] = {
                "before": before,
                "after": evaluate(edited, x, target, mask),
                "scratch": evaluate(scratch, x, target, mask),
                "edit_training": edit_training,
                "scratch_training": scratch_training,
            }
        runs.append(run)
    return {
        "experiment": "nonlinear_synthetic_memory_editing",
        "config": asdict(config),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
        "revision": _revision(),
        "device": "cpu",
        "world": world.to_metadata(),
        "protocol": {
            "paired_edit_initialization": "identical trained checkpoint; fresh Adam per case",
            "paired_scratch_initialization": "identical untrained initialization; all parameters",
            "replay": "all retained facts available; uniform CE within each edit/retain subset",
            "success": "both edit and retain accuracy >= threshold at evaluation checkpoints",
            "summary_filter": "only seeds passing baseline threshold on both subsets",
            "limits": [
                "Target surprisal is measured, not matched.",
                "Generator structure is not a measure of learned neural degrees of freedom.",
                "Scratch failure at finite budget is not an impossibility result.",
                "Scratch trains all parameters; FFN-only edit failures may reflect "
                "a restricted update space.",
                "No natural-language paraphrases or pretrained LLM are evaluated.",
                "Cross-platform and cross-version bitwise reproducibility is not guaranteed.",
            ],
        },
        "runs": runs,
        "summary": _summary(runs),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("mlp", "transformer"), default="mlp")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--edit-scope", choices=("ffn", "all"), default="ffn")
    defaults = asdict(NeuralConfig())
    for name in (
        "data_seed",
        "entities",
        "attributes",
        "groups",
        "answers",
        "width",
        "hidden",
        "layers",
        "heads",
        "pretrain_steps",
        "edit_steps",
        "scratch_steps",
        "eval_every",
        "threads",
    ):
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=defaults[name])
    for name in ("lr", "retention_weight", "success_threshold"):
        parser.add_argument("--" + name.replace("_", "-"), type=float, default=defaults[name])
    parser.add_argument("--output", type=Path, help="save the same JSON printed to stdout")
    args = vars(parser.parse_args(argv))
    output = args.pop("output")
    args["seeds"] = tuple(args["seeds"])
    try:
        report = run_experiment(NeuralConfig(**args))
        serialized = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(serialized + "\n", encoding="utf-8")
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
