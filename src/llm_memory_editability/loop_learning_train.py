"""Paired Loop learning with full old-distribution replay and independent audit.

The scoped adapter reuses the audited sequential trainer without changing any
historical implementation. Each experiment runs in its own Python process.
"""

from __future__ import annotations

import argparse
import json
from contextlib import contextmanager
from pathlib import Path

from . import sequential_transfer as st
from .loop_learning import architecture_manifest, construct, execution_digest
from .realworld_composition_data import write_json

_ORIGINAL_PLAN = st.training_plan


def training_plan(data, spec):
    """Preserve the new-fact stream; replay the actual first-stage stream."""
    if spec["history"] != "sequential_composition":
        raise ValueError("This first Loop matrix requires sequential_composition")
    if spec.get("replay_source") != "full_stage_a":
        raise ValueError("Declare full_stage_a replay explicitly")
    records, plan = _ORIGINAL_PLAN(data, spec)
    sa, sb, batch = (spec[k] for k in ("stage_a_steps", "stage_b_steps", "batch_size"))
    new = spec.get("stage_b_new_per_batch", batch // 2)
    old = batch - new
    if batch % 2 or sb % 2 or old != batch // 2:
        raise ValueError("Full replay requires an even batch, even B steps and half new facts")
    if spec.get("stage_a_composition_per_batch", batch // 2) != batch // 2:
        raise ValueError("First stage must balance old atoms and old compositions")
    stream = plan[:sa].reshape(-1)
    needed = sb * old
    if needed > len(stream):
        raise ValueError("Replay cannot exceed the actual first-stage stream")
    plan[sa:, :old] = stream[:needed].reshape(sb, old)
    return records, plan


def diagnostic_records(data, count=2):
    """Fixed, training-only records; no held-out composition labels enter this."""
    return {
        "old_atoms": [r for r in data["atoms"] if r["subset"] == "A"][:count],
        "new_atoms": [r for r in data["atoms"] if r["subset"] == "B"][:count],
        "old_compositions": data["train_compositions"][:count],
    }


@contextmanager
def adapter(spec, out, diagnostics=False):
    """Install process-local hooks and always restore the original functions."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    originals = {
        key: getattr(st, key) for key in ("construct", "training_plan", "save_checkpoint", "update")
    }
    nodes = set(spec.get("diagnostic_nodes", [])) if diagnostics else set()

    def build(_config, current_spec, device):
        model = construct(current_spec, device)
        target = out / "architecture.json"
        if not target.exists():
            write_json(
                target,
                {
                    **architecture_manifest(model),
                    "initial_execution_sha256": execution_digest(model),
                },
            )
        return model

    def save(path, model, optimizer, step, *args):
        originals["save_checkpoint"](path, model, optimizer, step, *args)
        target = out / f"diagnostics-{step:07d}.json"
        if step in nodes and not target.exists():
            from .loop_learning_diagnostics import diagnose

            data = json.loads(Path(spec["data_file"]).read_text())
            from transformers import GPT2TokenizerFast

            tokenizer = GPT2TokenizerFast.from_pretrained(spec["tokenizer"])
            device = next(model.parameters()).device
            batches = diagnostic_records(data, spec.get("diagnostic_examples_per_role", 2))
            results = {
                role: diagnose(model, records, spec, device, tokenizer.eos_token_id)
                for role, records in batches.items()
            }
            write_json(
                target,
                {
                    "step": step,
                    "scope": "fixed training records; eval-mode local gradients",
                    "record_ids": {k: [r["id"] for r in v] for k, v in batches.items()},
                    "roles": results,
                },
            )

    def update(model, optimizer, records, current_spec, device, pad):
        first = next(model.parameters())
        state_step = optimizer.state.get(first, {}).get("step", 0)
        step = int(state_step) + 1
        selected = step in nodes or step - 1 in nodes
        before = (
            {
                name: p.detach().clone()
                for name, p in model.named_parameters()
                if name.startswith("transformer.h.")
            }
            if selected
            else {}
        )
        result = originals["update"](model, optimizer, records, current_spec, device, pad)
        if selected:
            squares = {}
            for name, parameter in model.named_parameters():
                if name in before:
                    family = ".".join(name.split(".")[:4])
                    value = (parameter.detach() - before[name]).double().square().sum().item()
                    squares[family] = squares.get(family, 0.0) + value
            write_json(
                out / f"actual-update-{step:07d}.json",
                {
                    "step": step,
                    "gradient_norm_before_clipping": result["gradient_norm"],
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "unique_block_update_l2": {k: v**0.5 for k, v in squares.items()},
                },
            )
        return result

    st.construct, st.training_plan = build, training_plan
    st.save_checkpoint, st.update = save, update
    try:
        yield
    finally:
        for key, value in originals.items():
            setattr(st, key, value)


def run(spec, out, device):
    with adapter(spec, out, diagnostics=True):
        return st.run(spec, out, device)


def audit(out, device):
    spec = json.loads((Path(out) / "run.json").read_text())["spec"]
    with adapter(spec, out):
        return st.audit(out, device)


def preflight(spec, out, device):
    with adapter(spec, out):
        return st.preflight(spec, out, device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--run")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    if args.audit:
        result = audit(args.out, args.device)
    else:
        config = json.loads(args.config.read_text())
        spec = next(row for row in config["specs"] if row["name"] == args.run)
        result = (preflight if args.preflight else run)(spec, args.out, args.device)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
