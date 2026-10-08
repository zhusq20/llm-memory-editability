"""Three-model comparison using the frozen Loop data and optimizer contract."""

from __future__ import annotations

import argparse
import json
from contextlib import contextmanager
from pathlib import Path

from . import sequential_transfer as st
from .loop_kv_learning import architecture_manifest, construct, extra_attention_flops
from .loop_learning import execution_digest
from .loop_learning_train import training_plan
from .realworld_composition_data import write_json


@contextmanager
def adapter(spec, out):
    originals = {key: getattr(st, key) for key in ("construct", "training_plan", "update")}
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

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

    def update(model, optimizer, records, current_spec, device, pad):
        result = originals["update"](model, optimizer, records, current_spec, device, pad)
        result["estimated_matmul_training_flops"] += extra_attention_flops(
            model, records, current_spec["microbatch_size"]
        )
        return result

    st.construct, st.training_plan, st.update = build, training_plan, update
    try:
        yield
    finally:
        for key, value in originals.items():
            setattr(st, key, value)


def run(spec, out, device):
    with adapter(spec, out):
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
