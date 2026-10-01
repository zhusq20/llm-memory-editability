"""Fixed full-target and relation-support breakdown, separate from main scores."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from llm_memory_editability.grok_depth import utc, write_json
from llm_memory_editability.text_pretrain import construct, evaluate, sha
from llm_memory_editability.text_structure import TESTS, score_predictions, two_calls, verify_lock


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["freeze", "analyze"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    output = Path(cfg["source_lock"]).parent
    lock_path = output / "secondary-analysis-lock.json"
    script = "scripts/analyze_text_structure.py"
    if args.command == "freeze":
        if lock_path.exists():
            raise FileExistsError(lock_path)
        write_json(
            lock_path,
            dict(
                created_utc=utc(),
                sources={script: sha(script)},
                config_sha256=sha(args.config),
                definitions=dict(
                    full_target=(
                        "All 512 target paths; positive arm includes supervised paths; "
                        "main arms have zero target combination exposure"
                    ),
                    relation_support=(
                        "Fixed restricted support: r1<15 with even-index r2, "
                        "r1>=15 with odd-index r2; identical strata across arms"
                    ),
                    status=(
                        "Secondary analysis added during development after early-node reads; "
                        "confirmation analysis fixed before confirmation; "
                        "primary endpoint remains common strict_test"
                    ),
                ),
            ),
        )
        return
    lock = json.loads(lock_path.read_text())
    assert sha(script) == lock["sources"][script]
    assert sha(args.config) == lock["config_sha256"]
    verify_lock(args.config, cfg["source_lock"])
    torch.set_num_threads(2)
    results = []
    for name in cfg["runs"]:
        root = Path(cfg["output_root"]) / name
        spec = json.loads((root / "spec.json").read_text())
        saved = torch.load(
            root / f"weights-{spec['steps']:06d}.pt", map_location=args.device, weights_only=False
        )
        model = construct(spec, args.device).eval()
        model.load_state_dict(saved["model"])
        world = dict(np.load(root / "world.npz"))
        metrics, pred = evaluate(model, world["strict_all"], args.device, composite=True)
        calls, cp = two_calls(model, world["strict_all"], args.device)
        np.savez_compressed(
            output / f"{name}-full-target.npz",
            rows=world["strict_all"],
            **pred,
            **{"two_calls_" + k: v for k, v in cp.items()},
        )
        stored = dict(np.load(root / f"predictions-{spec['steps']:06d}.npz"))
        groups = {}
        for split in TESTS:
            rows = world[split]
            restricted_support = (rows[:, 1] < 15) == ((rows[:, 3] - 17) % 2 == 0)
            for label, mask in (
                ("restricted_supported", restricted_support),
                ("restricted_unsupported", ~restricted_support),
            ):
                values = {
                    k: stored[split + "_" + k][mask]
                    for k in ("answer", "target", "stops", "probability")
                }
                groups[split + "_" + label] = score_predictions(values) if mask.any() else dict(n=0)
        results.append(
            dict(
                run=name,
                world=spec["world"],
                initialization=spec["initialization"],
                arm=spec["arm"],
                full_target=metrics,
                full_target_two_calls=calls,
                strict_test_coverage=len(world["strict_test"]) / 512,
                relation_support=groups,
                checkpoint_sha256=sha(root / f"weights-{spec['steps']:06d}.pt"),
            )
        )
    write_json(
        output / "secondary-analysis.json",
        dict(status="complete", utc=utc(), lock_sha256=sha(lock_path), runs=results),
    )


if __name__ == "__main__":
    main()
