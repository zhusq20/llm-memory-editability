"""Fixed-weight recurrence intervention for both entity representations."""

import argparse
import json
import os

import diagnose_realworld_entities as entity
import torch

common = entity.common
ROOT = common.PROJECT / "results/realworld-loop-entity-depth-v1"


def main(condition, gpu):
    device = common.configure(gpu)
    data, manifest = entity.load(condition)
    source = entity.ROOT / "development" / f"loop4x2-{condition}"
    assert (source / "complete.json").exists()
    model = common.construct(manifest["model"], entity.spec_for("loop4x2"), device)
    model.load_state_dict(
        torch.load(source / "latest.pt", map_location="cpu", weights_only=False)["model"]
    )
    digest = common.model_digest(model)
    tokenizer = entity.NodeCodec()
    for repeats in [2, 1, 4]:
        model.repeats = repeats
        metrics, predictions = common.evaluate_selected(model, data, [], tokenizer, device, True)
        if repeats == 2:
            assert metrics == json.loads((source / "endpoint.json").read_text())
            assert predictions == json.loads((source / "endpoint-predictions.json").read_text())
        assert common.model_digest(model) == digest
        out = ROOT / "development" / f"{condition}-r{repeats}"
        common.write_json(
            out / "run.json",
            {
                "spec": {"condition": condition, "repeats": repeats, "steps": 0},
                "pid": os.getpid(),
                "gpu": gpu,
                "source": str(source),
                "tracking_group": "realworld-loop-diagnosis-v1",
                "job_type": "entity-depth-evaluation",
                "model_sha256": digest,
            },
        )
        common.write_json(out / "learning.json", [{"step": 0, "metrics": metrics}])
        common.write_json(out / "endpoint.json", metrics)
        common.write_json(out / "endpoint-predictions.json", predictions)
        common.write_json(
            out / "complete.json",
            {
                "passed": True,
                "unchanged_parameters": True,
                "trained_recurrence_exact_control_passed": True,
            },
        )
        print(
            json.dumps(
                {
                    "condition": condition,
                    "repeats": repeats,
                    "metrics": {
                        s: metrics[s]["alias_em"] for s in ["atomic", "test_ii", "test_oo"]
                    },
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=["natural", "entity"], required=True)
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()
    main(args.condition, args.gpu)
