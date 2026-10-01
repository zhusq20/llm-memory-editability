"""Independently rescore predictions, inspect paired inputs, and reload endpoints."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from llm_memory_editability.grok_depth import EpochStream, write_json
from llm_memory_editability.text_pretrain import construct, evaluate, sha

SPLITS = ("atomic", "background_train", "background_test", "target_train", "target_test")


def audit(config, device):
    cfg = json.loads(Path(config).read_text())
    lock = json.loads(Path(cfg["source_lock"]).read_text())
    assert lock["config_sha256"] == sha(config)
    for file, expected in lock["sources"].items():
        assert sha(file) == expected
    endpoints, trajectory, audits, initializations, pair_tables = [], [], [], {}, {}
    for name in cfg["runs"]:
        root = Path(cfg["output_root"]) / name
        complete = json.loads((root / "complete.json").read_text())
        spec = json.loads((root / "spec.json").read_text())
        history = json.loads((root / "learning.json").read_text())
        assert [r["step"] for r in history] == spec["nodes"]
        assert complete["final"] == history[-1]
        assert complete["world_sha256"] == sha(root / "world.npz")
        world = dict(np.load(root / "world.npz"))
        lookup = {(int(h), int(r)): int(t) for h, r, t in world["atomic"]}
        trainpairs, testpairs = set(), set()
        for split in SPLITS[1:]:
            for h, r1, b, r2, t in world[split]:
                assert lookup[h, r1] == b and lookup[b, r2] == t
                (testpairs if split.endswith("test") else trainpairs).add((h, t))
        assert not trainpairs & testpairs
        key = (spec["world"], spec["initialization"])
        initial = torch.load(root / "weights-000000.pt", map_location="cpu", weights_only=False)[
            "model"
        ]
        if key in initializations:
            for k, v in initial.items():
                assert torch.equal(v, initializations[key][k])
        else:
            initializations[key] = initial
        table = dict(np.load(root / "training-table.npz"))
        pair_tables.setdefault(key, {})[spec["arm"]] = table
        # Independently decode the role records instead of trusting the renderer.
        role_start = len(world["atomic"]) + len(world["background_train"])
        for packed, row in zip(table["tokens"][role_start:], world["target_train"], strict=True):
            h, r1, b, r2, t = map(int, row)
            assert packed[:16].tolist() == [2, h, 3, r1, 4, b, 5, 1, 2, b, 3, r2, 4, t, 5, 1]
            expected = (
                [2, h, 3, r1, 3, r2, 4, t, 5]
                if spec["arm"] == "P2"
                else [2, 6, 7, 8, 9, 10, 11, 12, 5]
            )
            assert packed[16:].tolist() == expected
        assert not np.triu(table["mask"], 1).any()
        assert not table["mask"][role_start:, :, 16:, :16].any()
        if spec["arm"] == "P0":
            assert not table["mask"][role_start:, :, 8:16, :8].any()
        else:
            assert table["mask"][role_start:, :, 8:16, :8].all()
        base, background, role = (
            len(world["atomic"]),
            len(world["background_train"]),
            len(world["target_train"]),
        )
        bounds = np.cumsum([0, base, background, role])
        saved_streams = torch.load(root / "latest.pt", map_location="cpu", weights_only=False)[
            "streams"
        ]
        exposures = []
        for group, size in enumerate((base, background, role)):
            stream = EpochStream(size, spec["world"] + 100 + group)
            count = np.zeros(size, dtype=np.int64)
            for _ in range(spec["steps"]):
                np.add.at(count, stream.take(spec["batch_parts"][group]), 1)
            assert count.max() - count.min() <= 1
            replay = stream.state_dict()
            assert replay["size"] == saved_streams[group]["size"]
            assert replay["rng"] == saved_streams[group]["rng"]
            np.testing.assert_array_equal(replay["remaining"], saved_streams[group]["remaining"])
            exposures.append(
                dict(
                    group=group,
                    rows=size,
                    total=int(count.sum()),
                    minimum=int(count.min()),
                    maximum=int(count.max()),
                )
            )
        for lo, hi, expected in zip(bounds[:-1], bounds[1:], (7, 9, 23), strict=True):
            assert np.all((table["labels"][lo:hi] != -100).sum(1) == expected)
        row_checks = 0
        for node in history:
            pred = np.load(root / f"predictions-{node['step']:06d}.npz")
            for split in SPLITS:
                target = world[split][:, -1]
                assert np.array_equal(pred[split + "_target"], target)
                answer = pred[split + "_answer"] == target
                stops = pred[split + "_stops"]
                full = answer & (stops[:, 0] == 5) & (stops[:, 1] == 1)
                metrics = dict(
                    n=len(target),
                    answer_accuracy=float(answer.mean()),
                    accuracy=float(full.mean()),
                    answer_probability=float(pred[split + "_probability"].mean()),
                )
                for k, value in metrics.items():
                    assert abs(node["metrics"][split][k] - value) < 1e-12
                row_checks += len(target)
                trajectory.append(
                    dict(
                        run=name,
                        world=spec["world"],
                        initialization=spec["initialization"],
                        arm=spec["arm"],
                        step=node["step"],
                        split=split,
                        **metrics,
                    )
                )
            atomic = pred["atomic_answer"] == world["atomic"][:, -1]
            known = {(h, r): ok for (h, r, _), ok in zip(world["atomic"], atomic, strict=True)}
            eligible = np.array(
                [known[h, r1] and known[b, r2] for h, r1, b, r2, _ in world["target_test"]]
            )
            condition = node["metrics"]["target_atomic_correct"]
            assert condition["n"] == int(eligible.sum())
            assert condition["coverage"] == float(eligible.mean())
            if eligible.any():
                assert condition["answer_accuracy"] == float(
                    (pred["target_test_answer"] == world["target_test"][:, -1])[eligible].mean()
                )
        model = construct(spec, device).eval()
        final_path = root / f"weights-{spec['steps']:06d}.pt"
        model.load_state_dict(
            torch.load(final_path, map_location=device, weights_only=False)["model"]
        )
        saved = np.load(root / f"predictions-{spec['steps']:06d}.npz")
        max_error = 0.0
        for split in SPLITS:
            _, actual = evaluate(model, world[split], device, composite=split != "atomic")
            for k in ("answer", "stops", "target"):
                assert np.array_equal(actual[k], saved[split + "_" + k])
            error = float(np.max(np.abs(actual["probability"] - saved[split + "_probability"])))
            assert error < 1e-5
            max_error = max(max_error, error)
        del model
        endpoints.append(
            dict(
                run=name,
                world=spec["world"],
                initialization=spec["initialization"],
                arm=spec["arm"],
                **complete["final"],
            )
        )
        audits.append(
            dict(
                run=name,
                raw_scoring_rows=row_checks,
                endpoint_reload_max_probability_error=max_error,
                exposures=exposures,
                checkpoint_sha256=sha(final_path),
            )
        )
        print(json.dumps(audits[-1]), flush=True)
    for arms in pair_tables.values():
        assert set(arms) == {"P0", "P1", "P2"}
        for field in ("tokens", "positions", "labels"):
            np.testing.assert_array_equal(arms["P0"][field], arms["P1"][field])
            np.testing.assert_array_equal(arms["P1"][field][:, :16], arms["P2"][field][:, :16])
        assert np.array_equal(arms["P1"]["mask"], arms["P2"]["mask"])
        assert not np.array_equal(arms["P0"]["mask"], arms["P1"]["mask"])
    out = Path(cfg["source_lock"]).parent
    write_json(
        out / "endpoint-audit.json",
        dict(
            status="passed",
            runs=len(audits),
            raw_scoring_rows=sum(x["raw_scoring_rows"] for x in audits),
            audits=audits,
            sources={__file__: sha(__file__)},
        ),
    )
    write_json(out / "endpoint-scores.json", endpoints)
    write_json(out / "trajectory-scores.json", trajectory)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/text-pretrain-confirmation-v1.json")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.set_device(args.device)
    torch.cuda.set_per_process_memory_fraction(0.08, args.device)
    audit(args.config, args.device)
