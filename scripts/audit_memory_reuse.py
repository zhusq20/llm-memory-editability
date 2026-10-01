"""Recompute stored answers and reload all saved reader nodes without training."""

from __future__ import annotations

import argparse
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from llm_memory_editability.memory_reuse import (
    TwoHopReader,
    evaluate,
    hash_state,
    new_memory,
    sha,
    write_json,
)


def independent_scores(logits, world, keys, name):
    predictions = np.argmax(logits, axis=2)
    table = world[f"mapping_{name}"]
    correct_one = predictions[:, 0] == table[keys]
    correct_two = predictions[:, 1] == table[table[keys]]
    train = world["train_mask"][keys]
    a, b = world["mapping_A"], world["mapping_B"]
    changed = a[a[keys]] != b[b[keys]]
    n_entities = len(world["train_mask"])
    both = []
    for i, key in enumerate(keys):
        repeat_start = (i // n_entities) * n_entities
        bridge = int(table[key])
        other = repeat_start + bridge
        assert keys[other] == bridge
        both.append(bool(correct_one[i] and correct_one[other]))
    both = np.array(both)

    def mean(values):
        return float(np.mean(values)) if len(values) else None

    return dict(
        n=len(keys),
        one_hop=mean(correct_one),
        two_hop=mean(correct_two),
        two_hop_A_training_heads=mean(correct_two[train]),
        two_hop_A_heldout_heads=mean(correct_two[~train]),
        two_hop_changed=mean(correct_two[changed]),
        changed_n=int(changed.sum()),
        two_hop_both_atoms_correct=mean(correct_two[both]),
        both_atoms_correct_n=int(both.sum()),
        both_atoms_correct_coverage=mean(both),
    )


def compare(actual, expected):
    assert actual.keys() == expected.keys()
    for key, value in actual.items():
        if isinstance(value, float):
            assert abs(value - expected[key]) < 1e-12, (key, value, expected[key])
        else:
            assert value == expected[key], (key, value, expected[key])


def audit(config_path, device):
    config = json.loads(Path(config_path).read_text())
    lock = json.loads(Path(config["lock"]).read_text())
    assert lock["config_sha256"] == sha(config_path)
    for name, digest in lock["files"].items():
        assert sha(name) == digest, name
    device = torch.device(device)
    torch.set_num_threads(config["spec"]["cpu_threads"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if device.type == "cuda":
        torch.cuda.set_device(device)
    runs, groups = [], {}
    for world_seed, init, arm in itertools.product(
        config["worlds"], config["initializations"], config["arms"]
    ):
        directory = Path(config["output_root"]) / f"w{world_seed}-i{init}-{arm}"
        result = json.loads((directory / "summary.json").read_text())
        assert result["status"] == "complete"
        assert result["spec"] == config["spec"]
        spec = result["spec"]
        world = dict(np.load(directory / "world.npz"))
        frozen_world = dict(np.load(Path(config["lock"]).parent / f"world-{world_seed}.npz"))
        assert all(np.array_equal(value, frozen_world[key]) for key, value in world.items())
        arrays = dict(np.load(directory / "endpoints.npz"))
        inputs, keys = arrays["inputs"], arrays["keys"]
        assert np.all((inputs[:, :-1] < spec["num_entities"]).sum(1) == 1)
        assert np.array_equal(inputs[inputs < spec["num_entities"]], keys)
        for name, labels in (("A", "A"), ("B", "B"), ("A", "B")):
            condition = "wrong_A_on_B" if name != labels else name
            compare(
                independent_scores(arrays[f"logits_{name}"], world, keys, labels),
                result["endpoints"][condition],
            )

        memory = new_memory(spec, init, device)
        assert hash_state(memory) == result["memories"]["A"]["initial_hash"]
        assert result["memories"]["A"]["initial_hash"] == result["memories"]["B"]["initial_hash"]
        reader = TwoHopReader(spec, world["embeddings"], memory, init, device)
        assert hash_state(reader, reader_only=True) == result["reader"]["initial_hash"]
        max_error, prediction_checks, parameter_checks = 0.0, 0, 0
        for name in ("A", "B"):
            checkpoint = torch.load(
                directory / f"reader-{name}.pt", map_location="cpu", weights_only=True
            )
            reader.load_state_dict(checkpoint["state_dict"])
            assert hash_state(reader, reader_only=True) == result["reader"]["final_hash"]
            assert hash_state(memory) == result["memories"][name]["final_hash"]
            parameter_checks += 2
            reloaded = evaluate(reader, inputs, device, spec["num_entities"])
            saved = arrays[f"logits_{name}"]
            error = float(np.abs(reloaded - saved).max())
            max_error = max(max_error, error)
            np.testing.assert_allclose(reloaded, saved, atol=1e-5, rtol=1e-5)
            assert np.array_equal(reloaded.argmax(-1), saved.argmax(-1))
            prediction_checks += reloaded.shape[0] * 2
            embeddings = torch.as_tensor(world["embeddings"], device=device)
            with torch.no_grad():
                out = memory(embeddings).cpu().numpy()
            np.testing.assert_allclose(out, arrays[f"memory_{name}_output"], atol=1e-5, rtol=1e-5)
            pred = (out @ world["embeddings"].T).argmax(-1)
            n = spec["num_entities"]
            expected = result["memories"][name]["metrics"]
            assert (
                float(np.mean(pred[:n] == world[f"mapping_{name}"][:n]))
                == expected["atomic_accuracy"]
            )
        curve = json.loads((directory / "reader-curve.json").read_text())
        assert [row["step"] for row in curve] == spec["reader_nodes"]
        for row in curve:
            step = row["step"]
            saved = np.load(directory / f"reader-node-{step:05d}.npz")["logits"]
            compare(
                independent_scores(saved, world, keys, "A"),
                {k: v for k, v in row.items() if k not in ("step", "seconds")},
            )
            reader.load_state_dict(
                torch.load(
                    directory / f"reader-node-{step:05d}.pt", map_location="cpu", weights_only=True
                )
            )
            reloaded = evaluate(reader, inputs, device, spec["num_entities"])
            max_error = max(max_error, float(np.abs(reloaded - saved).max()))
            np.testing.assert_allclose(reloaded, saved, atol=1e-5, rtol=1e-5)
            assert np.array_equal(reloaded.argmax(-1), saved.argmax(-1))
            assert hash_state(memory) == result["memories"]["A"]["final_hash"]
            prediction_checks += reloaded.shape[0] * 2
        groups.setdefault((world_seed, init), []).append(result)
        runs.append(
            dict(
                world=world_seed,
                init=init,
                arm=arm,
                nodes=len(curve),
                reload_predictions=prediction_checks,
                parameter_checks=parameter_checks,
                max_logit_error=max_error,
                summary_sha256=sha(directory / "summary.json"),
            )
        )
        print(json.dumps(runs[-1]), flush=True)
    for pair in groups.values():
        assert len(pair) == 2
        assert len({r["reader"]["initial_hash"] for r in pair}) == 1
        assert len({r["reader"]["stream_hash"] for r in pair}) == 1
        assert len({r["memories"]["A"]["initial_hash"] for r in pair}) == 1
    report = dict(
        status="passed",
        utc=datetime.now(timezone.utc).isoformat(),
        config_sha256=sha(config_path),
        source_lock_sha256=sha(config["lock"]),
        audit_source_sha256=sha(__file__),
        runs=runs,
        pairs=len(groups),
        total_reload_predictions=sum(r["reload_predictions"] for r in runs),
        max_logit_error=max(r["max_logit_error"] for r in runs),
    )
    write_json(Path(config["lock"]).parent / "audit.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    audit(args.config, args.device)
