"""Count computation in archived prompt-only multi-hop answers; no training."""

import argparse
import hashlib
import importlib
import json
import sys
import types
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def frozen_module(root, sources):
    root = Path(root)
    for relative, expected in sources.items():
        assert sha(root / relative) == expected, relative
    for name in list(sys.modules):
        if name == "llm_memory_editability" or name.startswith("llm_memory_editability."):
            del sys.modules[name]
    package = types.ModuleType("llm_memory_editability")
    package.__path__ = [str(root / "src/llm_memory_editability")]
    sys.modules[package.__name__] = package
    module = importlib.import_module("llm_memory_editability.depth_step")
    for name, loaded in list(sys.modules.items()):
        if name.startswith(package.__name__ + ".") and getattr(loaded, "__file__", None):
            path = Path(loaded.__file__)
            assert sha(path) == sources[str(path.relative_to(root))], name
    return module


@torch.inference_mode()
def answers(model, rows, separator, batch_size=256):
    """Hook primitive modules, because LoopGPT does not call Block.forward."""
    assert not model.training
    events, handles, expected = [], [], ["model", "token", "position"]
    primitives = {"model": model, "token": model.token, "position": model.position}
    for i, block in enumerate(model.blocks):
        for label, primitive in (
            ("ln1", block.ln1),
            ("qkv", block.attention.qkv),
            ("proj", block.attention.proj),
            ("ln2", block.ln2),
            ("mlp", block.mlp),
            ("up", block.mlp.up),
            ("down", block.mlp.down),
        ):
            primitives[f"{i}.{label}"] = primitive
    for _ in range(model.repeats):
        for i in range(len(model.blocks)):
            expected.extend(f"{i}.{s}" for s in ("ln1", "qkv", "proj", "ln2", "mlp", "up", "down"))
    primitives["ln_final"] = model.ln_final
    expected.append("ln_final")
    for name, primitive in primitives.items():
        handles.append(
            primitive.register_forward_pre_hook(lambda _m, _x, label=name: events.append(label))
        )
    output, traces = [], []
    try:
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            # The final truth column is excluded. No generated token is fed back.
            prompt = np.column_stack(
                (np.full(len(batch), 2), batch[:, :-1], np.full(len(batch), separator))
            )
            events.clear()
            logits = model(torch.as_tensor(prompt, dtype=torch.long))[:, -1]
            assert events == expected, events
            output.append(logits.argmax(-1).cpu().numpy())
            traces.append({"n": len(batch), "prompt_length": prompt.shape[1], "calls": events[:]})
    finally:
        for handle in handles:
            handle.remove()
    return np.concatenate(output), traces


def guesses(train, rows):
    """Frequency tables use training labels only; never use test labels."""

    def mode(values):
        c = Counter(map(int, values))
        return min(c, key=lambda x: (-c[x], x))

    fallback = mode(train[:, -1])
    result = {"constant": np.full(len(rows), fallback)}
    for name, key in (
        ("head_only", lambda row: (int(row[0]),)),
        ("last_relation", lambda row: (int(row[-2]),)),
        ("last_two_relations", lambda row: tuple(map(int, row[-3:-1]))),
        ("relations_without_head", lambda row: tuple(map(int, row[1:-1]))),
    ):
        groups = defaultdict(list)
        for row in train:
            groups[key(row)].append(row[-1])
        table = {k: mode(v) for k, v in groups.items()}
        result[name] = np.array([table.get(key(row), fallback) for row in rows])
    return result


def audit(entry, out, agreement_floor=0.95):
    run = Path(entry["run_root"])
    original = json.loads((run / "complete.json").read_text())
    spec = original["spec"]
    assert json.loads((run / "audit.json").read_text())["passed"]
    assert [spec["layers"], spec["repeats"]] == entry["architecture"]
    module = frozen_module(entry["frozen_source"], original["source"])
    assert sha(run / "latest.pt") == original["checkpoint_sha256"]
    checkpoint = torch.load(run / "latest.pt", map_location="cpu", weights_only=False)
    assert checkpoint["spec"] == spec and checkpoint["step"] == spec["steps"]
    model = module.construct(spec, "cpu").eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    executed = list(model.iter_blocks())
    assert len(model.blocks) == spec["layers"]
    assert len(executed) == spec["layers"] * spec["repeats"]
    assert all(b is model.blocks[i % spec["layers"]] for i, b in enumerate(executed))
    metadata = json.loads((run / "world-metadata.json").read_text())
    with np.load(run / "world.npz") as archive:
        world = {k: archive[k] for k in archive.files}
    world["metadata"] = metadata
    assert module.audit_world(world)["dataset_sha256"] == original["data_sha256"]
    result = {
        "run": run.name,
        "group": entry["group"],
        "world": spec["world_seed"],
        "initialization": spec["initialization"],
        "width": spec["width"],
        "phi": spec.get("phi", 4),
        "unique_blocks": len(model.blocks),
        "repeats": model.repeats,
        "executed_blocks_per_answer": len(executed),
        "checkpoint_sha256": original["checkpoint_sha256"],
        "original_reload_audit_sha256": sha(run / "audit.json"),
        "data_sha256": original["data_sha256"],
        "scores": {},
        "traces": {},
    }
    saved = {}
    with np.load(run / f"predictions-{spec['steps']:06d}.npz") as old:
        for name in module.EVALUATION_SPLITS:
            rows = world[name]
            answer, traces = answers(model, rows, metadata["separator_token"])
            archived = old[name + "_generated"][:, 0]
            disagreements = int(np.count_nonzero(answer != archived))
            assert 1 - disagreements / len(rows) >= agreement_floor, (name, disagreements)
            score = {
                "n": len(rows),
                "correct": int((answer == rows[:, -1]).sum()),
                "answer_accuracy": float(np.mean(answer == rows[:, -1])),
                "archived_answer_disagreements": disagreements,
                "archived_answer_accuracy": float(np.mean(archived == rows[:, -1])),
            }
            if name.startswith(("familiar_", "strict_")):
                hop = int(name.rsplit("_", 1)[1])
                baseline = guesses(world[f"train_{hop}"], rows)
                score["baselines"] = {
                    k: float(np.mean(v == rows[:, -1])) for k, v in baseline.items()
                }
                nodes, _ = module.truth_path_details(world, rows)
                simple = np.array([len(set(p)) == len(p) for p in nodes])
                score["no_repeated_entity"] = {
                    "n": int(simple.sum()),
                    "correct": int(((answer == rows[:, -1]) & simple).sum()),
                    "answer_accuracy": float(np.mean(answer[simple] == rows[simple, -1]))
                    if simple.any()
                    else None,
                }
                saved.update({name + "_baseline_" + k: v for k, v in baseline.items()})
            result["scores"][name], result["traces"][name] = score, traces
            saved[name + "_answer"] = answer
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
    result["passed"] = True
    out.mkdir()
    write(out / "audit.json", result)
    np.savez_compressed(out / "answers.npz", **saved)
    return result


def aggregate(results):
    groups = defaultdict(list)
    for r in results:
        groups[(r["group"], r["width"], r["phi"], r["unique_blocks"], r["repeats"])].append(r)
    output = []
    for condition, records in sorted(groups.items()):
        row = {"condition": list(condition), "runs": len(records), "scores": {}}
        for split in records[0]["scores"]:
            worlds = defaultdict(list)
            for r in records:
                worlds[r["world"]].append(r["scores"][split])
            by_world = {}
            for world, scores in worlds.items():
                value = {"answer_accuracy": float(np.mean([s["answer_accuracy"] for s in scores]))}
                if "baselines" in scores[0]:
                    value["baselines"] = {
                        k: float(np.mean([s["baselines"][k] for s in scores]))
                        for k in scores[0]["baselines"]
                    }
                    simple = [s["no_repeated_entity"]["answer_accuracy"] for s in scores]
                    value["no_repeated_entity_accuracy"] = (
                        float(np.mean(simple)) if all(s is not None for s in simple) else None
                    )
                by_world[str(world)] = value
            score = {
                "by_world": by_world,
                "answer_accuracy": float(
                    np.mean([s["answer_accuracy"] for s in by_world.values()])
                ),
            }
            if "baselines" in next(iter(by_world.values())):
                score["baselines"] = {
                    k: float(np.mean([s["baselines"][k] for s in by_world.values()]))
                    for k in next(iter(by_world.values()))["baselines"]
                }
            row["scores"][split] = score
        output.append(row)
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    config = json.loads(args.config.read_text())
    assert config["training_updates"] == 0 and config["device"] == "cpu"
    args.out.mkdir(parents=True, exist_ok=False)
    write(args.out / "frozen-config.json", config)
    write(
        args.out / "execution.json",
        {
            "script_sha256": sha(__file__),
            "config_sha256": sha(args.config),
            "torch": torch.__version__,
            "device": "cpu",
            "training_updates": 0,
        },
    )
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    results = []
    for entry in config["runs"]:
        r = audit(
            entry, args.out / Path(entry["run_root"]).name, config["archived_agreement_floor"]
        )
        results.append(r)
        print(
            json.dumps(
                {
                    "run": r["run"],
                    "passed": r["passed"],
                    "three_hop": r["scores"]["familiar_3"]["answer_accuracy"],
                    "four_hop": r["scores"]["familiar_4"]["answer_accuracy"],
                }
            ),
            flush=True,
        )
    write(
        args.out / "summary.json",
        {
            "passed": True,
            "runs": len(results),
            "training_updates": 0,
            "conditions": aggregate(results),
        },
    )


if __name__ == "__main__":
    main()
