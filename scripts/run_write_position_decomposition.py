"""Write-position decomposition of saved sequential writes (zero training).

For write k of a stream, theta_{k-1} and theta_k are the saved MLP states before and after
the write. Every query is scored under four conditions: theta_{k-1} everywhere ("none"),
theta_k everywhere ("full"), theta_k only at the subject position (position 0, "subject"),
and theta_k only at the answer position (the last input token, "answer"). Two-hop queries
add theta_k only at the first relation (position 1, "middle").

Scored queries: the just-written fact, earlier new facts, old atomic items (two pools plus
mixed-query companions) and the five old two-hop pools. Answer NLL and argmax correctness
at the answer position; EOS is not scored.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import run_ce_cross_sample as parent  # noqa: E402
import torch  # noqa: E402
from torch.nn import functional as F  # noqa: E402

from llm_memory_editability.ce_cross_sample import facts_by_id  # noqa: E402
from llm_memory_editability.ce_scope_editing import arrays, load_edit  # noqa: E402
from llm_memory_editability.write_position import layer_values, masked_logits  # noqa: E402

PERSISTENT = Path("/ossfs/workspace/llm-memory-editability")
ARMS = {
    "transformer": ("transformer", "ce-cross-sample-double-new-v1/transformer/output/evaluate"),
    "loop": ("loop", "ce-cross-sample-double-new-v1/loop/output/evaluate"),
    "discoloop": ("discoloop", "ce-cross-sample-double-new-v1/discoloop/output/evaluate"),
    "loop-no-bound": ("loop", "ce-cross-sample-ablation-v1/loop-no-bound/output/science"),
    "discoloop-no-bound": (
        "discoloop",
        "ce-cross-sample-ablation-v1/discoloop-no-bound/output/science",
    ),
}
PLAN = "configs/ce-cross-sample-double-new-v1.json"


def score(reader, rows, metadata, new, region, batch_size=1024):
    """Answer NLL and argmax correctness with theta_k restricted to `region`."""
    rows = np.asarray(rows, dtype=np.int64)
    device = next(reader.parameters()).device
    nll, correct = [], []
    for offset in range(0, len(rows), batch_size):
        part = rows[offset : offset + batch_size]
        tokens, positions, labels = arrays(part, metadata, device)
        answer_position = positions[:, 0]
        mask = torch.zeros(tokens.shape, dtype=torch.bool, device=device)
        index = torch.arange(len(part), device=device)
        if region == "full":
            mask[:] = True
        elif region == "subject":
            mask[:, 0] = True
        elif region == "middle":
            mask[:, 1] = True
        elif region == "answer":
            mask[index, answer_position] = True
        elif region != "none":
            raise ValueError(region)
        logits = masked_logits(reader, tokens, positions, new, mask)[:, 0].float()
        nll.extend(F.cross_entropy(logits, labels[:, 0], reduction="none").cpu().tolist())
        correct.extend((logits.argmax(-1) == labels[:, 0]).cpu().tolist())
    return nll, correct


def query_sets(panel):
    old_atomic, seen = [], set()
    for rows in [panel["old_eval_pools"]["atomic_id"], panel["old_eval_pools"]["atomic_ood"]]:
        for row in rows:
            if tuple(row) not in seen:
                seen.add(tuple(row))
                old_atomic.append(list(row))
    for query in panel["mixed_queries"]:
        if tuple(query["old_fact"]) not in seen:
            seen.add(tuple(query["old_fact"]))
            old_atomic.append(list(query["old_fact"]))
    chains = [
        list(row)
        for name in ("train_ii", "test_ii", "test_io", "test_oi", "test_oo")
        for row in panel["old_eval_pools"][name]
    ]
    return old_atomic, chains


def run_arm(arm, out, device):
    architecture, source = ARMS[arm]
    plan = json.loads((ROOT / PLAN).read_text())
    reader, world, panel, base, anchor, _, _, _ = parent.initialize(plan, architecture, device)
    metadata = panel["metadata"]
    facts = facts_by_id(panel)
    old_atomic, chains = query_sets(panel)
    layers = len(reader.backbone.transformer.h)
    records = PERSISTENT / source / "records"
    states = PERSISTENT / source / "states"
    previous_values = {name: value.detach().cpu().clone() for name, value in anchor.items()}
    out.mkdir(parents=True, exist_ok=True)
    rows_path = out / f"{arm}.jsonl"
    check = {"max_full_vs_record": 0.0, "max_none_vs_previous_full": 0.0}
    with rows_path.open("w") as handle:
        for stream in ("forward", "reverse"):
            previous_values = {n: v.detach().cpu().clone() for n, v in anchor.items()}
            previous_full = None
            for step in range(1, 65):
                key = f"{stream}-{step:03d}"
                record = json.loads((records / f"{key}.json").read_text())
                payload = torch.load(states / f"{key}.pt", map_location="cpu", weights_only=True)
                if payload["written"] != record["written"]:
                    raise ValueError(f"State and record disagree at {key}")
                fact = facts[record["written"][-1]]
                written = [facts[w] for w in record["written"][:-1]]
                load_edit(reader, {n: v.to(device) for n, v in previous_values.items()}, "mlp")
                new = [
                    tuple(t.to(device) for t in params)
                    for params in layer_values(payload["values"], layers)
                ]
                groups = {
                    "current": [list(fact)],
                    "earlier_new": [list(f) for f in written],
                    "old_atomic": old_atomic,
                    "chain": chains,
                }
                result = {"key": key, "stream": stream, "step": step, "fact": list(fact)}
                for name, rows in groups.items():
                    if not rows:
                        result[name] = None
                        continue
                    regions = ("none", "full", "subject", "answer")
                    if name == "chain":
                        regions += ("middle",)
                    result[name] = {
                        "rows": rows,
                        **{r: score(reader, rows, metadata, new, r) for r in regions},
                    }
                # Consistency with the saved readouts of the same state.
                saved = record["readout"]["old"]["atomic_id"]
                full = score(reader, saved["rows"], metadata, new, "full")[0]
                check["max_full_vs_record"] = max(
                    check["max_full_vs_record"],
                    float(np.max(np.abs(np.array(full) - np.array(saved["answer_nll"])))),
                )
                none_chain = np.array(result["chain"]["none"][0])
                if previous_full is not None:
                    check["max_none_vs_previous_full"] = max(
                        check["max_none_vs_previous_full"],
                        float(np.max(np.abs(none_chain - previous_full))),
                    )
                previous_full = np.array(result["chain"]["full"][0])
                handle.write(json.dumps(result) + "\n")
                previous_values = payload["values"]
                print(arm, key, json.dumps(check), flush=True)
    (out / f"{arm}-check.json").write_text(json.dumps(check, indent=1))
    return check


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arms", nargs="+", default=list(ARMS))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    for arm in args.arms:
        start = time.time()
        check = run_arm(arm, args.out, args.device)
        print(arm, "done", round(time.time() - start, 1), json.dumps(check), flush=True)


if __name__ == "__main__":
    main()
