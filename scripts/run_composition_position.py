"""Which query position reads the second written fact during composition (zero training).

Uses the isolated two-fact states of ce-cross-sample-double-new-v1. For order A_then_B, the
second write is B = (b, r_B, t): theta_prev is the A-only state, theta_new is A_then_B. For order
B_then_A the second write is A = (h, r_A, b): theta_prev is B_only, theta_new is B_then_A.
The second write is applied everywhere, nowhere, or at one token position only, and these
queries are scored (answer token, argmax and NLL):

- the written atom [subject, relation]: position 0 = its subject, 1 = its answer position;
- the double-new query [h, r_A, r_B]: position 0 = h, 1 = r_A (where round one decodes b and
  DiscoLoop re-encodes it), 2 = r_B (the answer position);
- mixed queries that pair the written fact with one unchanged old fact.

If the second hop B is read where b is re-encoded (position 1), its write is used through
the entity; if it is read at r_B (position 2), the round-two state there reproduces the
atomic question [b, r_B] that the write was learned on.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import run_ce_cross_sample as parent  # noqa: E402
import torch  # noqa: E402
from torch.nn import functional as F  # noqa: E402

from llm_memory_editability.ce_cross_sample import formal_pairs  # noqa: E402
from llm_memory_editability.ce_scope_editing import arrays, load_edit  # noqa: E402
from llm_memory_editability.write_position import layer_values, masked_logits  # noqa: E402

PERSISTENT = Path("/ossfs/workspace/llm-memory-editability")
STATES = "ce-cross-sample-double-new-v1/{}/output/evaluate/states"
PLAN = "configs/ce-cross-sample-double-new-v1.json"


def score(reader, rows, metadata, new, region):
    rows = np.asarray(rows, dtype=np.int64)
    device = next(reader.parameters()).device
    tokens, positions, labels = arrays(rows, metadata, device)
    mask = torch.zeros(tokens.shape, dtype=torch.bool, device=device)
    if region == "full":
        mask[:] = True
    elif region.startswith("pos"):
        mask[:, int(region[3:])] = True
    elif region != "none":
        raise ValueError(region)
    logits = masked_logits(reader, tokens, positions, new, mask)[:, 0].float()
    return {
        "nll": F.cross_entropy(logits, labels[:, 0], reduction="none").cpu().tolist(),
        "correct": (logits.argmax(-1) == labels[:, 0]).cpu().tolist(),
    }


def load(path):
    return torch.load(path, map_location="cpu", weights_only=True)["values"]


def run(architecture, out, device):
    plan = json.loads((ROOT / PLAN).read_text())
    reader, _, panel, _, _, _, _, _ = parent.initialize(plan, architecture, device)
    metadata = panel["metadata"]
    layers = len(reader.backbone.transformer.h)
    states = PERSISTENT / STATES.format(architecture)
    a_only = {r["pair_id"]: r["state"] for r in panel["reused_A_only_states"][architecture]}
    rows = []
    for pair in formal_pairs(panel):
        pid = pair["pair_id"]
        double = [pair["query"] + [pair["answer"]]]
        for order, prev_path, new_path, written in (
            ("A_then_B", a_only[pid], states / f"{pid}-A_then_B.pt", "B"),
            ("B_then_A", states / f"{pid}-B_only.pt", states / f"{pid}-B_then_A.pt", "A"),
        ):
            fact = pair[written]["fact"]
            fid = pair[written]["fact_id"]
            mixed = [
                q["query"] + [q["answer"]] for q in panel["mixed_queries"] if q["fact_id"] == fid
            ]
            load_edit(reader, {n: v.to(device) for n, v in load(prev_path).items()}, "mlp")
            new = [tuple(t.to(device) for t in p) for p in layer_values(load(new_path), layers)]
            entry = {"pair_id": pid, "order": order, "written": written, "fact": fact}
            for name, queries, regions in (
                ("atom", [fact], ("none", "full", "pos0", "pos1")),
                ("double_new", double, ("none", "full", "pos0", "pos1", "pos2")),
                ("mixed", mixed, ("none", "full", "pos0", "pos1", "pos2")),
            ):
                entry[name] = {
                    "rows": queries,
                    **{r: score(reader, queries, metadata, new, r) for r in regions},
                }
            rows.append(entry)
            print(
                architecture,
                pid,
                order,
                json.dumps(
                    {
                        k: entry["double_new"][k]["correct"][0]
                        for k in ("none", "full", "pos0", "pos1", "pos2")
                    }
                ),
                flush=True,
            )
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{architecture}.json").write_text(json.dumps(rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--architectures", nargs="+", default=["discoloop", "loop", "transformer"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    for architecture in args.architectures:
        run(architecture, args.out, args.device)


if __name__ == "__main__":
    main()
