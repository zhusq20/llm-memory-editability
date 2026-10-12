"""Fixed-weight development diagnostics; never select training arms by outcomes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from transformers import GPT2Config

from llm_memory_editability.grokking_reproduction import encode_rows
from llm_memory_editability.self_decode_alignment import SelfDecodeGPT, valid_token_mask


@torch.no_grad()
def run(config_path, output):
    cfg = json.loads(Path(config_path).read_text())
    root = Path(cfg["input_root"])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = True
    meta = json.loads((root / "data/complete.json").read_text())
    panels = dict(np.load(root / "data/panels.npz"))
    atoms = np.load(root / "data/world.npz")["atoms"]
    lookup = {(int(h), int(r)): int(t) for h, r, t in atoms}
    model_cfg = GPT2Config.from_pretrained(root / "reference/gpt2", local_files_only=True)
    model_cfg.n_layer = 4
    model_cfg.vocab_size = meta["vocab_size"]
    model_cfg.use_cache = False
    records = []
    for source in cfg["diagnostics"]["runs"]:
        for step in cfg["diagnostics"]["nodes"]:
            checkpoint = root / "development" / source / f"checkpoint-{step:07d}.pt"
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            model = SelfDecodeGPT(model_cfg, repeats=2).cuda().eval()
            model.load_state_dict(saved["model"])
            del saved
            for panel in cfg["diagnostics"]["panels"]:
                rows = panels[panel][: cfg["diagnostics"]["rows_per_panel"]]
                packed = encode_rows(rows, meta)
                arrays = {k: [] for k in ("first", "final", "confidence", "cosine", "valid")}
                predictions = {s: [] for s in cfg["diagnostics"]["replacement_scopes"]}
                for start in range(0, len(rows), 64):
                    x, positions, labels = [
                        torch.as_tensor(a[start : start + 64], device="cuda") for a in packed
                    ]
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        _, first, final = model.forward_with_states(x, positions)
                        first_logits = F.linear(
                            model.transformer.ln_f(first), model.transformer.wte.weight
                        )
                        final_logits = F.linear(
                            model.transformer.ln_f(final), model.transformer.wte.weight
                        )
                        probability = first_logits.float().softmax(-1)
                        confidence, token = probability.max(-1)
                        final_token = final_logits.argmax(-1)
                        cosine = F.cosine_similarity(
                            first.float(), model.transformer.wte(token).float(), dim=-1
                        )
                        valid = valid_token_mask(positions, labels, x.shape[1])
                        for key, value in zip(
                            arrays, (token, final_token, confidence, cosine, valid), strict=True
                        ):
                            arrays[key].append(value.cpu().numpy())
                        for scope in predictions:
                            logits = model.forward_with_replacement(
                                x, positions, scope=scope, valid_mask=valid
                            )
                            answer = logits[:, 0].argmax(-1)
                            generated = x.clone()
                            generated[torch.arange(len(x), device="cuda"), positions[:, 1]] = answer
                            stop = model.forward_with_replacement(
                                generated, positions, scope=scope, valid_mask=valid
                            )[:, 1].argmax(-1)
                            predictions[scope].append(torch.stack((answer, stop), -1).cpu().numpy())
                arrays = {k: np.concatenate(v) for k, v in arrays.items()}
                predictions = {k: np.concatenate(v) for k, v in predictions.items()}
                by_position = []
                for pos in range(4):
                    mask = arrays["valid"][:, pos]
                    if not mask.any():
                        continue
                    first = arrays["first"][mask, pos]
                    by_position.append(
                        {
                            "position": pos,
                            "n": int(mask.sum()),
                            "confidence": float(arrays["confidence"][mask, pos].mean()),
                            "cosine": float(arrays["cosine"][mask, pos].mean()),
                            "first_final_agreement": float(
                                (first == arrays["final"][mask, pos]).mean()
                            ),
                            "entity_fraction": float(
                                (
                                    (first >= meta["entity_offset"])
                                    & (first < meta["relation_offset"])
                                ).mean()
                            ),
                            "relation_fraction": float(
                                (
                                    (first >= meta["relation_offset"])
                                    & (first < meta["relation_offset"] + meta["relations"])
                                ).mean()
                            ),
                            "end_fraction": float((first == meta["end_marker"]).mean()),
                        }
                    )
                bridge = (
                    np.array([lookup[int(row[0]), int(row[1])] for row in rows])
                    + meta["entity_offset"]
                )
                record = {
                    "source_run": source,
                    "step": step,
                    "panel": panel,
                    "n": len(rows),
                    "first_round_bridge_accuracy": float((arrays["first"][:, 1] == bridge).mean()),
                    "positions": by_position,
                    "replacement_accuracy": {
                        s: float(
                            (
                                (p[:, 0] == rows[:, -1] + meta["entity_offset"])
                                & (p[:, 1] == meta["end_marker"])
                            ).mean()
                        )
                        for s, p in predictions.items()
                    },
                }
                stem = f"{source}-{step:07d}-{panel}"
                np.savez_compressed(
                    output / f"{stem}.npz",
                    rows=rows,
                    **arrays,
                    **{f"pred_{k}": v for k, v in predictions.items()},
                )
                records.append(record)
                (output / "progress.json").write_text(json.dumps(record, indent=2) + "\n")
                print(json.dumps(record), flush=True)
            del model
            torch.cuda.empty_cache()
    result = {
        "state": "complete",
        "optimization_updates": 0,
        "records": records,
        "interpretation": (
            "Acute replacement does not identify the effect of learned adaptation; "
            "one observed world."
        ),
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.config, args.output)
