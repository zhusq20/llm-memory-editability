"""Resolve short-panel reload differences using the original packed shape and batch."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import GPT2Config

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
from evaluate_loop_beyond_four_hops import CONTROLS  # noqa: E402
from evaluate_loop_position_controlled_long_hops import sha, state_hash, write_json  # noqa: E402

from llm_memory_editability.loop_hop_data import pack_rows  # noqa: E402
from llm_memory_editability.loop_position_invariance import LoopPositionGPT  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--persistent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(0.16)
    torch.backends.cuda.matmul.allow_tf32 = True
    start = time.monotonic()
    p = args.persistent
    data = p / "loop-hop-development-v1/inputs/data"
    metadata = json.loads((data / "complete.json").read_text())
    assert sha(data / "panels.npz") == metadata["panels_sha256"]
    panels = dict(np.load(data / "panels.npz"))
    folder = p / "loop-position-invariance-development-v1/runs/disco-rope-old"
    checkpoint = folder / "checkpoint-1500000.pt"
    grid = json.loads((folder / "grid.json").read_text())
    assert sha(checkpoint) == grid["checkpoint_sha256"]
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert saved["step"] == 1500000
    config = json.loads((ROOT / "configs/loop-position-invariance-development-v1.json").read_text())
    hf = GPT2Config(
        **config["model"],
        n_layer=1,
        vocab_size=metadata["vocab_size"],
        bos_token_id=metadata["pad_token"],
        eos_token_id=metadata["end_marker"],
        pad_token_id=metadata["pad_token"],
        use_cache=False,
    )
    args.output.mkdir(parents=True, exist_ok=False)
    result = {
        "state": "running",
        "training_updates": 0,
        "completed_pools": 0,
        "checkpoint_sha256": sha(checkpoint),
        "weights": {},
    }
    write_json(args.output / "status.json", result)
    for weight in ["ema", "raw"]:
        model = LoopPositionGPT(
            hf,
            4,
            handoff="additive",
            position="rope",
            rope_base=config["rope_base"],
            **config["handoff"],
        )
        model.load_state_dict(saved["model"], strict=True)
        if weight == "ema":
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    parameter.copy_(saved["ema"][name])
        model.to("cuda:0").eval().requires_grad_(False)
        before = state_hash(model)
        checks = {}
        with torch.no_grad():
            for panel in CONTROLS:
                tokens, positions, _ = pack_rows(panels[panel], metadata)
                parts = []
                for i in range(0, len(tokens), config["evaluation_batch_size"]):
                    x = torch.as_tensor(
                        tokens[i : i + config["evaluation_batch_size"]], device="cuda:0"
                    )
                    pos = torch.as_tensor(
                        positions[i : i + config["evaluation_batch_size"]], device="cuda:0"
                    )
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        logits, _, _ = model.trace(x, pos, 32)
                    parts.append(logits[:, :, 0].argmax(-1).cpu().numpy())
                answer = np.concatenate(parts, axis=1)
                with np.load(folder / "grid" / weight / f"{panel}.npz") as old:
                    mismatches = int((answer != old["answer"]).sum())
                    old_acc = (
                        old["answer"] == panels[panel][None, :, -1] + metadata["entity_offset"]
                    ).mean(1)
                acc = (answer == panels[panel][None, :, -1] + metadata["entity_offset"]).mean(1)
                checks[panel] = {
                    "token_mismatches": mismatches,
                    "accuracy": acc.tolist(),
                    "maximum_accuracy_difference_pp": float(np.max(abs(acc - old_acc)) * 100),
                }
                np.savez_compressed(args.output / f"{weight}-{panel}.npz", answer=answer)
                result["completed_pools"] += 1
                print(json.dumps({"weight": weight, "panel": panel, **checks[panel]}), flush=True)
        assert before == state_hash(model)
        result["weights"][weight] = {"state_hash_before_after": before, "checks": checks}
        write_json(args.output / "status.json", result)
        del model
        torch.cuda.empty_cache()
    result.update(
        state="complete",
        elapsed_seconds=time.monotonic() - start,
        peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
    )
    write_json(args.output / "status.json", result)


if __name__ == "__main__":
    main()
