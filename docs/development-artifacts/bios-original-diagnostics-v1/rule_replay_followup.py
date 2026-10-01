"""Post hoc replay of original rule training examples, separately reported."""

import json
import time
from pathlib import Path

import torch
from tokenizers import Tokenizer

from llm_memory_editability.bios_original_data import digest, load_json, write_json
from llm_memory_editability.bios_original_train import (
    generate,
    load_model,
    normalize,
    rule_rows,
    setup,
)

ROOT = Path(__file__).resolve().parent
plan = load_json(ROOT / "plan.json")
cfg = plan["config"]
tok = Tokenizer.from_file(str(Path(cfg["data_root"]) / "tokenizer/tokenizer.json"))
inputs = []
for i, row in enumerate(rule_rows(tok)):
    inputs.append(
        dict(
            index=i,
            task="parity" if i < 12 else "date_comparison",
            prompt=tok.decode(row["ids"][: row["answer_start"]]),
            target=tok.decode(row["ids"][row["answer_start"] :], skip_special_tokens=True).strip(),
        )
    )
input_path = ROOT / "rule-replay-inputs.json"
if input_path.exists():
    assert load_json(input_path) == inputs
else:
    write_json(input_path, inputs)
setup(1427, "cuda:0")
torch.cuda.set_per_process_memory_fraction(0.045, "cuda:0")
for endpoint in plan["endpoints"]:
    if endpoint["stage"] != "task":
        continue
    output = ROOT / "rule-replay" / f"{endpoint['id']}.json"
    if output.exists():
        continue
    assert digest(endpoint["checkpoint"]) == endpoint["checkpoint_sha256"]
    start = time.monotonic()
    model = load_model(endpoint["checkpoint"], cfg, "cuda:0", adapters=True).eval()
    model.requires_grad_(False)
    results = []
    for i in range(0, len(inputs), 16):
        batch = inputs[i : i + 16]
        generated, ended = generate(model, tok, [r["prompt"] for r in batch], "cuda:0", 96)
        results.extend(
            dict(row, generated=text, eos=eos, correct=normalize(text) == normalize(row["target"]))
            for row, text, eos in zip(batch, generated, ended, strict=True)
        )
    summary = {
        task: dict(
            n=sum(r["task"] == task for r in results),
            correct=sum(r["correct"] for r in results if r["task"] == task),
        )
        for task in ("parity", "date_comparison")
    }
    write_json(
        output,
        dict(
            endpoint=endpoint,
            summary=summary,
            rows=results,
            seconds=time.monotonic() - start,
            input_sha256=digest(input_path),
            script_sha256=digest(__file__),
            exploratory=True,
            shared_gpu=True,
            precision="FP32 parameters, BF16 autocast, TF32 enabled",
        ),
    )
    print(json.dumps(dict(endpoint=endpoint["id"], summary=summary)), flush=True)
    del model
