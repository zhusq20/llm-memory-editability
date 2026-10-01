"""Reproduce the tokenizer-only corpus audit without loading model weights.

Usage: /path/to/environment/bin/python tokenizer-runtime-collector.py auto output.json
Use ``generic`` in place of ``auto`` for the official tokenizer.json loader.
"""

import gzip
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer, PreTrainedTokenizerFast

root = Path(__file__).resolve().parents[3]
artifact_dir = Path(__file__).resolve().parent
config = json.loads((root / "configs/architecture-bridge-v1.json").read_text())
with gzip.open(artifact_dir / "tokenizer-runtime-raw-ids.json.gz", "rt") as handle:
    texts = json.load(handle)["corpus"]
mode, output_path = sys.argv[1:3]
if mode not in {"auto", "generic"}:
    raise ValueError("mode must be auto or generic")
result = {
    "python": sys.executable,
    "versions": {
        name: importlib.metadata.version(name)
        for name in ("transformers", "tokenizers")
    },
    "mode": mode,
    "models": {},
}
for key, entry in config["models"].items():
    cls = AutoTokenizer if mode == "auto" else PreTrainedTokenizerFast
    tokenizer = cls.from_pretrained(root / entry["path"], local_files_only=True)
    backend = json.loads(tokenizer.backend_tokenizer.to_str())
    items = []
    for item in texts:
        ids = tokenizer.encode(item["text"], add_special_tokens=False)
        if item["append_eos"]:
            ids.append(tokenizer.eos_token_id)
        items.append(
            {
                "ids": ids,
                "count": len(ids),
                "default_count": len(tokenizer.encode(item["text"])),
                "no_special_count": len(
                    tokenizer.encode(item["text"], add_special_tokens=False)
                ),
                "ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
            }
        )
    result["models"][key] = {
        "class": type(tokenizer).__name__,
        "module": type(tokenizer).__module__,
        "is_fast": tokenizer.is_fast,
        "eos_token_id": tokenizer.eos_token_id,
        "bos_token_id": tokenizer.bos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "special_tokens_map": tokenizer.special_tokens_map,
        "backend_components": {
            key: backend[key]
            for key in ("normalizer", "pre_tokenizer", "post_processor", "decoder")
        },
        "items": items,
    }
Path(output_path).write_text(json.dumps(result, ensure_ascii=False))
print(mode, len(texts), {key: value["class"] for key, value in result["models"].items()})
