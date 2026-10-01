#!/usr/bin/env python3
"""Explicit execution entry point for the prospective Hebbian study."""

import argparse
import importlib.metadata
import json
import platform
import re
import subprocess
import sys

from llm_memory_editability.hebbian_learning import (
    ARTIFACTS,
    CONFIG_PATH,
    DATA,
    calibrate,
    config,
    now,
    read_json,
    sha256,
    write_json,
)


def prepare_sources():
    import requests
    from transformers import AutoTokenizer

    from llm_memory_editability.hebbian_data import clean_candidates

    cfg = config()
    source = DATA / "source"
    source.mkdir(parents=True, exist_ok=True)

    def fetch(url, path):
        if path.exists():
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with requests.get(url, stream=True, timeout=(30, 180)) as response:
            response.raise_for_status()
            tmp = path.with_name(path.name + ".part")
            with tmp.open("wb") as f:
                for block in response.iter_content(4 * 1024 * 1024):
                    f.write(block)
            tmp.replace(path)

    fetch("https://rome.baulab.info/data/dsets/counterfact.json", source / "counterfact.json")
    model = cfg["model"]
    fetch(
        f"https://huggingface.co/api/models/{model['id']}/revision/{model['revision']}",
        source / "model-info.json",
    )
    model_info = read_json(source / "model-info.json")
    assert model_info["sha"] == model["revision"]
    for item in model_info["siblings"]:
        name = item["rfilename"]
        if name != ".gitattributes":
            fetch(
                f"https://huggingface.co/{model['id']}/resolve/{model['revision']}/{name}",
                source / "qwen3-0.6b-base" / name,
            )
    rev = cfg["author_revision"]
    for name in [
        "src/hebbian/methods/hebbian/model.py",
        "src/hebbian/methods/hebbian/construction.py",
        "LICENSE",
        "README.md",
    ]:
        fetch(
            f"https://raw.githubusercontent.com/HazyResearch/hebbian-mlps/{rev}/{name}",
            source / "hebbian-mlps" / name,
        )
    fetch("https://huggingface.co/api/datasets/Salesforce/wikitext", source / "wikitext-info.json")
    wiki_info = read_json(source / "wikitext-info.json")
    tokenizer = AutoTokenizer.from_pretrained(source / "qwen3-0.6b-base", local_files_only=True)
    import pyarrow.parquet as pq

    texts = {}
    for split in ["train", "test"]:
        name = f"wikitext-2-raw-v1/{split}-00000-of-00001.parquet"
        path = source / name
        fetch(
            f"https://huggingface.co/datasets/Salesforce/wikitext/resolve/{wiki_info['sha']}/{name}",
            path,
        )
        lines = pq.read_table(path).column("text").to_pylist()
        documents, current = [], []
        for line in lines:
            if re.fullmatch(r"\s*= [^=]+ =\s*", line) and current:
                documents.append("".join(current))
                current = []
            current.append(line)
        if current:
            documents.append("".join(current))
        segments = []
        for doc_id, doc in enumerate(documents):
            tokens = tokenizer(doc, add_special_tokens=False)["input_ids"]
            for offset in range(0, len(tokens) - 127, 128):
                ids = tokens[offset : offset + 128]
                segments.append(
                    {
                        "document_id": doc_id,
                        "token_offset": offset,
                        "document_sha256": __import__("hashlib").sha256(doc.encode()).hexdigest(),
                        "input_ids": ids,
                        "loss_mask": [0] + [1] * 127,
                        "answer_start": 1,
                        "source_split": split,
                    }
                )
                if len(segments) == 128:
                    break
            if len(segments) == 128:
                break
        assert len(segments) == 128
        texts[split] = segments
    write_json(DATA / "text.json", texts)
    files = {
        str(p.relative_to(source)): sha256(p)
        for p in source.rglob("*")
        if p.is_file() and not p.name.endswith(".part")
    }
    write_json(
        ARTIFACTS / "source-manifest.json",
        {
            "time": now(),
            "model_revision": model["revision"],
            "author_revision": rev,
            "wikitext_revision": wiki_info["sha"],
            "sha256": files,
            "model_license": "Apache-2.0",
            "author_license": (source / "hebbian-mlps/LICENSE").read_text(),
            "wikitext_card": wiki_info.get("cardData"),
            "config_sha256": sha256(CONFIG_PATH),
        },
    )
    if not (DATA / "candidates.json").exists():
        clean_candidates(tokenizer)
    packages = {}
    for name in [
        "torch",
        "transformers",
        "tokenizers",
        "datasets",
        "huggingface-hub",
        "numpy",
        "matplotlib",
        "safetensors",
        "pyarrow",
    ]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not installed"
    write_json(
        ARTIFACTS / "environment-lock.json",
        {
            "time": now(),
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "packages": packages,
            "dtype": "float32",
            "attention": "eager",
            "tf32": False,
            "deterministic_algorithms": True,
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "gpu_query": subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,uuid,ecc.errors.uncorrected.volatile.total",
                    "--format=csv",
                ],
                text=True,
            ),
            "excluded_gpu_indices": [0],
        },
    )
    (ARTIFACTS / "environment-freeze.txt").write_text(
        subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=[
            "prepare",
            "calibrate",
            "baseline",
            "audit-data",
            "lock-data",
            "freeze-b",
            "preflight",
            "diagnose",
            "form",
            "adapt",
        ],
    )
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--split", choices=["dev", "eval"], default="dev")
    parser.add_argument("--run-id", default="hebbian-learning-v1")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare_sources()
    elif args.command == "calibrate":
        calibrate()
    elif args.command == "baseline":
        from llm_memory_editability.hebbian_model import run_baseline

        run_baseline(args.device, args.shard, args.shards)
    elif args.command == "audit-data":
        from llm_memory_editability.hebbian_data import semantic_audit

        semantic_audit()
    elif args.command == "freeze-b":
        from llm_memory_editability.hebbian_statistics import freeze_b

        freeze_b()
    elif args.command == "lock-data":
        from llm_memory_editability.hebbian_data import lock_data

        rows = []
        for p in sorted(DATA.glob("baseline-shard-*.jsonl")):
            rows.extend(json.loads(line) for line in p.read_text().splitlines())
        assert len({r["case_id"] for r in rows}) == len(rows)
        write_json(DATA / "baseline.json", sorted(rows, key=lambda r: r["case_id"]))
        print(lock_data())
    else:
        from llm_memory_editability.hebbian_train import dispatch

        dispatch(args)


if __name__ == "__main__":
    main()
