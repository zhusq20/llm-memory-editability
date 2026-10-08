"""Download pinned public model files and the authors' original covariance caches."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import subprocess
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
MODEL_REVISIONS = {
    "gpt-j-6B": ("EleutherAI/gpt-j-6b", "47e169305d2e8376be1d31e765533382721b2cc1"),
    "gpt2-xl": ("openai-community/gpt2-xl", "15ea56dee5df4983c59b2538573817e1667135e2"),
}


def sha(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def assets():
    files = []
    common = ["config.json", "merges.txt", "tokenizer.json", "tokenizer_config.json", "vocab.json"]
    for name, (repository, revision) in MODEL_REVISIONS.items():
        extra = (
            ["pytorch_model.bin", "added_tokens.json", "special_tokens_map.json"]
            if name == "gpt-j-6B"
            else ["model.safetensors", "generation_config.json"]
        )
        for filename in common + extra:
            files.append(
                (
                    PROJECT / "data/paper-reproduction-models-v1" / name / filename,
                    f"https://huggingface.co/{repository}/resolve/{revision}/{filename}?download=true",
                )
            )
    for name, layers, projection in (
        ("EleutherAI_gpt-j-6B", range(3, 9), "fc_out"),
        ("gpt2-xl", [17], "c_proj"),
    ):
        for layer in layers:
            relative = (
                f"{name}/wikipedia_stats/transformer.h.{layer}.mlp."
                f"{projection}_float32_mom2_100000.npz"
            )
            files.append(
                (
                    PROJECT / "data/paper-reproduction-stats-v1" / relative,
                    f"https://memit.baulab.info/data/stats/{relative}",
                )
            )
    return files


def download(item):
    path, url = item
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = path.with_name(path.name + ".download.json")
    if manifest.exists():
        recorded = json.loads(manifest.read_text())
        assert recorded["url"] == url and recorded["sha256"] == sha(path)
        return recorded
    partial = path.with_name(path.name + ".partial")
    print(json.dumps({"downloading": str(path)}), flush=True)
    subprocess.run(
        [
            "curl",
            "--fail",
            "--location",
            "--retry",
            "6",
            "--retry-delay",
            "3",
            "--continue-at",
            "-",
            "--silent",
            "--show-error",
            "--output",
            str(partial),
            url,
        ],
        check=True,
    )
    assert partial.stat().st_size > 0
    partial.rename(path)
    record = {"path": str(path), "url": url, "bytes": path.stat().st_size, "sha256": sha(path)}
    manifest.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"downloaded": str(path), "bytes": record["bytes"]}), flush=True)
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    destination = (
        PROJECT
        / "docs/development-artifacts/paper-reproductions-parallel-v1/download-manifest.json"
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        records = list(executor.map(download, assets()))
    destination.write_text(
        json.dumps({"model_revisions": MODEL_REVISIONS, "files": records}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
