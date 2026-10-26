#!/usr/bin/env python3
"""Restore only the pinned TTCD experiment's old graph/config from HF.

No model weights are present in this dataset revision. Credentials are read
from a private local file and are never included in metadata or command output.
Large numerical bundles are downloaded only with --arrays, verified in full,
then extracted using the project's existing safe, per-file-hash restore code.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from download_experiment_artifacts import restore_array_archive

REPO = "zsqzz/llm-memory-editability"
REVISION = "6c580455ce1667274ea592c1c88f5d0fb1e5a190"
PREFIX = "results/grokking-reproduction-v1"
ARRAYS = [f"{PREFIX}/data/{name}.npz" for name in ("world", "panels")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/ttcd-second-hop-assets"))
    parser.add_argument(
        "--token-file", type=Path, default=Path("/home/siqizhu4/.cache/llm-memory-ttcd/hf-token")
    )
    parser.add_argument("--arrays", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("HF_XET_CACHE", str(args.output.resolve() / "hf-cache/xet"))
    from huggingface_hub import HfApi, hf_hub_download

    token = args.token_file.read_text().strip() if args.token_file.is_file() else None
    cache = args.output / "hf-cache"
    args.output.mkdir(parents=True, exist_ok=True)

    def download(filename, local=False):
        return Path(
            hf_hub_download(
                REPO,
                filename,
                repo_type="dataset",
                revision=REVISION,
                token=token,
                cache_dir=cache,
                **({"local_dir": args.output} if local else {}),
            )
        )

    info = HfApi(token=token).dataset_info(REPO, revision=REVISION)
    names = [s.rfilename for s in info.siblings]
    index = json.loads(download("ARRAY_ARCHIVES.json").read_text())
    needed = [a for a in index["archives"] if set(a["members"]) & set(ARRAYS)]
    restored = []
    for filename in (f"{PREFIX}/data/complete.json", f"{PREFIX}/reference/gpt2/config.json"):
        download(filename, local=True)
        restored.append(filename)
    if args.arrays:
        manifest = {
            f["path"]: f for f in json.loads(download("ARCHIVE_MANIFEST.json").read_text())["files"]
        }
        for archive in needed:
            bundle = download(archive["path"])
            restore_array_archive(bundle, archive, manifest, ARRAYS, args.output)
            restored.extend(n for n in archive["members"] if n in ARRAYS)
    record = {
        "repo": REPO,
        "revision": info.sha,
        "restored": restored,
        "checkpoint_files": [
            n for n in names if n.endswith((".pt", ".pth", ".safetensors", ".ckpt"))
        ],
        "discoloop_batch_files": [n for n in names if "supervised-alignment-development-v1" in n],
        "required_array_bundles": [{k: v for k, v in a.items() if k != "members"} for a in needed],
        "missing_checkpoint": (
            "Restore original audited DiscoLoop/Loop weights from the experiment host. "
            "Do not substitute random weights or restart old training."
        ),
    }
    (args.output / "hf-availability.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
