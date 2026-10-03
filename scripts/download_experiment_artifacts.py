#!/usr/bin/env python3
"""Restore experiment artifacts from Hugging Face to their original local paths."""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import tarfile
from pathlib import Path


def restore_array_archive(archive_path, archive_info, manifest, patterns, output):
    """Verify a bundle and restore only selected, hash-checked regular files."""
    digest = hashlib.sha256()
    with Path(archive_path).open("rb") as handle:
        while chunk := handle.read(4 * 1024**2):
            digest.update(chunk)
    if digest.hexdigest() != archive_info["sha256"]:
        raise ValueError(f"Archive SHA256 mismatch: {archive_info['path']}")
    root = Path(output).resolve()
    expected = set(archive_info["members"])
    seen = set()
    restored = 0
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive:
            if member.name not in expected or member.name in seen or not member.isfile():
                raise ValueError(f"Unexpected archive member: {member.name}")
            seen.add(member.name)
            target = (root / member.name).resolve()
            if not target.is_relative_to(root) or target == root:
                raise ValueError(f"Unsafe archive path: {member.name}")
            if not any(fnmatch.fnmatchcase(member.name, pattern) for pattern in patterns):
                continue
            wanted = manifest[member.name]
            handle = archive.extractfile(member)
            if handle is None:
                raise ValueError(f"Unreadable archive member: {member.name}")
            with handle:
                payload = handle.read()
            if len(payload) != wanted["bytes"]:
                raise ValueError(f"Array size mismatch: {member.name}")
            if hashlib.sha256(payload).hexdigest() != wanted["sha256"]:
                raise ValueError(f"Array SHA256 mismatch: {member.name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
            restored += 1
    if seen != expected:
        raise ValueError(f"Missing archive members: {archive_info['path']}")
    return restored


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="zsqzz/llm-memory-editability")
    parser.add_argument("--revision", default="main", help="Dataset branch or commit SHA")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--include",
        action="append",
        help="Path glob to download; repeat for multiple batches (default: data, results, docs)",
    )
    args = parser.parse_args()
    try:
        from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    except ImportError as error:
        raise SystemExit("Install archive dependencies: pip install -e '.[archive]'") from error

    revision = HfApi().dataset_info(args.repo_id, revision=args.revision).sha
    patterns = args.include or ["data/**", "results/**", "docs/**"]
    common = dict(
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=revision,
        local_dir=args.output,
    )
    index_path = hf_hub_download(filename="ARRAY_ARCHIVES.json", **common)
    manifest_path = hf_hub_download(filename="ARCHIVE_MANIFEST.json", **common)
    index = json.loads(Path(index_path).read_text())
    manifest = {item["path"]: item for item in json.loads(Path(manifest_path).read_text())["files"]}
    snapshot_download(allow_patterns=patterns, **common)
    restored = 0
    for archive in index["archives"]:
        if not any(
            fnmatch.fnmatchcase(name, pattern)
            for name in archive["members"]
            for pattern in patterns
        ):
            continue
        path = hf_hub_download(filename=archive["path"], **common)
        restored += restore_array_archive(path, archive, manifest, patterns, args.output)
    print(f"Downloaded revision {revision}; restored {restored} numerical arrays.")


if __name__ == "__main__":
    main()
