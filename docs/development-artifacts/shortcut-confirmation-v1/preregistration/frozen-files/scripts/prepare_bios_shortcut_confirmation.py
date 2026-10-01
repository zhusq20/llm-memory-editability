"""Prepare all prospectively locked base worlds, without training or evaluation.

The formal entry point requires the shared frozen-design guard. A private helper
also permits development seeds for tests in isolated temporary directories.
Committed worlds are immutable; resume regenerates and compares deterministic
bytes instead of overwriting existing files.
"""

import argparse
import fcntl
import hashlib
import json
import os
import tempfile
from pathlib import Path

from llm_memory_editability.bios_data import REVISION, audit_world, load_world, make_world

ROOT = Path(__file__).resolve().parents[1]
WORLD_FILES = ("world.npz", "metadata.json", "audit.json")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verified_sources(repo_root, source_root, frozen_files):
    repo_root, source_root = Path(repo_root).resolve(), Path(source_root).resolve()
    manifest_path = source_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("revision") != REVISION:
        raise ValueError("Official generator source revision changed")
    expected = {str(manifest_path.relative_to(repo_root)): digest(manifest_path)}
    seen = set()
    for entry in manifest["files"]:
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts or str(relative) in seen:
            raise ValueError("Source manifest path is duplicate or outside source root")
        seen.add(str(relative))
        path = source_root / relative
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Source is not an immutable regular file: {relative}")
        checksum = digest(path)
        if checksum != entry["sha256"] or (
            "bytes" in entry and path.stat().st_size != entry["bytes"]
        ):
            raise ValueError(f"Source manifest entry changed: {relative}")
        expected[str(path.relative_to(repo_root))] = checksum
    if expected != frozen_files:
        raise ValueError("Prospective source lock must cover manifest and every entry exactly")
    return expected


def immutable_json(path, value):
    """Atomic create; an existing receipt is accepted only when byte-identical."""
    path = Path(path)
    contents = (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != contents:
            raise ValueError(f"Immutable receipt mismatch: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{path.name}-", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(contents)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        # Hard-link creation fails rather than replacing a concurrent existing file.
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _prepare_worlds(
    worlds,
    base_root,
    source_root,
    *,
    repo_root,
    frozen_sources,
    config_sha256,
    lock_sha256,
    world_factory=make_world,
):
    """Private implementation used with development 0/1 only in unit tests."""
    worlds = list(worlds)
    if not worlds or len(set(worlds)) != len(worlds) or any(type(w) is not int for w in worlds):
        raise ValueError("Expected a nonempty ordered sequence of unique integer worlds")
    base_root, source_root, repo_root = [
        Path(p).resolve() for p in (base_root, source_root, repo_root)
    ]
    protected = (source_root, repo_root / "data/bios-organization-v1", repo_root / "data/bios-v1")
    if any(base_root == p or p in base_root.parents or base_root in p.parents for p in protected):
        raise ValueError("Confirmation base root must be isolated from source and development data")
    source_identity = verified_sources(repo_root, source_root, frozen_sources)
    base_root.mkdir(parents=True, exist_ok=True)
    with (base_root / ".preparation.lock").open("a+") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        expected_directories = {f"world-{seed}" for seed in worlds}
        if any(
            p.name.startswith("world-") and p.name not in expected_directories
            for p in base_root.iterdir()
        ):
            raise ValueError("Isolated base root contains worlds outside the locked sequence")
        prior_receipts = [
            base_root / "preparation-audit.json",
            *[base_root / ".preparation-receipts" / f"world-{seed}.json" for seed in worlds],
        ]
        for path in prior_receipts:
            if path.exists() or path.is_symlink():
                if path.is_symlink() or not path.is_file():
                    raise ValueError(f"Invalid prior preparation receipt: {path}")
                prior = json.loads(path.read_text())
                identity = {
                    "config_sha256": config_sha256,
                    "lock_sha256": lock_sha256,
                    "source_files_sha256": source_identity,
                }
                if any(prior.get(key) != value for key, value in identity.items()):
                    raise ValueError(
                        f"Prior preparation belongs to a different frozen lock: {path}"
                    )
                if path.name == "preparation-audit.json" and prior.get("worlds") != worlds:
                    raise ValueError("Prior preparation world sequence changed")
        file_hashes = {}
        for seed in worlds:
            verified_sources(repo_root, source_root, frozen_sources)
            final = base_root / f"world-{seed}"
            if final.is_symlink():
                raise ValueError(f"Prepared world cannot be a symlink: {final}")
            # A killed process can leave an unpublished hidden staging directory.
            # It is never loaded or selected; a fresh deterministic candidate is
            # generated and checked against any already committed three-file world.
            with tempfile.TemporaryDirectory(
                prefix=f".world-{seed}-staging-", dir=base_root
            ) as tmp:
                staging = Path(tmp)
                world = world_factory(seed, source_root)
                if world.seed != seed:
                    raise ValueError("Generator returned a different world identity")
                world.save(staging)
                restored = load_world(staging)
                recorded_audit = json.loads((staging / "audit.json").read_text())
                if restored.seed != seed or audit_world(restored) != recorded_audit:
                    raise ValueError("Saved world does not reproduce its truth audit")
                expected = {name: digest(staging / name) for name in WORLD_FILES}
                verified_sources(repo_root, source_root, frozen_sources)
                if final.exists():
                    if not final.is_dir() or {p.name for p in final.iterdir()} != set(WORLD_FILES):
                        raise ValueError(f"Committed world has missing or extra files: {final}")
                    if any(
                        (final / name).is_symlink() or digest(final / name) != checksum
                        for name, checksum in expected.items()
                    ):
                        raise ValueError(
                            f"Committed world differs from deterministic generation: {final}"
                        )
                else:
                    # Whole-directory publication is atomic on this same filesystem.
                    staging.rename(final)
                receipt = {
                    "world": seed,
                    "config_sha256": config_sha256,
                    "lock_sha256": lock_sha256,
                    "files_sha256": expected,
                    "source_files_sha256": source_identity,
                }
                immutable_json(base_root / ".preparation-receipts" / f"world-{seed}.json", receipt)
                file_hashes.update(
                    {f"world-{seed}/{name}": checksum for name, checksum in expected.items()}
                )
        verified_sources(repo_root, source_root, frozen_sources)
        for relative, expected in file_hashes.items():
            if digest(base_root / relative) != expected:
                raise ValueError(f"Prepared world changed before completion: {relative}")
        audit = {
            "complete": True,
            "config_sha256": config_sha256,
            "lock_sha256": lock_sha256,
            "worlds": worlds,
            "files_sha256": file_hashes,
            "source_files_sha256": source_identity,
        }
        immutable_json(base_root / "preparation-audit.json", audit)
        return audit


def prepare(config_path, lock_path):
    # Imported here so pure preparation-helper tests never initialize a model.
    from llm_memory_editability.bios_shortcut_confirmation import validate_frozen_design

    config_path, lock_path = Path(config_path).resolve(), Path(lock_path).resolve()
    config, lock = validate_frozen_design(config_path, lock_path)
    if config["worlds"] != list(range(100, 108)):
        raise ValueError("Formal preparation requires all eight reserved worlds in order")
    return _prepare_worlds(
        config["worlds"],
        ROOT / config["base_world_root"],
        ROOT / config["source_root"],
        repo_root=ROOT,
        frozen_sources=lock["data_source_files"],
        config_sha256=digest(config_path),
        lock_sha256=digest(lock_path),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    args = parser.parse_args()
    audit = prepare(args.config, args.lock)
    print(
        json.dumps(
            {
                "complete": audit["complete"],
                "worlds": audit["worlds"],
                "files": len(audit["files_sha256"]),
            }
        )
    )


if __name__ == "__main__":
    main()
