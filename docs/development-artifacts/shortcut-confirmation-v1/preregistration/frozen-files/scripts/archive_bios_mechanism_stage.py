"""Archive a quiescent local stage and verify every file against its source hash."""

import argparse
import hashlib
import json
import os
import tarfile
from pathlib import Path


def digest(stream):
    value = hashlib.sha256()
    while chunk := stream.read(4 * 1024 * 1024):
        value.update(chunk)
    return value.hexdigest()


def archive(source, destination, statuses, allow_incomplete=False):
    source, destination = source.resolve(), destination.absolute()
    if source == destination or source in destination.parents:
        raise ValueError("Archive must be outside the source tree")
    if not statuses:
        raise ValueError("At least one queue status is required")
    stage_complete = True
    status_records = {}
    for path in statuses:
        state = json.loads(path.read_text())
        complete = state["state"] == "complete" and not state.get("failed")
        if state.get("active") or (not complete and not allow_incomplete):
            raise ValueError(f"Queue has unfinished work: {path}")
        if not complete and state["state"] not in ("failed", "stopped"):
            raise ValueError(f"Incomplete queue must be terminal before archival: {path}")
        stage_complete &= complete
        status_records[str(path.resolve())] = state
    entries = {}
    snapshots = {}
    for path in sorted(source.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Archive source contains a symbolic link: {path}")
        if not path.is_file():
            continue
        relative = str(path.relative_to(source))
        before = path.stat()
        with path.open("rb") as stream:
            checksum = digest(stream)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError(f"Source changed while hashing: {path}")
        entries[relative] = {"bytes": after.st_size, "sha256": checksum}
        snapshots[relative] = (after.st_size, after.st_mtime_ns)
    if destination.exists() or destination.with_suffix(destination.suffix + ".json").exists():
        raise FileExistsError("Refusing to replace a published archive")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".partial-{os.getpid()}")
    with tarfile.open(temporary, "w") as output:
        for relative in entries:
            path = source / relative
            output.add(path, arcname=relative, recursive=False)
            state = path.stat()
            if snapshots[relative] != (state.st_size, state.st_mtime_ns):
                raise ValueError(f"Source changed while archiving: {path}")
    verified = set()
    with tarfile.open(temporary, "r") as saved:
        for member in saved:
            if not member.isfile() or member.name not in entries or member.name in verified:
                raise ValueError(f"Unexpected archive member: {member.name}")
            with saved.extractfile(member) as stream:
                checksum = digest(stream)
            expected = entries[member.name]
            if checksum != expected["sha256"] or member.size != expected["bytes"]:
                raise ValueError(f"Archive verification failed: {member.name}")
            verified.add(member.name)
    if verified != set(entries):
        raise ValueError("Archive is missing source files")
    os.replace(temporary, destination)
    index = {
        "complete": True,
        "stage_complete": stage_complete,
        "archive_scope": "all extant files; archival completion is not experimental completion",
        "source": str(source),
        "archive": str(destination),
        "files": len(entries),
        "bytes": sum(item["bytes"] for item in entries.values()),
        "every_member_verified": True,
        "queue_statuses": [str(path.resolve()) for path in statuses],
        "queue_status_snapshots": status_records,
        "entries": entries,
        "restore": "Extract into the original source directory to preserve frozen absolute paths.",
    }
    destination.with_suffix(destination.suffix + ".json").write_text(
        json.dumps(index, indent=2) + "\n"
    )
    print(json.dumps({key: value for key, value in index.items() if key != "entries"}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--status", type=Path, nargs="+", required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    archive(args.source, args.destination, args.status, args.allow_incomplete)


if __name__ == "__main__":
    main()
