"""Copy verified runtime bytes into each container without changing host permissions."""

from __future__ import annotations

import hashlib
import io
import json
import subprocess
import tarfile
import threading
from pathlib import Path


def digest(data):
    return hashlib.sha256(data).hexdigest()


def runtime_archive(source, manifest_path):
    source, manifest_path = Path(source).resolve(), Path(manifest_path).resolve()
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if not manifest.get("numerical_verification", {}).get("passed"):
        raise ValueError("A passing exact numerical verification is required")
    files = {}
    for relative, expected in manifest["files"].items():
        path = source / relative
        if not path.resolve().is_relative_to(source) or path.is_symlink():
            raise ValueError("Runtime file escapes its frozen source")
        content = path.read_bytes()
        if digest(content) != expected:
            raise ValueError(f"Frozen runtime bytes changed: {relative}")
        files[f"ttcd-speedup/{relative}"] = content
    files["ttcd-speedup-manifest.json"] = manifest_bytes
    directories = {
        str(parent) for name in files for parent in Path(name).parents if str(parent) != "."
    }
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for directory in sorted(directories, key=lambda value: (value.count("/"), value)):
            info = tarfile.TarInfo(directory)
            info.type, info.mode = tarfile.DIRTYPE, 0o555
            info.uid = info.gid = 0
            archive.addfile(info)
        for name, content in sorted(files.items()):
            info = tarfile.TarInfo(name)
            info.size, info.mode = len(content), 0o444
            info.uid = info.gid = 0
            archive.addfile(info, io.BytesIO(content))
    return buffer.getvalue(), digest(manifest_bytes)


def create_command(args, source, manifest_path, manifest_sha256):
    args = list(args)
    if args[0] != "run" or "/run/ttcd-speedup/scripts/run_ttcd_speedup.py" not in args:
        raise ValueError("Only the registered future runtime command may be copied")
    expected = {
        f"type=bind,source={Path(source).resolve()},target=/run/ttcd-speedup,readonly",
        f"type=bind,source={Path(manifest_path).resolve()},target=/run/ttcd-speedup-manifest.json,readonly",
    }
    if f"LM_TTCD_SPEEDUP_MANIFEST_SHA256={manifest_sha256}" not in args:
        raise ValueError("Runtime manifest environment identity differs")
    command, removed, i = ["create"], [], 1
    while i < len(args):
        if args[i] == "--detach":
            i += 1
        elif args[i] == "--mount" and args[i + 1] in expected:
            removed.append(args[i + 1])
            i += 2
        else:
            command.append(args[i])
            i += 1
    if len(removed) != 2 or set(removed) != expected:
        raise ValueError("Exactly the two reviewed runtime mounts must be removed")
    return command


def install(resources, source, manifest_path, receipt_directory, *, copy_runner=subprocess.run):
    """Keep Docker options and scientific argv; change only extension transport."""
    original = resources.docker
    archive, manifest_sha256 = runtime_archive(source, manifest_path)
    receipt_directory = Path(receipt_directory)
    receipt_directory.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()

    def docker(*args, **kwargs):
        if (
            not args
            or args[0] != "run"
            or "/run/ttcd-speedup/scripts/run_ttcd_speedup.py" not in args
        ):
            return original(*args, **kwargs)
        create = create_command(args, source, manifest_path, manifest_sha256)
        name = create[create.index("--name") + 1]
        path = receipt_directory / f"{name}.json"
        with lock:
            with path.open("x") as handle:
                json.dump(
                    {
                        "state": "creating",
                        "container": name,
                        "actual_create_argv": create,
                        "manifest_sha256": manifest_sha256,
                        "archive_sha256": digest(archive),
                        "host_permissions_changed": False,
                        "container_runtime_readonly_to_user": True,
                    },
                    handle,
                    indent=2,
                )
        created = original(*create, **kwargs)
        # The archive owns runtime files as root:root with 0444/0555 modes.
        # No --archive is needed: Docker's default ownership is also root.
        copied = copy_runner(
            ["docker", "--context", "lm-memory", "cp", "-", f"{name}:/run"],
            input=archive,
            capture_output=True,
            timeout=45,
            check=False,
        )
        if copied.returncode:
            raise RuntimeError(f"Runtime copy failed: {copied.stderr.decode(errors='replace')}")
        started = original("start", name, **kwargs)
        if started.returncode:
            raise RuntimeError("Copied-runtime container did not start")
        receipt = json.loads(path.read_text())
        receipt.update(state="started", container_id=created.stdout.strip())
        temporary = path.with_suffix(".partial")
        temporary.write_text(json.dumps(receipt, indent=2) + "\n")
        temporary.replace(path)
        return created

    resources.docker = docker
    return original
