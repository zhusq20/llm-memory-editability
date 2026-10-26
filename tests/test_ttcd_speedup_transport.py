import hashlib
import importlib.util
import io
import json
import subprocess
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location(
    "transport", Path(__file__).resolve().parents[1] / "scripts/ttcd_speedup_transport.py"
)
transport = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transport)


@pytest.fixture
def runtime(tmp_path):
    source = tmp_path / "private/source"
    source.mkdir(parents=True)
    (source / "code.py").write_text("value = 3\n")
    manifest = source.parent / "source-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "files": {"code.py": hashlib.sha256((source / "code.py").read_bytes()).hexdigest()},
                "numerical_verification": {"passed": True},
            }
        )
    )
    source.parent.chmod(0o700)
    manifest.chmod(0o600)
    return source, manifest


def command(source, manifest):
    return [
        "run",
        "--detach",
        "--name",
        "formal",
        "--gpus",
        "device=GPU-fixed",
        "--mount",
        "type=bind,source=/old,target=/frozen,readonly",
        "--mount",
        f"type=bind,source={source},target=/run/ttcd-speedup,readonly",
        "--mount",
        f"type=bind,source={manifest},target=/run/ttcd-speedup-manifest.json,readonly",
        "--env",
        f"LM_TTCD_SPEEDUP_MANIFEST_SHA256={hashlib.sha256(manifest.read_bytes()).hexdigest()}",
        "image",
        "python",
        "/run/ttcd-speedup/scripts/run_ttcd_speedup.py",
        "--",
        "/frozen/scripts/run_ttcd_architecture.py",
        "train",
        "--model",
        "stream",
    ]


def test_archive_exact_bytes_private_host_unchanged(runtime):
    source, manifest = runtime
    data, sha = transport.runtime_archive(source, manifest)
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        for item in archive.getmembers():
            assert item.uid == item.gid == 0
            assert item.mode == (0o555 if item.isdir() else 0o444)
        assert (
            archive.extractfile("ttcd-speedup/code.py").read() == (source / "code.py").read_bytes()
        )
        assert archive.extractfile("ttcd-speedup-manifest.json").read() == manifest.read_bytes()
    assert sha == hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert source.parent.stat().st_mode & 0o777 == 0o700
    assert manifest.stat().st_mode & 0o777 == 0o600


def test_command_changes_only_transport(runtime):
    source, manifest = runtime
    args = command(source, manifest)
    transformed = transport.create_command(
        args, source, manifest, hashlib.sha256(manifest.read_bytes()).hexdigest()
    )
    assert transformed[0] == "create" and "--detach" not in transformed
    assert transformed[transformed.index("image") :] == args[args.index("image") :]
    assert "device=GPU-fixed" in transformed
    assert "type=bind,source=/old,target=/frozen,readonly" in transformed
    assert not any("target=/run/ttcd-speedup" in x for x in transformed)


def test_copy_before_start_and_preserve_container_id(runtime, tmp_path):
    source, manifest = runtime
    calls = []

    def docker(*args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, "container-id\n", "")

    def copy(args, **kwargs):
        calls.append(tuple(args))
        assert kwargs["input"]
        return subprocess.CompletedProcess(args, 0, b"", b"")

    resources = SimpleNamespace(docker=docker)
    transport.install(resources, source, manifest, tmp_path / "receipts", copy_runner=copy)
    result = resources.docker(*command(source, manifest))
    assert result.stdout == "container-id\n"
    assert calls[0][0] == "create" and calls[1][3] == "cp" and calls[2] == ("start", "formal")
    assert json.loads((tmp_path / "receipts/formal.json").read_text())["state"] == "started"
    resources.docker("inspect", "formal")
    assert calls[-1] == ("inspect", "formal")


def test_modified_runtime_rejected(runtime):
    source, manifest = runtime
    (source / "code.py").write_text("changed\n")
    with pytest.raises(ValueError, match="bytes changed"):
        transport.runtime_archive(source, manifest)


def test_missing_mount_rejected(runtime):
    source, manifest = runtime
    args = command(source, manifest)
    index = args.index(f"type=bind,source={source},target=/run/ttcd-speedup,readonly")
    del args[index - 1 : index + 1]
    with pytest.raises(ValueError, match="Exactly"):
        transport.create_command(
            args, source, manifest, hashlib.sha256(manifest.read_bytes()).hexdigest()
        )
