import hashlib
import io
import runpy
import tarfile
from pathlib import Path

import pytest

restore_array_archive = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts/download_experiment_artifacts.py")
)["restore_array_archive"]


def make_bundle(tmp_path, members):
    path = tmp_path / "arrays.tar.gz"
    manifest = {}
    with tarfile.open(path, "w:gz") as archive:
        for name, payload in members.items():
            entry = tarfile.TarInfo(name)
            entry.size = len(payload)
            archive.addfile(entry, io.BytesIO(payload))
            manifest[name] = {
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
    info = {
        "path": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "members": list(members),
    }
    return path, info, manifest


def test_selected_arrays_restore_to_original_paths(tmp_path):
    path, info, manifest = make_bundle(
        tmp_path, {"results/a/one.npz": b"one", "results/b/two.npz": b"two"}
    )
    output = tmp_path / "restored"
    assert restore_array_archive(path, info, manifest, ["results/a/**"], output) == 1
    assert (output / "results/a/one.npz").read_bytes() == b"one"
    assert not (output / "results/b/two.npz").exists()


def test_corrupt_array_does_not_overwrite_existing_file(tmp_path):
    name = "results/a/one.npz"
    path, info, manifest = make_bundle(tmp_path, {name: b"corrupt"})
    manifest[name]["sha256"] = hashlib.sha256(b"correct").hexdigest()
    output = tmp_path / "restored"
    (output / name).parent.mkdir(parents=True)
    (output / name).write_bytes(b"existing")
    with pytest.raises(ValueError, match="Array SHA256 mismatch"):
        restore_array_archive(path, info, manifest, ["results/**"], output)
    assert (output / name).read_bytes() == b"existing"


def test_archive_cannot_write_outside_destination(tmp_path):
    path, info, manifest = make_bundle(tmp_path, {"../outside.npz": b"unsafe"})
    with pytest.raises(ValueError, match="Unsafe archive path"):
        restore_array_archive(path, info, manifest, ["*"], tmp_path / "restored")
    assert not (tmp_path / "outside.npz").exists()


def test_modified_bundle_is_rejected_before_extraction(tmp_path):
    path, info, manifest = make_bundle(tmp_path, {"results/a.npz": b"value"})
    path.write_bytes(path.read_bytes() + b"modified")
    with pytest.raises(ValueError, match="Archive SHA256 mismatch"):
        restore_array_archive(path, info, manifest, ["*"], tmp_path / "restored")
