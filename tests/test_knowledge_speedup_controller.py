import hashlib
import importlib.util
import io
import json
import sys
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "knowledge_speedup", ROOT / "scripts/accelerate_knowledge_factorial.py"
)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def test_admission_only_shares_with_known_same_batch_containers():
    row = {"used": 33000, "free": 65000}
    allowed = ["lm-memory:/known", "active CUDA process 123"]
    assert M.admission(row, allowed, ["known"], [123], [40000], 35000)
    for foreign in ["default:/reserved", "lm-memory:/other", "active CUDA process 124"]:
        assert not M.admission(row, allowed + [foreign], ["known"], [123], [40000], 35000)


def test_admission_keeps_peak_memory_margin_and_two_container_limit():
    assert not M.admission({"used": 80000, "free": 18000}, [], [], [], [40000], 20000)
    assert not M.admission({"used": 10000, "free": 88000}, [], [], [], [60000], 35000)
    assert not M.admission({"used": 10000, "free": 88000}, [], [], [], [10000, 10000], 10000)


def test_completed_stage_never_reenters_training():
    assert M.next_stage("train") == "audit"
    assert M.next_stage("audit") == "evaluate"
    assert M.next_stage("evaluate") is None
    with pytest.raises(ValueError):
        M.next_stage("prepare-shard")


def test_extension_copy_is_root_owned_readonly_and_checks_bytes(tmp_path):
    content = b"print('checked')\n"
    (tmp_path / "entry.py").write_bytes(content)
    (tmp_path / "manifest.json").write_text(
        json.dumps({"sha256": {"entry.py": hashlib.sha256(content).hexdigest()}})
    )
    with tarfile.open(fileobj=io.BytesIO(M.extension_archive(tmp_path))) as archive:
        assert archive.getmember("speedup").mode == 0o555
        entry = archive.getmember("speedup/entry.py")
        assert entry.uid == 0 and entry.mode == 0o444
        assert archive.extractfile(entry).read() == content
    (tmp_path / "entry.py").write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed before copy"):
        M.extension_archive(tmp_path)
