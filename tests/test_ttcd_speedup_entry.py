"""Runtime entry plumbing only: no model, optimizer, GPU or network calls."""

import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def fixture(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "runtime_entry_fixture", ROOT / "scripts/run_ttcd_speedup.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    runtime = tmp_path / "source"
    files = {}
    for name in ("ttcd_train_speedup", "ttcd_eval_speedup"):
        relative = f"src/llm_memory_editability/{name}.py"
        path = runtime / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, path)
        files[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "files": files,
                "numerical_verification": {
                    "passed": True,
                    "engineering_fixture_only": True,
                },
            }
        )
    )
    monkeypatch.setattr(module, "ROOT", runtime)
    monkeypatch.setenv("LM_TTCD_SPEEDUP_MANIFEST", str(manifest))
    monkeypatch.setenv("LM_TTCD_SPEEDUP_MANIFEST_SHA256", module.sha(manifest))
    original_sha = module.sha
    monkeypatch.setattr(
        module,
        "sha",
        lambda p: (
            "frozen-fixture-digest"
            if str(p).startswith("/frozen/scripts/")
            else original_sha(p)
        ),
    )
    return module, manifest


def test_exact_original_arguments_checkpoint_and_tracking_provenance(
    tmp_path, monkeypatch
):
    module, _ = fixture(tmp_path, monkeypatch)
    out = tmp_path / "output"
    args = [
        "/frozen/scripts/run_ttcd_architecture.py",
        "train",
        "--model",
        "stream",
        "--out",
        str(out),
    ]
    observed = {}
    monkeypatch.setattr(
        sys, "argv", [str(ROOT / "scripts/run_ttcd_speedup.py"), "--", *args]
    )
    monkeypatch.setitem(
        sys.modules,
        "wandb",
        SimpleNamespace(init=lambda **kw: observed.update(tracking=kw)),
    )

    def frozen_cli(path, run_name):
        assert path == args[0] and run_name == "__main__" and sys.argv == args
        torch.save(
            {"schema": "ttcd-verified-base-v1", "identity": {"unchanged": True}},
            out / "fixture.pt",
        )
        sys.modules["wandb"].init(config={"original": True})
        raise SystemExit(0)

    monkeypatch.setattr(module.runpy, "run_path", frozen_cli)
    original_save = torch.save
    with pytest.raises(SystemExit) as result:
        module.main()
    assert result.value.code == 0 and torch.save is original_save
    checkpoint = torch.load(out / "fixture.pt", weights_only=True)
    assert checkpoint["identity"] == {"unchanged": True}
    assert checkpoint["runtime_acceleration"]["original_argv"] == args
    assert observed["tracking"]["config"]["original"] is True
    receipt = json.loads((out / "runtime-acceleration-train.json").read_text())
    assert receipt["state"] == "complete"
    assert receipt["optimizer_steps_for_verification"] == 0
    assert receipt["evaluation_optimization"]["verification_passed"] is True
    assert receipt["training_optimization"]["pruned_calls"] == 0


def test_reject_native_foundation_and_changed_manifest(tmp_path, monkeypatch):
    module, manifest = fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "entry",
            "--",
            "/frozen/scripts/run_ttcd_architecture_foundation.py",
            "train",
            "--out",
            str(tmp_path / "out"),
        ],
    )
    with pytest.raises(ValueError, match="unrelated"):
        module.main()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "entry",
            "--",
            "/frozen/scripts/run_ttcd_architecture.py",
            "train",
            "--out",
            str(tmp_path / "out"),
        ],
    )
    manifest.write_text(manifest.read_text() + " ")
    with pytest.raises(ValueError, match="manifest differs"):
        module.main()


def test_load_missing_supplementary_dependency_only(tmp_path, monkeypatch):
    import llm_memory_editability

    module, _ = fixture(tmp_path, monkeypatch)
    calls, supplied = [], SimpleNamespace()
    monkeypatch.delattr(llm_memory_editability, "ttcd_double_new_eval", raising=False)
    monkeypatch.setattr(module.importlib.util, "find_spec", lambda _: None)
    monkeypatch.setattr(
        module, "load_extension", lambda name: calls.append(name) or supplied
    )
    module.ensure_eval_dependencies()
    assert calls == ["ttcd_double_new_eval"]
    assert llm_memory_editability.ttcd_double_new_eval is supplied


def test_existing_frozen_dependency_is_preserved(tmp_path, monkeypatch):
    import llm_memory_editability

    module, _ = fixture(tmp_path, monkeypatch)
    existing = SimpleNamespace()
    monkeypatch.setattr(
        llm_memory_editability, "ttcd_double_new_eval", existing, raising=False
    )
    monkeypatch.setattr(module.importlib.util, "find_spec", lambda _: SimpleNamespace())
    monkeypatch.setattr(
        module, "load_extension", lambda _: pytest.fail("Existing module replaced")
    )
    module.ensure_eval_dependencies()
    assert llm_memory_editability.ttcd_double_new_eval is existing
