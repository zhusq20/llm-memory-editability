"""Confirmation refuses unregistered identities before generating any world."""

import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import torch

from llm_memory_editability.bios_shortcut_confirmation import (
    confirmation_sources,
    file_hash,
    validate_config,
    validate_lock,
)

ROOT = Path(__file__).resolve().parents[1]


def test_frozen_design_rejects_budget_or_sampling_changes():
    config = json.loads((ROOT / "configs/bios-shortcut-confirmation-v1.json").read_text())
    validate_config(config)
    for key, value in (
        ("worlds", list(range(100, 109))),
        ("steps", 30720),
        ("edit_scope", "all"),
        ("supports", [0]),
        ("edit_sampling", "different after viewing outcomes"),
    ):
        with pytest.raises(ValueError):
            validate_config({**config, key: value})


def test_prospective_lock_binds_sources_environment_output_and_matrix(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_bytes((ROOT / "configs/bios-shortcut-confirmation-v1.json").read_bytes())
    output_root = tmp_path / "outputs"
    args = Namespace(
        world=100,
        seed=0,
        condition="company",
        phase="low",
        device="cuda",
        output=output_root / "low/width-256/world-100-seed-0-company",
    )
    lock = {
        "status": "frozen_before_any_confirmation_model_or_outcome",
        "config_sha256": file_hash(config_path),
        "sources": confirmation_sources(),
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "numpy": np.__version__,
        },
        "output_root": str(output_root),
        "device_type": "cuda",
        "data_source_files": {
            "data/bios-source/manifest.json": file_hash(ROOT / "data/bios-source/manifest.json")
        },
        "analysis_sources": {
            "scripts/archive_bios_mechanism_stage.py": file_hash(
                ROOT / "scripts/archive_bios_mechanism_stage.py"
            )
        },
    }
    lock_path = tmp_path / "lock.json"
    lock_path.write_text(json.dumps(lock))
    validate_lock(config_path, lock_path, args)
    for key, value in (
        ("status", "prepared"),
        ("sources", {}),
        ("environment", {}),
        ("config_sha256", "changed"),
        ("analysis_sources", {}),
        ("data_source_files", {}),
    ):
        lock_path.write_text(json.dumps({**lock, key: value}))
        with pytest.raises(ValueError):
            validate_lock(config_path, lock_path, args)
    lock_path.write_text(json.dumps(lock))
    args.device = "cpu"
    with pytest.raises(ValueError, match="CUDA BF16"):
        validate_lock(config_path, lock_path, args)
    args.device = "cuda"
    args.world = 0
    with pytest.raises(ValueError, match="reserved worlds"):
        validate_lock(config_path, lock_path, args)
    args.world = 100
    args.output = tmp_path / "other-output"
    with pytest.raises(ValueError, match="Output differs"):
        validate_lock(config_path, lock_path, args)
    assert not output_root.exists()
