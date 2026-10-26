#!/usr/bin/env python3
"""Archive the registered matrix, data and exact runtime sources once."""

import argparse
import concurrent.futures
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BATCH = "loop-invariant-robustness-v1"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--continue-preparation", action="store_true")
    args = parser.parse_args()
    base = args.base
    if args.continue_preparation:
        if (base / "freeze.json").exists() or (base / "controller-process.json").exists():
            raise ValueError("Only unfinished preparation can continue")
    base.mkdir(parents=True, exist_ok=args.continue_preparation)
    source = base / "source"
    files = [*ROOT.glob("src/**/*.py"), *ROOT.glob("scripts/*.py")]
    files += [
        ROOT / "pyproject.toml",
        ROOT / f"configs/{BATCH}.json",
        ROOT / "configs/loop-native-handoff-development-v1.json",
        *ROOT.glob(f"configs/{BATCH}/*.json"),
        ROOT / f"docs/development-artifacts/{BATCH}/protocol.md",
        ROOT / "tests/test_loop_native_handoff.py",
        ROOT / "tests/test_loop_robustness.py",
        ROOT / "tests/test_loop_robustness_controller.py",
    ]

    def copy_source(path):
        destination = source / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(copy_source, files))
    shutil.copytree(ROOT / f"results/{BATCH}/inputs", base / "inputs", dirs_exist_ok=True)
    shutil.copy2(ROOT / f"results/{BATCH}/existing-evidence.json", base / "existing-evidence.json")
    engineering = base / "engineering"
    engineering.mkdir(exist_ok=True)
    for name in (
        "invariant-stability-env-preflight-20261021",
        "invariant-stability-env-compat-20261021",
        "invariant-stability-contracts-20261021",
        "invariant-stability-contracts-v2-20261021",
        "invariant-stability-contracts-v3-20261021",
        "invariant-stability-check-v4-20261021",
        "invariant-stability-lint-v3-20261021",
        "invariant-stability-prepare-20261021",
    ):
        out = subprocess.run(
            ["docker", "--context", "lm-memory", "logs", name],
            capture_output=True,
            text=True,
            check=True,
        )
        (engineering / f"{name}.log").write_text(out.stdout + out.stderr)
    manifest = {
        "batch": BATCH,
        "image": json.loads((ROOT / f"configs/{BATCH}.json").read_text())["runtime"]["image"],
        "files": {
            str(p.relative_to(base)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(base.rglob("*"))
            if p.is_file()
        },
        "engineering": {
            "contract_tests_passed": 40,
            "ruff_check_passed": True,
            "old_initial_backbone_exact": True,
            "old_checkpoint_prediction_replay": "pending before scientific training",
            "failed_attempts": (
                "Two test collection failures from UID/file access; "
                "no model execution; logs retained."
            ),
        },
    }
    (base / "freeze.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"base": str(base), "frozen_files": len(manifest["files"])}))


if __name__ == "__main__":
    main()
