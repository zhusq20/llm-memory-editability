#!/usr/bin/env python3
"""Freeze the authorized baseline extension without altering active batch sources."""

import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BATCH = "loop-baseline-robustness-v1"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    args = parser.parse_args()
    base = args.base.resolve()
    plan = json.loads((ROOT / f"configs/{BATCH}.json").read_text())
    previous = Path(plan["runtime"]["persistent_root"]) / "loop-invariant-robustness-v1"
    parent = previous / "source"
    base.mkdir(parents=True, exist_ok=False)
    source = base / "source"
    # Freeze the already validated dependencies, overlaying only the explicit changes.
    entries = {
        p.relative_to(parent): p
        for pattern in ("src/**/*.py", "scripts/*.py")
        for p in parent.glob(pattern)
    }
    for p in [
        ROOT / "src/llm_memory_editability/loop_native_handoff.py",
        ROOT / "scripts/run_loop_native_handoff.py",
        ROOT / "scripts/run_loop_robustness.py",
        ROOT / "scripts/launch_loop_robustness.py",
        Path(__file__),
        ROOT / "pyproject.toml",
        ROOT / "configs/loop-native-handoff-development-v1.json",
        ROOT / f"configs/{BATCH}.json",
        *ROOT.glob(f"configs/{BATCH}/*.json"),
        *ROOT.glob("configs/loop-invariant-robustness-v1/*.json"),
        ROOT / f"docs/development-artifacts/{BATCH}/protocol.md",
        ROOT / "tests/test_loop_native_handoff.py",
        ROOT / "tests/test_loop_robustness.py",
        ROOT / "tests/test_loop_robustness_controller.py",
        ROOT / "tests/test_loop_baseline_robustness.py",
    ]:
        entries[p.relative_to(ROOT)] = p

    def copy_entry(item):
        relative, path = item
        target = source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        return str(target.relative_to(base)), hashlib.sha256(target.read_bytes()).hexdigest()

    with concurrent.futures.ThreadPoolExecutor(max_workers=24) as pool:
        files = dict(pool.map(copy_entry, entries.items()))
    for condition in ("original", "new-world", "many-to-one"):
        config = json.loads((source / f"configs/{BATCH}/{condition}.json").read_text())
        link = base / "data-reference" / condition
        link.parent.mkdir(exist_ok=True)
        link.symlink_to(config["input_root"], target_is_directory=True)
        for name in ("world.npz", "panels.npz", "complete.json"):
            path = link / name
            files[str(path.relative_to(base))] = hashlib.sha256(path.read_bytes()).hexdigest()
    (base / "depth").mkdir()
    (base / "depth/old-invariant").symlink_to(
        previous / "attempt-2/depth/old-invariant", target_is_directory=True
    )
    engineering = base / "engineering"
    engineering.mkdir()
    for name in ("lbr-v1-contracts", "lbr-v1-contracts-v2", "lbr-v1-contracts-v3"):
        result = subprocess.run(
            ["docker", "--context", "lm-memory", "logs", name],
            capture_output=True,
            text=True,
            check=True,
        )
        (engineering / f"{name}.log").write_text(result.stdout + result.stderr)
    # Explicit directory access for the root-squashed Docker bind-mount daemon.
    for path in [base, *base.rglob("*")]:
        if path.is_dir() and not path.is_symlink():
            path.chmod(0o755)
    (base / "freeze.json").write_text(
        json.dumps({"batch": BATCH, "image": plan["runtime"]["image"], "files": files}, indent=2)
        + "\n"
    )
    print(json.dumps({"base": str(base), "frozen_files": len(files)}), flush=True)


if __name__ == "__main__":
    os.umask(0o022)
    main()
