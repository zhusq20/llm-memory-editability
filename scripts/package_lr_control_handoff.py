#!/usr/bin/env python3
"""Package only the paused four arms, frozen inputs, and their existing checks."""

import hashlib
import json
import shutil
import tarfile
from pathlib import Path

from continue_lr_control_handoff import WAITING


def main():
    batch = Path("/ossfs/workspace/llm-memory-editability/loop-discoloop-lr-control-v1")
    handoff = batch / "handoff"
    receipt = json.loads((handoff / "paused-queue.json").read_text())
    if set(receipt["paused_jobs"]) != set(WAITING):
        raise ValueError("Four-arm pause receipt required")
    scripts = Path(__file__).resolve().parent
    for name in ("continue_lr_control_handoff.py", "run_lr_control_four.sh"):
        shutil.copy2(scripts / name, handoff / name)
    shutil.copy2(
        scripts.parent
        / "docs/development-artifacts/loop-discoloop-lr-control-v1/four-arm-handoff.md",
        handoff / "README.md",
    )
    relatives = [
        "source",
        "input-mount",
        "baseline-reference",
        "source-manifest.json",
        "baseline-reuse.json",
        "decay-contract.json",
        "contracts",
        "handoff/paused-queue.json",
        "handoff/continue_lr_control_handoff.py",
        "handoff/run_lr_control_four.sh",
        "handoff/README.md",
    ]
    for job in WAITING:
        if (batch / "execution" / job / "train").exists():
            raise ValueError(f"Scientific claim exists for {job}; do not export")
        relatives += [f"jobs/{job}", f"execution/{job}"]
    archive = handoff / "four-arm-inputs.tar"
    if archive.exists():
        raise FileExistsError("Archive already exists; do not overwrite")
    with tarfile.open(archive, "w") as stream:
        for relative in relatives:
            stream.add(batch / relative, arcname=relative, recursive=True)
    digest = hashlib.sha256()
    with archive.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    (handoff / "four-arm-inputs.tar.sha256").write_text(f"{digest.hexdigest()}  {archive.name}\n")
    manifest = {
        "archive": str(archive),
        "sha256": digest.hexdigest(),
        "bytes": archive.stat().st_size,
        "jobs": WAITING,
        "credentials_included": False,
        "docker_image_included": False,
    }
    (handoff / "package.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
