"""Freeze and execute the two-arm, fixed-reader memory replacement study."""

from __future__ import annotations

import argparse
import itertools
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from llm_memory_editability.memory_reuse import build_world, data_audit, run, sha, write_json

ROOT = Path(__file__).resolve().parents[1]
SOURCES = [
    "src/llm_memory_editability/memory_reuse.py",
    "src/llm_memory_editability/hebbian_interface.py",
    "scripts/run_memory_reuse.py",
    "tests/test_memory_reuse.py",
]


def utc():
    return datetime.now(timezone.utc).isoformat()


def freeze(config_path):
    config = json.loads(Path(config_path).read_text())
    target = Path(config["lock"])
    if target.exists():
        raise FileExistsError(target)
    source = Path(config["spec"]["source_path"])
    commit = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    assert commit == config["spec"]["source_commit"]
    assert not subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain"], text=True
    )
    paths = SOURCES + [config_path, config["plan"]]
    paths += [str(p) for p in sorted((source / "src/hebbian").rglob("*.py"))]
    paths += [str(source / "LICENSE")]
    for path in paths:
        destination = target.parent / "source" / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    audits = {}
    for world_seed in config["worlds"]:
        world = build_world(config["spec"], world_seed)
        audits[world_seed] = data_audit(config["spec"], world)
        np.savez_compressed(target.parent / f"world-{world_seed}.npz", **world)
    write_json(target.parent / "data-audit.json", audits)
    write_json(
        target,
        dict(
            created_utc=utc(),
            config=config_path,
            config_sha256=sha(config_path),
            files={p: sha(p) for p in paths},
            git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            official_commit=commit,
            environment=dict(
                python=platform.python_version(),
                torch=torch.__version__,
                numpy=np.__version__,
                cuda=torch.version.cuda,
            ),
        ),
    )


def verify(config_path):
    config = json.loads(Path(config_path).read_text())
    lock = json.loads(Path(config["lock"]).read_text())
    assert lock["config_sha256"] == sha(config_path)
    for path, digest in lock["files"].items():
        assert sha(path) == digest, f"Changed after freeze: {path}"
    return config


def run_name(world, init, arm):
    return f"w{world}-i{init}-{arm}"


def execute_one(config_path, world, init, arm, device):
    config = verify(config_path)
    assert world in config["worlds"] and init in config["initializations"] and arm in config["arms"]
    output = Path(config["output_root"]) / run_name(world, init, arm)
    metadata = output / "metadata.json"
    if metadata.exists():
        prior = json.loads(metadata.read_text())
        assert prior["lock_sha256"] == sha(config["lock"])
        if (output / "summary.json").exists():
            print(json.dumps(dict(status="already_complete", path=str(output))))
            return
    else:
        write_json(
            metadata,
            dict(
                started_utc=utc(),
                world=world,
                init=init,
                arm=arm,
                lock_sha256=sha(config["lock"]),
                argv=sys.argv,
            ),
        )
    try:
        result = run(config["spec"], world, init, arm, output, device)
    except Exception as error:
        write_json(output / "failure.json", dict(utc=utc(), error=repr(error)))
        raise
    write_json(
        output / "completed.json",
        dict(completed_utc=utc(), summary_sha256=sha(output / "summary.json")),
    )
    print(
        json.dumps(
            dict(status="complete", run=run_name(world, init, arm), endpoints=result["endpoints"])
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["freeze", "run", "run-all", "status"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--world", type=int)
    parser.add_argument("--init", type=int)
    parser.add_argument("--arm", choices=["ce", "aligned"])
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.action == "freeze":
        freeze(args.config)
        return
    config = verify(args.config)
    if args.action == "run":
        execute_one(args.config, args.world, args.init, args.arm, args.device)
        return
    for world, init, arm in itertools.product(
        config["worlds"], config["initializations"], config["arms"]
    ):
        if args.action == "run-all":
            execute_one(args.config, world, init, arm, args.device)
        else:
            path = Path(config["output_root"]) / run_name(world, init, arm) / "status.json"
            print(run_name(world, init, arm), path.read_text() if path.exists() else "pending")


if __name__ == "__main__":
    main()
