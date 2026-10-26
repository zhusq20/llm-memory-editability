"""Private-volume execution, immutable source and independent audit for CE controls."""

from __future__ import annotations

import fcntl
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BATCH = "ce-scope-discoloop-novel-v1"
VOLUME = BATCH + "-private-output"
FILES = (
    "src/llm_memory_editability/ce_scope_editing.py",
    "scripts/run_ce_scope_discoloop.py",
    "scripts/launch_ce_scope_discoloop.py",
    "tests/test_ce_scope_editing.py",
    f"configs/{BATCH}.json",
    f"docs/development-artifacts/{BATCH}/protocol.md",
)


def write(path, value):
    path = Path(path)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def docker(*args, **kwargs):
    return subprocess.run(
        ["docker", "--context", "lm-memory", *args],
        check=True,
        capture_output=True,
        timeout=60,
        **kwargs,
    )


def load_transport(source):
    spec = importlib.util.spec_from_file_location(
        "alpha_transport", source / "scripts/launch_alphaedit_discoloop.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def freeze(plan, batch):
    source = batch / "source"
    if source.exists():
        raise FileExistsError("Batch already frozen; no duplicate submissions")
    parent = Path(plan["parent_source"])
    manifest = parent / "source-manifest.json"
    if sha(manifest) != plan["parent_source_manifest_sha256"]:
        raise ValueError("Paired AlphaEdit source changed")
    original = json.loads(manifest.read_text())
    source.mkdir(mode=0o700, parents=True)
    for relative, expected in original["files"].items():
        path = parent / relative
        if path.is_symlink() or sha(path) != expected:
            raise ValueError(f"Frozen parent changed: {relative}")
        target = source / relative
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    for relative in FILES:
        target = source / relative
        if target.exists():
            raise ValueError("New batch must not replace any parent science source")
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    files = {str(p.relative_to(source)): sha(p) for p in sorted(source.rglob("*")) if p.is_file()}
    write(
        source / "source-manifest.json",
        {
            "batch": BATCH,
            "files": files,
            "parent_manifest_sha256": sha(manifest),
            "new_composition_supervision": 0,
        },
    )
    return source, sha(source / "source-manifest.json")


def persist(batch, name, stage, *, runtime_root="/out", host_root=None):
    destination = host_root or batch / "output"
    destination.mkdir(mode=0o700, parents=True, exist_ok=True)
    process = subprocess.Popen(
        ["docker", "--context", "lm-memory", "cp", f"{name}:{runtime_root}/{stage}", "-"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    files, skipped = {}, []
    with tarfile.open(fileobj=process.stdout, mode="r|") as archive:
        for member in archive:
            relative = Path(member.name)
            if relative.is_absolute() or ".." in relative.parts or relative.parts[0] != stage:
                raise ValueError("Stage archive escaped its output root")
            target = destination / relative
            if member.isdir():
                target.mkdir(mode=0o700, parents=True, exist_ok=True)
            elif member.isfile():
                target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                incoming = archive.extractfile(member)
                digest = hashlib.sha256()
                if target.exists():
                    for part in iter(lambda incoming=incoming: incoming.read(8 * 1024 * 1024), b""):
                        digest.update(part)
                    if sha(target) != digest.hexdigest():
                        raise ValueError("Refusing to overwrite differing existing evidence")
                else:
                    with target.open("xb") as handle:
                        for part in iter(
                            lambda incoming=incoming: incoming.read(8 * 1024 * 1024), b""
                        ):
                            handle.write(part)
                            digest.update(part)
                files[str(relative)] = digest.hexdigest()
            else:
                skipped.append(member.name)
    errors = process.stderr.read().decode()
    if process.wait() != 0:
        raise RuntimeError(f"Private stage persistence failed: {errors}")
    write(
        batch / f"persistence-{stage}.json",
        {
            "files": files,
            "skipped_nonregular": skipped,
            "host_permissions_unchanged": True,
            "private_volume": VOLUME,
        },
    )


def create(plan, source, manifest, stage, gpu):
    name = f"{BATCH}-{stage}"
    args = [
        "create",
        "--name",
        name,
        "--user",
        "1015:1016",
        "--network",
        "host",
        "--cpuset-cpus",
        plan["resources"]["cpus"],
        "--cpus",
        "4",
        "--memory",
        "32g",
        "--tmpfs",
        "/tmp:rw,size=2g",
        "--env",
        "PYTHONDONTWRITEBYTECODE=1",
        "--env",
        "PYTHONPATH=/frozen/src:/frozen/scripts",
        "--env",
        f"LM_SOURCE_MANIFEST_SHA256={manifest}",
        "--env",
        "XDG_CACHE_HOME=/tmp/cache",
        "--env",
        "WANDB_CACHE_DIR=/tmp/wandb-cache",
        "--env",
        "WANDB_CONFIG_DIR=/tmp/wandb-config",
        "--env",
        "WANDB_DATA_DIR=/tmp/wandb-data",
        "--env",
        "RUFF_CACHE_DIR=/tmp/ruff-cache",
        "--mount",
        "type=bind,source=/ossfs/workspace/llm-memory-editability,target=/ossfs/workspace/llm-memory-editability,readonly",
        "--mount",
        f"type=volume,source={VOLUME},target=/out",
        "--mount",
        "type=bind,source=/home/siqizhu4/.cache/llm-memory-ttcd/wandb-token,target=/run/secrets/wandb,readonly",
        "--workdir",
        "/frozen",
    ]
    if stage not in ("contracts", "cloud"):
        args += ["--gpus", f"device={gpu['uuid']}"]
    if stage == "contracts":
        paths = [p for p in FILES if p.endswith(".py")]
        commands = [
            [
                "python",
                "-m",
                "pytest",
                "-q",
                "-p",
                "no:cacheprovider",
                "tests/test_ce_scope_editing.py",
                "tests/test_alphaedit_loop.py",
                "tests/test_alphaedit_summary.py",
            ],
            ["ruff", "check", *paths],
            ["ruff", "format", "--check", *paths],
        ]
        program = "import subprocess; " + "; ".join(
            f"subprocess.run({cmd!r},check=True)" for cmd in commands
        )
        args += [plan["image"], "python", "-c", program]
    else:
        kind, _, scope = stage.partition("-")
        args += [
            plan["image"],
            "python",
            "scripts/run_ce_scope_discoloop.py",
            kind,
            "--plan",
            f"configs/{BATCH}.json",
            "--root",
            "/out",
            "--device",
            "cuda:0",
        ]
        if scope:
            args += ["--scope", scope]
    created = docker(*args).stdout.decode().strip()
    transport = load_transport(source)
    docker("cp", "-", f"{name}:/", input=transport.archive(source))
    docker("start", name)
    return name, created, args


def run_stage(plan, batch, source, manifest, stage, gpu):
    if shutil.disk_usage("/workspaces").free < plan["resources"]["minimum_disk_free_bytes"]:
        raise RuntimeError("Insufficient temporary disk headroom; no existing jobs stopped")
    name, identifier, args = create(plan, source, manifest, stage, gpu)
    write(
        batch / f"launch-{stage}.json",
        {"name": name, "id": identifier, "argv": args, "gpu": gpu},
    )
    started = time.monotonic()
    while True:
        state = json.loads(docker("inspect", "--format", "{{json .State}}", name).stdout)
        write(
            batch / "controller-state.json",
            {
                "state": "running",
                "stage": stage,
                "container": name,
                "container_state": state,
                "gpu": gpu,
                "elapsed_seconds": time.monotonic() - started,
            },
        )
        if not state["Running"]:
            logs = docker("logs", name)
            (batch / f"{stage}.log").write_bytes(logs.stdout + logs.stderr)
            write(batch / f"exit-{stage}.json", state)
            if stage != "contracts":
                persist(batch, name, stage)
            if state["ExitCode"]:
                raise RuntimeError(f"Stage failed and preserved: {stage}")
            if stage == "contracts":
                write(
                    batch / "contracts.json",
                    {
                        "passed": True,
                        "source_sha256": manifest,
                        "log_sha256": sha(batch / "contracts.log"),
                    },
                )
            return
        if time.monotonic() - started > plan["stage_timeout_seconds"]:
            docker("stop", "--time", "30", name)
            raise TimeoutError(f"Only this batch stage stopped after timeout: {stage}")
        time.sleep(15)


def main():
    plan = json.loads((ROOT / f"configs/{BATCH}.json").read_text())
    batch = Path(plan["output_root"])
    batch.mkdir(mode=0o700, parents=True, exist_ok=True)
    with (batch / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (batch / "controller-process.json").exists():
            raise FileExistsError("This authorized batch has already been submitted")
        source, manifest = freeze(plan, batch)
        transport = load_transport(source)
        write(
            batch / "controller-process.json",
            {
                "pid": os.getpid(),
                "source_sha256": manifest,
                "plan_sha256": sha(ROOT / f"configs/{BATCH}.json"),
                "submitted_utc": time.time(),
            },
        )
        gpu, gpu_lock, started = None, None, time.monotonic()
        try:
            docker("volume", "create", VOLUME)
            docker(
                "run",
                "--name",
                BATCH + "-private-init",
                "--user",
                "0:0",
                "--network",
                "none",
                "--read-only",
                "--cpus",
                "1",
                "--memory",
                "1g",
                "--mount",
                f"type=volume,source={VOLUME},target=/out",
                plan["image"],
                "python",
                "-c",
                "import os; os.chown('/out',1015,1016); os.chmod('/out',0o700)",
            )
            run_stage(plan, batch, source, manifest, "contracts", gpu)
            while gpu is None:
                gpu, gpu_lock = transport.idle_gpu()
                if gpu is None:
                    write(
                        batch / "controller-state.json",
                        {"state": "waiting_for_exclusive_idle_gpu"},
                    )
                    if time.monotonic() - started > plan["resource_wait_seconds"]:
                        raise TimeoutError("No exclusive idle GPU became available")
                    time.sleep(30)
            run_stage(plan, batch, source, manifest, "prepare", gpu)
            for scope in ("down", "mlp", "all"):
                run_stage(plan, batch, source, manifest, f"development-{scope}", gpu)
            write(
                batch / "selected-recipes.json",
                {
                    scope: {
                        "development_sha256": sha(
                            batch / f"output/development-{scope}/result.json"
                        ),
                        "selected": json.loads(
                            (batch / f"output/development-{scope}/result.json").read_text()
                        )["selected"],
                    }
                    for scope in ("down", "mlp", "all")
                },
            )
            for scope in ("down", "mlp", "all"):
                run_stage(plan, batch, source, manifest, f"evaluation-{scope}", gpu)
                run_stage(plan, batch, source, manifest, f"audit-{scope}", gpu)
            run_stage(plan, batch, source, manifest, "cloud", gpu)
            write(
                batch / "completion-manifest.json",
                {
                    "passed": True,
                    "source_sha256": manifest,
                    "scopes": ["down", "mlp", "all"],
                    "formal_edits": 96,
                    "formal_updates": 96 * plan["ce"]["steps"],
                    "audit_optimizer_updates": 0,
                    "alphaedit_reused": True,
                    "new_composition_supervision": 0,
                },
            )
            write(
                batch / "controller-state.json",
                {"state": "complete", "passed": True, "gpu": gpu},
            )
        except BaseException as error:
            write(batch / "controller-failure.json", {"error": repr(error)})
            raise
        finally:
            if gpu_lock is not None:
                gpu_lock.close()


if __name__ == "__main__":
    main()
