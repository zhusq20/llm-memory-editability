"""Run only the original frozen post-training stages; never train or retry."""

from __future__ import annotations

import concurrent.futures
import fcntl
import importlib.util
import json
import os
import shutil
import subprocess
import time
import traceback
from pathlib import Path

PROJECT = Path("/mnt/disk2_from_server2/siqizhu4/llm-memory-editability")
BATCH = Path("/ossfs/workspace/llm-memory-editability/ttcd-second-hop-scratch-v1")
CONTROL = Path(__file__).resolve().parent
IMAGE = "sha256:9643a7d6aeda1b4c9ac2ac50c2d84474b082d649a87e423bc6e0bdc0994097ee"
STAGES = ("train-audit", "calibrate", "evaluate", "audit", "cloud-audit")
ARMS = ("discoloop", "loop")
spec = importlib.util.spec_from_file_location("resource_helpers", CONTROL / "resource_helpers.py")
resource = importlib.util.module_from_spec(spec)
spec.loader.exec_module(resource)
resource.ROOT = PROJECT


def verify_frozen():
    manifest = resource.read_json(BATCH / "source-manifest.json")
    for relative, digest in manifest["files"].items():
        if resource.sha256(BATCH / "source" / relative) != digest:
            raise ValueError(f"Original frozen source changed: {relative}")
    if resource.sha256(BATCH / "source/configs/experiment.json") != manifest["config_sha256"]:
        raise ValueError("Frozen configuration changed")
    return manifest


def prepare_inputs(config):
    directory = BATCH / "input-mount" / config["data_dir"]
    directory.mkdir(parents=True, exist_ok=False)
    for path in (BATCH / "input-mount", directory.parent, directory):
        path.chmod(0o755)
    files = {}
    for name in (*resource.INPUTS, "model-config.json"):
        original = BATCH / "source/input-backup" / name
        target = directory / ("gpt2-config.json" if name == "model-config.json" else name)
        shutil.copy2(original, target)
        target.chmod(0o644)
        if resource.sha256(target) != resource.sha256(original):
            raise ValueError("Input copy differs from original frozen bytes")
        files[str(target.relative_to(BATCH / "input-mount"))] = resource.sha256(target)
    resource.write_json(
        BATCH / "input-mount-manifest.json",
        {"files": files, "source_manifest_sha256": resource.sha256(BATCH / "source-manifest.json")},
    )


def command_for(config, arm, stage, row):
    name, command = resource.build_command(
        config,
        arm,
        stage,
        IMAGE,
        row,
        BATCH / arm,
        BATCH,
        BATCH,
        Path("/home/siqizhu4/.cache/llm-memory-ttcd/wandb-token"),
    )
    # The science entrypoint is the untouched original, not the continual entrypoint.
    command[command.index("/frozen/scripts/run_ttcd_continual.py")] = (
        "/post-control/cloud_audit.py"
        if stage == "cloud-audit"
        else "/frozen/scripts/run_ttcd_second_hop.py"
    )
    image_index = command.index(IMAGE)
    command[image_index:image_index] = [
        "--mount",
        f"type=bind,source={CONTROL},target=/post-control,readonly",
    ]
    return name, command


def run_arm(config, arm):
    out = BATCH / arm
    state = {"model": arm}

    def update(value):
        state.update(value)
        resource.write_json(CONTROL / f"{arm}-state.json", {**state, "utc": resource.utc_now()})

    try:
        with (out / "post-training.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with (out / "execution.lock").open("a") as science_lock:
                fcntl.flock(science_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(science_lock, fcntl.LOCK_UN)
            deadline = time.monotonic() + 12 * 3600
            lock_dir = Path("/tmp") / f"lm-memory-gpu-locks-{os.getuid()}"
            lock_dir.mkdir(exist_ok=True)
            for stage in STAGES:
                verify_frozen()
                update({"state": "admitting", "stage": stage})
                with (lock_dir / "shared-admission-gpu-8.lock").open("a") as admission:
                    row = resource.wait_for_resources(config, arm, admission, deadline, update)
                    resource.reserve_stage(out, stage)
                    name, command = command_for(config, arm, stage, row)
                    if resource.inspect_container(name) is not None:
                        raise FileExistsError(f"Refusing existing container: {name}")
                    resource.write_json(CONTROL / f"{arm}-{stage}-admission.json", row)
                    resource.write_json(CONTROL / f"{arm}-{stage}-command.json", command)
                    with (out / f"{stage}.log").open("x") as log:
                        started = subprocess.run(
                            command, capture_output=True, text=True, timeout=60
                        )
                        log.write(started.stdout + started.stderr)
                        if started.returncode:
                            raise RuntimeError(f"Docker start failed: {started.stderr.strip()}")
                    update(
                        {
                            "state": "running",
                            "stage": stage,
                            "container": name,
                            "container_id": started.stdout.strip(),
                            "gpu": 8,
                        }
                    )
                with (out / f"{stage}.log").open("a") as log:
                    code = resource.monitor_container(name, log, deadline, update)
                resource.write_json(
                    out / f"exit-{stage}.json",
                    {
                        "returncode": code,
                        "gpu": 8,
                        "gpu_uuid": row["uuid"],
                        "image": IMAGE,
                        "cpu_set": config["runtime"]["cpu_sets"][arm],
                        "container": name,
                        "utc": resource.utc_now(),
                        "frozen_science_unchanged": True,
                    },
                )
                print(json.dumps({"model": arm, "stage": stage, "returncode": code}), flush=True)
                if code:
                    update({"state": "needs_attention", "returncode": code})
                    return code
            verify_frozen()
            update({"state": "complete", "stages": list(STAGES)})
            return 0
    except Exception:
        update({"state": "needs_attention", "traceback": traceback.format_exc()})
        print(traceback.format_exc(), flush=True)
        return 1


def main():
    with (BATCH / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = verify_frozen()
        original = resource.read_json(BATCH / "source/configs/experiment.json")
        for arm in ARMS:
            summary = resource.read_json(BATCH / arm / "training/summary.json")
            if summary["step"] != 30000 or summary["parent_sha256"] != resource.sha256(
                BATCH / arm / "training/parent.pt"
            ):
                raise ValueError("Completed terminal checkpoint is not bound to training receipt")
            for stage in STAGES:
                if (BATCH / arm / f"claim-{stage}.json").exists():
                    raise FileExistsError("This post-training stage was already claimed")
        prepare_inputs(original)
        config = json.loads(json.dumps(original))
        cross_stages = ("train", "train-audit", "calibrate", "evaluate", "audit", "cloud-audit")
        own_names = [f"{original['batch']}-{a}-{s}" for a in ARMS for s in STAGES]
        config["parent"] = {"parent_image_sha256": IMAGE}
        config["runtime"].update(
            {
                "gpus": dict.fromkeys(ARMS, 8),
                "cpu_sets": {"discoloop": [4, 5, 6, 7], "loop": [0, 1, 2, 3]},
                "shared_gpu_allowlist": {
                    arm: [f"ttcd-cross-sample-v1-single-{s}" for s in cross_stages] + own_names
                    for arm in ARMS
                },
                "shared_gpu_min_free_mib": 16384,
                "uid": os.getuid(),
                "gid": os.getgid(),
            }
        )
        resource.write_json(
            CONTROL / "plan.json",
            {
                "started_utc": resource.utc_now(),
                "pid": os.getpid(),
                "stages": list(STAGES),
                "authorization": (
                    "User requested completion of original independent reload audit "
                    "and formal evaluation"
                ),
                "frozen_source_manifest_sha256": resource.sha256(BATCH / "source-manifest.json"),
                "frozen_config_sha256": manifest["config_sha256"],
                "execution_only_runtime": config["runtime"],
                "new_training_updates": 0,
                "reason_for_loop_postprocessing_gpu8": (
                    "GPU9 has unrelated CUDA PIDs; GPU8 admits only the existing "
                    "cross-sample single arm and these postprocessing containers"
                ),
                "automatic_retry": False,
            },
        )
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            codes = dict(zip(ARMS, pool.map(lambda arm: run_arm(config, arm), ARMS), strict=True))
        resource.write_json(
            CONTROL / "completion.json",
            {
                "state": "complete" if not any(codes.values()) else "needs_attention",
                "exit_codes": codes,
                "finished_utc": resource.utc_now(),
                "new_training_updates": 0,
            },
        )
        return int(any(codes.values()))


if __name__ == "__main__":
    raise SystemExit(main())
