"""W&B sidecar for future experiments; reads trainer artifacts without changing training."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

DEFAULTS = Path(__file__).resolve().parents[2] / "configs/experiment-tracking-defaults.json"
PRIVATE_KEYS = {
    "api_key",
    "wandb_api_key",
    "access_token",
    "auth_token",
    "password",
    "credentials",
    "secret",
}


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temp.replace(path)


def public_config(value):
    if isinstance(value, dict):
        return {k: public_config(v) for k, v in value.items() if k.lower() not in PRIVATE_KEYS}
    if isinstance(value, list):
        return [public_config(v) for v in value]
    return value


def numeric_fields(value, prefix):
    result = {}
    for key, item in value.items():
        name = f"{prefix}/{key}"
        if isinstance(item, dict):
            result.update(numeric_fields(item, name))
        elif isinstance(item, int | float) and not isinstance(item, bool) and math.isfinite(item):
            result[name] = item
    return result


def learning_metrics(record, previous=None):
    """Keep the actual optimizer step and task denominators; derive interval throughput."""
    result = {"training/step": int(record["step"])}
    result.update(numeric_fields(record.get("metrics", {}), "eval"))
    groups = {
        "training_seconds": "perf/training_seconds",
        "wall_seconds": "perf/wall_seconds",
        "examples": "exposure/examples",
        "supervised_tokens": "exposure/supervised_tokens",
        "executed_input_tokens": "compute/executed_input_tokens",
        "estimated_matmul_training_flops": "compute/estimated_matmul_training_flops",
        "atomic_epochs": "exposure/atomic_epochs",
        "composition_epochs": "exposure/composition_epochs",
    }
    for key, value in record.items():
        if key in {"step", "metrics"}:
            continue
        if isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value):
            result[groups.get(key, f"training/{key}")] = value
    if previous is not None and record["step"] > previous["step"]:
        steps = record["step"] - previous["step"]
        if "training_seconds" in record and "training_seconds" in previous:
            seconds = record["training_seconds"] - previous["training_seconds"]
            if seconds > 0:
                result["perf/training_ms_per_update"] = 1000 * seconds / steps
                result["perf/updates_per_second"] = steps / seconds
                if "examples" in record and "examples" in previous:
                    result["perf/examples_per_second"] = (
                        record["examples"] - previous["examples"]
                    ) / seconds
    return result


class ArtifactRunTracker:
    def __init__(self, sdk, batch, path, settings):
        self.path = Path(path)
        self.state_path = Path(batch) / "tracking-wandb" / self.path.name / "state.json"
        self.metadata = read_json(self.path / "run.json")
        self.settings = settings
        identity = {
            "batch": Path(batch).name,
            "name": self.path.name,
            "spec": public_config(self.metadata.get("spec", {})),
            "world_sha256": self.metadata.get("world_sha256"),
            "initial_model_sha256": self.metadata.get("initial_model_sha256"),
        }
        fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        saved = read_json(self.state_path) if self.state_path.exists() else {}
        if saved:
            assert saved["fingerprint"] == fingerprint, "Source trajectory identity changed"
            assert saved["entity"] == settings["entity"] and saved["project"] == settings["project"]
        self.state = {
            "run_id": fingerprint[:16],
            "fingerprint": fingerprint,
            "entity": settings["entity"],
            "project": settings["project"],
            "last_logged_step": -1,
            **saved,
        }
        options = {"disable_git": True}
        if not settings["system_metrics"]:
            options["x_disable_stats"] = True
        else:
            if "gpu" in self.metadata:
                options["x_stats_gpu_device_ids"] = (self.metadata["gpu"],)
            if "pid" in self.metadata:
                options["x_stats_pid"] = self.metadata["pid"]
        mode = settings["mode"]
        run_id = self.state["run_id"]
        if mode == "offline" and saved:
            segment = saved.get("offline_segment", 0) + 1
            self.state["offline_segment"] = segment
            run_id = f"{run_id[:12]}{segment:04x}"
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        config = {
            **identity,
            "scientific_run_id": self.state["run_id"],
            **{
                k: self.metadata[k]
                for k in [
                    "model",
                    "phase",
                    "evaluation_nodes",
                    "execution_lock_sha256",
                    "parameters",
                    "vocab_size",
                    "atomic_examples",
                    "composition_examples",
                    "dtype",
                    "gpu_name",
                ]
                if k in self.metadata
            },
        }
        self.run = sdk.init(
            entity=settings["entity"],
            project=settings["project"],
            group=self.metadata.get("tracking_group", Path(batch).name),
            name=self.path.name,
            id=run_id,
            config=config,
            mode=mode,
            resume="allow" if mode == "online" else None,
            job_type=self.metadata.get("job_type", "training"),
            tags=self.metadata.get("tags", []),
            dir=str(self.state_path.parent),
            save_code=False,
            reinit="create_new",
            settings=sdk.Settings(**options),
        )
        self.run.define_metric("training/step")
        self.run.define_metric("*", step_metric="training/step")
        if mode == "online":
            # The public history is authoritative: an SDK restart can briefly report step 0,
            # and a local cursor can include points queued before a network interruption.
            remote = sdk.Api(timeout=30).run(f"{settings['entity']}/{settings['project']}/{run_id}")
            self.cursor = remote.lastHistoryStep
        else:
            self.cursor = self.state["last_logged_step"]
        self.state.update(url=self.run.url, mode=mode, wandb_run_id=run_id)
        write_json(self.state_path, self.state)

    def poll(self):
        history = read_json(self.path / "learning.json")
        previous = None
        for record in history:
            step = int(record["step"])
            if step > self.cursor:
                self.run.log(learning_metrics(record, previous), step=step)
                self.cursor = step
                self.state["last_logged_step"] = step
                write_json(self.state_path, self.state)
            previous = record
        status_path = self.path / "status.json"
        if status_path.exists():
            status = read_json(status_path)
            self.run.summary["training_state"] = status.get("state", "unknown")
        complete = (self.path / "complete.json").exists()
        failure_record = (self.path / "failure.json").exists() or (
            self.path / "failure-detail.json"
        ).exists()
        failed = failure_record and not complete
        self.run.summary["has_failure_record"] = failure_record
        self.run.summary["independently_reloaded"] = (self.path / "audit.json").exists()
        if complete or failed:
            self.run.summary["scientific_final_step"] = self.cursor
            self.run.finish(exit_code=1 if failed else 0)
            return True
        return False

    def close(self):
        self.run.finish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--runs-dir", type=Path)
    parser.add_argument("--defaults", type=Path, default=DEFAULTS)
    parser.add_argument("--entity")
    parser.add_argument("--project")
    parser.add_argument("--mode", choices=["online", "offline"])
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--detach", action="store_true")
    args = parser.parse_args()
    settings = read_json(args.defaults)
    for name in ["entity", "project", "mode"]:
        settings[name] = (
            getattr(args, name) or os.environ.get(f"WANDB_{name.upper()}") or settings[name]
        )
    if not settings["enabled"]:
        raise SystemExit("Experiment tracking is disabled in the selected defaults")
    local = args.root / "tracking-wandb"
    local.mkdir(parents=True, exist_ok=True)
    if args.detach:
        metadata_path = local / "process.json"
        if metadata_path.exists():
            pid = read_json(metadata_path)["pid"]
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                pass
            else:
                raise SystemExit(f"Tracking process already running: {pid}")
        arguments = [a for a in sys.argv[1:] if a != "--detach"]
        command = [
            sys.executable,
            "-u",
            "-m",
            "llm_memory_editability.experiment_tracking",
            *arguments,
        ]
        with (local / "tracker.log").open("a") as log:
            process = subprocess.Popen(
                command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
        write_json(metadata_path, {"pid": process.pid, "command": command})
        print(json.dumps({"tracker_pid": process.pid, "log": str(local / "tracker.log")}))
        return
    try:
        sdk = importlib.import_module("wandb")
    except ImportError as error:
        raise SystemExit("Install the tracking extra, or use .venv-wandb/bin/python") from error
    with (local / "tracker.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        active, finished = {}, set()
        runs_dir = args.runs_dir or args.root / "development"
        try:
            while True:
                for path in sorted(runs_dir.glob("*")):
                    if path in active or path in finished:
                        continue
                    if not (path / "run.json").exists() or not (path / "learning.json").exists():
                        continue
                    active[path] = ArtifactRunTracker(sdk, args.root, path, settings)
                    print(json.dumps({"run": path.name, "url": active[path].run.url}), flush=True)
                for path, tracker in list(active.items()):
                    current = read_json(path / "run.json")
                    if current.get("pid") != tracker.metadata.get("pid"):
                        tracker.close()
                        tracker = ArtifactRunTracker(sdk, args.root, path, settings)
                        active[path] = tracker
                    if tracker.poll():
                        finished.add(path)
                        del active[path]
                if args.once:
                    break
                manifest = args.root / "controller-state.json"
                if manifest.exists() and not active:
                    if read_json(manifest).get("state") in {"complete", "finished_with_failures"}:
                        break
                time.sleep(settings["poll_seconds"])
        finally:
            for tracker in active.values():
                tracker.close()


if __name__ == "__main__":
    main()
