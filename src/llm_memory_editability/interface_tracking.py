"""Use the existing W&B sidecar, finishing only after the independent audit."""

import argparse
import fcntl
import importlib
import json
import os
import time
from pathlib import Path

from . import experiment_tracking as tracking


class ContainerSDK:
    def __init__(self, sdk, runtime):
        self.sdk, self.runtime = sdk, runtime

    def __getattr__(self, name):
        return getattr(self.sdk, name)

    def Settings(self, **options):
        options.pop("x_stats_pid", None)
        if self.runtime:
            options["x_stats_pid"] = self.runtime["host_pid"]
            options["x_stats_gpu_device_ids"] = (self.runtime["physical_gpu"],)
        return self.sdk.Settings(**options)


class AuditedRunTracker(tracking.ArtifactRunTracker):
    def __init__(self, sdk, batch, path, settings):
        runtime_path = path / "container-runtime.json"
        runtime = tracking.read_json(runtime_path)
        super().__init__(ContainerSDK(sdk, runtime), batch, path, settings)
        self.run.config.update(
            {
                key: self.metadata[key]
                for key in (
                    "parent_spec",
                    "cases",
                    "learning_step_unit",
                    "status_step_unit",
                )
                if key in self.metadata
            }
        )
        if runtime:
            self.run.config.update({"container_runtime": runtime})

    def poll(self):
        history = tracking.read_json(self.path / "learning.json")
        previous = None
        for record in history:
            step = int(record["step"])
            if step > self.cursor:
                metrics = tracking.learning_metrics(record, previous)
                for key in ("case", "case_id", "condition", "variant", "arm"):
                    if key in record:
                        metrics["condition/" + key] = record[key]
                if "optimizer_updates" in record:
                    metrics["evaluation/index"] = step
                    metrics["condition/step_unit"] = "evaluation_index"
                    metrics.pop("perf/updates_per_second", None)
                    metrics.pop("perf/training_ms_per_update", None)
                    if previous is not None:
                        seconds = record.get("wall_seconds", 0) - previous.get("wall_seconds", 0)
                        updates = record["optimizer_updates"] - previous.get("optimizer_updates", 0)
                        if seconds > 0:
                            metrics["perf/updates_per_second"] = updates / seconds
                self.run.log(metrics, step=step)
                self.cursor = step
                self.state["last_logged_step"] = step
                tracking.write_json(self.state_path, self.state)
            previous = record
        status_path = self.path / "status.json"
        if status_path.exists():
            self.run.summary["training_state"] = tracking.read_json(status_path).get("state")
        failed = (self.path / "failure.json").exists()
        audit_path = self.path / "audit.json"
        audit = tracking.read_json(audit_path) if audit_path.exists() else {}
        audited = audit.get("passed") is True
        complete = (self.path / "complete.json").exists() and audited
        self.run.summary["has_failure_record"] = failed
        self.run.summary["independently_reloaded"] = audited
        if complete or failed:
            self.run.summary["scientific_final_step"] = self.cursor
            self.run.finish(exit_code=1 if failed else 0)
            # Preserve the authoritative resume cursor even when every point
            # was already on the server and this process logged no new points.
            self.state["last_logged_step"] = self.cursor
            tracking.write_json(self.state_path, self.state)
            return True
        return False


def discover_runs(runs_dir):
    return {
        path
        for path in Path(runs_dir).glob("*")
        if (path / "run.json").exists() and (path / "learning.json").exists()
    }


def final_step(path):
    history = tracking.read_json(path / "learning.json")
    return max((int(record["step"]) for record in history), default=-1)


def drain_runs(sdk, root, runs_dir, settings, *, once=False, tracker_factory=None):
    """Drain late-created runs and late final nodes before terminal exit.

    SDK initialization and uploads may take longer than an entire scientific
    run. Consequently, a scan made before those calls is never an exit proof.
    Finished runs stay in this process's cursor map and are reopened only if
    their saved learning trajectory acquired an unlogged final node.
    """
    root, runs_dir = Path(root), Path(runs_dir)
    tracker_factory = tracker_factory or AuditedRunTracker
    active, finished = {}, {}
    try:
        while True:
            for path in sorted(discover_runs(runs_dir)):
                if path in active or path in finished:
                    continue
                active[path] = tracker_factory(sdk, root, path, settings)
                print(json.dumps({"run": path.name, "url": active[path].run.url}), flush=True)
            for path, tracker in list(active.items()):
                current = tracking.read_json(path / "run.json")
                if current.get("pid") != tracker.metadata.get("pid"):
                    tracker.close()
                    tracker = tracker_factory(sdk, root, path, settings)
                    active[path] = tracker
                if tracker.poll():
                    finished[path] = tracker.cursor
                    del active[path]
            if once:
                return {str(path): cursor for path, cursor in finished.items()}
            manifest = root / "controller-state.json"
            state = tracking.read_json(manifest) if manifest.exists() else {}
            terminal = state.get("state") in {
                "complete",
                "finished_with_failures",
                "development_prerequisite_not_met",
            }
            # Rescan AFTER all SDK calls and after reading the controller's
            # terminal state, so runs born during initialization cannot vanish.
            fresh = discover_runs(runs_dir)
            stale = {path for path, cursor in finished.items() if final_step(path) > cursor}
            for path in stale:
                del finished[path]
            if terminal and not active:
                expected = {runs_dir / name for name in state.get("completed", [])}
                if missing := expected - fresh:
                    raise RuntimeError(f"Completed runs lack logging artifacts: {sorted(missing)}")
                if fresh <= finished.keys():
                    result = {
                        "passed": True,
                        "registered_runs": len(finished),
                        "runs": {
                            path.name: {
                                "last_logged_step": finished[path],
                                "final_step": final_step(path),
                            }
                            for path in sorted(fresh)
                        },
                    }
                    tracking.write_json(root / "tracking-completion.json", result)
                    return result
            if fresh - (active.keys() | finished.keys()):
                continue
            time.sleep(settings["poll_seconds"])
    finally:
        for tracker in active.values():
            tracker.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--runs-dir", type=Path)
    parser.add_argument("--defaults", type=Path, default=tracking.DEFAULTS)
    parser.add_argument("--entity")
    parser.add_argument("--project")
    parser.add_argument("--mode", choices=("online", "offline"))
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    settings = tracking.read_json(args.defaults)
    for key in ("entity", "project", "mode"):
        settings[key] = (
            getattr(args, key) or os.environ.get(f"WANDB_{key.upper()}") or settings[key]
        )
    if not settings["enabled"]:
        raise SystemExit("Experiment tracking is disabled in the selected defaults")
    local = args.root / "tracking-wandb"
    local.mkdir(parents=True, exist_ok=True)
    sdk = importlib.import_module("wandb")
    with (local / "tracker.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        tracking.write_json(args.root / "tracking-ready.json", {"sdk_version": sdk.__version__})
        drain_runs(sdk, args.root, args.runs_dir or args.root / "runs", settings, once=args.once)


if __name__ == "__main__":
    main()
