"""Handle simultaneous GPU admissions while preserving every training process."""

import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/ossfs/workspace/llm-memory-editability/results/residual-cache-comparison-v1")
BATCH = "residual-cache-comparison-v1"


def snapshot():
    result = []
    for context in ["lm-memory", "default", "d157"]:
        ids = subprocess.check_output(
            ["docker", "--context", context, "ps", "-q"], text=True
        ).split()
        if ids:
            items = json.loads(
                subprocess.check_output(
                    ["docker", "--context", context, "inspect", *ids], text=True
                )
            )
            result.extend((context, item) for item in items)
    return result


def devices(item):
    ids = set()
    for request in item["HostConfig"].get("DeviceRequests") or []:
        ids.update(
            map(str, range(8)) if request.get("Count") == -1 else request.get("DeviceIDs") or []
        )
    return ids


if __name__ == "__main__":
    events, managed = [], {}
    initial = ROOT / "gpu2-scheduling-intervention.json"
    if initial.exists():
        row = json.loads(initial.read_text())
        if row["state"] == "paused_for_gpu_collision":
            managed[row["container"]] = time.monotonic()
    while True:
        records = snapshot()
        paused_names = {
            item["Name"].lstrip("/")
            for context, item in records
            if context == "lm-memory" and item["State"]["Paused"]
        }
        managed = {name: began for name, began in managed.items() if name in paused_names}
        for context, own in records:
            if context != "lm-memory" or (own["Config"].get("Labels") or {}).get("batch") != BATCH:
                continue
            name = own["Name"].lstrip("/")
            if "-eng-" in name:
                continue
            competing = [
                other["Name"]
                for _, other in records
                if other["Id"] != own["Id"] and devices(own) & devices(other)
            ]
            action = None
            if competing and not own["State"]["Paused"]:
                action = "pause"
            elif not competing and name in managed and own["State"]["Paused"]:
                action = "unpause"
            if action:
                result = subprocess.run(
                    ["docker", "--context", "lm-memory", action, name], capture_output=True
                )
                if result.returncode == 0:
                    entry = dict(
                        action=action,
                        container=name,
                        competing=competing,
                        utc=datetime.now(timezone.utc).isoformat(),
                    )
                    if action == "pause":
                        managed[name] = time.monotonic()
                    else:
                        entry["paused_seconds"] = time.monotonic() - managed.pop(name)
                    events.append(entry)
                    (ROOT / "gpu-guard-events.json").write_text(json.dumps(events, indent=2) + "\n")
        state = json.loads((ROOT / "controller-state.json").read_text())
        if state["state"] in {"complete", "finished_with_failures"} and not managed:
            break
        time.sleep(3)
