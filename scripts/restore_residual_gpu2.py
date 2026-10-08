"""Release the recorded admission collision without restarting either training."""

import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

CONTAINER = "lm-residual-cache-2-w730011-mhc-local-r4-scale_off"
ROOT = Path("/ossfs/workspace/llm-memory-editability/results/residual-cache-comparison-v1")
GPU = "2"


def inspect(context, names):
    return (
        json.loads(
            subprocess.check_output(["docker", "--context", context, "inspect", *names], text=True)
        )
        if names
        else []
    )


def conflicts():
    result = []
    for context in ["lm-memory", "default", "d157"]:
        names = subprocess.check_output(
            ["docker", "--context", context, "ps", "-q"], text=True
        ).split()
        for item in inspect(context, names):
            if item["Name"].lstrip("/") == CONTAINER:
                continue
            for request in item["HostConfig"].get("DeviceRequests") or []:
                if request.get("Count") == -1 or GPU in (request.get("DeviceIDs") or []):
                    result.append(
                        dict(
                            context=context,
                            container=item["Name"],
                            started=item["State"]["StartedAt"],
                        )
                    )
    return result


if __name__ == "__main__":
    record = dict(
        state="paused_for_gpu_collision",
        container=CONTAINER,
        gpu=int(GPU),
        paused_utc=datetime.now(timezone.utc).isoformat(),
        reason="two independent controllers admitted the same free GPU within 138ms",
        conflicts=conflicts(),
        scientific_recipe_changed=False,
        restart=False,
        timing_interpretation=(
            "exclude this interrupted interval from isolated throughput comparisons"
        ),
    )
    path = ROOT / "gpu2-scheduling-intervention.json"
    path.write_text(json.dumps(record, indent=2) + "\n")
    started = time.monotonic()
    while True:
        own = inspect("lm-memory", [CONTAINER])[0]
        if not own["State"]["Running"]:
            record["state"] = "container_ended_before_restore"
            break
        if not own["State"]["Paused"]:
            record["state"] = "already_restored"
            break
        if not conflicts():
            subprocess.run(["docker", "--context", "lm-memory", "unpause", CONTAINER], check=True)
            record["state"] = "restored"
            break
        time.sleep(5)
    record.update(
        restored_utc=datetime.now(timezone.utc).isoformat(),
        paused_seconds=time.monotonic() - started,
    )
    path.write_text(json.dumps(record, indent=2) + "\n")
