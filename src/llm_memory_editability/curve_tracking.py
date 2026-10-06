"""Retry transient W&B connection failures without changing scientific trajectories."""

import json
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from . import interface_tracking
from .experiment_tracking import write_json


def utc():
    return datetime.now(timezone.utc).isoformat()


def main():
    import sys

    root = Path(sys.argv[sys.argv.index("--root") + 1])
    for attempt in range(1, 145):
        try:
            interface_tracking.main()
            return
        except Exception:
            path = root / "tracking-retries.json"
            history = json.loads(path.read_text()) if path.exists() else []
            history.append({"attempt": attempt, "utc": utc(), "error": traceback.format_exc()})
            write_json(path, history)
            print(json.dumps({"tracking_retry": attempt, "utc": utc()}), flush=True)
            time.sleep(min(60, 5 * attempt))
    raise RuntimeError("W&B retry budget exhausted; scientific artifacts remain intact")


if __name__ == "__main__":
    main()
