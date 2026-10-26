"""One sidecar per coverage trajectory, using the project's artifact tracker."""

import argparse
import os
import time
from pathlib import Path

import wandb

from llm_memory_editability.experiment_tracking import ArtifactRunTracker, read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    secret = Path("/run/secrets/wandb")
    if secret.exists():
        os.environ["WANDB_API_KEY"] = secret.read_text().strip()
    settings = read_json("configs/experiment-tracking-defaults.json")
    tracker = ArtifactRunTracker(wandb, args.batch, args.run, settings)
    try:
        while not tracker.poll():
            time.sleep(settings["poll_seconds"])
    except BaseException:
        tracker.close()
        raise


if __name__ == "__main__":
    main()
