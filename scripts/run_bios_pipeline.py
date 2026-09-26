"""Run the development phases and preserve their completion or failure status."""

import argparse
import datetime
import subprocess
import sys
from pathlib import Path

from llm_memory_editability.bios_data import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", nargs="+", type=int, required=True)
    parser.add_argument("--output", default="results/bios-dev-v1")
    args = parser.parse_args()
    records = []
    for phase in ("learning", "editing", "measurement"):
        start = datetime.datetime.now(datetime.timezone.utc).isoformat()
        command = [
            sys.executable,
            "scripts/run_bios_development.py",
            "--gpus",
            *map(str, args.gpus),
            "--output",
            args.output,
            "--phase",
            phase,
        ]
        result = subprocess.run(command, check=False)
        records.append(
            {
                "phase": phase,
                "start_utc": start,
                "end_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "returncode": result.returncode,
            }
        )
        write_json(Path(args.output) / "pipeline-status.json", records)
        if result.returncode:
            raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
