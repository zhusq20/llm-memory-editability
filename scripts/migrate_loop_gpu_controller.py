#!/usr/bin/env python3
"""Adopt the two explicitly migrated runs and preserve the existing queue order."""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import fcntl
import os
import socket
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--wandb-token-file", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.base / "control"))
    import resume_loop_transfer as previous

    original, read = previous.original, previous.read
    amendment = read(args.amendment)

    class Controller(previous.Controller):
        def __init__(self):
            original.Controller.__init__(self, args.base, args.wandb_token_file)
            self.amendment_path, self.amendment = args.amendment, amendment
            for file, expected in amendment["original_sha256"].items():
                if original.digest(args.base / file) != expected:
                    raise ValueError(f"Frozen operational source changed: {file}")
            self.plan["gpus"] = [1, 6, 3, 2]
            self.plan["cpu_sets"] = ["70,71", "72,73", "74,75", "78,79"]
            self.plan["gpu_uuids"].update(amendment["target_gpu_uuids"])
            self.plan["ignored_created_ids"].append(amendment["released_reservation_id"])
            self.state = copy.deepcopy(read(args.base / "controller-state.json"))
            self.state.pop("waiting_gpu", None)
            self.state.pop("waiting_reasons", None)
            self.checked = set()

        def existing_container(self, job, stage):
            if stage != "train" or job not in amendment["migrations"]:
                return super().existing_container(job, stage)
            expected = amendment["migrations"][job]
            receipt = read(self.base / "execution" / job / stage / "container.json")
            if receipt["id"] != expected["new_container_id"]:
                raise ValueError("Migrated container receipt differs")
            c = previous.inspect_container(receipt["id"])
            if (
                c["Name"] != "/" + expected["new_container_name"]
                or c["Image"] != self.plan["image"]
            ):
                raise ValueError("Migrated container identity differs")
            argv = c["Config"]["Cmd"]
            if argv[argv.index("--job") + 1] != job or "train" not in argv:
                raise ValueError("Migrated container command differs")
            return c

        def run(self):
            locks = []
            for path in [
                self.base / "controller.lock",
                *map(Path, self.plan["source_controller_locks"]),
            ]:
                handle = path.open("a+")
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                locks.append(handle)
            if (args.amendment.parent / "controller-process.json").exists():
                raise FileExistsError("Migration controller already submitted")
            for job in amendment["migrations"]:
                if not read(args.amendment.parent / "engineering" / job / "complete.json")[
                    "passed"
                ]:
                    raise RuntimeError("Fresh-process CUDA resume check not passed")
            for job in self.tasks:
                task = self.tasks[job]
                if (
                    task["kind"] == "new"
                    and original.digest(Path(task["source"]) / task["config"])
                    != task["config_sha256"]
                ):
                    raise ValueError("Frozen scientific configuration changed")
            process = {
                "pid": os.getpid(),
                "host": socket.gethostname(),
                "utc": original.utc(),
                "controller_sha256": original.digest(__file__),
                "resumed_from_pid": amendment["previous_controller"]["pid"],
                "schedule_amendment": str(args.amendment),
            }
            original.write(args.amendment.parent / "controller-process.json", process)
            original.write(self.base / "controller-process.json", process)
            self.update(
                state="running",
                available_gpu_pool=self.plan["gpus"],
                resource_expansion=False,
                gpu_migration=True,
            )
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                    for index, phase in enumerate(self.plan["phases"]):
                        self.update(phase=index)
                        self.run_phase(phase, pool)
                self.update(state="complete")
            except Exception as error:
                self.update(state="failed", error=str(error))
                raise

    Controller().run()


if __name__ == "__main__":
    os.umask(0o022)
    main()
