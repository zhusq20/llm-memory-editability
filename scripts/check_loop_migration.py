#!/usr/bin/env python3
"""Compare uninterrupted and fresh-process resumed CUDA updates, without W&B."""

import argparse
import copy
import json
import os
import sys
from pathlib import Path

sys.path[:0] = ["/project/src", "/project/scripts", str(Path(__file__).parent)]
sys.path.append(str(Path(os.environ["LOOP_PARENT_SOURCE"]) / "scripts"))

import numpy as np  # noqa: E402
import run_loop_robustness as runner  # noqa: E402
import torch  # noqa: E402
from resume_loop_training import build_resume  # noqa: E402


def equal(a, b, path="state"):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b), path
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b), path
    elif isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for key in a:
            equal(a[key], b[key], f"{path}/{key}")
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            equal(x, y, f"{path}/{i}")
    else:
        assert a == b, path


class PlannedInterruption(Exception):
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "resume"))
    parser.add_argument("--job", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    native = runner.native
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda:0")
    plan = json.loads(Path("configs/loop-invariant-robustness-v1.json").read_text())
    job = plan["jobs"][args.job]
    config = copy.deepcopy(json.loads((Path("/project") / job["config"]).read_text()))
    config["training"]["steps"] = 8
    config["evaluation_nodes"] = [0, 4, 8]
    config["heartbeat_steps"] = 2
    config_path = args.output / "engineering-config.json"
    if args.mode == "prepare":
        args.output.mkdir(parents=True, exist_ok=False)
        native.write_json(config_path, config)
        native.train(
            config,
            job["arm"],
            args.output / "continuous",
            False,
            config_path,
            None,
            device,
            limit_rows=8,
        )
        interrupted = args.output / "interrupted"

        def track(_):
            def log(values, step):
                if step == 4 and (interrupted / "latest.pt").exists():
                    saved = torch.load(
                        interrupted / "latest.pt", weights_only=False, map_location="cpu"
                    )
                    if saved["step"] == 4:
                        raise PlannedInterruption()

            return log

        try:
            native.train(
                config, job["arm"], interrupted, False, config_path, track, device, limit_rows=8
            )
        except PlannedInterruption:
            pass
        else:
            raise AssertionError("Planned checkpoint interruption did not occur")
        native.write_json(args.output / "prepared.json", {"passed": True, "step": 4})
        return
    assert json.loads((args.output / "prepared.json").read_text())["passed"]
    resume = build_resume(native)
    resume(
        config,
        job["arm"],
        args.output / "interrupted",
        False,
        config_path,
        None,
        device,
        limit_rows=8,
    )
    left = torch.load(args.output / "continuous/latest.pt", map_location="cpu", weights_only=False)
    right = torch.load(
        args.output / "interrupted/latest.pt", map_location="cpu", weights_only=False
    )
    for key in left:
        if key != "training_seconds":
            equal(left[key], right[key], key)
    lh = json.loads((args.output / "continuous/learning.json").read_text())
    rh = json.loads((args.output / "interrupted/learning.json").read_text())
    for a, b in zip(lh, rh, strict=True):
        for key in a:
            if key not in ("wall_seconds", "training_seconds"):
                equal(a[key], b[key], key)
    native.write_json(
        args.output / "complete.json",
        {
            "passed": True,
            "fresh_process_resume": True,
            "bitwise_equal": [k for k in left if k != "training_seconds"],
            "learning_metrics_equal": True,
            "engineering_updates": 16,
            "device": torch.cuda.get_device_name(),
            "utc": native.utc(),
        },
    )
    print("CUDA checkpoint recovery matches uninterrupted training exactly", flush=True)


if __name__ == "__main__":
    os.umask(0o022)
    main()
