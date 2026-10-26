"""Add newly authorized GPU4 to the waiting norm-handoff batch, once only."""

from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace


def eligible(module, gpu, prior, contexts):
    if gpu in (1, 2) and not module.prior_lane_released(prior, gpu):
        return False
    try:
        module.gpu_available(gpu, contexts)
    except RuntimeError:
        return False
    return True


def dynamic_wave(controller, stages, module, prior):
    pending = list(controller.config["arms"])
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        active = {}

        def work(arm, gpu):
            for stage in stages:
                controller.stage(arm, stage, None if stage == "cloud" else gpu)

        while pending or active:
            for gpu, future in list(active.items()):
                if future.done():
                    future.result()  # Failed work is never returned to the queue.
                    del active[gpu]
            for gpu in (4, 1, 2):
                if not pending:
                    break
                if gpu in active or not eligible(module, gpu, prior, controller.contexts):
                    continue
                arm = pending.pop(0)
                module.write(
                    controller.output / "dispatch" / f"{arm}-{stages[0]}.json",
                    {"gpu": gpu, "arm": arm, "stages": stages, "utc": module.utc()},
                )
                active[gpu] = pool.submit(work, arm, gpu)
            if pending or active:
                module.time.sleep(15)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o022)
    base = args.base.resolve()
    source = base / "launcher-source/scripts/launch_norm_handoff.py"
    spec = importlib.util.spec_from_file_location("frozen_local_launcher", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = base / "run"
    amendment = module.read(base / "gpu4-amendment.json")
    request = module.read(output / "launch-request.json")
    if (
        module.sha(source) != request["launcher_sha256"]
        or module.sha(base / "bundle/bundle.json") != request["bundle_sha256"]
        or module.sha(Path(__file__)) != amendment["dispatcher_sha256"]
        or module.sha(output / "launch-request.json") != amendment["original_request_sha256"]
        or module.platform.node() != request["hostname"]
        or amendment["gpus"] != [4, 1, 2]
    ):
        raise ValueError("Scheduling amendment identity mismatch")
    config = module.verify_bundle(base / "bundle")
    config["runtime"]["image"] = request["image"]
    actual = json.loads(
        module.checked(["docker", "--context", "lm-memory", "image", "inspect", request["image"]])
    )[0]["Id"]
    if actual != request["image"]:
        raise ValueError("Runtime image changed")
    launch_args = SimpleNamespace(
        bundle=base / "bundle",
        output=output,
        docker_context="lm-memory",
        reservation_context=["default"],
        gpus=[4, 1, 2],
        cpus_per_run=request["cpus_per_run"],
        memory_gib=request["memory_gib"],
        wandb_key_file=Path(amendment["wandb_key_file"]),
        wait_for_prior_batch=None,
    )
    for context in ("lm-memory", "default"):
        module.local_engine(context)
    prior = Path(request["wait_for_prior_batch"])
    with (output / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        receipt = base / "gpu4-controller-process.json"
        if receipt.exists():
            raise FileExistsError("GPU4 dispatcher already claimed; inspect, do not retry")
        for folder in (output / "execution").glob("*/*"):
            if not (folder / "exit.json").is_file() or module.read(folder / "exit.json")["code"]:
                raise RuntimeError(f"Unfinished or failed stage requires review: {folder}")
        module.write(receipt, {"pid": os.getpid(), "utc": module.utc()})
        module.write(output / "controller-process.json", module.read(receipt))

        class Controller(module.Controller):
            def stage(self, arm, stage, gpu=None):
                if stage == "baseline-replay":
                    while True:
                        available = [
                            index
                            for index in (4, 1, 2)
                            if eligible(module, index, prior, self.contexts)
                        ]
                        if available:
                            gpu = available[0]
                            module.write(
                                output / "waiting/all-baseline-replay.json",
                                {"state": "released", "gpu": gpu, "utc": module.utc()},
                            )
                            break
                        module.write(
                            output / "waiting/all-baseline-replay.json",
                            {"state": "waiting", "candidate_gpus": [4, 1, 2], "utc": module.utc()},
                        )
                        module.time.sleep(15)
                return super().stage(arm, stage, gpu)

            def wave(self, stages):
                dynamic_wave(self, stages, module, prior)

        module.write(output / "controller-state.json", {"state": "running", "utc": module.utc()})
        try:
            Controller(launch_args, config).run()
        except BaseException as error:
            module.write(
                output / "controller-state.json",
                {"state": "needs_attention", "error": repr(error), "utc": module.utc()},
            )
            raise
        module.write(output / "controller-state.json", {"state": "complete", "utc": module.utc()})


if __name__ == "__main__":
    main()
