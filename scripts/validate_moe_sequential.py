"""Run actual isolated CUDA preflights and read-only ordinary endpoint checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from execute_learning_use import claim_gpu
from run_memory_interface_next import now, write


def validate(config_path):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    project = Path(__file__).resolve().parents[1]
    artifact = project / "docs/development-artifacts" / config["batch"]
    source = Path(config.get("source_root", project))
    jobs = [
        (spec["name"], index, gpu, False)
        for index, (spec, gpu) in enumerate(zip(config["specs"], config["gpus"], strict=True))
    ]
    jobs.append(("reused-ordinary", None, 4, True))
    locks, workers = [], []
    try:
        for name, index, gpu, reused in jobs:
            lock = claim_gpu(gpu)
            if lock is None:
                raise RuntimeError(f"GPU {gpu} is occupied; no other experiment is preempted")
            locks.append(lock)
            out = artifact / "preflight" / name
            out.mkdir(parents=True, exist_ok=False)
            container = "lm-moe-preflight-" + hashlib.sha256(str(out).encode()).hexdigest()[:16]
            command = [
                "docker",
                "--context",
                "lm-memory",
                "run",
                "--name",
                container,
                "--label",
                "project=llm-memory-editability",
                "--label",
                "batch=" + config["batch"],
                "--network=none",
                "--gpus",
                f"device={gpu}",
                "--cpus",
                "4",
                "--memory",
                "32g",
                "--read-only",
                "--tmpfs",
                "/tmp:rw,size=2g",
                "--shm-size=1g",
                "--mount",
                f"type=bind,src={project},dst={project},readonly",
                "--mount",
                f"type=bind,src={out},dst={out}",
                "--workdir",
                str(project),
                "--env",
                "MEMORY_INTERFACE_CONTAINER=1",
                "--env",
                "PYTHONDONTWRITEBYTECODE=1",
                "--env",
                "OMP_NUM_THREADS=1",
                "--env",
                "PYTHONPATH=" + str(source / "src"),
                config["runtime"]["image"],
                config["runtime"]["python"],
                "-u",
                "-m",
                "llm_memory_editability.moe_sequential",
                "--config",
                str(config_path),
                "--out",
                str(out),
                "--device",
                "cuda:0",
            ]
            if reused:
                command.append("--verify-reused")
            else:
                command += ["--preflight", "--run", config["specs"][index]["name"]]
            write(out / "command.json", {"command": command, "physical_gpu": gpu})
            log = (out / "output.log").open("w")
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            workers.append((name, out, process, log, reused))
        results = []
        for name, out, process, log, reused in workers:
            returncode = process.wait()
            log.close()
            path = out / ("reuse-validation.json" if reused else "preflight.json")
            result = json.loads(path.read_text()) if path.exists() else {}
            passed = returncode == 0 and result.get("passed") is True
            results.append(
                {"name": name, "passed": passed, "returncode": returncode, "evidence": str(path)}
            )
        # Check real-sized ordinary, expert and wide parameter counts and common tensors.
        ledgers = {}
        for entry in config["reused_runs"]:
            ledger = artifact / "preflight/reused-ordinary" / entry["name"] / "architecture.json"
            ledgers.setdefault("D4", json.loads(ledger.read_text()))
        for spec in config["specs"]:
            ledger = artifact / "preflight" / spec["name"] / "architecture.json"
            if ledger.exists():
                ledgers.setdefault(spec["architecture"], json.loads(ledger.read_text()))
        counts = {
            arm: {
                key: ledger[key]
                for key in ("total_parameters", "nominal_parameters_selected_per_token")
            }
            for arm, ledger in ledgers.items()
        }
        matching = False
        if set(ledgers) == {"D4", "M4", "W4"}:
            m, w, d = (ledgers[arm] for arm in ("M4", "W4", "D4"))
            matching = (
                abs(m["total_parameters"] - w["total_parameters"]) < 0.001 * m["total_parameters"]
            )
            matching &= (
                abs(m["nominal_parameters_selected_per_token"] - d["total_parameters"])
                < 0.001 * d["total_parameters"]
            )
            dense = {
                key: value
                for key, value in d["initialization"]["shared_tensor_sha256"].items()
                if ".mlp." not in key
            }
            matching &= dense == m["initialization"]["shared_tensor_sha256"]
            matching &= dense == w["initialization"]["shared_tensor_sha256"]
        summary = {
            "passed": matching and all(row["passed"] for row in results),
            "jobs": results,
            "real_size_parameter_ledger": counts,
            "paired_common_tensors_and_parameter_matching": matching,
            "created_utc": now(),
        }
        write(artifact / "preflight-summary.json", summary)
        print(json.dumps(summary), flush=True)
        if not summary["passed"]:
            raise SystemExit(1)
    finally:
        # Running isolated preflights are preserved if orchestration fails.
        for _, _, process, log, _ in workers:
            if process.poll() is not None:
                log.close()
        for lock in locks:
            lock.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    validate(parser.parse_args().config)
