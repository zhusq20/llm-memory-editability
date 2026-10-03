"""Frozen, degree-matched 512-chain support experiment and one-process GPU slots."""

from __future__ import annotations

import argparse
import json
import os
import queue
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from llm_memory_editability.grok_depth import utc, write_json
from llm_memory_editability.latent_scaling import build_world as scaling_world
from llm_memory_editability.latent_support import (
    SOURCE_FILES,
    audit,
    build_world,
    construct,
    data_digest,
    exposure_signature,
    file_hash,
    model_digest,
    role_coverage,
    run_name,
    support_components,
    train,
    validate_nodes,
)

ARTIFACTS = Path("docs/development-artifacts/latent-support-v1")
RESULTS = Path("results/latent-support-v1")
CONFIG = Path("configs/latent-support-v1.json")
DESIGN = ARTIFACTS / "design.md"
GRAPHS = Path("docs/development-artifacts/memory-scaling-theory-v1/support-graphs.json")
PRIOR = Path("configs/latent-confirmation-v1.json")
SOURCES = list(
    dict.fromkeys(
        [
            *SOURCE_FILES,
            *json.loads(PRIOR.read_text())["source"],
            "scripts/run_latent_support.py",
            "tests/test_latent_support.py",
            str(GRAPHS),
            str(PRIOR),
        ]
    )
)


def specifications():
    prior = json.loads(PRIOR.read_text())
    graphs = {item["world"]: item for item in json.loads(GRAPHS.read_text())["worlds"]}
    specs = []
    for historical in prior["specs"]:
        if (historical["layers"], historical["repeats"], historical["composition_count"]) != (
            1,
            2,
            "all",
        ):
            continue
        graph = graphs[historical["world"]]
        for arm, key in (
            ("connected", "forest_matched512_available_composite_indices"),
            ("split", "degree_matched_disconnected512_available_composite_indices"),
        ):
            specs.append(
                {
                    **historical,
                    "composition_count": 512,
                    "support": arm,
                    "composition_indices": graph[key],
                }
            )
    if len(specs) != len({run_name(s) for s in specs}) or len(specs) != 12:
        raise ValueError("Expected exactly 12 distinct frozen support runs")
    return specs


def verify_pairs(specs, graphs):
    data = {}
    for world in sorted({s["world"] for s in specs}):
        worlds = {}
        for arm in ("connected", "split"):
            spec = next(s for s in specs if s["world"] == world and s["support"] == arm)
            value = build_world(spec)
            worlds[arm] = value
            components = support_components(value)
            key = "forest_matched512" if arm == "connected" else "degree_matched_disconnected512"
            assert (
                components["components"]
                == graphs[world]["cases"][key]["components_including_uncovered_roles"]
            )
            assert components["isolated_roles"] == 0
            data[f"{world}:{arm}"] = {
                "sha256": data_digest(value),
                "composition_examples": len(value["train_composite"]),
                "familiar_test_examples": len(value["familiar_test"]),
                "strict_test_examples": len(value["strict_test"]),
                "role_coverage": role_coverage(value),
                "components": components,
                "role_degrees_and_marginals": exposure_signature(value),
            }
        assert set(worlds["connected"]) == set(worlds["split"])
        for key in worlds["connected"]:
            if key != "train_composite":
                np.testing.assert_array_equal(worlds["connected"][key], worlds["split"][key])
        assert exposure_signature(worlds["connected"]) == exposure_signature(worlds["split"])
        assert (
            support_components(worlds["connected"])["components"]
            < support_components(worlds["split"])["components"]
        )
    return data


def prepare():
    if CONFIG.exists() or (ARTIFACTS / "frozen-config.json").exists():
        raise FileExistsError("Do not overwrite a frozen experiment")
    specs = specifications()
    source = {path: file_hash(path) for path in SOURCES}
    prior = json.loads(PRIOR.read_text())
    assert all(source[path] == digest for path, digest in prior["source"].items())
    graphs = {item["world"]: item for item in json.loads(GRAPHS.read_text())["worlds"]}
    for world, graph in graphs.items():
        assert file_hash(graph["data_path"]) == graph["data_sha256"]
        spec = next(s for s in specs if s["world"] == world)
        rebuilt = scaling_world(dict(spec, composition_count="all"))
        with np.load(graph["data_path"]) as archived:
            assert set(rebuilt) == set(archived.files)
            for key, value in rebuilt.items():
                np.testing.assert_array_equal(value, archived[key])
    data = verify_pairs(specs, graphs)
    initial = {}
    for spec in specs:
        validate_nodes(spec)
        assert spec["nodes"][-1] == spec["steps"] == 128000
        spec["frozen_data_sha256"] = data[f"{spec['world']}:{spec['support']}"]["sha256"]
        key = (spec["world"], spec["initialization"])
        digest = model_digest(construct(spec, "cpu"))
        if key in initial:
            assert initial[key] == digest
        initial[key] = digest
    config = {
        "created_utc": utc(),
        "phase": "Observed-world mechanism follow-up; fixed degree-matched support comparison",
        "specs": specs,
        "source": source,
        "design_sha256": file_hash(DESIGN),
        "data": data,
        "prior_config_sha256": file_hash(PRIOR),
        "support_graphs": {"path": str(GRAPHS), "sha256": file_hash(GRAPHS)},
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "primary": "Connected minus split full familiar-test generation accuracy at R2, 128k",
        "secondary": ["Fixed full learning trajectories", "Same-weight R1/R2/R3/R4 at 32k/128k"],
        "analysis_unit": "Average two paired initializations within each observed world; "
        "equal-weight mean of three world differences. Retain every pair and denominator.",
        "limitations": "Observed worlds; treatment changes contexts and higher-order pairings. "
        "Equal role counts do not match per-batch order or identify a unique graph mechanism.",
        "budget": {
            "runs": len(specs),
            "updates": sum(s["steps"] for s in specs),
            "supervised_tokens": sum(s["steps"] * s["batch_size"] * 23 // 3 for s in specs),
            "learning_nodes": sum(len(s["nodes"]) for s in specs),
            "repeat_checks": sum(len(s["repeat_nodes"]) * len(s["test_repeats"]) for s in specs),
        },
    }
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    write_json(CONFIG, config)
    for path in SOURCES:
        target = ARTIFACTS / "source" / path
        if target.exists():
            raise FileExistsError(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    shutil.copy2(CONFIG, ARTIFACTS / "frozen-config.json")
    print(json.dumps({"budget": config["budget"], "data": data}), flush=True)


def load_config(path):
    config = json.loads(Path(path).read_text())
    if not all(file_hash(p) == h for p, h in config["source"].items()):
        raise RuntimeError("Experiment source differs from the frozen configuration")
    if file_hash(DESIGN) != config["design_sha256"]:
        raise RuntimeError("Design differs from the frozen configuration")
    if file_hash(GRAPHS) != config["support_graphs"]["sha256"]:
        raise RuntimeError("Support selections differ from the frozen graph analysis")
    if config != json.loads((ARTIFACTS / "frozen-config.json").read_text()):
        raise RuntimeError("Configuration differs from its frozen snapshot")
    for spec in config["specs"]:
        validate_nodes(spec)
        expected = config["data"][f"{spec['world']}:{spec['support']}"]["sha256"]
        if spec["frozen_data_sha256"] != expected or data_digest(build_world(spec)) != expected:
            raise RuntimeError("Run data differ from the frozen matrix")
    return config


def engineering_spec(spec):
    return dict(spec, steps=8, nodes=[0, 8], repeat_nodes=[8], checkpoint_nodes=[0, 8])


def subprocess_stage(command, config_path, spec, device, engineering=False):
    name = run_name(spec)
    folder = RESULTS / f"engineering-preflight-{spec['support']}" if engineering else RESULTS / name
    folder.mkdir(parents=True, exist_ok=True)
    argv = [
        sys.executable,
        __file__,
        command,
        "--config",
        str(config_path),
        "--name",
        name,
        "--device",
        device,
    ]
    if engineering:
        argv.append("--engineering")
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    with (folder / f"{command}-process.log").open("a") as log:
        process = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, env=env, check=False)
    status = {"run": name, "stage": command, "device": device, "returncode": process.returncode}
    write_json(folder / f"{command}-process-status.json", {**status, "utc": utc()})
    print(json.dumps(status), flush=True)
    return status


def preflight(config_path, config, device):
    if (ARTIFACTS / "preflight.json").exists():
        raise FileExistsError("Do not overwrite a completed preflight")
    checks = {}
    for arm in ("connected", "split"):
        spec = next(s for s in config["specs"] if s["support"] == arm)
        folder = RESULTS / f"engineering-preflight-{arm}"
        if (folder / "audit.json").exists():
            raise FileExistsError("Do not overwrite a completed preflight")
        for stage in ("run", "audit"):
            status = subprocess_stage(stage, config_path, spec, device, engineering=True)
            if status["returncode"]:
                raise RuntimeError(f"Engineering {arm} {stage} failed; retained logs in {folder}")
        checks[arm] = json.loads((folder / "audit.json").read_text())
        assert checks[arm]["passed"]
    write_json(
        ARTIFACTS / "preflight.json",
        {
            "scope": "Engineering only; two arms separately trained and independently reloaded",
            "passed": True,
            "source": config["source"],
            "design_sha256": config["design_sha256"],
            "checks": checks,
        },
    )


def execute(config_path, gpus):
    config = load_config(config_path)
    if not gpus or len(gpus) != len(set(gpus)):
        raise ValueError("GPU slots must be nonempty and unique")
    preflight_result = json.loads((ARTIFACTS / "preflight.json").read_text())
    assert preflight_result["passed"]
    assert preflight_result["source"] == config["source"]
    assert preflight_result["design_sha256"] == config["design_sha256"]
    slots = queue.Queue()
    for gpu in gpus:
        slots.put(gpu)
    launched, started = utc(), time.perf_counter()

    def launch(spec):
        name = run_name(spec)
        folder = RESULTS / name
        folder.mkdir(parents=True, exist_ok=True)
        if (folder / "complete.json").exists() and (folder / "audit.json").exists():
            complete = json.loads((folder / "complete.json").read_text())
            checked = json.loads((folder / "audit.json").read_text())
            assert complete["spec"] == spec and complete["source"] == config["source"]
            assert complete["data_sha256"] == spec["frozen_data_sha256"]
            assert checked["passed"] and checked["frozen_data_exact"]
            return {"run": name, "skipped_complete": True}
        gpu = slots.get()
        try:
            commands = ["audit"] if (folder / "complete.json").exists() else ["run", "audit"]
            stages = []
            for command in commands:
                status = subprocess_stage(command, config_path, spec, f"cuda:{gpu}")
                stages.append({**status, "gpu": gpu})
                if status["returncode"]:
                    break
            return {"run": name, "stages": stages, "returncode": stages[-1]["returncode"]}
        finally:
            slots.put(gpu)

    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        statuses = list(pool.map(launch, config["specs"]))
    write_json(
        ARTIFACTS / "execution.json",
        {
            "launched_utc": launched,
            "finished_utc": utc(),
            "seconds": time.perf_counter() - started,
            "gpus": gpus,
            "statuses": statuses,
        },
    )
    if any(s.get("returncode", 0) for s in statuses):
        raise RuntimeError("Failed attempts retained; inspect process logs before any retry")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "preflight", "execute", "run", "audit"))
    parser.add_argument("--config", default=str(CONFIG))
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--name")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--engineering", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
        return
    config = load_config(args.config)
    if args.command == "preflight":
        preflight(args.config, config, args.device)
    elif args.command == "execute":
        execute(args.config, [int(g) for g in args.gpus.split(",")])
    else:
        spec = next(s for s in config["specs"] if run_name(s) == args.name)
        folder = RESULTS / args.name
        if args.engineering:
            folder = RESULTS / f"engineering-preflight-{spec['support']}"
            spec = engineering_spec(spec)
        if args.command == "run":
            train(spec, folder, config["source"], args.device)
        else:
            print(json.dumps(audit(folder, args.device)), flush=True)


if __name__ == "__main__":
    main()
