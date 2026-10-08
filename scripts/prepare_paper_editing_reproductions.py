"""Build reviewable full benchmark configs and deterministic development smoke configs."""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
ARTIFACT = PROJECT / "docs/development-artifacts/paper-reproductions-parallel-v1"


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    assert not path.exists(), path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def source_files(root):
    return sorted(
        [path for path in root.rglob("*.py") if ".git" not in path.parts]
        + list(root.rglob("hparams/**/*.json"))
        + list(root.rglob("globals.yml"))
        + [path for path in root.rglob("LICENSE*") if path.is_file()]
        + [path for path in (root / "README.md",) if path.exists()]
    )


def build(benchmark):
    batch = benchmark.lower() + "-paper-reproduction-v1"
    frozen = PROJECT / "docs/development-artifacts" / batch / "source"
    image = subprocess.check_output(
        [
            "docker",
            "--context",
            "lm-memory",
            "image",
            "inspect",
            "lm-memory-paper-editors:torch2.6.0-hf4.31.0-v2",
            "--format",
            "{{.Id}}",
        ],
        text=True,
    ).strip()
    assets = read(ARTIFACT / "download-manifest.json")["files"]
    model = "gpt-j-6B" if benchmark == "MQuAKE" else "gpt2-xl"
    model_name = "EleutherAI/gpt-j-6B" if benchmark == "MQuAKE" else "gpt2-xl"
    upstream = ARTIFACT / ("memit-upstream" if benchmark == "MQuAKE" else "ripple-upstream")
    author = upstream if benchmark == "MQuAKE" else upstream / "src/memit"
    models = PROJECT / "data/paper-reproduction-models-v1" / model
    relevant = [
        r
        for r in assets
        if str(models) + "/" in r["path"]
        or ("stats-v1/" + model_name.replace("/", "_") + "/") in r["path"]
    ]
    fixed_inputs = {r["path"]: r["sha256"] for r in relevant}
    weights = next(
        r for r in relevant if Path(r["path"]).name in {"pytorch_model.bin", "model.safetensors"}
    )
    additional = source_files(upstream)
    if benchmark == "MQuAKE":
        data_root = ARTIFACT / "mquake-upstream"
        additional += list((data_root / "prompts").iterdir())
        additional += [data_root / "datasets/MQuAKE-CF-3k.json", data_root / "README.md"]
    else:
        additional += list((upstream / "data/benchmark").glob("*.json"))
    additional += [
        PROJECT / "tests/test_paper_editing_reproduction.py",
        Path(__file__).resolve(),
        PROJECT / "scripts/fetch_paper_reproduction_assets.py",
    ]
    additional = sorted(set(p for p in additional if p.is_file()))
    input_hashes = dict(fixed_inputs)
    input_hashes.update(
        {
            str(frozen / p.relative_to(PROJECT)): sha(p)
            for p in additional
            if p.is_relative_to(upstream) or (benchmark == "MQuAKE" and p.is_relative_to(data_root))
        }
    )
    config = {
        "batch": batch,
        "module": "mquake_reproduction" if benchmark == "MQuAKE" else "ripple_reproduction",
        "gpus": [4, 5] if benchmark == "MQuAKE" else [6, 7],
        "runtime": {
            "docker_context": "lm-memory",
            "docker_host": "unix:///run/docker-lm-memory/docker.sock",
            "image": image,
            "python": "/opt/conda/bin/python",
            "cpus": 4,
            "memory": "80g" if benchmark == "MQuAKE" else "32g",
            "env": {
                "PYTHONHASHSEED": "0",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "MPLCONFIGDIR": "/tmp/matplotlib",
            },
        },
        "additional_sources": [str(p.relative_to(PROJECT)) for p in additional],
        "specs": [],
    }
    for arm in ["ROME", "MEMIT"] if benchmark == "MQuAKE" else ["popular", "random", "recent"]:
        method = arm if benchmark == "MQuAKE" else "ROME"
        data = (
            data_root / "datasets/MQuAKE-CF-3k.json"
            if benchmark == "MQuAKE"
            else upstream / f"data/benchmark/{arm}.json"
        )
        spec = {
            "name": f"{model}-{method.lower()}" + ("" if benchmark == "MQuAKE" else "-" + arm),
            "benchmark": benchmark,
            "method": method,
            "seed": 42,
            "model_dir": str(models),
            "model_name": model_name,
            "model_weights_sha256": weights["sha256"],
            "model_revision": read(ARTIFACT / "download-manifest.json")["model_revisions"][model][
                1
            ],
            "editor_source": str(frozen / author.relative_to(PROJECT)),
            "hparams_file": str(
                frozen
                / author.relative_to(PROJECT)
                / f"hparams/{method}/{model_name.replace('/', '_')}.json"
            ),
            "stats_dir": str(PROJECT / "data/paper-reproduction-stats-v1"),
            "data_file": str(data),
            "data_sha256": sha(data),
            "total_source_cases": len(read(data)),
            "max_cases": len(read(data)),
            "input_hashes": input_hashes,
            "tracking_group": batch,
            "original_experiment": "Table 3, GPT-J, per-instance edits"
            if benchmark == "MQuAKE"
            else f"Tables 3–5, GPT-2 XL ROME, {arm}",
            "editor_commit": "80426fd9316cf9a50c5ba15e0912f2c2c5bfe84b"
            if benchmark == "MQuAKE"
            else "54f3b88af4895a3aacb580ec63ce7ae857185040",
            "dataset_commit": "fb43dadc2d8cd19d08ce81c63d957b59deb3f3cd"
            if benchmark == "MQuAKE"
            else "54f3b88af4895a3aacb580ec63ce7ae857185040",
        }
        if benchmark == "MQuAKE":
            spec.update(
                prompt_dir=str(frozen / data_root.relative_to(PROJECT) / "prompts"),
                answer_max_new_tokens=64,
                cot_max_new_tokens=256,
                dataset_version=(
                    "Original MQuAKE-CF-3k, matching ACL 2023 Table 3, not the 2024 v2 correction"
                ),
            )
        else:
            spec.update(
                benchmark_source=str(frozen / upstream.relative_to(PROJECT)),
                split=arm,
                paper_max_new_tokens=20,
                dataset_version=(
                    "All cases in the pinned authors' release; counts differ from paper Table 1"
                ),
            )
        config["specs"].append(spec)
    write(PROJECT / "configs" / (batch + ".json"), config)
    smoke = copy.deepcopy(config)
    smoke.update(
        batch=benchmark.lower() + "-paper-validation-v1",
        repository=str(PROJECT),
        source_root=str(PROJECT),
        results_root=str(PROJECT / "results" / (benchmark.lower() + "-paper-validation-v1")),
    )
    if benchmark == "RippleEdits":
        smoke["specs"] = [s for s in smoke["specs"] if s["split"] == "popular"]
    for spec in smoke["specs"]:
        spec["name"] += "-smoke"
        spec["max_cases"] = 4 if benchmark == "MQuAKE" else 1
        spec["tracking_group"] = smoke["batch"]
        for key in ("editor_source", "hparams_file", "prompt_dir", "benchmark_source"):
            if key in spec:
                spec[key] = spec[key].replace(str(frozen), str(PROJECT), 1)
        spec["input_hashes"] = {
            key.replace(str(frozen), str(PROJECT), 1): value
            for key, value in spec["input_hashes"].items()
        }
        (Path(smoke["results_root"]) / "runs" / spec["name"]).mkdir(parents=True)
    write(PROJECT / "docs/development-artifacts" / batch / "smoke-config.json", smoke)
    print(
        json.dumps(
            {
                "batch": batch,
                "runs": len(config["specs"]),
                "cases": [s["max_cases"] for s in config["specs"]],
                "image": image,
            }
        )
    )


if __name__ == "__main__":
    build("MQuAKE")
    build("RippleEdits")
