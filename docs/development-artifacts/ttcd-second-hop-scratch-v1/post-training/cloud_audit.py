"""Read back all five original W&B stage receipts without creating new runs."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import wandb


def plain(value):
    if hasattr(value, "items"):
        result = {str(k): plain(v) for k, v in value.items()}
        if result.get("_type") == "large-array" and "value" in result:
            return result["value"]
        return result
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def compare(expected, actual, path="result"):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(expected) != set(actual):
            raise AssertionError(f"Cloud dictionary differs: {path}")
        for key, value in expected.items():
            compare(value, actual[key], f"{path}/{key}")
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            raise AssertionError(f"Cloud list differs: {path}")
        for i, (a, b) in enumerate(zip(expected, actual, strict=True)):
            compare(a, b, f"{path}/{i}")
    elif isinstance(expected, float):
        if not math.isclose(expected, actual, rel_tol=1e-5, abs_tol=1e-5):
            raise AssertionError(f"Cloud float differs: {path}")
    elif expected != actual:
        raise AssertionError(f"Cloud value differs: {path}")


def flattened(value, prefix="result"):
    if isinstance(value, dict):
        result = {}
        for key, child in value.items():
            result.update(flattened(child, f"{prefix}.{key}"))
        return result
    return {prefix: value}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["cloud-audit"])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device")
    parser.add_argument("--tracking")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    out = args.out
    paths = {
        "train": "training/summary.json",
        "train-audit": "training/audit.json",
        "calibrate": "calibration.json",
        "evaluate": "summary.json",
        "audit": "audit.json",
    }
    api = wandb.Api(timeout=90)
    rows = []
    for stage, relative in paths.items():
        receipt = json.loads((out / f"wandb-{stage}.json").read_text())
        expected = json.loads((out / relative).read_text())
        if stage == "calibrate":
            expected = {"passed": expected["passed"], "groups": expected["groups"]}
        run = api.run(f"zhusq20/llm-memory-editability/{receipt['id']}")
        if run.state != "finished" or run.config["batch"] != config["batch"]:
            raise AssertionError(f"Cloud stage unfinished or wrong batch: {stage}")
        if run.config["model_name"] != args.model or run.config["stage"] != stage:
            raise AssertionError(f"Cloud stage identity differs: {stage}")
        summary = plain(run.summary)
        # W&B's HTTPSummary exposes flattened dotted keys in this fixed SDK.
        # The remotely downloaded canonical file preserves original nesting.
        downloaded = run.file("wandb-summary.json").download(
            root=str(out / "cloud-summary-downloads-v4" / stage), replace=False
        )
        cloud_path = Path(downloaded.name)
        cloud = json.loads(cloud_path.read_text())
        compare(expected, cloud["result"])
        if "result" in summary:
            compare(expected, summary["result"])
        else:
            for key, value in flattened(expected).items():
                if key not in summary:
                    raise AssertionError(f"Cloud API field missing: {key}")
                # The API may summarize floating arrays as histograms rather
                # than return their values. Full lists were already compared
                # recursively against the remote canonical file above.
                if not isinstance(value, list):
                    compare(value, summary[key], key)
        if stage == "train" and summary["training/step"] != 30000:
            raise AssertionError("Cloud terminal step differs")
        rows.append(
            {
                "stage": stage,
                "id": receipt["id"],
                "url": receipt["url"],
                "state": run.state,
                "result_matches": True,
                "full_arrays_verified_from_remote_canonical_file": True,
                "downloaded_summary_sha256": hashlib.sha256(cloud_path.read_bytes()).hexdigest(),
            }
        )
    result = {
        "passed": True,
        "model": args.model,
        "stages": rows,
        "terminal_training_step": 30000,
        "new_training_updates": 0,
    }
    target = out / "cloud-audit.json"
    with target.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
