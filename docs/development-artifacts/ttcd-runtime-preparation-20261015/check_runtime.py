"""Container CPU/storage/W&B verification; never launches scientific training."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import time
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--credential-file", type=Path, default=Path("/run/secrets/wandb"))
    parser.add_argument("--online", action="store_true")
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    checkpoint = out / "cpu-roundtrip.pt"
    if args.replay:
        state = torch.load(checkpoint, weights_only=True, map_location="cpu")
        x = state["input"].requires_grad_()
        value = (x @ x.T).square().mean()
        value.backward()
        torch.testing.assert_close(value.detach(), state["output"], rtol=0, atol=0)
        torch.testing.assert_close(x.grad, state["gradient"], rtol=0, atol=0)
        result = {
            "passed": True,
            "independent_container_reload": True,
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "uid": os.getuid(),
            "gid": os.getgid(),
        }
        (out / "storage-replay.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result))
        return
    if checkpoint.exists():
        raise FileExistsError("Preserve earlier runtime check; choose a new output directory")
    torch.manual_seed(20261015)
    x = torch.randn(32, 32, requires_grad=True)
    output = (x @ x.T).square().mean()
    output.backward()
    if not torch.isfinite(x.grad).all():
        raise RuntimeError("CPU matrix/backward check failed")
    torch.save({"input": x.detach(), "output": output.detach(), "gradient": x.grad}, checkpoint)
    result = {
        "passed": True,
        "scientific_training": False,
        "training_updates": 0,
        "python": platform.python_version(),
        "torch_cuda": torch.version.cuda,
        "uid": os.getuid(),
        "gid": os.getgid(),
        "versions": {
            p: importlib.metadata.version(p)
            for p in (
                "torch",
                "transformers",
                "numpy",
                "pytest",
                "ruff",
                "wandb",
                "huggingface-hub",
            )
        },
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    }
    if args.online:
        import wandb

        key = args.credential_file.read_text().strip()
        os.environ["WANDB_API_KEY"] = key
        run = wandb.init(
            entity="zhusq20",
            project="llm-memory-editability",
            name="ttcd-runtime-preparation-20261015",
            group="ttcd-runtime-preparation-20261015",
            job_type="environment-check",
            mode="online",
            dir=str(out),
            save_code=False,
            config={
                "scientific_training": False,
                "training_updates": 0,
                "purpose": "Check independent LM image, NFS persistence, and W&B upload",
            },
            settings=wandb.Settings(disable_git=True),
        )
        run.log({"environment/online_check": 1, "training_updates": 0}, step=0)
        run.summary["environment_passed"] = True
        run.summary["scientific_training"] = False
        run_id, url = run.id, run.url
        run.finish()
        api = wandb.Api(api_key=key, timeout=30)
        remote = None
        for _attempt in range(6):
            remote = api.run(f"zhusq20/llm-memory-editability/{run_id}")
            if remote.state == "finished":
                break
            time.sleep(2)
            api.flush()
        if remote.state != "finished" or not remote.summary.get("environment_passed"):
            raise RuntimeError("W&B remote terminal state/summary verification failed")
        rows = list(
            remote.scan_history(keys=["_step", "environment/online_check", "training_updates"])
        )
        if not any(
            r.get("environment/online_check") == 1 and r.get("training_updates") == 0 for r in rows
        ):
            raise RuntimeError("W&B cloud history verification failed")
        result["wandb"] = {
            "id": run_id,
            "url": url,
            "state": remote.state,
            "cloud_summary_verified": True,
            "cloud_history_verified": True,
        }
    (out / "runtime-check.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
