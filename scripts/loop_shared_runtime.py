"""Execute unchanged frozen Loop code and record permissions and CUDA peaks."""

import json
import os
import runpy
import sys
from pathlib import Path


def main():
    profile, stage, *arguments = sys.argv[1:]
    profile = Path(profile)
    os.umask(0o022)
    secret = Path("/run/secrets/wandb")
    if secret.exists():
        os.environ["WANDB_API_KEY"] = secret.read_text().strip()
    os.setgroups([])
    os.setgid(int(os.environ["LM_LOOP_GID"]))
    os.setuid(int(os.environ["LM_LOOP_UID"]))
    files = 0
    for root in [Path("/project"), Path("/loop-parent-source")]:
        for path in [root, *root.rglob("*")]:
            stat = path.stat()
            expected = 0o555 if path.is_dir() else 0o444
            if stat.st_uid != 0 or stat.st_mode & 0o777 != expected:
                raise ValueError(f"Frozen code ownership/permissions differ: {path}")
            files += path.is_file()
    try:
        if stage == "preflight":
            import torch
            import wandb

            torch.set_num_threads(2)
            torch.cuda.set_device(0)
            torch.cuda.set_per_process_memory_fraction(0.16, 0)
            x = torch.randn(1024, 1024, device="cuda", requires_grad=True)
            loss = (x @ x.T).square().mean()
            loss.backward()
            torch.cuda.synchronize()
            assert torch.isfinite(loss) and torch.isfinite(x.grad).all()
            api = wandb.Api(timeout=60)
            assert api.viewer is not None
            list(api.runs("zsqzz/llm-memory-editability", per_page=1)[:1])
            print("CUDA matrix/backward/sync and W&B access passed", flush=True)
        else:
            sys.argv = ["/project/scripts/run_loop_robustness.py", stage, *arguments]
            runpy.run_path(sys.argv[0], run_name="__main__")
    finally:
        import torch

        data = dict(root_owned_immutable_source_files=files, stage=stage)
        if torch.cuda.is_initialized():
            torch.cuda.synchronize()
            data.update(
                gpu=torch.cuda.get_device_name(0),
                peak_allocated_mib=torch.cuda.max_memory_allocated(0) / 2**20,
                peak_reserved_mib=torch.cuda.max_memory_reserved(0) / 2**20,
            )
        profile.write_text(json.dumps(data, indent=2) + "\n")


if __name__ == "__main__":
    main()
