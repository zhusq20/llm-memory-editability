"""One-line lifetime repair to the frozen zero-update preflight, never training."""

import argparse
import hashlib
import importlib.util
import inspect
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, "/frozen/scripts")
spec = importlib.util.spec_from_file_location(
    "frozen_query_preflight_repair", "/frozen/scripts/run_loop_query_control.py"
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("stage", choices=["preflight"])
parser.add_argument("--config", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
config = json.loads(args.config.read_text())
m.validate_config(config)
torch.set_num_threads(4)
torch.set_num_interop_threads(1)
torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
source = inspect.getsource(m.preflight)
old = "del gradients, reference, outputs, candidate"
if source.count(old) != 1:
    raise ValueError("Expected exact frozen preflight lifetime site")
patched = source.replace(old, old + ", logits, loss")
exec(compile(patched, "<frozen_preflight_lifetime_repair>", "exec"), m.__dict__)
result = m.preflight(config, args.out)
m.write_json(
    args.out / "result-preflight-repair.json",
    {
        "passed": result["passed"],
        "original_failed_preflight_retained": True,
        "surviving_updates": 0,
        "only_change": "Release eager logits/loss graph references before CUDA graph capture",
        "original_function_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "patched_function_sha256": hashlib.sha256(patched.encode()).hexdigest(),
        "utc": m.utc(),
    },
)
print(json.dumps(result), flush=True)
