"""Zero-surviving-update capture probe of the exact frozen scientific run path."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/frozen/scripts")
spec = importlib.util.spec_from_file_location(
    "query_capture_probe", "/frozen/scripts/run_loop_query_control.py"
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
config = json.loads(Path("/frozen/configs/loop-query-control-development-v1.json").read_text())
torch.set_num_threads(4)
torch.set_num_interop_threads(1)
torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
model, metadata, provenance = m.initialize(config, "cuda:0", config["arms"]["query_control"])
world = dict(np.load(Path(config["input_root"]) / "data/world.npz"))
table, _ = m.make_table(world, metadata, config["spec"], "cuda:0")
stream = m.MergedBatchStream(len(table[0]) - 1, config["spec"]["stream_seed"])
indices = torch.as_tensor(stream.batch(config["spec"]["batch_size"])[0], device="cuda:0")
m.gradient_diagnostic(
    model, table, indices, config["arms"]["query_control"], torch.device("cuda:0")
)
lr = torch.tensor(0.0, device="cuda:0")
optimizer = m.optimizer_for(model, lr, config["spec"]["weight_decay"])
print(json.dumps({"state": "capturing_exact_formal_training_path"}), flush=True)
graph = m.AlignmentStep(
    model, optimizer, table, config["spec"]["batch_size"], config["arms"]["query_control"]
)
torch.cuda.synchronize()
assert m.model_digest(model) == provenance["initial_model_sha256"]
print(
    json.dumps({"capture_passed": True, "all_weights_restored": True, "surviving_updates": 0}),
    flush=True,
)
