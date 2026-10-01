#!/usr/bin/env python3
"""GPU capture and recapture checks on the historical development checkpoint."""

import copy
from pathlib import Path

import numpy as np
import torch

from llm_memory_editability.grok_depth import make_optimizer, write_json
from llm_memory_editability.grok_loop_supervision import (
    SupervisionStep,
    answer_losses,
    from_state,
    load_world,
    restore_rng,
    tensor_digest,
)
from llm_memory_editability.grok_multihop import pack_rows

torch.set_num_threads(1)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device = torch.device("cuda:0")
source = Path("results/grok-loop-v1/development/dev-h2-l2")
state = torch.load(source / "latest.pt", map_location=device, weights_only=False)
world = load_world(source)
packed = [pack_rows(world[name], 4) for name in ("atomic", "train_composite")]
table = tuple(torch.as_tensor(np.concatenate(a), device=device) for a in zip(*packed, strict=True))
report = {}
for arm in ("single", "multi"):
    model = from_state(state, device)
    optimizer = make_optimizer(model, torch.tensor(1e-4, device=device), 0.1)
    optimizer.load_state_dict(copy.deepcopy(state["optimizer"]))
    restore_rng(state, device)
    before = tensor_digest(model.state_dict())
    graph = SupervisionStep(model, optimizer, table, 256, arm)
    assert tensor_digest(model.state_dict()) == before
    saved_rng, saved_cpu = torch.cuda.get_rng_state(), torch.get_rng_state()
    indices = torch.arange(256, device=device)
    graph(indices)
    torch.cuda.synchronize()
    after = copy.deepcopy(model.state_dict())
    opt_after = copy.deepcopy(optimizer.state_dict())
    native = from_state(state, device)
    native_opt = make_optimizer(native, torch.tensor(1e-4, device=device), 0.1)
    native_opt.load_state_dict(copy.deepcopy(state["optimizer"]))
    torch.cuda.set_rng_state(saved_rng)
    torch.set_rng_state(saved_cpu)
    losses = answer_losses(native, *(t[indices] for t in table))
    weights = torch.tensor([0, 0, 1] if arm == "single" else [1 / 3] * 3, device=device)
    (losses * weights).sum().backward()
    torch.nn.utils.clip_grad_norm_(native.parameters(), 1, foreach=True)
    native_opt.step()
    error = max(float((after[k] - v).abs().max()) for k, v in native.state_dict().items())
    print(arm, "capture_eager_weight_error", error, flush=True)
    assert error < 2e-6
    saved_rng, saved_cpu = torch.cuda.get_rng_state(), torch.get_rng_state()
    graph(indices)
    torch.cuda.synchronize()
    expected = copy.deepcopy(model.state_dict())
    model.load_state_dict(after)
    optimizer.load_state_dict(copy.deepcopy(opt_after))
    torch.cuda.set_rng_state(saved_rng)
    torch.set_rng_state(saved_cpu)
    regraph = SupervisionStep(model, optimizer, table, 256, arm)
    regraph(indices)
    torch.cuda.synchronize()
    resume_error = max(float((expected[k] - v).abs().max()) for k, v in model.state_dict().items())
    print(arm, "recapture_resume_error", resume_error, flush=True)
    assert resume_error < 2e-6
    report[arm] = {
        "capture_eager_max_weight_error": error,
        "recapture_resume_max_weight_error": resume_error,
        "restore_passed": True,
    }
write_json("docs/development-artifacts/grok-loop-supervision-v1/gpu-preflight.json", report)
