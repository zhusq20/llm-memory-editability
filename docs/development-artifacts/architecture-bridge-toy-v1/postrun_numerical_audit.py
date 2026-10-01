"""Independent CPU matrix-recurrence audit; no training or model selection."""

import json
from pathlib import Path

import torch
from torch.nn import functional as F

from llm_memory_editability.architecture_bridge_toy import GatedDelta, ToyLM, initialize
from llm_memory_editability.twohop_depth import digest, write

ROOT = Path(__file__).resolve().parents[3]
ART = Path(__file__).resolve().parent
CFG = json.loads((ROOT / "configs/architecture-bridge-toy-v1.json").read_text())
torch.set_num_threads(2)


@torch.no_grad()
def main():
    model = ToyLM(CFG["architectures"][1], CFG)
    initialize(model, 171)
    model.double()
    mixer = next(block.mixer for block in model.blocks if isinstance(block.mixer, GatedDelta))
    x = torch.randn(2, 6, CFG["width"], generator=torch.Generator().manual_seed(271)).double()
    output, cache = mixer(x)
    raw = mixer.qkv(x)
    sequence = F.pad(raw.transpose(1, 2), (CFG["conv_kernel"] - 1, 0))
    convolution = F.conv1d(
        sequence,
        mixer.conv_weight.T[:, None, :],
        groups=3 * CFG["width"],
    ).transpose(1, 2)
    q, k, v = F.silu(convolution).chunk(3, -1)
    shape = (2, 6, CFG["heads"], CFG["width"] // CFG["heads"])
    q, k, v = q.reshape(shape), k.reshape(shape), v.reshape(shape)
    q = F.normalize(q, dim=-1, eps=1e-6)
    k = F.normalize(k, dim=-1, eps=1e-6)
    alpha = torch.exp(-F.softplus(mixer.decay(x)))
    beta = mixer.write(x).sigmoid()
    states, values = [], []
    state = torch.zeros(2, CFG["heads"], shape[-1], shape[-1], dtype=torch.float64)
    eye = torch.eye(shape[-1], dtype=torch.float64)[None, None]
    for t in range(shape[1]):
        key = k[:, t, :, :, None]
        val = v[:, t, :, :, None]
        transition = alpha[:, t, :, None, None] * (
            eye - beta[:, t, :, None, None] * (key @ key.transpose(-1, -2))
        )
        state = state @ transition + beta[:, t, :, None, None] * (val @ key.transpose(-1, -2))
        states.append(state)
        values.append((state @ q[:, t, :, :, None]).squeeze(-1))
    values = torch.stack(values, 1)
    reference = mixer.out((mixer.out_norm(values) * mixer.gate(x).sigmoid()[..., None]).flatten(-2))
    state_error = float((state - cache["state"]).abs().max())
    output_error = float((reference - output).abs().max())
    assert state_error < 1e-14 and output_error < 1e-12

    cache_errors, parameters = {}, {}
    tokens = torch.tensor([[1, 3, 35, 37, 5, 2], [1, 7, 38, 36, 9, 2]])
    for architecture in CFG["architectures"]:
        net = ToyLM(architecture, CFG)
        initialize(net, 171)
        net.double()
        full = net(tokens)
        pieces, memory = [], None
        # One-token decoding exercises all historical convolution and KV offsets.
        for t in range(tokens.shape[1]):
            logits, memory, _ = net(tokens[:, t : t + 1], cache=memory, return_cache=True)
            pieces.append(logits)
        error = float((full - torch.cat(pieces, 1)).abs().max())
        assert error < 1e-12
        cache_errors[architecture["name"]] = error
        parameters[architecture["name"]] = sum(p.numel() for p in net.parameters())

    report = {
        "scope": (
            "Independent float64 numerical reference; no additional training or scientific cases"
        ),
        "reference": "Depthwise conv1d plus explicit right-multiplied delta transition matrices",
        "max_delta_state_error": state_error,
        "max_delta_output_error": output_error,
        "one_token_cache_continuation_max_errors": cache_errors,
        "parameters": parameters,
        "hybrid_parameter_excess_fraction": parameters["hybrid"] / parameters["gqa"] - 1,
        "all_passed": True,
        "audit_script_sha256": digest(Path(__file__)),
    }
    write(ART / "independent-numerical-audit.json", report)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
