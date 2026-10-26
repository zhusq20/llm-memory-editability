"""AlphaEdit observer: reconstruct the full boundary before selecting a position.

The original frozen TTCD observer is unchanged. A GPU forensic replay showed
that RMS on a strided position slice can differ by 9.54e-7 from the native full
operation. Full reconstruction is bitwise equal; no tolerance is relaxed and
no model state, parameter, answer or scoring rule is changed.
"""

import numpy as np
import torch
from check_discoloop_edit_rounds import RoundTrace, scores
from torch.nn import functional as F

from llm_memory_editability.grokking_reproduction import encode_rows
from llm_memory_editability.ttcd_editing import tensor_digest


@torch.no_grad()
def inspect_rows(reader, fast, rows, metadata, expected, *, kind, bridges=None):
    """Use the parent's full query layout and greedy answer/EOS procedure."""
    device = next(reader.parameters()).device
    arrays = encode_rows(np.asarray(rows, dtype=np.int64), metadata)
    tokens, positions, labels = (torch.as_tensor(v, device=device) for v in arrays)
    old = labels[:, 0]
    new = torch.as_tensor(expected, device=device) + metadata["entity_offset"]
    before = tensor_digest(fast)
    ordinary = reader(tokens, positions, fast=fast)
    with RoundTrace(reader) as trace:
        observed = reader(tokens, positions, fast=fast)
    if not torch.equal(ordinary, observed):
        raise AssertionError("Observation changed final logits")
    states = trace.states()
    index = torch.arange(len(tokens), device=device)
    answer = observed[:, 0].argmax(-1)
    tokens[index, positions[:, 1]] = answer
    stop = reader(tokens, positions, fast=fast)[:, 1].argmax(-1)
    result = {
        "kind": kind,
        "rows": np.asarray(rows).tolist(),
        "answer_position": positions[:, 0].cpu().tolist(),
        "answer": answer.cpu().tolist(),
        "stop": stop.cpu().tolist(),
        "expected_new": new.cpu().tolist(),
        "old": old.cpu().tolist(),
        "final_new_joint": ((answer == new) & (stop == metadata["end_marker"])).cpu().tolist(),
        "final_old_joint": ((answer == old) & (stop == metadata["end_marker"])).cpu().tolist(),
        "hook_logits_exact": True,
        "stages": {},
    }
    model = reader.backbone
    for stage, state in states.items():
        logits = F.linear(model.transformer.ln_f(state), model.transformer.wte.weight)
        stage_result = {"answer": scores(logits[index, positions[:, 0]], new, old, new)}
        if bridges is not None:
            bridge = torch.as_tensor(bridges, device=device) + metadata["entity_offset"]
            stage_result["bridge"] = scores(logits[:, 1], bridge)
        result["stages"][stage] = stage_result
    if bridges is not None:
        # This is the boundary's actual probability-weighted embedding calculation.
        stage1_logits = F.linear(trace.norm_outputs[0], model.transformer.wte.weight)
        probability = (stage1_logits.float() / model.discoloop_temperature).softmax(-1)
        soft_full = F.linear(probability, model.transformer.wte.weight.T)
        soft = soft_full[:, 1]
        expected_embedding = model.transformer.wte.weight[
            torch.as_tensor(bridges, device=device) + metadata["entity_offset"]
        ]

        def rms(x):
            return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + model.discoloop_eps)

        channel_full = rms(soft_full)
        channel = channel_full[:, 1]
        actual_input = states["round2_input"][:, 1]
        reconstructed = (states["round1_end"] + model.discoloop_alpha * channel_full)[:, 1]
        if not torch.equal(actual_input, reconstructed):
            raise AssertionError("Reconstructed bridge boundary differs from actual input")
        result["bridge_channel"] = {
            "soft_embedding_cosine_to_true": F.cosine_similarity(soft, expected_embedding)
            .cpu()
            .tolist(),
            "normalized_embedding_relative_error": (
                (channel - rms(expected_embedding)).norm(dim=-1)
                / rms(expected_embedding).norm(dim=-1)
            )
            .cpu()
            .tolist(),
            "residual_to_channel_norm_ratio": (
                states["round1_end"][:, 1].norm(dim=-1)
                / (model.discoloop_alpha * channel).norm(dim=-1)
            )
            .cpu()
            .tolist(),
            "boundary_reconstruction_exact": True,
        }
    if tensor_digest(fast) != before:
        raise AssertionError("Diagnostic mutated a fast state")
    return result
