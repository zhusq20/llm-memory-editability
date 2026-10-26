"""Second-hop editing with a GPT/DiscoLoop adaptation of TTCD equation (6).

The real GPT residual, LayerNorm, causal attention, GELU MLP, positional
embeddings and tied decoder are retained. This implements the ideal local
TTCD update, not the paper's learned conv/projection IP-TTCD-v9 recipe.
Teacher/student local views share Q/K/V and the incoming residual. Writes use
only the atomic-query position in the FIRST round, once per physical layer.
The resulting down-projection correction is shared by every loop execution.
It lives outside the backbone and can be saved and read without context/KV.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def tensor_digest(tensors):
    result = hashlib.sha256()
    for name, value in sorted(tensors.items(), key=lambda item: str(item[0])):
        value = value.detach().cpu().contiguous()
        result.update(str(name).encode())
        result.update(str(value.dtype).encode())
        result.update(str(tuple(value.shape)).encode())
        result.update(value.view(torch.uint8).numpy().tobytes())
    return result.hexdigest()


def causal_window(length, window, device):
    if window is not None and (
        isinstance(window, bool) or not isinstance(window, int) or window < 1
    ):
        raise ValueError("window must be a positive integer or None")
    i = torch.arange(length, device=device)
    distance = i[:, None] - i[None, :]
    return (distance >= 0) & ((distance < window) if window is not None else True)


def local_ttcd_update(z_student, y_teacher, y_student, learning_rate):
    """Negative detached local gradient, in GPT Conv1D's [intermediate, hidden] layout.

    Inputs are [write positions, features]. The teacher is a stopped target;
    inputs to this local gradient are held fixed. This is NOT the full gradient
    through upstream layers or multiple loop executions. No norm/clipping or
    learned feature transform silently changes equation (6).
    """
    if not math.isfinite(learning_rate) or learning_rate < 0:
        raise ValueError("learning_rate must be finite and nonnegative")
    if z_student.ndim != 2 or y_student.ndim != 2 or y_teacher.shape != y_student.shape:
        raise ValueError("Expected 2-D features and matching 2-D teacher/student outputs")
    if not len(z_student) or len(z_student) != len(y_student):
        raise ValueError("At least one aligned write position is required")
    dtype = torch.float64 if z_student.dtype == torch.float64 else torch.float32
    z = z_student.detach().to(dtype)
    residual = y_teacher.detach().to(dtype) - y_student.detach().to(dtype)
    update = learning_rate * z.T @ residual / len(z)
    if not torch.isfinite(update).all():
        raise FloatingPointError("Nonfinite TTCD update")
    return update


class TTCDReader(nn.Module):
    """Audited backbone plus explicitly passed session fast weights.

    Frozen by default. The scratch trainer explicitly enables backbone gradients
    and batched independent fast states; it retains zero-dropout eval computation.

    Evaluation is stateless: no hidden state, KV cache, attention output or
    previous query is stored. Only a caller-supplied ``fast`` dict can persist.
    Native query token positions start at zero; context packets use sequential
    positions, an intentionally visible potential distribution mismatch.
    """

    def __init__(self, backbone, *, trainable_backbone=False, native_causal_attention=False):
        super().__init__()
        if getattr(backbone, "first_round_kv", False):
            raise ValueError("This experiment does not reopen the old KV-fusion grid")
        self.trainable_backbone = trainable_backbone
        self.native_causal_attention = native_causal_attention
        self.backbone = backbone.eval().requires_grad_(trainable_backbone)
        if trainable_backbone and any(
            getattr(backbone.config, name) != 0
            for name in ("embd_pdrop", "resid_pdrop", "attn_pdrop")
        ):
            raise ValueError(
                "Scratch training uses explicit zero dropout and eval-mode computation"
            )
        # Accounting only; never read by the model or serialized as memory.
        self.compute = {
            "forward_calls": 0,
            "executed_input_tokens": 0,
            "estimated_dense_forward_matmul_flops": 0,
            "estimated_backward_matmul_flops_proxy": 0,
        }

    def empty_fast(self):
        dtype = torch.float64 if next(self.parameters()).dtype == torch.float64 else torch.float32
        return {
            i: torch.zeros_like(block.mlp.c_proj.weight, dtype=dtype)
            for i, block in enumerate(self.backbone.transformer.h)
        }

    def validate_fast(self, fast):
        expected = {
            i: block.mlp.c_proj.weight for i, block in enumerate(self.backbone.transformer.h)
        }
        if set(fast) != set(expected):
            raise ValueError("One fast-weight correction is required per physical layer")
        for key, value in fast.items():
            shape = value.shape[-2:] if self.trainable_backbone else value.shape
            if shape != expected[key].shape or value.device != expected[key].device:
                raise ValueError(f"Fast-weight shape/device mismatch in layer {key}")
            if value.ndim not in ((2, 3) if self.trainable_backbone else (2,)):
                raise ValueError("Fast weights must be matrices or independent training batches")
            if not torch.isfinite(value).all():
                raise FloatingPointError(f"Nonfinite fast weights in layer {key}")

    def forward(self, tokens, positions=None, *, fast=None, window=None, capture=None):
        if self.training or self.backbone.training:
            raise ValueError(
                "Use reader.eval(); this reader requires deterministic eval computation"
            )
        if tokens.ndim != 2 or tokens.shape[1] < 1:
            raise ValueError("tokens must be a nonempty [batch, length] tensor")
        model, cfg = self.backbone, self.backbone.config
        if tokens.shape[1] > cfg.n_positions:
            raise ValueError("Context packet exceeds the existing positional embedding table")
        batch, length = tokens.shape
        width, flops = cfg.n_embd, 0
        for block in model.transformer.h:
            intermediate = block.mlp.c_proj.weight.shape[0]
            cost = (
                8 * length * width**2
                + 4 * length**2 * width
                + (6 if fast is not None else 4) * length * width * intermediate
            )
            flops += model.repeats * cost
            if capture is not None:
                flops += (
                    2 * length * width**2
                    + 4 * length**2 * width
                    + (6 if fast is not None else 4) * length * width * intermediate
                )
        if getattr(model, "discoloop", False):
            flops += (model.repeats - 1) * 4 * length * width * cfg.vocab_size
        scored = length if positions is None else positions.shape[1]
        flops = batch * (flops + 2 * scored * width * cfg.vocab_size)
        self.compute["forward_calls"] += 1
        self.compute["executed_input_tokens"] += batch * length
        self.compute["estimated_dense_forward_matmul_flops"] += flops
        if torch.is_grad_enabled() and (
            self.trainable_backbone
            or (fast is not None and any(v.requires_grad for v in fast.values()))
        ):
            # Conventional 2x-forward backward proxy. Frozen-weight kernels may
            # execute less; keep this separate from the forward estimate.
            self.compute["estimated_backward_matmul_flops_proxy"] += 2 * flops
        if fast is not None:
            self.validate_fast(fast)
        if capture is not None:
            if window is None or capture["student_window"] >= window:
                raise ValueError("Capture requires teacher_window > student_window")
            selected = capture["positions"]
            if selected.shape != (len(tokens),):
                raise ValueError("One causal atomic-query write position per sequence is required")
            if (selected < 0).any() or (selected >= tokens.shape[1]).any():
                raise ValueError("Invalid write position")
            capture["features"] = {}
        length = tokens.shape[1]
        # A full boolean mask and is_causal=True can choose different CUDA
        # kernels. The verified foundation reader must use the native kernel
        # path for ordinary queries; bounded teacher/student views retain their
        # existing explicit masks. The opt-in preserves historical readers.
        mask = (
            None
            if self.native_causal_attention and window is None
            else causal_window(length, window, tokens.device)
        )
        t = model.transformer
        x = t.wte(tokens) + t.wpe(torch.arange(length, device=tokens.device))
        for round_index in range(model.repeats):
            for layer, block in enumerate(t.h):
                residual = x
                q, k, v = block.attn.c_attn(block.ln_1(x)).split(cfg.n_embd, dim=2)
                shape = (len(tokens), length, cfg.n_head, cfg.n_embd // cfg.n_head)
                q, k, v = (a.view(shape).transpose(1, 2) for a in (q, k, v))

                def attention(attention_mask, q=q, k=k, v=v, residual=residual, block=block):
                    out = F.scaled_dot_product_attention(
                        q, k, v, attn_mask=attention_mask, is_causal=attention_mask is None
                    )
                    out = out.transpose(1, 2).reshape(len(tokens), length, cfg.n_embd)
                    return residual + block.attn.c_proj(out)

                def mlp(incoming, block=block, layer=layer):
                    z = block.mlp.act(block.mlp.c_fc(block.ln_2(incoming)))
                    output = block.mlp.c_proj(z)
                    if fast is not None:
                        if fast[layer].ndim == 3 and fast[layer].shape[0] != len(tokens):
                            raise ValueError(
                                "Independent fast-state batch size differs from tokens"
                            )
                        delta = torch.matmul(z.to(fast[layer].dtype), fast[layer])
                        output = output + delta.to(output.dtype)
                    return z, output

                u = attention(mask)
                _, y_teacher = mlp(u)
                if capture is not None and round_index == 0:
                    student_mask = causal_window(length, capture["student_window"], tokens.device)
                    z_student, y_student = mlp(attention(student_mask))
                    batch = torch.arange(len(tokens), device=tokens.device)
                    features = tuple(a[batch, selected] for a in (z_student, y_teacher, y_student))
                    capture["features"][layer] = (
                        features
                        if capture.get("outer_grad", False)
                        else tuple(a.detach().clone() for a in features)
                    )
                x = u + y_teacher
            if getattr(model, "discoloop", False) and round_index < model.repeats - 1:
                x = model._discoloop_boundary(x)
        x = t.ln_f(x)
        if positions is not None:
            x = x[torch.arange(len(tokens), device=tokens.device)[:, None], positions]
        return F.linear(x, t.wte.weight)

    @torch.no_grad()
    def write(self, packet, fast, *, teacher_window, student_window, learning_rate):
        """Read the complete atomic packet, then commit all layer writes together.

        No update is visible in the packet that generated it. No composite
        queries, their labels, or EOS prediction labels are accepted by this API.
        ``packet`` ends at the atomic relation; its new answer occurs only in
        the earlier correction record. Readout and update use the same prefix.
        """
        self.validate_fast(fast)
        capture = {
            "student_window": student_window,
            "positions": torch.full(
                (len(packet),), packet.shape[1] - 1, device=packet.device, dtype=torch.long
            ),
        }
        positions = capture["positions"][:, None]
        logits = self(packet, positions, fast=fast, window=teacher_window, capture=capture)[:, 0]
        updates = {
            layer: local_ttcd_update(*features, learning_rate)
            for layer, features in capture["features"].items()
        }
        self.compute["estimated_dense_forward_matmul_flops"] += sum(
            2 * len(features[0]) * features[0].shape[1] * features[1].shape[1]
            for features in capture["features"].values()
        )
        updated = {layer: fast[layer].detach() + updates[layer] for layer in fast}
        self.validate_fast(updated)
        info = {
            "teacher_argmax": logits.argmax(-1).cpu().tolist(),
            "write_positions": len(packet),
            "layers_written": len(updates),
            "update_norm": math.sqrt(sum(float(a.square().sum()) for a in updates.values())),
            "local_gaps": {
                str(layer): float((features[1] - features[2]).square().mean())
                for layer, features in capture["features"].items()
            },
        }
        return updated, info


def correction_packet(atom, new_tail, query, metadata, *, gap, teacher_window, student_window):
    """Record [b,r,new,EOS], filler, then an unanswered atomic/composite query.

    Only the atomic form is sent to ``write`` by the runner. The composite form
    is exclusively an open-context evaluation reference. Visibility is checked
    using sequence indices, never repeated/reset position embeddings.
    """
    atom = [int(v) for v in atom]
    query = [int(v) for v in query]
    if gap < 0 or len(atom) != 3 or len(query) not in (2, 3):
        raise ValueError("Invalid correction packet")
    if not 0 <= new_tail < metadata["entities"] or new_tail == atom[2]:
        raise ValueError("The new target must be an existing different entity")
    eo, ro = metadata["entity_offset"], metadata["relation_offset"]
    record = [atom[0] + eo, atom[1] + ro, new_tail + eo, metadata["end_marker"]]
    encoded = [query[0] + eo, *[r + ro for r in query[1:]]]
    packet = record + [metadata["pad_token"]] * gap + encoded
    position = len(packet) - 1
    if position >= teacher_window or position - 3 < student_window:
        raise ValueError("Complete correction must be visible to long and absent from short window")
    return packet


def build_case_bank(world, *, seed, calibration_cases, evaluation_cases, affected, controls, phi):
    """Choose target facts and untrained compositions BEFORE observing predictions.

    Calibration/evaluation use disjoint edited facts in the SAME observed world.
    This is development evidence, not independent-world confirmation. No
    correctness or post-edit outcome filters choose cases or queries.
    """
    if any(n < 1 for n in (calibration_cases, evaluation_cases, affected, controls)):
        raise ValueError("Case/query budgets must be positive")
    rng = np.random.RandomState(seed)
    support = world.support(phi)
    by_second = {}
    for chain in np.flatnonzero(~support & (world.first != world.second)):
        by_second.setdefault(int(world.second[chain]), []).append(int(chain))
    ordering = {
        group: rng.permutation(
            [a for a in by_second if bool(world.ood[a]) == (group == "ood")]
        ).tolist()
        for group in ("id", "ood")
    }
    cases = []
    for split, count in (("calibration", calibration_cases), ("evaluation", evaluation_cases)):
        for group in ("id", "ood"):
            for _ in range(count):
                if not ordering[group]:
                    raise ValueError(f"Not enough {group} facts with held-out two-hop dependents")
                atom_id = int(ordering[group].pop())
                old = int(world.atoms[atom_id, 2])
                new = int(rng.randint(world.entities - 1))
                new += new >= old
                dependent = rng.permutation(by_second[atom_id])[:affected].tolist()
                unrelated = np.flatnonzero(
                    (world.first != atom_id) & (world.second != atom_id) & ~support
                )
                atom_controls = np.flatnonzero(np.arange(len(world.atoms)) != atom_id)
                same_answer = np.flatnonzero(
                    (world.atoms[:, 2] == old) & (np.arange(len(world.atoms)) != atom_id)
                )
                cases.append(
                    {
                        "case_id": f"{split}-{group}-a{atom_id}",
                        "split": split,
                        "group": group,
                        "atom": atom_id,
                        "new_tail": new,
                        "D": dependent,
                        "U_atoms": rng.permutation(atom_controls)[:controls].tolist(),
                        "U_chains": rng.permutation(unrelated)[:controls].tolist(),
                        "same_answer_atoms": rng.permutation(same_answer)[:controls].tolist(),
                    }
                )
    return cases


def save_fast(path, reader, fast, provenance):
    reader.validate_fast(fast)
    payload = {
        "schema": "ttcd-fast-v1",
        "backbone_digest": tensor_digest(reader.backbone.state_dict()),
        "fast_digest": tensor_digest(fast),
        "fast": {k: v.detach().cpu() for k, v in fast.items()},
        "provenance": provenance,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return payload["fast_digest"]


def load_fast(path, reader):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["schema"] != "ttcd-fast-v1":
        raise ValueError("Unknown fast-state schema")
    if payload["backbone_digest"] != tensor_digest(reader.backbone.state_dict()):
        raise ValueError("Fast state belongs to different backbone weights")
    if payload["fast_digest"] != tensor_digest(payload["fast"]):
        raise ValueError("Fast-state hash mismatch")
    device = next(reader.parameters()).device
    fast = {k: v.to(device) for k, v in payload["fast"].items()}
    reader.validate_fast(fast)
    return fast, payload["provenance"]
