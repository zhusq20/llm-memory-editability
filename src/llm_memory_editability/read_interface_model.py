"""Model loading, chat prompts, scoring and residual-stream interventions for Qwen3 / Qwen3.5.

Interventions act on decoder-layer outputs (the residual stream after layer i) at chosen token
positions. They apply only to the prefill pass, so generation after a patched prompt continues
from the patched states through the model's own cache.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = {
    "qwen3-1.7b": {
        "repo": "Qwen/Qwen3-1.7B",
        "revision": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
        "path": "/hf/hub/models--Qwen--Qwen3-1.7B/snapshots/"
        "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
    },
    "qwen3.5-2b": {
        "repo": "Qwen/Qwen3.5-2B",
        "revision": "15852e8c16360a2fea060d615a32b45270f8a8fc",
        "path": "/hf/hub/models--Qwen--Qwen3.5-2B/snapshots/"
        "15852e8c16360a2fea060d615a32b45270f8a8fc",
    },
}


def load(name, weights=None, attn="sdpa", device="cuda"):
    spec = MODELS[name]
    tok = AutoTokenizer.from_pretrained(spec["path"])
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        spec["path"], dtype=torch.float32, attn_implementation=attn
    )
    if weights is not None:
        state = torch.load(weights, map_location="cpu", weights_only=True)
        missing, unexpected = model.load_state_dict(state, strict=False)
        assert not unexpected, unexpected
        assert all("lm_head" in k for k in missing), missing
    model.to(device).eval()
    return model, tok


def layers(model):
    return model.model.layers


def layer_kinds(model):
    return ["linear" if hasattr(layer, "linear_attn") else "attention" for layer in layers(model)]


def chat(tok, user, thinking=False, system=None):
    messages = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": user}
    ]
    return tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=thinking
    )


def span_positions(offsets, start, end):
    return [i for i, (a, b) in enumerate(offsets) if b > start and a < end and b > a]


def encode(tok, texts, spans=None, answers=None):
    """Left-padded batch; `spans` are per-text lists of (char_start, char_end) to locate.

    Returns input ids, mask, located positions in padded coordinates, and answer token slices.
    """
    full = [t + (a or "") for t, a in zip(texts, answers or [None] * len(texts), strict=True)]
    enc = tok(
        full,
        return_tensors="pt",
        padding=True,
        return_offsets_mapping=True,
        add_special_tokens=False,
    )
    width = enc["input_ids"].shape[1]
    positions, answer_slices = [], []
    for i, text in enumerate(texts):
        offsets = enc["offset_mapping"][i].tolist()
        pad = width - int(enc["attention_mask"][i].sum())
        real = offsets[pad:]
        found = []
        for start, end in spans[i] if spans else []:
            found.append([pad + p for p in span_positions(real, start, end)])
        positions.append(found)
        if answers:
            first = next(j for j, (a, b) in enumerate(real) if b > len(text) and b > a)
            answer_slices.append((pad + first, width))
    return enc["input_ids"], enc["attention_mask"], positions, answer_slices


def find_span(text, needle, after=0):
    start = text.find(needle, after)
    if start < 0:
        raise ValueError(f"{needle!r} not in prompt")
    return start, start + len(needle)


class Interventions:
    """Forward hooks on decoder layers: capture, overwrite, and token-mixer knockout."""

    def __init__(self, model):
        self.model = model
        self.handles = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def capture(self, layer_ids, store):
        """store[layer] = full output tensor (batch, seq, dim) from the prefill pass."""
        for idx in layer_ids:

            def hook(module, args, output, idx=idx):
                if output.shape[1] > 1:
                    store[idx] = output.detach()

            self.handles.append(layers(self.model)[idx].register_forward_hook(hook))

    def overwrite(self, layer_idx, rows, positions, vectors):
        """Replace layer output at (rows[k], positions[k]) with vectors[k] during prefill."""
        rows = torch.as_tensor(rows)
        positions = torch.as_tensor(positions)

        def hook(module, args, output):
            if output.shape[1] == 1:
                return output
            output = output.clone()
            output[rows.to(output.device), positions.to(output.device)] = vectors.to(
                output.device, output.dtype
            )
            return output

        self.handles.append(layers(self.model)[layer_idx].register_forward_hook(hook))

    def add(self, layer_idx, rows, positions, vectors):
        """Add vectors[k] to the layer output at (rows[k], positions[k]) during prefill; the
        result stays differentiable with respect to `vectors`."""
        rows = torch.as_tensor(rows, dtype=torch.long)
        positions = torch.as_tensor(positions, dtype=torch.long)

        def hook(module, args, output):
            if output.shape[1] == 1 or len(rows) == 0:
                return output
            r, p = rows.to(output.device), positions.to(output.device)
            output = output.clone()
            output[r, p] = output[r, p] + vectors.to(output.device, output.dtype)
            return output

        self.handles.append(layers(self.model)[layer_idx].register_forward_hook(hook))

    def capture_mixer_inputs(self, store):
        """store[layer] = list of (batch, seq, dim) inputs to each DeltaNet token mixer."""
        for idx, layer in enumerate(layers(self.model)):
            if not hasattr(layer, "linear_attn"):
                continue

            def pre(module, args, kwargs, idx=idx):
                hidden = kwargs.get("hidden_states", args[0] if args else None)
                store.setdefault(idx, []).append(hidden.detach())

            self.handles.append(layer.linear_attn.register_forward_pre_hook(pre, with_kwargs=True))

    def block_attention(self, layer_idx, rows, query_pos, key_pos):
        """Attention knockout: queries at query_pos cannot attend to key_pos (Geva et al.)."""
        attn = layers(self.model)[layer_idx].self_attn

        def pre(module, args, kwargs):
            mask = kwargs.get("attention_mask")
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            b, t = hidden.shape[:2]
            if mask is None:
                mask = torch.ones(t, t, dtype=torch.bool, device=hidden.device).tril()
                mask = mask[None, None].expand(b, 1, t, t)
            mask = mask.clone()
            blocked = False if mask.dtype == torch.bool else torch.finfo(mask.dtype).min
            for r, q, keys in zip(rows, query_pos, key_pos, strict=True):
                mask[r, :, q, keys] = blocked
            kwargs["attention_mask"] = mask
            return args, kwargs

        self.handles.append(attn.register_forward_pre_hook(pre, with_kwargs=True))

    def block_linear_mixer(self, layer_idx, rows, query_pos, key_pos, mean_input):
        """DeltaNet knockout: the mixer output at query_pos comes from a second pass in which the
        mixer inputs at key_pos are replaced by a reference mean, so it cannot read those tokens.
        Outputs at all other positions are unchanged."""
        mixer = layers(self.model)[layer_idx].linear_attn
        state = {"inner": False}

        def hook(module, args, kwargs, output):
            if state["inner"]:
                return output
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            if hidden.shape[1] == 1:
                return output
            edited = hidden.clone()
            for r, keys in zip(rows, key_pos, strict=True):
                edited[r, keys] = mean_input.to(edited.device, edited.dtype)
            state["inner"] = True
            try:
                if args:
                    second = module(edited, *args[1:], **kwargs)
                else:
                    second = module(**{**kwargs, "hidden_states": edited})
            finally:
                state["inner"] = False
            second = second[0] if isinstance(second, tuple) else second
            out = (output[0] if isinstance(output, tuple) else output).clone()
            for r, q in zip(rows, query_pos, strict=True):
                out[r, q] = second[r, q]
            return (out, *output[1:]) if isinstance(output, tuple) else out

        self.handles.append(mixer.register_forward_hook(hook, with_kwargs=True))


def autocast():
    return torch.autocast("cuda", dtype=torch.bfloat16)


@torch.no_grad()
def answer_scores(model, ids, mask, answer_slices):
    """Teacher-forced greedy match and log-prob of each answer continuation."""
    with autocast():
        logits = model(
            input_ids=ids.cuda(), attention_mask=mask.cuda(), use_cache=False
        ).logits.float()
    logp = torch.log_softmax(logits, -1)
    out = []
    for i, (start, end) in enumerate(answer_slices):
        target = ids[i, start:end].cuda()
        pred = logp[i, start - 1 : end - 1]
        out.append(
            {
                "greedy": bool((pred.argmax(-1) == target).all()),
                "logprob": float(pred.gather(-1, target[:, None]).sum()),
            }
        )
    return out


@torch.no_grad()
def generate(model, tok, ids, mask, max_new_tokens):
    with autocast():
        stop = [tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id]
        out = model.generate(
            eos_token_id=sorted(set(stop)),
            input_ids=ids.cuda(),
            attention_mask=mask.cuda(),
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tok.pad_token_id,
            top_p=None,
            top_k=None,
            temperature=None,
        )
    return tok.batch_decode(out[:, ids.shape[1] :], skip_special_tokens=True)


def final_answer(text, stepwise):
    """Direct mode: the whole reply. Step-by-step mode: the text after the last
    'Final answer:' marker, or empty if the reply never gave one."""
    if not stepwise:
        return text.strip()
    marker = "final answer:"
    at = text.lower().rfind(marker)
    return text[at + len(marker) :].strip() if at >= 0 else ""
