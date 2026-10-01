"""FP32 Qwen adapter. All CE/KL reductions preserve per-sequence weighting."""

from __future__ import annotations

import copy
import hashlib
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

from .hebbian_data import TEMPLATE, score_answer
from .hebbian_learning import ARTIFACTS, DATA, config, now, read_json, write_json


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def state_hashes(model):
    return {name: tensor_hash(p) for name, p in model.state_dict().items()}


def aggregate_hash(hashes):
    return hashlib.sha256(
        "\n".join(f"{k}:{v}" for k, v in sorted(hashes.items())).encode()
    ).hexdigest()


def set_determinism(seed=0):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


class QwenExperiment:
    def __init__(self, device="cuda:3"):
        set_determinism()
        self.cfg = config()
        self.device = torch.device(device)
        source = DATA / "source/qwen3-0.6b-base"
        self.tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                source,
                local_files_only=True,
                torch_dtype=torch.float32,
                attn_implementation=self.cfg["model"]["attention"],
            )
            .to(self.device)
            .eval()
        )
        self.model.requires_grad_(False)
        self.layer = self.model.model.layers[self.cfg["model"]["layer"]]
        self.mlp = self.layer.mlp
        assert self.mlp.down_proj.weight.shape == (1024, 3072)
        assert self.mlp.up_proj.weight.numel() + self.mlp.gate_proj.weight.numel() == 6291456
        self.base_local = {n: p.detach().clone() for n, p in self.mlp.named_parameters()}
        self.reference = None

    def restore_base(self):
        with torch.no_grad():
            for n, p in self.mlp.named_parameters():
                p.copy_(self.base_local[n])

    def configure_trainable(self, phase):
        self.model.requires_grad_(False)
        names = ["down_proj.weight"] if phase == "adapt" else ["up_proj.weight", "gate_proj.weight"]
        for name, p in self.mlp.named_parameters():
            p.requires_grad_(name in names)
        return [p for p in self.model.parameters() if p.requires_grad]

    def make_reference(self):
        self.reference = copy.deepcopy(self.model).eval().requires_grad_(False)

    def rows(self, records, views=(0,)):
        return [(r, v, r["encoded"][v]) for r in records for v in views]

    def batch(self, encoded, prompt_only=False):
        sequences = [
            x["input_ids"][: x["answer_start"]] if prompt_only else x["input_ids"] for x in encoded
        ]
        width = max(map(len, sequences))
        ids = torch.full(
            (len(sequences), width),
            self.tokenizer.pad_token_id,
            dtype=torch.long,
            device=self.device,
        )
        attention = torch.zeros_like(ids)
        answer = torch.zeros_like(ids, dtype=torch.bool)
        for i, (seq, enc) in enumerate(zip(sequences, encoded, strict=True)):
            ids[i, : len(seq)] = torch.tensor(seq, device=self.device)
            attention[i, : len(seq)] = 1
            if not prompt_only:
                answer[i, : len(seq)] = torch.tensor(enc["loss_mask"], device=self.device).bool()
        return ids, attention, answer

    def hidden(self, ids, attention, model=None):
        return (
            (model or self.model)
            .model(input_ids=ids, attention_mask=attention, use_cache=False)
            .last_hidden_state
        )

    def ce(self, encoded, model=None):
        ids, attention, mask = self.batch(encoded)
        hidden = self.hidden(ids, attention, model)
        b, t = torch.where(mask[:, 1:])
        logits = (model or self.model).lm_head(hidden[b, t])
        losses = F.cross_entropy(logits, ids[b, t + 1], reduction="none")
        sums = torch.zeros(len(encoded), device=self.device).scatter_add_(0, b, losses)
        counts = torch.bincount(b, minlength=len(encoded))
        return sums / counts

    def kl(self, encoded):
        if self.reference is None:
            raise RuntimeError("An episode-specific parent reference is required")
        ids, attention, mask = self.batch(encoded)
        b, t = torch.where(mask[:, 1:])
        with torch.no_grad():
            ref_hidden = self.hidden(ids, attention, self.reference)
            ref_log = F.log_softmax(self.reference.lm_head(ref_hidden[b, t]), dim=-1)
        logits = self.model.lm_head(self.hidden(ids, attention)[b, t])
        losses = F.kl_div(
            F.log_softmax(logits, dim=-1), ref_log, reduction="none", log_target=True
        ).sum(-1)
        sums = torch.zeros(len(encoded), device=self.device).scatter_add_(0, b, losses)
        return sums / torch.bincount(b, minlength=len(encoded))

    def features(self, encoded, detach=True):
        captured = {}

        def capture(module, args):
            captured["phi"] = args[0]

        def capture_input(module, args):
            captured["x"] = args[0]

        handle = self.mlp.down_proj.register_forward_pre_hook(capture)
        input_handle = self.mlp.register_forward_pre_hook(capture_input)
        try:
            ids, attention, _ = self.batch(encoded, prompt_only=True)
            if detach:
                with torch.no_grad():
                    hidden = self.hidden(ids, attention)
                    positions = attention.sum(1) - 1
                    self.last_prompt_logits = self.model.lm_head(
                        hidden[torch.arange(len(encoded), device=self.device), positions]
                    ).detach()
                    x = captured["x"][torch.arange(len(encoded), device=self.device), positions]
                    self.last_layer_inputs = x.detach()
                    self.original_features = (
                        F.linear(x, self.base_local["up_proj.weight"])
                        * F.silu(F.linear(x, self.base_local["gate_proj.weight"]))
                    ).detach()
            else:
                self.hidden(ids, attention)
            phi = captured["phi"][
                torch.arange(len(encoded), device=self.device), attention.sum(1) - 1
            ]
            return phi.detach() if detach else phi
        finally:
            handle.remove()
            input_handle.remove()

    @torch.no_grad()
    def generate(self, prompts, max_new_tokens=24):
        # Left-padding generation; positions are computed from the attention mask.
        self.tokenizer.padding_side = "left"
        inputs = self.tokenizer(
            prompts, padding=True, return_tensors="pt", add_special_tokens=False
        ).to(self.device)
        prompt_width = inputs.input_ids.shape[1]
        tokenizer = self.tokenizer

        class StopAtNewline(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                texts = tokenizer.batch_decode(
                    input_ids[:, prompt_width:], skip_special_tokens=True
                )
                return torch.tensor(["\n" in text for text in texts], device=input_ids.device)

        output = self.model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            use_cache=True,
            logits_to_keep=1,
            stopping_criteria=StoppingCriteriaList([StopAtNewline()]),
        )
        tail = output[:, inputs.input_ids.shape[1] :]
        predictions = self.tokenizer.batch_decode(tail, skip_special_tokens=True)
        # Padding after a newline stop is not evidence of the model emitting EOS.
        eos_emitted = []
        for row in tail:
            emitted = False
            for length, token in enumerate(row, 1):
                if int(token) == self.tokenizer.eos_token_id:
                    emitted = True
                    break
                if "\n" in self.tokenizer.decode(row[:length], skip_special_tokens=True):
                    break
            eos_emitted.append(emitted)
        return predictions, eos_emitted

    @torch.no_grad()
    def evaluate(self, records, views=(0, 1, 2), batch_size=24):
        rows = self.rows(records, views)
        result = []
        for start in range(0, len(rows), batch_size):
            chunk = rows[start : start + batch_size]
            encoded = [x[2] for x in chunk]
            losses = self.ce(encoded).tolist()
            predictions, eos = self.generate([TEMPLATE.format(r["views"][v]) for r, v, _ in chunk])
            phi = self.features(encoded)
            for i, (r, v, enc) in enumerate(chunk):
                prediction = predictions[i]
                first_token = enc["input_ids"][enc["answer_start"]]
                logits = self.last_prompt_logits[i]
                competitor_logits = logits.clone()
                competitor_logits[first_token] = -torch.inf
                competitor = int(competitor_logits.argmax())
                result.append(
                    {
                        "case_id": r["case_id"],
                        "relation_id": r["relation_id"],
                        "subject_group": r["subject_group"],
                        "view_id": v,
                        "answer_nll": losses[i],
                        "prediction": prediction,
                        "answer_em": score_answer(prediction, r.get("aliases", [r["answer"]])),
                        "eos_emitted": eos[i],
                        "feature_norm": float(phi[i].norm()),
                        "first_answer_nll": float(-logits.log_softmax(-1)[first_token]),
                        "first_answer_margin": float(logits[first_token] - logits[competitor]),
                        "current_competitor_token": competitor,
                        "answer_tokens": enc["answer_tokens"],
                        "prompt_tokens": enc["prompt_tokens"],
                    }
                )
        return result

    def gradient(self, encoded):
        loss = self.ce([encoded]).mean()
        grad = torch.autograd.grad(loss, self.mlp.down_proj.weight)[0]
        return grad.detach(), float(loss.detach())

    def verify_hooks_and_gradients(self, encoded):
        self.configure_trainable("adapt")
        captured = {}

        def capture_input(module, args):
            captured["x"] = args[0].detach()

        def capture_down(module, args, output):
            captured["phi"] = args[0].detach()
            output.retain_grad()
            captured["output"] = output

        handles = [
            self.mlp.register_forward_pre_hook(capture_input),
            self.mlp.down_proj.register_forward_hook(capture_down),
        ]
        try:
            self.model.zero_grad(set_to_none=True)
            self.ce([encoded]).mean().backward()
            with torch.no_grad():
                expected_phi = self.mlp.up_proj(captured["x"]) * F.silu(
                    self.mlp.gate_proj(captured["x"])
                )
                phi_error = float((expected_phi - captured["phi"]).abs().max())
                delta = captured["output"].grad.double().flatten(0, 1)
                phi = captured["phi"].double().flatten(0, 1)
                reconstructed = delta.T @ phi
                actual = self.mlp.down_proj.weight.grad.double()
                grad_error = float((reconstructed - actual).norm() / actual.norm())
                nonanswer_positions = encoded["answer_start"] - 1
                prompt_grad_norm = float(delta[:nonanswer_positions].norm())
            assert phi_error == 0.0
            assert grad_error < 1e-5
            return {
                "phi_max_absolute_error": phi_error,
                "gradient_relative_error": grad_error,
                "earlier_prompt_delta_norm": prompt_grad_norm,
                "all_positions_used": True,
                "trainable_parameters": sum(
                    p.numel() for p in self.model.parameters() if p.requires_grad
                ),
                "pass": True,
            }
        finally:
            for handle in handles:
                handle.remove()
            self.model.zero_grad(set_to_none=True)


def run_baseline(device, shard=0, shards=1):
    engine = QwenExperiment(device)
    candidates = read_json(DATA / "candidates.json")[shard::shards]
    out = DATA / f"baseline-shard-{shard}.jsonl"
    finished = {}
    if out.exists():
        for line in out.read_text().splitlines():
            row = __import__("json").loads(line)
            finished[row["case_id"]] = row
    candidates = [r for r in candidates if r["case_id"] not in finished]
    start_time = time.monotonic()
    for start in range(0, len(candidates), 8):
        chunk = candidates[start : start + 8]
        rows = engine.evaluate(chunk)
        with out.open("a") as f:
            for record in chunk:
                item = {
                    "case_id": record["case_id"],
                    "views": [r for r in rows if r["case_id"] == record["case_id"]],
                }
                f.write(__import__("json").dumps(item, ensure_ascii=False) + "\n")
        if start % 80 == 0:
            print(
                f"baseline shard={shard} {start + len(chunk)}/{len(candidates)} "
                f"elapsed={time.monotonic() - start_time:.1f}s",
                flush=True,
            )
    write_json(
        ARTIFACTS / f"baseline-shard-{shard}-complete.json",
        {"time": now(), "rows": len(candidates) + len(finished), "shards": shards},
    )
