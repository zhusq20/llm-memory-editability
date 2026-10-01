#!/usr/bin/env python3
"""Frozen-model behavior and MLP input interventions. No optimizer is constructed."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import transformers
from prepare_twohop_frozen import ART, DATA, ROOT, digest, normalize, read, write
from transformers import AutoModelForCausalLM, AutoTokenizer

RESULTS = ROOT / "results/twohop-frozen-v1"


def extract(text):
    if "Final answer:" in text:
        text = text.rsplit("Final answer:", 1)[1]
    text = text.strip().split("\n")[0].strip()
    text = re.sub(r"^(?:Answer|Intermediate entity|Bridge):\s*", "", text, flags=re.I)
    return text.strip().strip('"').strip("*")


def grade(text, answer, aliases=()):
    predicted = normalize(extract(text))
    gold = normalize(answer)
    x, y = Counter(predicted.split()), Counter(gold.split())
    common = sum((x & y).values())
    f1 = 2 * common / (sum(x.values()) + sum(y.values())) if common else 0.0
    return dict(
        em=predicted == gold,
        f1=f1,
        alias_em=predicted in {normalize(a) for a in [answer, *aliases]},
    )


def prompt_content(question, context="", bridge=None, proposal=False, cot=False):
    header = (
        "Use the supplied passages to answer the question."
        if context
        else "Answer the factual question."
    )
    if proposal:
        instruction = (
            "Identify the intermediate entity needed to connect the two relations. "
            "Output only that entity's name, not the final answer."
        )
    elif cot:
        instruction = (
            "Reason briefly through the two relations, then write Final answer: "
            "followed by only the answer."
        )
    else:
        instruction = "Output only the short answer, without an explanation."
    parts = [header, instruction]
    if context:
        parts.append("Passages:\n" + context)
    parts.append("Question: " + question)
    if bridge is not None:
        parts.append("Proposed intermediate entity: " + bridge)
    return "\n\n".join(parts)


class Engine:
    def __init__(self, model_cfg, cfg):
        self.cfg, self.model_cfg = cfg, model_cfg
        self.device = torch.device(model_cfg["device"])
        self.tok = AutoTokenizer.from_pretrained(ROOT / model_cfg["path"], local_files_only=True)
        self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "left"
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                ROOT / model_cfg["path"],
                local_files_only=True,
                torch_dtype=torch.bfloat16,
                attn_implementation=cfg["attention"],
            )
            .to(self.device)
            .eval()
            .requires_grad_(False)
        )
        self.layers = sorted(
            {int(len(self.model.model.layers) * f) for f in cfg["layer_fractions"]}
        )
        self.stats = Counter()

    def prompt(self, question, **kwargs):
        content = prompt_content(question, **kwargs)
        if self.model_cfg["chat"]:
            return self.tok.apply_chat_template(
                [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True
            )
        # Fixed format examples teach only output formatting, not benchmark relations.
        return "Question: What is 2 + 2?\nAnswer: 4\n\n" + content + "\nAnswer:"

    def tokens(self, prompts):
        batch = self.tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(
            self.device
        )
        if batch.input_ids.shape[1] > self.cfg["max_input_tokens"]:
            raise ValueError("Input exceeds frozen context budget; no silent truncation")
        return batch

    @contextmanager
    def patch(self, layer, deltas, positions=None):
        handles = []
        if layer is not None:

            def hook(module, args):
                x = args[0]
                # Generation patches the prefill only. Later cached decoding is unmodified.
                if positions is None and x.shape[1] == 1:
                    return None
                pos = (
                    torch.full((x.shape[0],), x.shape[1] - 1, device=x.device)
                    if positions is None
                    else positions
                )
                out = x.clone()
                out[torch.arange(len(x), device=x.device), pos] += deltas.to(x.dtype)
                return (out, *args[1:])

            handles.append(self.model.model.layers[layer].mlp.register_forward_pre_hook(hook))
        try:
            yield
        finally:
            for h in handles:
                h.remove()

    def generate(self, prompts, max_tokens=None, layer=None, deltas=None):
        batch = self.tokens(prompts)
        started = time.perf_counter()
        with self.patch(layer, deltas), torch.no_grad():
            output = self.model.generate(
                **batch,
                max_new_tokens=max_tokens or self.cfg["generation_tokens"],
                do_sample=False,
                use_cache=True,
                pad_token_id=self.tok.pad_token_id,
                eos_token_id=self.tok.eos_token_id,
            )
        suffix = output[:, batch.input_ids.shape[1] :]
        self.stats["generation_calls"] += 1
        self.stats["prompt_tokens"] += int(batch.attention_mask.sum())
        self.stats["generated_nonpad_tokens"] += int((suffix != self.tok.pad_token_id).sum())
        self.stats["generation_seconds"] += time.perf_counter() - started
        return self.tok.batch_decode(suffix, skip_special_tokens=True)

    def generate_many(self, prompts, max_tokens=None):
        outputs = []
        for start in range(0, len(prompts), self.cfg["batch"]):
            outputs += self.generate(prompts[start : start + self.cfg["batch"]], max_tokens)
        return outputs

    def capture(self, prompts):
        # One prompt at a time ensures source coordinates do not depend on padding length.
        all_states = []
        for prompt in prompts:
            states, handles = {}, []
            for layer in self.layers:

                def hook(module, args, layer=layer, states=states):
                    states[layer] = args[0][0, -1].detach().float()

                handles.append(self.model.model.layers[layer].mlp.register_forward_pre_hook(hook))
            try:
                with torch.no_grad():
                    self.model(**self.tokens([prompt]), use_cache=False, logits_to_keep=1)
            finally:
                for h in handles:
                    h.remove()
            all_states.append(states)
        return all_states

    def scores(self, prompt, candidates, layer=None, deltas=None, gradients=False):
        # Each row is a full candidate continuation; mean token log probability is explicit.
        prefix = self.tok.encode(prompt, add_special_tokens=False)
        targets = [self.tok.encode(" " + s, add_special_tokens=False) for s in candidates]
        sequences = [prefix + t for t in targets]
        width = max(map(len, sequences))
        ids = torch.full(
            (len(sequences), width), self.tok.pad_token_id, device=self.device, dtype=torch.long
        )
        attention = torch.zeros_like(ids)
        mask = torch.zeros_like(ids, dtype=torch.bool)
        for i, seq in enumerate(sequences):
            ids[i, : len(seq)] = torch.tensor(seq, device=self.device)
            attention[i, : len(seq)] = 1
            mask[i, len(prefix) : len(seq)] = True
        positions = torch.full((len(sequences),), len(prefix) - 1, device=self.device)
        phi_holder = []
        handle = None
        if gradients:

            def retain(module, args):
                args[0].retain_grad()
                phi_holder.append(args[0])

            handle = self.model.model.layers[layer].mlp.down_proj.register_forward_pre_hook(retain)
        try:
            with self.patch(layer, deltas, positions), torch.set_grad_enabled(gradients):
                logits = self.model(input_ids=ids, attention_mask=attention, use_cache=False).logits
                lp = logits[:, :-1].float().log_softmax(-1).gather(-1, ids[:, 1:, None]).squeeze(-1)
                scores = (lp * mask[:, 1:]).sum(1) / mask[:, 1:].sum(1)
                if gradients:
                    gap = scores[0] - scores[1]
                    gap.backward()
                    grad_x = deltas.grad.detach().float().sum(0)
                    grad_phi = phi_holder[0].grad[:, len(prefix) - 1].detach().float().sum(0)
                    return scores.detach().float(), grad_x, grad_phi
                return scores.detach().float()
        finally:
            if handle is not None:
                handle.remove()

    def phi(self, layer, x):
        mlp = self.model.model.layers[layer].mlp
        with torch.no_grad():
            x = x.to(next(mlp.parameters()).dtype)
            return (mlp.act_fn(mlp.gate_proj(x)) * mlp.up_proj(x)).float()


def append(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def behavior(engine, cases, out):
    if out.exists():
        raise FileExistsError(out)
    for dataset in ["mquake", "2wiki"]:
        for split in ["development", "evaluation"]:
            part = [r for r in cases if r["dataset"] == dataset and r["split"] == split]
            outputs = {}
            for name, question_key in [
                ("direct", "question"),
                ("first", "q1"),
                ("second_oracle", "q2"),
            ]:
                outputs[name] = engine.generate_many(
                    [engine.prompt(r[question_key], context=r["context"]) for r in part]
                )
            outputs["bridge_proposal"] = engine.generate_many(
                [engine.prompt(r["question"], context=r["context"], proposal=True) for r in part]
            )
            outputs["self_bridge"] = engine.generate_many(
                [
                    engine.prompt(r["question"], context=r["context"], bridge=extract(b))
                    for r, b in zip(part, outputs["bridge_proposal"], strict=True)
                ]
            )
            outputs["oracle_bridge"] = engine.generate_many(
                [
                    engine.prompt(r["question"], context=r["context"], bridge=r["bridge"])
                    for r in part
                ]
            )
            outputs["scaffold_chain"] = engine.generate_many(
                [
                    engine.prompt(
                        r["q2_template"].replace("{bridge}", extract(b)), context=r["context"]
                    )
                    for r, b in zip(part, outputs["first"], strict=True)
                ]
            )
            outputs["cot"] = engine.generate_many(
                [engine.prompt(r["question"], context=r["context"], cot=True) for r in part],
                engine.cfg["cot_tokens"],
            )
            for i, r in enumerate(part):
                row = {k: r[k] for k in ["id", "dataset", "split", "group"]}
                row["outputs"] = {k: v[i] for k, v in outputs.items()}
                row["metrics"] = {}
                for name, value in row["outputs"].items():
                    bridge_task = name in ["first", "bridge_proposal"]
                    row["metrics"][name] = grade(
                        value,
                        r["bridge"] if bridge_task else r["answer"],
                        r["bridge_aliases"] if bridge_task else r["aliases"],
                    )
                    if not bridge_task:
                        row["metrics"][name]["filtered_alias_em"] = grade(
                            value, r["answer"], r.get("unambiguous_aliases", r["aliases"])
                        )["alias_em"]
                row["scaffold_template_valid"] = r["scaffold_template_valid"]
                append(out, row)
            print(
                json.dumps(
                    dict(
                        stage="behavior",
                        dataset=dataset,
                        split=split,
                        rows=len(part),
                        stats=engine.stats,
                    )
                ),
                flush=True,
            )


def mechanism(engine, cases, behavior_path, out):
    if out.exists():
        raise FileExistsError(out)
    behavior_rows = {
        (r["dataset"], r["id"]): r
        for r in [json.loads(line) for line in behavior_path.read_text().splitlines()]
    }
    for index, r in enumerate([r for r in cases if r["mechanism"]]):
        base = {k: r[k] for k in ["dataset", "id", "split", "group"]}
        donor = r["wrong_donor"]
        if donor is None:
            append(out, {**base, "status": "missing_same_relation_donor"})
            continue
        context = r["support_context"]
        prompt = engine.prompt(r["question"], context=context)
        proposed = extract(behavior_rows[(r["dataset"], r["id"])]["outputs"]["bridge_proposal"])
        first_prompt = engine.prompt(r["q1"], context=r["first_context"])
        oracle_prompt = engine.prompt(r["q2"], context=context)
        wrong_prompt = engine.prompt(donor["q2"], context=donor["support_context"])
        self_prompt = engine.prompt(r["question"], context=context, bridge=proposed)
        states = engine.capture([prompt, first_prompt, oracle_prompt, wrong_prompt, self_prompt])
        for layer in engine.layers:
            x, first, oracle, wrong, own = [s[layer] for s in states]
            error = oracle - x
            rng = torch.Generator(device=engine.device).manual_seed(
                int(
                    hashlib.sha256(f"{r['dataset']}:{r['id']}:{layer}".encode()).hexdigest()[:8], 16
                )
            )
            random = torch.randn(x.shape, generator=rng, device=engine.device)
            random *= error.norm() / random.norm().clamp_min(1e-12)
            wrong_direction = wrong - x
            wrong_direction *= error.norm() / wrong_direction.norm().clamp_min(1e-12)
            directions = dict(
                zero=torch.zeros_like(x),
                first_only=first - x,
                oracle_025=error,
                oracle_050=error,
                oracle_100=error,
                wrong_oracle=wrong_direction,
                random=random,
                reverse=error,
                self_bridge=own - x,
            )
            names = list(engine.cfg["patches"])
            deltas = torch.stack([directions[n] * engine.cfg["patches"][n] for n in names])
            # Gold and designated wrong successor are used only for offline sensitivity measurement.
            candidates = [r["answer"], donor["answer"]]
            probe = torch.zeros(
                (2, len(x)), device=engine.device, dtype=torch.float32, requires_grad=True
            )
            clean_scores, gx, gp = engine.scores(prompt, candidates, layer, probe, gradients=True)
            probe_gap = float(clean_scores[0] - clean_scores[1])
            scores = engine.scores(
                prompt, candidates * len(names), layer, deltas.repeat_interleave(2, dim=0)
            ).reshape(-1, 2)
            clean_gap = float(scores[0, 0] - scores[0, 1])
            generations = engine.generate([prompt] * len(names), layer=layer, deltas=deltas)
            delta_phi = engine.phi(layer, x[None] + deltas) - engine.phi(layer, x[None])
            pred_x = deltas @ gx
            pred_phi = delta_phi @ gp
            archive = (
                RESULTS / engine.name / "vectors" / f"{r['dataset']}-{r['id']}-layer-{layer}.npz"
            )
            archive.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                archive,
                x=x.cpu().numpy(),
                deltas=deltas.cpu().numpy(),
                delta_phi=delta_phi.cpu().numpy(),
                grad_x=gx.cpu().numpy(),
                grad_phi=gp.cpu().numpy(),
                scores=scores.cpu().numpy(),
                clean_scores=clean_scores.cpu().numpy(),
            )
            for j, name in enumerate(names):
                gap = float(scores[j, 0] - scores[j, 1])
                append(
                    out,
                    {
                        **base,
                        "status": "complete",
                        "layer": layer,
                        "condition": name,
                        "output": generations[j],
                        "metrics": grade(generations[j], r["answer"], r["aliases"]),
                        "wrong_successor_em": grade(generations[j], donor["answer"])["em"],
                        "clean_gap": clean_gap,
                        "probe_batch_gap_difference": probe_gap - clean_gap,
                        "gap": gap,
                        "gap_change": gap - clean_gap,
                        "predicted_input_change": float(pred_x[j]),
                        "predicted_feature_change": float(pred_phi[j]),
                        "input_error": float(deltas[j].norm()),
                        "feature_error": float(delta_phi[j].norm()),
                        "first_donor_final_answer_literal": normalize(r["answer"])
                        in normalize(first_prompt),
                        "wrong_donor_id": donor["id"],
                        "reference_scope": "2wiki_support_only"
                        if context
                        else "mquake_closed_book",
                    },
                )
        print(json.dumps(dict(stage="mechanism", case=index + 1, **base)), flush=True)


def parameter_digest(model):
    h = hashlib.sha256()
    for name, parameter in model.named_parameters():
        h.update(name.encode())
        h.update(parameter.detach().cpu().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["small", "main"])
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    cfg = read(ROOT / "configs/twohop-frozen-v1.json")
    torch.set_num_threads(4)
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    engine = Engine(cfg["models"][args.model], cfg)
    engine.name = args.model
    out = RESULTS / args.model
    out.mkdir(parents=True, exist_ok=True)
    if args.preflight:
        prompt = engine.prompt("What is the capital of France?")
        baseline = engine.generate([prompt])
        zero = torch.zeros((1, engine.model.config.hidden_size), device=engine.device)
        patched = engine.generate([prompt], layer=engine.layers[0], deltas=zero)
        assert baseline == patched
        x = engine.capture([prompt])[0][engine.layers[0]]
        probe = torch.zeros((2, len(x)), device=engine.device, requires_grad=True)
        scores, gx, gp = engine.scores(prompt, ["Paris", "London"], engine.layers[0], probe, True)
        assert torch.isfinite(gx).all() and gx.norm() > 0 and gp.norm() > 0
        write(
            ART / f"preflight-{args.model}.json",
            dict(
                generation=baseline,
                zero_equivalent=True,
                scores=scores.tolist(),
                grad_input_norm=float(gx.norm()),
                grad_feature_norm=float(gp.norm()),
                layers=engine.layers,
                parameters=sum(p.numel() for p in engine.model.parameters()),
                trainable=sum(p.numel() for p in engine.model.parameters() if p.requires_grad),
                environment=dict(
                    python=sys.version,
                    torch=torch.__version__,
                    transformers=transformers.__version__,
                    numpy=np.__version__,
                ),
            ),
        )
        return
    lock = ART / f"execution-lock-{args.model}.json"
    if lock.exists():
        raise FileExistsError("Execution already started; inspect state instead of overwriting")
    before = parameter_digest(engine.model)
    write(
        lock,
        dict(
            started_utc=datetime.now(timezone.utc).isoformat(),
            model=cfg["models"][args.model],
            config_sha256=digest(ROOT / "configs/twohop-frozen-v1.json"),
            script_sha256=digest(Path(__file__)),
            preparation_script_sha256=digest(ROOT / "scripts/prepare_twohop_frozen.py"),
            cases_sha256=digest(DATA / "cases.json"),
            protocol_sha256=digest(ROOT / "docs/hebbian-learning-plan-v1.md"),
            parameter_sha256=before,
            layers=engine.layers,
            environment=dict(
                python=sys.version,
                torch=torch.__version__,
                transformers=transformers.__version__,
                numpy=np.__version__,
            ),
            files={
                p.name: digest(p)
                for p in (ROOT / cfg["models"][args.model]["path"]).iterdir()
                if p.is_file()
            },
        ),
    )
    started = time.perf_counter()
    cases = read(DATA / "cases.json")
    behavior(engine, cases, out / "behavior.jsonl")
    mechanism(engine, cases, out / "behavior.jsonl", out / "mechanism.jsonl")
    after = parameter_digest(engine.model)
    assert before == after and all(not p.requires_grad for p in engine.model.parameters())
    write(
        ART / f"completion-{args.model}.json",
        dict(
            state="complete",
            finished_utc=datetime.now(timezone.utc).isoformat(),
            seconds=time.perf_counter() - started,
            unchanged_parameters=before == after,
            parameter_sha256_after=after,
            stats=engine.stats,
            max_memory_bytes=torch.cuda.max_memory_allocated(engine.device),
            files={
                str(p.relative_to(out)): digest(p) for p in sorted(out.rglob("*")) if p.is_file()
            },
        ),
    )


if __name__ == "__main__":
    main()
