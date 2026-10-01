#!/usr/bin/env python3
"""Fixed eight-case FP32 replication of every layer and intervention."""

import argparse
import time

import torch
from run_twohop_frozen import ART, DATA, RESULTS, ROOT, Engine, digest, mechanism, read, write


class TailScoreEngine(Engine):
    def scores(self, prompt, candidates, layer=None, deltas=None, gradients=False):
        prefix = self.tok.encode(prompt, add_special_tokens=False)
        leading = "" if self.model_cfg["chat"] else " "
        targets = [self.tok.encode(leading + s, add_special_tokens=False) for s in candidates]
        length = max(map(len, targets))
        width = len(prefix) + length
        ids = torch.full(
            (len(targets), width), self.tok.pad_token_id, device=self.device, dtype=torch.long
        )
        attention = torch.zeros_like(ids)
        mask = torch.zeros((len(targets), length), device=self.device, dtype=torch.bool)
        for i, target in enumerate(targets):
            seq = prefix + target
            ids[i, : len(seq)] = torch.tensor(seq, device=self.device)
            attention[i, : len(seq)] = 1
            mask[i, : len(target)] = True
        positions = torch.full((len(targets),), len(prefix) - 1, device=self.device)
        holder, handle = [], None
        if gradients:

            def retain(module, args):
                args[0].retain_grad()
                holder.append(args[0])

            handle = self.model.model.layers[layer].mlp.down_proj.register_forward_pre_hook(retain)
        try:
            with self.patch(layer, deltas, positions), torch.set_grad_enabled(gradients):
                logits = self.model(
                    input_ids=ids,
                    attention_mask=attention,
                    use_cache=False,
                    logits_to_keep=length + 1,
                ).logits
                lp = (
                    logits[:, :-1]
                    .float()
                    .log_softmax(-1)
                    .gather(-1, ids[:, len(prefix) :, None])
                    .squeeze(-1)
                )
                scores = (lp * mask).sum(1) / mask.sum(1)
                if gradients:
                    (scores[0] - scores[1]).backward()
                    return (
                        scores.detach().float(),
                        deltas.grad.detach().float().sum(0),
                        holder[0].grad[:, len(prefix) - 1].detach().float().sum(0),
                    )
                return scores.detach().float()
        finally:
            if handle is not None:
                handle.remove()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["small", "main"])
    parser.add_argument("--device")
    args = parser.parse_args()
    cfg = read(ROOT / "configs/twohop-frozen-v1.json")
    torch.set_num_threads(4)
    torch.manual_seed(cfg["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    model_cfg = dict(cfg["models"][args.model])
    if args.device:
        model_cfg["device"] = args.device
    engine = TailScoreEngine(model_cfg, cfg)
    engine.model.float()
    example = engine.prompt("What is the capital of France?")
    full_scores = Engine.scores(engine, example, ["Paris", "London"])
    tail_scores = engine.scores(example, ["Paris", "London"])
    assert torch.allclose(full_scores, tail_scores, atol=1e-5, rtol=1e-5)
    engine.name = args.model + "-fp32"
    cases = read(DATA / "cases.json")
    selected = []
    for dataset in ["mquake", "2wiki"]:
        selected += [
            r
            for r in cases
            if r["dataset"] == dataset and r["split"] == "evaluation" and r["mechanism"]
        ][:4]
    out = RESULTS / engine.name
    out.mkdir(parents=True, exist_ok=True)
    lock = ART / f"precision-lock-{args.model}.json"
    if lock.exists():
        raise FileExistsError(lock)
    write(
        lock,
        dict(
            cases=[dict(dataset=r["dataset"], id=r["id"]) for r in selected],
            dtype="float32",
            device=model_cfg["device"],
            tail_full_score_max_difference=float((full_scores - tail_scores).abs().max()),
            weight_note="BF16 weights promoted to FP32; computation changes, no training.",
            selection="First four evaluation mechanism cases per dataset; no outcome filter.",
            runner_sha256=digest(ROOT / "scripts/run_twohop_frozen.py"),
            script_sha256=digest(ROOT / "scripts/check_twohop_frozen_precision.py"),
        ),
    )
    start = time.perf_counter()
    mechanism(engine, selected, RESULTS / args.model / "behavior.jsonl", out / "mechanism.jsonl")
    assert all(not p.requires_grad for p in engine.model.parameters())
    write(
        ART / f"precision-completion-{args.model}.json",
        dict(
            state="complete",
            cases=len(selected),
            seconds=time.perf_counter() - start,
            stats=engine.stats,
            max_memory_bytes=torch.cuda.max_memory_allocated(engine.device),
            files={str(p.relative_to(out)): digest(p) for p in out.rglob("*") if p.is_file()},
        ),
    )


if __name__ == "__main__":
    main()
