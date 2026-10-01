#!/usr/bin/env python3
"""Post-hoc output-budget control; identical evaluation cases and CoT prompt."""

import time

import torch
from run_twohop_frozen import (
    ART,
    DATA,
    RESULTS,
    ROOT,
    Engine,
    append,
    digest,
    grade,
    parameter_digest,
    read,
    write,
)


def main():
    cfg = read(ROOT / "configs/twohop-frozen-v1.json")
    torch.set_num_threads(4)
    torch.manual_seed(cfg["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    model_cfg = dict(cfg["models"]["main"], device="cuda:2")
    engine = Engine(model_cfg, cfg)
    cases = [r for r in read(DATA / "cases.json") if r["split"] == "evaluation"]
    out = RESULTS / "main-cot256.jsonl"
    if out.exists():
        raise FileExistsError(out)
    before = parameter_digest(engine.model)
    write(
        ART / "cot256-lock.json",
        dict(
            scope="Post-hoc format/budget control; not original matched-output-budget result.",
            max_tokens=256,
            cases=[dict(dataset=r["dataset"], id=r["id"]) for r in cases],
            prompt="Identical to primary CoT",
            model=model_cfg,
            parameter_sha256=before,
            runner_sha256=digest(ROOT / "scripts/run_twohop_frozen.py"),
            source_sha256=digest(ROOT / "scripts/check_twohop_frozen_cot.py"),
        ),
    )
    start = time.perf_counter()
    for dataset in ["mquake", "2wiki"]:
        part = [r for r in cases if r["dataset"] == dataset]
        outputs = engine.generate_many(
            [engine.prompt(r["question"], context=r["context"], cot=True) for r in part],
            max_tokens=256,
        )
        for row, output in zip(part, outputs, strict=True):
            append(
                out,
                dict(
                    dataset=dataset,
                    id=row["id"],
                    output=output,
                    has_final_marker="Final answer:" in output,
                    metrics=grade(output, row["answer"], row["aliases"]),
                ),
            )
        print(dataset, len(part), engine.stats, flush=True)
    after = parameter_digest(engine.model)
    assert before == after
    write(
        ART / "cot256-completion.json",
        dict(
            state="complete",
            n=len(cases),
            seconds=time.perf_counter() - start,
            unchanged_parameters=True,
            parameter_sha256_after=after,
            stats=engine.stats,
            output_sha256=digest(out),
        ),
    )


if __name__ == "__main__":
    main()
