"""Measure native recall, extraction and parity at the same frozen bioS weights."""

import argparse
import calendar
import json
import time
from collections import defaultdict
from pathlib import Path

import torch
from run_bios_original_trajectory import measure, sha
from tokenizers import Tokenizer

from llm_memory_editability.bios_original_data import answer, date, question
from llm_memory_editability.bios_original_diagnostics import operation_prompt, parse_date
from llm_memory_editability.bios_original_train import generate, load_model, normalize, setup
from llm_memory_editability.grok_depth import write_json

ROOT = Path("docs/development-artifacts/bios-same-weight-v1")
CONFIG = Path("configs/bios-original-development-v1.json")
CASES = Path("docs/development-artifacts/bios-original-trajectory-v1/cases.json")
SOURCES = [
    Path(__file__),
    Path("scripts/run_bios_original_trajectory.py"),
    *[
        Path(f"src/llm_memory_editability/{name}.py")
        for name in (
            "bios_original_data",
            "bios_original_model",
            "bios_original_train",
            "bios_original_diagnostics",
            "grok_depth",
        )
    ],
]


def prepare():
    cfg = json.loads(CONFIG.read_text())
    endpoints = []
    for world in cfg["world_seeds"]:
        for condition in ("S", "M", "MP"):
            for stage in ("pretrain", "adapt", "task"):
                name = "pass-540.pt" if stage == "pretrain" else "final.pt"
                checkpoint = Path(cfg["result_root"]) / (
                    f"world-{world}/init-1427/{condition}/{stage}/{name}"
                )
                endpoints.append(
                    dict(
                        world=world,
                        condition=condition,
                        stage=stage,
                        checkpoint=str(checkpoint),
                        checkpoint_sha256=sha(checkpoint),
                    )
                )
    lock = dict(
        created_unix=time.time(),
        config=cfg,
        cases_sha256=sha(CASES),
        sources={str(p): sha(p) for p in SOURCES},
        endpoints=endpoints,
        people_per_world=32,
        views=[0, 1],
        native_batch=4,
        qa_batch=16,
        max_new_tokens=96,
        native_max_new_tokens=12,
        analysis="Paired fixed development people; no new training or selection by results",
        limitations=[
            "Native birthcity prefix supplies the true date.",
            "Pretraining/adapt endpoints have not been trained on the parity rule format.",
            "Autonomous and oracle conditions change prompts and computation budget.",
        ],
        inputs={
            str(p): sha(p)
            for p in [
                CONFIG,
                CASES,
                Path(cfg["data_root"]) / "tokenizer/tokenizer.json",
                *[Path(cfg["data_root"]) / f"world-{w}/people.json" for w in cfg["world_seeds"]],
            ]
        },
    )
    if (ROOT / "lock.json").exists():
        raise FileExistsError("Existing frozen plan")
    write_json(ROOT / "lock.json", lock)


@torch.inference_mode()
def run(world, device):
    lock = json.loads((ROOT / "lock.json").read_text())
    for path, expected in (lock["sources"] | lock["inputs"]).items():
        assert sha(path) == expected, path
    setup(14700, device)
    torch.cuda.set_per_process_memory_fraction(0.12, device)
    cfg = lock["config"]
    tok = Tokenizer.from_file(str(Path(cfg["data_root"]) / "tokenizer/tokenizer.json"))
    cases = [r for r in json.loads(CASES.read_text()) if r["world"] == world]
    people = json.loads((Path(cfg["data_root"]) / f"world-{world}/people.json").read_text())
    selected = [people[i] for i in sorted({r["person_id"] for r in cases})]
    for endpoint in lock["endpoints"]:
        if endpoint["world"] != world:
            continue
        key = f"w{world}-{endpoint['condition']}-{endpoint['stage']}"
        out = ROOT / "endpoints" / f"{key}.json"
        if out.exists():
            continue
        assert sha(endpoint["checkpoint"]) == endpoint["checkpoint_sha256"]
        start = time.monotonic()
        model = load_model(
            endpoint["checkpoint"], cfg, device, adapters=endpoint["stage"] != "pretrain"
        ).eval()
        model.requires_grad_(False)
        native = []
        for i in range(0, len(cases), lock["native_batch"]):
            native.extend(
                measure(
                    model,
                    cases[i : i + lock["native_batch"]],
                    tok,
                    device,
                    lock["native_max_new_tokens"],
                )
            )
        rows = []
        for p in selected:
            for view in lock["views"]:
                for attr in ("date", "birthcity"):
                    rows.append(
                        dict(
                            person_id=p["id"],
                            view=view,
                            mode="qa_" + attr,
                            prompt=question(p, attr, view),
                            target=answer(p, attr),
                        )
                    )

        def generate_all(batch_rows, current_model):
            for i in range(0, len(batch_rows), lock["qa_batch"]):
                batch = batch_rows[i : i + lock["qa_batch"]]
                texts, eos = generate(
                    current_model, tok, [r["prompt"] for r in batch], device, lock["max_new_tokens"]
                )
                for row, text, ended in zip(batch, texts, eos, strict=True):
                    row.update(
                        generated=text,
                        eos=ended,
                        correct=normalize(text) == normalize(row["target"]),
                    )
                    row["correct_with_eos"] = row["correct"] and ended

        generate_all(rows, model)
        lookup = {(r["person_id"], r["view"]): r for r in rows if r["mode"] == "qa_date"}
        tasks = []
        for p in selected:
            for view in lock["views"]:
                generated_date = lookup[p["id"], view]["generated"]
                parsed = parse_date(generated_date)
                for mode, prompt in (
                    ("direct_parity", question(p, "parity", view)),
                    ("oracle_parity", operation_prompt("parity", date(p))),
                    ("autonomous_parity", operation_prompt("parity", generated_date)),
                ):
                    tasks.append(
                        dict(
                            person_id=p["id"],
                            view=view,
                            mode=mode,
                            prompt=prompt,
                            target=answer(p, "parity"),
                            generated_date=generated_date,
                            input_parseable=parsed is not None,
                            input_month_correct=parsed is not None and parsed[1] == p["month"],
                            true_month=calendar.month_name[p["month"]],
                        )
                    )
        generate_all(tasks, model)
        totals = defaultdict(lambda: dict(n=0, correct=0))
        for row in native:
            t = totals["native_" + row["attribute"]]
            t["n"] += 1
            t["correct"] += int(row["attribute_exact"])
        for row in rows + tasks:
            t = totals[f"{row['mode']}/view{row['view']}"]
            t["n"] += 1
            t["correct"] += int(row["correct"])
        write_json(
            out,
            dict(
                endpoint=endpoint,
                seconds=time.monotonic() - start,
                lock_sha256=sha(ROOT / "lock.json"),
                native=native,
                predictions=rows + tasks,
                totals=dict(totals),
            ),
        )
        print(json.dumps(dict(endpoint=key, totals=dict(totals))), flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "run"])
    parser.add_argument("--world", type=int)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    else:
        run(args.world, args.device)
