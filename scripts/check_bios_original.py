"""Small end-to-end engineering checks; never count these as development results."""

import copy
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer

from llm_memory_editability.bios_original_data import TASKS, load_json, prepare_world, write_json
from llm_memory_editability.bios_original_train import evaluate, finetune, pretrain


def main():
    cfg = copy.deepcopy(load_json("configs/bios-original-development-v1.json"))
    cfg.update(
        data_root="data/bios-original-preflight-v1",
        result_root="results/bios-original-preflight-v3",
        people=200,
    )
    cfg["model"].update(width=64, layers=2, heads=4)
    cfg["pretrain"].update(passes=2, nodes=[0, 1, 2], warmup_steps=1)
    cfg["adapt"].update(epochs=1, qv_rank=2, embedding_rank=4)
    cfg["task"].update(epochs=1, rule_repetitions_per_epoch=1)
    cfg["evaluation"].update(splits=["dev"], views=[0], max_new_tokens=4)
    tok = Tokenizer.from_file("data/bios-original-v1/tokenizer/tokenizer.json")
    prepare_world(cfg, 142701, tok)
    pretrain(cfg, 142701, 1427, "MP", "cuda:0", stop_after_pass=1)
    parent = pretrain(cfg, 142701, 1427, "MP", "cuda:0")
    base = Path(cfg["result_root"]) / "world-142701/init-1427"
    uninterrupted = copy.deepcopy(cfg)
    uninterrupted["result_root"] += "-uninterrupted"
    comparison = pretrain(uninterrupted, 142701, 1427, "MP", "cuda:0")
    a = torch.load(parent, map_location="cpu", weights_only=False)["model"]
    b = torch.load(comparison, map_location="cpu", weights_only=False)["model"]
    max_diff = max((a[k] - b[k]).abs().max().item() for k in a)
    assert max_diff == 0, f"Restart changed final weights: {max_diff}"
    adapted = finetune(cfg, 142701, 1427, "MP", "adapt", parent, "cuda:0", tok)
    for cond in ["MP", "MP-R", "MP-Random", "MP-CoT"]:
        end = finetune(cfg, 142701, 1427, cond, "task", adapted, "cuda:0", tok)
        evaluate(
            cfg,
            142701,
            end,
            base / cond / "task/evaluation",
            "cuda:0",
            tok,
            tasks=TASKS,
            cot=cond == "MP-CoT",
        )
    result = dict(
        status="passed",
        scope="200-person 2-layer engineering check; not science evidence",
        pretrain_passes=2,
        restart_weights_max_difference=max_diff,
        adaptations=1,
        task_branches=4,
        evaluations=4,
    )
    write_json("docs/development-artifacts/bios-original-v1/pipeline-preflight.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
