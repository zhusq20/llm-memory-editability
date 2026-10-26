import importlib.util
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_knowledge_use_coverage.py"
SPEC = importlib.util.spec_from_file_location("coverage_runner", SCRIPT)
R = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R)


@pytest.mark.parametrize("objective", ["pd", "sft"])
def test_accumulation_matches_whole_objective_and_counts_actual_forward_inputs(
    monkeypatch, objective
):
    samples = [
        {"id": "1", "student": [1, 2], "teacher": [1, 2, 3], "answer": [4, 5]},
        {"id": "2", "student": [1], "teacher": [1, 2], "answer": [6]},
        {"id": "3", "student": [1, 2, 3], "teacher": [1, 2, 3, 4], "answer": [7, 8, 9]},
    ]
    collate = R.K.collate
    monkeypatch.setattr(R.K, "collate", lambda rows, pad: collate(rows, pad, "cpu"))
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    model = SimpleNamespace(config=SimpleNamespace(pad_token_id=0), parameter=parameter)

    def loss(model, batch, arm, config, teacher):
        student = batch["student"]
        if objective == "pd":
            assert teacher is not None
            value = torch.stack(
                [
                    y[m].float().mean()
                    for y, m in zip(student["labels"], student["answer_mask"], strict=True)
                ]
            ).mean()
        else:
            assert teacher is None
            value = student["labels"][student["answer_mask"]].float().mean()
        return model.parameter * value

    monkeypatch.setattr(R.F, "batch_loss", loss)
    value, counts = R.loss_backward(
        model, object(), samples, objective, {"microbatch_size": 2, "scope": "down"}
    )
    expected = (4.5 + 6 + 8) / 3 if objective == "pd" else sum(range(4, 10)) / 6
    assert value == pytest.approx(expected)
    assert parameter.grad.item() == pytest.approx(expected)
    expected_counts = {}
    for side in ("student", "teacher") if objective == "pd" else ("student",):
        chunks = [collate(samples[:2], 0, "cpu")[side], collate(samples[2:], 0, "cpu")[side]]
        expected_counts[f"{side}_tokens"] = sum(int(b["attention_mask"].sum()) for b in chunks)
        expected_counts[f"{side}_padded_tokens"] = sum(b["input_ids"].numel() for b in chunks)
    assert counts == expected_counts


def test_cloud_requires_exact_endpoint_counts_and_reads_back_node(tmp_path, monkeypatch):
    batch = tmp_path / "batch"
    run = batch / "runs/formal-pd-A-addition"
    node = run / "node-1024"
    config = {"scope": "down"}
    R.K.write(run / "run.json", {"spec": {"config": config}})
    R.K.write(run / "complete.json", {"step": 2, "supervised_tokens": 1024, "nodes": [1024]})
    R.K.write(node / "audit.json", {"passed": True, "weights_sha256": "frozen"})
    R.K.write(node / "evaluation/summary.json", {"complete": True, "groups": {}})
    R.K.write(
        batch / f"tracking-wandb/{run.name}/state.json",
        {
            "entity": "zsqzz",
            "project": "p",
            "run_id": "id",
            "url": "https://wandb.ai/zsqzz/p/runs/id",
        },
    )
    remote = SimpleNamespace(
        config={"spec": {"config": config}},
        state="finished",
        summary={"scientific_final_step": 2, "exposure/supervised_tokens": 512},
    )
    monkeypatch.setitem(
        sys.modules,
        "wandb",
        SimpleNamespace(Api=lambda **kw: SimpleNamespace(run=lambda identity: remote)),
    )
    with pytest.raises(ValueError, match="count differs"):
        R.sync_node_tracking(config, batch, run, node)
    assert not (node / "cloud-verification.json").exists()
    remote.summary["exposure/supervised_tokens"] = 1024
    R.sync_node_tracking(config, batch, run, node)
    assert json.loads((node / "cloud-verification.json").read_text())["passed"]
    assert remote.summary["independently_reloaded"]


def test_cloud_summary_accepts_flattened_transport_but_not_wrong_counts_or_weights():
    payload = {
        "audit": {"passed": True, "weights_sha256": "original"},
        "evaluation": {"seconds": 29.292897284030914, "correct": 64, "accuracy": 1.0},
    }
    key = "nodes/node-200192"
    remote = R.flatten_summary(payload, key)
    remote[f"{key}.evaluation.seconds"] = 29.29289728403091
    assert R.node_summary_equal(remote, key, payload)
    assert R.node_summary_equal({key: payload}, key, payload)
    for field, bad in (
        ("audit.weights_sha256", "different"),
        ("evaluation.correct", 63),
        ("evaluation.accuracy", 0.999),
        ("audit.passed", False),
    ):
        changed = {**remote, f"{key}.{field}": bad}
        assert not R.node_summary_equal(changed, key, payload)
    del remote[f"{key}.audit.weights_sha256"]
    assert not R.node_summary_equal(remote, key, payload)


def test_reload_rejects_wrong_arm_before_loading_model(tmp_path, monkeypatch):
    config = {"scope": "down"}
    node = tmp_path / "runs/formal-pd-A-addition/node-512"
    node.mkdir(parents=True)
    torch.save(
        {
            "config": config,
            "tokens": 512,
            "identity": {"cohort": "formal", "objective": "pd", "arm": "B", "stage": "addition"},
        },
        node / "resume.pt",
    )
    monkeypatch.setattr(R.K, "load_model", lambda config: pytest.fail("Must reject before CUDA"))
    with pytest.raises(ValueError, match="trajectory identity"):
        R.reload_and_evaluate(config, tmp_path, "formal", "pd", "A", "addition", 512)


@pytest.mark.parametrize("objective", ["pd", "sft"])
def test_cpu_trajectory_saves_exact_token_nodes_and_restorable_adam(
    tmp_path, monkeypatch, objective
):
    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 0

        def encode(self, text, **kwargs):
            return [ord(c) % 31 + 1 for c in text]

        def apply_chat_template(self, messages, **kwargs):
            return messages[-1]["content"] + "?"

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(32, 4)
            self.model = torch.nn.Module()
            self.model.mlp = torch.nn.Module()
            self.model.mlp.down_proj = torch.nn.Linear(4, 4, bias=False)
            self.head = torch.nn.Linear(4, 32, bias=False)
            self.config = SimpleNamespace(pad_token_id=0, use_cache=False)

        def forward(self, input_ids, attention_mask, use_cache):
            values = self.embed(input_ids) * attention_mask[..., None]
            values = values.cumsum(1) / attention_mask.cumsum(1).clamp_min(1)[..., None]
            return SimpleNamespace(logits=self.head(self.model.mlp.down_proj(values)))

    def load(config):
        torch.manual_seed(123)
        return Model(), Tokenizer()

    monkeypatch.syspath_prepend(str(SCRIPT.parent))
    monkeypatch.setattr(R.K, "load_model", load)
    monkeypatch.setattr(R.K, "autocast", nullcontext)
    monkeypatch.setattr(R.F, "autocast", nullcontext)
    collate = R.K.collate
    monkeypatch.setattr(R.K, "collate", lambda samples, pad: collate(samples, pad, "cpu"))
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    config = {
        "scope": "down",
        "arms": ["pd_down", "sft_down"],
        "max_sequence_tokens": 128,
        "seed": 23,
        "microbatch_size": 2,
        "learning_rate": 0.01,
        "weight_decay": 0.1,
        "max_grad_norm": 1.0,
        "development_token_nodes": [4, 8],
        "main_tokens_per_update": 4,
        "replay_tokens_per_update": 2,
        "warmup_token_fraction": 0.02,
        "kl_temperature": 2.0,
        "retention_kl_weight": 0.2,
    }
    data = tmp_path / "data/development"
    rows = [
        {
            "id": "a",
            "context": "document",
            "question": "question",
            "pd_answer": "abc",
            "sft_answer": "abc",
            "fact_ids": ["fact-a"],
            "context_fact_ids": ["fact-a"],
        }
    ]
    for name in ("prepared-A.jsonl", "prepared-replay.jsonl"):
        R.write_rows(data / name, rows)
    R.K.write(
        data / "prepare-complete.json",
        {"sha256": {p.name: R.K.digest(p) for p in data.glob("*.jsonl")}},
    )
    R.train(config, tmp_path, "development", objective, "A", "addition", [4, 8], smoke=True)
    directory = tmp_path / f"engineering/development-{objective}-A-addition"
    done = R.read(directory / "complete.json")
    assert done["supervised_tokens"] == 8 and done["replay_tokens"] == 4
    assert done["nodes"] == [4, 8] and done["smoke"] is True
    for node in (4, 8):
        saved = torch.load(directory / f"node-{node}/resume.pt", weights_only=False)
        assert saved["tokens"] == saved["main_stream"]["tokens"] == node
        assert saved["replay_stream"]["tokens"] == node // 2
        assert saved["identity"]["objective"] == objective
        assert saved["optimizer"]["state"]
        model, _ = load(config)
        model = R.F.configure_arm(model, config, f"{objective}_down")
        R.K.restore(model, saved["weights"])
        assert all(
            torch.equal(p.detach(), saved["weights"][name])
            for name, p in model.named_parameters()
            if p.requires_grad
        )
