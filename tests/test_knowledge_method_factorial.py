import contextlib
import copy
import json
import random

import pytest
import torch
import torch.nn.functional as F

from llm_memory_editability import knowledge_method_factorial as factorial


def settings():
    return {
        "arms": list(factorial.ARMS),
        "lora_r": 2,
        "lora_alpha": 4,
        "lora_dropout": 0.0,
        "lora_targets": [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head",
        ],
        "gradient_checkpointing": False,
        "kl_temperature": 2.0,
    }


def tiny_model():
    transformers = pytest.importorskip("transformers")
    return transformers.Qwen3ForCausalLM(
        transformers.Qwen3Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=24,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            tie_word_embeddings=True,
        )
    )


def minibatch():
    return factorial.shared.collate(
        [
            {"student": [1, 2], "teacher": [8, 9, 1, 2], "answer": [3, 4]},
            {"student": [5], "teacher": [9, 5], "answer": [7]},
        ],
        pad_id=0,
        device="cpu",
    )


@pytest.mark.parametrize("arm", factorial.ARMS)
def test_eight_parameter_scopes_and_reload(arm):
    if "lora" in arm:
        pytest.importorskip("peft")
    model = factorial.configure_arm(tiny_model(), settings(), arm)
    contract = factorial.parameter_contract(model, arm)
    names = contract["trainable_names"]
    _, scope = factorial.arm_spec(arm)
    if scope == "full":
        assert len(names) == len(list(model.named_parameters()))
    elif scope == "down":
        assert names == [f"model.layers.{layer}.mlp.down_proj.weight" for layer in range(2)]
    elif scope == "down_lora":
        assert len(names) == 4
        assert all(".mlp.down_proj.lora_" in name for name in names)
        assert model.peft_config["default"].target_modules == {"down_proj"}
    else:
        assert all(".lora_" in name for name in names)
        assert any(".lm_head.lora_" in name for name in names)
    assert contract["trainable_parameters"] == sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    saved = factorial.shared.state(model)
    with torch.no_grad():
        next(p for p in model.parameters() if p.requires_grad).add_(1)
    factorial.shared.restore(model, saved)
    for name, parameter in model.named_parameters():
        if name in saved:
            torch.testing.assert_close(parameter, saved[name], rtol=0, atol=0)


@pytest.mark.parametrize("arm", ["pd_full", "pd_down"])
def test_dense_pd_teacher_stays_frozen_when_student_updates(monkeypatch, arm):
    torch.manual_seed(18)
    original = tiny_model()
    student = factorial.configure_arm(copy.deepcopy(original), settings(), arm)
    teacher = copy.deepcopy(original).requires_grad_(False)
    teacher.train()  # The loss must restore evaluation mode even for an external caller.
    monkeypatch.setattr(factorial, "autocast", contextlib.nullcontext)
    batch = minibatch()
    teacher_before = {name: p.detach().clone() for name, p in teacher.named_parameters()}
    student_before = factorial.shared.state(student)
    loss = factorial.batch_loss(student, batch, arm, settings(), teacher)
    s, t = batch["student"], batch["teacher"]
    expected = factorial.shared.forward_kl(
        student(input_ids=s["input_ids"], attention_mask=s["attention_mask"]).logits[
            s["answer_mask"]
        ],
        teacher(input_ids=t["input_ids"], attention_mask=t["attention_mask"]).logits[
            t["answer_mask"]
        ],
        s["answer_mask"].sum(1),
        2.0,
    )
    torch.testing.assert_close(loss, expected)
    loss.backward()
    torch.optim.SGD([p for p in student.parameters() if p.requires_grad], lr=0.1).step()
    assert not teacher.training
    assert all(p.grad is None and not p.requires_grad for p in teacher.parameters())
    for name, parameter in teacher.named_parameters():
        torch.testing.assert_close(parameter, teacher_before[name], rtol=0, atol=0)
    assert any(
        not torch.equal(dict(student.named_parameters())[name], value)
        for name, value in student_before.items()
    )


def test_dense_pd_rejects_missing_trainable_and_storage_alias_teachers(monkeypatch):
    monkeypatch.setattr(factorial, "autocast", contextlib.nullcontext)
    student = factorial.configure_arm(tiny_model(), settings(), "pd_full")
    with pytest.raises(ValueError, match="independent"):
        factorial.batch_loss(student, minibatch(), "pd_full", settings())
    with pytest.raises(ValueError, match="independent"):
        factorial.batch_loss(student, minibatch(), "pd_full", settings(), student)
    teacher = tiny_model()
    with pytest.raises(ValueError, match="trainable"):
        factorial.batch_loss(student, minibatch(), "pd_full", settings(), teacher)
    teacher.requires_grad_(False)
    teacher.model.layers[0].mlp.down_proj.weight = torch.nn.Parameter(
        student.model.layers[0].mlp.down_proj.weight.detach(), requires_grad=False
    )
    with pytest.raises(ValueError, match="storage"):
        factorial.batch_loss(student, minibatch(), "pd_full", settings(), teacher)


@pytest.mark.parametrize("arm", ["pd_lora", "pd_down_lora"])
def test_adapter_teacher_is_base_and_all_base_parameters_stay_frozen(monkeypatch, arm):
    pytest.importorskip("peft")
    torch.manual_seed(33)
    original = tiny_model().eval()
    student = factorial.configure_arm(copy.deepcopy(original), settings(), arm)
    # Nonzero adapters make using the current student as its teacher detectably wrong.
    with torch.no_grad():
        for name, p in student.named_parameters():
            if ".lora_B." in name:
                p.fill_(0.1)
    student.train()
    monkeypatch.setattr(factorial, "autocast", contextlib.nullcontext)
    batch = minibatch()
    loss = factorial.batch_loss(student, batch, arm, settings())
    s, t = batch["student"], batch["teacher"]
    with torch.no_grad():
        targets = original(input_ids=t["input_ids"], attention_mask=t["attention_mask"]).logits[
            t["answer_mask"]
        ]
    logits = student(input_ids=s["input_ids"], attention_mask=s["attention_mask"]).logits[
        s["answer_mask"]
    ]
    expected = factorial.shared.forward_kl(logits, targets, s["answer_mask"].sum(1), 2.0)
    torch.testing.assert_close(loss, expected)
    assert student.training
    loss.backward()
    assert all(p.grad is None for name, p in student.named_parameters() if ".lora_" not in name)
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in student.parameters()
        if p.requires_grad
    )
    # A mistakenly unfrozen base is rejected before the teacher forward pass.
    next(p for name, p in student.named_parameters() if ".lora_" not in name).requires_grad_(True)
    with pytest.raises(ValueError, match="frozen"):
        factorial.batch_loss(student, batch, arm, settings())


@pytest.mark.parametrize("arm", ["sft_full", "sft_lora", "sft_down", "sft_down_lora"])
def test_sft_preserves_shared_token_mean_and_never_loads_teacher(monkeypatch, arm):
    if "lora" in arm:
        pytest.importorskip("peft")
    model = factorial.configure_arm(tiny_model(), settings(), arm)
    monkeypatch.setattr(factorial, "autocast", contextlib.nullcontext)
    monkeypatch.setattr(factorial, "load_model", lambda _: pytest.fail("SFT teacher load"))
    assert factorial.load_teacher(model, settings(), arm) is None
    batch = minibatch()
    actual = factorial.batch_loss(model, batch, arm, settings())
    s = batch["student"]
    logits = model(input_ids=s["input_ids"], attention_mask=s["attention_mask"]).logits
    expected = F.cross_entropy(logits[s["answer_mask"]], s["labels"][s["answer_mask"]])
    torch.testing.assert_close(actual, expected)


def test_dense_teacher_loads_base_not_student_and_lora_does_not_reload(monkeypatch):
    original = tiny_model()
    student = factorial.configure_arm(copy.deepcopy(original), settings(), "pd_down")
    with torch.no_grad():
        next(p for p in student.parameters() if p.requires_grad).add_(1)
    monkeypatch.setattr(factorial, "load_model", lambda _: (copy.deepcopy(original), None))
    teacher = factorial.load_teacher(student, settings(), "pd_down")
    for name, p in teacher.named_parameters():
        torch.testing.assert_close(p, dict(original.named_parameters())[name], rtol=0, atol=0)
    assert not teacher.training and not any(p.requires_grad for p in teacher.parameters())
    monkeypatch.setattr(factorial, "load_model", lambda _: pytest.fail("LoRA teacher load"))
    assert factorial.load_teacher(student, settings(), "pd_lora") is None
    assert factorial.load_teacher(student, settings(), "pd_down_lora") is None


@pytest.mark.parametrize("arm", factorial.ARMS)
def test_encoding_and_fit_choose_objective_not_parameter_scope(monkeypatch, arm):
    expected = "pd_lora" if arm.startswith("pd_") else "sft_lora"
    monkeypatch.setattr(factorial.shared, "encode_sample", lambda tok, row, mapped, limit: mapped)
    monkeypatch.setattr(
        factorial.shared,
        "fit_diagnostic",
        lambda model, tok, cfg, batch, mapped: {"mapped": mapped},
    )
    assert factorial.encode_sample(None, {}, arm, 10) == expected
    assert factorial.fit_diagnostic(None, None, {}, None, arm) == {"mapped": expected, "arm": arm}


def test_train_builds_order_once_per_epoch_and_lazy_encodes_only_selected_rows(
    tmp_path, monkeypatch
):
    model = tiny_model()
    cfg = {
        **settings(),
        "seed": 17,
        "expected_training_rows": 5,
        "max_sequence_tokens": 128,
        "batch_size": 2,
        "epochs": 2,
        "learning_rate": 0.01,
        "weight_decay": 0.0,
        "warmup_steps": 100,
        "max_grad_norm": 1.0,
        "checkpoint_steps": 1,
        "log_steps": 1,
    }
    shared_dir = tmp_path / "shared"
    shared_dir.mkdir()
    (shared_dir / "prepare-complete.json").write_text("{}")
    (shared_dir / "train.jsonl").write_text(
        "".join(json.dumps({"id": str(i)}) + "\n" for i in range(5))
    )
    monkeypatch.setattr(
        factorial, "load_model", lambda _: (model, type("Tok", (), {"pad_token_id": 0})())
    )
    monkeypatch.setattr(factorial, "autocast", contextlib.nullcontext)
    monkeypatch.setattr(factorial.shared, "seed_all", lambda seed: torch.manual_seed(seed))
    monkeypatch.setattr(factorial.shared, "rng_state", lambda: {})
    actual_collate = factorial.shared.collate
    monkeypatch.setattr(
        factorial.shared, "collate", lambda rows, pad: actual_collate(rows, pad, device="cpu")
    )
    selected, seeds = [], []
    real_random = random.Random

    def random_for_epoch(seed):
        seeds.append(seed)
        return real_random(seed)

    def encode(tok, row, arm, limit):
        selected.append(int(row["id"]))
        return {"id": row["id"], "student": [1, 2], "teacher": [8, 1, 2], "answer": [3]}

    monkeypatch.setattr(factorial.random, "Random", random_for_epoch)
    monkeypatch.setattr(factorial, "encode_sample", encode)
    factorial.train(cfg, tmp_path, "sft_down", smoke_steps=6)
    expected = []
    for epoch in range(2):
        order = list(range(5))
        real_random(cfg["seed"] + epoch).shuffle(order)
        expected.extend(order)
    assert seeds == [17, 18]
    assert selected == expected + [0, 1]  # The final two encodings are the reload audit sample.
    manifest = json.loads((tmp_path / "runs/sft_down/manifest.json").read_text())
    assert manifest["rows"] == 5 and manifest["steps"] == 6
    assert manifest["teacher_mode"] == "none"


def test_mid_epoch_resume_matches_uninterrupted_weights_and_exposure(tmp_path, monkeypatch):
    torch.manual_seed(91)
    original = tiny_model()
    cfg = {
        **settings(),
        "seed": 17,
        "expected_training_rows": 5,
        "max_sequence_tokens": 128,
        "batch_size": 2,
        "epochs": 2,
        "learning_rate": 0.01,
        "weight_decay": 0.0,
        "warmup_steps": 100,
        "max_grad_norm": 1.0,
        "checkpoint_steps": 1,
        "log_steps": 1,
    }
    reference, resumed = tmp_path / "reference", tmp_path / "resumed"
    for directory in (reference, resumed):
        shared_dir = directory / "shared"
        shared_dir.mkdir(parents=True)
        (shared_dir / "prepare-complete.json").write_text("{}")
        (shared_dir / "train.jsonl").write_text(
            "".join(json.dumps({"id": str(i)}) + "\n" for i in range(5))
        )
    interrupt = False

    def load(_):
        model = copy.deepcopy(original)
        if interrupt:
            forward, calls = model.forward, 0

            def interrupted_forward(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 3:
                    raise RuntimeError("simulated interruption before third optimizer step")
                return forward(*args, **kwargs)

            model.forward = interrupted_forward
        return model, type("Tok", (), {"pad_token_id": 0})()

    monkeypatch.setattr(factorial, "load_model", load)
    monkeypatch.setattr(factorial, "autocast", contextlib.nullcontext)
    monkeypatch.setattr(factorial.shared, "seed_all", lambda seed: torch.manual_seed(seed))
    monkeypatch.setattr(factorial.shared, "rng_state", lambda: {"torch": torch.get_rng_state()})
    monkeypatch.setattr(
        factorial.shared, "restore_rng", lambda state: torch.set_rng_state(state["torch"])
    )
    actual_collate = factorial.shared.collate
    monkeypatch.setattr(
        factorial.shared, "collate", lambda rows, pad: actual_collate(rows, pad, device="cpu")
    )
    selected = []

    def encode(tok, row, arm, limit):
        index = int(row["id"])
        selected.append(index)
        return {"id": row["id"], "student": [1, 2], "teacher": [8, 1, 2], "answer": [3 + index]}

    monkeypatch.setattr(factorial, "encode_sample", encode)
    factorial.train(cfg, reference, "sft_down", smoke_steps=6)
    reference_order = selected[:-2]
    selected.clear()
    interrupt = True
    with pytest.raises(RuntimeError, match="simulated interruption"):
        factorial.train(cfg, resumed, "sft_down", smoke_steps=6)
    checkpoint = torch.load(resumed / "runs/sft_down/resume.pt", weights_only=False)
    assert checkpoint["step"] == 2 and checkpoint["exposure"] == 4
    assert not (resumed / "runs/sft_down/complete.json").exists()
    prefix = selected[: checkpoint["exposure"]]
    selected.clear()
    interrupt = False
    factorial.train(cfg, resumed, "sft_down", smoke_steps=6)
    assert prefix + selected[:-2] == reference_order
    expected = torch.load(reference / "runs/sft_down/weights.pt", weights_only=True)
    actual = torch.load(resumed / "runs/sft_down/weights.pt", weights_only=True)
    assert expected.keys() == actual.keys()
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
    complete = json.loads((resumed / "runs/sft_down/complete.json").read_text())
    assert complete["step"] == 6 and complete["exposure"] == 10
