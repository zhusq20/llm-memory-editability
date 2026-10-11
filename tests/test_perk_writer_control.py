"""CPU development contracts for the prospective controlled PERK extension."""

from __future__ import annotations

import copy
import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from llm_memory_editability.perk_writer_control import (  # noqa: E402
    ControlledPERKWriter,
    PairedWriterBatch,
    assert_paired_writer_initial_state,
)

AUTHOR_SOURCE = Path(
    os.environ.get(
        "PERK_AUTHOR_SOURCE",
        str(
            Path(__file__).resolve().parents[1]
            / "results/perk-paper-reproduction-v1/reference/perk"
        ),
    )
)


def features(rows):
    ids = torch.tensor(rows, dtype=torch.long)
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": ids.clone()}


@pytest.fixture()
def pair():
    # Explicit development tensor fixtures, with no claim that these are QA2
    # atomic or compositional labels. The contract accepts independently reviewed
    # supervised inputs; it never derives QA labels from materials.
    return PairedWriterBatch(
        features([[2, 3, 4, 5, 1], [6, 7, 8, 9, 1]]),
        features([[2, 4, 1], [6, 8, 1]]),
        features([[2, 6, 9, 1]]),
        material_id="explicit-tensor-development-fixture",
    )


@pytest.fixture()
def writer(tmp_path, request):
    if not AUTHOR_SOURCE.exists():
        pytest.skip("Frozen PERK author source is required")
    for package in ("lightning", "peft", "transformers", "omegaconf", "datasets"):
        pytest.importorskip(package)
    from omegaconf import OmegaConf
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    from llm_memory_editability.perk_reproduction import load_author, resolve_config, seed_all

    torch.set_num_threads(1)
    seed_all(617)
    model_dir = tmp_path / "tiny-gpt2"
    raw = Tokenizer(WordLevel({f"t{i}": i for i in range(32)} | {"[UNK]": 32}, unk_token="[UNK]"))
    raw.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw,
        eos_token="t1",
        pad_token="t1",
        unk_token="[UNK]",
        model_max_length=128,
    )
    tokenizer.save_pretrained(model_dir)
    GPT2LMHeadModel(
        GPT2Config(
            vocab_size=33,
            n_positions=128,
            n_embd=16,
            n_layer=2,
            n_head=2,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            eos_token_id=1,
            pad_token_id=1,
        )
    ).save_pretrained(model_dir)
    # Build an author module with externally supplied features. The subclass
    # bypasses only file loading, and leaves CausalLM/LoRA/Higher/unroll intact.
    author = load_author(AUTHOR_SOURCE)

    class TensorFixtureModule(author.MetaLMModule):
        def load_dataset(self):
            pass

    overrides = getattr(request, "param", {})
    config = resolve_config(
        {
            "reproduction_kind": "development_smoke",
            "model_name_or_path": str(model_dir),
            "data_dir": "unused-explicit-tensor-fixture",
            "num_workers": 0,
            "lora_rank": 2,
            "lora_alpha": 2,
            "bf16": False,
            "n_inner_iter": 2,
            "inner_lr": 0.003,
            "inner_funct": "default",
            "unroll_start": 0,
            "max_new_tokens": 4,
        }
        | overrides,
        tmp_path / "output",
    )
    result = ControlledPERKWriter(TensorFixtureModule(OmegaConf.create(config)))
    result.eval()
    return result


def state_copy(model):
    return {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}


def assert_state_equal(model, before):
    after = model.state_dict()
    assert after.keys() == before.keys()
    for name, tensor in before.items():
        torch.testing.assert_close(after[name], tensor, rtol=0, atol=0)


def test_paired_exposure_shares_material_and_records_distinct_outer_supervision(pair):
    first = pair.exposure("atomic", n_inner_iter=4)
    second = pair.exposure("compositional", n_inner_iter=4)
    for field in first:
        if field.startswith("inner_") or field.startswith("material_"):
            assert first[field] == second[field]
    assert first["inner_subsequence_presentations"] == 8
    assert first["inner_supervised_tokens"] == 32
    assert first["outer_queries"] == 2
    assert second["outer_queries"] == 1
    assert first["outer_sha256"] != second["outer_sha256"]
    with pytest.raises(ValueError, match="Unknown"):
        pair.outer_features("unreviewed")


def test_material_is_copied_and_query_labels_are_required():
    material = features([[2, 3, 4]])
    pair = PairedWriterBatch(material, features([[2, 4]]), features([[3, 4]]), material_id="m")
    before = pair.exposure("atomic", n_inner_iter=1)
    material["input_ids"].fill_(9)
    exported = pair.material_features()
    exported["input_ids"].fill_(10)
    assert pair.exposure("atomic", n_inner_iter=1) == before
    with pytest.raises(ValueError, match="exactly"):
        PairedWriterBatch(
            pair.material_features(), {}, features([[3, 4]]), material_id="missing-labels"
        )
    with pytest.raises(ValueError, match="next-token"):
        unsupervised = features([[2, 4]])
        unsupervised["labels"][:, 1:] = -100
        PairedWriterBatch(
            pair.material_features(), unsupervised, features([[3, 4]]), material_id="m"
        )


@pytest.mark.parametrize("field", ["input_ids", "attention_mask", "labels"])
def test_author_batch_pair_rejects_any_write_difference(pair, field):
    def as_author(condition):
        outer = pair.outer_features(condition)
        return outer | {f"train_{k}": v for k, v in pair.material_features().items()}

    atomic = as_author("atomic")
    composed = as_author("compositional")
    accepted = PairedWriterBatch.from_author_batches(atomic, composed, material_id="matched")
    assert accepted.exposure("atomic", n_inner_iter=2)["inner_input_tokens"] == 20
    composed[f"train_{field}"][0, 0] += 1
    with pytest.raises(ValueError, match=f"train_{field}"):
        PairedWriterBatch.from_author_batches(atomic, composed, material_id="mismatched")


def test_outer_optimizer_excludes_reader_and_lora_but_inner_adapts_them(writer, pair):
    outer = writer.outer_optimizer()
    membership = {id(p) for group in outer.param_groups for p in group["params"]}
    assert membership == {id(p) for _, p in writer.outer_named_parameters()}
    for name, parameter in writer.author_module.model.lm_model.named_parameters():
        assert id(parameter) not in membership
        assert parameter.requires_grad == ("lora" in name)
    before = state_copy(writer)
    initial_adapters = {
        name: parameter.detach().clone()
        for name, parameter in writer.author_module.model.named_parameters()
        if "lora" in name
    }
    with writer.adapt_material(pair.material_features()) as adapted:
        assert torch.isfinite(adapted.inner_loss)
        values = adapted.adapter_values()
        assert values.keys() == initial_adapters.keys()
        assert any(
            not torch.equal(values[name], tensor) for name, tensor in initial_adapters.items()
        )
    assert_state_equal(writer, before)


@pytest.mark.parametrize(
    "writer",
    [
        {"n_inner_iter": 2, "inner_funct": "default", "unroll_start": 0},
        {"n_inner_iter": 4, "inner_funct": "truncated", "unroll_start": 3},
    ],
    indirect=True,
    ids=["default-two-step", "author-four-step-truncated"],
)
@pytest.mark.parametrize("condition", ["atomic", "compositional"])
def test_real_gpt2_control_metagradients_and_only_controls_outer_update(writer, pair, condition):
    before = state_copy(writer)
    outer = writer.outer_optimizer()
    writer.zero_grad(set_to_none=True)
    torch.manual_seed(971)
    output = writer.meta_step(pair, condition)
    assert torch.isfinite(output["loss"])
    output["loss"].backward()
    assert_state_equal(writer, before)
    for prefix in ("weighting_model.", "inner_schedular."):
        gradients = [
            parameter.grad
            for name, parameter in writer.outer_named_parameters()
            if name.startswith(prefix) and parameter.grad is not None
        ]
        assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
        assert sum(float(gradient.abs().sum()) for gradient in gradients) > 0
    # Gradients are retained through the initial LoRA for the inner graph, but
    # those values are excluded from the outer optimizer and cannot be updated.
    assert any(
        parameter.grad is not None
        for name, parameter in writer.author_module.model.named_parameters()
        if "lora" in name
    )
    outer.step()
    controls = {"author_module.model.weighting_model.", "author_module.inner_schedular."}
    changed = []
    for name, tensor in writer.state_dict().items():
        if not torch.equal(tensor, before[name]):
            changed.append(name)
            assert any(name.startswith(prefix) for prefix in controls)
    assert changed
    writer.zero_grad(set_to_none=True)
    assert all(parameter.grad is None for parameter in writer.parameters())


def test_material_resets_and_queries_are_read_only_without_shared_kv(writer, pair):
    before = state_copy(writer)
    modes = {name: module.training for name, module in writer.named_modules()}
    query = {key: value for key, value in pair.outer_features("atomic").items() if key != "labels"}
    torch.manual_seed(673)
    with torch.no_grad(), writer.adapt_material(pair.material_features()) as first:
        values = first.adapter_values()
        logits = first.read(query)["logits"]
        first.read(pair.outer_features("compositional"))
        repeated = first.read(query)["logits"]
        torch.testing.assert_close(repeated, logits, rtol=0, atol=0)
        for name, tensor in first.adapter_values().items():
            torch.testing.assert_close(tensor, values[name], rtol=0, atol=0)
        assert not logits.requires_grad
        with pytest.raises(ValueError, match="optional labels only"):
            first.read(query | {"past_key_values": ()})
        with pytest.raises(RuntimeError, match="overlapping"):
            with writer.adapt_material(pair.material_features()):
                pass
    with pytest.raises(RuntimeError, match="context has ended"):
        first.read(query)
    torch.manual_seed(673)
    with writer.adapt_material(pair.material_features()) as reset:
        for name, tensor in reset.adapter_values().items():
            torch.testing.assert_close(tensor, values[name], rtol=0, atol=0)
        torch.testing.assert_close(reset.read(query)["logits"], logits, rtol=0, atol=0)
    assert_state_equal(writer, before)
    assert {name: module.training for name, module in writer.named_modules()} == modes


def test_fast_generation_uses_adapted_logits_and_left_padding(writer, pair):
    before = state_copy(writer)
    ids = torch.tensor([[1, 2, 3], [2, 3, 4]], dtype=torch.long)
    mask = torch.tensor([[0, 1, 1], [1, 1, 1]], dtype=torch.long)
    torch.manual_seed(673)
    with writer.adapt_material(pair.material_features()) as adapted:
        adapted._fast_model.eval()
        positions = mask.cumsum(-1) - 1
        positions.masked_fill_(mask == 0, 1)
        with torch.no_grad():
            expected = (
                adapted._fast_model(
                    {
                        "input_ids": ids,
                        "attention_mask": mask,
                        "position_ids": positions,
                        "use_cache": False,
                    }
                )["logits"][:, -1]
                .argmax(-1)
                .tolist()
            )
        actual = adapted.generate(ids, mask, eos_token_id=1, max_new_tokens=3)
        assert [row[0] for row in actual] == expected
        assert actual == adapted.generate(ids, mask, eos_token_id=1, max_new_tokens=3)
        with pytest.raises(ValueError, match="left padding"):
            adapted.generate(ids, mask.flip(1), eos_token_id=1, max_new_tokens=3)
        with pytest.raises(ValueError, match="context window"):
            adapted.generate(ids, mask, eos_token_id=1, max_new_tokens=128)
    assert_state_equal(writer, before)


@pytest.mark.skipif(
    os.environ.get("PERK_WRITER_CUDA_CONTRACT") != "1",
    reason="CUDA contract runs only in the assigned independent container",
)
@pytest.mark.parametrize(
    "writer", [{"n_inner_iter": 4, "inner_funct": "truncated", "unroll_start": 3}], indirect=True
)
def test_actual_cuda_matrix_backward_and_two_writer_updates(writer, pair):
    assert torch.cuda.is_available()
    writer.cuda()
    torch.manual_seed(971)
    matrix = torch.randn(128, 128, device="cuda", requires_grad=True)
    (matrix @ matrix.T).square().mean().backward()
    torch.cuda.synchronize()
    assert torch.isfinite(matrix.grad).all()
    pair = PairedWriterBatch(
        {k: v.cuda() for k, v in pair.material_features().items()},
        {k: v.cuda() for k, v in pair.outer_features("atomic").items()},
        {k: v.cuda() for k, v in pair.outer_features("compositional").items()},
        material_id=pair.material_id,
    )
    before = state_copy(writer.author_module.model.lm_model)
    optimizer = writer.outer_optimizer()
    for condition in ("atomic", "compositional"):
        writer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = writer.meta_step(pair, condition)
        output["loss"].backward()
        assert all(
            torch.isfinite(p.grad).all()
            for _, p in writer.outer_named_parameters()
            if p.grad is not None
        )
        optimizer.step()
        torch.cuda.synchronize()
    assert_state_equal(writer.author_module.model.lm_model, before)
    with writer.adapt_material(pair.material_features()) as view:
        query = pair.outer_features("compositional")
        ids = view.generate(
            query["input_ids"], query["attention_mask"], eos_token_id=1, max_new_tokens=3
        )
        assert len(ids) == 1 and 1 <= len(ids[0]) <= 3


@pytest.mark.parametrize("scope", ["base", "lora", "control"])
def test_paired_initialization_checks_reader_adapter_and_write_controls(writer, scope):
    second = copy.deepcopy(writer)
    assert_paired_writer_initial_state(writer, second)
    if scope == "control":
        _, parameter = next(second.outer_named_parameters())
    else:
        parameter = next(
            parameter
            for name, parameter in second.author_module.model.lm_model.named_parameters()
            if ("lora" in name) == (scope == "lora")
        )
    with torch.no_grad():
        parameter.add_(0.01)
    with pytest.raises(ValueError, match="initial state differs"):
        assert_paired_writer_initial_state(writer, second)


def test_paired_reader_mathematical_configuration_is_checked(writer):
    second = copy.deepcopy(writer)
    second.author_module.model.config.layer_norm_epsilon *= 2
    with pytest.raises(ValueError, match="architecture/configuration"):
        assert_paired_writer_initial_state(writer, second)
    second = copy.deepcopy(writer)
    next(
        module
        for module in second.author_module.model.modules()
        if isinstance(module, torch.nn.Dropout)
    ).p += 0.1
    with pytest.raises(ValueError, match="dropout"):
        assert_paired_writer_initial_state(writer, second)


@pytest.mark.parametrize(
    "field",
    ["eps", "scale_attn_weights", "scale_attn_by_inverse_layer_idx", "reorder_and_upcast_attn"],
)
def test_paired_actual_reader_math_is_checked_even_with_identical_saved_config(writer, field):
    second = copy.deepcopy(writer)
    target = next(
        module
        for module in second.author_module.model.lm_model.modules()
        if (isinstance(module, torch.nn.LayerNorm) if field == "eps" else hasattr(module, field))
    )
    setattr(target, field, target.eps * 2 if field == "eps" else not getattr(target, field))
    assert (
        writer.author_module.model.config.to_dict() == second.author_module.model.config.to_dict()
    )
    assert_state_equal(second, state_copy(writer))
    with pytest.raises(ValueError, match="LayerNorm.eps|reader computation"):
        assert_paired_writer_initial_state(writer, second)


def test_paired_computing_config_is_checked_independently_of_saved_config(writer):
    second = copy.deepcopy(writer)
    second.author_module.model.lm_model.config.scale_attn_weights = False
    assert (
        writer.author_module.model.config.to_dict() == second.author_module.model.config.to_dict()
    )
    with pytest.raises(ValueError, match="architecture/configuration.*computing"):
        assert_paired_writer_initial_state(writer, second)


@pytest.mark.parametrize(
    "writer",
    [
        {"n_inner_iter": 2, "inner_funct": "default", "unroll_start": 0},
        {"n_inner_iter": 4, "inner_funct": "truncated", "unroll_start": 3},
    ],
    indirect=True,
    ids=["default-two-step", "author-four-step-truncated"],
)
def test_extension_retains_author_weighted_clm_adamw_unroll_and_gradients(writer, pair):
    author = writer.author_module
    batch = (
        pair.outer_features("atomic")
        | {f"train_{key}": value for key, value in pair.material_features().items()}
        | {"print_out": {}}
    )
    before = state_copy(writer)
    torch.manual_seed(823)
    reference = author.step(batch)
    reference["loss"].backward()
    reference_gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in writer.outer_named_parameters()
        if parameter.grad is not None
    }
    writer.zero_grad(set_to_none=True)
    torch.manual_seed(823)
    controlled = writer.meta_step(pair, "atomic")
    controlled["loss"].backward()
    torch.testing.assert_close(controlled["loss"], reference["loss"], rtol=0, atol=0)
    torch.testing.assert_close(controlled["inner_loss"], reference["inner_loss"], rtol=0, atol=0)
    for name, parameter in writer.outer_named_parameters():
        if name in reference_gradients:
            torch.testing.assert_close(parameter.grad, reference_gradients[name], rtol=0, atol=0)
    assert_state_equal(writer, before)
    second = copy.deepcopy(writer)
    second.author_module.hparams.n_inner_iter += 1
    with pytest.raises(ValueError, match="n_inner_iter"):
        assert_paired_writer_initial_state(writer, second)


def test_write_uses_only_material_when_paired_outer_supervision_changes(writer, pair):
    original = state_copy(writer)
    adapter_values = []
    inner_exposure = []
    for condition in ("atomic", "compositional"):
        torch.manual_seed(791)
        with writer.adapt_material(pair.material_features(), track_higher_grads=True) as adapted:
            adapter_values.append(adapted.adapter_values())
            assert torch.isfinite(adapted.supervised_loss(pair.outer_features(condition)))
        inner_exposure.append(
            {
                k: v
                for k, v in pair.exposure(condition, n_inner_iter=2).items()
                if k.startswith("inner_")
            }
        )
    for name, tensor in adapter_values[0].items():
        torch.testing.assert_close(adapter_values[1][name], tensor, rtol=0, atol=0)
    assert inner_exposure[0] == inner_exposure[1]
    assert_state_equal(writer, original)


def test_controlled_writer_rejects_trainable_base_reader(writer):
    parameter = next(
        parameter
        for name, parameter in writer.author_module.model.lm_model.named_parameters()
        if "lora" not in name
    )
    parameter.requires_grad_(True)
    with pytest.raises(ValueError, match="Base reader must be frozen"):
        ControlledPERKWriter(writer.author_module)
