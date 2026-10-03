"""Architecture contracts: pairing, routing objective, connections and reload."""

import io

import pytest
import torch

from llm_memory_editability.parametric_architecture import (
    ARMS,
    HyperConnection,
    combine_router_fractions,
    construct,
    flop_ledger,
    initialization_manifest,
    optimizer_for,
    parameter_ledger,
    residual_mix,
    router_balance_loss,
    sinkhorn,
)
from llm_memory_editability.realworld_composition import construct as historical_construct

torch.set_num_threads(2)
TINY = dict(vocab_size=43, positions=24, hidden_size=16, attention_heads=4, dropout=0.0)


def model_for(arm, config=None, seed=137):
    return construct(
        TINY if config is None else config, {"architecture": arm, "initialization": seed}, "cpu"
    )


@pytest.mark.parametrize("arm", ["D4", "D8", "L4R2"])
def test_dense_path_exact_historical_initialization_logits_and_gradients(arm):
    model = model_for(arm)
    blocks, repeats, _ = ARMS[arm]
    old = historical_construct(
        TINY, dict(initialization=137, unique_layers=blocks, repeats=repeats), "cpu"
    )
    for name, parameter in old.state_dict().items():
        assert torch.equal(parameter, model.state_dict()[name]), name
    x = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
    positions = torch.tensor([[1, 3], [0, 2]])
    actual, expected = model(x, positions), old(x, positions)
    assert torch.equal(actual, expected)
    actual.square().sum().backward()
    expected.square().sum().backward()
    for (name, p), (_, q) in zip(model.named_parameters(), old.named_parameters(), strict=True):
        assert torch.equal(p.grad, q.grad), name


@pytest.mark.parametrize("arm", list(ARMS))
def test_actual_counts_and_readout_tied(arm):
    model = model_for(arm)
    ledger = parameter_ledger(model)
    assert ledger["total_parameters"] == ledger["expected_parameters"]
    assert ledger["trainable_parameters"] == ledger["total_parameters"]
    assert sum("wte.weight" in n for n, _ in model.named_parameters()) == 1
    assert not any("lm_head" in n for n, _ in model.named_parameters())


@pytest.mark.parametrize(
    "arm,expected",
    [
        ("D4", 67736832),
        ("D8", 96088320),
        ("L4R2", 67736832),
        ("M8", 209500416),
        ("W8", 209408256),
        ("IHC8", 96530848),
        ("HC8", 97317552),
        ("M4", 124442880),
        ("LM4R2", 124442880),
    ],
)
def test_full_size_actual_shapes_match_frozen_parameter_counts(arm, expected):
    config = dict(
        vocab_size=50257, positions=1024, hidden_size=768, attention_heads=12, dropout=0.1
    )
    with torch.device("meta"):
        model = construct(config, {"architecture": arm, "initialization": 137}, "meta")
    assert parameter_ledger(model)["total_parameters"] == expected


def test_paired_common_tensors_and_independent_module_rng():
    reference = model_for("D8")
    baseline = initialization_manifest(reference)["shared_tensor_sha256"]
    for arm in ARMS:
        model = model_for(arm)
        manifest = initialization_manifest(model)
        assert all(
            baseline[name] == value for name, value in manifest["shared_tensor_sha256"].items()
        )
        if arm in {"M8", "W8", "HC8", "IHC8"}:
            assert manifest["introduced_tensor_sha256"]
    dense, loop = model_for("M4"), model_for("LM4R2")
    assert all(torch.equal(p, loop.state_dict()[name]) for name, p in dense.state_dict().items())
    assert len(loop.transformer.h) == 4 and loop.repeats == 2
    identity, full = model_for("IHC8"), model_for("HC8")
    for left, right in zip(identity.connections, full.connections, strict=True):
        assert torch.equal(left.projection.weight, right.projection.weight[:8])
        assert torch.equal(left.pre_bias, right.pre_bias)
        assert torch.equal(left.post_bias, right.post_bias)
        assert not hasattr(left, "alpha_res") and not hasattr(left, "res_bias")


@pytest.mark.parametrize("arm", ["D8", "W8", "M8", "LM4R2", "IHC8", "HC8"])
def test_causality_padding_and_batch_invariant_logits(arm):
    model = model_for(arm).eval()
    short = torch.tensor([[1, 2, 3]])
    alone = model(short)
    batched = torch.tensor([[1, 2, 3, 0, 0], [8, 7, 6, 5, 4]])
    mask = torch.tensor([[True, True, True, False, False], [True] * 5])
    joined = model(batched, valid_mask=mask)
    torch.testing.assert_close(joined[0, :3], alone[0], atol=2e-6, rtol=2e-5)
    changed = batched.clone()
    changed[0, 3:] = torch.tensor([39, 40])
    torch.testing.assert_close(model(changed)[0, :3], alone[0], atol=2e-6, rtol=2e-5)
    positions = torch.tensor([[0, 2], [1, 4]])
    gathered = model(batched, positions, valid_mask=mask)
    torch.testing.assert_close(gathered, joined[torch.arange(2)[:, None], positions])


def test_moe_dropless_padding_stats_and_exact_microbatch_aux_gradients():
    model = model_for("M4").eval()
    tokens = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 6, 7, 8], [3, 2, 1, 5, 0]])
    mask = torch.tensor([[True] * 3 + [False] * 2, [True] * 5, [True] * 4 + [False]])
    _, aux = model(tokens, valid_mask=mask, return_aux=True)
    for stats in aux["router_stats"]:
        assert stats["valid_tokens"].item() == 12
        assert stats["assignment_counts"].sum().item() == 24
        assert stats["processed_assignments"].item() == 24
        assert stats["dropped_assignments"].item() == 0
        assert stats["probability_sum"].sum().item() == pytest.approx(12)
    aux["balance_loss"].backward()
    expected = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        prepass = [
            model(tokens[i : i + 1], valid_mask=mask[i : i + 1], return_aux=True)[1]["router_stats"]
            for i in range(3)
        ]
    fractions, total = combine_router_fractions(prepass)
    accumulated = 0.0
    for i in range(3):
        _, small = model(tokens[i : i + 1], valid_mask=mask[i : i + 1], return_aux=True)
        loss = router_balance_loss(small["router_stats"], fractions, total)
        accumulated += loss.detach().item()
        loss.backward()
    assert accumulated == pytest.approx(aux["balance_loss"].item(), rel=1e-6)
    for name, parameter in model.named_parameters():
        if name in expected:
            torch.testing.assert_close(parameter.grad, expected[name], atol=3e-7, rtol=1e-4)


def test_moe_eval_padding_statistics_unchanged_and_top2_task_router_gradient():
    model = model_for("M4").eval()
    short = torch.tensor(
        [[0, 1, 2]]
    )  # zero may be a real EOS: never infer validity from token IDs.
    _, first = model(short, return_aux=True)
    padded = torch.tensor([[0, 1, 2, 0, 0]])
    mask = torch.tensor([[True, True, True, False, False]])
    logits, second = model(padded, valid_mask=mask, return_aux=True)
    torch.testing.assert_close(first["balance_loss"], second["balance_loss"])
    for a, b in zip(first["router_stats"], second["router_stats"], strict=True):
        assert torch.equal(a["assignment_counts"], b["assignment_counts"])
        torch.testing.assert_close(a["probability_sum"], b["probability_sum"])
    logits[:, :3].square().mean().backward()
    assert model.transformer.h[0].mlp.router.weight.grad.abs().sum().item() > 0


def test_loop_recomputes_routes_and_accumulates_shared_gradients():
    model = model_for("LM4R2")
    visits = []
    handle = model.transformer.h[0].mlp.register_forward_hook(
        lambda module, inputs, output: visits.append(output[1])
    )
    logits, aux = model(torch.tensor([[1, 2, 3]]), return_aux=True)
    logits.square().mean().backward()
    handle.remove()
    assert len(visits) == 2 and len(aux["router_stats"]) == 8
    assert model.transformer.h[0].mlp.router.weight.grad is not None
    assert len({id(block) for block in model.transformer.h}) == 4


def test_residual_orientation_and_permutation_covariance():
    x = torch.tensor([[[1.0], [10.0], [100.0]]])
    matrix = torch.tensor([[[0.1, 0.7, 0.2], [0.3, 0.2, 0.5], [0.6, 0.1, 0.3]]])
    torch.testing.assert_close(residual_mix(x, matrix)[0, :, 0], torch.tensor([63.1, 12.7, 35.2]))
    permutation = torch.tensor([2, 0, 1])
    permuted = residual_mix(x[:, permutation], matrix[:, permutation][:, :, permutation])
    torch.testing.assert_close(permuted, residual_mix(x, matrix)[:, permutation])


def test_sinkhorn_accepted_iteration_order_extreme_finiteness_and_gradients():
    logits = torch.tensor(
        [[20.0, -20.0, 0.0], [18.0, 0.0, -20.0], [-20.0, 20.0, 0.0]], requires_grad=True
    )
    matrix = sinkhorn(logits)
    expected = logits.float().softmax(-1) + 1e-6
    expected = expected / (expected.sum(-2, keepdim=True) + 1e-6)
    for _ in range(19):
        expected = expected / (expected.sum(-1, keepdim=True) + 1e-6)
        expected = expected / (expected.sum(-2, keepdim=True) + 1e-6)
    assert torch.equal(matrix, expected)
    assert torch.isfinite(matrix).all() and (matrix >= 0).all()
    assert (matrix.sum(-2) - 1).abs().max() < 2e-6
    (matrix * torch.arange(9).reshape(3, 3)).sum().backward()
    assert torch.isfinite(logits.grad).all()
    extreme = torch.tensor([[1e20, -1e20], [1e20, -1e20]], requires_grad=True)
    sinkhorn(extreme).square().sum().backward()
    assert torch.isfinite(extreme.grad).all()
    regular = sinkhorn(torch.randn(4, 4))
    assert (regular.sum(-1) - 1).abs().max() < 1e-5
    assert torch.linalg.matrix_norm(regular, ord=2).item() < 1.0001


@pytest.mark.parametrize("identity", [True, False])
def test_mhc_controller_nonsymmetric_inputs_gradients_and_dtype(identity):
    wrapper = HyperConnection(5, 0, identity=identity)
    x = torch.randn(2, 3, 4, 5, requires_grad=True)
    with torch.no_grad():
        wrapper.projection.weight.normal_(0, 0.1)
    output = wrapper(x, lambda value: value.square())
    output.square().mean().backward()
    for name, parameter in wrapper.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name
    with torch.autocast("cpu", dtype=torch.bfloat16):
        pre, post, residual = wrapper.coefficients(x)
    assert pre.dtype == post.dtype == torch.float32
    if residual is not None:
        assert residual.dtype == torch.float32


def test_mhc_dense_equivalent_limit_and_mean_readout():
    wrapper = HyperConnection(5, 0, identity=True)
    base = torch.randn(2, 3, 5)
    x = base.unsqueeze(-2).expand(-1, -1, 4, -1)

    def forced_coefficients(value):
        pre = value.new_tensor([1, 0, 0, 0]).expand(*value.shape[:-2], 4)
        post = torch.ones_like(pre)
        return pre, post, None

    wrapper.coefficients = forced_coefficients
    result = wrapper(x, torch.sin)
    torch.testing.assert_close(result.mean(-2), base + torch.sin(base))
    assert torch.equal(result[..., 0, :], result[..., 3, :])


@pytest.mark.parametrize("arm", ["D8", "M8", "LM4R2", "W8", "IHC8", "HC8"])
def test_optimizer_groups_and_checkpoint_round_trip(arm):
    model = model_for(arm)
    optimizer = optimizer_for(model, 1e-3, 0.1)
    ids = [id(parameter) for group in optimizer.param_groups for parameter in group["params"]]
    assert len(ids) == len(set(ids)) == len(list(model.parameters()))
    decay = {
        id(p): group["weight_decay"] for group in optimizer.param_groups for p in group["params"]
    }
    for name, parameter in model.named_parameters():
        if name.startswith("connections."):
            assert decay[id(parameter)] == (0.1 if ".projection." in name else 0.0)
    x = torch.tensor([[1, 2, 3]])
    logits, aux = model(x, return_aux=True)
    (logits.square().mean() + aux["balance_loss"] * 0.01).backward()
    optimizer.step()
    expected = model(x).detach()
    buffer = io.BytesIO()
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict()), buffer)
    buffer.seek(0)
    state = torch.load(buffer, weights_only=True)
    restored = model_for(arm, seed=999)
    restored.load_state_dict(state["model"])
    reoptimizer = optimizer_for(restored, 1e-3, 0.1)
    reoptimizer.load_state_dict(state["optimizer"])
    assert torch.equal(expected, restored(x))
    optimizer.zero_grad(set_to_none=True)
    reoptimizer.zero_grad(set_to_none=True)
    for instance, optim in [(model, optimizer), (restored, reoptimizer)]:
        output, auxiliary = instance(x, return_aux=True)
        (output.square().mean() + auxiliary["balance_loss"] * 0.01).backward()
        optim.step()
    for name, value in model.state_dict().items():
        assert torch.equal(value, restored.state_dict()[name]), name


def test_flop_ledger_tracks_actual_experts_padding_and_repeats():
    kwargs = dict(executed_tokens=20, projected_positions=6, attention_pairs=100, valid_tokens=12)
    dense, wide, moe, loop = [
        flop_ledger(model_for(arm), **kwargs) for arm in ["D8", "W8", "M8", "LM4R2"]
    ]
    assert wide["forward_ffn_matmul_flops"] == dense["forward_ffn_matmul_flops"] * 4
    assert moe["forward_ffn_matmul_flops"] == dense["forward_ffn_matmul_flops"] * 12 / 20
    assert moe["selected_expert_assignments"] == 8 * 12 * 2
    assert moe == loop
    hc = flop_ledger(model_for("HC8"), **kwargs)
    assert hc["sinkhorn_normalization_passes"] == 39
    assert hc["forward_controller_matmul_flops"] > 0
