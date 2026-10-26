"""Prune discarded training-write computation without changing TTCD updates.

The original frozen reader supplies the complete implementation. A checked AST
transformation returns immediately after round-zero capture and corrects its
compute estimates. Native queries and inference writes call the original
function unchanged. Installation is explicit and reversible within one process.
"""

from __future__ import annotations

import ast
import contextlib
import functools
import hashlib
import inspect
import textwrap
from pathlib import Path

from .ttcd_editing import TTCDReader

ORIGINAL_FILE_SHA256 = "391790498d6412081baa06d73d2105c1a8a2c78c0cca0256a11d201e2e72d28f"


def _capture_prefix(original):
    source_path = Path(inspect.getsourcefile(original))
    digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
    if digest != ORIGINAL_FILE_SHA256:
        raise ValueError("Training speedup requires the exact audited TTCDReader source bytes")
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise ValueError("Expected exactly the original forward function")
    forward = tree.body[0]
    loops = [
        n for n in forward.body if isinstance(n, ast.For) and ast.unparse(n.target) == "round_index"
    ]
    if len(loops) != 1:
        raise ValueError("The original loop-round structure changed")
    rounds = loops[0]
    if (
        ast.unparse(rounds.iter) != "range(model.repeats)"
        or len(rounds.body) != 2
        or not isinstance(rounds.body[0], ast.For)
        or ast.unparse(rounds.body[0].target) != "(layer, block)"
        or not isinstance(rounds.body[1], ast.If)
        or "_discoloop_boundary" not in ast.unparse(rounds.body[1])
    ):
        raise ValueError("Expected a complete physical-layer pass before the Disco boundary")
    captures = [
        n
        for n in ast.walk(rounds.body[0])
        if isinstance(n, ast.Assign)
        and any(ast.unparse(target) == "capture['features'][layer]" for target in n.targets)
    ]
    if len(captures) != 1 or "outer_grad" not in ast.unparse(captures[0]):
        raise ValueError("The original differentiable first-round capture changed")
    # All physical-layer features exist here; no later value is consumed by
    # session_write. Returning before the boundary also removes the decoder.
    # The discarded decoder normally creates a cached autocast view of the
    # tied embedding parameter. Preserve that cast node with a zero-row linear
    # operation, so BF16 gradient accumulation uses the same shared cast graph.
    cache = ast.parse(
        "if torch.is_autocast_enabled(tokens.device.type):\n"
        "    F.linear(x.new_empty((0, cfg.n_embd)), t.wte.weight)\n"
    ).body[0]
    rounds.body.insert(1, cache)
    rounds.body.insert(2, ast.Return(value=ast.Constant(None)))
    repeated_cost = boundary_cost = decoder_cost = 0
    for node in ast.walk(forward):
        if isinstance(node, ast.AugAssign) and ast.unparse(node.target) == "flops":
            if ast.unparse(node.value) == "model.repeats * cost":
                node.value = ast.Name(id="cost", ctx=ast.Load())
                repeated_cost += 1
        if (
            isinstance(node, ast.If)
            and ast.unparse(node.test) == "getattr(model, 'discoloop', False)"
        ):
            node.test = ast.Constant(False)
            boundary_cost += 1
        if isinstance(node, ast.Assign) and any(ast.unparse(t) == "flops" for t in node.targets):
            if ast.unparse(node.value) == "batch * (flops + 2 * scored * width * cfg.vocab_size)":
                node.value = ast.BinOp(
                    left=ast.Name(id="batch", ctx=ast.Load()),
                    op=ast.Mult(),
                    right=ast.Name(id="flops", ctx=ast.Load()),
                )
                decoder_cost += 1
    if (repeated_cost, boundary_cost, decoder_cost) != (1, 1, 1):
        raise ValueError("The original FLOP accounting structure changed")
    forward.name, forward.decorator_list = "_capture_only_forward", []
    ast.fix_missing_locations(tree)
    namespace = dict(original.__globals__)
    exec(compile(tree, str(source_path) + ":capture-prefix", "exec"), namespace)
    return namespace[forward.name], {
        "original_file_sha256": digest,
        "original_forward_sha256": hashlib.sha256(
            textwrap.dedent(inspect.getsource(original)).encode()
        ).hexdigest(),
        "transformation": (
            "Return after round-zero capture; zero-row linear preserves tied-weight autocast cache"
        ),
        "compute_policy": (
            "Original four keys; pruned forward estimate and conventional2x backward proxy"
        ),
        "pruned_calls": 0,
    }


def install():
    """Install once and return an idempotent undo callable with provenance."""
    original = TTCDReader.forward
    if getattr(original, "_ttcd_train_speedup", False):
        raise RuntimeError("Training speedup is already installed in this process")
    prefix, provenance = _capture_prefix(original)

    @functools.wraps(original)
    def forward(self, tokens, positions=None, *, fast=None, window=None, capture=None):
        if capture is not None and capture.get("outer_grad") is True:
            result = prefix(self, tokens, positions, fast=fast, window=window, capture=capture)
            provenance["pruned_calls"] += 1
            return result
        return original(self, tokens, positions, fast=fast, window=window, capture=capture)

    forward._ttcd_train_speedup = True
    TTCDReader.forward = forward
    undone = False

    def undo():
        nonlocal undone
        if not undone:
            if TTCDReader.forward is not forward:
                raise RuntimeError(
                    "Another runtime modification replaced the reader; refuse silent undo"
                )
            TTCDReader.forward = original
            undone = True

    undo.provenance = provenance
    return undo


@contextlib.contextmanager
def enabled():
    """Apply the optimization only within the explicit numerical/run scope."""
    undo = install()
    try:
        yield undo.provenance
    finally:
        undo()
