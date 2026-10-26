"""Runtime I/O amendment for frozen TTCD evaluation, with reversible installation.

The scientific readers, writer, query batches and numerical operations are
unchanged. Only registered frozen backbone digests are cached. Fast states are
always hashed in full and remain independently loadable in the original format.
The evaluation modules' original ``tensor_digest`` aliases deliberately remain
uncached, including their phase and read-only-state checks.
"""

from __future__ import annotations

import threading
from pathlib import Path

import torch

from . import ttcd_continual_eval, ttcd_double_new_eval, ttcd_editing, ttcd_experiment

_ACTIVE = None


def _signature(tensors):
    """Identify live tensor storage and its shared mutation version counter.

    ``state_dict()`` creates new detached Tensor objects, so Python object ids
    alone are insufficient. Holding the registered views keeps these storage
    identities alive. Unsupported/inference tensors safely bypass the cache.
    """
    result = []
    for name, value in sorted(tensors.items(), key=lambda item: str(item[0])):
        if not isinstance(value, torch.Tensor) or value.layout != torch.strided:
            return None
        try:
            version = value._version
            storage = value.untyped_storage()._cdata
        except RuntimeError:
            return None
        result.append(
            (
                type(name).__name__,
                str(name),
                storage,
                value.data_ptr(),
                value.storage_offset(),
                tuple(value.shape),
                tuple(value.stride()),
                str(value.dtype),
                str(value.device),
                str(value.layout),
                value.requires_grad,
                version,
            )
        )
    return tuple(result)


class _Installation:
    def __init__(self):
        self.original_digest = ttcd_editing.tensor_digest
        self.original_save = ttcd_editing.save_fast
        self.original_load = ttcd_editing.load_fast
        self.entries = {}
        self.patches = []
        self.lock = threading.RLock()
        self.closed = False
        self.stats = {
            "schema": "ttcd-eval-io-amendment-v1",
            "scientific_training_updates": 0,
            "fast_state_schema": "ttcd-fast-v1",
            "query_sequence_batch_precision_unchanged": True,
            "fast_digests_always_complete": True,
            "evaluation_digest_aliases_unchanged": True,
            "registered_backbones": 0,
            "backbone_cache_hits": 0,
            "backbone_cache_misses": 0,
            "backbone_signature_invalidations": 0,
            "uncached_digest_calls": 0,
            "save_fast_calls": 0,
            "load_fast_calls": 0,
            "cpu_fast_snapshots": 0,
            "avoided_backbone_digest_bytes": 0,
            "avoided_duplicate_fast_copy_bytes": 0,
            "complete_verifications": 0,
            "verification_passed": None,
            "restored": False,
        }

    def _register(self, reader):
        model = reader.backbone
        # Training callers retain the original complete digest on every call.
        if model.training or any(p.requires_grad for p in model.parameters()):
            return
        snapshot = model.state_dict()
        signature = _signature(snapshot)
        if signature is None:
            return
        with self.lock:
            entry = self.entries.get(id(model))
            if entry is not None and entry["signature"] == signature:
                return
            observed = self.original_digest(snapshot)
            self.stats["backbone_cache_misses"] += 1
            if entry is None:
                entry = {"model": model, "initial_digest": observed}
                self.entries[id(model)] = entry
                self.stats["registered_backbones"] += 1
            else:
                self.stats["backbone_signature_invalidations"] += 1
            entry.update(
                signature=signature,
                snapshot=snapshot,
                digest=observed,
                bytes=sum(v.numel() * v.element_size() for v in snapshot.values()),
            )

    def digest(self, tensors):
        signature = _signature(tensors)
        if signature is not None:
            with self.lock:
                for entry in self.entries.values():
                    if entry["signature"] == signature:
                        # Do not trust a previously frozen model after it is
                        # switched to training or any live storage is replaced.
                        model = entry["model"]
                        if (
                            not model.training
                            and not any(p.requires_grad for p in model.parameters())
                            and _signature(model.state_dict()) == signature
                        ):
                            self.stats["backbone_cache_hits"] += 1
                            self.stats["avoided_backbone_digest_bytes"] += entry["bytes"]
                            return entry["digest"]
        self.stats["uncached_digest_calls"] += 1
        return self.original_digest(tensors)

    def save_fast(self, path, reader, fast, provenance):
        reader.validate_fast(fast)
        self._register(reader)
        # A single CPU snapshot supplies both the complete digest and payload.
        # Its original dtype/layout and every value are retained.
        snapshot = {k: v.detach().cpu() for k, v in fast.items()}
        payload = {
            "schema": "ttcd-fast-v1",
            "backbone_digest": self.digest(reader.backbone.state_dict()),
            "fast_digest": self.digest(snapshot),
            "fast": snapshot,
            "provenance": provenance,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)
        self.stats["save_fast_calls"] += 1
        self.stats["cpu_fast_snapshots"] += 1
        self.stats["avoided_duplicate_fast_copy_bytes"] += sum(
            v.numel() * v.element_size() for v in fast.values() if v.device.type != "cpu"
        )
        return payload["fast_digest"]

    def load_fast(self, path, reader):
        self._register(reader)
        # The original loader still checks schema, complete fast hash, frozen
        # backbone binding, shapes/devices, finiteness and returns provenance.
        result = self.original_load(path, reader)
        self.stats["load_fast_calls"] += 1
        return result

    def verify(self):
        """Force full bytes, independent of version counters or cached hashes."""
        try:
            with self.lock:
                for entry in self.entries.values():
                    observed = self.original_digest(entry["model"].state_dict())
                    self.stats["complete_verifications"] += 1
                    if observed != entry["initial_digest"]:
                        raise AssertionError("Runtime speedup detected a changed frozen backbone")
        except BaseException:
            self.stats["verification_passed"] = False
            raise
        self.stats["verification_passed"] = True
        return self.stats

    def patch(self, module, name, expected, replacement):
        # Rebind only known aliases to the original function. Never overwrite
        # an independently installed wrapper or an evaluation digest alias.
        if getattr(module, name, None) is expected:
            self.patches.append((module, name, expected, replacement))
            setattr(module, name, replacement)

    def restore(self, *, verify=True):
        global _ACTIVE
        if self.closed:
            return self.stats
        try:
            if verify:
                self.verify()
        finally:
            for module, name, original, installed in reversed(self.patches):
                if getattr(module, name, None) == installed:
                    setattr(module, name, original)
            self.closed = True
            self.stats["restored"] = True
            if _ACTIVE is self:
                _ACTIVE = None
        return self.stats


def install():
    """Return ``(undo, stats)``; ``undo()`` fully verifies and restores aliases.

    Install once per independent stage process, before calling the frozen CLI.
    Keep the returned stats as runtime-amendment provenance, separate from the
    original scientific result. On an existing failure, ``undo(verify=False)``
    restores aliases without masking that failure; it does not certify audit.
    """
    global _ACTIVE
    if _ACTIVE is not None:
        raise RuntimeError("TTCD evaluation speedup is already installed")
    state = _Installation()
    state.patch(ttcd_editing, "tensor_digest", state.original_digest, state.digest)
    for module in (ttcd_editing, ttcd_continual_eval, ttcd_double_new_eval, ttcd_experiment):
        state.patch(module, "save_fast", state.original_save, state.save_fast)
        state.patch(module, "load_fast", state.original_load, state.load_fast)
    _ACTIVE = state
    return state.restore, state.stats


def verify():
    """Explicit complete phase-end verification before writing a receipt."""
    if _ACTIVE is None:
        raise RuntimeError("TTCD evaluation speedup is not installed")
    return _ACTIVE.verify()
