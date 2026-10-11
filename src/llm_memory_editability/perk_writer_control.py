"""Preparation for an outer-supervision comparison with a fixed PERK reader.

This is an extension, not the paper reproduction. The supplied author module
owns the complete language model, weighted CLM, learned inner rates, and inner
unroll. Only the weighting network and rate controls enter the outer optimizer;
the reader and LoRA initial values remain fixed. No dataset or QA labels are
constructed here, and this module does not launch training.
"""

from __future__ import annotations

import hashlib
import importlib
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any, Literal

import torch
from torch import Tensor, nn

OuterCondition = Literal["atomic", "compositional"]
_FEATURE_FIELDS = ("input_ids", "attention_mask", "labels")


def _supervised_features(features: Mapping[str, Tensor], *, name: str) -> dict[str, Tensor]:
    """Validate supplied causal-LM labels without generating any supervision."""
    if set(features) != set(_FEATURE_FIELDS):
        raise ValueError(f"{name} must supply exactly input_ids, attention_mask, and labels")
    result = {}
    for field in _FEATURE_FIELDS:
        tensor = features[field]
        if not isinstance(tensor, Tensor) or tensor.ndim != 2:
            raise ValueError(f"{name}.{field} must be a rank-2 tensor")
        result[field] = tensor.detach().clone()
    ids, mask, labels = (result[field] for field in _FEATURE_FIELDS)
    if ids.shape != mask.shape or ids.shape != labels.shape or 0 in ids.shape or ids.shape[1] < 2:
        raise ValueError(f"{name} tensors must have identical nonempty [rows, tokens>=2] shapes")
    if ids.dtype != torch.long or labels.dtype != torch.long:
        raise ValueError(f"{name} input_ids and labels must be int64")
    if mask.dtype not in (torch.long, torch.bool):
        raise ValueError(f"{name} attention_mask must be int64 or bool")
    if any(tensor.device != ids.device for tensor in result.values()):
        raise ValueError(f"{name} tensors must be on one device")
    if (ids < 0).any() or not ((mask == 0) | (mask == 1)).all():
        raise ValueError(f"{name} requires nonnegative IDs and a binary attention mask")
    active = labels != -100
    if (active & (mask == 0)).any() or not torch.equal(labels[active], ids[active]):
        raise ValueError(f"{name} labels must match input IDs, with padding labeled -100")
    if not (labels[:, 1:] != -100).any(dim=1).all():
        raise ValueError(f"Each {name} row must supply a next-token supervision target")
    return result


def _features_digest(features: Mapping[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for field in _FEATURE_FIELDS:
        tensor = features[field].detach().cpu().contiguous()
        digest.update(f"{field}:{tensor.dtype}:{tuple(tensor.shape)}\n".encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _exposure(features: Mapping[str, Tensor]) -> dict[str, int]:
    return {
        "rows": features["input_ids"].shape[0],
        "input_tokens": int(features["attention_mask"].sum()),
        # Causal LM predicts position i from the preceding position. Position 0
        # has no target prediction and is excluded even if its label is present.
        "supervised_tokens": int((features["labels"][:, 1:] != -100).sum()),
        "tensor_tokens_including_padding": features["input_ids"].numel(),
    }


class PairedWriterBatch:
    """One material stream with two explicitly supplied outer supervision sets.

    Construction copies the tensors. Accessors return fresh copies, so neither
    an author unroll nor a caller can change the paired stream by tensor aliasing.
    The condition names describe supplied supervision, not a semantic validation
    of its labels. A scientific protocol must separately review those labels and
    the train/evaluation split.
    """

    def __init__(
        self,
        material: Mapping[str, Tensor],
        atomic_outer: Mapping[str, Tensor],
        compositional_outer: Mapping[str, Tensor],
        *,
        material_id: str,
    ) -> None:
        if not isinstance(material_id, str) or not material_id.strip():
            raise ValueError("An explicit material_id is required")
        self.material_id = material_id
        self._material = _supervised_features(material, name="material")
        self._outer = {
            "atomic": _supervised_features(atomic_outer, name="atomic_outer"),
            "compositional": _supervised_features(compositional_outer, name="compositional_outer"),
        }

    @classmethod
    def from_author_batches(
        cls,
        atomic_batch: Mapping[str, Any],
        compositional_batch: Mapping[str, Any],
        *,
        material_id: str,
    ) -> PairedWriterBatch:
        """Accept author-collated QA tensors only if all write tensors match."""
        material = {}
        for field in _FEATURE_FIELDS:
            key = f"train_{field}"
            if key not in atomic_batch or key not in compositional_batch:
                raise ValueError(f"Both paired author batches must supply {key}")
            first, second = atomic_batch[key], compositional_batch[key]
            if (
                not isinstance(first, Tensor)
                or not isinstance(second, Tensor)
                or first.dtype != second.dtype
                or first.device != second.device
                or not torch.equal(first, second)
            ):
                raise ValueError(f"Paired write input differs: {key}")
            material[field] = first
        return cls(
            material,
            {field: atomic_batch[field] for field in _FEATURE_FIELDS},
            {field: compositional_batch[field] for field in _FEATURE_FIELDS},
            material_id=material_id,
        )

    def material_features(self) -> dict[str, Tensor]:
        return {field: tensor.clone() for field, tensor in self._material.items()}

    def outer_features(self, condition: OuterCondition) -> dict[str, Tensor]:
        if condition not in self._outer:
            raise ValueError(f"Unknown outer supervision condition: {condition}")
        return {field: tensor.clone() for field, tensor in self._outer[condition].items()}

    def exposure(self, condition: OuterCondition, *, n_inner_iter: int) -> dict[str, Any]:
        if isinstance(n_inner_iter, bool) or not isinstance(n_inner_iter, int) or n_inner_iter < 1:
            raise ValueError("n_inner_iter must be a positive integer")
        inner = _exposure(self._material)
        outer = _exposure(self.outer_features(condition))
        return {
            "material_id": self.material_id,
            "material_sha256": _features_digest(self._material),
            "outer_sha256": _features_digest(self._outer[condition]),
            "outer_condition": condition,
            "inner_subsequence_presentations": inner["rows"] * n_inner_iter,
            "inner_input_tokens": inner["input_tokens"] * n_inner_iter,
            "inner_supervised_tokens": inner["supervised_tokens"] * n_inner_iter,
            "inner_tensor_tokens_including_padding": (
                inner["tensor_tokens_including_padding"] * n_inner_iter
            ),
            "outer_queries": outer["rows"],
            "outer_input_tokens": outer["input_tokens"],
            "outer_supervised_tokens": outer["supervised_tokens"],
            "outer_tensor_tokens_including_padding": outer["tensor_tokens_including_padding"],
        }


class AdaptedPERKReadView:
    """Ephemeral LoRA state for read-only queries within one material context."""

    def __init__(self, fast_model: nn.Module, inner_loss: Tensor) -> None:
        self._fast_model = fast_model
        self._active = True
        self.inner_loss = inner_loss.detach()

    def _ensure_active(self) -> None:
        if not self._active:
            raise RuntimeError("The material context has ended; its fast state was discarded")

    def adapter_values(self) -> dict[str, Tensor]:
        self._ensure_active()
        return {
            name: parameter.detach().clone()
            for name, parameter in self._fast_model.named_parameters()
            if "lora" in name
        }

    def supervised_loss(self, features: Mapping[str, Tensor]) -> Tensor:
        """Outer loss through the author unroll, without an optimizer update."""
        self._ensure_active()
        checked = _supervised_features(features, name="outer")
        return self._fast_model(checked)["loss"]

    def read(self, features: Mapping[str, Tensor]) -> dict[str, Tensor]:
        """Evaluate a new query without carrying attention/KV state or gradients."""
        self._ensure_active()
        required = {"input_ids", "attention_mask"}
        if not required <= set(features) or set(features) - required - {"labels"}:
            raise ValueError("Query must supply input_ids/attention_mask and optional labels only")
        if "labels" in features:
            checked = _supervised_features(features, name="query")
        else:
            # Validate inputs directly without constructing any query labels.
            ids = features["input_ids"]
            mask = features["attention_mask"]
            if not isinstance(ids, Tensor) or not isinstance(mask, Tensor):
                raise ValueError("Query input_ids and attention_mask must be tensors")
            if ids.ndim != 2 or mask.shape != ids.shape or 0 in ids.shape:
                raise ValueError("Query tensors must have identical nonempty [rows, tokens] shapes")
            if (
                ids.dtype != torch.long
                or mask.dtype not in (torch.long, torch.bool)
                or mask.device != ids.device
            ):
                raise ValueError("Query requires int64 IDs and tensors on one device")
            if (ids < 0).any() or not ((mask == 0) | (mask == 1)).all():
                raise ValueError("Query requires nonnegative IDs and a binary attention mask")
            if not mask.any(dim=1).all():
                raise ValueError("Each query row must contain an attended token")
            checked = {"input_ids": ids.detach().clone(), "attention_mask": mask.detach().clone()}
        modes = [(module, module.training) for module in self._fast_model.modules()]
        try:
            self._fast_model.eval()
            with torch.no_grad():
                outputs = self._fast_model({**checked, "use_cache": False})
                return {
                    field: outputs[field].detach().clone()
                    for field in ("logits", "loss")
                    if outputs.get(field) is not None
                }
        finally:
            for module, training in modes:
                module.training = training

    def generate(
        self, input_ids: Tensor, attention_mask: Tensor, *, eos_token_id: int, max_new_tokens: int
    ) -> list[list[int]]:
        """Greedy batched generation through the actual fast forward, without KV.

        Calling a bound ``generate`` method on Higher's monkey-patched child can
        bypass its functional parameters. Every step here explicitly calls the
        fast model. Left padding uses the same position IDs as HF generation.
        Returned IDs are the actual sampled IDs, not text re-tokenization.
        """
        self._ensure_active()
        if max_new_tokens < 1 or input_ids.ndim != 2 or input_ids.shape != attention_mask.shape:
            raise ValueError("Generation needs aligned rank-2 inputs and a positive budget")
        if (
            input_ids.dtype != torch.long
            or not ((attention_mask == 0) | (attention_mask == 1)).all()
        ):
            raise ValueError("Generation requires int64 IDs and a binary attention mask")
        if input_ids.device != attention_mask.device or not attention_mask.any(dim=1).all():
            raise ValueError("Generation inputs must share a device and contain attended tokens")
        # Right padding would read the pad position as the next-token query.
        if (
            not (attention_mask[:, -1] == 1).all()
            or (attention_mask[:, 1:] < attention_mask[:, :-1]).any()
        ):
            raise ValueError("Generation requires left padding")
        if (
            input_ids.shape[1] + max_new_tokens
            > self._fast_model.lm_model.config.max_position_embeddings
        ):
            raise ValueError("Generation exceeds the actual reader context window")
        modes = [(module, module.training) for module in self._fast_model.modules()]
        ids, mask = input_ids.clone(), attention_mask.clone()
        done = torch.zeros(ids.shape[0], dtype=torch.bool, device=ids.device)
        outputs: list[list[int]] = [[] for _ in range(ids.shape[0])]
        try:
            self._fast_model.eval()
            with torch.no_grad():
                for _ in range(max_new_tokens):
                    positions = mask.long().cumsum(-1) - 1
                    positions.masked_fill_(mask == 0, 1)
                    logits = self._fast_model(
                        {
                            "input_ids": ids,
                            "attention_mask": mask,
                            "position_ids": positions,
                            "use_cache": False,
                        }
                    )["logits"]
                    next_ids = logits[:, -1].argmax(-1)
                    for row, token in enumerate(next_ids.tolist()):
                        if not done[row]:
                            outputs[row].append(token)
                    active = ~done
                    next_ids = torch.where(active, next_ids, eos_token_id)
                    ids = torch.cat((ids, next_ids[:, None]), dim=1)
                    mask = torch.cat((mask, active[:, None].to(mask.dtype)), dim=1)
                    done |= next_ids == eos_token_id
                    if done.all():
                        break
                    del logits
            return outputs
        finally:
            for module, training in modes:
                module.training = training


class ControlledPERKWriter(nn.Module):
    """Reuse an author module while outer-learning only its write controls.

    LoRA remains differentiable for the author's inner AdamW. Excluding its
    initial values from the outer optimizer, rather than disabling gradients,
    fixes the reader and fast-state initialization without breaking adaptation.
    Call ``zero_grad`` on this wrapper between outer steps: it also clears the
    excluded LoRA gradients that higher-order differentiation can produce.
    """

    def __init__(self, author_module: nn.Module) -> None:
        super().__init__()
        self.author_module = author_module
        config = author_module.hparams
        for setting in ("peft_lora", "weighted_loss", "dyna_lr"):
            if not config.get(setting):
                raise ValueError(f"The controlled PERK writer requires {setting}=True")
        if config.get("packing"):
            raise ValueError(
                "The controlled GPT-2 writer uses independent unpacked material chunks"
            )
        weighting = author_module.model.weighting_model
        if weighting is None or not list(author_module.inner_schedular.parameters()):
            raise ValueError("Author weighting network and learned inner rates are required")
        reader = list(author_module.model.lm_model.named_parameters())
        if not any("lora" in name for name, _ in reader):
            raise ValueError("The author reader must contain LoRA parameters")
        for name, parameter in reader:
            if parameter.requires_grad != ("lora" in name):
                raise ValueError(f"Base reader must be frozen and LoRA differentiable: {name}")
        if not all(parameter.requires_grad for _, parameter in self.outer_named_parameters()):
            raise ValueError("All write controls must be differentiable")
        self._adapting = False

    def outer_named_parameters(self) -> Iterator[tuple[str, nn.Parameter]]:
        for name, parameter in self.author_module.model.weighting_model.named_parameters():
            yield f"weighting_model.{name}", parameter
        for name, parameter in self.author_module.inner_schedular.named_parameters():
            yield f"inner_schedular.{name}", parameter

    def outer_optimizer(self) -> torch.optim.AdamW:
        """Use author outer AdamW settings, restricting its parameter membership."""
        config = self.author_module.hparams
        decayed, no_decay, rates = [], [], []
        for name, parameter in self.outer_named_parameters():
            if name.startswith("inner_schedular."):
                rates.append(parameter)
            elif any(excluded in name for excluded in ("bias", "LayerNorm.weight")):
                no_decay.append(parameter)
            else:
                decayed.append(parameter)
        return torch.optim.AdamW(
            [
                {"params": decayed, "weight_decay": config.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
                {"params": rates, "weight_decay": 0.0},
            ],
            lr=config.learning_rate,
            eps=config.adam_epsilon,
            betas=(0.9, 0.95),
        )

    @contextmanager
    def adapt_material(
        self, material: Mapping[str, Tensor], *, track_higher_grads: bool = False
    ) -> Iterator[AdaptedPERKReadView]:
        """Reset inner weights/buffers per material; only yielded fast state persists.

        The inner optimizer and its AdamW moments are created anew. The complete
        author forward and configured inner unroll are retained. No query or QA
        supervision is accepted by this write operation.
        """
        if self._adapting:
            raise RuntimeError("A writer cannot open overlapping material contexts")
        checked = _supervised_features(material, name="material")
        higher = importlib.import_module("higher")
        author = self.author_module
        inner_optimizer = author.config_inner_optimizer()
        expected = {
            id(parameter) for name, parameter in author.model.named_parameters() if "lora" in name
        }
        actual = {
            id(parameter) for group in inner_optimizer.param_groups for parameter in group["params"]
        }
        if actual != expected:
            raise ValueError("Author inner optimizer must update exactly the differentiable LoRA")
        modes = [(module, module.training) for module in author.model.modules()]
        self._adapting = True
        view = None
        try:
            # Inference still needs inner autograd even if its caller uses
            # no_grad. The read view itself always disables query gradients.
            with torch.enable_grad():
                author.model.train()
                with higher.innerloop_ctx(
                    author.model,
                    inner_optimizer,
                    copy_initial_weights=False,
                    track_higher_grads=track_higher_grads,
                    accumulation_steps=1,
                ) as (fast_model, diffopt):
                    inner_loss = author.inner_funct([checked], fast_model, diffopt)
                    view = AdaptedPERKReadView(fast_model, inner_loss)
                    yield view
        finally:
            if view is not None:
                view._active = False
            for module, training in modes:
                module.training = training
            self._adapting = False

    def meta_step(self, batch: PairedWriterBatch, condition: OuterCondition) -> dict[str, Any]:
        """Compute a differentiable outer loss and separate write/QA exposure."""
        outer = batch.outer_features(condition)
        with self.adapt_material(batch.material_features(), track_higher_grads=True) as adapted:
            loss = adapted.supervised_loss(outer)
            inner_loss = adapted.inner_loss
        return {
            "loss": loss,
            "inner_loss": inner_loss,
            "outer_loss": loss.detach(),
            "exposure": batch.exposure(
                condition, n_inner_iter=self.author_module.hparams.n_inner_iter
            ),
        }


def assert_paired_writer_initial_state(
    first: ControlledPERKWriter, second: ControlledPERKWriter
) -> None:
    """Check a pair before outer learning, including control initialization."""
    fields = (
        "peft_lora",
        "weighted_loss",
        "dyna_lr",
        "packing",
        "n_inner_iter",
        "inner_lr",
        "inner_funct",
        "unroll_start",
        "first_order",
        "weight_decay",
        "learning_rate",
        "adam_epsilon",
        "attn_implementation",
        "weight_normalize",
        "weight_non_linearity",
        "lora_rank",
        "lora_alpha",
    )
    for field in fields:
        if first.author_module.hparams.get(field) != second.author_module.hparams.get(field):
            raise ValueError(f"Paired writer configuration differs: {field}")
    for config_scope in ("saved", "computing"):
        configs = [
            (
                writer.author_module.model.config
                if config_scope == "saved"
                else writer.author_module.model.lm_model.config
            ).to_dict()
            for writer in (first, second)
        ]
        for config in configs:
            # These two serialized metadata fields do not change the reader's
            # computation. All mathematical/architectural fields must match.
            config.pop("_name_or_path", None)
            config.pop("transformers_version", None)
        if configs[0] != configs[1]:
            raise ValueError(f"Paired reader architecture/configuration differs: {config_scope}")
    modules = [dict(writer.author_module.model.named_modules()) for writer in (first, second)]
    if modules[0].keys() != modules[1].keys():
        raise ValueError("Paired reader/control module structure differs")
    for name, module in modules[0].items():
        other = modules[1][name]
        if type(module) is not type(other):
            raise ValueError(f"Paired reader/control module type differs: {name}")
        if isinstance(module, nn.Dropout) and module.p != other.p:
            raise ValueError(f"Paired reader/control dropout differs: {name}")
        if isinstance(module, nn.LayerNorm) and module.eps != other.eps:
            raise ValueError(f"Paired reader LayerNorm.eps differs: {name}")
        # GPT-2 caches these settings on its attention modules. Comparing config
        # alone misses changes to the actual forward calculation after loading.
        # LoRA also keeps its actual scale and adapter selection outside tensors.
        for field in (
            "scale_attn_weights",
            "scale_attn_by_inverse_layer_idx",
            "reorder_and_upcast_attn",
            "layer_idx",
            "scaling",
            "fan_in_fan_out",
            "active_adapters",
            "merged_adapters",
            "disable_adapters",
        ):
            if hasattr(module, field):
                value, other_value = getattr(module, field), getattr(other, field)
                # PeftModel exposes disable_adapters as a method; individual
                # LoRA layers expose the actual boolean under that same name.
                if not callable(value) and value != other_value:
                    raise ValueError(f"Paired reader computation differs: {name}.{field}")
    states = first.state_dict(), second.state_dict()
    if states[0].keys() != states[1].keys():
        raise ValueError("Paired writer state fields differ")
    for name, tensor in states[0].items():
        other = states[1][name]
        if tensor.dtype != other.dtype or not torch.equal(tensor.cpu(), other.cpu()):
            raise ValueError(f"Paired writer initial state differs: {name}")
