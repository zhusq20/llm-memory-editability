"""Trainable nonlinear fact models; importing this module requires PyTorch."""

import torch
from torch import nn


class FactMLP(nn.Module):
    """Separate entity/attribute embeddings followed by a nonlinear MLP."""

    def __init__(
        self,
        n_entities: int,
        n_attributes: int,
        n_answers: int,
        width: int,
        hidden: int,
        layers: int,
    ) -> None:
        super().__init__()
        self.entity_embedding = nn.Embedding(n_entities, width)
        self.attribute_embedding = nn.Embedding(n_attributes, width)
        blocks = []
        for layer in range(layers):
            blocks.extend((nn.Linear(2 * width if layer == 0 else hidden, hidden), nn.GELU()))
        self.hidden = nn.Sequential(*blocks)
        self.answer_head = nn.Linear(hidden, n_answers)

    def forward(self, facts: torch.Tensor) -> torch.Tensor:
        _validate_facts(facts)
        embedded = torch.cat(
            (
                self.entity_embedding(facts[:, 0]),
                self.attribute_embedding(facts[:, 1]),
            ),
            dim=-1,
        )
        return self.answer_head(self.hidden(embedded))


class FactTransformer(nn.Module):
    """Causal [ENTITY, ATTRIBUTE, QUERY] transformer with an answer classification head."""

    def __init__(
        self,
        n_entities: int,
        n_attributes: int,
        n_answers: int,
        width: int,
        hidden: int,
        layers: int,
        heads: int,
    ) -> None:
        super().__init__()
        self.n_entities = n_entities
        self.query_id = n_entities + n_attributes
        self.token_embedding = nn.Embedding(self.query_id + 1, width)
        self.position_embedding = nn.Embedding(3, width)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=hidden,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=layers,
            norm=nn.LayerNorm(width),
            enable_nested_tensor=False,
        )
        # TransformerEncoder clones its prototype; initialize each layer independently.
        for layer in self.encoder.layers:
            nn.init.xavier_uniform_(layer.self_attn.in_proj_weight)
            for module in layer.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
        self.answer_head = nn.Linear(width, n_answers)
        self.register_buffer("causal_mask", torch.triu(torch.ones(3, 3, dtype=torch.bool), 1))

    def forward(self, facts: torch.Tensor) -> torch.Tensor:
        _validate_facts(facts)
        tokens = torch.stack(
            (
                facts[:, 0],
                facts[:, 1] + self.n_entities,
                torch.full_like(facts[:, 0], self.query_id),
            ),
            dim=1,
        )
        embedded = self.token_embedding(tokens)
        embedded = embedded + self.position_embedding(torch.arange(3, device=facts.device))
        encoded = self.encoder(embedded, mask=self.causal_mask)
        return self.answer_head(encoded[:, -1])


def _validate_facts(facts: torch.Tensor) -> None:
    if facts.ndim != 2 or facts.shape[1] != 2 or facts.dtype != torch.long:
        raise ValueError("facts must be an int64 tensor with shape (n_facts, 2)")


def build_model(
    model_type: str,
    n_entities: int,
    n_attributes: int,
    n_answers: int,
    width: int = 32,
    hidden: int = 64,
    layers: int = 2,
    heads: int = 4,
) -> nn.Module:
    """Construct an MLP or a causal transformer; initialization uses the torch RNG."""
    for name, value in {
        "n_entities": n_entities,
        "n_attributes": n_attributes,
        "n_answers": n_answers,
        "width": width,
        "hidden": hidden,
        "layers": layers,
        "heads": heads,
    }.items():
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if model_type == "mlp":
        return FactMLP(n_entities, n_attributes, n_answers, width, hidden, layers)
    if model_type == "transformer":
        if width % heads:
            raise ValueError("transformer width must be divisible by heads")
        return FactTransformer(n_entities, n_attributes, n_answers, width, hidden, layers, heads)
    raise ValueError("model_type must be 'mlp' or 'transformer'")


def edit_parameters(model: nn.Module, scope: str = "ffn") -> list[nn.Parameter]:
    """Freeze outside the chosen edit scope, returning the exact optimizer parameters."""
    if scope not in {"ffn", "all"}:
        raise ValueError("scope must be 'ffn' or 'all'")
    if scope == "all":
        selected = list(model.parameters())
    elif isinstance(model, FactMLP):
        selected = list(model.hidden.parameters())
    elif isinstance(model, FactTransformer):
        selected = [
            parameter
            for layer in model.encoder.layers
            for linear in (layer.linear1, layer.linear2)
            for parameter in linear.parameters()
        ]
    else:
        raise ValueError("ffn scope requires a FactMLP or FactTransformer")
    selected_ids = {id(parameter) for parameter in selected}
    for parameter in model.parameters():
        parameter.requires_grad_(id(parameter) in selected_ids)
        parameter.grad = None
    return selected
