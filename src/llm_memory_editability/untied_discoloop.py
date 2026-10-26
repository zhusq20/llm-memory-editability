"""DiscoLoop's vocabulary handoff on a standard Transformer with independent layers.

The paper's architecture part separates two conditions for composing facts that
were only learned atomically: the second hop needs its fact stored at the depth
where the bridge appears (parameter sharing gives this), and the bridge must be
handed over in the entity form the store reads (passing through the vocabulary
gives this). DiscoLoop 4x2 has both. This module keeps the vocabulary handoff
and removes sharing: eight independently parameterized blocks, with the same
soft decode/re-encode channel added once after block ``split`` (default 4).
"""

from __future__ import annotations

from torch.nn import functional as F

from .self_decode_alignment import _readout
from .supervised_alignment import SupervisedAlignmentGPT


class UntiedDiscoLoopGPT(SupervisedAlignmentGPT):
    """One pass over independent blocks with a DiscoLoop channel after ``split`` blocks.

    With weights of blocks ``split:`` copied from blocks ``:split``, the forward is
    identical to DiscoLoop with ``split`` shared blocks run twice. The returned
    first state is the residual after block ``split``, before the channel.
    """

    def __init__(
        self,
        config,
        repeats=1,
        first_round_kv=False,
        discoloop=True,
        discoloop_alpha=1.0,
        discoloop_temperature=1.0,
        discoloop_eps=1e-6,
        discoloop_split=4,
    ):
        if repeats != 1 or first_round_kv or not discoloop:
            raise ValueError("Untied DiscoLoop is one pass, no KV, with the channel enabled")
        super().__init__(
            config,
            repeats=1,
            first_round_kv=False,
            discoloop=True,
            discoloop_alpha=discoloop_alpha,
            discoloop_temperature=discoloop_temperature,
            discoloop_eps=discoloop_eps,
        )
        if not isinstance(discoloop_split, int) or not 0 < discoloop_split < config.n_layer:
            raise ValueError("The channel must sit strictly between blocks")
        self.discoloop_split = discoloop_split

    def _blocks(self, x, blocks):
        cfg = self.config
        batch, length, _ = x.shape
        for block in blocks:
            z = block.ln_1(x)
            q, k, v = block.attn.c_attn(z).split(cfg.n_embd, dim=2)
            shape = (batch, length, cfg.n_head, cfg.n_embd // cfg.n_head)
            q, k, v = (a.view(shape).transpose(1, 2) for a in (q, k, v))
            y = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, dropout_p=cfg.attn_pdrop if self.training else 0.0
            )
            y = block.attn.c_proj(y.transpose(1, 2).reshape(batch, length, cfg.n_embd))
            x = x + F.dropout(y, cfg.resid_pdrop, self.training)
            x = x + block.mlp(block.ln_2(x))
        return x

    def forward_with_states(self, tokens, positions=None):
        blocks = self.transformer.h
        x = self._blocks(self._embedded(tokens), blocks[: self.discoloop_split])
        first_state = x
        x = self._discoloop_boundary(x)
        x = self._blocks(x, blocks[self.discoloop_split :])
        return _readout(self, x, positions), first_state, x
