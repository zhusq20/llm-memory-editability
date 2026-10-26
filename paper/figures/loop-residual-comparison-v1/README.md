# Vanilla Loop, NDR, and Invariant Loop

The current comparison uses [an adapted Highway-gated Loop in v3](../loop-residual-comparison-v3/README.md). The [original Highway layer mechanism is in v2](../loop-residual-comparison-v2/README.md). This version preserves the earlier NDR drawing and its original source trace.

Three separate English figures use the same vertical geometry. Only the third token's gate is expanded. Every displayed NDR/Invariant token has its own complete old-state bypass, routed on the left to keep the gate input and control lines clear.

| Figure | Main distinction | Files |
| --- | --- | --- |
| Vanilla Loop | Direct transfer | [Preview](vanilla-loop-preview.png) · [PDF](vanilla-loop.pdf) · [SVG](vanilla-loop.svg) · [PNG](vanilla-loop.png) |
| NDR | Post-attention input; vector sigmoid gate | [Preview](ndr-preview.png) · [PDF](ndr.pdf) · [SVG](ndr.svg) · [PNG](ndr.png) |
| Invariant Loop | Five statistics; scalar weight normalized by historical mass | [Preview](invariant-loop-preview.png) · [PDF](invariant-loop.pdf) · [SVG](invariant-loop.svg) · [PNG](invariant-loop.png) |

## Captions

**Vanilla Loop.** Each iteration applies the shared Transformer to the preceding hidden states; its output is passed directly to the next iteration. The block includes its internal attention, MLP, normalization and residual connections. The ordinary recurrence follows [DiscoLoop Eq. (2)](https://arxiv.org/html/2607.00341v2#S3.SS1).

**NDR.** A post-attention state `a_t` inside the shared layer feeds a two-layer MLP and sigmoid. Its vector gate `g_t` combines the candidate `u_t` with that token's old state, channel by channel. The drawing follows [NDR §2.1 / Eqs. (1)–(5) and Figure 5](https://arxiv.org/pdf/2110.07732v4#page=19); publication is confirmed by the [ICLR 2022 conference entry](https://iclr.cc/virtual/2022/poster/6053). The internal tap is after attention, its residual addition and LayerNorm. Geometric attention and the data FFN remain abstract inside the NDR band. The original candidate FFN has no ordinary FFN residual; this is the paper's encoder layer, not a claim that NDR uses the same GPT block as our model.

**Invariant Loop.** Soft vocabulary feedback `f_t` forms a candidate. The MLP input consists of five statistics derived from `c_t,h_(t−1),p_t,p_(t−1)`. Its score is clipped and exponentiated into a weight; mass normalization combines that weight with the retained historical mass `βD_(t−1)` to produce scalar `γ_t`. The control arrow reaches the highlighted aggregate. Each token preserves its own old state and mixes it with its own candidate. After input reinjection and LayerNorm, `LN₁(h_t+e)` is the common source for next-loop Q, K and V projections. Soft feedback is adopted from [DiscoLoop §4](https://arxiv.org/html/2607.00341v2#S4).

The two rows inside each compact gate module expose the key distinction: NDR uses a sigmoid; our implementation uses historical-mass normalization. `βD_(t−1)` enters only the lower normalization row, not the MLP. Dashed arrows carry weights or scalar control; solid arrows carry state vectors. The first two token columns are faded to focus the explanation; they execute the same respective update. The small ports in each band belong to one joint sequence transformation.

## Equations and implementation trace

The loop index is `t`; token and channel indices are suppressed. NDR uses an elementwise gate, while our model uses a single scalar per token. Both recurrent schemes can carry information from earlier loops through the previous state.

```text
Vanilla:
  H_t = Fθ(h_previous)
  h_t = H_t

NDR:
  a_t = LN(MHA(h_previous, H_previous, H_previous) + h_previous)
  u_t = LN(FFN_data(a_t))
  g_t = sigmoid(FFN_gate(a_t))
  h_t = (1-g_t) ⊙ h_previous + g_t ⊙ u_t

Invariant (later loops):
  H_t = Fθ(h_previous + e)
  p_t = softmax(E LN_f(H_t))
  f_t = RMS(Eᵀ p_t)
  c_t = RMS(H_t + f_t)
  φ_t = [normalized entropy, maximum probability,
         p_tᵀ p_previous, relative candidate change,
         cosine(c_t,h_previous)]
  s_t = MLP(φ_t)
  w_t = exp(clip(s_t,-8,8))
  D_t = β D_previous + w_t
  γ_t = w_t / D_t
  h_t = (1-γ_t) h_previous + γ_t c_t
```

Invariant equations follow [features, score and aggregate_round](../../../src/llm_memory_editability/loop_native_handoff.py). The common Q/K/V input follows [the actual projection code](../../../src/llm_memory_editability/self_decode_alignment.py). The complete running state also includes `N_t=βN_previous+w_t c_t`, with `h_t=N_t/D_t`, and the previous probability. Their carry edges remain in the caption to keep the drawing focused. Only retained mass is surfaced because it directly explains the gate's normalization. The compact MLP row includes score clipping and exponentiation; it does not assert that the raw MLP output is `γ_t`.

The [current recipe](../../../configs/loop-invariant-poisson-terminal-v1.json) starts with `Fθ(e)` and uses `Fθ(h_previous+e)` afterwards. Initialization and final readout are outside this intermediate-loop view. This is an architecture comparison; it contains no performance claim or assertion of a single-variable ablation. It adds no training or NDR implementation to the project.

## Reproduction

[Generator](build_figures.py) · [References](references.json) · [Source hashes and wiring trace](manifest.json) · [Three independent LaTeX figures](figures.tex)

```bash
LD_LIBRARY_PATH="/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
MPLCONFIGDIR=/tmp/loop-residual-comparison-v1-mpl \
python paper/figures/loop-residual-comparison-v1/build_figures.py
```

Each figure is exported separately as a single-page PDF, editable SVG, 300 dpi PNG and preview. Older [soft-feedback comparison figures](../loop-architecture-comparison-v1/README.md) and the [v12 Invariant drawing](../invariant-loop-v12/README.md) are preserved.
