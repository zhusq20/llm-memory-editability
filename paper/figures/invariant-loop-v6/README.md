# Invariant Loop: vertical token-state diagram

2026-10-24. Revised to follow the vertical layout of DiscoLoop Figure 1, with a shared Transformer layer at each loop and individual token states between layers. All figure labels are in English.

[Preview](invariant-loop-vertical-preview.png) · [300 dpi PNG](invariant-loop-vertical.png) · [Vector PDF](invariant-loop-vertical.pdf) · [Editable SVG](invariant-loop-vertical.svg)

## Caption

**Invariant Loop.** Each column represents a token position. Both Transformer layers use the same parameters `θ`. Between successive loop calls, the block output `H_t` supplies a continuous path and a soft vocabulary-feedback path, shown by the green squares. Adding and RMS-normalizing these paths produces the candidate `c_t`. A weight computed from five statistics and the historical normalization mass combines each candidate with its previous state to form `h_t`, which enters the next loop. The highlighted region contains the inter-loop computation. One old-state bypass is drawn in full for readability; the same aggregation is applied independently at every token position.

The three displayed token positions are illustrative. Causal attention inside the shared Transformer exchanges information between positions. Circles represent token hidden-state vectors; squares represent feedback embedding vectors; the outer green rings mark aggregated states. Colors and sizes do not encode measured accuracy or weights.

## Implementation correspondence

The figure follows [LoopNativeHandoffGPT.aggregate_round](../../../src/llm_memory_editability/loop_native_handoff.py):

```text
H_t = Fθ(h_(t−1) + e)
p_t = softmax(E LN_f(H_t))
q_t = RMS(Eᵀ p_t)
c_t = RMS(H_t + q_t)
h_t = (1−γ_t) h_(t−1) + γ_t c_t
```

Input reinjection `+e` and normalization are specified here to keep the drawing sparse. A green square denotes the complete normalized soft feedback `q_t`, with no hard token selection. The same tied embedding matrix `E` supplies the vocabulary projection and soft re-encoding.

The five statistics are normalized entropy, maximum probability, adjacent prediction-distribution inner product, relative candidate change, and candidate/previous-state cosine. A shared `5→16→1` GELU MLP produces the score. With `w_t=exp(clip(s_t,−8,8))` and learned scalar `β`, the implementation computes:

```text
N_t = β N_(t−1) + w_t c_t
D_t = β D_(t−1) + w_t
γ_t = w_t / D_t
h_t = N_t / D_t
```

The complete carried tuple is `(h,N,D,p)`. Auxiliary `N,D,p` and their edges are omitted from the drawing. `p_t` is decoded from `H_t`, rather than from the aggregate `h_t`. Initialization is `h₀=N₀=e, D₀=1`, with `p₀=softmax(E LN_f(e))`. The final readout uses `h_R`. Further implementation details remain in the [detailed figure notes](../invariant-loop-v2/README.md).

## Reference and reproduction

The layout follows [DiscoLoop v2, Figure 1](https://arxiv.org/pdf/2607.00341v2#page=3): repeated horizontal layers, vertical token-state columns, and a local feedback branch. The history aggregation and five-statistic weighting follow this repository's implementation. The previous horizontal version is preserved in [v5](../invariant-loop-v5/README.md).

[Drawing source](build_figure.py) · [Source hashes and trace](manifest.json) · [LaTeX inclusion](figure.tex)

The script requires matplotlib and produces one independent figure in PDF, SVG, and PNG. It does not import a model or run training.

```bash
LD_LIBRARY_PATH="/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
MPLCONFIGDIR=/tmp/invariant-loop-v6-mpl \
python paper/figures/invariant-loop-v6/build_figure.py
```
