# Invariant Loop: vertical token-state diagram

2026-10-24. The compact MLP module now has an explicitly labeled five-statistic input, and its `γ_t` output connects directly to the highlighted token's aggregate. The figure keeps the vertical layout, parallel carry paths, feedback `f_t`, and common next-loop Q/K/V input. All labels are in English.

[Preview](invariant-loop-vertical-preview.png) · [300 dpi PNG](invariant-loop-vertical.png) · [Vector PDF](invariant-loop-vertical.pdf) · [Editable SVG](invariant-loop-vertical.svg)

The earlier expanded aggregation diagram is preserved as [historical detail](../invariant-loop-v10/README.md). The current figure uses the compact MLP representation.

## Caption

**Invariant Loop.** Each column represents a token position. Both Transformer layers use the same parameters `θ`. One token is highlighted for detail; the other two show the same computation in context. Soft vocabulary feedback `f_t` combines with the block output to form the candidate `c_t`. The compact MLP-based module produces the update weight used to combine the candidate with its previous state, giving `h_t`. Input reinjection and the first LayerNorm give `LN₁(h_t+e)`, the common source from which next-loop Q, K, and V are projected. Every displayed token retains its own complete blue bypass into its matching aggregate.

Above the MLP, `φ_t` is labeled as five statistics derived from `c_t,h_(t−1),p_t,p_(t−1)`. These tensors are the sources of the statistics, rather than raw concatenated MLP inputs. A dashed scalar-control arrow runs from the module's output directly into the highlighted token's aggregate ring `h_t`. It is separate from both the old-state bypass and the next-loop hidden-state input.

The `MLP → γ_t` arrow summarizes the learned weighting mechanism. In the implementation, the MLP outputs a score, followed by clipping, exponentiation, and normalization against the retained historical mass. These scalar operations are folded into the graphical abstraction and remain specified below. The figure does not assert the raw equality `γ_t=MLP(φ_t)`.

Each blue bypass turns right from its old state, runs straight down a dedicated channel, and enters the matching aggregate from the right. The three token lanes are disjoint; there are no wire crossings, bridges, or routes around another token. Each aggregate receives its own candidate from above and its own previous state from the right. Each token has its own acceptance weight; sharing the update rule does not require equal weights across positions.

The Transformer is displayed as one horizontal layer band with three token ports. The ports belong to a single joint causal Transformer call, not three independent token models. White channels separate the external carry paths from the computation ports. The same layer representation and parameters are used in both loops.

The three displayed token positions are illustrative. Causal attention inside the shared Transformer exchanges information between positions. Circles represent token hidden-state vectors; squares represent feedback embedding vectors; the outer green rings mark aggregated states. Fading the first two positions is a drawing choice: all positions still execute the same update. Colors, opacity, and sizes do not encode measured accuracy or weights.

## Implementation correspondence

The figure follows [LoopNativeHandoffGPT.aggregate_round](../../../src/llm_memory_editability/loop_native_handoff.py):

```text
H_t = Fθ(h_(t−1) + e)
p_t = softmax(E LN_f(H_t))
f_t = RMS(Eᵀ p_t)
c_t = RMS(H_t + f_t)
h_t = (1−γ_t) h_(t−1) + γ_t c_t
```

A green square denotes the complete normalized soft feedback `f_t`, with no hard token selection. The same tied embedding matrix `E` supplies the vocabulary projection and soft re-encoding. The implementation names this returned tensor `feedback`; `f_t` is its figure notation. Attention queries retain the distinct symbol `Q`.

The annotation at the next-loop entrance corresponds to [the actual Q/K/V projection](../../../src/llm_memory_editability/self_decode_alignment.py):

```text
z_(t+1) = LN₁(h_t + e)
Q_(t+1) = Linear_Q(z_(t+1))
K_(t+1) = Linear_K(z_(t+1))
V_(t+1) = Linear_V(z_(t+1))
```

Each `Linear` includes its bias. Aggregation changes the hidden-state input to all three projections; it does not mix Q, K, or V separately. Feedback and candidate RMS normalization remain specified in the caption, while input reinjection and the first LayerNorm are displayed in the common-input annotation.

The five statistics are normalized entropy, maximum probability, adjacent prediction-distribution inner product, relative candidate change, and candidate/previous-state cosine. A shared `5→16→1` GELU MLP produces the score. With `w_t=exp(clip(s_t,−8,8))` and learned scalar `β`, the implementation computes:

```text
N_t = β N_(t−1) + w_t c_t
D_t = β D_(t−1) + w_t
γ_t = w_t / D_t
h_t = N_t / D_t
```

The complete carried tuple is `(h,N,D,p)`. Auxiliary `N,D,p` and their edges are omitted from the drawing. `p_t` is decoded from `H_t`, rather than from the aggregate `h_t`. Initialization is `h₀=N₀=e, D₀=1`, with `p₀=softmax(E LN_f(e))`. The final readout uses `h_R`. Further implementation details remain in the [detailed figure notes](../invariant-loop-v2/README.md).

## Reference and reproduction

The layout follows [DiscoLoop v2, Figure 1](https://arxiv.org/pdf/2607.00341v2#page=3): repeated horizontal layers, vertical token-state columns, and a local feedback branch. The history aggregation and MLP-based weighting follow this repository's implementation. The prior overview with the detached MLP annotation is preserved in [v11](../invariant-loop-v11/README.md).

[Drawing source](build_figure.py) · [Source hashes and trace](manifest.json) · [LaTeX inclusion](figure.tex)

The script requires matplotlib and produces one independent figure in PDF, SVG, and PNG. It does not import a model or run training.

```bash
LD_LIBRARY_PATH="/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
MPLCONFIGDIR=/tmp/invariant-loop-v12-mpl \
python paper/figures/invariant-loop-v12/build_figure.py
```
