# Vanilla Loop, Highway Networks, and Invariant Loop

The current aggregation comparison uses [the shared-Transformer Highway adaptation in v3](../loop-residual-comparison-v3/README.md). This version preserves the original layer-to-layer Highway mechanism as a reference.

The middle figure now depicts Highway Networks. It uses a direct layer-input gate, replacing the earlier NDR post-attention input and two-layer gate MLP. All figures retain the vertical layout, one highlighted position, and disjoint left-side carry paths.

| Figure | Mechanism | Files |
| --- | --- | --- |
| Vanilla Loop | Direct state transfer | [Preview](vanilla-loop-preview.png) · [PDF](vanilla-loop.pdf) · [SVG](vanilla-loop.svg) · [PNG](vanilla-loop.png) |
| Highway Networks | Layer input → Linear → Sigmoid → vector gate | [Preview](highway-preview.png) · [PDF](highway.pdf) · [SVG](highway.svg) · [PNG](highway.png) |
| Invariant Loop | Five statistics → MLP → mass normalization → scalar gate | [Preview](invariant-loop-preview.png) · [PDF](invariant-loop.pdf) · [SVG](invariant-loop.svg) · [PNG](invariant-loop.png) |

## Captions

**Highway Networks.** The layer input `h_(ℓ−1)` feeds both a nonlinear transform and a linear-plus-sigmoid transform gate. The vector gate mixes the candidate `u_ℓ` with the carried input, channel by channel. Each displayed position has its own complete carry path, and the highlighted gate connects directly to its corresponding mixture. This follows *Training Very Deep Networks*, [§2, Eq. (3), and §2.2](https://proceedings.neurips.cc/paper/2015/file/215a71a12769b056c3c32e7299f1c5ed-Paper.pdf), published at [NeurIPS (NIPS) 2015](https://proceedings.neurips.cc/paper/2015/hash/215a71a12769b056c3c32e7299f1c5ed-Abstract.html).

```text
u_ℓ = H_ℓ(h_(ℓ−1))
g_ℓ = sigmoid(W_T,ℓᵀ h_(ℓ−1) + b_T,ℓ)
h_ℓ = (1−g_ℓ) ⊙ h_(ℓ−1) + g_ℓ ⊙ u_ℓ
```

Here `g_ℓ` denotes the original paper's transform gate `T`; the carry gate is `1−T`. `Linear` includes its bias. The original highway network connects successive layers; the diagram therefore uses `Layer ℓ`, `Layer ℓ+1` and distinct transforms `H_ℓ`, `H_(ℓ+1)`. It does not assert weight sharing or a Transformer backbone. Position lanes align the gate mechanism with the token-state views in the other figures; they do not assert that the original experiments used language tokens. Reusing this gate around a shared Transformer would be a Highway-style Loop adaptation.

**Vanilla Loop.** A shared Transformer passes its output directly to the next iteration. Internal Transformer residuals and normalization remain abstract within the block. The recurrence follows [DiscoLoop §3.1 / Eq. (2)](https://arxiv.org/html/2607.00341v2#S3.SS1).

**Invariant Loop.** Soft feedback `f_t` forms candidate `c_t`. An MLP reads five statistics from `c_t,h_(t−1),p_t,p_(t−1)`. Score processing and normalization against retained historical mass `βD_(t−1)` produce a scalar `γ_t` for each token. Retained mass enters the normalization row, not the MLP. Each token's own state is preserved through a local carry. Input reinjection and LayerNorm yield `LN₁(h_t+e)`, the common next-loop Q/K/V source. This follows [the actual implementation](../../../src/llm_memory_editability/loop_native_handoff.py), including `w_t=exp(clip(MLP(φ_t),−8,8))` and `γ_t=w_t/(βD_(t−1)+w_t)`. Soft feedback follows [DiscoLoop §4](https://arxiv.org/html/2607.00341v2#S4). Exact feature definitions and auxiliary state remain in the [implementation notes](../loop-residual-comparison-v1/README.md#equations-and-implementation-trace).

Solid arrows carry vectors; dashed arrows carry gate weights or scalar control. The other two positions remain visible in context and use the same respective rule. Highway has layer-to-layer carries; the two Loop models iterate through shared weights. This comparison illustrates update mechanisms and makes no performance claim.

## Reproduction

[Generator](build_figures.py) · [References](references.json) · [Source hashes and wiring](manifest.json) · [Separate LaTeX figures](figures.tex)

```bash
LD_LIBRARY_PATH="/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
MPLCONFIGDIR=/tmp/loop-residual-comparison-v2-mpl \
python paper/figures/loop-residual-comparison-v2/build_figures.py
```

Every figure has an independent PDF, editable SVG, 300 dpi PNG and preview. The Vanilla and Invariant drawings retain the prior geometry and content. The [previous NDR comparison](../loop-residual-comparison-v1/README.md) is preserved. No model or training code is executed.
