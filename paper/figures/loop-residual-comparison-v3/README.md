# Vanilla Loop, adapted Highway-gated Loop, and Invariant Loop

**The main aggregation comparison now uses a shared Transformer on both sides.** Highway Networks supplies the published gate mechanism; the middle figure is explicitly labeled **Highway-gated Loop (adapted)**. The original layer-to-layer Highway drawing is preserved in [v2](../loop-residual-comparison-v2/README.md).

| Figure | Role | Files |
| --- | --- | --- |
| Vanilla Loop | Overall architectural reference | [Preview](vanilla-loop-preview.png) · [PDF](vanilla-loop.pdf) · [SVG](vanilla-loop.svg) · [PNG](vanilla-loop.png) |
| Highway-gated Loop (adapted) | Proposed aggregation control with matched candidate computation | [Preview](highway-loop-preview.png) · [PDF](highway-loop.pdf) · [SVG](highway-loop.svg) · [PNG](highway-loop.png) |
| Invariant Loop | Implemented statistical scoring and normalized historical aggregation | [Preview](invariant-loop-preview.png) · [PDF](invariant-loop.pdf) · [SVG](invariant-loop.svg) · [PNG](invariant-loop.png) |

## Matching the comparison

The two aggregation figures use the same Transformer architecture, input reinjection, tied decode/encode, soft feedback `f_t`, candidate RMS normalization and initial hidden state. Transformer and gate parameters are shared across loops within each model. The proposed experiment would use matched backbone initialization and independent training, rather than tie the two models' trained parameters together.

For the matched aggregation pair, the common computation is:

```text
h_0 = e
H_1 = Fθ(e)
H_t = Fθ(h_(t−1) + e)       for t >= 2
p_t = softmax(E LN_f(H_t))
f_t = RMS(Eᵀ p_t)
c_t = RMS(H_t + f_t)
```

The adapted Highway update reads the recurrent carry directly:

```text
g_t = sigmoid(W_g h_(t−1) + b_g)     vector gate, shared parameters
h_t = (1−g_t) ⊙ h_(t−1) + g_t ⊙ c_t
```

Invariant Loop uses the implementation's five statistics, score processing and scalar historical mass:

```text
s_t = MLP(φ_t)          φ_t from c_t, h_(t−1), p_t, p_(t−1)
w_t = exp(clip(s_t,−8,8))
D_t = βD_(t−1) + w_t
γ_t = w_t / D_t
h_t = (1−γ_t) h_(t−1) + γ_t c_t
```

This compares the aggregation modules as a whole: content input versus statistics, vector versus scalar weights, and sigmoid versus historical-mass normalization. It does not isolate those individual choices. The ordinary Vanilla Loop remains an overall reference: its raw output is carried directly and its candidate path is not the matched soft-feedback path. A strict direct-handoff control for the matched aggregation pair would instead use `h_t=c_t`.

## Captions and status

**Highway-gated Loop (adapted).** The recurrent state feeds a shared Transformer candidate path and a shared linear-plus-sigmoid gate. Candidate computation, including soft feedback and normalization, matches Invariant Loop. Vector `g_t` combines each token's own previous state with its candidate. The control arrow terminates at the highlighted aggregate; all displayed tokens have disjoint local carries. The gate is adapted from *Training Very Deep Networks*, [§2 Eq. (3) and §2.2](https://proceedings.neurips.cc/paper/2015/file/215a71a12769b056c3c32e7299f1c5ed-Paper.pdf). The original paper does not specify this shared-Transformer architecture.

**Invariant Loop.** Five statistics feed the MLP; its processed score is normalized against retained historical mass to give scalar `γ_t`. The mass enters normalization rather than the MLP. Soft feedback, candidate formation and the common next-loop Q/K/V source follow the [actual model](../../../src/llm_memory_editability/loop_native_handoff.py) and [projection code](../../../src/llm_memory_editability/self_decode_alignment.py). The figure retains the preceding version's drawing of the implemented model. Detailed auxiliary state is specified in the [implementation notes](../loop-residual-comparison-v1/README.md#equations-and-implementation-trace).

**Status:** the Highway-gated Loop figure is a proposed control design. There is no sigmoid Highway handoff in the current model's supported handoffs. The existing `state-mlp` handoff uses mass normalization and cannot be relabeled as this control. No model implementation, training, or changes to the frozen [four-arm experiment](../../../results/loop-fourway-terminal-20261024/protocol.md) are included in this drawing update.

## Reproduction

[Drawing source](build_figures.py) · [References and design status](references.json) · [Source hashes and wiring](manifest.json) · [Separate LaTeX figures](figures.tex)

```bash
LD_LIBRARY_PATH="/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
MPLCONFIGDIR=/tmp/loop-residual-comparison-v3-mpl \
python paper/figures/loop-residual-comparison-v3/build_figures.py
```

Each figure is exported separately as a single-page PDF, editable SVG, 300 dpi PNG and preview. The generator imports no model code.
