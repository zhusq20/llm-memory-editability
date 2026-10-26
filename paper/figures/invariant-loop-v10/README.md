# Invariant Loop: overview and enlarged aggregation detail

2026-10-24. The shared-Transformer overview and the enlarged `γ_t` detail are separate figures. Both use English labels and show one representative token in detail.

| Figure | Preview | Vector files |
| --- | --- | --- |
| Architecture overview | [PNG](../invariant-loop-v9/invariant-loop-vertical-preview.png) | [PDF](../invariant-loop-v9/invariant-loop-vertical.pdf) · [SVG](../invariant-loop-v9/invariant-loop-vertical.svg) |
| Enlarged aggregation detail | [PNG](invariant-loop-gamma-detail-preview.png) | [PDF](invariant-loop-gamma-detail.pdf) · [SVG](invariant-loop-gamma-detail.svg) |

[300 dpi aggregation PNG](invariant-loop-gamma-detail.png) · [Drawing source](build_detail.py) · [Source hashes](manifest.json) · [LaTeX inclusion](figures.tex)

## Detail caption

**Computing and applying the update weight.** At one token position, five statistics of the current prediction, previous prediction, candidate, and previous state enter a shared scoring MLP. Clipping the score and exponentiating produces a positive candidate weight `w_t`. The retained historical mass `βD_(t−1)` is a separate input to normalization. Their ratio gives `γ_t`, the fraction of the new candidate in the updated hidden state. The previous state receives the complementary fraction `1−γ_t`. This same rule is applied independently at every token position.

The enlargement follows the actual `features`, `score`, and `aggregate_round` methods in [loop_native_handoff.py](../../../src/llm_memory_editability/loop_native_handoff.py). The small network is `5→16→1` with GELU. Its width, score clipping, and initial decay come from the [frozen configuration](../../../configs/loop-native-handoff-development-v1.json).

## Five input statistics

The token index is omitted from all expressions.

| Figure label | Implemented statistic |
| --- | --- |
| Normalized entropy | `−Σ p_t(v) log p_t(v) / log |V|` |
| Max probability | `max_v p_t(v)` |
| Distribution overlap | `p_tᵀ p_(t−1)` |
| Relative change | `‖c_t−h_(t−1)‖ / (‖h_(t−1)‖+ε)` |
| Cosine similarity | `cos(c_t,h_(t−1))` |

Current and previous distributions are decoded from their respective block outputs. The candidate is `c_t=RMS(H_t+f_t)`, where `f_t` denotes the normalized soft vocabulary feedback used in the overview. No Query symbol is used for this feedback.

## Normalization and state update

```text
s_t = MLP(φ_t)
w_t = exp(clip(s_t,−8,8))
γ_t = w_t / (βD_(t−1) + w_t)
h_t = (1−γ_t)h_(t−1) + γ_t c_t
```

`β=sigmoid(b)` is a learned scalar shared across tokens and loops, initialized at 0.55. It is independent of the current candidate score. `D_(t−1)` stores the previous accumulated mass. The code computes the equivalent streaming recurrence:

```text
N_t = βN_(t−1) + w_t c_t
D_t = βD_(t−1) + w_t
h_t = N_t / D_t
```

Positive weights and positive retained mass give `0<γ_t<1`. The acceptance fraction depends on both the candidate's score and the retained mass. The figure does not identify it with answer correctness or a single confidence statistic. State aggregation acts between Transformer calls. The next-loop Q/K/V common input remains `LN₁(h_t+e)`, as labeled in the overview.

## Presentation and reproduction

The separate enlargement follows the local-detail approach of [DiscoLoop Figure 1](https://arxiv.org/pdf/2607.00341v2#page=3), while its equations follow this repository's Invariant Loop implementation. Vector files remain independent so they can be placed beside the overview without combining them into a single raster image.

The script uses matplotlib only and does not import a model or execute training. PDF / SVG are vector exports; PNG is 300 dpi. Dashed edges indicate scalar control and solid edges carry states or statistics. Colors and widths do not encode empirical results.

```bash
LD_LIBRARY_PATH="/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
MPLCONFIGDIR=/tmp/invariant-loop-v10-mpl \
python paper/figures/invariant-loop-v10/build_detail.py
```
