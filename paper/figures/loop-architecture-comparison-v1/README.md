# Loop architecture comparison

**本组三图保留为软反馈组件对照；最新主图为[普通Loop／Highway-gated Loop（适配设计）／Invariant](../loop-residual-comparison-v3/README.md)。** Highway适配分支与Invariant统一候选计算，用于比较聚合模块；该分支尚未实现或训练。DiscoLoop标注为采用的软反馈组件来源。已核实 DiscoLoop 的 [NeurIPS 2026 Poster 会议记录](https://neurips.cc/virtual/2026/poster/155886)；arXiv 旧版本的 Preprint 页眉不能据此解释成尚未接收。

三张图分别保存，保持同一纵向布局、两轮共享 Transformer、三个 token 位置和一个突出展示的位置。前两张依据论文原始递推；第三张沿用已确认的 [v12 实现图](../invariant-loop-v12/README.md)。三图只比较架构，不表示已经完成同条件效果归因。

| Figure | Inter-loop operation | Files |
| --- | --- | --- |
| Vanilla Loop | Direct state transfer | [Preview](vanilla-loop-preview.png) · [PDF](vanilla-loop.pdf) · [SVG](vanilla-loop.svg) · [PNG](vanilla-loop.png) |
| DiscoLoop | Add soft feedback | [Preview](discoloop-preview.png) · [PDF](discoloop.pdf) · [SVG](discoloop.svg) · [PNG](discoloop.png) |
| Invariant Loop | Soft feedback + MLP-based state aggregation | [Preview](invariant-loop-preview.png) · [PDF](invariant-loop.pdf) · [SVG](invariant-loop.svg) · [PNG](invariant-loop.png) |

## Cross-loop residual comparison

用户最新重点是旧状态经过一轮计算后的保留与合并。**当前三张独立图为[Vanilla Loop → Highway-gated Loop (adapted) → Invariant Loop](../loop-residual-comparison-v3/README.md)。** [原始Highway层间图](../loop-residual-comparison-v2/README.md)、[NDR图](../loop-residual-comparison-v1/README.md)及本目录的软反馈组件对照保留。

| Reference | Verified acceptance | Connection relevant to the figure |
| --- | --- | --- |
| The Neural Data Router: Adaptive Control Flow in Transformers Improves Systematic Generalization | [ICLR 2022 Poster, official entry](https://iclr.cc/virtual/2022/poster/6053) | Per-token, per-channel copy gate around a complete recurrent transformation |
| Fixed-Point Reasoning: Stable and Adaptive Deep Looped Models (FPRM) | [NeurIPS 2026 Poster, official entry](https://neurips.cc/virtual/2026/poster/148540) | Damped iteration combining the old state with a new proposal in its public method |
| InfiLoop | Acceptance not verified; [preprint](https://arxiv.org/abs/2610.11570) | Content-weighted, temporally decayed running aggregation |

**NDR 是门控保留的已发表实例。** 原论文 [§2.1 / Eqs. (1)–(4) 与 Figure 5](https://arxiv.org/pdf/2110.07732#page=19) 将共享层展开成 attention 后的状态 `a_t`、候选 `u_t`、门 `g_t=σ(FFN_gate(a_t))`，再计算 `h_t=(1−g_t)⊙h_(t−1)+g_t⊙u_t`。门是每个 token 的向量，旧状态支路跨过完整变换。其原始编码器使用 geometric attention，FFN 候选分支也不同于普通 GPT 残差块；画图应标为共享 NDR 变换，不能把普通 GPT block 加门称作原版 NDR。

我们的门采用每个 token 的五个统计量，经 MLP 预测分数，再与衰减后的历史质量归一化得到标量 `γ_t`。比较应突出门的输入、向量／标量权重与历史质量归一化。两种递推都能保留更早历史；不能把 NDR 描述成完全没有历史保留，也不能仅凭更新式相似断言机制相同。软反馈 `f_t` 则明确归属 DiscoLoop。

**FPRM 是另一篇已接收候选。** 官方会议标题与旧 arXiv 标题不同，但作者名单相符；作者[实验室首页](https://camail.org/)的 2026-09-24 公告也明确列为 NeurIPS full paper。其公开 [arXiv §3.3 / Algorithm 1](https://arxiv.org/html/2606.18206v1#S3.SS3) 使用 `z_next=(1−η)z+η fθ(z;x)`，推理时按残差停滞衰减阻尼系数。该阻尼与我们的逐 token、MLP 驱动权重不同。此次方法核查依据公开 arXiv 版本，会议条目已核实；未逐项核对会议最终 PDF 的所有改动。旧 ICML workshop 记录不覆盖后来的 NeurIPS 接收记录。

## Sources for the rendered figures

- **DiscoLoop — NeurIPS 2026 Poster**，软反馈组件来源。其 [§3.1 / Eq. (2)](https://arxiv.org/html/2607.00341v2#S3.SS1) 还给出了普通 Loop，因而前两张图均有同一已接收论文的原始定义。
- **Loop, Think, & Generalize — COLM 2026**，作者[官方代码仓库](https://github.com/OSU-NLP-Group/Loop-Think-Generalize)明确标注会议；[§4](https://arxiv.org/html/2604.07822v2#S4) 的直接递推也支持普通 Loop 的画法。此次未独立取得会议决定页，来源层级单列记录。
- **InfiLoop**，与历史状态聚合最直接相关，但目前核实到的是[预印本](https://arxiv.org/abs/2610.11570)，未找到正式接收记录。继续作为方法来源引用，不标成已发表对照；相关机制核查见[已有评审](../../../docs/reports/loop-native-handoff-review-20261020.md)。

DiscoLoop 的“当前输出加软反馈”与 InfiLoop 的“当前候选与历史摘要聚合”是两种不同连接。三张图中的 Transformer 块内部均保留原本的残差、归一化、attention 和 MLP；普通 Loop 没有的是图外单独的历史聚合支路。输入重注入 `e` 也不等于历史状态保留。

## English captions

**Vanilla Loop.** The same Transformer parameters are reused at each iteration. Its output becomes the next loop's input directly: `H_t=Fθ(h_(t−1)); h_t=H_t`. This follows DiscoLoop §3.1 and the recurrent block in Kohli et al. §4. Internal Transformer residuals remain inside the abstract block.

**DiscoLoop.** At an intermediate boundary, normalized soft vocabulary feedback `f_t=RMS(Eᵀp_t)` is added to the current output: `h_t=H_t+f_t`. The drawing uses the fixed-gate `α=1`, temperature `τ=1` variant from §5.1. It does not depict the paper's learned-α variant or its hard-token intervention. The final-loop feedback is disabled in the original method; the pictured boundary is not the final readout.

**Invariant Loop.** Soft feedback forms `c_t=RMS(H_t+f_t)`. An MLP-based module controls `h_t=(1−γ_t)h_(t−1)+γ_t c_t`, with a separate carry path for each token. The MLP reads five statistics of `c_t,h_(t−1),p_t,p_(t−1)`; its dashed `γ_t` arrow terminates at the highlighted token's aggregate. In the implementation, the MLP predicts a score; clipping, exponentiation and normalization against retained historical mass are folded into the compact module. After input reinjection, `LN₁(h_t+e)` supplies the common source for next-loop Q, K and V projections.

The Invariant Loop figure follows [aggregate_round](../../../src/llm_memory_editability/loop_native_handoff.py) and the actual [Q/K/V projection](../../../src/llm_memory_editability/self_decode_alignment.py). For later rounds, `H_t=Fθ(h_(t−1)+e)`. The current terminal-supervision recipe's first call is `Fθ(e)`; initialization is outside this intermediate-loop view. Candidate RMS, input reinjection, prediction decoding and final readout differ from the original published baselines and must not be described as a single-variable ablation. The project baseline implementations are adaptations; these two reference figures use the papers' stated recurrences.

All positions run the same rule. Fading two columns only focuses the explanation. The three ports belong to one joint causal Transformer call. Auxiliary accumulators and probability state, exact weight normalization and initialization remain in the [v12 implementation notes](../invariant-loop-v12/README.md).

## Reproduction

[Generator](build_figures.py) · [Venue verification](reference-status.json) · [Source hashes and figure trace](manifest.json) · [Three separate LaTeX figures](figures.tex)

```bash
LD_LIBRARY_PATH="/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
MPLCONFIGDIR=/tmp/loop-architecture-comparison-v1-mpl \
python paper/figures/loop-architecture-comparison-v1/build_figures.py
```

The generator reuses the v12 drawing source without modifying it. Each figure has its own PDF, SVG, 300 dpi PNG and preview. No model is imported and no training is run.
