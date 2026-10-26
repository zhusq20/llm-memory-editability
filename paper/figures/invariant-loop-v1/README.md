# Invariant Loop 架构示意图

2026-10-24，依据 `LoopNativeHandoffGPT(handoff="invariant")` 的实际实现绘制。图示内容是当前 invariant 臂，`state-mlp` 是另一个比较臂。这里只绘制架构，没有新模型前向、训练或调度操作。

| 用途 | 中文 | 英文 |
| --- | --- | --- |
| 预览 | [PNG](invariant-loop-zh.png) | [PNG](invariant-loop.png) |
| 论文插图，矢量 | [PDF](invariant-loop-zh.pdf) | [PDF](invariant-loop.pdf) |
| 可编辑文字与模块 | [SVG](invariant-loop-zh.svg) | [SVG](invariant-loop.svg) |

英文版可按 6.9 英寸宽放入论文双栏，缩放后最小正文字号为 8.1 pt；数学下标随公式缩放。PNG 为 300 dpi，PDF 嵌入字体，SVG 保留文字。重绘源码见 [build_figure.py](build_figure.py)，来源哈希和检查记录见 [manifest.json](manifest.json)。[figure.tex](figure.tex) 提供英文图注与双栏插入代码。

## 怎样读图

- **(a) 整体循环。** 输入先经过词嵌入、可学习绝对位置嵌入和 embedding dropout，得到 `e`。每轮把 `e` 再加到旧聚合状态上，经过同一个 GPT-2 块、候选生成和交接模块。主干保留真实的 Pre-LN、多头因果注意力、MLP 和两条残差；图中的残差加法与 LayerNorm 分开。最终从聚合状态 `h_R` 读出答案 / EOS logits。
- **(b) 候选生成。** `H_t` 经共享 `LN_f` 和绑定词嵌入矩阵 `E` 解码，再对**全词表**做 softmax。用 `Eᵀp_t` 软再编码，经过无可学习仿射参数的 RMS 归一化，与 `H_t` 相加，再做一次同类 RMS 归一化得到候选 `c_t`。这里没有 argmax、硬 token 选择或离散中间答案。
- **(c) 跨轮交接。** 五个统计量依次是归一化熵、最大概率、相邻解码分布内积、候选相对旧状态的变化、候选与旧状态余弦。共享 `5→16→1` GELU 网络给出分数，经截断、取指数后，用 `N/D` 流式递推聚合。`β=σ(b)` 是所有位置与轮次共用的可学习标量，初值 0.55。打分网络 113 个参数，加上衰减参数共 114 个。

图 (a) 的循环箭头表示旧隐藏状态进入下一轮的主通路；完整实现同时携带 `S_t=(h_t,N_t,D_t,p_t)`，见 (c)。`N` 是加权历史分子，`D` 是累计权重。初始化为 `h₀=N₀=e, D₀=1`，且 `p₀=softmax(E LN_f(e))`；首轮块输入因此是 `2e`。上一轮概率指上轮块输出 `H_{t−1}` 的解码分布，初始轮用 `p₀`，不是每轮另从聚合状态重新解码。

每个 token 位置独立维护统计量与历史分子 / 分母；跨位置的信息交换发生在主干因果注意力中。词表投影、反馈再编码与最终输出共享 `E`；中间和最终解码也共用 `LN_f`。RMS 不替代主干 LayerNorm。初始 `e` 保留原始 embedding 尺度，没有预先 RMS 归一化。

归一化聚合也可等价写为 `h_t=(1−γ_t)h_{t−1}+γ_t c_t`，其中 `γ_t=w_t/(βD_{t−1}+w_t)`；图中保留与代码一致的 `N/D` 形式，避免把接收比例误画成独立 sigmoid 门。完整携带元组中 `h=N/D` 是实现保存的冗余状态。

“Invariant”指打分特征在同时置换两个概率分布的词表坐标、同时正交旋转新旧隐藏状态时保持不变，不表示完整模型对实体、位置或数据分布不敏感，也不保证统计量与实体身份没有关联。

## 实现对应

| 图中内容 | 实现入口 |
| --- | --- |
| `e`、词与绝对位置嵌入、embedding dropout | [self_decode_alignment.py](../../../src/llm_memory_editability/self_decode_alignment.py) 的 `SelfDecodeGPT._embedded` |
| Pre-LN、因果 MHA、attention dropout、输出投影、残差与 MLP | 同文件的 `SelfDecodeGPT._round`；继承路径经 `SupervisedAlignmentGPT._round`，invariant 调用不传首轮 KV bank |
| 最终 / 中间 `LN_f` 与绑定 `E` | 同文件的 `_readout` |
| 软解码再编码、无仿射 RMS | [loop_native_handoff.py](../../../src/llm_memory_editability/loop_native_handoff.py) 的 `_decoded_feedback`、`rms_normalize` |
| 五维统计量与 GELU 打分 | 同文件的 `features`、`score` |
| 裁剪为 `[-8,8]`、取指数、`N/D` 递推 | 同文件的 `aggregate_round` |
| 状态初始化、每轮注入与最终聚合状态读出 | 同文件的 `initial_state`、`run` |
| 宽度 768、12 heads、1 个独立块、dropout 0.1 | [冻结科学配方](../../../configs/loop-native-handoff-development-v1.json) 的 `model`、`training` 和 `handoff` |

训练中 embedding、attention 和 residual dropout 均为 0.1，推理时关闭。图中 MHA 包含 Q/K/V 投影、缩放点积注意力、attention dropout 和输出投影；MLP 使用 GPT-2 配置的 `gelu_new`。图 (c) 的小打分 MLP 使用 `F.gelu`，两处激活没有混同。

训练计算图另有两个监督段：第一段轮数从 `[4,6]` 均匀抽取，第二段从 `[1,4]` 抽取，段间 detach，每段末仅用答案 / EOS 交叉熵。每次 `run` 重采 embedding dropout，因此分段训练时第二段注入 `e` 可以不同；同一段内每轮复用同一个 `e`。这些训练操作和参数 EMA 不作为推理模块画入总图。图中的 `R` 是调用时指定的循环次数，不是由问题跳数控制，也没有学习式退出模块。

## 参考论文与画法

1. [Attention Is All You Need](https://arxiv.org/abs/1706.03762)，Figure 1：借鉴自下而上的模块流、嵌入加法、模块边框与旁路残差。原图采用 Post-LN；本图依实际 GPT-2 代码画成 Pre-LN。没有引入原图的 encoder-decoder 交叉注意力。
2. [InfiLoop: Scaling to Tens of Thousands of Test-Time Iterations with Loop-Native Attention Residuals](https://arxiv.org/abs/2610.11570)，Figure 2、§3.1–3.2：参照共享块重复执行、历史加权与常量历史存储的流式 `N/D` 表达。本实现的词表反馈和五统计量 MLP 按本项目代码绘制，不将其归为 InfiLoop 原论文的模块。
3. [Ouro: Scaling Latent Reasoning via Looped Language Models](https://arxiv.org/abs/2510.25741)，Figure 1：参照用回路表达参数共享与重复计算的画法。本图的 `R` 对应显式执行轮数。

参考图只用于表达方式与方法归属；最终连接逐项按代码核对。原 [2026-10-21 架构图](../../../results/invariant-architecture-diagram-20261021/invariant-architecture.pdf) 保留。本版将主干、反馈与交接分开展开，没有修改科学模型。

## 重绘

从仓库根目录执行：

```bash
uv run --no-project --with 'matplotlib>=3.9,<4' python paper/figures/invariant-loop-v1/build_figure.py
```

可用 `--language en` 或 `--language zh` 只重绘一种语言；默认同时生成两种。中文需要 Noto Sans CJK 字体。生成器检查配置关键值、五特征顺序、文字画布边界与模块标签边界；源码哈希记录在 manifest 中。生成结果还经人工查看图像，确认各条残差与支路连接。
