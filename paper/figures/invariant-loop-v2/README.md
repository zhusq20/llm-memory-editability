# Invariant Loop：架构与跨轮状态

依据当前 `LoopNativeHandoffGPT(handoff="invariant")` 实现重绘，2026-10-24。两张图分别保存；[旧版](../invariant-loop-v1/README.md)保留。

| 图 | 预览 | 高清 | 矢量 |
| --- | --- | --- | --- |
| 1：架构总览 | [预览](01-architecture-preview.png) | [PNG](01-architecture.png) | [PDF](01-architecture.pdf) · [SVG](01-architecture.svg) |
| 2：跨轮状态与权重计算 | [预览](02-state-handoff-preview.png) | [PNG](02-state-handoff.png) | [PDF](02-state-handoff.pdf) · [SVG](02-state-handoff.svg) |

## 图注与阅读顺序

**图 1. Invariant Loop 架构总览。** 每轮将输入表示 `e` 加到上一轮聚合状态 `h`，调用共享的因果 Transformer。块输出 `H` 经共享输出头转换为全词表概率 `p`，再对词嵌入加权求和并归一化。该反馈与 `H` 相加并再次归一化，形成候选 `c`。五统计量打分给出本轮权重；结合历史累计权重，得到接收比例 `γ`，更新跨轮状态。终轮从聚合状态读出答案 / EOS logits。

**图 2. 跨轮状态与权重计算。** 代码保存 `(h,N,D,p)`：`h` 进入下一轮 Transformer，并用于打分与聚合；`N,D` 是加权历史的分子与分母；`p` 是本轮块输出的解码分布，留给下一轮计算分布内积。五统计量经共享 `5→16→1` GELU 网络打分，截断并取指数得到正权重。历史衰减 `β` 是另一个独立可学习标量。图中下部给出代码实际使用的递推式。

相较旧图，Transformer 的 Pre-LN、注意力、MLP、残差和 dropout 收入一个共享块；将“交接”等笼统术语换为具体操作；把跨轮携带内容与各自去向独立展开。图 1 的主回路只标 `h`，图 2 明示全部四项状态。

## 符号与实现核对

| 符号 | 含义 | 实现 |
| --- | --- | --- |
| `e` | 词嵌入 + 可学习绝对位置嵌入，训练时再做 embedding dropout | [self_decode_alignment.py](../../../src/llm_memory_editability/self_decode_alignment.py)，`_embedded` |
| `Fθ`、`H_t` | 共享因果 GPT-2 块及其本轮输出；`H_t=Fθ(h_(t−1)+e)` | 同文件 `_round`；配置为 1 个独立块、宽 768、12 heads |
| `p_t` | `softmax(E LN_f(H_t))`，全词表分布，温度 1 | [loop_native_handoff.py](../../../src/llm_memory_editability/loop_native_handoff.py)，`_decoded_feedback` |
| `q_t`、`c_t` | 概率加权的词嵌入反馈；与块输出相加后得到归一化候选 | 同文件 `_decoded_feedback`、`aggregate_round` |
| `φ_t`、`w_t` | 五个标量；打分经 `clip(-8,8)` 与 `exp` 得到正权重 | 同文件 `features`、`score`、`aggregate_round` |
| `N_t,D_t,h_t` | 衰减后的历史加上本轮贡献；`h_t=N_t/D_t` | 同文件 `aggregate_round` |
| `β` | 共享可学习标量 `sigmoid(b)`，初值 0.55 | 同文件 `beta`；[冻结配方](../../../configs/loop-native-handoff-development-v1.json) |
| `γ_t` | 本轮候选占更新后总权重的比例 `w_t/(βD_(t−1)+w_t)` | `aggregate_round` 返回的 `weight / denominator` |

图 1 的凸组合式是代码递推的代数等价表达：代入 `N_(t−1)=D_(t−1)h_(t−1)` 即得。`γ` 不是独立 sigmoid 门；`β` 也不由本轮 `w` 产生。`N` 虽可由 `hD` 恢复，实际代码仍保存它，图 2 保留这一实现事实。

`RMS(x)=x/sqrt(mean(x²)+ε)`，这里没有可学习仿射参数，`ε=10⁻⁶`。它不替代 GPT-2 主干内的 LayerNorm。中间解码和最终输出共享 `LN_f`；输入词嵌入、解码投影、软再编码和最终输出共享同一矩阵 `E`。

初始化为 `h₀=N₀=e, D₀=1`，`p₀=softmax(E LN_f(e))`，因此首轮块输入是 `2e`。同一次 `run` 内每轮复用 `e`；分段训练重新调用 `run` 时会重新采 embedding dropout。图示为单次推理的架构，未展开分段深监督、段间 detach 或用于评测的参数 EMA。

所有状态与统计量逐 token 保存和计算。`h,N,c,H,q` 每位置为宽度 `d` 的向量，`p` 为词表大小 `|V|` 的向量，`D,w,γ` 每位置为标量；`β` 全位置、全轮共享。循环次数 `R` 由调用指定，图中的回路不表示自适应退出或生成中间文本。

“Invariant”指五维打分输入对同时置换两个概率分布的词表坐标，以及同时正交变换新旧隐藏状态保持不变。它不表示完整模型对实体、位置或数据分布不敏感。图 2 使用“分布内积”而非“预测正确性／相同答案”，避免给统计量赋予未验证的语义。

## 论文参照与本实现的区别

- [DiscoLoop，v2，Figure 1、§4、式 (4)–(6)](https://arxiv.org/html/2607.00341v2#S4)：参照块输出分出软解码—再编码支路的表达。当前实现固定反馈系数 1、温度 1；反馈后另做候选 RMS，再进入历史聚合。没有照搬原论文的学习注入门。
- [InfiLoop，v2，Figure 2、§3.1–3.3、式 (2)–(5)](https://arxiv.org/html/2610.11570v2#S3)：参照共享块和归一化历史递推。原方法用共享 pseudo-query 对候选内容打分；本实现用五统计量 MLP，并保存上一轮全词表分布。这两处是本图必须展开的区别。

论文提供模块组织与表达参考；图中的每条计算关系以本仓库实现为准。图示不承载性能结论。

## 导出、复核与重绘

每张图独立导出 PDF、保留文字的 SVG 与 300 dpi PNG；半宽缩放到 6.9 英寸时最小正文为 8.25 pt。中文矢量 PDF 嵌入字体。图 1 中绿色支路跨越蓝色反馈线时留出断口，表示无连接。已检查文字画布边界、图像渲染、公式与实现对应关系。

绘图源码：[build_figures.py](build_figures.py)。来源哈希与图表追踪：[manifest.json](manifest.json)。论文插入代码：[figures.tex](figures.tex)。

```bash
LD_LIBRARY_PATH="/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
MPLCONFIGDIR=/tmp/invariant-loop-v2-mpl \
python paper/figures/invariant-loop-v2/build_figures.py
```

仅生成图与来源记录，不导入模型、不执行模型前向、训练或调度。
