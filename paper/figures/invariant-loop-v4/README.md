# Invariant Loop：相邻两轮的状态节点图

2026-10-24，根据用户“用几个状态节点展开相邻两轮”的要求重绘。主图只有五个状态节点，无模块方框、无公式符号。

[预览](invariant-loop-nodes-preview.png) · [高清 PNG](invariant-loop-nodes.png) · [矢量 PDF](invariant-loop-nodes.pdf) · [可编辑 SVG](invariant-loop-nodes.svg)

## 图注

**Invariant Loop 的相邻两轮更新。** 蓝色节点表示跨轮状态，橙色节点表示新候选。每轮从旧状态分出两条路径：水平路径保留旧状态供聚合使用；下方路径通过共享 Transformer、软词表反馈和候选归一化产生新候选。两条路径在带绿色圆环的节点按比例合并。两轮共用的中间蓝色节点既是上一轮的合并结果，也是下一轮的旧状态。五统计量打分与历史累计权重共同决定合并比例。节点面积、颜色和箭头粗细不编码实测权重或性能。

## 与代码的对应

以[模型实现](../../../src/llm_memory_editability/loop_native_handoff.py)的 `aggregate_round` 为准，依次表示：

```text
旧状态：h_(t−1)
生成候选：H_t = F(h_(t−1) + e)
          p_t = softmax(E LN_f(H_t))
          c_t = RMS(H_t + RMS(Eᵀ p_t))
加权合并：h_t = (1−γ_t) h_(t−1) + γ_t c_t
下一轮：  以 h_t 作为旧状态，重复同一组计算
```

图中“Transformer + 软词表反馈”是候选生成路径的抽象，包含每轮重新加入输入表示 `e` 与最终候选 RMS。共享块仍是实际因果 GPT-2，未改动主干。

五统计量是归一化预测熵、最大预测概率、相邻预测分布内积、候选相对旧状态的变化和二者余弦。网络给出分数，经截断、指数化得到 `w_t`；`γ_t=w_t/(βD_(t−1)+w_t)`。实现用 `N_t=βN_(t−1)+w_tc_t`、`D_t=βD_(t−1)+w_t` 和 `h_t=N_t/D_t` 计算，图中的加权合并与此完全等价。

代码另外保存历史分子 `N`、累计权重 `D` 和上一轮块输出的分布 `p`。为突出状态流，主图不逐项展开这些辅助变量。概率分布来自块输出，不从合并后的状态重新解码。初始 `h₀=N₀=e, D₀=1`，初始分布由 `e` 解码；最后一轮从聚合状态读出答案。完整细节见[实现说明](../invariant-loop-v2/README.md)。

这里的节点表示同一序列的隐藏状态；聚合逐 token 进行，Transformer 的因果注意力负责跨 token 信息交换。连续两轮没有对应两跳推理的预设含义。

## 来源与文件

画法参照此前已实际查看的 [DiscoLoop Figure 2 左侧状态节点图](https://arxiv.org/pdf/2607.00341v2#page=5)，保留本项目的软反馈与加权历史更新，没有引入该论文图中的硬 token 替换干预。

[绘图入口](build_figure.py)复用[绘图工具](../invariant-loop-v2/build_figures.py)；[来源哈希](manifest.json)；[论文插入片段](figure.tex)。PDF / SVG 为矢量，PNG 为 300 dpi。每个文件均为一张独立图，未执行模型前向或训练。

```bash
LD_LIBRARY_PATH="/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
MPLCONFIGDIR=/tmp/invariant-loop-v2-mpl \
python paper/figures/invariant-loop-v4/build_figure.py
```
