# Invariant Loop：简洁架构图

2026-10-24，按用户对 DiscoLoop Figure 1、2 简洁程度的要求重绘。主图只展示相邻轮次之间的核心计算；公式和辅助状态移至图注。

[预览](invariant-loop-preview.png) · [高清 PNG](invariant-loop.png) · [矢量 PDF](invariant-loop.pdf) · [可编辑 SVG](invariant-loop.svg)

## 图注

**Invariant Loop。** 同一个 Transformer 在相邻轮次复用。每轮将输入嵌入 `e` 加到旧状态 `h_(t−1)`，得到块输出 `H_t`。软解码后的全词表分布 `p_t` 加权词嵌入，并经 RMS 归一化形成反馈 `q_t`；它与块输出相加、再次归一化，形成新候选 `c_t`。共享的五统计量网络打分，结合历史累计权重计算接收比例 `γ_t`。旧状态经顶部旁路参与加权更新，更新后的 `h_t` 进入下一轮；最后一轮从 `h_R` 读出答案。

蓝色模块表示软词表反馈及候选生成；绿色模块表示统计量打分和历史加权。图中两个 `Fθ` 使用同一组参数。

## 主图省略的实现细节

打分网络读取归一化熵、最大预测概率、相邻预测分布内积、候选相对旧状态的变化和二者余弦，网络为 `5→16→1` GELU。正权重 `w_t=exp(clip(s_t,−8,8))`，`γ_t=w_t/(βD_(t−1)+w_t)`；`β` 为共享可学习标量。主图的“打分 + 权重归一化”同时概括这两步，不能解释成只由当前候选决定的独立 sigmoid 门。

代码实际执行：

```text
N_t = β N_(t−1) + w_t c_t
D_t = β D_(t−1) + w_t
h_t = N_t / D_t = (1−γ_t) h_(t−1) + γ_t c_t
```

完整携带元组为 `(h,N,D,p)`。主图只展开隐藏状态主路；上一轮 `p` 用于分布内积，`N,D` 用于历史递推，均未单独画出。`p_t` 来自块输出 `H_t`，不是从聚合后的 `h_t` 解码。初始化 `h₀=N₀=e, D₀=1`，`p₀=softmax(E LN_f(e))`。各位置分别计算统计量与聚合，跨位置信息由因果 Transformer 交换。

Transformer 内部残差、LayerNorm、attention、MLP；输出头与词嵌入权重绑定；具体 RMS、初始化和训练配方，均以[上一版实现说明](../invariant-loop-v2/README.md)及[模型代码](../../../src/llm_memory_editability/loop_native_handoff.py)为准。架构未改动，未执行训练。

## 参考与复现

实际查看了 [DiscoLoop v2](https://arxiv.org/pdf/2607.00341v2) 的 Figure 1（PDF 第 3 页）与 Figure 2（第 5 页）：前者用共享层和局部软反馈支路表达方法，后者用少量状态节点聚焦两轮之间的干预。这里借鉴它们的信息取舍；Figure 2 的硬解码干预没有移入我们的模型。历史加权递推来自 [InfiLoop §3](https://arxiv.org/html/2610.11570v2#S3)，五统计量打分按本项目实现绘制。

绘图入口：[build_figure.py](build_figure.py)，复用[上一版绘图样式工具](../invariant-loop-v2/build_figures.py)。[manifest.json](manifest.json)保存代码和样式哈希。每张输出是一张独立图；PDF / SVG 为矢量，PNG 为 300 dpi。

```bash
LD_LIBRARY_PATH="/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
MPLCONFIGDIR=/tmp/invariant-loop-v2-mpl \
python paper/figures/invariant-loop-v3/build_figure.py
```
