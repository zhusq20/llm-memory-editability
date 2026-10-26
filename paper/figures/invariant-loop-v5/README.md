# Invariant Loop：共享参数与状态节点图

2026-10-24，根据用户“以 loop 参数、hidden states 等状态为节点，参考 DiscoLoop 图 1”重绘。

[预览](invariant-loop-state-graph-preview.png) · [高清 PNG](invariant-loop-state-graph.png) · [矢量 PDF](invariant-loop-state-graph.pdf) · [可编辑 SVG](invariant-loop-state-graph.svg)

## 图注

**Invariant Loop 的共享参数与状态流。** 两个灰色 `Fθ` 节点表示相邻两轮调用同一个 Transformer，底部的参数节点 `θ` 用虚线连接两次调用。蓝色圆节点表示传入及传出的隐藏状态，橙色圆节点为 Transformer 块输出，绿色圆节点为候选状态或更新权重；`q` 方节点为软反馈 embedding。图中圆节点表示序列状态张量，未展开 batch 和 token 维度。

块输出 `H_t` 分别流向预测分布 `p_t` 和候选 `c_t`。预测分布经概率加权的词嵌入与 RMS 得到 `q_t`；它和块输出相加、再次 RMS 归一化，形成候选。统计量打分及历史累计权重确定接收比例 `γ_t`。旧状态由顶部旁路进入新状态节点，与候选按 `1−γ_t` 和 `γ_t` 合并。新状态 `h_t` 随即进入下一轮，继续使用同一个 `θ`。

底部橙色虚线只表示参数共享，绿色虚线表示标量权重控制。节点大小和线宽不编码实测权重或性能。

## 对应实际实现

依照 [loop_native_handoff.py](../../../src/llm_memory_editability/loop_native_handoff.py) 的 `aggregate_round`：

```text
H_t = Fθ(h_(t−1) + e)
p_t = softmax(E LN_f(H_t))
q_t = RMS(Eᵀ p_t)
c_t = RMS(H_t + q_t)
h_t = (1−γ_t) h_(t−1) + γ_t c_t
```

每次 loop 调用前重新加入同一个输入嵌入 `e`，这一步在图底说明，不额外画成操作方框。最后一轮使用共享输出头从聚合状态 `h_R` 读出答案。

图中 `γ_t` 节点抽象五统计量提取、共享 MLP 打分、截断、取指数与历史权重归一化。五个统计量是归一化熵、最大概率、相邻分布内积、候选相对旧状态的变化、候选与旧状态的余弦。它还使用上一轮保存的 `h_(t−1),p_(t−1),D_(t−1)`；这些历史输入没有逐项连线。

代码保存 `(h,N,D,p)`，执行 `N_t=βN_(t−1)+w_tc_t`、`D_t=βD_(t−1)+w_t`、`h_t=N_t/D_t`，其中 `γ_t=w_t/D_t`、`w_t=exp(clip(MLP(φ_t),−8,8))`、`β=sigmoid(b)`。这与主图的加权合并等价。`p_t` 来自块输出 `H_t`，不从聚合后的状态重新解码。

初始化、训练和归一化等完整说明仍见[实现细节](../invariant-loop-v2/README.md)。所有统计量与历史聚合逐 token 计算；跨 token 信息由共享 Transformer 的因果注意力交换。主图没有把循环次数等同于推理跳数。

## 画法与复现

参考 [DiscoLoop v2 Figure 1](https://arxiv.org/pdf/2607.00341v2#page=3) 的层、隐藏状态及局部软反馈支路表达；这里显式增加共享参数节点和旧状态到新状态的连接，以表现本实现的跨轮聚合。此前版本保留在 [v3](../invariant-loop-v3/README.md) 和 [v4](../invariant-loop-v4/README.md)。

[绘图源码](build_figure.py)复用[绘图工具](../invariant-loop-v2/build_figures.py)，[来源哈希](manifest.json)。每个输出文件均为一张独立图，PDF / SVG 为矢量，PNG 为 300 dpi。未执行模型前向或训练。

```bash
LD_LIBRARY_PATH="/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
MPLCONFIGDIR=/tmp/invariant-loop-v2-mpl \
python paper/figures/invariant-loop-v5/build_figure.py
```
