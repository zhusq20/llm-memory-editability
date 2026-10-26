# 用途覆盖：收益、保持与成本

状态：`incomplete`。缺失结果保留为未知，不记作零分。

开发用于选择预算；正式受控比较与真实文档应用分别解释。PD/SFT保留各自采样配方。

## 开发选择

校准只用 `atomic_seen/direct`，不使用未练问法或正式结果。

选择状态：`incomplete`；目标是否达到：`None`；开发／正式／改写 token：None／None／None。

## 开发全部节点

| 运行 | 状态 | 原问法 | 未练问法 | 双新组合直接 | 分步 | 自生成桥 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| development-pd-A-addition/node-200192 | incomplete | 待测 | 待测 | 待测 | 待测 | 待测 |
| development-pd-B-addition/node-200192 | incomplete | 待测 | 待测 | 待测 | 待测 | 待测 |
| development-pd-C-addition/node-200192 | incomplete | 待测 | 待测 | 待测 | 待测 | 待测 |
| development-pd-A-addition/node-400384 | incomplete | 待测 | 待测 | 待测 | 待测 | 待测 |
| development-pd-B-addition/node-400384 | incomplete | 待测 | 待测 | 待测 | 待测 | 待测 |
| development-pd-C-addition/node-400384 | incomplete | 待测 | 待测 | 待测 | 待测 | 待测 |

## 正式受控比较

| 运行 | 状态 | 原问法 | 未练问法 | 双新组合直接 | 分步 | 自生成桥 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 尚未选择或未规划执行节点 | incomplete | 待测 | 待测 | 待测 | 待测 | 待测 |

## 真实文档应用

| 运行 | 状态 | 闭卷 | 原文在场 | 闭卷判分覆盖 |
| --- | --- | ---: | ---: | ---: |

该64案例已在既有批次使用，作为应用验证；自然应用没有安排分步或自生成桥评价。

## 保持与配对

JSON逐组保存A→B、B→C、A→C的配对差值和聚类95%区间，以及基座原正确题的保持率。改写以前一阶段为参考；变化的依赖答案不当作应保持旧答案。location保持以前次写入终点为基线。

背景、原文在场、原子掌握及联合前提子集均保留分母与覆盖率，完整结果始终同时保存。直接、显式分步、自生成桥链不混合。自然语义评分无效记录为未知，另外报有效覆盖率。

缺失或不完整项：14；保留训练失败：0。

完整分母、前提子集、配对区间、计数与缺口见 `summary.json`；完整性见 `validation.json`。

| 配对比较 | 主用途差值及95%区间（百分点） | 配对问题数 | 独立块数 |
| --- | ---: | ---: | ---: |
| development/pd/addition/node-200192/B-minus-A | 待测 | 待测 | 待测 |
| development/pd/addition/node-200192/C-minus-B | 待测 | 待测 | 待测 |
| development/pd/addition/node-200192/C-minus-A | 待测 | 待测 | 待测 |
| development/pd/addition/node-400384/B-minus-A | 待测 | 待测 | 待测 |
| development/pd/addition/node-400384/C-minus-B | 待测 | 待测 | 待测 |
| development/pd/addition/node-400384/C-minus-A | 待测 | 待测 | 待测 |

## 成本与完整性

训练成本只读取精确token节点的learning/complete计数。输入、padding、教师、主监督、回放分列；缺少的FLOPs、独占GPU时间和中间节点exposure不估算补齐。共享生成准备不重复计入每臂。

| 运行 | 基座原正确旧题保持 | 丢失旧题 | 主／回放监督token | 学生／教师处理token | 训练秒 |
| --- | ---: | ---: | ---: | ---: | ---: |
| development-pd-A-addition/node-200192 | 待测 | 待测 | None/None | None/None | None |
| development-pd-B-addition/node-200192 | 待测 | 待测 | None/None | None/None | None |
| development-pd-C-addition/node-200192 | 待测 | 待测 | None/None | None/None | None |
| development-pd-A-addition/node-400384 | 待测 | 待测 | None/None | None/None | None |
| development-pd-B-addition/node-400384 | 待测 | 待测 | None/None | None/None | None |
| development-pd-C-addition/node-400384 | 待测 | 待测 | None/None | None/None | None |

表内None表示该精确节点成本未知；重载、评测和共享生成时间另存，不混入训练秒。
