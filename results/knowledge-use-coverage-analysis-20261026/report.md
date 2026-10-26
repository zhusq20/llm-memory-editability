# 用途覆盖：收益、保持与成本

状态：`incomplete`。缺失结果保留为未知，不记作零分。

开发用于选择预算；正式受控比较与真实文档应用分别解释。PD/SFT保留各自采样配方。

## 开发选择

校准只用 `atomic_seen/direct`，不使用未练问法或正式结果。

选择状态：`complete`；目标是否达到：`True`；开发／正式／改写 token：200192／800768／200192。

## 开发全部节点

| 运行 | 状态 | 原问法 | 未练问法 | 双新组合直接 | 分步 | 自生成桥 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| development-pd-A-addition/node-200192 | complete | 100.00% | 92.19% | 0.00% | 3.12% | 100.00% |
| development-pd-B-addition/node-200192 | complete | 100.00% | 99.22% | 3.12% | 3.12% | 100.00% |
| development-pd-C-addition/node-200192 | complete | 100.00% | 100.00% | 3.12% | 3.12% | 100.00% |
| development-pd-A-addition/node-400384 | complete | 100.00% | 92.19% | 0.00% | 3.12% | 100.00% |
| development-pd-B-addition/node-400384 | complete | 100.00% | 99.22% | 3.12% | 6.25% | 100.00% |
| development-pd-C-addition/node-400384 | complete | 100.00% | 100.00% | 3.12% | 3.12% | 100.00% |

## 正式受控比较

| 运行 | 状态 | 原问法 | 未练问法 | 双新组合直接 | 分步 | 自生成桥 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| formal-pd-A-addition/node-800768 | complete | 99.22% | 79.49% | 1.56% | 3.91% | 91.41% |
| formal-pd-A-rewrite/node-200192 | complete | 87.11% | 54.30% | 0.00% | 0.00% | 76.56% |
| formal-pd-B-addition/node-800768 | complete | 99.61% | 99.61% | 1.56% | 1.56% | 99.22% |
| formal-pd-B-rewrite/node-200192 | complete | 77.34% | 69.92% | 0.00% | 0.00% | 57.81% |
| formal-pd-C-addition/node-800768 | complete | 99.22% | 98.83% | 18.75% | 19.53% | 99.22% |
| formal-pd-C-rewrite/node-200192 | complete | 98.05% | 96.88% | 0.78% | 0.78% | 96.88% |
| formal-sft-A-addition/node-800768 | complete | 99.22% | 73.83% | 2.34% | 2.34% | 95.31% |
| formal-sft-A-rewrite/node-200192 | incomplete | 待测 | 待测 | 待测 | 待测 | 待测 |
| formal-sft-B-addition/node-800768 | complete | 99.22% | 99.22% | 3.12% | 3.12% | 99.22% |
| formal-sft-B-rewrite/node-200192 | incomplete | 待测 | 待测 | 待测 | 待测 | 待测 |
| formal-sft-C-addition/node-800768 | complete | 99.22% | 98.63% | 6.25% | 5.47% | 99.22% |
| formal-sft-C-rewrite/node-200192 | incomplete | 待测 | 待测 | 待测 | 待测 | 待测 |

## 真实文档应用

| 运行 | 状态 | 闭卷 | 原文在场 | 闭卷判分覆盖 |
| --- | --- | ---: | ---: | ---: |
| natural-pd-A-addition/node-800768 | incomplete | 待测 | 待测 | 待测 |
| natural-pd-B-addition/node-800768 | incomplete | 待测 | 待测 | 待测 |
| natural-pd-C-addition/node-800768 | incomplete | 待测 | 待测 | 待测 |
| natural-sft-A-addition/node-800768 | incomplete | 待测 | 待测 | 待测 |
| natural-sft-B-addition/node-800768 | incomplete | 待测 | 待测 | 待测 |
| natural-sft-C-addition/node-800768 | incomplete | 待测 | 待测 | 待测 |

该64案例已在既有批次使用，作为应用验证；自然应用没有安排分步或自生成桥评价。

## 保持与配对

JSON逐组保存A→B、B→C、A→C的配对差值和聚类95%区间，以及基座原正确题的保持率。改写以前一阶段为参考；变化的依赖答案不当作应保持旧答案。location保持以前次写入终点为基线。

背景、原文在场、原子掌握及联合前提子集均保留分母与覆盖率，完整结果始终同时保存。直接、显式分步、自生成桥链不混合。自然语义评分无效记录为未知，另外报有效覆盖率。
旧知识317题按国家共同聚类；源ID无法识别时保留点估计，区间记未知，不按独立题替代。

缺失或不完整项：10；保留训练失败：0。

完整分母、前提子集、配对区间、计数与缺口见 `summary.json`；完整性见 `validation.json`。

| 配对比较 | 主用途差值及95%区间（百分点） | 配对问题数 | 独立块数 |
| --- | ---: | ---: | ---: |
| development/pd/addition/node-200192/B-minus-A | +3.12 [+0.00, +9.38] | 32 | 32 |
| development/pd/addition/node-200192/C-minus-B | +0.00 [-9.38, +9.38] | 32 | 32 |
| development/pd/addition/node-200192/C-minus-A | +3.12 [+0.00, +9.38] | 32 | 32 |
| development/pd/addition/node-400384/B-minus-A | +3.12 [+0.00, +9.38] | 32 | 32 |
| development/pd/addition/node-400384/C-minus-B | +0.00 [-9.38, +9.38] | 32 | 32 |
| development/pd/addition/node-400384/C-minus-A | +3.12 [+0.00, +9.38] | 32 | 32 |
| formal/pd/addition/node-800768/B-minus-A | +0.00 [-2.34, +2.34] | 128 | 128 |
| formal/pd/addition/node-800768/C-minus-B | +17.19 [+10.94, +24.22] | 128 | 128 |
| formal/pd/addition/node-800768/C-minus-A | +17.19 [+10.16, +24.22] | 128 | 128 |
| formal/pd/rewrite/node-200192/B-minus-A | +0.00 [+0.00, +0.00] | 128 | 64 |
| formal/pd/rewrite/node-200192/C-minus-B | +0.78 [+0.00, +2.34] | 128 | 64 |
| formal/pd/rewrite/node-200192/C-minus-A | +0.78 [+0.00, +2.34] | 128 | 64 |
| formal/sft/addition/node-800768/B-minus-A | +0.78 [-3.12, +4.69] | 128 | 128 |
| formal/sft/addition/node-800768/C-minus-B | +3.12 [-2.34, +8.59] | 128 | 128 |
| formal/sft/addition/node-800768/C-minus-A | +3.91 [+0.00, +8.59] | 128 | 128 |
| formal/sft/rewrite/node-200192/B-minus-A | 待测 | 待测 | 待测 |
| formal/sft/rewrite/node-200192/C-minus-B | 待测 | 待测 | 待测 |
| formal/sft/rewrite/node-200192/C-minus-A | 待测 | 待测 | 待测 |
| natural/pd/addition/node-800768/B-minus-A | 待测 | 待测 | 待测 |
| natural/pd/addition/node-800768/C-minus-B | 待测 | 待测 | 待测 |
| natural/pd/addition/node-800768/C-minus-A | 待测 | 待测 | 待测 |
| natural/sft/addition/node-800768/B-minus-A | 待测 | 待测 | 待测 |
| natural/sft/addition/node-800768/C-minus-B | 待测 | 待测 | 待测 |
| natural/sft/addition/node-800768/C-minus-A | 待测 | 待测 | 待测 |

## 成本与完整性

训练成本只读取精确token节点的learning/complete计数。输入、padding、教师、主监督、回放分列；缺少的FLOPs、独占GPU时间和中间节点exposure不估算补齐。共享生成准备不重复计入每臂。

| 运行 | 基座原正确旧题保持 | 丢失旧题 | 主／回放监督token | 学生／教师处理token | 训练秒 |
| --- | ---: | ---: | ---: | ---: | ---: |
| development-pd-A-addition/node-200192 | 62.70% | 91 | 200192/50048 | 2931782/6457748 | 1341.7228658040985 |
| development-pd-B-addition/node-200192 | 32.79% | 164 | 200192/50048 | 2975022/6471894 | 1352.041954150889 |
| development-pd-C-addition/node-200192 | 54.92% | 110 | 200192/50048 | 3544660/7955434 | 1610.0976533510257 |
| development-pd-A-addition/node-400384 | 63.11% | 90 | 400384/100096 | 5863280/12915181 | 3214.7659694261383 |
| development-pd-B-addition/node-400384 | 33.20% | 163 | 400384/100096 | 5950243/12944825 | 3243.647845939966 |
| development-pd-C-addition/node-400384 | 53.28% | 114 | 400384/100096 | 7087155/15906074 | 3741.0412024960387 |
| formal-pd-A-addition/node-800768 | 53.91% | 112 | 800768/200192 | 11733072/25893532 | 5876.455298872897 |
| formal-pd-A-rewrite/node-200192 | 41.56% | 142 | 200192/50048 | 2356038/3894248 | 1156.7120619558264 |
| formal-pd-B-addition/node-800768 | 61.32% | 94 | 800768/200192 | 11981038/26137605 | 6401.1425317840185 |
| formal-pd-B-rewrite/node-200192 | 47.74% | 127 | 200192/50048 | 2356038/3894248 | 1130.4728798992 |
| formal-pd-C-addition/node-800768 | 52.67% | 115 | 800768/200192 | 14137345/31748818 | 7370.136897146003 |
| formal-pd-C-rewrite/node-200192 | 55.14% | 109 | 200192/50048 | 2356038/3894248 | 1087.064785968978 |
| formal-sft-A-addition/node-800768 | 60.91% | 95 | 800768/200192 | 11741648/2269855 | 4798.278144327924 |
| formal-sft-A-rewrite/node-200192 | 待测 | 待测 | 200192/50048 | 2356430/567630 | 982.2834885120392 |
| formal-sft-B-addition/node-800768 | 74.90% | 61 | 800768/200192 | 12000372/2269855 | 4835.943369987886 |
| formal-sft-B-rewrite/node-200192 | 待测 | 待测 | 200192/50048 | 2356430/567630 | 972.3222824239638 |
| formal-sft-C-addition/node-800768 | 74.07% | 63 | 800768/200192 | 14097244/2269855 | 5623.324709119042 |
| formal-sft-C-rewrite/node-200192 | 待测 | 待测 | None/None | None/None | None |
| natural-pd-A-addition/node-800768 | 待测 | 待测 | None/None | None/None | None |
| natural-pd-B-addition/node-800768 | 待测 | 待测 | None/None | None/None | None |
| natural-pd-C-addition/node-800768 | 待测 | 待测 | None/None | None/None | None |
| natural-sft-A-addition/node-800768 | 待测 | 待测 | None/None | None/None | None |
| natural-sft-B-addition/node-800768 | 待测 | 待测 | None/None | None/None | None |
| natural-sft-C-addition/node-800768 | 待测 | 待测 | None/None | None/None | None |

表内None表示该精确节点成本未知；重载、评测和共享生成时间另存，不混入训练秒。
