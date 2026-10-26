# Loop历史：按问题查重

更新：2026-10-22。此页回答“以前问过什么、得到什么、现在怎样用”，不维护另一份计划。当前优先级与授权只看[Loop路线图](loop-roadmap.md)。先定位相关行，再读链接原文；无需每次通读全部历史。

## 1. 已研究的问题

| 问题／实际改变 | 已有证据 | 对当前判断的作用 |
| --- | --- | --- |
| 参数共享与交接形式：普通Loop、DiscoLoop、独立层加词表交接 | Wang两跳任务中，共享＋词表交接OO95.66%，独立层＋词表交接6.19%；[比较](results.md#untied-discoloop-development-v1) | 支持该配方中的二者配合；不能外推成所有任务的必要条件，也不是新idea |
| 对齐能否替代交接：自预测方向、原标签方向、中间CE及各自KV组合 | OI部分改善伴随其他池下降；两批KV融合均未超过对齐单独。[自解码](results.md#self-decode-alignment-development-v1)／[原标签](results.md#supervised-alignment-development-v1) | 保留正负结果；两类损失与KV范围不同，不合成一个单因素结论 |
| 旧KV能否改善组合与顺序新知识使用 | 当前知识任务扩展未建立稳定新增知识收益，用户已结束配方扩展。[顺序比较](results.md#loop-kv-learning-v1)／[残差缓存](results.md#residual-cache-comparison-v1) | 重访须有足以改变旧判断的新条件；不能换组合或名字继续旧网格 |
| 门放哪里、读什么：注入门、连续更新门、共同门、融合候选门、逐维门 | P在300k有三跳收益，其他参数化不一致；后期会有低谷和答案破坏。[门系列](results.md#dual-gate-multihop-v1)／[候选门](results.md#candidate-update-gate-multihop-v1) | 门控已做大量比较；“增加一个门”本身不是新突破 |
| 为什么训练高低波动：门交换、模块干预、状态与候选记录 | 旧P与新invariant都存在形成失败和后续覆盖；单换门或衰减不能解释所有低谷。[P诊断](results.md#p-gate-root-cause-research-20261018)／[invariant诊断](results.md#invariant-checkpoint-dip-20261022) | 局部修复有证据，统一根因未建立；固定删除某轮MLP会伤其他题型 |
| 用稳定性规则控制时刻：解码一致、依赖闭合、门时刻表移植 | 一致性规则5/5未通过，依赖闭合仅1/5通过；移植可改善旧模型部分池。[规则](results.md#loop-decode-stability-gate-test-20261019)／[依赖](results.md#loop-readiness-gate-test-20261019)／[移植](results.md#loop-gate-timetable-intervention-20261019) | 解码稳定不等于答案正确或推理就绪；不能直接当部署规则 |
| 历史归一化聚合与接收分数：native四臂 | 原世界终点invariant强于ema/content/加性，但低谷、长循环退化仍在。[当前证据](loop-roadmap.md#evidence) | 继续检验的是重复性与直接状态门；不能把invariant名称当作数据无关性证明 |
| 第二跳怎样读取：局部attention干预与统一温度 | 候选可明显改善，完整四跳改善有限。[读取](results.md#invariant-second-hop-20261022)／[温度](results.md#invariant-attention-temperature-20261022) | 读取接口有作用，后续保持及其他跳仍待解释；未改当前训练配方 |
| 五跳从哪一步失败：原生逐阶段轨迹、单轮桥路由与原子调用供体 | B五跳99.14%前16轮从未正确；R5强制读原生桥使A19.24→94.98%、B0.66→8.90%；完整原子供体B当轮99.80%、R16为61.64%。[诊断](results.md#invariant-five-hop-diagnosis-20261025) | A有强路由证据，B未被该路由干预充分恢复，完整供体后仍有保持困难；完整供体可能已含答案，不能归成一个模块根因 |
| DiscoLoop是否是同一种五跳故障：同轨迹与同六类干预 | RoPE B99.88%从未正确、原生后续丢失0%；R5单源路由几乎无效，R4规范真桥31.98%；完整原子结果从当轮100%保持到R16的99.98%。旧绝对位置版第四桥已几乎不可读。[比较](results.md#disco-five-hop-diagnosis-20261025) | 共同失败在可读第五答案形成前，但干预时机、上游桥和供体保持不同；不能据相似最终分数归为同一机制或换数据敏感性 |
| RoPE第5轮是丢关系还是接收状态不再接受桥 | 同一原子attention下，B原残差R4/R5 raw31.98/0.10%，缩放或清零后均100%；原生attention配缩放残差仍0%。invariant纯桥＋缩放接收输入在配对末关系池B78.66%，443/500前缀可答多个不同后继。[拆分诊断](results.md#loop-receiver-attention-20261025) | RoPE旧残差干扰可逆，但原生读取亦不足，未证明关系彻底丢失；invariant进入块的h+e保留可用关系、形成后仍会丢。全头单源桥V的RoPE fresh正控失败已保留，不能用该负结果作关系丢失证据 |
| 模型深度与数据支持：旧4×2成功配方、1–4跳新任务、层轮网格 | 多个数据与计算条件同时改变；300k八组原子接近满分而组合弱，不能作单因素归因。[多跳](results.md#loop-hop-unified-v1)／[旧层轮](results.md#loop-block-depth-v1) | 长期可能晚学会；工程未开训、20k停止、科学负结果分别记录 |
| 新知识写入能否接入旧推理：TTCD、MLP原子CE、连续写入与保护 | 单事实CE可组合；连续写入原配方严重遗忘，邻居保护下DiscoLoop新事实100%、双新98.44%、旧组合保持96.99%。[单事实](results.md#ce-architecture-novel-v1)／[连续](results.md#ce-cross-sample-double-new-v1)／[保护](results.md#neighbor-protection-development-v1) | 与知识主线共享证据；这些结果不等于当前多跳模型已具备更新传播，也不自动重开写入网格 |

<a id="position-depth"></a>
## 2. 深度与位置：三个问题分开

| 实际问题 | 已有结果 | 已被后续证据修正的说法 |
| --- | --- | --- |
| 同一题增加循环次数 | 旧无门模型R4–R16基本不变；新invariant四跳R8→R32小幅上升，R256下降，固定子集更深后平台。[旧](results.md#loop-round-extrapolation-diagnostic-20261019)／[新](results.md#loop-invariant-robustness-v1) | “所有模型多算完全没用”已不成立；“多算一直改善”也无证据 |
| 回答更多跳的问题 | 旧5–8跳测试受答案位置影响；同权重无padding位置平移已使invariant B四跳82.18→0.28%。插值恢复content A五跳至44.60%，仍未恢复各组B长跳。[位置干预](results.md#loop-position-controlled-long-hops-20261025) | 位置输入对组合有直接影响；去位置破坏短组合，替换位置未恢复B长跳，不能据此宣称位置无关或存在四跳硬上限 |
| 换位置编码是否解决问题 | position三臂R8 EMA的A四跳／B三跳／B四跳：ema-rope为4.64／63.98／0%，disco-rope-old为99.96／88.86／45.14%，ema-absolute-old为100／25.84／25.96%。新统一前缀R16下RoPE DiscoLoop的A/B四跳99.92/44.92%，五跳0.22/0.12%，六至八跳未恢复；原形状重放逐token一致。[旧证据](reports/loop-evidence-history-20261022.md#evidence)／[长跳补测](results.md#invariant-five-hop-diagnosis-20261025) | “换成RoPE即可解决”未成立；共同长跳失败不等于排除所有位置效应，也不等于检验换数据敏感性 |

旧“1/r冻结”是特定加性交接轨迹的相对更新观察，不是通用失败定理。[原测量](results.md#additive-carry-freezing-20261020)

## 3. 旧方案与执行状态怎样读

- **已完成的科学比较：** 以各批完成清单为准，保留独立重载、云端核验是否通过的区别，不重复训练。native、position、两批对齐作为旧对照复用。
- **仍在收齐的当前比较：** 重复性／换数据、普通Loop／基础DiscoLoop基线、state-mlp；全部由[当前路线图](loop-roadmap.md#current)登记。历史迁移顺序与GPU编号不作新的启动依据。
- **只准备、工程中止或未补验终态：** 例如norm-handoff、旧深度支持增补、规则门与ListOps的历史启动段落；按[原记录](reports/loop-evidence-history-20261022.md#evidence-records)查状态，不默认仍运行、已完成或已取消。文档归档不改变原授权或控制器。
- **原契约停止与解释撤回：** 六臂20k pilot未过原门槛，正式矩阵未启动；“为前缀一致必须末轮加反馈”的解释已撤回。[语义更正](loop_semantics_audit.md)
- **历史候选：** 旧报告里的优先级、MLP单独门、旧历史状态门、规则、课程、停止等建议均属于当时讨论。与当前实现是否相同须查公式和协议，不从关键词推断已做或未做。

## 4. 查证入口与维护

[完整结果与入口档案](reports/loop-evidence-history-20261022.md)保留此次移出的原文、完整表格、失败、旧预算及核验状态；[阶段PDF](reports/loop-transformer-summary-20261021.pdf)是10-21历史快照。相关详细报告按需打开：[门设计](reports/loop-history-gate-design-review.md)、[旧残差诊断](reports/residual-gate-architecture-and-diagnosis.md)、[深度文献与规则诊断](reports/loop-depth-extrapolation-gate-stability-survey-20261019.md)、[native设计评审](reports/loop-native-handoff-review-20261020.md)。旧PDF与冻结协议原样保留。

新结果进入结果汇总；只有改变当前判断时才更新路线图与本页对应行。调度、PID、检查数量、失败重试写回批次记录，入口保留恢复链接。文档头部不再累加逐时播报；提出新候选前核对对应历史，已有问题未解决不等于新问题。
