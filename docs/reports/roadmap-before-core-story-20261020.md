> 历史快照：2026-10-20 改写为“知识、推理与连接”核心故事之前的 roadmap 原文，仅供追溯，不是当前规划。当前规划见 [roadmap](../roadmap.md)。

# 研究路线：让Transformer持续积累可复用的知识

更新：2026-10-19（America/Chicago）。本文是唯一的当前研究规划；结果、推导与执行记录分别维护在[结果汇总](../results.md)、[理论依据](../memory-scaling-theory.md)和各批实验协议中。

<a id="mission-20261019"></a>
<a id="question"></a>
## 1. 使命、愿景与论文问题

**使命：理解Transformer怎样把学到的事实变成可以灵活使用、持续积累和可靠更新的知识，并找到实现这种能力的条件。**

愿景是：模型学会新事实后，已经掌握的问答方式与推理操作能够使用它；事实改变后，相关判断随之改变；继续学习时，无关的已有能力能够保留。希望减少为每种问法、每条组合后果重新提供答案监督的需要。这里讨论已学任务与操作内的迁移，不要求新事实自动产生尚未学过的技能。

**本文聚焦：事实的学习，能在多大程度上与具体用途的学习分开？这种分离需要什么条件，又为什么会在知识更新时失效？**

使命决定选题，文献帮助理解已有进展。从第一性原理推导出已有方法可以有价值：需要说明它为什么成立、依赖什么条件、能解释哪些观察。与已有论文重合不是放弃重要问题的理由，也不能据此声称新的方法或定理。解释整合、条件分析和有依据的设计原则都可以构成研究推进。

近期的[参数知识访问研究](https://arxiv.org/html/2602.22193v1)、[Engram](https://arxiv.org/abs/2601.07372)和2026年10月的预印本[EngramEdit](https://arxiv.org/abs/2610.10533)，分别研究知识访问、记忆与计算分工、事实更新及后续使用。它们提供选题背景，不指定本项目必须采用哪种架构。

## 2. 三个层次与已有结论

三个层次各有独立实验和解释。第一层不是组合实验的掌握率检查；第三层同时包括新增、改写与后续保持。下表直接复用已完成证据，不重新训练以确认同一现象。各结论限于原实验条件，不将跨批次分数差当作配对因果效应。

<a id="recall"></a>
### 第一层：知识怎样被学会并回答

**问题：同一事实怎样从训练中的表达，变成不同问法都能提取的知识？**

| 已有证据 | 可以采用的结论 | 解释边界 |
| --- | --- | --- |
| [bioS表达组织](../results.md#bios)：规范QA为5.07/35.65/99.96% | 训练表达与组织影响之后的知识提取 | token数等也有变化，不能唯一归因于某一种表示机制 |
| [同权重的提取与规则使用](../results.md#bios-same-weight) | 同一模型在不同提取和使用方式下有明显差异；分步提取可以改善使用 | 分步增加输入与计算；不能把调用变化直接解释为主体参数里的知识被擦除 |
| [顺序注入后的问法迁移](../results.md#sequential-phrasing-20261011)：B原问法100%，留出问法37.61–48.62% | 原问法学会不保证换问法可答 | 留出表达同时改变词汇与指令格式，不能直接叫遗忘 |

需要解释的是表达、事实学习与提取行为怎样联系。已有知识量和曝光曲线可作补充，固定预算低分不直接当作存储容量上限。

<a id="composition"></a>
### 第二层：知识怎样被组合

**问题：已经学会的事实，为什么有时不能进入已掌握的推理操作？**

| 已有证据 | 可以采用的结论 | 解释边界 |
| --- | --- | --- |
| [组合使用经历](../results.md#grok-usage)：有组合经历100%，匹配原子重复25.01% | 使用经历的作用超过简单重复事实 | 组合练习同时改变上下文、监督和优化 |
| [小型共享模型的表示训练](../results.md#representation-alignment)：桥实体都可解码，对齐使分布外组合12.60→92.32% | 可解码与可使用不同，训练可以改善后续使用 | 改变了多个模块，不能只归因于MLP |
| [普通两层模型复核](../results.md#independent-alignment-v1)：对齐后余弦.9997，分布外组合5.27% | 表示方向接近不足以保证组合，收益有架构条件 | 与共享模型跨批次、参数量和图不同，不能据差值唯一归因于共享 |

解释须容纳标准模型的成功条件和循环模型的失败条件。直接回答、模型自行分步回答、中间解码与因果干预分别提供不同证据，不能相互代替。

<a id="updating"></a>
<a id="sequential-transfer"></a>
<a id="write-interface"></a>
### 第三层：知识变化后怎样继续使用

**问题：新事实怎样接入已有用法，并在继续学习时与旧能力一起保留？**

| 已有证据 | 可以采用的结论 | 解释边界 |
| --- | --- | --- |
| [写入目标](../results.md#write-target-editing-v1)：同样24/24原子写入成功，组合传播3.13/68.75/85.42% | 写入目标影响后续可用性；改对最终答案与让后续计算用上新事实不同 | 传播改善伴随旧组合损伤，不宣称兼顾新旧能力 |
| [同次联合编辑的权重恢复](../results.md#edit-restoration-20261011)：仅保留第二轮改动可传播78.91%，原只优化第二轮为0% | 优化没有找到有效更新，不等于该执行阶段无法承载更新 | 同一图与初始化上的事后机制证据，不是普遍定理 |
| [连续写入消融](../results.md#ce-cross-sample-ablation-v1)：低学习率Loop新原子82.03%、双新组合0%；取消截断DiscoLoop双新95.31%、旧组合保持80.93% | 保住原子不自动恢复组合；写入约束会影响新旧能力取舍 | 两架构的基础能力与成本不同，不能把全部差异归因于架构 |

新增事实与改写旧事实分开解释，后者还涉及旧答案竞争。[顺序知识注入](../results.md#sequential-transfer-v1)、[旧组合回放](../results.md#sequential-replay-v1)、[知识变化中学习调用](../results.md#knowledge-change-v1)均已完成，直接作为证据，不换名再做。自然模型的[编辑复现与跳位分析](../results.md#path-hop-analysis-v1)保留其不一致结果，不将它们改写为统一机制的支持。

<a id="theory"></a>
<a id="path-consistency"></a>
## 3. 需要建立的解释

论文连接三个对象：**事实以什么情境被训练；完整计算怎样使用它；实际参数更新改变了哪些行为。** 研究MLP的作用，同时保留embedding、attention、归一化、残差以及各模块的协作，不预设事实只存于某一个模块。

| 解释问题 | 现成证据的作用 | 还不能直接断言的内容 |
| --- | --- | --- |
| 同一事实在不同用途中的计算是否兼容 | 问法迁移、表示对齐、分步回答与状态干预约束解释 | 同一个向量或同一组参数是所有用途复用的必要条件 |
| 写入能否影响后续真正使用的计算 | 写入目标和同次更新恢复区分行为成功、优化过程与承载位置 | 每一轮对应一跳；最终原子成功意味着中间计算已更新 |
| 使用改善与无关能力损伤是否可分离 | 配对编辑、回放和连续学习展示实际收益与代价 | 某配方的取舍是不可避免的能力上限 |

路径一致性是一种候选解释：新的用途需要有效访问训练或编辑所改变的内容，并以兼容状态接入后续计算；完整路径不必逐项相同。[Z1的原冻结判读为partial](../results.md#z1-path-overlap-20261011)，自然模型也未建立统一路径机制。因此不把它预设为唯一主线或已成立结论。

数学从完整Transformer的计算图出发。局部关系`Δf(q) ≈ J(q)Δθ`可以解释实际更新对不同查询的影响，不能代替长期学习证明；共享参数改变更新耦合，不保证不同调用都受益。推导及条件见[实际计算与更新](../memory-scaling-theory.md#transformer-conditional-results)、[跨调用关系](../memory-scaling-theory.md#loop-cross-call-kernel)、[功能复用](../memory-scaling-theory.md#functional-reuse-20261004)。数学结论、数值校验、实验观察与机制假说分别标明。

<a id="next-experiments"></a>
<a id="read-interface"></a>
## 4. 当前主线：注入的知识为什么用不起来——知识调用接口

**用户提出的主线（2026-10-19）。** 训练好的模型除了存储知识，还有检索和运用知识的功能。直接向训练好的模型注入知识（上下文蒸馏、MLP down投影编辑等）可以让事实被答出，却常常缺乏有效运用这些知识的能力。目标是找出并定义这一调用功能，再据此提出轻量复用注入知识的算法或架构。它是§1问题在预训练模型上的具体形式：事实学习能否与用途学习分开，取决于新事实能否接到已有的调用功能上。

**可检验的定义。** 用因果干预定义两个对象，不预设它们是一个干净的区域：

- **读取接口：** 哪些token位置与层段上的实体表示，一旦换成另一实体的表示，所有用途（问答、换问法、作第一跳、CoT）都改用后者的属性。用模型原有的自然事实标定。
- **读取器：** 接口之后跨事实、跨用途都必需的组件。消融后使用失败，而移植回完整模型可确认接口内容未变。

判据是跨事实通用、跨用途共享、因果必要且充分。它可能是分布式的；[Z1](../results.md#z1-path-overlap-20261011)在小模型上没有建立独立的查表回路。Hase等2023表明定位结果不一定指示最佳编辑位置，所以最终以写入后的端到端使用为准。理论上对应[式31](../memory-scaling-theory.md#untied-second-hop-kernel-20261010)：读取器是下游Jacobian，地址是被写参数的输入激活；某个用途能读到一次写入，要求两者都与写入重合。

**“缺乏运用能力”拆成三种可区分的失败，分别对应不同的算法：**

| 假说 | 含义 | 对应算法 | 已有证据（均为待检验的线索） |
| --- | --- | --- | --- |
| A 读取器缺失 | 调用功能需要为新知识重新学习 | 一次训练、跨事实共享的读取适配器 | Zucchet等2025回忆电路先于存储形成；Balesni等（2411.16353 v4）两条合成事实即使分层微调仍为机会水平；[知识变化中学习调用](../results.md#knowledge-change-v1)收益只来自一个图 |
| B 地址错位 | 通用调用功能已存在，注入写到了它不读的位置或格式 | 写到接口的轻量写入，或把知识搬回接口的转接 | Physics 3.1名人带动少数群体；Mehri & Schütze（2610.22300）；[写入目标](../results.md#write-target-editing-v1)早期写入使传播3.13→68.75%且读取器未重训；Dai等（2607.08393）后层存储与中层计算错位 |
| C 调用次序 | 单遍前向只调用一次读取器，第一跳结果来不及被再次调用 | 重复调用读取器：CoT、Loop、解码再编码 | Hopping too late；He等（2601.04600）；[TTCD新增事实](../results.md#ttcd-novel-fact-transfer-v1)分步251/251、内部两跳0/251；[DiscoLoop独立写入两条新事实](../results.md#ce-cross-sample-double-new-v1)组合85.94% |

**范围。** 第一步只做**已知实体的新增事实**，不覆盖旧值；全新实体与改写旧事实放第二步。主模型为Qwen3-1.7B（标准Transformer）和Qwen3.5-2B（Gated DeltaNet与全注意力3:1混合，24层），两者分别标定，不据此作架构优劣结论。

| 阶段 | 内容 | 新增训练 | 完成标准 |
| --- | --- | --- | --- |
| 0 标定 | 在自然事实上找接口层段与读取器；筛选模型可靠掌握的实体、桥事实和两跳关系 | 0 | 两个模型各有接口与读取器的因果证据；数据前提通过 |
| 1 注入 | 同一批新增事实，用陈述SFT、多样数据SFT、上下文蒸馏、MEMIT、仅down投影微调注入；测原问法、换问法、新事实作第一跳/第二跳（直接与CoT）、上下文上限及保持 | 小规模微调 | 原子写入达标，或如实记录未达标 |
| 2 移植 | 注入模型与原模型之间双向移植接口表示，把每个“方法×用途”的失败按预定规则归入A/B/C | 0 | 归类规则在看结果前固定 |
| 3 算法 | 按诊断选择：B写到接口；C重复调用读取器；A在留出的注入事实上检验共享读取适配器 | 另行授权 | 匹配数据与计算，计入保持 |

阶段0–2的协议见[知识调用接口开发协议](../development-artifacts/knowledge-read-interface-development-v1/protocol.md)，已于2026-10-20完成，[结果](../results.md#knowledge-read-interface-development-v1)在第二事实集上复现。

- **梯度注入绕开读取接口。** M1/M2/M5/M3都没有把新事实写进主语表示（正向移植≈0），而是写在主语下游。按预定规则归类以B（地址错位）为主：在桥位置补上显式主语状态，注入模型就能用上新事实。但关系是虚构的，原模型是否有读取它的能力未经检验，T1低同样符合A；这一归类目前只作“T1低、T2高”的描述。
- **写进主语表示就能被读取。** MEMIT写进了主语表示，原模型读取器能把它当第一跳使用；但它绑定编辑语境，且大批量写入不成立。

[后续批](../results.md#knowledge-read-interface-followup-v1)（2026-10-20，[协议](../development-artifacts/knowledge-read-interface-followup-v1/protocol.md)）进一步收窄了这两条：

- **U3不是“桥形成太晚”。** 直接回答时，U3提示中的桥位置从未形成能触发新事实的h（两个模型、两组事实、五种方法全部如此），回补也救不回。U3需要让桥实体显式出现。
- **只改写入层段不够。** 把梯度写入限制在接口层段的MLP，事实进入了主语位置，但它的读取还依赖其它位置被一并改动的状态，原读取器读不出（T1≤.04），第一跳用途没有改善；第二事实集复现。下游段写入则如预期完全在下游。

单种子开发证据，尚未区分A（读取器缺失）与B。

<a id="read-interface-v2"></a>
**2026-10-20修正：[知识调用接口v2计划](../development-artifacts/knowledge-read-interface-v2/protocol.md)（同日授权并完成，结果见下文）。** 对已有生成做零训练重新分析（[结果](../results.md#knowledge-read-interface-v2-reanalysis)），发现U2失败分两种：
- 接口段写入时，事实被触发，但“在哪国”的操作被覆盖，模型直接答城市v。
- 下游段写入时，操作保留，但事实不被触发，模型答主语自己的国家。

两臂只差训练层段，这一翻转在两个模型、两组事实上都成立。A/B/C没有“操作被覆盖”这一类，因此改用四个环节分解一次使用：关系键、触发、交接、操作保持。

工作假说是：单跳监督不能确定写入的键覆盖与值形式，写入的参数形式和位置决定梯度落到“实体解”还是“答案解”。计划依次为：
- **R：** 测量修正，包括干扰上下文上限和扩充的自然参照。
- **E1：** 只改主语状态的最优写入上限，检验B的前提，也为“只在主语位置写入”给出上限。原候选“只在主语位置施加写入增量”因此暂缓。
- **E2：** 已有关系，按E1结果决定是否启动。
- **E3：** 新增通道与低秩增量的写入形式对照，复现[Cortex](https://arxiv.org/abs/2610.20399)的关键比较。
- **E4：** 用途练习能否迁移到没练过的操作，以及只有单跳训练的事实。

已有小模型证据只作线索，不在小模型上重做。相关工作：Geva等2023的回忆三步分解、ROME/MEMIT因果追踪、Dai等2026、O'Neill等（2607.11020）、User as Engram（2606.19172）、Zhang等ICLR 2025的Editing Overfit、Cortex（2610.20399）。

可主张的贡献是：
- 用四环节分解与受控对照解释注入事实为什么用不上；
- 检验写入形式与用途迁移；
- 由诊断推出、能在留出注入事实上复用的轻量算法。

现象本身与“存储/读取分离”的说法不是新贡献。

**v2结果（2026-10-20，[结果](../results.md#knowledge-read-interface-v2)）：**
- **用途练习跨事实迁移（E4）。** 只在一半事实上练习“大洲”“官方语言”两个操作，另一半从未练习的事实在“在哪国”上也能用：分步回答比数据量对照高.39（Qwen3-1.7B）与.70（Qwen3.5-2B，开发集），第二事实集一致；直接回答的增益小（0–.19）。练习修复的主要是操作覆盖（G3），不是直接组合。
- **主语状态写入（E1）不答v，但读不全。** 四组都没有达到“读取可用”：h在前的留出问法只有.12–.41。Qwen3.5-2B换成已有关系（E2，用户选定(a)）后h在前提高到.48–.67、混答减少，第一跳仍失败；Qwen3-1.7B两类关系无差别。
- **写入形式（E3）没有复现Cortex。** 新增通道与低秩增量在第一跳上无稳定差别，新增通道略多答v。但Qwen3.5-2B在这一协议下（新参数、逐条写入、各层隐藏状态对齐、学习率1.5e-4）写在前段，第一跳U2直接达到.35–.58，是本主线首次；Qwen3-1.7B（学习率1.5e-3）写在后段、为答案形式。
- **测量修正。** 加三条同关系干扰后，上下文中的直接U3由.70/.84降到.35/.39；R3按Biran等筛选后每类只剩6–39条，未达50条。

**下一步候选（均需另行授权）：**
- 在Qwen3-1.7B上以学习率1.5e-4重跑E3两臂，区分“写入位置随学习率变化”与模型差别；再在Qwen3.5-2B上对E3的四个成分（新参数、隐藏状态对齐、逐条写入、学习率）做消融，找出让写入可作第一跳的那一项。
- 协议§5的算法分支“E4跨事实迁移”已触发：把少量事实上的用途练习与E3式写入结合，在F_test上比较四个环节、保持与成本。

<a id="loop-native-handoff"></a>
**2026-10-20：Loop架构评审与原生交接方案（用户要求；训练未授权）。** 对照[InfiLoop](https://arxiv.org/abs/2610.11570)评审现有Loop设计，见[评审报告](loop-native-handoff-review-20261020.md)。
- **算法结构从哪里进入。** 主线损失只监督答案。算法结构来自三处：10-12两批的辅助损失（中间CE OI 7.03%，已撤除）；"前缀即原子问句"的模板，DiscoLoop通道依赖它；固定R=4。
- **不稳定的来源。** 主要是图B等不受监督的行为在漂移。学习门能读实体身份，因此学成依赖图的时刻表。
- **零训练再分析。** 超过训练轮数后，四类模型每轮的相对更新都约按1/r衰减（[结果](../results.md#additive-carry-freezing-20261020)）。
- **新方案。** 以InfiLoop为骨架：归一化凸组合交接、只监督答案的深监督与随机轮数、EMA；保留DiscoLoop解码—再编码作为候选成分；新增对实体身份不变的接收分数。

[批次协议](../development-artifacts/loop-native-handoff-development-v1/protocol.md)为四臂各150万步，已完成代码、30项契约测试与CPU冒烟。它属于§6排除的架构配方扩展，由用户这次的要求重新打开。2026-10-20用户授权训练，04:05 UTC在GPU 6–9提交（smoke→train→grid）；尚无科学结果。

按三个层次编排已有证据、连接理论的论文写作继续进行，不占用新实验预算。当前不指定某一架构或算法为必做方案；投影保护、表示对齐、共享计算和解码再编码都是按诊断选用的工具。

新增实验的判断顺序为：**问题的重要性 → 解释能否被检验 → 现有证据是否已经足够 → 资源代价是否合理。** 不另建内部特征预测器，也不以扩大模型、种子或超参数网格代替机制判断。文献用于理解、借鉴和准确归属，不再默认把“先复现另一篇论文”作为每次推进的前提。

本次重构的新增科学训练与编辑预算为0。新研究方向不等于新增执行授权；已经提交的批次按自己的协议收尾，不因本文件重写而取消、重启或更改。

<a id="main-setup"></a>
<a id="evaluation"></a>
<a id="efficiency-evaluation"></a>
## 5. 数据、评价与执行约束

第一层使用已有bioS与问法实验，组合与更新优先复用已有Wang知识图、检查点及兼容对照；小模型、2Wiki和自然模型结果各自写明范围。同一关键比较统一数据、初始化规则、监督与评价，变化因素单列。不同设定之间不直接相减分数，也不为强行统一历史设定而重训。

- **行为评价：** 全量结果为主，同时保留前提子集及覆盖率。分别报告直接回答、组合后果、无关原子与组合保持；旧正确题的丢失不能被新增正确抵消。新增与改写、隐式组合与显式分步、已训练长度与长度外推分开。
- **前提核查：** 事实是否学会或写成、必要后继是否具备、任务操作是否已掌握分别核查；未达标样本保留，失败不自动归因于组合机制。
- **成本与统计：** 曝光、token、更新数、训练/推理计算、总/激活参数和耗时分别记录，第三层同时计旧能力保持。同一图的种子、不同问法与重复测量不当独立世界；开发与确认分开。
- **执行与恢复：** 依[实验协议](../experimental-protocol.md)和[项目入口](../../AGENTS.md)核对冻结配置、控制器、执行锁及实际LM Docker进程。先读该批`controller-process.json`、`controller-state.json`及调度修订，不按历史GPU表启动。训练完成、独立重载、云端核验分别确认。
- **追踪：** 新批默认使用[W&B团队项目](https://wandb.ai/zsqzz/llm-memory-editability)及[默认配置](../../configs/experiment-tracking-defaults.json)；原批身份和契约不追溯修改。

本次是文档重构，没有实时核查任务进度；[结果汇总](../results.md)提供批次入口，实际状态以对应控制器和进程为准。

<a id="scope"></a>
<a id="history-dedup"></a>
## 6. 范围与避免重复

当前研究以参数知识为主要证据，不预设知识必须只存在于稠密权重，也不认为外部存储自动解决了知识使用。DiscoLoop是已有实例，门控及其配方不再承担论文的中心使命。

以下内容不进入默认的新实验队列：

- 已完成的表达/提取、原子重复/组合经历、表示对齐及逐轮编辑范围比较。
- 已完成的顺序注入、旧组合回放、知识变化训练、学习率/Adam/截断消融。
- DiscoLoop门控、KV、mHC、MoE及其它架构的配方扩展；跳数、宽度、循环次数的大网格。
- 容量幂律、bioS 100k主矩阵、日期更新、少迭代蒸馏及独立TTT论文等旧草案；它们不是当前承诺。

新比较先核对[结果汇总](../results.md)及[历史去重记录](roadmap-before-restructure-20261019.md#history-dedup)，说明原批结论、关键变量与新增问题。用户的逐项评价按原记录解释，未回复不视为认可。负结果限于原条件；有新的机制依据时可以重新研究，但不能仅换数据呈现、架构组合或名称延续同一失败问题。

<a id="glossary"></a>
## 7. 核心术语

| 术语 | 本文含义 |
| --- | --- |
| 原子事实 | 单条关系事实`(h, r, t)`；模型学习后通过查询提取 |
| 组合与桥实体 | 两跳`(h, r₁, r₂)`先得到桥实体b，再由`(b, r₂)`得到答案；事实跳数不同于模型层数或循环轮数 |
| ID/OOD与II/IO/OI/OO | 在相应Wang批次中，ID事实参加过训练组合，OOD事实只有原子训练；两个字母依次表示两跳事实的类别。不同批次以原协议为准 |
| 普通Transformer / Loop L×T | 前者各层参数独立；后者将L层共享块执行T次。复制初值对照不等同通常的独立初始化基线 |
| 写入情境 / 使用方式 | 事实以何种样本被学习或编辑 / 之后在哪类查询或计算中被调用 |
| 新增 / 改写 / 顺序学习 | 添加原来没有的事实 / 改变已有键的事实值 / 连续接收知识并检验早先知识保持 |
| E / D / R / U | 目标事实 / 应随之改变的后果 / 回放材料 / 不参与选择的保持评价；保护用题与留出保持题分开 |
| 分步回答 | 用模型自己的第一步答案构造下一步查询；需计入额外输入与计算，不等同内部组合 |
| 记忆状态与TTT | 参数、快权重、隐藏状态、KV与外部记忆分别说明。测试时训练是否跨样本保留状态由具体方法规定 |

<a id="literature"></a>
<a id="maintenance"></a>
## 8. 文档与历史入口

| 入口 | 用途 |
| --- | --- |
| 本路线图 | 当前使命、问题、证据用途、推进顺序与范围 |
| [结果汇总](../results.md) | 完成结果、条件、反例及批次状态入口 |
| [理论依据](../memory-scaling-theory.md) | 完整推导、方法依据、假设与边界 |
| [实验协议](../experimental-protocol.md) | 共同规则与历史脚本所依赖的冻结契约 |
| [重构前历史记录](roadmap-before-restructure-20261019.md) | 旧讨论、未采用方案与调度记录，仅供追溯，不作为另一份规划 |

README只作导航。今后直接修改对应主题，不在顶部累加互相覆盖的“当前方案”。科学数据、权重、日志、原协议和失败记录不因本次整理改变。

<!-- Historical anchor redirects are generated below to preserve existing citations. -->

<details>
<summary>旧链接兼容入口（历史内容，不是当前待办）</summary>

| 旧主题或批次 | 记录 |
| --- | --- |
| <a id="first-principles-plan-20261019"></a>第一性原理旧提案与文献复核 | [历史原文](roadmap-before-restructure-20261019.md#first-principles-plan-20261019) |
| <a id="sequence-dependency-gate-v1"></a>sequence-dependency-gate-v1 | [历史原文](roadmap-before-restructure-20261019.md#sequence-dependency-gate-v1) |
| <a id="residual-gate-injection-ablation-v1"></a>residual-gate-injection-ablation-v1 | [历史原文](roadmap-before-restructure-20261019.md#residual-gate-injection-ablation-v1) |
| <a id="candidate-update-gate-gpu1-v3"></a>candidate-update-gate-gpu1-v3 | [历史原文](roadmap-before-restructure-20261019.md#candidate-update-gate-gpu1-v3) |
| <a id="compact-wang-joint-gate-v1"></a>compact-wang-joint-gate-v1 | [历史原文](roadmap-before-restructure-20261019.md#compact-wang-joint-gate-v1) |
| <a id="candidate-update-gate-multihop-v1"></a>candidate-update-gate-multihop-v1 | [历史原文](roadmap-before-restructure-20261019.md#candidate-update-gate-multihop-v1) |
| <a id="joint-update-gate-wang-v1"></a>joint-update-gate-wang-v1 | [历史原文](roadmap-before-restructure-20261019.md#joint-update-gate-wang-v1) |
| <a id="joint-update-gate-multihop-v1"></a>joint-update-gate-multihop-v1 | [历史原文](roadmap-before-restructure-20261019.md#joint-update-gate-multihop-v1) |
| <a id="gate-generalization-extension-v1"></a>gate-generalization-extension-v1 | [历史原文](roadmap-before-restructure-20261019.md#gate-generalization-extension-v1) |
| <a id="paper-update-gate-continuation-v1"></a>paper-update-gate-continuation-v1 | [历史原文](roadmap-before-restructure-20261019.md#paper-update-gate-continuation-v1) |
| <a id="paper-update-gate-multihop-u95-v1"></a>paper-update-gate-multihop-u95-v1 | [历史原文](roadmap-before-restructure-20261019.md#paper-update-gate-multihop-u95-v1) |
| <a id="paper-update-gate-multihop-v1"></a>paper-update-gate-multihop-v1 | [历史原文](roadmap-before-restructure-20261019.md#paper-update-gate-multihop-v1) |
| <a id="gate-evidence-refinement-20261017"></a>门控证据整理历史 | [历史原文](roadmap-before-restructure-20261019.md#gate-evidence-refinement-20261017) |
| <a id="gated-loop-depth-extrapolation-v1"></a>gated-loop-depth-extrapolation-v1 | [历史原文](roadmap-before-restructure-20261019.md#gated-loop-depth-extrapolation-v1) |
| <a id="zhang2026"></a>两跳组合的文献分析 | [历史原文](roadmap-before-restructure-20261019.md#zhang2026) |
| <a id="discoloop2026"></a>DiscoLoop文献分析 | [历史原文](roadmap-before-restructure-20261019.md#discoloop2026) |
| <a id="weight-tying-pooling"></a>权重绑定与梯度池化提案 | [历史原文](roadmap-before-restructure-20261019.md#weight-tying-pooling) |
| <a id="feng2025"></a>提取结构文献分析 | [历史原文](roadmap-before-restructure-20261019.md#feng2025) |
| <a id="pretraining-continual"></a>预训练与持续学习旧论述 | [历史原文](roadmap-before-restructure-20261019.md#pretraining-continual) |
| <a id="wang-benchmark-boundaries"></a>旧主设定可行性复核 | [历史原文](roadmap-before-restructure-20261019.md#wang-benchmark-boundaries) |
| <a id="innovation-rnn-ttt"></a>RNN与TTT旧候选 | [历史原文](roadmap-before-restructure-20261019.md#innovation-rnn-ttt) |
| <a id="memory-mosaics"></a>Memory Mosaics阅读记录 | [历史原文](roadmap-before-restructure-20261019.md#memory-mosaics) |
| <a id="memory-reuse-solution"></a>旧方法整合目标 | [历史原文](roadmap-before-restructure-20261019.md#memory-reuse-solution) |
| <a id="dual-gate-design-20261017"></a>双门设计 | [历史原文](roadmap-before-restructure-20261019.md#dual-gate-design-20261017) |
| <a id="norm-handoff-minimal-proposal-20261017"></a>归一化交接提案 | [历史原文](roadmap-before-restructure-20261019.md#norm-handoff-minimal-proposal-20261017) |
| <a id="loop-solution-integration-20261012"></a>Loop方法整合提案 | [历史原文](roadmap-before-restructure-20261019.md#loop-solution-integration-20261012) |
| <a id="paper-reproduction-first"></a>论文复现历史 | [历史原文](roadmap-before-restructure-20261019.md#paper-reproduction-first) |
| <a id="zhang2026-next"></a>10月10日候选实验 | [历史原文](roadmap-before-restructure-20261019.md#zhang2026-next) |
| <a id="nl-edit-layer"></a>自然模型编辑层候选 | [历史原文](roadmap-before-restructure-20261019.md#nl-edit-layer) |
| <a id="history-audit-20261010"></a>历史去重旧清单 | [历史原文](roadmap-before-restructure-20261019.md#history-audit-20261010) |
| <a id="controlled-multihop"></a>组合再学习开销旧提案 | [历史原文](roadmap-before-restructure-20261019.md#controlled-multihop) |
| <a id="architecture-efficiency"></a>架构效率旧比较 | [历史原文](roadmap-before-restructure-20261019.md#architecture-efficiency) |
| <a id="loop-kv-next"></a>Loop与KV比较历史 | [历史原文](roadmap-before-restructure-20261019.md#loop-kv-next) |
| <a id="moe-sequential-next"></a>MoE顺序学习历史 | [历史原文](roadmap-before-restructure-20261019.md#moe-sequential-next) |

</details>
