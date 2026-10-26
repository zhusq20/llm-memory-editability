# LM Knowledge路线图：2026-10-25整理前存档

> 历史存档，仅供复核。下文的“当前／下一步”、编号和运行状态均指整理前的记录；新的论文定位与优先级只读[当前路线图](../roadmap.md)。本次仅压缩重复表达、调整阅读顺序，不新增实验授权，也不修改原始证据。相对链接已按存档位置调整。

更新：2026-10-25 UTC。本文是LM Knowledge唯一当前规划；先读§1，核对证据时再读[结果摘要](../results.md)。Loop的实验与计划由[独立路线图](../loop-roadmap.md)维护。

<a id="natural-knowledge-methods-20261021"></a>
<a id="next-experiments"></a>
## 1. 当前优先：真实模型怎样用上新知识

**目标：让部署中的模型学到新事实后，能在自然问题中调用、与已有知识组合，并在事实修改后使用新的后果，同时保留旧能力。** 两路工作相接：先形成新事实的多用途训练recipe，再研究能否通过共享训练学会更新规则，减少未来每条事实所需的数据与计算。现有八组按原协议及已授权接续修订完成；后续用途覆盖比较以现有off-policy PD为主，并用SFT做对应验证。

### 1.1 已有八组方法比较

| 工作 | 当前安排 |
| --- | --- |
| 八组方法比较 | `knowledge-method-factorial-1000-v1`：Qwen3-1.7B，SFT／Prompt Distillation × full／原范围LoRA／down-only／down-LoRA；作者HotpotQA前1000案例、297840条训练问答，每组5轮、1489200次样本曝光；按用户10-22授权从已有检查点增大batch，后续更新数随batch变化 |
| 评价 | Qwen3-4B关闭thinking，沿用作者评分消息与true/false解析；报告true／有效判分数及无效数量、覆盖率。保留旧32案例EM/F1，语义重评独立存放 |
| 执行 | 10-25 06:55 UTC：八组训练、独立重载、终点评测、云端核对及评测容器退出均完成。闭卷收益、原正确题保持和成本已统一分析；见[统一分析](../../results/knowledge-factorial-analysis-20261025/report.md) |
| 已有开发批 | 32案例四组的训练、独立重载与云端核验已完成。训练拟合改善，但EM受长回答格式影响，不能据此给方法排名或断言无效 |

完整预算、教师与数据冻结、更新范围及配方差异见[八组协议](../development-artifacts/knowledge-method-factorial-1000-v1/protocol.md)。这是Qwen3扩展；旧M3、E3与小模型TTCD可帮助解释结果，不能代替原方法比较。

10-22用户新增执行要求：保留当前训练进度，试测更大物理batch以提高吞吐。按[接续修订](../development-artifacts/knowledge-method-batchsize-20261022/amendment.md)保留参数、Adam、RNG和样本位置，以同一五轮数据曝光结束；原372300步仅描述batch4阶段。各组batch与切换点单列，不再描述为全程统一batch4比较。

八组均提高闭卷表现，同时损失部分旧正确题和原文档在场能力。PD-full闭卷最高46.75%；PD-LoRA为45.55%，原文档在场表现较好。收益、保持和成本共同解释，不作单一最优范围排名。用途覆盖已由用户指定全参数范围，继续按下述独立契约执行。

<a id="knowledge-use-recipe-plan"></a>
### 1.2 下一阶段：在现有PD上比较用途覆盖

**方法选择（2026-10-22用户确认）：context distillation不要求on-policy。当前PD是off-policy：提前生成并固定回答轨迹，训练时教师看到原文、学生不看原文，两者沿同一回答前缀计算分布，以正向KL（教师‖学生）训练。** 每步重新计算教师logits、教师与学生来自同一底座，都不改变轨迹来源。依据见[当前数据编码与损失](../../src/llm_memory_editability/knowledge_method_diagnosis.py)、[八组训练实现](../../src/llm_memory_editability/knowledge_method_factorial.py)。

主问题是：相同监督token预算下，表达多样性和用途多样性分别怎样影响新事实的提取、未练用途的泛化及旧能力保持？先完成PD三组，再用SFT三组做对应验证；首轮保持off-policy轨迹来源，不同时加入on-policy因素。

| 数据组织 | 现有off-policy PD（主比较） | SFT（对应验证） |
| --- | --- | --- |
| A：少量原子QA重复 | 固定回答轨迹上的教师分布监督 | 固定回答的硬目标监督 |
| B：多种问法的原子QA | 扩大措辞、格式与别名覆盖 | 相同用途范围的问法多样化 |
| C：原子QA＋多用途QA | 增加关系组合、比较、判断等用途 | 相同用途范围的多用途训练 |

A→B检验表达覆盖，B→C检验用途覆盖。每种方法内部固定底座、事实及背景资料、更新范围、监督token预算、调参预算和保持训练；记录更新数、事实曝光、生成与教师计算成本。各臂都从同一底座开始，将同一批事实写入一个模型。PD与SFT如保留各自的回答采样配方，按整套方法比较，不称纯KL／CE消融。

主要材料用自然语言表达的可控新事实，另在真实文档上验证应用效果。开发事实用于选择配方，正式写入事实与开发集分开；正式事实会进入写入训练，其测试用途问题留出。生成训练数据不读取测试问题或答案，并核对多用途材料是否额外引入背景事实。模型的背景知识及原文在场时的用途能力单独记录，全量结果和前提成立子集一起报告。

评价同时覆盖原子提取、未练问法、已练用途的新问题、留出的用途类别／关系组合、两条新事实的组合、事实再次修改后的依赖答案，以及无关旧正确题保持。直接回答与显式分步、新增与改写分别报告。保持集包含语义相近但答案不应变化的问题及一般能力问题，保护训练与保持评价分开；确实依赖新事实的旧答案不纳入保持约束。

已有[用途适配实验](../development-artifacts/use-adapter-development-v1/protocol.md)同时提供跨事实收益和保持损失的证据。本轮借鉴这些观察，在现有PD／SFT框架中明确控制用途覆盖，形成可比较的数据成本与保持曲线；兼容数据、评价和对照优先复用。

### 1.3 后续：成本、TTT与共享更新规则

1. **配方成本与保持。** 选代表性数据条件，比较不同样本量；区分固定训练预算下扩大覆盖与允许训练成本增长两种问题。联合报告未练用途表现、旧能力损失与总成本，得到需要哪些用途、多少数据及怎样保持旧能力的recipe。
2. **TTT应用。** 比较测试时只训练原材料、测试时增加用途数据、共享训练已学习用途而测试时只给原材料。跨样本任务在同一学习状态中先后写入材料A／B，清除原文与普通KV后测试组合；独立场景重置初始状态，场景内连续写入，评价答案不参与训练。
3. **少量数据的更新规则。** 借鉴[PropMEND](https://arxiv.org/html/2506.08920v1)／[PERK](https://arxiv.org/html/2507.06415v3)，比较普通原子更新、以原子目标训练的更新器、以多用途目标训练的更新器，并以第一路充分用途训练作为效果参照。共享训练与部署测试的事实、实体及部分关系组合分开；部署更新只接收原子材料。衡量共享训练能减少多少未来事实的用途数据需求，同时单列共享训练成本与旧能力保持。
4. **On-policy作为独立后续因素。** 在相同用途覆盖下再比较固定轨迹与当前学生生成轨迹。轨迹来源与KL方向分别控制；若按原论文同时改变二者，则报告整套配方差异。On-policy不是上述PD主比较、TTT或学习更新规则的前提，也不预设它必然更优。

**2026-10-24（芝加哥）用户最新决定：“立即开始，固定全参数”。** `knowledge-use-coverage-v1`的用途覆盖、原子改写传播与真实文档验证统一采用全参数PD／SFT，在GPU4／6／8执行；这替代先等八组完整分析再定范围的顺序。原PD-LoRA现已完成，启动时七组快照与最终八组结果分别保留。32对开发事实、128对正式事实和64个真实文档案例已经准备；开发／正式回答生成与自然问题开发校准可以并行，正式训练仍在开发选择共同监督token预算后进行，训练材料不读取测试问答。见[全参数修订](../development-artifacts/knowledge-use-coverage-v1/full-start-amendment-20261025.md)、[批次协议](../development-artifacts/knowledge-use-coverage-v1/protocol.md)与[执行交接](../../results/knowledge-use-coverage-20261025/RUNNING.txt)。这项授权不恢复其他历史候选或Loop暂缓训练。

**2026-10-25 18:07（芝加哥）状态复核：** 开发三组训练与两节点共六份重载／评价／云端核验收据齐全；已按原规则选择开发200192、正式800768、改写200192主监督token。正式集基线已正常完成，PD A／B／C三组正式新增训练实际运行且已有有限损失与梯度；SFT与原子改写尚无终点结果。自然问题修订已生成，结构有效46/96，语义审核仍待完成；不能把生成完成写成应用验证通过。详见[最新结果与快照](../results.md#knowledge-use-coverage-v1)、[四卡修订](../development-artifacts/knowledge-use-coverage-v1/four-card-amendment-20261025.md)。

原生记忆架构保留[已完成的调研](../reports/lm-knowledge-methods-and-memory-20261021.md)：Memory+跨深度共享持久内容、Hope-Attention多频率MLP可供后续比较；目前没有新增架构训练安排。

<a id="mission-20261020"></a>
<a id="mission-20261019"></a>
<a id="question"></a>
<a id="practical-questions"></a>
## 2. 从理论愿景组织论文

<a id="paper-vision-20261025"></a>
**知识在哪里、推理在哪里、二者怎样连接。我们的目标是让一次学到的事实成为后续计算可重复调用的资源：换问法仍能提取，放入未见组合仍能使用，事实修改后其依赖答案随之更新，同时保留不应改变的能力。** 这是长期研究愿景；完整实现可能需要新的架构与方法，不作为本稿必须全面兑现的承诺。本文解释实现这些能力所需的计算条件，并据此组织、检验解决原则。上下文蒸馏、参数微调、知识编辑与循环架构是不同的实现与检验入口。

**2026-10-25用户明确的组织原则：** 从理论愿景推导问题和关键比较，再安排实验支持、修正或反驳主张；不能从已有成绩反向拼装论证。不能为了规避论点与前文的重复性而放弃有价值的论点。与前文章节、此前讨论或已有论文重叠时，保留其科学作用，精简措辞并准确归属。本文对历史证据的重新组织是事后综合，不把它写成所有历史实验共有的事前假设。

### 2.1 先定义希望实现什么

用知识真值集合K、事实变更ΔK和查询q描述目标：更新后的模型应回答`Ans(K⊕ΔK, q)`。对确实依赖变更的查询，答案应改变；对不受变更影响的原正确查询，答案应保持。这里的K是评价真值，不假设网络内部有显式知识表，也不要求知识与推理物理上分属MLP和attention。

这导出三个行为要求与一个架构问题：**跨表达提取、跨用途组合、更新后的一致使用，以及使这些调用能够发生的计算结构。** “会答—能组合—更新后能用”是三个评价维度，不是所有模型都必须逐项经过的训练阶段。原子问答正确可以帮助排除缺事实的解释，但不是所有内部推理的普遍必要条件；Ye复现中原子模板0%与部分组合成功并存的结果必须保留。

### 2.2 论文要建立什么

**中心解释：答案监督约束的是当前输出；跨用途的知识复用还取决于哪些状态能触发已写入的关联、写入内容能否被后续计算使用，以及架构是否提供再次调用的机会。** 同一次参数变化又会影响其他用途，因此更新传播与旧能力保持必须联合研究。

正文把贡献落实为三项可核查内容：在完整Transformer计算上明确回答、组合与更新的非蕴含关系及调用条件；用表达、写入目标和架构的关键比较检验这些解释；把解释用于自然模型的数据配方与收益／保持／成本评价。前两项已有大量证据，第三项的八组比较完成、用途覆盖与改写正式结果尚缺。标准外积梯度和链式法则是分析工具；本研究的贡献应由实际建立的解释、比较及方法结果承担。

**2026-10-25用户补充的贡献定位：** 从知识与推理连接的统一视角归纳解决方案，常规知识编辑可以承担代表性实现，不要求每条原则都对应新算法。论证要说明每种操作改变哪个计算条件、为什么在某类用途上有效，以及什么问题仍未解决。局部改善、受控架构实现和长期愿景分别陈述。“当前方法未实现”“给定计算或更新约束下不可行”“所有现有架构都不可能”是不同层级；现有失败不能直接证明最后一项。

<a id="parts"></a>
| 部分 | 要解释的问题 | 关键已有证据 |
| --- | --- | --- |
| <a id="recall"></a>会答 | 学过的事实怎样被不同问法提取 | bioS表达组织改变提取率与地址；顺序写入原问法100%，留出问法37.61–48.62% |
| <a id="composition"></a>能组合 | 第一跳的结果怎样成为下一次读取的输入 | 标准小模型两条新事实分别答对64/64，组合0/64；另一个实验显式分步251/251，内部两跳0/251 |
| <a id="updating"></a><a id="sequential-transfer"></a><a id="write-interface"></a>更新后能用 | 新事实、依赖答案与旧能力怎样一起变化 | 小模型改写目标改变传播；邻居保护已有组合与保持的正结果，不能概括为持续积累仍全无进展 |
| <a id="architecture"></a>架构 | 写到哪一深度、以什么形式交接 | 同一受控比较中，未共享参数＋词表交接OO 6.19%，共享＋词表交接95.66% |

四部分各有端到端证据，架构是核心层次。表中的历史观察用于回答上述问题，不反过来定义研究目标。关键数字、反向结果和完成状态见[主题综合](../results.md#knowledge-paper-synthesis-20261025)。

<a id="pretraining-insight-role-20261025"></a>
### 2.3 预训练后难以更新这一观察怎样辅助主线

**2026-10-25用户明确：不比较同一模型的不同预训练阶段，开销过大；当前论文围绕给定模型中知识与推理的连接展开。** “预训练后注入知识常伴随旧能力损失”作为背景观察，帮助提出问题，不承担本文中心结论，也不作为新增预训练实验的理由。

它的作用贯穿三处：引言中说明学到很多知识之后，能否继续修改并一致使用仍是独立要求；机制讨论中提出新旧用途可能调用共同计算、一次写入因而可能同时带来迁移与干扰的解释；实验中要求同时评价新事实掌握、依赖答案变化和无关旧能力保持。与变更有关的答案应改变，无关答案应保持，二者共同定义一次更新是否成功。

现有写入目标、架构、邻居保护及PD／SFT比较，加上已授权用途覆盖与原子改写，分别检验上述问题；不需要重建预训练历程。当前证据不能确定干扰由预训练造成，更不能推出“注入必然遗忘”。邻居保护改善保持的正结果必须保留。本文应解释什么条件让知识可调用、可组合、可更新，预训练观察为这个愿景提供动机。

<a id="principles"></a>
<a id="theory"></a>
<a id="path-consistency"></a>
<a id="associative-memory-contribution"></a>
## 3. 解释框架：地址、内容与后续计算

**单向连接。** 一次因果Transformer前向只能使用已有的上游状态；推理得到的新实体要再调用知识，需要后续可用的读取计算，或重新经过已经学会该关联的计算。以实体／问题为地址分别描述随实体／问题变化的计算，是功能定义，具体位置由实验判断。

**答案监督的写入。** MLP输出矩阵在普通SGD下一步的更新满足 `ΔW = −η Σ_p δ_p a_pᵀ`：`a_p`是写入时的激活，`−δ_p`是完整后续计算传回的下降方向。已有实验中，后段写入常集中于问句末位置，推动模型说出答案；这不自动把答案变成下一次读取认得的实体表示。外积恒等式本身不能证明所有模型、所有训练都会写在相同位置。

<a id="write-modes"></a>
上下文读取的键值来自上下文状态；梯度写入的地址与方向来自激活和损失反传。两者可以借联想记忆语言比较，但真实softmax注意力、LoRA参数化、多步Adam与完整残差计算须保留各自条件。详细公式见[理论依据](../memory-scaling-theory.md#associative-memory-to-use-20261020)，不将它们压成一种无条件等价算法。

<a id="two-routes"></a>
| 调用路线 | 已有解释与例子 |
| --- | --- |
| 重演原计算 | CoT把实体变回输入；受控DiscoLoop重新经过共享层与兼容的问句位置。早期“路径一致性”属于这条工作假说 |
| 把共享地址放到实体 | bioS表达多样化、实体式写入与用途练习，为不同用途共享一次写入提供条件 |

两条路线是组织已有结果的解释，不是已经证明的唯一解。旧能力的干扰与新用途的迁移来自同一次更新，需要一起解释。

<a id="solution-principles-20261025"></a>
### 3.1 从计算条件组织解决原则

从愿景出发，一条知识要进入后续推理，至少需要相关关联被改变、使用时能够触发它、取得的内容能参与后续计算。依赖答案是否更新由这些条件共同决定。下面按实际改变的环节组织方法，三条路线可以组合，不构成互斥分类，也不把历史方法换名当作新发现。

**事实变化不意味着每次重学推理规则。** 更新后的依赖答案可以通过逐项维护已学结论来获得，也可以在使用时重新读取已修改的事实并计算出来；两者可以混合，但所需监督、覆盖范围和推理成本不同。长期愿景更强调后者的可复用性。实验须区分直接训练过的依赖答案与只修改原子事实后自行变化的答案，才能判断具体方案实现了什么。

| 解决路线 | 改变什么，预期解决什么 | 可复用证据与论证任务 |
| --- | --- | --- |
| 改善知识写入 | 改变写入地址、目标内容或更新约束，使后续用途能取得适合继续计算的内容 | 常规ROME／MEMIT复现、答案／早期实体／方向对齐写入比较。分别报告事实改对和依赖传播；常规编辑既是实际方案，也是检验写入条件的工具 |
| 改善知识调用 | 通过表达与用途训练改善触发方式，或把生成的中间实体重新作为输入，调用已有知识计算 | bioS、用途经历、当前覆盖比较及自生成桥链。区分训练改善、单次分步和外部多次调用；收益与额外监督、推理成本一起解释 |
| 改变架构中的访问与交接 | 让不同推理步骤有机会访问共享的关联，并以可继续使用的形式交接 | 已完成的参数共享×词表交接比较提供受控实现。说明哪些条件联合有效，不将这一实现写成唯一结构；自然模型的通用架构方案仍属后续目标 |

**保持要求贯穿三条路线。** 一次改写既要作用于依赖该事实的用途，也要控制对无关用途的影响；回放和邻居保护用于检验这种控制。架构提供调用机会，也不自动保证写入正确、推理正确或无遗忘。

论文贡献在于这些计算条件、操作与行为之间有证据支撑的联系，而非方法名录。代表性正结果说明原则可以落实，失败和模型差异说明还缺哪些条件。完整愿景尚未实现，不妨碍建立上述解释与方法论；具体声称有效的方案仍需对应结果支持。

<a id="evidence-chain"></a>
<a id="layer-evidence-chain"></a>
## 4. ICML论文的实验论证与开展顺序

### 4.1 从愿景到可检验的问题

每个实验节依次写“希望解释什么—哪个比较能区分解释—观察是什么—据此保留或修正什么判断”。下表是当前写作组织，不新增训练预算，也不改原批次契约。

| 理论导出的问题 | 正文关键比较与复用证据 | 它应回答什么；什么结果会收窄解释 |
| --- | --- | --- |
| E1：同一语义事实怎样获得可迁移的读取地址？ | bioS的S／M／MP及换名与前文冲突；自然Qwen同提示逐层表示、写入梯度与读取干预 | 连接训练表达与后来调用。若问法收益只发生在训练形式，就不能宣称跨表达可用；表示份额本身不能定位知识或推理 |
| E2：后续计算如何再次调用已学关联？ | 组合经历对匹配原子重复；参数共享×词表交接四格；直接回答对自生成中间答案再输入 | 将使用经历、再次访问和交接形式分开。共享或交接单独改善多少、联合改善多少都报告；成功不要求内部状态与embedding严格相等 |
| E3：一次事实修改怎样改变其后果，并保留无关能力？ | 同一实体标签的最终答案／早期实体／方向对齐目标；只复习对照；连续写入的随机／邻居保护；MQuAKE与RippleEdits | 编辑成功、依赖传播、旧原子和旧组合保持分别评价。传播改善若伴随保持损伤，就不能宣布完整解决；小模型邻居保护的正结果也不能因自然模型仍困难而删除 |
| E4：这些原则怎样指导真实模型学习新知识？ | 已完成PD／SFT×四参数范围；当前固定全参数A原子重复／B多问法／C多用途；随后只给原子材料再次改写 | 八组测实际方法的收益与代价；A→B测表达覆盖，B→C测用途覆盖，改写检验用途答案是否跟随事实变化。只改善所练用途的新问法，仍不足以宣称未练用途或双新组合已解决 |

E2承载架构主论点，不能缩成附录的一项消融。E4贯穿会答、能组合和更新后能用，形成受控解释与自然模型应用之间的连接；PD不是待被证明失败的靶子。当前八组是[PD原方法](https://arxiv.org/html/2412.14964v2)的Qwen扩展；旧M3／E3不能替代它，PD／SFT跨臂比较也不能改称纯KL／CE效果。

**共同读数。** 每个相关比较同时列基座、原文在场参照、原子掌握、直接／单次分步／外部自生成桥链、依赖答案与原正确旧题保持；不将不同调用方式合并成一个成绩。用途题按已练用途的新问法、留出用途、留出关系组合及双新事实分别计分。按事实对、知识世界、改写交换块或自然案例采用各批适当的配对分析，保留全量和前提成立子集及覆盖率。相同监督token、相同事实曝光和相同计算预算是不同比较；参数量、教师成本、保持成本与架构额外计算单列。历史四格中两条件联合成功是该配方的实测，不是所有Transformer都必须共享参数与词表交接的定理。

### 4.2 正文怎样展开

1. **引言：** 从一次学习应支持多种使用与后续更新的愿景出发，用一个贯穿全文的例子说明：机构总部从城市b迁到b′，问法变化、所在国家、两机构比较及迁址后答案分别要求什么。用已有结果时明确标注实际案例；未核对的例子仅作示意。
2. **定义与理论：** 定义知识真值、使用任务、更新及保持；保留真实Transformer计算。先给原子行为不能唯一确定组合行为的条件构造，再用“写入地址／内容／后续计算”解释为什么监督、数据和架构都可能改变调用。证明细节进附录，正文保留关键条件。`J(q)Δθ`同样描述新用途迁移与旧用途干扰。
3. **实验：** 按E1—E4展开。每节保留一个能区分解释的核心比较及其主要反证，其他兼容批次作补充。会答与能用的区别可在定义、受控检验和真实方法评价中再次出现：三处作用不同，不能因字面重复删掉后两处。
4. **解决原则、讨论与相关工作：** 用§3.1连接各节已经检验的条件与可行操作，说明现有架构内的改善、调用流程的改变和架构实现分别解决什么。准确归属已有方法，把本研究建立的解释、代表性改善和未实现目标分开。保留模型差异、失败预测与尚未解决的自然模型更新问题。TTT与共享更新规则在尚无对应正式结果时作为后续方向。

**图表服务问题。** 逐层机制主图继续复用[已有六面板](../reports/layer-evidence-chain-20261021.md)，标明不同面板的样本来源，不画成同一事实的完整因果轨迹；架构四格比较作为正文核心表；更新目标与保持并列展示；真实Qwen八组用收益／保持／成本表；用途覆盖用同预算下的能力分项表及新增→改写的配对结果。完整曲线、所有节点、全部失败与复现差异放附录，正文同步交代影响主张的负结果。

[ICML 2026评审说明](https://icml.cc/Conferences/2026/ReviewerInstructions)将主张的证据支持、方法可靠性、重要性与原创性分别评价，并明确原创性可以来自对既有方法的新理解，不要求每篇论文发明全新方法。这里据此检查论证是否完整，不推定其他年份的具体投稿规则或接受结果；目标年份未指定，版式与页数在成稿时另核。

### 4.3 当前能写到哪一步

**现在可以写引言、定义与理论，以及E1—E3和E4八组比较的结果正文。** 已有受控正结果足以支撑比“会答未必会用”更深入的解释：使用经历改变泛化，架构的访问与交接共同起作用，写入目标改变传播，针对干扰结构的保护可以改善持续积累。与此同时，真实模型已经显示闭卷收益与保持损失，不能写成参数注入普遍无效，也不能写成知识复用已经全面解决。

当前应用验证需要收齐的是：同一自然模型、同一受控事实流中，**用途覆盖的正式泛化结果、仅改写原子事实后的依赖变化，以及旧能力的配对保持**。已授权的用途覆盖协议正好检验这三个问题；取得并解释结果是工作要求，不把三项全部改善设为论文成文条件。自然文档材料的质量审核尚未通过，不能把生成完成算作应用验证完成。

当前开发结果里，C在所练用途的新问法上明显优于A／B，但留出用途与双新直接组合仍低；这使正式比较需要保留全部能力维度，而不能只保留C的有利列。即便后续仍如此，“学习用途扩大了哪一类调用、哪些调用还没有迁移”仍是有价值的结果；不能事后把原本的跨用途目标改成只测措辞变化。

### 4.4 工作优先级与历史对应

| 顺序 | 具体工作 | 历史对应与本次新增问题 |
| --- | --- | --- |
| 立即成文 | 完成上述定义、理论、历史结果正文与图表索引 | 对已有证据进行理论组织，属于综合，不包装成新增实验或事前预测 |
| 当前已授权批次 | 完成正式PD／SFT A/B/C及六组原子改写，按预算选择与原评价契约执行 | 八组只比较方法／参数范围；旧use-adapter有用途收益与损失，但逐事实分支与共享练习交织。本批固定事实、范围、监督预算，检验表达／用途覆盖与改写后复用 |
| 当前应用分支 | 完成真实文档问题的语义审核，再按原协议验证 | 结构合法与确由共同文档支持是不同条件；现在仍缺可靠应用结果，不凭空补齐或自动删去失败案例 |
| 主结果之后决定 | 如需强化推广性，优先补影响中心主张的事实世界／种子或独立自然材料验证 | 属于稳健性补证；当前八组单seed、正式结果未齐，不把扩大模型／网格本身称新科学目标，不在本次启动 |
| 后续研究 | §1.3的成本曲线、TTT及共享更新规则 | 复用顺序迁移、回放、PERK与用途练习的正负证据；新增问题是共享训练能否减少未来每条事实所需用途数据。尚无正式答案，不作为本稿已经完成的贡献 |

保留的重要反证包括：第二跳角色覆盖失败；RippleEdits不支持普遍第一跳优势；MEMIT上CoT没有缩小预定跳位差距；两个Qwen的读取阻断效应不同；“先形成桥实体再读第二跳上下文”的预定判断四组均失败；用途练习伴随自然两跳退化。它们用来修正解释，不取消有价值的问题。当前不恢复旧共享KV配方，也不把Loop独立路线的新配方结果混入已完成的四格比较。

<a id="scope"></a>
<a id="history-dedup"></a>
## 5. 历史结果怎样约束下一步

<a id="idea-history-check"></a>
**研究判断更正（2026-10-22用户明确）：不应该把之前论文讨论过的任何方法视为“不能再作为新发现”。** 文献重叠不能成为排除研究方向或否定发现与贡献的理由；应依据本研究实际建立的结果、解释和方法论判断，并准确说明已有工作的贡献。不能仅因已有论文或此前讨论涉及相关内容，就断言“多用途训练有帮助”“训练模型适应知识变化”不能成为本研究的发现。下面的历史检查用于理解证据、复用成果和避免无意义重复，不作为先验否定这些研究目标的门槛。

**选题要求（2026-10-22用户明确）：寻找新的突破点。提出idea之前先看历史跑过什么实验、检验过什么假设，不能把旧实验探究换一种说法当作新的研究目标。**

先核对本路线图、[结果汇总](../results.md)、[历史去重记录](../reports/roadmap-before-restructure-20261019.md#history-dedup)，并回到相关批次的协议、结果与[项目入口中的用户评价](../../AGENTS.md)。核查覆盖已完成、失败、停止、运行中与已规划的工作，按实际研究问题、假设和操作判断是否重复，不能只比较实验名称。

每个候选想法先简要写清：

1. **历史做过什么：** 最接近的假设、实验批次、实际操作与结果，包括负结果及用户已否定的延伸。
2. **还缺什么：** 历史证据具体留下了哪个未回答的问题；不能把已有结论重新表述成待验证目标。
3. **这次改变什么：** 实质改变的假设、关键条件或机制，为什么旧实验不能回答，哪些对照与产物可以复用。
4. **可能推进什么：** 新结果将怎样改变现有认识、区分尚未解决的解释或打开新方向；不能只以多跑一组、提高一个指标或补齐网格作为研究价值。

更名、换术语、换叙事，或仅换数据、模型、模块组合和扩大网格，本身不构成新目标。复现、补证与旧方向延伸可以有价值，但必须如实标注；重新研究旧问题须说明具体的新依据和新增问题。找不到实质新增问题时直接说明，不为提出idea而包装旧问题。与外部论文话题重叠的重要问题仍可研究，按上述要求说明本项目的新增认识。

| 历史事项 | 现在的处理 |
| --- | --- |
| 旧M3/E3、TTCD、知识调用接口诊断 | 已有机制证据；不再反复证明“会答未必会用”，不据此预判PD／SFT／LoRA原方法无效 |
| 写入目标、顺序迁移、旧组合回放、MoE与表示对齐 | 保留正负结果和兼容对照；已完成批次不重跑 |
| 当前知识任务的旧共享KV扩展 | 用户已因反复失败结束该配方扩展；不以改名、换呈现或组合架构继续同一问题 |
| 容量幂律、bioS 100k、日期更新、旧TTT／宽度／轮次／mHC网格 | 历史候选，退出默认待办；不能从草案叠加预算 |
| Loop的gate、位置、深度与重复性实验 | 已有授权由[Loop路线图](../loop-roadmap.md)独立维护；本表不取消其运行，也不引入LM Knowledge待办 |

新建议先核对[历史批次索引](../results.md#history-index)，说明已有批次、真正改变的变量和新增问题。用户逐项评价（原引用：`../../results/history-design-review-20261007/user-reviews.json`，整理时该文件在本地缺失）与方法判断分开，未回复不视为认可。原始负结果、失败、停止原因和小模型正结果都保留。

这里“不再反复证明”约束的是无新增问题的重复实验，不是删除论文中必要的论证。已有的“会答不保证会用”、用途训练收益、架构交接与更新传播等论点，仍按其在理论愿景中的作用进入正文。

<a id="main-setup"></a>
<a id="evaluation"></a>
<a id="efficiency-evaluation"></a>
## 6. 评价与执行

- 全量结果为主；直接／分步回答、组合后果、旧原子／组合保持分开。原正确题丢失不能被新增正确抵消。
- 曝光、token、更新数、计算、总／激活参数和耗时分别记录。共享GPU耗时不作独占效率证据。
- 恢复按[工作入口](../../AGENTS.md)与[实验协议](../experimental-protocol.md)，读取最新控制器、修订和实际LM Docker；不按文档中的旧GPU或PID启动。
- 训练完成、独立重载、W&B核验分别记录；本次文档整理不新增科学预算。

<a id="glossary"></a>
## 7. 术语

| 术语 | 含义 |
| --- | --- |
| 原子事实／桥实体 | 单条关系 `(h,r,t)`／两跳中连接两条事实的实体b |
| 地址／内容 | 什么状态触发写入／写入的方向；不与MLP输入权重的特征检测方向混用 |
| 实体形式／答案形式 | 读入实体词形成的表示／推高该词元输出的方向 |
| 普通Transformer／Loop L×T | 各层独立参数／L层共享块执行T次 |
| II／IO／OI／OO | 两跳事实各自参加过组合训练(I)，或只参加原子训练(O)；以批次契约为准 |
| E／D／R／U | 目标事实／依赖后果／回放材料／未参与选择的保持评价 |

<a id="read-interface"></a>
<a id="read-interface-v2"></a>
<a id="loop-native-handoff"></a>
<a id="literature"></a>
<a id="maintenance"></a>
## 8. 按需查阅

[结果与状态](../results.md) · [完整推导](../memory-scaling-theory.md) · [共同协议](../experimental-protocol.md) · [文献调研](../reports/lm-knowledge-methods-and-memory-20261021.md) · [Loop路线](../loop-roadmap.md) · [历史文档](../reports/knowledge-history-20261022/README.md)。知识调用接口开发、followup及v2均作为已完成证据归档。

本页只维护当前问题、优先级和选择依据；详细数据写入结果，调度写入批次交接。历史报告中的“下一步”不自动恢复为待办。

<details>
<summary>历史链接兼容索引（按需展开）</summary>

| 旧入口 | 记录 |
| --- | --- |
| <a id="first-principles-plan-20261019"></a>first-principles-plan-20261019 | [原文](../reports/knowledge-history-20261022/roadmap.md#first-principles-plan-20261019) |
| <a id="sequence-dependency-gate-v1"></a>sequence-dependency-gate-v1 | [原文](../reports/knowledge-history-20261022/roadmap.md#sequence-dependency-gate-v1) |
| <a id="residual-gate-injection-ablation-v1"></a>residual-gate-injection-ablation-v1 | [原文](../reports/knowledge-history-20261022/roadmap.md#residual-gate-injection-ablation-v1) |
| <a id="candidate-update-gate-gpu1-v3"></a>candidate-update-gate-gpu1-v3 | [原文](../reports/knowledge-history-20261022/roadmap.md#candidate-update-gate-gpu1-v3) |
| <a id="compact-wang-joint-gate-v1"></a>compact-wang-joint-gate-v1 | [原文](../reports/knowledge-history-20261022/roadmap.md#compact-wang-joint-gate-v1) |
| <a id="candidate-update-gate-multihop-v1"></a>candidate-update-gate-multihop-v1 | [原文](../reports/knowledge-history-20261022/roadmap.md#candidate-update-gate-multihop-v1) |
| <a id="joint-update-gate-wang-v1"></a>joint-update-gate-wang-v1 | [原文](../reports/knowledge-history-20261022/roadmap.md#joint-update-gate-wang-v1) |
| <a id="joint-update-gate-multihop-v1"></a>joint-update-gate-multihop-v1 | [原文](../reports/knowledge-history-20261022/roadmap.md#joint-update-gate-multihop-v1) |
| <a id="gate-generalization-extension-v1"></a>gate-generalization-extension-v1 | [原文](../reports/knowledge-history-20261022/roadmap.md#gate-generalization-extension-v1) |
| <a id="paper-update-gate-continuation-v1"></a>paper-update-gate-continuation-v1 | [原文](../reports/knowledge-history-20261022/roadmap.md#paper-update-gate-continuation-v1) |
| <a id="paper-update-gate-multihop-u95-v1"></a>paper-update-gate-multihop-u95-v1 | [原文](../reports/knowledge-history-20261022/roadmap.md#paper-update-gate-multihop-u95-v1) |
| <a id="paper-update-gate-multihop-v1"></a>paper-update-gate-multihop-v1 | [原文](../reports/knowledge-history-20261022/roadmap.md#paper-update-gate-multihop-v1) |
| <a id="gate-evidence-refinement-20261017"></a>gate-evidence-refinement-20261017 | [原文](../reports/knowledge-history-20261022/roadmap.md#gate-evidence-refinement-20261017) |
| <a id="gated-loop-depth-extrapolation-v1"></a>gated-loop-depth-extrapolation-v1 | [原文](../reports/knowledge-history-20261022/roadmap.md#gated-loop-depth-extrapolation-v1) |
| <a id="zhang2026"></a>zhang2026 | [原文](../reports/knowledge-history-20261022/roadmap.md#zhang2026) |
| <a id="discoloop2026"></a>discoloop2026 | [原文](../reports/knowledge-history-20261022/roadmap.md#discoloop2026) |
| <a id="weight-tying-pooling"></a>weight-tying-pooling | [原文](../reports/knowledge-history-20261022/roadmap.md#weight-tying-pooling) |
| <a id="feng2025"></a>feng2025 | [原文](../reports/knowledge-history-20261022/roadmap.md#feng2025) |
| <a id="pretraining-continual"></a>pretraining-continual | [原文](../reports/knowledge-history-20261022/roadmap.md#pretraining-continual) |
| <a id="wang-benchmark-boundaries"></a>wang-benchmark-boundaries | [原文](../reports/knowledge-history-20261022/roadmap.md#wang-benchmark-boundaries) |
| <a id="innovation-rnn-ttt"></a>innovation-rnn-ttt | [原文](../reports/knowledge-history-20261022/roadmap.md#innovation-rnn-ttt) |
| <a id="memory-mosaics"></a>memory-mosaics | [原文](../reports/knowledge-history-20261022/roadmap.md#memory-mosaics) |
| <a id="memory-reuse-solution"></a>memory-reuse-solution | [原文](../reports/knowledge-history-20261022/roadmap.md#memory-reuse-solution) |
| <a id="dual-gate-design-20261017"></a>dual-gate-design-20261017 | [原文](../reports/knowledge-history-20261022/roadmap.md#dual-gate-design-20261017) |
| <a id="norm-handoff-minimal-proposal-20261017"></a>norm-handoff-minimal-proposal-20261017 | [原文](../reports/knowledge-history-20261022/roadmap.md#norm-handoff-minimal-proposal-20261017) |
| <a id="loop-solution-integration-20261012"></a>loop-solution-integration-20261012 | [原文](../reports/knowledge-history-20261022/roadmap.md#loop-solution-integration-20261012) |
| <a id="paper-reproduction-first"></a>paper-reproduction-first | [原文](../reports/knowledge-history-20261022/roadmap.md#paper-reproduction-first) |
| <a id="zhang2026-next"></a>zhang2026-next | [原文](../reports/knowledge-history-20261022/roadmap.md#zhang2026-next) |
| <a id="nl-edit-layer"></a>nl-edit-layer | [原文](../reports/knowledge-history-20261022/roadmap.md#nl-edit-layer) |
| <a id="history-audit-20261010"></a>history-audit-20261010 | [原文](../reports/knowledge-history-20261022/roadmap.md#history-audit-20261010) |
| <a id="controlled-multihop"></a>controlled-multihop | [原文](../reports/knowledge-history-20261022/roadmap.md#controlled-multihop) |
| <a id="architecture-efficiency"></a>architecture-efficiency | [原文](../reports/knowledge-history-20261022/roadmap.md#architecture-efficiency) |
| <a id="loop-kv-next"></a>loop-kv-next | [原文](../reports/knowledge-history-20261022/roadmap.md#loop-kv-next) |
| <a id="moe-sequential-next"></a>moe-sequential-next | [原文](../reports/knowledge-history-20261022/roadmap.md#moe-sequential-next) |
| <a id="read-interface-20261020"></a>read-interface-20261020 | [原文](../reports/knowledge-history-20261022/roadmap.md#read-interface-20261020) |
| <a id="研究路线知识在哪里推理在哪里二者怎样连接"></a>研究路线：知识在哪里、推理在哪里、二者怎样连接 | [原文](../reports/knowledge-history-20261022/roadmap.md#研究路线知识在哪里推理在哪里二者怎样连接) |
| <a id="1-核心故事"></a>1. 核心故事 | [原文](../reports/knowledge-history-20261022/roadmap.md#1-核心故事) |
| <a id="从联想记忆到可调用的知识"></a>从联想记忆到可调用的知识 | [原文](../reports/knowledge-history-20261022/roadmap.md#从联想记忆到可调用的知识) |
| <a id="2-两条原则"></a>2. 两条原则 | [原文](../reports/knowledge-history-20261022/roadmap.md#2-两条原则) |
| <a id="原则一单向连接"></a>原则一：单向连接 | [原文](../reports/knowledge-history-20261022/roadmap.md#原则一单向连接) |
| <a id="原则二答案监督的地址与内容"></a>原则二：答案监督的地址与内容 | [原文](../reports/knowledge-history-20261022/roadmap.md#原则二答案监督的地址与内容) |
| <a id="总纲重演原计算或把地址挪到实体"></a>总纲：重演原计算，或把地址挪到实体 | [原文](../reports/knowledge-history-20261022/roadmap.md#总纲重演原计算或把地址挪到实体) |
| <a id="3-四个部分"></a>3. 四个部分 | [原文](../reports/knowledge-history-20261022/roadmap.md#3-四个部分) |
| <a id="31-会答知识的地址"></a>3.1 会答：知识的地址 | [原文](../reports/knowledge-history-20261022/roadmap.md#31-会答知识的地址) |
| <a id="32-能组合单向连接"></a>3.2 能组合：单向连接 | [原文](../reports/knowledge-history-20261022/roadmap.md#32-能组合单向连接) |
| <a id="33-更新后能用注入就是答案监督"></a>3.3 更新后能用：注入就是答案监督 | [原文](../reports/knowledge-history-20261022/roadmap.md#33-更新后能用注入就是答案监督) |
| <a id="34-架构写到哪一深度以什么形式交接"></a>3.4 架构：写到哪一深度、以什么形式交接 | [原文](../reports/knowledge-history-20261022/roadmap.md#34-架构写到哪一深度以什么形式交接) |
| <a id="4-成文方式与下一步"></a>4. 成文方式与下一步 | [原文](../reports/knowledge-history-20261022/roadmap.md#4-成文方式与下一步) |
| <a id="5-数据评价与执行约束"></a>5. 数据、评价与执行约束 | [原文](../reports/knowledge-history-20261022/roadmap.md#5-数据评价与执行约束) |
| <a id="6-避免重复"></a>6. 避免重复 | [原文](../reports/knowledge-history-20261022/roadmap.md#6-避免重复) |
| <a id="7-核心术语"></a>7. 核心术语 | [原文](../reports/knowledge-history-20261022/roadmap.md#7-核心术语) |
| <a id="8-文档与历史入口"></a>8. 文档与历史入口 | [原文](../reports/knowledge-history-20261022/roadmap.md#8-文档与历史入口) |

</details>
