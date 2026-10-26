# LM Knowledge：论文与研究路线图

更新：2026-10-25。本文维护LM Knowledge的论文主线与工作优先级；证据和批次状态见[结果摘要](results.md)，Loop工作见[独立路线图](loop-roadmap.md)。

<a id="mission-20261020"></a>
<a id="mission-20261019"></a>
<a id="question"></a>
<a id="practical-questions"></a>
<a id="paper-vision-20261025"></a>
## 1. 愿景与本稿贡献

**知识在哪里、推理在哪里、二者怎样连接。** 长期愿景是让学到的事实成为可重复调用的资源：换问法仍能提取，进入新组合仍能使用，事实修改后相关答案随之更新，同时保留无关能力。完整实现可能需要新的架构与方法，不作为本稿必须全面兑现的承诺。

**本稿要建立的是：解释知识写入、调用与后续计算的条件，据此归纳解决原则，并用代表性方法检验。** 常规知识编辑、训练配方和架构比较都可承担实现与证据，不要求每条原则对应新算法。贡献应落在计算条件、具体操作与行为之间有证据支持的联系。

用知识真值K、事实变更ΔK和查询q描述能力目标：更新后应回答`Ans(K⊕ΔK, q)`；依赖变更的答案应改变，无关原正确答案应保持。K是评价真值，不假设内部存在知识表，也不预设MLP负责知识、attention负责推理。“会答、能组合、更新后能用”是三个评价维度，架构决定这些调用如何发生。

**组织原则（2026-10-25用户明确）：**

1. 从理论愿景提出问题与关键比较，再用实验支持、修正或反驳；不从已有成绩反向拼装主张。历史证据的整合如实写成事后综合。
2. 不因与前文、此前讨论或已有论文重叠而放弃有价值的论点；保留它在定义、解释和检验中的作用，精简重复措辞并准确归属。
3. 分清愿景、数学条件、工作假说与实测。“当前方法未实现”不等于“现有架构不可能”；局部改善不等于完整解决。

<a id="pretraining-insight-role-20261025"></a>
**预训练观察的作用。** “预训练后编辑容易干扰旧知识”用于引出更新与保持的问题，不作为本文中心结论。不比较同一模型的不同预训练阶段，不归因于未经检验的预训练历史，也不预设遗忘必然发生。

<a id="principles"></a>
<a id="theory"></a>
<a id="path-consistency"></a>
<a id="associative-memory-contribution"></a>
## 2. 解释框架与解决原则

**中心解释：答案监督约束当前输出，跨用途复用还依赖写入地址、内容和后续计算。** 单次因果前向中，新得到的实体要继续调用知识，需要后续可用的读取计算，或重新经过已学会该关联的计算。普通SGD下，MLP输出矩阵一步更新为`ΔW = −η Σ_p δ_p a_pᵀ`：激活`a_p`参与决定写入地址，反传方向`−δ_p`服务于当前目标；推动模型说出答案，不自动保证其表示可被下一跳使用。

<a id="write-modes"></a>
上下文读取与参数写入可借地址／内容语言比较，但softmax注意力、LoRA、多步Adam和完整残差计算保留各自条件。外积与链式法则是分析工具；一阶响应`J(q)Δθ`同时描述新用途迁移与旧用途干扰。条件推导、非蕴含关系及反例见[理论依据](memory-scaling-theory.md#associative-memory-to-use-20261020)。

<a id="two-routes"></a>
已有解释包含两条调用路线：**重演原计算**，如把生成的实体重新输入、循环访问共享层；**使不同用途共享实体地址**，如表达多样化与实体式写入。它们是待证据检验的机制解释，不是唯一解。

<a id="solution-principles-20261025"></a>
**从这些条件组织三类解决方案：**

| 路线 | 改变的环节 | 代表性实现与检验 |
| --- | --- | --- |
| 改善知识写入 | 写入地址、目标内容、更新约束 | ROME／MEMIT；答案／早期实体／方向对齐目标；分别测编辑成功与依赖传播 |
| 改善知识调用 | 触发关联的方式，以及取得内容后的使用 | 表达与用途训练、自生成中间答案再输入；区分训练收益与额外推理成本 |
| 改变架构的访问与交接 | 不同推理步骤访问共享关联的机会及交接形式 | 参数共享×词表交接；检验联合条件与各自作用，不将一个受控实现写成唯一结构 |

三类方案可以组合，旧能力保持贯穿其中，回放与邻居保护提供已有检验。架构提供调用机会，也不自动保证正确写入、推理或无遗忘。

**更新事实不等于重学推理规则。** 依赖答案既可逐项改写，也可在使用时读取新事实再计算；两者可以混合，但监督、覆盖和计算成本不同。愿景更强调后者，因此必须区分直接训练过的依赖答案与只改原子事实后自行变化的答案。

<a id="parts"></a>
<a id="evidence-chain"></a>
<a id="layer-evidence-chain"></a>
## 3. 论文结构与实验论证

正文按**愿景与问题 → 定义和计算条件 → E1—E4关键比较 → 解决原则与讨论**展开。每个实验节写清“要解释什么、哪个比较能区分解释、观察如何修正判断”；关键条件留在正文，证明细节与完整结果进附录。

| 实验节与主张 | 关键比较 | 论证重点 |
| --- | --- | --- |
| <a id="recall"></a>E1：会答——事实怎样跨表达被提取 | bioS表达组织及换名／前文冲突；自然Qwen逐层表示、写入梯度与读取干预 | 联系训练表达、写入地址和调用；表示份额本身不能定位知识或推理 |
| <a id="composition"></a><a id="architecture"></a>E2：能组合与架构——后续计算怎样再次调用知识 | 用途经历对匹配原子重复；参数共享×词表交接四格；直接回答对自生成桥链 | 分开使用经历、再次访问和交接形式；架构是正文核心，不降为附录消融 |
| <a id="updating"></a><a id="sequential-transfer"></a><a id="write-interface"></a>E3：更新后能用——依赖传播与无关保持 | 三种写入目标及只复习对照；随机／邻居保护；MQuAKE与RippleEdits | 编辑成功、依赖传播、旧原子／组合保持分别评价，保留改善保持的正结果 |
| E4：真实模型——原则怎样指导学习新知识 | PD／SFT×四参数范围；全参数A原子重复／B多问法／C多用途；随后仅改原子事实 | 八组测收益／保持／成本；A→B测表达覆盖，B→C测用途覆盖，改写测后果是否跟随 |

四节复用[主题证据](results.md#knowledge-paper-synthesis-20261025)。E4贯穿会答、组合和更新，连接受控解释与自然模型应用。PD作为有实际收益的方法评价；旧M3／E3与小模型TTCD不能替代原方法。原子模板成功不保证组合，原子模板失败也不等于内部完全没有事实。

图表保留[逐层六面板](reports/layer-evidence-chain-20261021.md)、架构四格、更新传播与保持对照、真实模型收益／保持／成本及用途分项表。不同批次注明来源，不拼成同一事实的完整因果轨迹。影响主张的反证进入正文，包括角色迁移失败、跨模型／方法不一致和用途收益伴随旧能力退化；细节与原冻结判据保留在结果及附录。

<a id="natural-knowledge-methods-20261021"></a>
<a id="next-experiments"></a>
## 4. 当前进度与工作顺序

**现在可以写引言、定义与理论、E1—E3，以及E4八组结果。** Qwen3-1.7B的PD／SFT×full／LoRA／down／down-LoRA八组已完成训练、独立重载、终点评价与云端核验；各组1489200次样本曝光，接续batch差异与成本见[统一分析](../results/knowledge-factorial-analysis-20261025/report.md)和[原协议](development-artifacts/knowledge-method-factorial-1000-v1/protocol.md)。八组均有闭卷收益，也有旧正确题损失，按收益、保持和成本共同解释。

| 优先级 | 工作与完成要求 |
| --- | --- |
| 立即成文 | 按§1—3整理正文及图表，说明每项证据支持的条件和未解决问题 |
| 完成已授权比较 | 收齐正式PD／SFT A/B/C、六组原子改写的泛化、传播和配对保持结果；取得并解释结果，不以所有指标改善为成文门槛 |
| 完成应用验证 | 先完成真实文档问题的语义审核，再按原协议验证；生成完成不等于应用验证通过 |
| 主结果之后再定 | 仅补影响中心主张的事实世界／种子或独立自然材料验证；不以扩大网格本身作为新目标 |

<a id="knowledge-use-recipe-plan"></a>
### 当前批次：表达与用途覆盖

全参数范围已由用户确认。固定底座、事实与背景资料、监督token预算、调参预算和保持训练，比较A少量原子QA重复、B多问法原子QA、C原子QA＋多用途QA；先PD，再SFT对应验证，随后仅提供原子材料再次改写。与旧用途适配相比，本批明确控制表达／用途覆盖，并检验修改后的复用。

PD保持off-policy：预先固定回答轨迹，教师看原文、学生不看原文，沿相同回答前缀使用正向KL。重算教师logits不改变轨迹来源；PD／SFT若保留不同回答配方，按整套方法比较，不称纯KL／CE消融。开发与正式事实分开，生成材料不读取测试问答，多用途材料不得暗中引入额外背景事实。

截至[2026-10-25 18:07芝加哥的已核对快照](results.md#knowledge-use-coverage-v1)：PD开发六节点完成，预算已选定，正式PD A／B／C已启动；SFT与改写尚无终点结果，自然问题仍待语义审核。开发C的收益主要见于所练用途的新问法，未练用途与双新直接组合仍低，正式比较保留全部能力维度。本段是有时间戳的记录，实时状态读[批次交接](../results/knowledge-use-coverage-20261025/RUNNING.txt)指向的controller与实际LM Docker；预算和执行以[协议](development-artifacts/knowledge-use-coverage-v1/protocol.md)、[全参数修订](development-artifacts/knowledge-use-coverage-v1/full-start-amendment-20261025.md)和[四卡修订](development-artifacts/knowledge-use-coverage-v1/four-card-amendment-20261025.md)为准。

### 后续问题

当前比较之后，再决定数据成本／保持曲线、TTT与共享更新规则：重点是能否通过共享训练减少未来每条事实所需的用途数据。部署时仅给原子材料，评价跨事实组合、保持与总成本；复用顺序迁移、回放、PERK的正负证据。On-policy是独立后续因素，不是当前PD或TTT的前提。详细旧设想见[整理前存档](reports/roadmap-before-condense-20261025.md)，不据此自动启动。原生记忆架构保留[调研入口](reports/lm-knowledge-methods-and-memory-20261021.md)，当前没有新增架构训练安排。

<a id="scope"></a>
<a id="history-dedup"></a>
<a id="idea-history-check"></a>
## 5. 研究取舍与评价

**新建议先查[历史批次](results.md#history-index)、原协议与[用户研究约定](../AGENTS.md#论文主线与既有研究约定)，写清最近的历史问题、已有结果、未回答之处及本次实质改变。** 换名称、模型或扩大网格本身不构成新目标；复现、补证与延伸如实标注。用户评价与方法判断分开，未回复不视为认可。文献重叠不排除有价值的发现，历史查重也不删除必要论点。

已完成的写入目标、顺序迁移、回放、MoE、对齐及旧KV批次复用证据，不重复提交；用户已结束的共享KV配方不换名延续。容量幂律、bioS 100k、日期更新与旧TTT／宽度／轮次／mHC网格退出默认待办。Loop由独立路线图管理，本稿不恢复其暂缓批次，也不混合不同批次结果。

<a id="main-setup"></a>
<a id="evaluation"></a>
<a id="efficiency-evaluation"></a>
**评价共同要求：**

- 同时报告基座、原文在场参照、原子掌握、用途泛化、依赖传播与原正确题保持；新增正确不能抵消原正确丢失，格式／别名失分另审。
- 直接回答、单次分步、外部自生成桥链分开；已练用途的新问法、留出用途／关系组合和双新事实分开；新增与改写分开。
- 全量结果为主，前提成立子集另报覆盖率。按事实对、知识世界或案例做适当配对；不把同一事实的问法当作独立事实。保护训练与保持评价分开，依赖变更的旧答案不受保持约束。
- 曝光、监督token、更新数与计算预算不混用；教师／数据生成／保持训练成本、参数量、架构额外计算和耗时分别记录，共享GPU耗时不作独占效率证据。
- 训练完成、独立重载、W&B核验分别记录；恢复按[工作入口](../AGENTS.md)及[实验协议](experimental-protocol.md)，不按旧GPU或PID启动。本次整理不改变授权、预算与冻结材料。

<a id="read-interface"></a>
<a id="read-interface-v2"></a>
<a id="loop-native-handoff"></a>
<a id="literature"></a>
<a id="maintenance"></a>
## 6. 文档入口

[结果与状态](results.md) · [完整推导](memory-scaling-theory.md) · [共同协议](experimental-protocol.md) · [文献调研](reports/lm-knowledge-methods-and-memory-20261021.md) · [Loop路线](loop-roadmap.md) · [历史文档](reports/knowledge-history-20261022/README.md) · [本次整理前存档](reports/roadmap-before-condense-20261025.md)。

本页维护问题、优先级与选择依据；详细数据写入结果，调度写入批次交接。历史报告中的“当前／下一步”不自动成为待办。投稿年份尚未指定，版式与要求在成稿时核对。

<a id="glossary"></a>
<details>
<summary>术语速查</summary>

| 术语 | 含义 |
| --- | --- |
| 原子事实／桥实体 | 单条关系`(h,r,t)`／连接两条关系的中间实体 |
| 地址／内容 | 什么状态触发关联／写入的方向；不与MLP输入权重的特征检测方向混用 |
| 实体形式／答案形式 | 读入实体词形成的表示／推高该词元输出的方向 |
| 普通Transformer／Loop L×T | 各层独立参数／L层共享块执行T次 |
| II／IO／OI／OO | 两条事实分别参加过组合训练(I)或只参加原子训练(O)，以批次契约为准 |
| E／D／R／U | 目标事实／依赖后果／回放材料／未参与选择的保持评价 |

</details>

<details>
<summary>历史链接兼容索引（按需展开）</summary>

| 旧入口 | 记录 |
| --- | --- |
| <a id="first-principles-plan-20261019"></a>first-principles-plan-20261019 | [原文](reports/knowledge-history-20261022/roadmap.md#first-principles-plan-20261019) |
| <a id="sequence-dependency-gate-v1"></a>sequence-dependency-gate-v1 | [原文](reports/knowledge-history-20261022/roadmap.md#sequence-dependency-gate-v1) |
| <a id="residual-gate-injection-ablation-v1"></a>residual-gate-injection-ablation-v1 | [原文](reports/knowledge-history-20261022/roadmap.md#residual-gate-injection-ablation-v1) |
| <a id="candidate-update-gate-gpu1-v3"></a>candidate-update-gate-gpu1-v3 | [原文](reports/knowledge-history-20261022/roadmap.md#candidate-update-gate-gpu1-v3) |
| <a id="compact-wang-joint-gate-v1"></a>compact-wang-joint-gate-v1 | [原文](reports/knowledge-history-20261022/roadmap.md#compact-wang-joint-gate-v1) |
| <a id="candidate-update-gate-multihop-v1"></a>candidate-update-gate-multihop-v1 | [原文](reports/knowledge-history-20261022/roadmap.md#candidate-update-gate-multihop-v1) |
| <a id="joint-update-gate-wang-v1"></a>joint-update-gate-wang-v1 | [原文](reports/knowledge-history-20261022/roadmap.md#joint-update-gate-wang-v1) |
| <a id="joint-update-gate-multihop-v1"></a>joint-update-gate-multihop-v1 | [原文](reports/knowledge-history-20261022/roadmap.md#joint-update-gate-multihop-v1) |
| <a id="gate-generalization-extension-v1"></a>gate-generalization-extension-v1 | [原文](reports/knowledge-history-20261022/roadmap.md#gate-generalization-extension-v1) |
| <a id="paper-update-gate-continuation-v1"></a>paper-update-gate-continuation-v1 | [原文](reports/knowledge-history-20261022/roadmap.md#paper-update-gate-continuation-v1) |
| <a id="paper-update-gate-multihop-u95-v1"></a>paper-update-gate-multihop-u95-v1 | [原文](reports/knowledge-history-20261022/roadmap.md#paper-update-gate-multihop-u95-v1) |
| <a id="paper-update-gate-multihop-v1"></a>paper-update-gate-multihop-v1 | [原文](reports/knowledge-history-20261022/roadmap.md#paper-update-gate-multihop-v1) |
| <a id="gate-evidence-refinement-20261017"></a>gate-evidence-refinement-20261017 | [原文](reports/knowledge-history-20261022/roadmap.md#gate-evidence-refinement-20261017) |
| <a id="gated-loop-depth-extrapolation-v1"></a>gated-loop-depth-extrapolation-v1 | [原文](reports/knowledge-history-20261022/roadmap.md#gated-loop-depth-extrapolation-v1) |
| <a id="zhang2026"></a>zhang2026 | [原文](reports/knowledge-history-20261022/roadmap.md#zhang2026) |
| <a id="discoloop2026"></a>discoloop2026 | [原文](reports/knowledge-history-20261022/roadmap.md#discoloop2026) |
| <a id="weight-tying-pooling"></a>weight-tying-pooling | [原文](reports/knowledge-history-20261022/roadmap.md#weight-tying-pooling) |
| <a id="feng2025"></a>feng2025 | [原文](reports/knowledge-history-20261022/roadmap.md#feng2025) |
| <a id="pretraining-continual"></a>pretraining-continual | [原文](reports/knowledge-history-20261022/roadmap.md#pretraining-continual) |
| <a id="wang-benchmark-boundaries"></a>wang-benchmark-boundaries | [原文](reports/knowledge-history-20261022/roadmap.md#wang-benchmark-boundaries) |
| <a id="innovation-rnn-ttt"></a>innovation-rnn-ttt | [原文](reports/knowledge-history-20261022/roadmap.md#innovation-rnn-ttt) |
| <a id="memory-mosaics"></a>memory-mosaics | [原文](reports/knowledge-history-20261022/roadmap.md#memory-mosaics) |
| <a id="memory-reuse-solution"></a>memory-reuse-solution | [原文](reports/knowledge-history-20261022/roadmap.md#memory-reuse-solution) |
| <a id="dual-gate-design-20261017"></a>dual-gate-design-20261017 | [原文](reports/knowledge-history-20261022/roadmap.md#dual-gate-design-20261017) |
| <a id="norm-handoff-minimal-proposal-20261017"></a>norm-handoff-minimal-proposal-20261017 | [原文](reports/knowledge-history-20261022/roadmap.md#norm-handoff-minimal-proposal-20261017) |
| <a id="loop-solution-integration-20261012"></a>loop-solution-integration-20261012 | [原文](reports/knowledge-history-20261022/roadmap.md#loop-solution-integration-20261012) |
| <a id="paper-reproduction-first"></a>paper-reproduction-first | [原文](reports/knowledge-history-20261022/roadmap.md#paper-reproduction-first) |
| <a id="zhang2026-next"></a>zhang2026-next | [原文](reports/knowledge-history-20261022/roadmap.md#zhang2026-next) |
| <a id="nl-edit-layer"></a>nl-edit-layer | [原文](reports/knowledge-history-20261022/roadmap.md#nl-edit-layer) |
| <a id="history-audit-20261010"></a>history-audit-20261010 | [原文](reports/knowledge-history-20261022/roadmap.md#history-audit-20261010) |
| <a id="controlled-multihop"></a>controlled-multihop | [原文](reports/knowledge-history-20261022/roadmap.md#controlled-multihop) |
| <a id="architecture-efficiency"></a>architecture-efficiency | [原文](reports/knowledge-history-20261022/roadmap.md#architecture-efficiency) |
| <a id="loop-kv-next"></a>loop-kv-next | [原文](reports/knowledge-history-20261022/roadmap.md#loop-kv-next) |
| <a id="moe-sequential-next"></a>moe-sequential-next | [原文](reports/knowledge-history-20261022/roadmap.md#moe-sequential-next) |
| <a id="read-interface-20261020"></a>read-interface-20261020 | [原文](reports/knowledge-history-20261022/roadmap.md#read-interface-20261020) |
| <a id="研究路线知识在哪里推理在哪里二者怎样连接"></a>研究路线：知识在哪里、推理在哪里、二者怎样连接 | [原文](reports/knowledge-history-20261022/roadmap.md#研究路线知识在哪里推理在哪里二者怎样连接) |
| <a id="1-核心故事"></a>1. 核心故事 | [原文](reports/knowledge-history-20261022/roadmap.md#1-核心故事) |
| <a id="从联想记忆到可调用的知识"></a>从联想记忆到可调用的知识 | [原文](reports/knowledge-history-20261022/roadmap.md#从联想记忆到可调用的知识) |
| <a id="2-两条原则"></a>2. 两条原则 | [原文](reports/knowledge-history-20261022/roadmap.md#2-两条原则) |
| <a id="原则一单向连接"></a>原则一：单向连接 | [原文](reports/knowledge-history-20261022/roadmap.md#原则一单向连接) |
| <a id="原则二答案监督的地址与内容"></a>原则二：答案监督的地址与内容 | [原文](reports/knowledge-history-20261022/roadmap.md#原则二答案监督的地址与内容) |
| <a id="总纲重演原计算或把地址挪到实体"></a>总纲：重演原计算，或把地址挪到实体 | [原文](reports/knowledge-history-20261022/roadmap.md#总纲重演原计算或把地址挪到实体) |
| <a id="3-四个部分"></a>3. 四个部分 | [原文](reports/knowledge-history-20261022/roadmap.md#3-四个部分) |
| <a id="31-会答知识的地址"></a>3.1 会答：知识的地址 | [原文](reports/knowledge-history-20261022/roadmap.md#31-会答知识的地址) |
| <a id="32-能组合单向连接"></a>3.2 能组合：单向连接 | [原文](reports/knowledge-history-20261022/roadmap.md#32-能组合单向连接) |
| <a id="33-更新后能用注入就是答案监督"></a>3.3 更新后能用：注入就是答案监督 | [原文](reports/knowledge-history-20261022/roadmap.md#33-更新后能用注入就是答案监督) |
| <a id="34-架构写到哪一深度以什么形式交接"></a>3.4 架构：写到哪一深度、以什么形式交接 | [原文](reports/knowledge-history-20261022/roadmap.md#34-架构写到哪一深度以什么形式交接) |
| <a id="4-成文方式与下一步"></a>4. 成文方式与下一步 | [原文](reports/knowledge-history-20261022/roadmap.md#4-成文方式与下一步) |
| <a id="5-数据评价与执行约束"></a>5. 数据、评价与执行约束 | [原文](reports/knowledge-history-20261022/roadmap.md#5-数据评价与执行约束) |
| <a id="6-避免重复"></a>6. 避免重复 | [原文](reports/knowledge-history-20261022/roadmap.md#6-避免重复) |
| <a id="7-核心术语"></a>7. 核心术语 | [原文](reports/knowledge-history-20261022/roadmap.md#7-核心术语) |
| <a id="8-文档与历史入口"></a>8. 文档与历史入口 | [原文](reports/knowledge-history-20261022/roadmap.md#8-文档与历史入口) |

</details>
