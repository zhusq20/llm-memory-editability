> 历史快照：2026-10-22文档压缩前的正文。文中的“当前／下一步／运行中”只代表记录当时；最新研究优先级见[主路线图](../../roadmap.md)，Loop见[独立路线图](../../loop-roadmap.md)，恢复须读[当前工作入口](../../../AGENTS.md)。此处保留完整证据、负结果与旧判断，不构成新任务授权。

# 研究路线：知识在哪里、推理在哪里、二者怎样连接

更新：2026-10-22（America/Chicago）。本文是唯一的当前研究规划；结果、推导与执行记录分别维护在[结果汇总](../../results.md)、[理论依据](../../memory-scaling-theory.md)和各批实验协议中。改写前的版本见[快照](../roadmap-before-core-story-20261020.md)。

<a id="mission-20261020"></a>
<a id="mission-20261019"></a>
<a id="question"></a>
## 1. 核心故事

**使命：理解Transformer怎样把学到的事实变成可以灵活使用、持续积累和可靠更新的知识，并找到实现这种能力的条件。**

**一句话：Transformer在实体出现的位置查知识，在下游做推理，一次前向里信息只往下游流；训练信号却来自最末端的答案。我们观察到，单靠答案写入的事实，地址往往是“问题”、内容是“答案”：能回答，却未必能被推理当作实体使用。推理得到的新实体，也回不到已经经过的知识层。架构决定能否再次访问这些参数，以及以什么形式交接。**

| 对象 | 功能定义 | 自然模型中的典型位置 |
| --- | --- | --- |
| 知识 | 以实体为地址的计算：换实体就变，换问题不变 | 实体所在位置（[实测](../../results.md#knowledge-reasoning-map-v1)：MLP地址以实体为主，问题在前时中段兼读问题） |
| 推理 | 以问题为地址的计算：换问题就变，对所有实体相同 | 最后位置（[实测](../../results.md#knowledge-reasoning-map-v1)：前2/3层主要随问题变化；约第20层实体依赖增强，后段联合份额上升。份额本身不等于推理或答案的直接定位，见[证据核对](../layer-evidence-chain-20261021.md)） |
| 连接 | 知识以实体形式交给推理；推理结果要再查知识，必须回到知识所在的深度，并以那里认得的形式到达 | — |

**实体形式**指模型读到该实体的词时自己形成的表示；**答案形式**指直接推高该实体词元的输出方向。

<a id="practical-questions"></a>
**现实意义（2026-10-20用户采纳）。** 这条主线回答几类常用方法的问题：
- 上下文蒸馏为什么在某些用途上失效；
- LoRA写入的知识能否被自然运用；
- 测试时训练写入的记忆能否支持推理；
- 上下文学习为什么能推理，它与把知识写进参数有什么机制区别。

四个问题共用一个答案：上下文学习与梯度写入都是外积写入，区别在谁提供键（[§3.3](#write-modes)）。上下文以实体为键、每层一份，组合用途在推理之前的早中段就把它读入；微调、LoRA、上下文蒸馏和TTT以问句为键，内容是答案形式，写在推理之后的位置与深度。引言以这组问题开篇。

论文由四个核心部分组成。第四部分架构由用户于2026-10-20确定为核心层次，不是旁证；其中Loop的实验证据与新设计归[Loop论文](../../loop-roadmap.md)，本文只引用其结论。

1. **会答：** 知识的地址。
2. **能组合：** 知识到推理的单向连接。
3. **更新后能用：** 注入就是答案监督。
4. **架构：** 写到哪一深度、以什么形式交接。

<a id="associative-memory-contribution"></a>
### 从联想记忆到可调用的知识

[Zhong等《Understanding Transformer from the Perspective of Associative Memory》（2025预印本）§2–3](https://arxiv.org/pdf/2505.19488)用键、值和匹配函数统一描述MLP与attention，并把读取质量与更新规则分开；DeltaFormer将非线性读取与delta更新结合。[Alex Wang的博客（2026）](https://alexwang.ai/posts/associative-memory/)从减少串扰解释线性映射、MLP和softmax读取，也指出梯度更新可以看成一次记忆写入。博客的干扰分析受前文启发，二者不是两份独立验证。

**我们进一步解释：训练形成的关联，怎样成为可跨用途调用、可组合、可持续更新的知识。** 沿用联想记忆的语言，把一次写入放回完整Transformer的层、位置与后续计算中，回答三个相连的问题：

- **谁能读到：** 写入使用实体状态还是完整问句作地址，决定它能服务哪些问法与用途。
- **读到后怎样继续：** “说出实体”的答案方向不自动成为下一跳认得的输入；架构须同时提供可访问的参数和兼容的交接形式。
- **改变它会影响谁：** 同一写入对其它用途的作用，可以是需要的迁移，也可以是对旧事实的干扰；更新须同时处理目标事实、依赖它的后果和地址邻居。

贡献在于把这些关系解释清楚，并用[已有实验的论证链](#evidence-chain)连接起来。外积梯度、键值表示和链式法则是已有数学工具；我们要建立的是它们在知识调用中的具体含义。完整推导见[理论文档：从联想记忆到跨用途调用](../../memory-scaling-theory.md#associative-memory-to-use-20261020)。

<a id="principles"></a>
<a id="theory"></a>
<a id="path-consistency"></a>
## 2. 两条原则

### 原则一：单向连接

自然事实的使用分三步（[Geva等2023](https://arxiv.org/abs/2304.14767)）：
1. 主语位置的前中段调出实体属性；
2. 关系信息移到最后位置；
3. 最后位置的注意力按关系读出属性。

[Back Attention](https://arxiv.org/abs/2502.10835)把它细分为四步。层只往上，位置只往前看，因此一次前向里只有“知识→推理”。推理得到的实体出现在下游。要以它为地址再查知识，只有两条路：在下游另存一份知识，或经过一条回到上游的回路。

### 原则二：答案监督的地址与内容

MLP输出权重在普通SGD下一步写入为

`ΔW = −η Σ_p δ_p a_pᵀ`

其中`a_p`是进入输出投影的激活，相当于写入地址；`−δ_p`是后续完整计算传回的下降方向，相当于写入内容。共享参数还须对各次执行求和。这个外积关系是恒等式；它不单独决定哪个位置梯度最大，也不保证内容等于答案实体的输入表示。结合实测，我们得到以下解释：

- **地址：** 答案损失直接作用于预测位置；更早位置经注意力接收间接信号。后段写入在实测中集中于问题末位置，地址因此携带整个问句。
- **内容：** 答案位置的`−δ`在后段接近unembedding中v的方向，推动模型“说出v”。
- **用途约束：** 单跳问答“h的R是v”无法区分“记住答案”与“认识v这个实体”。让多种用途（v在哪国、在哪个洲……）共享同一次写入，提供了形成可复用实体表示的约束；是否形成，要由新用途的表现与内部测量共同判断。

**两条原则合起来：** 答案监督容易学到服务原问句的关联；新的用途须重新触发这次写入，并把读出的内容接入后续计算。实体入口提供了跨用途共享地址的机会，回路则提供了重演原问句计算的机会。

**实测（[答案梯度的地址与内容](../../results.md#answer-gradient-address-v1)）：** 在两个Qwen模型上：
- 越往后层，down投影的梯度越集中在预测答案的位置，主语份额降到0；
- 答案位置的写入方向在后段就是“说出v”；
- 问v本身、问v在哪国、问v在哪个洲，只在主语位置要求相似的写入。

主语位置的梯度在中段指向实体v，这在Qwen3.5-2B上明显，在Qwen3-1.7B上很弱，与两者E3写入落在前段/后段一致。

**训练出的写入同样落在问句上（[写入位置分解](../../results.md#write-position-20261020)）：** 在三种小模型的连续写入中，把每次写入只放在答案位置，刚写入的事实全部答对；只放在主语位置，答对0–5%（Loop取消截断为22.7%）。

**已知事实上的位置×层地图（[知识与推理地图](../../results.md#knowledge-reasoning-map-v1)）：** 两个Qwen上，最后提示token在前2/3层的MLP地址主要随问题变化（问题份额.81–.92，实体份额.06–.15）；约第20层实体依赖明显增强，后段联合份额为.20–.23。这描述输入依赖的变化，不能仅据联合份额认定答案已经形成。答案监督的逐位置梯度范数质量在后段有.47–.57落在最后提示token、约.19在回复位置，实体token只有.04–.06。这支持写入主要作用于携带问句信息的末位置；这些份额不是实际训练后参数增量的加性分解。

**预训练写入的地址（[bioS属性的地址](../../results.md#bios-write-address-v1)）：** 单一传记的属性以前文为地址，换了名字仍答原人物的属性（92–94%）；多表达加句序变换的属性以名字为地址，换名后84–91%跟名字。同一事实不同表达的写入只在名字上一致，多表达训练因此把写入累加到名字上（名字组份额.52–.66，单一传记约.13）。这是“用途约束”在预训练中的直接测量。

**数学依据。** 固定输入激活，一次写入对另一地址的直接作用为`Δm(a′) = −η Σ_p δ_p〈a_p, a′〉`；完整模型还要经过后续计算。局部关系`Δf(u) ≈ J(u)Δθ`与[式31](../../memory-scaling-theory.md#untied-second-hop-kernel-20261010)把地址重合和下游作用接起来：有重合才可能传递，但项间抵消、内容方向和答案间隔仍决定行为是否改善。新用途上的作用是迁移，旧用途上的作用可能是干扰，二者由同一更新关系描述。

[J-space](https://transformer-circuits.pub/2026/workspace/index.html)以平均下游Jacobian作为读出，发现中段有一个供推理使用的工作区：多跳依赖它，一步回忆可以绕过它。这与“答案形式绕过推理”一致。我们借鉴它的写法：先给概念与功能判据，每条判据配一个直观单例再给汇总。不在Qwen上复现它的结论。

<a id="two-routes"></a>
### 总纲：重演原计算，或把地址挪到实体

按式31，一次写入要在新用途上起作用，需要同一组参数上的激活重合，并经下游计算产生有效影响。我们的答案监督写入主要在“原子问句走完那组层”时被触发。已有证据据此组织成两条调用路线：

| 路线 | 做法 | 实例 |
| --- | --- | --- |
| **甲：重演原计算** | 新用途走同一组层，输入同样形式，排成同样的问句 | CoT把桥实体变回输入词，从第0层重新算；DiscoLoop第二轮复用同一组层，词表把答案变回输入词，两跳题前缀`[h, r₁]`本身就是原子问句，r₂位置看到的近似原子问句`[b, r₂]` |
| **乙：把地址挪到实体** | 写入以所有用途都经过的实体入口为地址 | [表达多样化（bioS）](../../results.md#bios-write-address-v1)、多用途练习、MEMIT、早期实体监督、Qwen3.5-2B的E3写入 |

四个部分据此分工：
- **第1–3部分：** 答案监督写出的事实服务原问句，即路线甲；表达多样化、用途练习与实体式写入把写入推到乙。
- **第4部分：** 架构实现甲。“深度”与“形式”是重演的两个分量：深度让第二跳用上学会这条事实的同一组层，形式让进入这组层的输入与原子问句兼容。只修一个分量时OO ≤6.2%，两个都修时为95.66%。

早先的“路径一致性”假说就是路线甲；乙是第二条路。

**DiscoLoop走甲（零训练，[结果](../../results.md#write-position-20261020)）：** 把后写事实的写入只放在两跳问句的一个位置，第二跳在r₂位置被读到（32/32），在再编码桥实体的r₁位置读不到（0/32）；第一跳在r₁位置被读到（31/32）。每一跳都在重演自己原子问句的位置被读到，再编码的桥实体是输入，知识不在那里。

<a id="parts"></a>
## 3. 四个部分

<a id="recall"></a>
### 3.1 会答：知识的地址

**主张：** 会答说明训练过的查询能读出答案；跨问法提取还要求新查询触发同一关联。问题地址容易依赖模板，反向提问又需要另一个实体作输入。表达多样化把地址从前文移到名字：单一传记的属性以前文为地址，多表达加句序变换的属性以名字为地址；只换表达时，紧跟名字的属性以名字为地址，前文越长越以前文为地址。

| 证据 | 内容 |
| --- | --- |
| [bioS表达组织](../../results.md#bios) | 单篇/五种表达/表达加句序变换，规范QA为5.07/35.65/99.96% |
| [bioS属性的地址](../../results.md#bios-write-address-v1) | 换名不换前文：S 92–94%跟前文、0.2%跟名字；MP 84–91%跟名字；M的出生城市跟名字38–52%，公司跟前文85–91%。名字组写入份额MP .52–.66、M .29–.35、S .13 |
| [顺序注入后的问法迁移](../../results.md#sequential-phrasing-20261011) | 原问法100%，留出问法37.61–48.62% |
| [下游段写入](../../results.md#knowledge-read-interface-v2-reanalysis) | 只在原问句触发；问“在哪国”时答主语自己的国家（.60–.95） |

**要写进来的已有工作：**
- Physics of LMs 3.1：表达多样化后，属性才存到人名位置。我们在同一原版设定上用换名测试直接测出两种地址，并用写入分解给出原因（上表）。
- 反转诅咒。
- [同权重的提取与规则使用](../../results.md#bios-same-weight)：分步提取改善使用。

<a id="composition"></a>
### 3.2 能组合：单向连接

**主张：** 第一跳的结果在下游才出现；若第二跳所需的关联只在已经经过的层中可读，一次前向就无法再调用它。组合经历可以在更深处建立读取，显式写出桥实体则把第二跳变成一次新的原子查询。两条事实分别会答，仍不足以保证这次交接成功。

| 证据 | 内容 |
| --- | --- |
| [组合使用经历](../../results.md#grok-usage) | 有组合经历100%，匹配原子重复25.01% |
| [桥实体读出](../../results.md#knowledge-read-interface-followup-v1) | U3桥位置从未形成h（20/20格）；显式写出h即可恢复 |
| [新增事实两跳](../../results.md#ttcd-novel-fact-transfer-v1) | 分步251/251，内部两跳0/251 |
| [分别学两条新事实](../../results.md#ce-cross-sample-double-new-v1) | 标准Transformer两条都记住（64/64），组合0/64 |

**要写进来的已有工作：**
- [Wang等2024](https://arxiv.org/abs/2405.15071)：分布外组合失败，是因为上层没有第二跳事实。
- [Hopping too late](https://arxiv.org/abs/2406.12775)。
- [Balesni等two-hop curse](https://arxiv.org/abs/2411.16353)。
- Yang等2024，隐式多跳推理。
- J-space：多跳依赖中段工作区。

<a id="updating"></a>
<a id="sequential-transfer"></a>
<a id="write-interface"></a>
### 3.3 更新后能用：注入就是答案监督

**主张：** 当微调、上下文蒸馏或快权重只用最终答案监督时，都面临原则二的“问题地址＋答案内容”：
- 地址宽时，问什么都答v，操作被覆盖；
- 地址窄时，换个问法就不触发。

以实体为地址的写入几乎不答v，可以作第一跳。补救办法是让多种用途共享同一次写入。

<a id="write-modes"></a>
**一个公式看四类写入（2026-10-20用户采纳）。** 上下文学习与梯度写入都是外积写入，区别在谁提供键与值：

| 写入 | 键（地址） | 值（内容） | 可读深度 |
|---|---|---|---|
| 上下文：注意力`Σ_i v_i k_iᵀ` | 上下文中实体词元自己算出的状态 | 模型对该词元内容的编码，实体形式 | 每一层各一份 |
| 梯度：`−η Σ_p δ_p a_pᵀ`（微调、LoRA、上下文蒸馏、TTT） | 损失位置（问句末）的状态 | “说出答案”的方向 | 梯度选定的深度 |

[Dai等2023](https://arxiv.org/abs/2212.10559)把两式当作对偶、强调相似；我们强调键的来源不同。两条原则在常用方法上就是这个样子。四个论点：

1. **上下文蒸馏把教师的答案写到被问过的问句上。** 教师靠注意力以实体为键读上下文；蒸馏只监督教师在蒸馏问句上的输出，学生得到问句键与答案内容。二者在蒸馏问句及其近邻上一致，在需要把事实当实体输入的用途上分开。
   - 同一组事实上，M3换问法.87–.98，直接作第一跳.01（上下文中.23–.52）；T1≈0、T2 .87–.90，写在主语下游。
   - DiscoLoop上的测试时上下文蒸馏：新事实只在第二轮最后一块出现，内部两跳0/251，分步251/251；同一基础模型用原子CE写同样32条新事实，两跳100%。
   - 成功的蒸馏都扩大了用途覆盖：[Padmanabhan 2023](https://arxiv.org/abs/2306.09306)用实体续写作迁移集，[Cartridges](https://arxiv.org/abs/2506.06266)用合成对话做self-study，[Lampinen 2025](https://arxiv.org/abs/2505.00661)把上下文中推出的结论加入微调数据。本项目的实体式写入＋用途练习使未练事实直接作第一跳从.26升到.85。
2. **LoRA与全参数微调写到同一个地址；能否运用，由写入的键和深度决定。** LoRA的梯度`∂L/∂B = Σδ(Aa)ᵀ`、`∂L/∂A = BᵀΣδaᵀ`由同一组δ与a构成。
   - E3的逐事实秩一增量（近似逐事实LoRA）与新增通道在同一模型内写在同一段、第一跳同级：Qwen3-1.7B都在后段（.00–.02），Qwen3.5-2B都在前段（.35–.58）。
   - 只训练down投影与全参数同级；只换训练层段就翻转失败类型。
   - ROME/MEMIT同为秩一更新，键是主语末词元：T1 .94/.71，第一跳直接.24/.12，梯度方法约为0；交到答案位置的内容在两个模型、两组事实上都偏实体形式（[G2](../../results.md#write-mode-readout-v1)）。
3. **TTT把材料存成“前缀→下一词”，推理靠重新提问用上它。** 文档上的下一词损失在每个位置都是答案监督：键是前缀，值是下一个词。
   - 陈述SFT（M1，最接近在陈述文本上做TTT）在Qwen3-1.7B上换问法.45–.57、第一跳直接.00–.04；TTCD内部两跳0/251、分步251/251。
   - 支持推理的两条路正是[总纲](#two-routes)的甲（CoT、DiscoLoop重演原子问句）与乙（self-study、[SEAL](https://arxiv.org/abs/2506.10943)的推论数据）。
   - 本项目重建的[PERK](https://arxiv.org/abs/2507.06415)在BabiLong QA2上为13.5%，上下文推理微调FT-ICR为98.0%；原论文报告PERK占优，此条只作参照。
4. **上下文学习以实体为键、每层一份；组合用途在推理之前的早中段，从问句把事实读入。**
   - 地址宽：上下文中换问法1.00（参数写入.45–.99），分步两跳.94–1.0（上下文蒸馏.18–.35）。
   - 深度（[G3](../../results.md#write-mode-readout-v1)）：第一跳、第二跳与捷径题依赖问句在早中段读取上下文，读取重心比直接回忆早1.2–5层；直接回忆在后段由答案位置读取。第二跳由问句中的实体关联直接找到记录，不是先形成桥实体再读（预定的“先桥后读”四组都不成立）。
   - 内容（[G2](../../results.md#write-mode-readout-v1)）：上下文交到答案位置的v特异内容约为梯度写入的1.4–6倍。形式随模型而定，并与行为一致：Qwen3.5-2B以实体形式为主（干扰下第一跳.21–.25），Qwen3-1.7B以答案形式为主（干扰下答v .69–.71）。
   - 上下文写入随序列丢弃，不伤参数邻居；干扰变成注意力竞争（同关系干扰把直接使用减半），与下文的地址邻居对应。

[G1–G3](../../results.md#write-mode-readout-v1)（零训练，已完成）在同一组事实上比较了上下文与五种参数写入：四种梯度写入直接作第一跳≤.06、答v .09–.86；以主语为键的MEMIT几乎不答v，交到答案位置的内容偏实体形式；上下文的事实在组合所需的深度被读入。

改写旧事实时，答案式写入只改了原问法的输出，入口里仍是旧值，所以多跳题继续给出旧答案。

连续写入多条事实时，写入的影响按两种方式扩散：
- **约束框：** 所有写入锁在同一个参数框里时，后写的事实覆盖先写的，与两者是否相关无关。
- **地址邻居：** DiscoLoop放开约束后，新事实保留并能组合，损伤集中在与写入同主语、同关系的旧事实上；答案相同的旧问题反而被推高（内容是答案）。位置分解表明，损伤主要来自答案位置上的写入。答案位置的问句状态携带主语和关系，这支持按地址重合解释干扰；语义分组本身不等于已经测出全部激活距离。

据此选择旧知识保护对象：DiscoLoop不加截断、只保护新事实的同主语与同关系旧事实，连续写入64条后新事实100%、双新组合98.44%、旧原子保持99.22%、旧组合保持96.99%。没有进入保护集的同主语旧事实也随兄弟一起得到保护。Loop的写入影响超出这两个地址，邻居保护只部分有效。

**在联想记忆视角上进一步解释的内容：** 写入须兼顾应当改变的新用途与应当保留的旧用途。我们把干扰落实到哪些问题、哪些位置受影响，再据此选择保护对象。上下文蒸馏也须保留事实在其它问题中的作用；只复现原答案，没有约束这些用途。

| 证据 | 内容 |
| --- | --- |
| [层段翻转](../../results.md#knowledge-read-interface-v2-reanalysis) | 两臂只差训练层段：接口段答v .51–.82，下游段答主语的国家.60–.95 |
| [梯度注入与MEMIT](../../results.md#knowledge-read-interface-development-v1) | 四种梯度注入T1≈0，写在主语下游；MEMIT以主语为地址，T1 .94/.71 |
| [新增事实快写入](../../results.md#ttcd-novel-fact-transfer-v1) | 新事实只在第二轮最后一块末出现（32/32），首轮0/32，写在离损失最近的深度 |
| [主语状态写入E1](../../results.md#knowledge-read-interface-v2) | 几乎不答v（≤.08） |
| [新增参数写入E3](../../results.md#knowledge-read-interface-v2) | Qwen3.5-2B前段写入，第一跳直接.35–.58；Qwen3-1.7B后段写入，为答案形式；[改用同一学习率1.5e-4](../../results.md#e3-learning-rate-control-v1)后Qwen3-1.7B仍写在后段，写入深度随模型而定 |
| [用途练习E4](../../results.md#knowledge-read-interface-v2) | 只在一半事实上练两种用途，另一半事实的“在哪国”分步提高.39/.70，修复的是操作被覆盖 |
| [实体式写入＋共享用途练习](../../results.md#use-adapter-development-v1) | Qwen3.5-2B上，没练过的事实直接“在哪国”从.26到.85；Qwen3-1.7B的答案式写入只改善分步，直接仍≈0 |
| [写入目标](../../results.md#write-target-editing-v1) | 改写原子同样24/24成功；只用最终答案时多跳题仍给旧答案72/96、传播3.13%；加早期实体监督后传播68.75%，再加方向对齐85.42% |
| [连续写入64条事实](../../results.md#ce-cross-sample-double-new-v1) | DiscoLoop双新组合从单独一对的85.94%降到1/64，新原子只保留3/128 |
| [写入约束消融](../../results.md#ce-cross-sample-ablation-v1) | DiscoLoop取消截断后双新组合95.31%，旧组合保持从99.53%降到80.93%；Loop降低学习率保住新原子82.03%，组合仍为0% |
| [连续写入的损伤分组](../../results.md#sequential-write-interference-20261020) | 零训练。DiscoLoop取消截断后，每次写入使同主语旧事实NLL升4.82、同关系升2.40、无关旧事实−0.01，答案相同的旧问题降0.15；有截断时，每次无关写入后先写的新事实丢失48% |
| [按地址邻居保护](../../results.md#neighbor-protection-development-v1) | DiscoLoop无截断、KL只保护同主语与同关系旧事实：新事实128/128，双新63/64，旧原子99.22%，旧组合96.99%（随机保护时87.25%、80.93%）；Loop旧原子8.59%→50.00% |

**要写进来的已有工作：**
- MQuAKE、RippleEdits的传播失败。
- Editing Overfit（Zhang等ICLR 2025）：编辑后多跳题直接答新对象。
- ROME/MEMIT。
- Hase等2023：定位不等于最佳编辑位置。
- [Cortex](https://arxiv.org/abs/2610.20399)。
- 上下文蒸馏：[Snell等2022](https://arxiv.org/abs/2209.15189)；[Padmanabhan等2023](https://arxiv.org/abs/2306.09306)；[Cartridges](https://arxiv.org/abs/2506.06266)：在语料上直接做下一词训练追不上上下文学习，self-study追上。
- 上下文学习与微调：[Lampinen等2025](https://arxiv.org/abs/2505.00661)，数据匹配时上下文学习泛化更灵活；[Dai等2023](https://arxiv.org/abs/2212.10559)，注意力与微调的对偶形式。
- 参数写入与检索：[Ovadia等2023](https://arxiv.org/abs/2312.05934)，检索优于无监督微调；[Biderman等2024](https://arxiv.org/abs/2405.09673)，LoRA学得少、忘得少。
- 测试时训练：[PERK](https://arxiv.org/abs/2507.06415)、[SEAL](https://arxiv.org/abs/2506.10943)。
- [顺序知识注入](../../results.md#sequential-transfer-v1)、[旧组合回放](../../results.md#sequential-replay-v1)、[知识变化中学习调用](../../results.md#knowledge-change-v1)、[自然模型编辑复现](../../results.md#path-hop-analysis-v1)。

<a id="architecture"></a>
### 3.4 架构：写到哪一深度、以什么形式交接

**主张：**

- **A1 深度：** 标准Transformer每层参数独立，一层的写入不会自动出现在后面的层。若第二跳到达时已错过能读取该事实的层，就要在后续层另行学会读取。
- **A2 共享：** 参数共享让同一次参数更新在每轮执行时都可访问；是否真正被读到，还取决于该轮的输入。
- **A3 形式：** 上一轮末尾的隐藏状态不是入口认得的实体，交接必须以实体形式到达入口。有两种办法：
  - 用损失把交接状态对齐成实体形式；
  - 经过词表（解码再编码、CoT），在构造上给出实体形式，并让只用答案学来的事实也能组合。词表是天然连接两种形式的地方：unembedding把实体变成答案，embedding把答案变回实体。经过词表的交接要发生在一跳刚完成、模型说出的正是桥实体的时候。
- **A4 分工：** 回路解决组合；连续写入中的保持归第三部分。

**本文的证据：**

| 架构 | 写到哪一深度 | 交接形式 | 结果 |
| --- | --- | --- | --- |
| [标准Transformer](../../results.md#ce-cross-sample-double-new-v1) | 只在训练用途的深度 | — | 分别学两条新事实，都记住（64/64），组合0/64 |
| [标准两层＋表示对齐](../../results.md#independent-alignment-v1) | 只在训练用途的深度 | 对齐成实体（余弦.9997） | 分布外组合5.27% |
| [标准Transformer＋经过词表的交接](../../results.md#untied-discoloop-development-v1) | 只在训练用途的深度 | 经过词表 | 测试II 99.87%，OI/IO/OO 36.17/19.47/6.19% |
| [CoT](../../results.md#ttcd-novel-fact-transfer-v1) | 每一跳都变成第一跳 | 经过词表 | 新事实分步251/251，内部两跳0/251 |

**深度与形式缺一不可。** 同一Wang图与配方上，只修形式（未共享＋词表交接）与只修深度（共享＋隐藏状态交接，引自Loop论文）的OI都约35%，OO都≤6.2%；两者都有（DiscoLoop）时OO为95.66%，训练计算与只修形式的一格相同。深度与形式是[总纲](#two-routes)中路线甲的两个分量。

预定的“未共享＋词表交接OI接近DiscoLoop”没有成立，实测为36.17%；这一负结果保留，不能因OO比较支持当前解释而改写原预测。

**组合读的是重演出的问句位置（[零训练](../../results.md#write-position-20261020)）。** DiscoLoop中第二跳的写入在r₂位置被读到，第一跳在r₁位置被读到。同一次写入只放在自己的问句位置时，普通Loop能组合18/32与20/32，放在全部位置时只有3/32与0/32。DiscoLoop两种顺序分别为32/32对31/32、31/32对24/32：它能重演读取，也仍可能在其它位置误触发。这把交接与干扰接到同一个解释中。

**引用Loop论文的结论**（[Loop路线图](../../loop-roadmap.md)）：
- 参数共享时，同一次写入在每一轮各司其职：第一轮那份服务第一跳，第二轮那份服务第二跳；拆开参数只优化第二轮，找不到有效写入（[编辑恢复](../../results.md#edit-restoration-20261011)）。
- 共享参数但交接隐藏状态时，只学过原子的事实仍不能组合；把交接推向答案更差（[对照](../../results.md#supervised-alignment-development-v1)）。
- 交接对齐成实体后，共享模型的分布外组合大幅提高，标准模型不提高（[共享小模型](../../results.md#representation-alignment)）。
- 经过词表的交接（DiscoLoop）使只学过原子的事实几乎全部可组合，分别写入的两条新事实也能组合；代价约为2倍训练计算。
- 每轮层数太少、一跳跨了轮次时，交接点上还没有桥实体，经过词表的交接也失效（[轮数网格](../../results.md#loop-hop-unified-v1)）。
- MoE、加宽、跨轮KV、mHC、增加轮数都不改变深度和形式，没有稳定收益。

**要写进来的已有工作：**
- Wang等2024。
- Back Attention：低层读高层状态，是一条不经过词表的深度回路。
- Saunshi等2025，循环Transformer。
- [Coconut](https://arxiv.org/abs/2412.06769)：把隐藏状态直接送回输入，按A3会遇到形式不匹配。
- [DiscoLoop](https://arxiv.org/abs/2607.00341)、[InfiLoop](https://arxiv.org/abs/2610.11570)。
- [Engram](https://arxiv.org/abs/2601.07372)：把存储显式放在入口。

<a id="next-experiments"></a>
## 4. 成文方式与下一步

**写法（借鉴J-space的功能判据与单例、联想记忆博客的短公式解释）：**
1. **2026-10-21用户修订：先整理“逐层表示＋逐层干预＋写入分解”，再确定中心图。** 主图采用保留原层号的三列实测：实体/末位置的表示依赖，主语状态替换与读取阻断，写入位置与内容方向。方法终点比较用表格。此前“知识在左下、推理在右侧”的模块示意不再作为开篇主图，见[逐层证据整理与图稿](../layer-evidence-chain-20261021.md)。
2. 每条主张先给一个单例，再给汇总。例如同一条注入事实问三个问题：原问法答对；问“在哪国”答v（内容是答案）；换成下游段训练再问，答主语自己的国家（地址是问题）。
3. 第1–3部分统一用答案类型分解（正确／答v／答主语的属性），第4部分统一用“深度×交接形式”的架构表。
4. 用“不需要它的用途”划边界：答案形式对原问法完全够用，所以只看单跳正确率会高估“学会了”。
5. 第2–4部分的讨论按[总纲](#two-routes)的甲、乙两条路组织。
6. 数学先用“外积写入→地址重合→下游作用”讲清，再接完整式31；每处明确哪些是恒等式、哪些由实验识别，不用容量或SNR替代知识调用证据。
7. 引言以[现实问题](#practical-questions)开篇：上下文蒸馏、LoRA、TTT与上下文学习；第3部分用[四类写入表](#write-modes)回答。

<a id="layer-evidence-chain"></a>
**本次成文优先：内部过程的证据链。** [已整理](../layer-evidence-chain-20261021.md)各测量的层内位置、样本、评分与可连接范围，并从保存的逐层数值生成两模型的同布局图。表示地图与其写入位置分解共享提示；自然城市问答干预与新事实梯度分别提供补充证据，不能拼成同一案例的完整因果过程。实体/问题/联合份额描述输入依赖，不直接等同于存储/推理/答案的定位。Qwen3-1.7B的晚层读取干预清楚，Qwen3.5-2B对应阻断效应弱，正文须保留这一差异。桥位置读出未支持普遍的“桥实体形成太晚”；按实际功能互换与回补结果解释。本次仅整理、复算与作图，没有新增模型执行，既有方法比较按原授权继续。

<a id="evidence-chain"></a>
**已有实验的论证链。** 按下列问题顺序串联正文，各部分的完整结果仍保留在§3与结果汇总中。

| 环节 | 关键证据 | 得到的解释与下一步问题 |
| --- | --- | --- |
| 1. 原问题会答，知识是否已经可用？ | [问法迁移](../../results.md#sequential-phrasing-20261011)：原问法100%，留出问法37.61–48.62%；[两条新事实](../../results.md#ce-cross-sample-double-new-v1)：标准模型两条都记住64/64，组合0/64 | 回答成功只验证原查询；需要解释新用途为何读不到、读到后为何接不上 |
| 2. 答案监督究竟写了什么、写在哪里？ | [梯度分解](../../results.md#answer-gradient-address-v1)给出地址与内容；[位置分解](../../results.md#write-position-20261020)中，只施加答案位置写入即可答对刚写入的事实；[位置×层地图](../../results.md#knowledge-reasoning-map-v1)中，已知事实的答案监督写入在后段集中于以问题为地址的最后位置 | 问句地址与答案方向把行为失败接到实际计算；梯度解释与训练后位置干预相互补充 |
| 3. 怎样使取出的内容服务其它用途？ | [层段翻转](../../results.md#knowledge-read-interface-v2-reanalysis)改变错误答案类型；[实体式写入＋用途练习](../../results.md#use-adapter-development-v1)把未练事实直接使用从.26提到.85；[改写目标](../../results.md#write-target-editing-v1)在同为24/24原子编辑成功时，传播从3.13%提到68.75%／85.42%；[bioS地址](../../results.md#bios-write-address-v1)中，多表达让名字成为各表达共享的写入地址，换名后MP 84–91%跟名字、S 92–94%跟前文 | 写成原答案与写成后续能用的内容有区别；多个用途须能共享有效写入 |
| 4. 第二跳怎样再次访问同一关联？ | [深度×形式比较](../../results.md#untied-discoloop-development-v1)：未共享＋词表交接OO 6.19%，共享＋词表交接95.66%；[位置分解](../../results.md#write-position-20261020)中第二跳只在r₂施加写入为32/32、只在桥实体r₁为0/32 | 深度与形式共同支持重演读取；能组合并不要求先把答案式记忆全部搬到实体位置 |
| 5. 能用的新知识，怎样连续积累？ | [约束消融](../../results.md#ce-cross-sample-ablation-v1)保住新知识却损伤旧知识；[损伤分组](../../results.md#sequential-write-interference-20261020)与位置分解定位干扰；[邻居保护](../../results.md#neighbor-protection-development-v1)使DiscoLoop新事实100%、双新组合98.44%、旧原子／组合保持99.22%／96.99%，Loop只部分改善 | 同一写入既产生迁移也产生干扰；组合回路与写入保护各自解决一个问题 |

这条论证把联想记忆的存取语言推进到**监督如何选择写入的地址与内容、内容如何接入下一步、更新如何同时影响多个用途**。它由行为比较、实际梯度、位置干预和架构比较共同支撑；各环节按自己的实验解释，不合并成一项实验，也不把事后串联改写为事前统一预测。

未支持原解释的结果同样保留：损伤分组预先判断“同关系大于同主语”，实测相反；Qwen3-1.7B的答案式写入加用途练习只改善分步，直接使用仍近零；共享全参数用途练习还损伤自然两跳。这些差异要求同时检查地址、内容与旧能力保持。

<a id="natural-knowledge-methods-20261021"></a>
**当前优先：让真实部署的模型自然使用新知识，先诊断已有算法。** 用户于2026-10-21要求先查原论文并实际比较上下文蒸馏、SFT、LoRA和down projection写入，同时调研原生记忆架构；“SST”已明确为SFT。旧M3、E3及小模型TTCD仍是已有证据，不能代替原方法，也不能据其失败断言这些方法一般无效。

**本轮修订：模型固定Qwen3-1.7B，数据与主评价沿用论文。** 用户随后要求扩大数据并补全参数范围：使用作者HotpotQA前1000案例、9928段、每段30问，共297840条问答；SFT/PD分别配全参数、原范围LoRA、仅down矩阵、仅down-LoRA，共8组。主评价按最新要求改用Qwen3-4B（关闭thinking）、原评分消息内容和二元判分，以true数除以有效判分数，另报无效覆盖；新生成取消短答案额外指令。先重评旧320条生成，再三片并行补数据，全部完成后八臂从同一原基础权重开始。完整设计和预算见[新协议](../../development-artifacts/knowledge-method-factorial-1000-v1/protocol.md)；执行以[新批登记](../../results.md#knowledge-method-factorial-1000-v1)为准。原模型与新增对照不同于论文，因此称Qwen3扩展，不冒称原表格完整复现。

第一批采用Prompt Distillation作者的自然文档、单跳合成训练、多跳评价设计，在现有Qwen3-1.7B上比较 **PD-LoRA、SFT-LoRA、SFT-full、SFT-down**，已全部完成训练、独立重载和云端核验。前两臂比较作者各自的整体配方，后三臂固定SFT数据和预算比较更新范围。32个作者案例、312段、9360条训练问答，每臂5轮。训练拟合改善，但预定EM受到明显回答格式变化影响，故本次按用户要求用论文评分协议与Qwen3-4B裁判另行重评。原评分保留，不据EM低分宣称算法无效；扩大数据的八臂为独立新批，不续训冒充复现。[旧协议](../../development-artifacts/knowledge-method-diagnosis-development-v1/protocol.md)与[结果登记](../../results.md#knowledge-method-diagnosis-development-v1)。

后续按结果推进：若原方法有效，下一项比较围绕同一批事实，分别测试原问法、未练问法、与旧知识组合、事实再次修改后的依赖答案；训练材料不提供这些留出用途的答案，并保留原本正确的旧问题。新增事实与改写旧事实分开报告，直接组合与显式分步也分开报告，再解释数据覆盖、参数约束、地址与内容。若无效，先核对训练确实学到内容、教师在场时能完成任务和原文配方差异，不直接跳到架构必需的结论。Padmanabhan的实体续写蒸馏、SEAL与Cartridges分别提供迁移数据、学习材料生成和KV存储参照，按能回答的新问题选取，暂不扩成方法大全。

原生记忆调研优先 **Memory+跨深度共享持久内容**，其次 **Hope-Attention多频率MLP**；其现有证据尚未完整覆盖部署后事实写入、多用途组合和更新传播。Titans/TTT的会话快记忆与永久知识分开判断。详见[原方法与记忆架构调研](../lm-knowledge-methods-and-memory-20261021.md)。本轮不新启动原生架构训练，Loop的已有工作继续由单独路线图维护。

已有论证链继续整理。下面保留历史完成项与次级候选；未完成项不与上述原方法比较同时扩展：

| 项 | 作用 | 新增训练 |
| --- | --- | --- |
| DiscoLoop走甲还是乙 | 零训练的位置分解已回答：走甲（见论证链第4环）。可选的训练确认：第二跳不能逐字重演原子问句的模板 × 原子只学一种问法或多种问法，甲预测单一问法下组合下降、多种问法下恢复；新排布须不能被ID组合训练简单映射回原排布 | 历史候选，未排队；后续由Loop路线图统一判断 |
| Qwen3-1.7B上E3改用1.5e-4学习率 | 区分写入深度随学习率还是随模型变化。**已完成**：[结果](../../results.md#e3-learning-rate-control-v1)，两臂仍写在后段，U2直接.02，写入深度随模型而定 | 2臂各50条事实 |
| 知识/推理位置×层地图 | 开篇图的实测版。**已完成**：[结果](../../results.md#knowledge-reasoning-map-v1)，最后位置前2/3层以问题为地址，答案监督写在后段最后位置；实体token中段兼读问题（K1不成立），K2–K4成立 | 0 |
| bioS写入地址 | 第1部分“表达多样化是否把地址移向实体”。**已完成**：[结果](../../results.md#bios-write-address-v1)，S以前文为地址，MP以名字为地址，M随前文长度移动；B1–B5两世界全部成立 | 0 |
| [G1–G3 四类写入的读出比较](../../development-artifacts/write-mode-readout-v1/protocol.md) | 同一组事实上比较上下文与五种参数写入（陈述SFT、多样SFT、上下文蒸馏、仅down、MEMIT）。**已完成**：[结果](../../results.md#write-mode-readout-v1)。组合用途在早中段从问句读入上下文事实（G3-1“先桥后读”四组不成立，方向相反）；梯度写入交到答案位置的内容约为上下文的1/6–3/4；MEMIT偏实体形式。预定G2-1被全参数微调的整体漂移掩盖，改用事后v特异投影 | 0 |
| 换国家对照 | U3中把国家换成首都不是h的国家：以问句为键的写入仍推高v，以实体为键的上下文与MEMIT不推高。直接检验“谁触发写入”，补上G2-1未能回答的部分 | 0 |
| G4 蒸馏写在哪里 | 待原论文配方比较后再决定是否需要实体位置对齐；不将旧M3的失败预先归因于蒸馏本身 | 次级候选，未入队 |

<a id="read-interface"></a>
<a id="read-interface-v2"></a>
知识调用接口各批（[开发](../../development-artifacts/knowledge-read-interface-development-v1/protocol.md)、[后续](../../development-artifacts/knowledge-read-interface-followup-v1/protocol.md)、[v2](../../development-artifacts/knowledge-read-interface-v2/protocol.md)）已完成，作为第1、3部分的证据。原先的A/B/C与四环节分解由原则二的地址、内容取代。

<a id="loop-native-handoff"></a>
Loop架构方向由单独的[Loop路线图](../../loop-roadmap.md)维护，按[问题索引](../../loop-history.md)核对历史。native四臂已完成；本页不维护Loop排程。

新增实验的判断顺序：**先核对历史实验与假设 → 明确尚未回答的问题及可能的新认识 → 能否让核心故事更直观或更完整 → 现有证据是否已经足够 → 资源代价。** 具体遵循[选题要求](#idea-history-check)。不另建内部特征预测器，不以扩大模型、种子或超参数网格代替机制判断。

<a id="main-setup"></a>
<a id="evaluation"></a>
<a id="efficiency-evaluation"></a>
## 5. 数据、评价与执行约束

- **评价：** 全量结果为主。分别报告直接回答、分步回答、组合后果、无关原子与组合保持；旧正确题的丢失不能被新增正确抵消。新增与改写分开。
- **成本：** 曝光、token、更新数、训练/推理计算、总/激活参数和耗时分别记录；第三部分同时计旧能力保持。
- **执行与恢复：** 依[实验协议](../../experimental-protocol.md)和[项目入口](../../../AGENTS.md)，先读该批`controller-process.json`、`controller-state.json`及调度修订，核对实际LM Docker进程，不按历史GPU表启动。
- **追踪：** 新批默认使用[W&B团队项目](https://wandb.ai/zsqzz/llm-memory-editability)及[默认配置](../../../configs/experiment-tracking-defaults.json)。

<a id="scope"></a>
<a id="history-dedup"></a>
## 6. 避免重复

<a id="idea-history-check"></a>
**选题要求（2026-10-22用户明确）：寻找新的突破点。提出idea之前先看历史跑过什么实验、检验过什么假设，不能把旧实验探究换一种说法当作新的研究目标。**

先核对本路线图、[结果汇总](../../results.md)、[历史去重记录](../roadmap-before-restructure-20261019.md#history-dedup)，并回到相关批次的协议、结果与[项目入口中的用户评价](../../../AGENTS.md)。核查覆盖已完成、失败、停止、运行中与已规划的工作，按实际研究问题、假设和操作判断是否重复，不能只比较实验名称。

每个候选想法先简要写清：

1. **历史做过什么：** 最接近的假设、实验批次、实际操作与结果，包括负结果及用户已否定的延伸。
2. **还缺什么：** 历史证据具体留下了哪个未回答的问题；不能把已有结论重新表述成待验证目标。
3. **这次改变什么：** 实质改变的假设、关键条件或机制，为什么旧实验不能回答，哪些对照与产物可以复用。
4. **可能推进什么：** 新结果将怎样改变现有认识、区分尚未解决的解释或打开新方向；不能只以多跑一组、提高一个指标或补齐网格作为研究价值。

更名、换术语、换叙事，或仅换数据、模型、模块组合和扩大网格，本身不构成新目标。复现、补证与旧方向延伸可以有价值，但必须如实标注；重新研究旧问题须说明具体的新依据和新增问题。找不到实质新增问题时直接说明，不为提出idea而包装旧问题。与外部论文话题重叠的重要问题仍可研究，按上述要求说明本项目的新增认识。

以下内容不进入默认的新实验队列：

- 已完成的表达/提取、原子重复/组合经历、表示对齐、逐轮编辑范围、顺序注入、旧组合回放、知识变化训练、学习率/Adam/截断消融。
- 不改变写入深度或交接形式的架构配方扩展，包括门控、KV、mHC、MoE，以及跳数、宽度、循环次数的大网格。
- 容量幂律、bioS 100k主矩阵、日期更新、少迭代蒸馏及独立TTT论文等旧草案。

用户的逐项评价按原记录解释，未回复不视为认可。

<a id="glossary"></a>
## 7. 核心术语

| 术语 | 本文含义 |
| --- | --- |
| 原子事实 | 单条关系事实`(h, r, t)` |
| 组合与桥实体 | 两跳`(h, r₁, r₂)`先得到桥实体b，再由`(b, r₂)`得到答案 |
| 知识／推理 | 以实体为地址的计算／以问题为地址的计算 |
| 地址／内容 | 写入被什么状态触发／写入了什么方向 |
| 读取键／写入地址 | MLP输入权重中的特征检测方向／一次输出权重更新使用的隐藏激活；两种“键”不混用 |
| 迁移／干扰 | 同一次写入对其它用途产生的有益／有害作用；须结合任务目标与完整下游计算判断 |
| 实体形式／答案形式 | 模型读到该实体词时形成的表示／直接推高该词元的输出方向 |
| 入口 | 实体作为输入词被读入、知识被调出的位置与深度 |
| 交接形式 | 推理结果被送回入口时的表示：隐藏状态、经过词表或对齐后的状态 |
| ID/OOD与II/IO/OI/OO | ID事实参加过训练组合，OOD事实只有原子训练；两个字母依次为两跳事实的类别 |
| 普通Transformer／Loop L×T | 各层参数独立／L层共享块执行T次 |
| 分步回答 | 用模型自己的第一步答案构造下一步查询 |
| E/D/R/U | 目标事实／应随之改变的后果／回放材料／不参与选择的保持评价 |

<a id="literature"></a>
<a id="maintenance"></a>
## 8. 文档与历史入口

| 入口 | 用途 |
| --- | --- |
| 本路线图 | 核心故事、四个部分、下一步 |
| [Loop路线图](../../loop-roadmap.md) | 另一工作线：只监督答案的Loop架构设计与批次 |
| [结果汇总](../../results.md) | 完成结果与批次状态入口 |
| [理论依据](../../memory-scaling-theory.md) | 推导 |
| [实验协议](../../experimental-protocol.md) | 共同规则与历史冻结契约 |
| [改写前快照（10-20）](../roadmap-before-core-story-20261020.md) | 知识调用接口主线、A/B/C分解等原文 |
| [重构前历史记录（10-19）](../roadmap-before-restructure-20261019.md) | 更早的讨论与调度记录 |

README只作导航。今后直接修改对应主题，不在顶部累加“当前方案”。

<details>
<summary>旧链接兼容入口（历史内容，不是当前待办）</summary>

| 旧主题或批次 | 记录 |
| --- | --- |
| <a id="first-principles-plan-20261019"></a>第一性原理旧提案与文献复核 | [历史原文](../roadmap-before-restructure-20261019.md#first-principles-plan-20261019) |
| <a id="sequence-dependency-gate-v1"></a>sequence-dependency-gate-v1 | [历史原文](../roadmap-before-restructure-20261019.md#sequence-dependency-gate-v1) |
| <a id="residual-gate-injection-ablation-v1"></a>residual-gate-injection-ablation-v1 | [历史原文](../roadmap-before-restructure-20261019.md#residual-gate-injection-ablation-v1) |
| <a id="candidate-update-gate-gpu1-v3"></a>candidate-update-gate-gpu1-v3 | [历史原文](../roadmap-before-restructure-20261019.md#candidate-update-gate-gpu1-v3) |
| <a id="compact-wang-joint-gate-v1"></a>compact-wang-joint-gate-v1 | [历史原文](../roadmap-before-restructure-20261019.md#compact-wang-joint-gate-v1) |
| <a id="candidate-update-gate-multihop-v1"></a>candidate-update-gate-multihop-v1 | [历史原文](../roadmap-before-restructure-20261019.md#candidate-update-gate-multihop-v1) |
| <a id="joint-update-gate-wang-v1"></a>joint-update-gate-wang-v1 | [历史原文](../roadmap-before-restructure-20261019.md#joint-update-gate-wang-v1) |
| <a id="joint-update-gate-multihop-v1"></a>joint-update-gate-multihop-v1 | [历史原文](../roadmap-before-restructure-20261019.md#joint-update-gate-multihop-v1) |
| <a id="gate-generalization-extension-v1"></a>gate-generalization-extension-v1 | [历史原文](../roadmap-before-restructure-20261019.md#gate-generalization-extension-v1) |
| <a id="paper-update-gate-continuation-v1"></a>paper-update-gate-continuation-v1 | [历史原文](../roadmap-before-restructure-20261019.md#paper-update-gate-continuation-v1) |
| <a id="paper-update-gate-multihop-u95-v1"></a>paper-update-gate-multihop-u95-v1 | [历史原文](../roadmap-before-restructure-20261019.md#paper-update-gate-multihop-u95-v1) |
| <a id="paper-update-gate-multihop-v1"></a>paper-update-gate-multihop-v1 | [历史原文](../roadmap-before-restructure-20261019.md#paper-update-gate-multihop-v1) |
| <a id="gate-evidence-refinement-20261017"></a>门控证据整理历史 | [历史原文](../roadmap-before-restructure-20261019.md#gate-evidence-refinement-20261017) |
| <a id="gated-loop-depth-extrapolation-v1"></a>gated-loop-depth-extrapolation-v1 | [历史原文](../roadmap-before-restructure-20261019.md#gated-loop-depth-extrapolation-v1) |
| <a id="zhang2026"></a>两跳组合的文献分析 | [历史原文](../roadmap-before-restructure-20261019.md#zhang2026) |
| <a id="discoloop2026"></a>DiscoLoop文献分析 | [历史原文](../roadmap-before-restructure-20261019.md#discoloop2026) |
| <a id="weight-tying-pooling"></a>权重绑定与梯度池化提案 | [历史原文](../roadmap-before-restructure-20261019.md#weight-tying-pooling) |
| <a id="feng2025"></a>提取结构文献分析 | [历史原文](../roadmap-before-restructure-20261019.md#feng2025) |
| <a id="pretraining-continual"></a>预训练与持续学习旧论述 | [历史原文](../roadmap-before-restructure-20261019.md#pretraining-continual) |
| <a id="wang-benchmark-boundaries"></a>旧主设定可行性复核 | [历史原文](../roadmap-before-restructure-20261019.md#wang-benchmark-boundaries) |
| <a id="innovation-rnn-ttt"></a>RNN与TTT旧候选 | [历史原文](../roadmap-before-restructure-20261019.md#innovation-rnn-ttt) |
| <a id="memory-mosaics"></a>Memory Mosaics阅读记录 | [历史原文](../roadmap-before-restructure-20261019.md#memory-mosaics) |
| <a id="memory-reuse-solution"></a>旧方法整合目标 | [历史原文](../roadmap-before-restructure-20261019.md#memory-reuse-solution) |
| <a id="dual-gate-design-20261017"></a>双门设计 | [历史原文](../roadmap-before-restructure-20261019.md#dual-gate-design-20261017) |
| <a id="norm-handoff-minimal-proposal-20261017"></a>归一化交接提案 | [历史原文](../roadmap-before-restructure-20261019.md#norm-handoff-minimal-proposal-20261017) |
| <a id="loop-solution-integration-20261012"></a>Loop方法整合提案 | [历史原文](../roadmap-before-restructure-20261019.md#loop-solution-integration-20261012) |
| <a id="paper-reproduction-first"></a>论文复现历史 | [历史原文](../roadmap-before-restructure-20261019.md#paper-reproduction-first) |
| <a id="zhang2026-next"></a>10月10日候选实验 | [历史原文](../roadmap-before-restructure-20261019.md#zhang2026-next) |
| <a id="nl-edit-layer"></a>自然模型编辑层候选 | [历史原文](../roadmap-before-restructure-20261019.md#nl-edit-layer) |
| <a id="history-audit-20261010"></a>历史去重旧清单 | [历史原文](../roadmap-before-restructure-20261019.md#history-audit-20261010) |
| <a id="controlled-multihop"></a>组合再学习开销旧提案 | [历史原文](../roadmap-before-restructure-20261019.md#controlled-multihop) |
| <a id="architecture-efficiency"></a>架构效率旧比较 | [历史原文](../roadmap-before-restructure-20261019.md#architecture-efficiency) |
| <a id="loop-kv-next"></a>Loop与KV比较历史 | [历史原文](../roadmap-before-restructure-20261019.md#loop-kv-next) |
| <a id="moe-sequential-next"></a>MoE顺序学习历史 | [历史原文](../roadmap-before-restructure-20261019.md#moe-sequential-next) |
| <a id="read-interface-20261020"></a>知识调用接口主线与A/B/C分解（10-20） | [历史原文](../roadmap-before-core-story-20261020.md#read-interface) |

</details>
