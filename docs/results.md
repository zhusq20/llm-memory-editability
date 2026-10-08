# 研究结果：知识的回答、组合与更新后使用

本页按论文的三个层次整理已有证据：**知识能被回答，能被组合，更新后能继续被使用。** 当前问题、实验优先级和未执行计划统一见 [roadmap](roadmap.md)。2026-10-07核对历史报告与控制器，删除过时的排队和GPU播报；2026-10-09补入九条论文复现的终点验证与五条冻结跳位分析。旧审计仍是既有证据，不宣称本轮重新重载了全部历史模型。

熟悉组合指必要事实已有组合使用经历的新查询；严格组合指必要事实只接受过原子训练。不同批次的图、评分和角色划分并不完全相同，以下数字须连同各批协议解释。除特别说明，多世界结果先平均世界内初始化，再对世界等权；查询、初始化、编辑和节点不增加独立世界数。外部自产桥调用提供关系分解并增加计算，与原生组合分别报告。

| 层次 | 已有主要证据 | 当前解释 |
| --- | --- | --- |
| [一、能被回答](#recall) | 表达组织、查询方式、知识负载和优化均影响事实提取 | 记住某种续写与通过不同查询提取，是需要分别测量的能力 |
| [二、能被组合](#composition) | 使用经历、支持分布、表示训练和执行结构改变新组合表现；共享表示对齐有强严格组合正例，独立两层复核未复现同等收益 | 单跳掌握和首跳可读出并不足以保证后继使用；条件性正例与负例需要一起解释 |
| [三、更新后继续使用](#updating) | 顺序新增实验中，有/无先前组合练习的BB为20.57/3.12%；局部编辑中，最终答案/早期实体/再加对齐的传播为3.12/68.75/85.42% | 使用经历和写入方式能改善新内容的使用，两项仍有旧组合损伤；新增与改写分别解释 |

<a id="recall"></a>
## 一、知识能被回答：形成、提取与保持

<a id="bios"></a>
<a id="bios-original"></a>
### 表达组织与事实提取：bioS

两个10k人物开发世界中，同样语义事实曝光、不同传记组织，之后接受相同QA适配。固定传记S、五种表达M、表达加句序变化MP的规范六属性QA为 **5.07%／35.65%／99.96%**，五个留出问法均值为0.57%／14.06%／32.90%。表达组织改变提取，但token、难度与实际措辞重复也变化。

预训练540曝光端点的原生日期续写为100.00%／98.44%／95.31%，出生地均100%。这些分数与后来QA来自不同权重，不能据此认定同一权重下只有查询接口失败。任务适配后的MP日期／年份／月份奇偶／日期比较为94.16%／34.10%／75.02%／0%；同一MP-CoT端点允许CoT后年份／奇偶为84.06%／99.39%，日期比较仍0%。新日期对在给真实日期时规则比较仍0/48，原训练规则却全部正确：此处失败含规则迁移问题。

六条预训练、六个QA适配、十二个任务分支及已有权重诊断保留；570万条预测复算、18端点抽样重载通过。100k主矩阵和日期更新未执行。证据：[完整复盘](development-artifacts/bios-original-review-v1/report.md)、[原生学习曲线](development-artifacts/bios-original-trajectory-v1/report.md)、[规则诊断](development-artifacts/bios-original-diagnostics-v1/summary.json)。

<a id="bios-same-weight"></a>
### 同一权重的提取与规则使用

18个既有bioS端点的零训练诊断发现，S原生日期／出生地在预训练端点为100%／100%，QA适配后为32.81%／78.13%；同一适配权重的常用QA为14.06%／1.56%。LoRA之外的主体参数保持不变，因此这是调用行为变化，不能说主体事实被擦除。

MP任务端点的常用／留出问法，日期QA为96.88%／59.38%，直接月份奇偶为76.56%／35.94%，自主生成日期再判断为100%／93.75%，给真实日期判断均100%。分步提取能改善使用，但同时改变输入、离散中间文本和计算量。固定开发人物、两个既有世界，非新世界确认。[汇总与逐世界证据](development-artifacts/bios-same-weight-v1/summary.json)。

<a id="capacity-scaling-development-v1"></a>
### 早期知识负载开发及学习控制

普通两层、宽64、102,976参数；同一世界和初始化，每例3,000曝光。九条校准／负载／背景控制均完成。下表为固定终点，H是该批负载变量。

| H | 数据bits/参数 | 实测学习bits/参数 | 全原子 | 训练组合 | 留出II |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 0.159 | 0.159 | 100.00% | 100.00% | 0.68% |
| 1024 | 0.398 | 0.383 | 93.34% | 94.58% | 0.54% |
| 4096 | 1.352 | 0.627 | 21.81% | 45.75% | 0.83% |
| 16384 | 5.171 | 0.198 | 1.39% | 8.79% | 1.32% |

固定H=256目标、只加背景事实时，总H=1024／4096／16384的全原子为100.00%／74.46%／7.54%，目标II为1.02%／0.68%／0.45%。最低负载也没有建立组合能力，因此不能从这些曲线拟合组合容量尺度律；高负载训练组合也未拟合，不能直接归因为容量或串扰。原前提失败停止记录及用户随后授权的完整扫描都保留。[汇总](../results/capacity-scaling-development-v1/summary.json)、[完整扫描修订](development-artifacts/capacity-scaling-full-scan-20261006/amendment.md)。

<a id="capacity-recall-controls-v1"></a>

**连续调用与训练内容控制：17项完成。** 九个旧权重中，最低负载自产桥的II／OO均100%，原生II仅0.68%，错桥II为0%；其余主负载的自产桥II为90.09%／12.84%／0.59%。八条新增训练包含原子、等计算原子复习和长期续训；最低负载额外280k更新后，wd=.01／.1的原生II仍0.57%／1.02%。H=256原子／原子复习终点为100%／29.10%，后者中途曾100%，说明优化轨迹不能省略。[完整报告](../results/capacity-recall-controls-v1/report.md)、[汇总](../results/capacity-recall-controls-v1/summary.json)。

<a id="capacity-stability-check-v1"></a>

**稳定性附批：四条训练及重载完成。** 改用lr=.001，H=256／4096的mixed原子为11.91%／31.30%，等计算原子复习为100%／95.49%；复习的自产桥II为100%／90.28%，直接II均0%。降低学习率没有普遍解决混合训练问题。本批controller的`finished_with_failures`来自W&B记录进程退出，科学四条均完成；不能把它写成云端同步完成。[报告](../results/capacity-stability-check-v1/report.md)、[状态](../results/capacity-stability-check-v1/controller-state.json)。

<a id="standard-storage-composition"></a>
### 标准Transformer的负载与优化稳定性

24开发、三新世界的30正式训练及2条退化诊断完成。低／高负载为1024／6144事实，固定词表、共同必要原子曝光和组合流；额外槽位从背景锚点复习改为新事实。宽256、16k步，主池为熟悉事实新链。

| 架构 | 低负载组合 | 高负载组合 |
| --- | ---: | ---: |
| 普通1层 | 32.66% | 22.10% |
| 普通2层 | 33.07% | 47.69% |
| 普通3层 | 53.26% | 39.25% |
| 单块Loop×2 | 40.66% | 32.27% |
| 单块Loop×3 | 43.33% | 37.00% |

普通2层低负载的第三世界训练组合从8k的100%退化到16k的35.51%，留出仅0.37%；该格保留在均值。必要原子及自主调用仍接近100%，严格组合各端点仅0–1.76%。降低lr续训4096步没有修复该退化。不能把上述非单调负载曲线直接称容量边界。[正式汇总](development-artifacts/storage-composition-v1/confirmation/summary.json)、[退化诊断](development-artifacts/storage-composition-v1/recovery/summary.json)。

<a id="storage-stability"></a>

后续一个新开发世界×三初始化的24条16k校准，余弦方案12端点的原子和训练组合均100%。普通2层低／高负载的恒定→余弦组合为49.00→57.56%／33.87→40.96%；Loop×2为49.13→57.30%／40.83→46.59%。12配对九正三负，余弦选择依据是学习稳定，不是测试最高分；严格组合仍近地板。[稳定性汇总](development-artifacts/storage-stability-v1/summary.json)。

<a id="composition-data-curves-v1"></a>
### 已建立低负载组合前提后的支持与负载曲线

**38条训练全部完成，失败0；三个新世界。** 开发四层宽64通过预定前提后固定464,896参数、4096实体词表、32关系；正式7档嵌套独立组合支持与6档N=128–4096负载均完成。每个N的随机关系图知识量为16N log₂N，组合答案不增加独立bits；不同N为不同闭合图，不能称嵌套事实干预。

N=128时，φ=0／0.5／1／2／4／7.2／12.6在等128k更新下的II为 **0／5.24／6.53／8.45／20.23／85.29／99.93%**；等每例4000曝光为0／4.64／5.47／7.91／22.04／95.48／100%。原子均100%，非零支持的训练组合均100%，OO仍0–3.09%。支持数量增长同时改变使用覆盖、组合比例及每例重复，不能称纯数量效应。

| 实体N；φ=7.2 | 等128k更新原子 | 等128k更新II | 等4000曝光原子 | 等4000曝光II |
| ---: | ---: | ---: | ---: | ---: |
| 128（支持臂复用） | 100.00% | 85.29% | 100.00% | 95.48% |
| 256 | 98.13% | 34.23% | 99.97% | 76.63% |
| 512 | 69.21% | 3.69% | 92.27% | 29.49% |
| 1024 | 23.62% | 0.98% | 60.86% | 6.97% |
| 2048 | 7.81% | 0.47% | 25.15% | 1.87% |
| 4096 | 3.04% | 0.23% | 2.41% | 0.42% |

增加曝光能改善部分负载，却未消除高负载下的提取与训练拟合问题；最高负载也非单调改善。这是固定架构与优化协议下的学习曲线，不是表达能力上限。数字按报告中三个世界等权复算，开发不并入。[全部预算节点](../results/composition-data-curves-v1/report/summary.json)、[支持图](../results/composition-data-curves-v1/report/support-curves.png)、[负载图](../results/composition-data-curves-v1/report/knowledge-load-curves.png)、[科学状态](../results/composition-data-curves-v1/controller-state.json)、[冻结协议](development-artifacts/composition-data-curves-v1/protocol.md)。

<a id="composition"></a>
## 二、知识能被组合：使用经历、表示与计算结构

<a id="implicit-reasoning-paper-reproduction-v2"></a>
<a id="second-hop-paper-reproduction-v1"></a>
### Ye 2025原实验：原子监督与第二跳角色覆盖

**四条各100万作者迭代的训练均完成，全池新进程重载逐token一致。** Figure 3监督比较使用同一图、初始化和273600条Train-II，分别不加原子监督、加入38000条ID原子监督；Test-II终点为1643/3000（54.77%）与2987/3000（99.57%）。不加原子监督时，标准单跳问法为0%；这不能证明内部没有事实表示，也说明标准单跳可回答性不是组合成功的硬性前提。加入监督后，首次记录到Test-II超过50%的节点从100万迭代提前到15万迭代；记录间隔有限，不将它解释为精确阈值步数。组合曝光分别为10.21亿与8.97亿，不能称为等组合曝光的纯监督效应。

第二跳覆盖消融中，1900条ID事实可作为训练首跳、禁止作为训练第二跳。只学Train-II／另加全部40000条原子监督的普通Test-II为2877/3000（95.90%）／2987/3000（99.57%）；受限Test-II-SR两臂均2/3000（0.07%）。全原子监督臂ID/OOD单跳为2996/3000、1993/2000；它支持本配方中“直接可提取与首跳使用经历不足以形成第二跳使用”，尚未定位独立层内部的唯一机制。

两比较均为seed42的单世界、单初始化，留出率等条件不同，不把跨批差值当作配对效应。使用作者数据函数和完整训练实现，但保留必要代码修正及Torch/CUDA环境差异；行为方向复现不代表原图数字逐项相同。W&B本地追踪已到终点，本次未独立核验云端。[Figure 3原协议](development-artifacts/implicit-reasoning-paper-reproduction-v1/protocol.md)、[解码修订](development-artifacts/implicit-reasoning-paper-reproduction-v2/protocol.md)、[Figure 3两臂审计](../results/implicit-reasoning-paper-reproduction-v2/runs/only_ii-seed42/audit.json)、[加ID监督审计](../results/implicit-reasoning-paper-reproduction-v2/runs/ii_plus_id-seed42/audit.json)、[角色限制协议](development-artifacts/second-hop-paper-reproduction-v1/protocol.md)、[全原子监督审计](../results/second-hop-paper-reproduction-v1/runs/full_atomic-seed42/audit.json)。

<a id="grok-usage"></a>
### 同一事实的组合经历

四臂开发与三个新世界12条确认训练完成。A/B事实组轮流参加组合练习，并设置逐批匹配构成原子出现次数的重复对照。固定128k端点，同一事实组的未见同组组合：有组合经历 **100%**，仅基础原子3.01%，对应原子重复25.01%，重复另一组原子6.99%。主比较相关原子全部正确，组合相对匹配重复提高74.99个百分点，三个世界方向一致。

这支持组合使用经历有独立于简单原子重复的价值；它仍同时改变上下文、监督和优化。事后角色分层：两事实均有经历／仅首条／仅后条／均无为100／41.48／48.94／3.01%，中间两行不是同题四格因果对照。[正式汇总](development-artifacts/grok-usage-v1/report-confirmation/summary.json)、[角色复盘](development-artifacts/knowledge-use-review-v1/summary.json)。

<a id="text-structure"></a>
### 组合搭配分布

三新世界×两初始化，固定每条事实角色次数、训练链数和token边际。受限／广搭配使熟悉事实新链从 **15.11%到66.30%**，仅后事实有经历的新链62.50→89.71%；独立人物和桥实体的严格目标仅1.57→2.16%。原子、训练组合、自主调用均100%。熟悉收益三世界均正；严格收益六配对四正两负，仍约2%。说明搭配支持改善局部组合，并未建立普遍跨事实迁移。[确认报告](development-artifacts/text-structure-v1/confirmation/summary.json)、[次要分组](development-artifacts/text-structure-v1/confirmation/secondary-analysis.json)。

<a id="text-pretraining"></a>
### 文本中的联合上下文与组合结果

三个新世界×两初始化×三臂，固定32k端点。独立原子陈述P0、相同句子开放共同上下文P1、再加非测试组合结果P2的目标组合为 **69.16%／82.88%／99.84%**，单跳均100%。P1−P0三个世界为+26.89／+20.24／−5.98个百分点，联合上下文收益并不稳定；P2−P1六配对均正，但P2同时改变监督和曝光。后继事实可参加背景组合，此处并非两条边都无使用经历的严格组合。

同桥首跳状态替换在固定子池使P0组合76.06→91.36%，仅MLP为87.34%；异桥完整状态转向率P0/P1/P2为29.53／15.18／9.07%。组合更好并不必然意味着所测单点通道更强。供体包含查询身份，MLP依赖注意力历史，不能唯一归因于MLP。[报告](development-artifacts/text-pretrain-v1/report/summary.json)、[干预表](development-artifacts/text-pretrain-v1/report/mechanism-cells.csv)。

<a id="text-loss-source"></a>
### 已学事实继续训练时，哪些损失维持组合使用

从六个共同父端点复制模型、优化器和数据流，继续4096步；24条冻结后续训练完成。保留原子A、背景组合B和中性N的不同损失，目标组合AB/A/B/N为 **84.59／43.10／65.69／69.27%**，单跳为100／100／71.09／95.76%。AB−A为+41.50个百分点，六配对均正；交互+45.08，不能概括为原子训练普遍有害。

A分支换入N分支MLP后43.10→70.96%，六配对均正、单跳仍100%；换注意力为61.10%。反向MLP交换并非每对都下降。训练损失、参数变化与行为之间有条件性因果联系，但未分离事实内容、读取方式和共同适应。三个世界此前已被观察，属于冻结后续而非新世界确认。[报告](development-artifacts/text-loss-source-v1/followup/report/summary.json)、[参数互换](development-artifacts/text-loss-source-v1/followup/report/weight-cells.csv)。

<a id="representation-alignment"></a>
### 共享GPT：首跳能读出与能继续使用

三新世界×两初始化×三臂，完整单共享GPT块执行两次、宽128、260,608参数、16k更新。首跳辅助CE两臂用相同实体标签，额外对齐把首执行块首关系位置的完整残差方向约束到该实体输入Embedding；推理没有额外输入或调用。

| 训练方式 | 原子／训练组合 | 熟悉新组合 | 严格组合 |
| --- | ---: | ---: | ---: |
| 全文CE | 100%／100% | 5.73% | 1.30% |
| 全文CE＋首跳CE | 100%／100% | 32.68% | 12.60% |
| 相同首跳CE＋表示对齐 | 100%／100% | 97.96% | 92.32% |

对齐的熟悉／严格收益六配对均正。两辅助臂的早期首跳实体读出均100%，严格状态余弦为.8005／.9992。故标签能读出不足以解释后继使用；此训练干预提供强严格组合正例。Embedding、注意力和MLP共同改变，未证明唯一模块或必要向量形式。[完整确认](development-artifacts/representation-alignment-v1/confirmation/summary.json)、[状态诊断](development-artifacts/representation-alignment-v1/confirmation/state-diagnostics.json)。

<a id="independent-alignment-v1"></a>
### 普通独立两层复核：强严格迁移没有复现

一个开发配对及三新世界×两初始化×两臂，共14条16k训练完成。普通两层、宽128、458,880参数；相同首跳CE，仅额外λ=.3对齐不同。原子、训练组合、早期首跳读出、必要事实覆盖和自主两次调用均100%。

| 普通两层 | 熟悉组合 | 严格组合 |
| --- | ---: | ---: |
| 首跳CE | 16.76% | 2.96% |
| 首跳CE＋对齐 | 55.88% | 5.27% |

熟悉收益六配对均正；严格差+2.31个百分点，四正一负一零。严格状态余弦已从.8190到.9997，但仍不能可靠组合。旧共享模型的强严格收益未复现；跨批次参数量、初始化和世界不同，不能把差值唯一归因于共享约束。14端点重载和远端14运行完整节点核验通过。[报告](../results/independent-alignment-v1/report/report.md)、[解释](../results/independent-alignment-v1/report/interpretation.md)、[云端审计](../results/independent-alignment-v1/tracking-cloud-audit.json)。

<a id="bridge-reencoding"></a>
### 自产桥重编码与几何监督覆盖

固定三世界×两初始化×三臂共18既有权重，零新增训练。在首关系位置用自产实体argmax的输入Embedding方向替换状态、保留范数，不提供真桥。全文CE／首跳CE／对齐臂的原生严格为1.30／12.60／92.32%，重编码后 **1.33／46.35／92.38%**，替代错误桥为1.14／1.89／1.40%。首跳CE六配对均改善且原子100%；正确身份重编码能部分恢复使用，仍未替代训练期对齐。[正式重编码汇总](../results/bridge-reencoding-confirmation-v1/report-confirmation/summary.json)。

另三新世界×两初始化×四臂的24条16k训练，均保留相同首跳CE。几何项不加／仅组合每例系数匹配／仅组合总项匹配／覆盖原子与组合的熟悉组合为26.73／78.70／79.31／97.97%，严格为 **10.25／25.55／23.80／92.12%**；原子与自主调用均100%。all相对两个组合对照的严格差+66.57／+68.33个百分点、六配对均正。该模板中原子与组合共享首关系因果前缀，all直接训练了严格事实后来复用的状态；不能单独区分首跳、次跳和锚点贡献。[覆盖确认](../results/alignment-coverage-confirmation-v1/report-confirmation/summary.json)。

<a id="grok-depth"></a>
### 深度与组合形成

三新世界×两初始化，原子与训练组合均100%，128k端点：普通1层宽128／普通1层宽180／普通2层宽128的留出组合为 **27.78／24.52／96.44%**；宽一层与两层参数接近。共同计算上限下，两层64k仍93.06%，窄一层128k为27.78%。两层原子及训练组合6k–8k已共同≥99%，组合随后继续提高；不是此前完全无泛化的突然觉醒。严格OOD池很小，两层仅2.28%。[主汇总](development-artifacts/grok-depth-v1/summary.json)、[完成证据](development-artifacts/grok-depth-v1/completion-manifest.json)。

<a id="depth-hop-extension"></a>

两跳3／4层的12条配对扩展为98.40／98.29%，达到90%的平均更新为27.33k／22k，原2层47k；达到90%的平均FLOPs以3层最低，更多层并非按计算一律更快。三跳开发将组合支持φ6→24，4层从9.91→95.03%；两个新世界3／4层分别90.29／97.04%，共同FLOPs的probe差+3.22个百分点。四跳6层φ24仅11.84%，虽原子／训练组合／外部四次调用100%，仍未建立可靠四跳基线。这些数据不支持“一跳必须一层”或普遍最少层数。[两跳扩展](development-artifacts/grok-depth-extension-v1/summary.json)、[三跳确认](development-artifacts/grok-multihop-v1/confirmation/summary.json)、[全开发多跳](development-artifacts/grok-multihop-v1/summary.json)。

<a id="grok-depth-bridge"></a>

纯首跳供体仅见[h′,r1]，换入两层模型第一块r1状态时，新路径正确转向 **89.31%**；换头实体位置仅0.13%，完整新问题95.71%。相关原子全正确，同桥替换保留原答案99.49%。主转向随4k／32k／64k／128k为8.66／75.16／83.21／89.31%，早期原子前提尚未充分。支持首跳状态参与后继计算，但完整状态不等于纯实体编码，也不证明MLP独占知识。40既有模型状态、零训练。[状态干预汇总](development-artifacts/grok-depth-bridge-v1/summary.json)。

<a id="grok-loop"></a>
### 共享参数Loop与执行次数

15开发、6敏感性、三新世界×两初始化的90正式训练及既定干预全部完成；主终点128k，最终答案与EOS监督。C1/C2为普通1/2层，CD为普通展开深度D，L1/L2为1/2个共享块循环到D；两／三跳D4、四跳D6。

| 架构 | 两跳ID／OOD | 三跳ID／OOD | 四跳ID／OOD |
| --- | ---: | ---: | ---: |
| C1 | 30.38／0.84% | 57.56／0.68% | 38.24／0.75% |
| C2 | 96.44／1.42% | 97.61／0.82% | 83.03／0.70% |
| CD | 98.14／1.61% | 99.32／0.84% | 100.00／0.94% |
| L1 | 100.00／4.55% | 100.00／0.78% | 100.00／0.94% |
| L2 | 99.86／6.29% | 99.99／1.02% | 100.00／0.84% |

L1/L2的熟悉ID组合接近满分，严格OOD仍低；四跳C1训练组合仅64.17%，其低分包含拟合不足。固定两跳L2权重，测试R2／3／4／8的OOD为6.29／26.28／38.50／17.42%，原子100／99.93／95.77／30.24%；额外计算收益有范围，超出训练深度亦可破坏提取。开发D8两种初始化的L2 OOD为18.70／17.59%，CD为1.67／1.48%，只是同开发世界敏感性。

正式ID合格供体子集的第一层异桥完整状态转向，L1两／三／四跳为99.43／100／99.98%，仅MLP为43.17／97.10／73.98%；覆盖82.92／87.60／69.66%。它支持早期信息使用，不证明唯一MLP或每轮一跳。[主汇总](development-artifacts/grok-loop-v1/main-report/summary.json)、[循环及干预](development-artifacts/grok-loop-v1/mechanism-report/summary.json)、[敏感性](development-artifacts/grok-loop-v1/sensitivity-report/summary.json)。

<a id="grok-loop-continuous"></a>

111轨迹、2,997节点的只读连续评分复盘：正式两跳L2的OOD正确答案概率6.1473%，答案＋EOS6.2932%；三／四跳概率1.0085／0.8452%。OOD低分不是EOS丢分，也没有高概率被硬阈值隐藏。90正式OOD轨迹未见两段均≥5个百分点的下降再回升；每8k保存的后期节点不能排除更快波动。[复盘](development-artifacts/grok-loop-continuous-v1/summary.json)。

<a id="grok-loop-same-bridge"></a>

同桥ID／OOD纯首跳供体比较，30既有正式端点、零训练。共同供体子池覆盖仅6.12%；L1基线8.25%，换ID／OOD完整状态40.58／8.06%；L2基线9.68%，换后38.94／12.33%。相同桥身份仍有状态可用性差异，但不同供体是不同事实，不能视为同一事实训练经历的随机干预，也没有完全修复后继使用。[完整分母与报告](development-artifacts/grok-loop-same-bridge-v1/report/summary.json)。

<a id="grok-loop-supervision"></a>

从同一R2父权重展开R4续训8k，单终点／多终点监督在R4的OOD为11.10／28.08%，R8为41.33／42.58%，R16为46.74／34.48%；父模型R4为38.50%。多终点没有保住原R4能力，也未普遍改善长循环。R32→64时原始残差幅度继续近线性增长、绝对步长非零，归一化变化变小不能当作不动点证据。14续训、18模型动力学诊断完成。[监督报告](development-artifacts/grok-loop-supervision-v1/followup-report/summary.json)、[动力学](development-artifacts/grok-loop-supervision-v1/dynamics/summary.json)。

<a id="latent-scaling"></a>
### 数据支持、宽度、独立深度与循环

一个开发世界的24条128k训练，原子／训练组合／自主调用均100%。宽128单块R1／2／3在64组合例时为0.77／0.77／0.38%，256例为9.96／6.13／8.43%，763例为26.05／39.85／49.81%。宽64／128／256、R2、最大支持时为24.52／39.85／42.91%；普通两层宽128／256为70.50／71.65%。宽128的R1／2／3／4／8为26.05／39.85／49.81／44.83／36.02%。不同规模轴不能互相替代，严格组合仍0–1.76%。[开发汇总](development-artifacts/latent-scaling-v1/summary.json)。

<a id="latent-confirmation"></a>

三新世界×两初始化×八条件48条确认：256／完整组合支持下，R1为7.97／26.30%，R2为11.05／47.99%，R3为7.49／48.81%，普通两层13.28／67.10%。支持×R2收益交互 **+18.61个百分点**，三世界、六配对均正；全部相关原子与训练组合100%。R3相对R2的世界方向混合，严格池均值0.46–1.11%。支持同时改变角色覆盖、搭配及重复。[正式汇总](development-artifacts/latent-confirmation-v1/summary.json)。

<a id="latent-support"></a>

等角色次数的12条后续对照：固定512链，连通保留／保度拆分的熟悉组合 **20.19／20.56%**，世界差−3.02／+0.79／+1.12个百分点，六配对三正三负。相关原子、训练组合、自主调用均100%；预定局部结构子池差+6.16个百分点、覆盖约9–13%，不能替代全池。没有建立稳定的连通性收益，亦不证明连通结构永远无效。[支持结构汇总](development-artifacts/latent-support-v1/summary.json)。

<a id="memory-scaling-theory"></a>

九组数学校验与三世界训练图分析为零LM训练。最小连通保留需442／434／434链，512链可匹配角色度数并改变分量；图秩不等于GPT特征Gram秩，实体链最长两跳也不等于支持图直径。理论适用条件见 [memory-scaling-theory](memory-scaling-theory.md)，对应实际训练结果见上段。[数学校验](development-artifacts/memory-scaling-theory-v1/README.md)。

<a id="multihop-scaling"></a>
### 多跳使用经历与执行计算

16开发及三新世界×两初始化的96条64k正式训练完成；原子、训练组合、自主逐跳路径均100%。提高组合支持的两／三／四跳熟悉新链收益为 **+30.47／+28.33／+14.62个百分点**，每hop全部48条件配对为正。宽256、Loop训练R2→R4的支持交互为+9.75／+12.53／+4.50个百分点；固定R4权重只把测试R2→R4，高支持收益+6.97／+15.47／+7.20，必要原子覆盖接近100%。继续到R6可退化，严格组合仍弱。任务联合训练2/3/4跳，不是未见长度外推。[主报告](development-artifacts/multihop-scaling-v1/confirmation/report/summary.json)、[同权重预算](development-artifacts/multihop-scaling-v1/confirmation/execution-budget-report/execution-budget-summary.json)。

<a id="depth-step"></a>

一个开发世界九架构64k比较，普通1／2／3／4／6层两跳为20.97／38.71／41.94／56.45／43.55%，Loop×2／3／4／6为30.65／34.68／36.29／43.55%；三／四跳完整表保留。原子和训练组合全对，深度收益不统一单调。Loop首桥更可读出不对应更高原生成绩；基线校正后的异桥单点干预未稳定转向。该批编辑另见第三层。[全部端点](development-artifacts/depth-step-v1/endpoints.csv)、[机制汇总](development-artifacts/depth-step-v1/mechanism-summary.json)。

<a id="recurrence-use"></a>

14既有权重的零训练干预，高支持R4下删首r1更新，两／三／四跳熟悉组合从51.61／49.06／29.28%降至35.61／29.29／24.19%；固定共同原子正确子集仍下降。更新响应的范数大不保证语义使用；同桥首跳替换未稳定修复严格组合。对单共享无时间条件块，删除任意整轮恒等于少执行一次F，已独立核验，不能由整轮删除定位哪个循环重要。[干预报告](development-artifacts/recurrence-use-v1/followup/report.md)、[等价式](development-artifacts/recurrence-use-v1/equivalence-note.md)。

<a id="loop-block-depth-v1"></a>
### 固定每轮四层，分别训练更大的R

**九条128k训练及重载完成，失败0。** 同一个历史开发世界、宽128，主比较固定四个独立层和参数量，允许总执行深度随训练R增加，按各自训练R评分；不是只对旧权重多算几轮。

| 四层块训练R | 1 | 2 | 3 | 4 | 6 | 8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 熟悉组合 | 77.01% | 66.67% | 50.96% | 64.37% | 60.54% | 46.36% |
| 严格组合 | 0.98% | 0.78% | 1.17% | 0.98% | 0.39% | 0.00% |

各端点原子、训练组合和必要原子覆盖均100%。附加2×2／2×4／8×1的熟悉组合为50.19／47.89／46.36%。在这套配方中，增大训练循环并未带来统一收益；只含一个世界/初始化，不是架构上限。等计算节点和同权重R扫描全部保留。[报告](../results/loop-block-depth-v1/report/summary.json)、[科学状态](../results/loop-block-depth-v1/controller-state.json)、[协议](development-artifacts/loop-block-depth-v1/protocol.md)。

<a id="grokking-dynamics"></a>
### 长期优化与原论文规模复现

小模板开发中12条512k训练：普通两层wd=0／.01／.1的熟悉组合90.50／93.39／95.45%，Loop×2为55.58／61.57／81.82%，严格仍0.39–1.56%；原子与训练组合终点全对。有拟合后改善，但没有确认清晰突变。MLP范数变化与正则处理相容，不证明唯一回路。异桥供体转向常调用了有组合经历的新路径，不能称修复原严格链。编辑及回放角色限制见第三层。[完整分析](development-artifacts/grokking-dynamics-v1/assessment.json)。

<a id="grokking-reproduction"></a>

原规模八条150万步训练全部完成：2000实体、200关系、40,000原子、宽768，普通8层与共享4×2，同展开深度。一个世界、一个初始化；答案与EOS监督。

| 支持φ／衰减wd | 普通8层严格OO | Loop 4×2严格OO |
| --- | ---: | ---: |
| 3.6／.1 | 0.40% | 25.65% |
| 7.2／.1 | 0.20% | 53.29% |
| 12.6／.1 | 0.00% | 66.17% |
| 7.2／.3 | 0.00% | 81.34% |

原子及训练面板8k–23k已共同≥99%，Loop明显OO改善晚得多；全量原子终点98.88–99.97%。这给出无显式桥监督的严格组合正例。参数量、初始化、支持比例和计算条件须分别考虑，不能把单个因素称充分条件。[端点独立复算](development-artifacts/first-principles-review-20261004/data-synthesis/grokking-recount.json)、[全部曲线](development-artifacts/first-principles-review-20261004/data-synthesis/grokking-all-eight-curves.png)、[完成清单](development-artifacts/grokking-reproduction-v1/completion-manifest.json)。

<a id="bridge-workspace"></a>
### 中间表示的因果复用

42个既有权重端点的干预矩阵已完成，包含18个小GPT与24个原尺寸700k／1M／1.5M端点；零新增训练。小模型严格子池中，全文CE／首跳CE／对齐的输入实体方向转向为1.72／17.09／92.34%，同范数随机为1.66／1.26／1.59%；对齐臂完整prefix转向92.54%。删除完整state实体分量使原生93.10→0.98%，仅删MLP分量为81.77%，随机MLP扰动反而46.68%，尚不支持唯一MLP投影。

原尺寸1M、第一轮末端、固定OOD首跳对，Loop四条件的输入方向转向10.53／45.61／35.67／59.06%，prefix Jacobian方向12.28／45.03／32.75／54.39%，完整prefix21.64／53.80／67.25／87.13%；随机方向0–0.58%。第二条事实可能有组合经历，此池不是严格OO基准。早期桥读出99.22–100%而功能转向差异很大，支持可读出与可使用的区分。全层、所有节点及微小BF16逆交换行为误差均保留。[全部端点](development-artifacts/bridge-workspace-v1/comparison/report/endpoints.csv)、[重计分](development-artifacts/bridge-workspace-v1/comparison/report/recount-audit.json)、[完成状态](../results/bridge-workspace-v1/comparison/controller-state.json)。

<a id="realworld-composition"></a>
### 真实事实的闭卷组合与架构适用范围

2Wiki原始事实改编为闭卷训练：65,968原子、56,121组合训练、3,729未见组合。九条300k正式训练（三架构×三初始化）及重载完成，只有一个真实图。

| 架构 | 原子 | 全池组合 | II | OO |
| --- | ---: | ---: | ---: | ---: |
| 普通8层 | 99.93% | 24.97% | 37.87% | 17.74% |
| Loop 4×2 | 99.95% | 23.12% | 35.15% | 16.73% |
| 普通4层 | 99.96% | 23.18% | 33.81% | 17.02% |

训练组合均≥99.96%，自主两次调用均≥99.84%；本配方没有观察到Loop优势。开发32k也未显示优势：全池44.08／40.12／39.93%，OO19.54／17.61／17.08%。官方别名EM为主，名称无歧义等次评分另报。任务不是原阅读理解排行榜，原注释存在歧义、证据不足和冲突；开发审核及名称隔离记录保留。[正式汇总](development-artifacts/realworld-composition-v1/confirmation/confirmation-summary.json)、[数据审核](development-artifacts/realworld-composition-v1/semantic-review.json)、[全部开发结果](development-artifacts/realworld-composition-v1/development-summary.json)。

<a id="realworld-loop-diagnosis"></a>

**实体表示和事实地址的22条诊断全部完成。** 同一真实图、32k、两个初始化，普通8层／Loop的自然名OO为17.15／16.33%，单实体token为84.22／85.11%；原子从95.93／90.52%到99.985／99.985%。表示改动同时改变长度、身份地址、消歧和监督分布；Loop相对优势仍不稳定。

小合成地址干预保持基础事实、训练链ID和词表，把多关系事实共享头地址改成每条首跳事实独立头地址：普通／Loop在128k的OO由2.61／86.41%变为96.46／98.70%，架构差从+83.80到+2.23个百分点。支持事实地址条件改变组合难度及共享收益；同时改变活跃输入槽与头实体新颖性，不是纯出度效应。[表示子集](development-artifacts/realworld-loop-diagnosis-v1/entity-subsets.json)、[地址全量复算](development-artifacts/realworld-loop-addressability-v1/summary.json)。

<a id="oo-budget-attribution-20261005"></a>

**预算与优化归因六条新增、两条复用均完成。** 历史合成Loop φ7.2/wd.3在1500k的81.34%，不能与真实300k的16.73%作同更新比较；两者训练GPU时长约6.50／6.64小时，真实执行token／估算FLOPs反而约2.13／1.96倍。实现、编码和图等不同，不能作纯数据效应。

固定真实图、Loop4×2、300k及两初始化，lr5e-5／1e-4×wd.1／.3的OO均值依次 **17.11／15.67／16.17／15.00%**，未恢复历史合成强严格组合。完整全池曲线、原64题面板和资源分开保留。[优化归因报告](../results/oo-optimizer-attribution-v1/report.json)、[预算复盘](development-artifacts/oo-budget-attribution-20261005/summary.json)。

<a id="parametric-architecture"></a>

**参数架构主矩阵15条300k新训练全部完成，失败0。** 开发独立选学习率后，在同一个真实图做三初始化配对；报告另列九条历史基线。各臂名称和结构按[冻结协议](development-artifacts/parametric-architecture-development-v1/source/docs/development-artifacts/parametric-architecture-execution-v1/protocol.md)定义。

| 新主实验臂 | 原子 | 训练组合 | 全池组合 | OO |
| --- | ---: | ---: | ---: | ---: |
| M8：八层MoE | 99.84% | 99.90% | 27.33% | 18.86% |
| W8：近等总参数的宽MLP | 99.94% | 99.95% | 25.50% | 17.20% |
| IHC8：Identity多流控制 | 99.88% | 99.92% | 22.45% | 16.37% |
| HC8：mHC跨流混合 | 99.93% | 99.96% | 22.75% | 17.43% |
| 本批D8配对基线 | 99.88% | 99.90% | 25.60% | 16.84% |

结果是有限预算下的架构及训练方案比较；没有出现接近完备的组合使用。三初始化不等于三新世界；共同必要原子正确子集为处理后描述，不能作因果调整。[完整结果、配对差和分母](../results/parametric-architecture-main-v1/report/summary.json)、[科学状态](../results/parametric-architecture-main-v1/controller-state.json)。

<a id="shared-cache-branch-v1"></a>
### 首轮共享KV

20条128k训练与128个edit/sham分支完成；三新世界确认、每世界一个配对初始化，四层共享块、R4、宽128、1,011,072参数。

| KV访问 | 原子 | 熟悉组合 | 严格组合 |
| --- | ---: | ---: | ---: |
| 原Loop local | 100% | 48.59% | 1.11% |
| 首轮共享＋本轮完整KV | 100% | 72.17% | 1.43% |
| 首轮共享＋本轮窗口 | 100% | 72.57% | 0.78% |
| 完整共享读取停止梯度 | 100% | 60.39% | 0.33% |

完整／窗口共享熟悉收益三世界均正，等计算节点方向保持；严格仍近地板。关闭完整共享使熟悉72.17→34.26%、原子100→75.17%，说明路径确实使用，但不能单独定位组合桥。编辑结果见第三层。[完整报告](../results/shared-cache-branch-v1/report/report.md)、[云端36运行核验](../results/shared-cache-branch-v1/cloud-final-audit.json)。

<a id="residual-cache-comparison-v1"></a>
### 残差结构与共享KV的配对扩展

**48条新训练、8条复用基线和16个新增编辑父模型全部完成；56/56训练格审计通过。** 三个配对世界此前已观察，属于扩展而非新独立确认。主训练保持四层、宽128、128k，比较普通残差、Identity-mHC、mHC及local/shared_full、R4/R8；开发缩放参照另列。

| 残差 | local R4／R8熟悉 | shared R4／R8熟悉 | shared R4／R8严格 |
| --- | ---: | ---: | ---: |
| 普通 | 48.59／39.68% | 72.17／70.99% | 1.43／0.72% |
| Identity-mHC | 41.32／47.63% | 67.83／76.77% | 0.78／0.46% |
| mHC | 43.96／44.77% | 70.91／71.53% | 0.85／1.11% |

共享KV的熟悉收益保留，但mHC没有稳定胜过Identity控制，也没有解决严格组合。额外控制器参数、Sinkhorn逐元素计算与等FLOPs节点分别报告；不能把全部差值归为残差混合机制。64次新增直接编辑均成功；三已知世界的各新增臂严格传播均0%，熟悉第二跳仅mHC/local为16.67%、其余0%，每臂角色只有三个图选事实，不能建立普遍编辑优势。[完整报告](../results/residual-cache-comparison-v1/report/report.md)、[逐例编辑与全部对照](../results/residual-cache-comparison-v1/report/summary.json)、[科学状态](../results/residual-cache-comparison-v1/controller-state.json)。

<a id="small-lm-composition"></a>
<a id="small-lm-composition-v2"></a>
### 完整Qwen的事实呈现与组合

Qwen3-0.6B-Base全596,049,920参数训练，一个开发世界。首轮独立／关联呈现的直接两跳6.25／28.13%，但关联组第二单跳仅21.88%，不能视作知识掌握相当的组合收益。

共同独立原子复习的v2完成三条512步训练：分开／无关联拼接／关联拼接的两单跳、自主两次调用、背景训练组合均100%；64目标直接两跳为 **6.25／4.69／15.63%**，留出问法10.94／4.69／15.63%。关联保留小幅开发收益，但事实同文档也建立头尾共现；不是严格跨世界确认。[v1汇总](development-artifacts/small-lm-composition-v1/summary.json)、[v2汇总](development-artifacts/small-lm-composition-v2/summary.json)。v2更新分支见第三层。

<a id="natural-composition"></a>

三新世界、完整Qwen3-0.6B六条512步支持复核中，低／高支持熟悉组合为 **2.08／0%**，严格均1.04%；相关原子、训练组合和自主调用均100%。未确认符号GPT的支持收益，有限预算与小分母不建立自然模型能力上限。修订前未见关系搭配开发与修订后四种搭配均出现的正式比较分开保留。[正式报告](development-artifacts/natural-composition-confirmation-v1/amendment-v2/confirmation/summary.json)。

<a id="architecture-bridge"></a>
### 其他架构与冻结自然模型诊断

两个世界×两初始化的受控GQA／混合GatedDelta-GQA，原子99.80／100%，训练组合99.51／100%，留出组合12.89／13.28%，外部两次调用99.61／100%。参数、FLOPs不同，0.39个百分点不建立架构优势；完整问题供体可能携带答案，不能把其patch收益解释为纯首跳修复。

Qwen3-0.6B与Qwen3.5-0.8B在各24例MQuAKE/2Wiki评价中的MLP／状态／联合干预没有建立稳定桥接选择性；错误供体和随机方向也能改善个别案例。每模型四次局部8步学习中，Qwen3唯一EM改善来自输出格式，D没有改善。四个down矩阵均满行秩，不能套秩亏反例解释它们；60个代数校验只是条件公式核验。[受控训练](development-artifacts/architecture-bridge-toy-v1/summary.json)、[自然模型](development-artifacts/architecture-bridge-v1/summary.json)、[格式复核](development-artifacts/architecture-bridge-v1/learning-format-review.json)。

<a id="frozen-twohop"></a>

冻结Qwen3-4B-Instruct在各128例MQuAKE／2Wiki上，直接回答23.44／28.13%，自主桥接25.78／32.81%，给分解串联36.72／35.16%，真桥33.59／32.81%，短CoT17.97／22.66%，长CoT36.72／36.72%。自主桥接差的配对区间跨零；长CoT增加生成计算。两单跳均通过子集仅30／37例，其长CoT73.33／81.08%，不能代替全池。

全FP32复核中，保留MLP有限非线性比输入一阶梯度更准确地描述已知标签下的局部响应，4B误差减少约46.8／41.3%；这是历史离线诊断，不是当前预测研究方向。隐藏干预仍无稳定桥接特异修复。模型规模与指令训练不同，数据别名、语义冲突和格式限制保留。11,128生成及主／FP32审计完成。[完整报告与复现](reports/twohop-frozen-v1.md)、[行为表](development-artifacts/twohop-frozen-v1/behavior-summary.csv)、[全精度结果](development-artifacts/twohop-frozen-v1/full-precision-summary.json)。

<a id="updating"></a>
## 三、新增或修改知识后，已有用法能否继续工作

<a id="paper-edit-reproductions-v1"></a>
<a id="path-hop-analysis-v1"></a>
### 自然模型编辑复现与冻结跳位分析

**MQuAKE两条全3000案例任务及独立重放均已通过。** 各方法6015条事实编辑、66030条生成查询，全部保存的低秩编辑在新进程中重新应用，生成token逐条一致。每案例独立编辑后恢复，不累积编辑。

| GPT-J 6B编辑方法 | 直接编辑成功，事实分母6015 | 编辑后多跳，案例分母3000 | 编辑后CoT，案例分母3000 | 编辑与必要单跳均成功的条件覆盖 | 条件多跳／CoT |
| --- | ---: | ---: | ---: | ---: | ---: |
| ROME | 88.10% | 7.37% | 21.20% | 2026/3000（67.53%） | 9.38%／28.92% |
| MEMIT | 95.53% | 7.63% | 13.13% | 860/3000（28.67%） | 16.86%／33.95% |

直接编辑成功与必要单跳掌握仍不保证派生问答更新。方法间条件池不同，不按条件准确率宣布方法优劣。数据采用原MQuAKE-CF-3k，原提示与默认编辑超参数；作者未公开参数编辑评价脚本，本次生成上限、答案截取及cloze前缀评分是显式实现约定，不冒称与原Table 3数字完全相同。[复现协议及差异](development-artifacts/paper-reproductions-parallel-v1/protocol.md)、[ROME审计](../results/mquake-paper-reproduction-v1/runs/gpt-j-6B-rome/audit.json)、[MEMIT审计](../results/mquake-paper-reproduction-v1/runs/gpt-j-6B-memit/audit.json)。

**2026-10-09，五条编辑复现全部通过独立重放，冻结跳位分析完成5/5条。** 五条全池审计覆盖237403条生成查询，按运行计数，不是独立事实数。分组、评分、等权2/3跳分层和10000次案例bootstrap均沿用2026-10-08契约；分组分析没有新增训练或模型前向。MQuAKE主池为单编辑的869案例（2跳首／后续213/300，3跳198/158），不同组是不同案例；CoT与直接回答在同一案例配对。

| MQuAKE主比较 | 直接回答首跳−后续跳，pp（95%区间） | CoT首跳−后续跳，pp（95%区间） | CoT使差距缩小，pp（95%区间） |
| --- | ---: | ---: | ---: |
| ROME | 25.34（20.89–29.67） | 17.00（11.75–22.20） | 8.34（2.90–13.71） |
| MEMIT | 24.15（19.87–28.34） | 24.15（19.22–29.02） | 0.003（−5.13–5.05） |

两方法都有首跳优势，但预定“CoT缩小跳位差距”只在ROME得到支持，MEMIT未测出缩小。不能将整体跳位差距唯一归为路径不一致；关系、实体类型及原能力差异仍需保留。条件子集仅作次要分析，不替代全池主比较。[冻结契约](../configs/path-hop-analysis-v1.json)、[ROME分组结果](../results/path-hop-analysis-v1/gpt-j-6B-rome.json)、[MEMIT分组结果](../results/path-hop-analysis-v1/gpt-j-6B-memit.json)。

RippleEdits的POPULAR/RANDOM/RECENT全发布池885/1922/1948案例均已通过独立重放及作者Evaluator重算；RECENT的109个空标签案例原样保留为排除记录。冻结主比较按成功编辑、原前提成立且结构合法的已执行测试计分，bootstrap按案例重采样。这与复现表中的“先案例平均再池平均”准确率不同，不能混用两种权重。

| 发布池 | CI首跳正确率（有效测试数） | CII次跳正确率（有效测试数） | CI−CII，pp（95%区间） | 同案例双轴描述性比较 |
| --- | ---: | ---: | ---: | --- |
| POPULAR | 40.00%（165） | 14.29%（28） | +25.71（+6.89–+39.61） | 2案例，均值差0pp |
| RANDOM | 36.07%（499） | 100.00%（1） | −63.93；单次跳测试不足以估计稳定规律 | 0案例 |
| RECENT | 38.49%（943） | 55.56%（54） | −17.06（−42.00–+3.14） | 6案例，均值差+29.17pp |

POPULAR支持条件池中的首跳优势；RANDOM的次跳覆盖仅1条，原bootstrap只有6446/10000次抽样含次跳，保留其原输出但不把狭窄区间解释为稳定反向效应。RECENT点估计反向、区间跨零；同案例描述性差值又为正，说明不同案例组成与筛选不能忽略，也不能用6个配对子集替代主比较。预定“每个发布池CI均高于CII”未得到支持。结合MEMIT的近零CoT交互，本批不足以建立普遍的路径一致性解释；保留角色不对称与ROME的条件性CoT收益，不据此宣布神经路径机制已经证实或完全否定。

RECENT原串行审计已核对的前缀保留。为了缩短等待，使用同一固定镜像和完全未改的冻结审计源码，将全部1948案例拆为7个不相交范围，在空闲GPU的独立容器中重新重放；范围视图只改变`start_case`、`max_cases`及输出目录，原记录与低秩编辑只读。七段共45042查询逐token一致，11034次恢复及580项完整参数哈希均通过；原输入哈希、全池覆盖及原评分汇总再次核对后，才发布替代审计并停止冗余串行审计。原控制器状态、进度、容器退出码和全部分片证据保留，当前控制器由完成监督程序按完整审计证据收尾，未重复提交科学训练或编辑。

[RECENT审计与替代方式](../results/rippleedits-paper-reproduction-v1/runs/gpt2-xl-rome-recent/audit.json)、[调度修订](../results/rippleedits-paper-reproduction-v1/schedule-amendment.json)、[完整替代证据](../results/rippleedits-paper-reproduction-v1/audit-scheduling-intervention-20261009/distributed-audit.json)、[POPULAR分析](../results/path-hop-analysis-v1/gpt2-xl-rome-popular.json)、[RANDOM分析](../results/path-hop-analysis-v1/gpt2-xl-rome-random.json)、[RECENT分析](../results/path-hop-analysis-v1/gpt2-xl-rome-recent.json)、[完成清单与哈希](../results/path-hop-analysis-v1/completion-manifest.json)、[汇总及图表来源](../results/path-hop-analysis-v1/summary.json)、[主图PNG](../results/path-hop-analysis-v1/hop-position-summary.png)/[PDF](../results/path-hop-analysis-v1/hop-position-summary.pdf)。七项既有分组契约测试通过；本次查阅本地追踪终点，未做新的独立W&B云端核验。

<a id="hebbian-interface"></a>
### 简化架构的事实模块替换

复用作者SwiGLU与固定读取器实现，一个开发世界后，三个独立世界完成单跳A→B事实模块整体替换。CE和完整向量MSE的A/B精确键读出、原A调用、固定读取器换B后的调用均100%；保留错误A模块时全池0.78%、实际改变的事实子集0%。CE输出与目标夹角约.753–.774弧度，仍能正确接入，因此精确向量相等不是本低负载归一化设置的必要条件。固定词向量、单位V/O、无残差，与普通GPT有明确结构差异。[结果](development-artifacts/hebbian-interface-v1/summary.json)、[论文实现对应](development-artifacts/hebbian-interface-v1/literature-alignment.json)。

<a id="memory-reuse"></a>

两跳扩展在三新世界×两初始化完成：先学A记忆及调用，再冻结读取器，换入仅用B原子训练的新MLP，B组合标签完全不参与训练。CE／CE加统一表示约束的A两跳99.46／99.97%，B单跳均100%，B两跳 **66.39／94.66%**；六配对收益均正，错误旧A记忆按B评分仅0.52%。支持本简化连续调用架构中的新事实复用，处理同时改变事实MLP和在其上学到的注意力，不能唯一归为输出几何。B与A共享词表、映射重排；不是普通端到端模型中后来新增实体的实验。[确认汇总](development-artifacts/memory-reuse-v1/confirmation/report/summary.json)、[冻结契约](development-artifacts/memory-reuse-v1/confirmation-plan.md)。

<a id="memory-interface-next"></a>
### 已有强组合能力之后的原子编辑传播

复用共享GPT的首跳CE和对齐父模型，局部更新共享MLP；目标原子CE加32条旧原子父分布KL，其余冻结。开发只用写入和保持选择lr=1e-5，不用D传播。正式三世界×16图选事实×两初始化×两训练臂，共192编辑及192复习分支，全部重载与独立计分完成。

| 父模型 | 原子写入E | 父正确U原子保持 | 派生新答案D，编辑／复习 | 编辑前D正确覆盖 |
| --- | ---: | ---: | ---: | ---: |
| 表示对齐 | 100% | 99.76% | **4.08%／0.26%** | 93.72% |
| 首跳CE | 100% | 99.93% | 9.38%／0.78% | 25.69% |

主表先逐目标、再初始化／世界等权。对齐原始D计数12/298，限制编辑前父正确且编辑成功为12/279；首跳CE为21/298及7/62。分母覆盖差很大，不能由CE较高条件率推断普遍优于对齐。回放及必要原子均100%；父正确U熟悉／严格组合保持，对齐99.02／96.59%，CE92.02／84.21%，有限KL未严格保护所有组合行为。

这直接表明：**在这一原本强组合的父模型中，事实改对并保持多数旧原子，仍不足以让派生答案更新。** 尚未唯一定位传播失败于早期状态。自产桥重编码和覆盖实验见[第二层](#bridge-reencoding)，后续写入方式比较见[roadmap](roadmap.md#write-interface)。[正式编辑汇总](../results/interface-editing-confirmation-v1/report/summary.json)、[独立复算](../results/interface-editing-confirmation-v1/report/independent-endpoint-review.json)、[开发选择](../results/interface-editing-development-v1/report/decision.json)。

<a id="write-target-editing-v1"></a>
### 同一新事实写到哪里，决定更新后能否继续使用

**开发3个作业、正式配对扩展18个作业均已完成并通过每节点独立重载。** 复用已有三个世界×两个初始化的强组合共享GPT父模型，只考察每父模型四个图选严格首跳事实。三种目标依次为最终原子CE、再加早期实体CE、再加新实体输入表示的方向对齐；每种目标都有原旧事实复习sham。可更新的共享MLP、32条原子回放、lr=1e-5和512步完全相同，额外两项权重均为.3，始终只用同一实体标签，不给第二关系或派生答案。开发仅按E、必要原子和原子保持确认操作有效，未按D选择配方。

| 写入目标 | 新原子E | 派生新答案D | 旧派生答案残留 | 早期新实体读出 | U原子保持 | 父正确U严格组合保持，世界均值 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 最终原子CE | 24/24 | **3/96（3.13%）** | 72/96 | 0/24 | 99.68% | 96.47% |
| 加早期实体CE | 24/24 | **66/96（68.75%）** | 3/96 | 24/24 | 99.57% | 92.42% |
| 再加方向对齐 | 24/24 | **82/96（85.42%）** | 0/96 | 24/24 | 99.51% | 89.84% |

D为全部答案发生改变的合法未训练严格组合，每臂24次编辑、96次查询；案例与初始化是世界内配对重复，不增加独立世界数。三个世界的D依次为最终CE **1/32、1/32、1/32**，早期CE **25/32、21/32、20/32**，再加对齐 **28/32、30/32、24/32**。D父模型正确覆盖86/96，对应新答案为3/86、61/86、75/86；表中完整池没有用该条件筛选。必要后继原子各96/96正确，早期新方向余弦依次为.5306、.8752、.9996。最终CE基线的192个保存节点与历史编辑的完整模型摘要逐一一致。

**传播改善伴随真实的组合保持损失。** 父模型原本答对的U严格查询共11258次评测，三个编辑臂仍答对10872、10416、10128次；合并保持率96.57／92.52／89.96%，对应表中先案例、再初始化、再世界平均的96.47／92.42／89.84%。U严格完整池准确率为89.90／86.16／83.75%，不能把下降仅解释为父模型原本较弱。三个自身sham的父正确U严格保持世界均值为98.94／97.28／99.11%，新D为1/96、1/96、0/96。早期CE的sham仅保留55/96旧D，最终CE和方向对齐sham分别为86/96、87/96，说明仅让实体可读出也可能扰动已有用法。

这支持一个清楚的结论：**只改好最终事实答案，与让后续计算用上新事实不是同一个训练目标；把同一实体更新写入早期状态能显著改善使用，对齐其表示还能进一步改善。** 但共享MLP在两次执行中都被改变，尚不能把效应唯一归因于一个局部状态；组合副作用也未解决。范围限定于既有三个世界的严格首跳配对扩展，不称新世界确认，不外推到第二跳或自然预训练模型。

[完整报告与逐世界sham/保持分母](../results/write-target-confirmation-v1/report/report.md)、[原始预测复算及报告源码哈希](../results/write-target-confirmation-v1/report/summary.json)、[开发报告](../results/write-target-development-v1/report/report.md)、[冻结正式配置](development-artifacts/write-target-confirmation-v1/frozen-config.json)、[本轮协议](development-artifacts/learning-use-20261007/protocol.md)。云端同步状态由独立追踪核验记录，不由科学完成自动代替。

<a id="sequential-transfer-v1"></a>
### 先学组合、再学新事实：有迁移，但远未形成完整的新知识使用

**正式24条训练与独立端点重载全部完成，失败0。** 在同一个2Wiki真实事实图的三个固定划分上，各做两个初始化、四种训练历史；这些划分不是三个独立世界。从随机初始化训练普通四层、宽256的完整Transformer，先13000步、再7000步。顺序组合臂先学A原子与A组合，随后仅学B原子及A原子回放；原子对照先做A原子重复；联合臂打乱顺序组合臂完全相同的训练样本多重集；A-only sham共享顺序组合臂的阶段A，后续只复习A。所有臂均不给含B的组合答案，BB实体已在A事实中出现；学习率按相同全局步数调度。原子对照匹配更新与样本槽，不声称监督token及事实提及完全相同。

| 训练历史 | 终点B原子，划分均值 | BB直接回答 | BB自主两次调用 | AA旧事实新组合 |
| --- | ---: | ---: | ---: | ---: |
| 先组合、后新原子 | 100% | **79/384（20.57%）** | 384/384（100%） | 117/384（30.47%） |
| 先原子重复、后新原子 | 100% | 12/384（3.13%） | 384/384（100%） | 1/384（0.26%） |
| 同材料联合训练 | 100% | 89/384（23.18%） | 384/384（100%） | 179/384（46.61%） |
| A-only sham，不学习B | 7.70%（58/752次答对） | 38/384（9.90%） | 40/384（10.42%） | 105/384（27.34%） |

先平均划分内初始化，再对划分等权；表中BB和AA每格分母相同，均值也等于合并比例。384是跨运行的查询评测次数，包含初始化重复及可能的划分重叠，不是384个独立事实。所有臂A原子均100%；自主两次调用提供关系分解并增加计算，不能与一次直接回答视为相同推理预算。

**组合经历与真正加入新知识的作用分别可见。** 顺序组合相对原子对照的BB提高17.45个百分点，六配对均正；但它在学B前已答对55/384（14.32%），不能把终点20.57%全部算作新知识迁移。学B后净增24次正确，即+6.25个百分点，六配对五正一负；相对同父A-only sham提高41/384，即+10.68个百分点，六配对均正，三个划分分别+10.94／+13.28／+7.81。顺序组合的三个划分BB为24.22／15.63／21.88%，六端点为7–17/64。联合训练比顺序组合仅高2.60个百分点，六配对三正两零一负；它还把A组合练习分散到整个训练过程，不能单独将差值归为新知识接入。

**旧组合没有完整保持。** 顺序组合的AA从134/384（34.90%）降至117/384（30.47%）；原本正确的134次只保留86次，丢失48次，同时新增31次正确。合并保持率64.18%，按初始化及划分平均为64.78%。A-only sham也只保留86/134，终点新增正确为19次，说明继续原子训练本身伴随旧能力变化。原A训练组合从4608/4608降至3335/4608（72.37%），不能只凭AA净下降4.43个百分点认定使用能力基本不变。联合臂AA终点比顺序臂高16.15个百分点。顺序臂BA／AB的划分均值为19.27／20.57%；BA分母不等，合并计数35/192另列，AB为79/384。

开发一个单独划分×两初始化×四历史的8条训练也均完成，预算为8000＋4000步；顺序组合AA为63/128→57/128，BB为20/128，对应原子／联合／sham为3/128、30/128、18/128。开发只按原子、A训练组合与AA确认操作及保持情况，不按BB选择配方；正式阶段仅按数据规模增加曝光预算。开发结果不并入正式均值。

这些结果支持：先前组合练习有利于后来知识的使用，新原子学习也带来超过旧事实复习的有限收益；但在单跳和自主逐跳调用均正确时，直接新组合仍低，旧组合也会改变。它尚未建立完整的知识复用，也不能把全部失败解释为新知识写入，因为原有组合能力同时发生了遗忘。

[完整逐划分／初始化报告](../results/sequential-transfer-confirmation-v1/report/report.md)、[计数复算、配对差与报告源码哈希](../results/sequential-transfer-confirmation-v1/report/summary.json)、[主图PDF](../results/sequential-transfer-confirmation-v1/report/sequential-transfer-summary.pdf)、[主图PNG](../results/sequential-transfer-confirmation-v1/report/sequential-transfer-summary.png)、[开发独立报告](../results/sequential-transfer-development-v1/report/report.md)、[开发决定](../results/sequential-transfer-development-v1/development-decision.json)、[冻结正式配置](development-artifacts/sequential-transfer-confirmation-v1/frozen-config.json)。

<a id="sequential-replay-v1"></a>
### 继续回放旧组合：保持明显改善，新组合仍停留在约21%

**文献配方复测已完成：开发4条、配对扩展12条，全部独立重载通过，失败0；W&B云端16条也均通过核验。** 本轮采用Ibrahim等TMLR 2024论文已测试的50%旧数据回放及阶段优化重启：新建AdamW，β=(0.9,0.95)、weight decay=0.1、clip=1，前1%更新warmup至3e-4，再余弦下降至3e-5。回放沿原训练顺序；没有扫描比例或按BB挑选配方。原论文验证大规模语言建模的保持与适应，本轮将该方法用于新事实组合使用。[原论文§4.2、§5、§6.2及附录B](https://arxiv.org/html/2403.08763v4)、[文献与官方代码核对](development-artifacts/sequential-replay-literature-v1/literature.md)。

沿用上一节同一个真实图的三个已观察划分、每划分两个初始化，从原阶段A组合父权重各分出两臂；这是配对扩展，不是新独立世界确认。每臂新增7000更新、batch128，每步64条完全相同且顺序一致的B原子。其余64个旧槽，`atom_replay`只回放A原子；`full_replay`沿原阶段A完整序列回放，累计224,000条A原子与224,000条AA训练组合，相邻步的旧槽交替来自原子与组合。两臂均重置优化器，不给评价AA或含B的组合答案。固定的是更新数、样本槽和B曝光；变长QA的监督token及计算量并非完全相等。原子对照只改变旧回放内容，A原子曝光也相应不同。[冻结协议](development-artifacts/sequential-replay-literature-v1/protocol.md)。

| 同父权重、同优化配方的阶段B | A/B原子 | 原AA训练题 | 未见旧组合AA | 新组合BB | BB自主两次调用 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 只回放旧原子 | 均100% | 613/4608（13.30%） | 16/384（4.17%） | 19/384（4.95%） | 384/384（100%） |
| 回放完整旧训练分布 | 均100% | 4608/4608（100%） | **145/384（37.76%）** | **80/384（20.83%）** | 384/384（100%） |

每臂A原子为8766/8766、B原子为752/752。先平均划分内初始化，再对划分等权；表中AA/BB各运行分母相同。计数包含重复初始化与划分重叠，不当作独立事实；自主两次调用由外部提供分解，单列而不当作直接组合成功。

**完整回放优于本轮匹配的原子回放，新组合仍与先前水平接近。** BB差值为+15.89个百分点，六个配对均正，三个划分的差值为+18.75／+14.06／+14.84；完整回放的三个划分BB为22.66／16.41／23.44%。两臂共享的学B前BB为55/384（14.32%）；完整回放后净增25次正确，+6.51个百分点，六次前后比较五正一平，不能把全部终点正确都算作新知识迁移。其80/384（20.83%）与上一轮顺序组合79/384（20.57%）接近，未突破已有水平。上一轮优化器与采样契约不同，只作背景；0.26个百分点的跨轮差值不能归因于回放。主比较表明，在本轮优化重启配方下，完整旧回放避免了只回放原子时的严重组合退化。

**训练题保持100%仍不等于旧组合能力完整保持。** 未见AA从共同的134/384（34.90%）变为完整回放145/384（37.76%）；原先正确的134次中保留99次、丢失35次，同时新增46次正确。合并保持率73.88%，按初始化及划分等权为73.97%；只回放原子保留13/134，等权保持率10.81%。完整回放六次AA前后比较仍有两次下降，净准确率提高不能掩盖旧正确题丢失。辅助BA的完整／原子回放均值为30.68%／4.98%，原始计数58/192与9/192；BA分母不等，均值不等于合并比例。AB为89/384（23.18%）与31/384（8.07%）。

开发单独使用两个初始化、每臂4000更新，完整／原子回放的AA为71/128／12/128，BB为33/128／17/128，A/B原子均100%；开发结果不并入扩展均值。开发仅按原子学习、训练AA与未见AA检查操作，没有依据BA/AB/BB改变配方、预算或样本。结果支持在学习新事实时继续练习旧组合以减轻用法遗忘，并帮助新组合回答；它没有解决“原子与逐跳调用均正确、直接新组合仍低”的问题，也尚未保证未见旧组合逐题保持。

[完整配对报告](../results/sequential-replay-paired-v1/report/report.md)、[原始计数与报告源码哈希](../results/sequential-replay-paired-v1/report/summary.json)、[主图PDF](../results/sequential-replay-paired-v1/report/sequential-replay-summary.pdf)、[主图PNG](../results/sequential-replay-paired-v1/report/sequential-replay-summary.png)、[开发独立报告](../results/sequential-replay-development-v1/report/report.md)、[开发决定](../results/sequential-replay-development-v1/development-decision.json)、[冻结配对配置](development-artifacts/sequential-replay-paired-v1/frozen-config.json)、[开发云端核验](../results/sequential-replay-development-v1/tracking-cloud-audit.json)、[配对扩展云端核验](../results/sequential-replay-paired-v1/tracking-cloud-audit.json)。

<a id="loop-learning-v1"></a>
### Loop同起点比较：容量、重复计算与新增知识使用

**16条开发训练与独立重载全部完成，失败0，共192000次更新。** 同一2Wiki开发划分、两个初始化，每次先8000步旧原子＋训练AA，再4000步新原子＋完整旧分布回放。相同宽度/循环数内，shared、untied、mlp_shared具有完全相同的初始展开权重与前向；所有配对使用相同训练样本顺序。浅层参照只执行一次基础块组。新增事实不等于旧事实改写，答案监督也不等于自然全token预训练。

这里的`untied`把同一组随机初值复制到不同层，然后分别训练；它是固定初始前向、改变参数绑定的辅助对照，不是各层独立随机初始化的普通Transformer。以下“共享/解共享”沿用历史名称，结论限定于这一初始化条件；普通模型的架构效率须另有适当基线。

宽256、两基础块重复两次的四臂结果如下，数值是两个初始化均值：

| 架构 | 唯一参数（M） | 未见AA：A末→B末 | 新＋新BB：A末→B末 | BB净增（百分点） | 原先正确AA保持 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 整块共享 | 14.51 | 40.63→50.00% | 18.75→22.66% | +3.91 | 88.39% |
| 完全解共享 | 16.09 | 42.97→52.34% | 17.19→21.09% | +3.91 | 87.00% |
| 仅MLP共享 | 15.04 | 42.19→48.44% | 18.75→22.66% | +3.91 | 87.50% |
| 两层浅层参照 | 14.51 | 39.84→45.31% | 17.19→22.66% | +5.47 | 82.18% |

三种等执行深度结构的BB净增相同，不能将共享末分较高解释为新增知识使用的优势。所有16条A末旧原子、B末新旧原子及训练AA均100%，但未见组合明显低于100%；当前知识负载没有测出原子容量上限。

本批`ReproductionGPT`只读取各执行位置本轮计算的KV，没有首轮共享KV通路。因此该批的参数共享结果，不能代替[历史共享KV](#shared-cache-branch-v1)的访问比较，也不能证明二者结合后的顺序新增知识使用无效。

在读取主批端点前冻结了宽度/循环扩展；共享与解共享每点各两个初始化。下表为B末结果，括号是A→B净增：

| 宽度；循环数 | 共享AA | 解共享AA | 共享BB（净增） | 解共享BB（净增） | 每次完整训练估算FLOPs |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128；2 | 31.25% | 35.16% | 15.63%（+2.34） | 14.06%（+1.56） | 1.19×10¹⁵ |
| 256；2 | 50.00% | 52.34% | 22.66%（+3.91） | 21.09%（+3.91） | 2.99×10¹⁵ |
| 256；4 | 40.63% | 53.13% | 23.44%（+10.94） | 22.66%（+7.03） | 4.25×10¹⁵ |

宽度增加时两架构的组合均值提高；循环2→4时，共享模型的AA下降、BB末分仅增加0.78个百分点，不能说更多循环统一改善知识使用。R4共享BB净增更大也伴随较低的A末基线，且继续回放旧组合，没有无B续训对照，暂不作新增知识因果归因。相同lr的AdamW未作逐架构最优校准；三个点、同图两初始化不能确定scaling指数或排除更长期grokking。全池AA/AB/BB各64题、BA14题，中间panel与完整端点评分分母分开保留。

微观诊断覆盖16条训练的80个节点及128个实际更新节点。固定训练池的每角色2例，当前权重临时解共享后的前向误差均0，梯度求和最大误差4.77×10⁻⁷，未发现非有限数。主批360个调用位置梯度对中359个余弦非负；这只测同一角色在不同调用位置的局部关系，不能排除新旧知识间干扰，也不能据此解释性能。实际AdamW更新另行记录。真实小Transformer的FP64跨调用核等式误差2.22×10⁻¹⁵，普通SGD一阶余项符合该例的二阶缩放；数学恒等式与训练效果分别成立。

证据：[主批报告](../results/loop-learning-development-v1/report/report.md)、[学习曲线](../results/loop-learning-development-v1/report/learning-steps.png)、[规模扩展](../results/loop-learning-scaling-development-v1/report/report.md)、[主批微观诊断](../results/loop-learning-development-v1/diagnostic-report/report.md)、[扩展诊断](../results/loop-learning-scaling-development-v1/diagnostic-report/report.md)、[数值核验](development-artifacts/loop-learning-development-v1/verify_loop_kernel.json)、[主批云审计](../results/loop-learning-development-v1/tracking-cloud-audit.json)、[扩展云审计](../results/loop-learning-scaling-development-v1/tracking-cloud-audit.json)。训练时间含少量更新范数采集，显存峰值包含诊断副本；FLOPs为矩阵乘估算，不是硬件测量。

<a id="loop-kv-learning-v1"></a>
### 普通独立四层与Loop加首轮KV：本配方没有稳定的新增知识收益

**新增四条12k训练、独立重载和W&B云端核验全部完成，失败0；新增48,000更新，复用两条已完成的原Loop。** 普通四层各层独立随机初始化；两层Loop执行两轮；同Loop第二轮额外读取同层首轮KV，保留完整当前因果KV、联合softmax及连通梯度。沿用同一2Wiki开发划分、两初始化、A8000步及B4000步、完整旧内容回放、相同样本顺序和优化器规则，不复制两层随机张量作为普通基线。

| 模型 | 唯一参数M | 未见旧AA：A末→B末 | 新事实BB：A末→B末 | BB净增pp | 父正确AA保持 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 普通独立四层 | 16.09 | 44.53→50.78% | 15.63→23.44% | +7.81 | 80.74% |
| 两层Loop两轮，复用 | 14.51 | 40.63→50.00% | 18.75→22.66% | +3.91 | 88.39% |
| 同Loop加首轮KV | 14.51 | 37.50→47.66% | 17.19→21.09% | +3.91 | 90.03% |

各数为两初始化均值，原子A/B及训练旧组合终点均100%。共享KV两个初始化的BB末分20.31/21.88%，相对原Loop为−3.13/0pp；A→B净增为0/7.81pp，相对原Loop为−1.56/+1.56pp。因此没有稳定的新增知识使用收益，预定无B续训门槛未通过，第二步未启动，不扩大R/窗口/mHC网格。普通模型BB终点配对一正一负，均值稍高且旧保持较低，不能据此宣称普适架构优劣。

普通模型与原Loop累计矩阵乘训练FLOPs均值均约2.993×10¹⁵，共享KV约3.010×10¹⁵；参数与计算分别报告。旧Loop诊断采集带来少量额外时间及副本显存，其墙钟不作为纯架构速度测量。AA/AB/BB各64题、BA14题，两个初始化来自同一划分，不是新世界确认。结果限定于答案监督、真实名称和本开发预算，未证明自然语言预训练或更长期训练的架构上限。

[完整端点报告](../results/loop-kv-learning-development-v1/report/report.md)、[逐题转移、曲线及成本](../results/loop-kv-learning-development-v1/report/summary.json)、[预定后续门槛分析](../results/loop-kv-learning-development-v1/report/analysis.json)、[独立重载完成清单](../results/loop-kv-learning-development-v1/completion-manifest.json)、[云端核验](../results/loop-kv-learning-development-v1/tracking-cloud-audit.json)、[冻结协议](development-artifacts/loop-kv-learning-development-v1/protocol.md)、[旧产物不变核验](../results/loop-kv-learning-development-v1/reuse-provenance-audit.json)。

<a id="moe-sequential-v1"></a>
### MoE、普通模型与宽模型：三层顺序学习成本比较

**新增四条12k训练、独立重载和W&B云端四条核验全部通过，失败0。** 同一已观察2Wiki开发划分、两个初始化，复用最近的两条普通独立四层D4；新增MoE M4和宽模型W4各两条，共48,000更新。旧D4的A/B四个端点共7,900条预测在新构建代码下完全复算，旧产物哈希未变。不把两个初始化或六个模型条件称新世界确认。

四层完整GPT-2、宽256，均只执行一次。普通为4d FFN，MoE为每层8个2d专家、每token选2个，宽模型为16d FFN。前者近等激活FFN计算，后者近等MoE总参数。固定A8000/B4000、完全相同的训练ID顺序、答案与EOS监督、优化器和完整旧分布回放；不给含B组合答案。MoE另加已验证的.01整批路由平衡项，不增加知识标签。

| 模型 | 总参数M／名义选中M | AA：A末→B末 | BB：A末→B末 | BB净增pp | 原正确AA保持 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 普通D4 | 16.091／16.091 | 44.53→50.78% | 15.63→23.44% | +7.81 | 80.74% |
| MoE M4 | 22.410／16.100 | 49.22→55.47% | 17.19→22.66% | +5.47 | 77.72% |
| 宽W4 | 22.395／22.395 | 54.69→49.22% | 17.19→21.09% | +3.91 | 72.86% |

三臂A末旧原子、B末新旧原子与训练旧组合均100%，BB自主两次调用也均100%；第一层完整学习曲线及中途退化另报，不因终点满分省略。MoE相对普通的B末AA差+3.13/+6.25pp，两个初始化都高；但BB差−4.69/+3.13pp、BB净增差−7.81/+3.13pp，方向混合，原正确AA保持差−5.81/−0.23pp。AA较高终点不等于旧正确题更好保持。MoE−宽模型的BB一平一正，保持一负一正，不称三层统一优势。

平均累计矩阵乘训练FLOPs为普通/MoE/宽 **2.993/2.574/5.447×10^15**；MoE较普通约少14%，较宽约少53%。MoE只计算真实有效token的专家，dense计算padding，差值含实现因素；名义选中参数包含完整Embedding计数，不是实际FLOPs。新MoE/宽模型训练计时均值426.33/154.57秒，较少矩阵估算没有成为本实现的速度优势；旧D4时间不是同期吞吐基准。

本轮没有测出原子容量上限，也没有稳定的新知识组合或旧正确AA保持优势。一个开发划分、两初始化、共同配方不建立普适架构结论；没有无B续训，BB净增不唯一归因B写入。三层成本、全部逐题转移与源文件哈希均保留，不自动扩大专家/路由/lr网格。

[完整报告](../results/moe-sequential-development-v1/report/report.md)、[结果解释](../results/moe-sequential-development-v1/report/interpretation.md)、[原始计数及配对差](../results/moe-sequential-development-v1/report/summary.json)、[学习曲线PNG](../results/moe-sequential-development-v1/report/learning-curves.png)/[PDF](../results/moe-sequential-development-v1/report/learning-curves.pdf)、[完成清单](../results/moe-sequential-development-v1/completion-manifest.json)、[云端核验](../results/moe-sequential-development-v1/tracking-cloud-audit.json)、[冻结协议](development-artifacts/moe-sequential-development-v1/protocol.md)。

<a id="knowledge-change-v1"></a>
### 在知识变化中学习调用

**开发及三个全新世界均已完成，报告无缺项。** local／shared_full×fixed／changed_atomic／changed_use六臂，从同架构父状态按同预算训练。每片段临时用原子事实修改首MLP，冻结它训练其余参数，再精确恢复首MLP；训练更新地址与16个测试编辑地址分开。不是二阶元学习或无限制变化世界预训练。changed_atomic使用不受影响的组合问题，与changed_use查询曝光不同；fixed与changed_use有相同查询前缀，但变化标签提供反事实监督。

确认终点原子、训练过程临时写入均100%；local三臂熟悉组合32.16／32.72／32.60%，shared_full为49.30／53.26／47.19%，严格0.65–1.11%。24个调用学习父模型及768个edit/sham分支包括开发，不把全部分支当独立世界。

| 架构／训练 | 熟悉首跳D，全池 | 同类D父正确覆盖 | 父正确D的新答案 | 熟悉次跳D |
| --- | ---: | ---: | ---: | ---: |
| local／fixed | 4.17% | 27.08% | 0% | 0% |
| local／changed_atomic | 6.25% | 41.67% | 0% | 0% |
| local／changed_use | 8.33% | 25.00% | 0% | 0% |
| shared／fixed | 14.58% | 56.25% | 4.17% | 0% |
| shared／changed_atomic | 14.58% | 54.17% | 11.11% | 0% |
| shared／changed_use | **31.25%** | 39.58% | **27.78%** | 0% |

所有测试编辑E均100%，各角色U原子保持99.29–99.97%。共享KV熟悉首跳的fixed→changed_use逐世界为0→0、18.75→18.75、25→75%，平均收益来自第三世界，不能称三世界一致改善。父正确条件池在不同臂之间变化，必须连同覆盖及sham解释，不能只比较条件百分比。严格首跳最高2.08%、严格次跳全0%，父正确严格池几乎为空，不能称普遍更新能力已形成。它与“先学组合、再只学全新原子事实”的顺序迁移不同；后者已完成，见[顺序新增结果](#sequential-transfer-v1)，不从本节旧状态重复启动。

[完整阶段报告](../results/knowledge-change-v1/report/report.md)、[所有世界、sham与逐例分母](../results/knowledge-change-v1/report/summary.json)、[科学完成状态](../results/knowledge-change-v1/controller-state.json)、[云端56运行核验](../results/knowledge-change-v1/cloud-final-audit.json)、[冻结协议](development-artifacts/knowledge-change-v1/protocol.md)。

### 其他编辑证据与前提边界

| 批次 | 已观察结果 | 可支持的解释及证据 |
| --- | --- | --- |
| 深度/step的高lr原编辑 | 96分支E和回放均100%，未回放原子仅40.58–68.16%，复习也退化 | 原保持前提不成立，不能把低D归因于接口；[机制汇总](development-artifacts/depth-step-v1/mechanism-summary.json) |
| 同批保持校准后的预留事实 | 72分支，各状态E/R/Kdev/U/后继原子均100%，全量D及父正确D传播均0% | 一个世界六事实、每状态8条D，支持写入与传播分离；[预留事实汇总](development-artifacts/depth-step-v1/edit-followup/summary.json) |
| Grokking长期训练编辑 | 576分支直接写入全成功；512k条件池首/次跳传播普通91.67/33.33%、Loop88.89/63.19% | 每角色仅两事实；早期保持差，原回放漏次跳角色。均衡校准四候选未通过共同保持门槛，依赖分支未启动；[分析](development-artifacts/grokking-dynamics-v1/assessment.json)、[失败决定](development-artifacts/grokking-dynamics-v1/balanced-editor/calibration/decision.json) |
| 共享KV原批 | 64次编辑E全成功；local/full/window/detached熟悉首跳66.67/66.67/33.33/66.67%，熟悉次跳和严格传播全0% | 每臂首跳只有三个图选事实，未建立稳定共享传播优势；detach同时改变编辑梯度；[报告](../results/shared-cache-branch-v1/report/report.md) |
| 残差/KV扩展 | 新增64次编辑E全成功，严格传播全0%，熟悉次跳仅mHC/local有16.67% | 复用已知世界、分母小，不能称独立确认；[逐例结果](../results/residual-cache-comparison-v1/report/summary.json) |
| Qwen共同复习v2 | 三父模型更新E均100%；D新答案更新/复习为6.25/9.38%、9.38/12.50%、18.75/12.50% | 关联组留出问法没有同方向差；未回放U事实明显下降，原本D仅0/32、2/32、3/32，未检验大量原成功组合的失效；[完整报告](development-artifacts/small-lm-composition-v2/summary.json) |

<a id="qwen"></a>
### 局部约束与参数更新保持

原始Qwen3-0.6B的约束诊断，12目标、64保持、64未约束事实和16自然文本：第14层qa局部参数代价相对无约束，功能保护1.20倍、全位置表示保护4.29倍；只保护末位置特征仍有最终输出漂移。四个受测down矩阵满行秩，不证明门控或长期学习瓶颈。[约束产物](development-artifacts/qwen-constraints-v1/)。

24条256步路径开发中，128步16/16位置恢复条件都增大回放分布漂移；完整更新R KL .00691–.01138，恢复实体末位置增大约4.8–20.1%、查询内容23.6–297.3%。预定推进判据未通过，96确认轨迹没有启动；U/自然文本没有更新后评价。与跨位置共同保持相容，尚未识别唯一抵消机制。[路径产物](development-artifacts/qwen-path-learning-v1/)、[事后复盘](development-artifacts/path-learning-reassessment-20260929/)。

<a id="organization"></a>
### 数据组织与关系编辑：历史系列

这些批次有复用，不能把行数相加当独立模型总数。保留它们是为了说明：学习改善、关系依赖与更新保持不能互相替代；它们不是当前自动扩展队列。

| 批次 | 结果及解释边界 | 证据 |
| --- | --- | --- |
| v2.3 顺序学习 | 8学习、256编辑；课程效应依方法，例外更新受默认/个人城市传播与保持损伤限制 | [配置入口](../configs/bios-symbolic-development-v1.json) |
| v2.4 文档组织 | 首次99%平均步数A/B6272、C9856；直接改96/96，例外联合成功0/48；编辑组织优势随参数范围反转 | [报告](development-artifacts/organization-v1/report.md) |
| v2.5 容量/QA留出 | 42容量诊断、24条半QA留出轨迹；更多预算仍改善，不证明容量上限 | [配置入口](../configs/bios-capacity-development-v1.json) |
| v2.6 关系交叉 | 12学习、96编辑；组合匹配+0.842个百分点、四块同向，编辑传播未继承正收益 | [报告](development-artifacts/cross-v1/report.md) |
| v2.7 规模交叉 | 48学习、384编辑；约221.8万参数档不匹配任务平均+22.45个百分点，规模也改变例外损伤 | [报告](development-artifacts/cross-scale-v1/report.md) |
| v2.8 机制开发 | 新增60学习/续训、480编辑；自主两次调用修复部分组合，监督/上下文只修复部分传播，内部候选无稳定选择性 | [审计](development-artifacts/mechanism-v1/development-completion-audit.json)、[勘误](development-artifacts/mechanism-v1/ERRATA.md) |
| v2.9 八新世界确认 | 96学习、768共同E39编辑；例外学习39.31→73.76%，一致更新传播87.73→64.58%，新例外传播12.62→15.10% | 差+2.49pp区间[−2.17,+7.15]、Holm p=.139；主要例外编辑收益未确认；[统计](development-artifacts/shortcut-confirmation-v1/statistics/report.md) |
| v2.11 上下文梯度 | 四中性轨迹；方向余弦.295/.417/.371高于打乱约.003–.005，向量相对误差仍>1 | [报告](development-artifacts/context-gradient-v1/report.md) |
| v2.12 监督与均匀attention | 冲突传播随四档监督16.02/18.07/20.58/23.87%，组织匹配劣势−7.40→−16.44pp | 真实/均匀梯度余弦约.998，统计近似仍失准；[报告](development-artifacts/followup-v1/report.md) |
| v2.13 路径与最小模型 | 24/24成熟冲突更新削弱正确相对冲突优势，23/24正确概率仍增加；原长期0/6没有给足另一关系目标 | 不能据原失败证明不可分离；[路径](development-artifacts/path-transfer-v1/report.md)、[最小模型](development-artifacts/path-minimal-v1/report.md) |
| v2.15 目标补齐/独立世界 | 补齐目标后普通联合微调6/6成功；八新世界GN严格联合约35%，Adam23.44% | 成本与曲率近似敏感性另报；[方向](development-artifacts/direction-v1/)、[汇总](development-artifacts/direction-report-v1/) |
| v2.16 四格关系响应 | 一致/冲突组合40/48和5/48；直接、归属、自主调用96/96；编辑前22/24及四格23/24个人事实响应更强 | 确认不期望的功能依赖，未定位唯一组件或形成原因；[产物](development-artifacts/relation-response-v1/) |

<a id="twohop-module"></a>
### 早期模块与深度证据

Hebbian E1模块使用MQuAKE原始事实图962原子、598去重链，人工实体向量、四宽度×三种子。宽512单跳98.61%、连续两跳68.67%，两原子都可读出子池仍69.38%；宽1024二者100%。范数充分界在宽128/256/512认证覆盖0、1024覆盖100%，不是自然语言基准成绩。[模块结果](development-artifacts/twohop-hebbian-v1/module-results.csv)、[审计](development-artifacts/twohop-hebbian-v1/completion-audit.json)。

早期两跳深度28训练、两世界×两初始化，在4096步原子、训练组合及外部调用均100%，留出组合架构均值仅0.88–3.34%；六层较一层+1.95pp，相对近等参数两层仅+0.71pp且方向混合。桥实体能读出没有对应可靠反事实使用；后来的成功批次同时改变图、支持、优化和预算，不能只归因于训练更久。[完整早期报告](development-artifacts/twohop-depth-v1/README.md)。

<a id="hebbian"></a>
### Hebbian启发学习及已结束支线

| 历史批次 | 已有结论 | 证据 |
| --- | --- | --- |
| Hebbian学习v1 | 24数学校验、127训练；C1相对C0/C2后续学习收益未获支持，C1−C0 Q-AUC约−.0005、区间跨零 | [执行与结果](development-artifacts/hebbian-learning-v1/resume-execution-status.json)、[修订](development-artifacts/hebbian-learning-v1/execution-amendments.md) |
| Future Work v1 | 24符号训练、72复习；相关性作用随层数配置改变，复习未稳定修复组合；核的表现预测仅属历史诊断 | [产物](development-artifacts/hebbian-future-v1/) |
| Future Work v2 | 108训练、540节点；完整答案下历史核增益缩小，固定竞争/近等参数控制未稳定复现原方向反转，原子和自主调用100%而直接组合有缺口 | [产物](development-artifacts/hebbian-future-v2/) |

C0/C1/C2使用相同SwiGLU和梯度优化，只改变训练目标，不能称“Hebbian与普通MLP”的架构比较。历史预测支线保留结果，不列为下一步研究；当前机制问题和理论使用范围见 [roadmap](roadmap.md#theory)。

<a id="reproduction"></a>
## 证据、完成状态与复现

本页把科学结果、证据边界和下一步规划分开；下一步只见 [roadmap](roadmap.md#next-experiments)。批次报告、冻结协议、原始数据、全部失败尝试及预测继续保存在 `docs/development-artifacts/` 与 `results/`。本次重整没有重跑实验、改写历史评分或删除科学产物。

2026-10-07本地核对：knowledge-change、residual-cache、composition-data-curves、loop-block-depth、parametric-architecture主矩阵、OO优化归因的控制器均为complete；bridge-workspace的42项也已complete。它们此前在本页的“排队／运行中”文字已删除。科学完成不等于远端追踪完整：已明确核验的批次在各节给出云端审计；其余以本地结果为据，不由controller状态推断W&B。capacity-stability的科学四条完成与W&B失败分别保留。

本轮[进度核查](../results/progress-review-20261007/audit.json)再次核对近期53项学习/写入、16项回放、16项Loop的既有完成证据；Loop的16份重载审计均passed，两批既有云审计passed，汇总源文件哈希一致。7个相关控制器无活动或排队任务，LM Docker无运行容器。本次只读取已有审计和当前进程状态，没有新增训练、再次模型重载、实时云端查询或容器清理。

复核优先使用原始生成、固定真值和冻结计分定义，其次是带分母的机器可读汇总，再是本页摘要。精确复现采用各批次冻结源码与配置；不能为统一格式改写旧协议、覆盖已有运行或把开发世界重新称作独立确认。新批次按 [实验协议](experimental-protocol.md) 记录。

本机启动Python前仍需在同一条命令中设置 `LD_LIBRARY_PATH="/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"`；这条修复限已验证主机，不自动套用容器。LM实验专用Docker与资源规则见项目工作入口。运行恢复应读取实际控制器、进程和容器状态，本结果页不充当实时GPU调度表。
