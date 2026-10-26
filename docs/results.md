# 研究结果：当前摘要与证据索引

LM Knowledge整合更新：2026-10-25（芝加哥）；Loop各条保留其原快照时间。本页先给LM Knowledge当前比较与核心证据；理论愿景和论文组织见[Roadmap](roadmap.md#paper-vision-20261025)。详细结果见[历史正文](reports/knowledge-history-20261022/results.md)，所有旧批次锚点在页末保留。

<a id="knowledge-method-factorial-1000-v1"></a>
## 八组比较完成：闭卷收益伴随保持损失

Qwen3-1.7B，SFT／Prompt Distillation各配full、原范围LoRA、down-only、down-LoRA；1000案例、9928段、297840条训练问答。八组各5轮、1489200次样本曝光；原batch4对应每组372300步。10-22用户授权从已有进度增大batch，后续实际更新数另报，不再固定原步数。[原协议](development-artifacts/knowledge-method-factorial-1000-v1/protocol.md)、[接续修订](development-artifacts/knowledge-method-batchsize-20261022/amendment.md)。

**2026-10-25 06:55 UTC核验：八组均达到1489200曝光，独立重载、本地终点评测、W&B配置／计数核对及八个评测容器正常退出全部通过。** [统一分析](../results/knowledge-factorial-analysis-20261025/report.md)保留逐题新增答对、原正确丢失、无效判分、按91个国家簇估计的旧题保持区间，以及原训练、接续、废弃重算和共享准备成本。

| 范围 | PD闭卷 | SFT闭卷 | PD旧正确保留 /247 | SFT旧正确保留 /247 |
| --- | ---: | ---: | ---: | ---: |
| full | 46.75% | 39.80% | 200 | 211 |
| 原范围LoRA | 45.55% | 39.84% | 213 | 213 |
| down-only | 40.84% | 35.90% | 217 | 229 |
| down-LoRA | 35.20% | 33.80% | 208 | 214 |

共同闭卷基座25.60%。PD在full／原范围LoRA／down的闭卷增益均高于对应SFT，配对差为6.95／5.71／4.94个百分点；down-LoRA差1.40个百分点，区间跨零。PD-LoRA相对PD-down闭卷高4.70个百分点，原文档在场几乎相同，旧保持差的区间跨零；不能把down称为统一最优范围。八组原文档在场分数均低于基座79.10%，且均有原正确题丢失。案例区间只反映单seed的案例变动，PD／SFT回答配方及接续batch差异不能解释为纯KL／CE消融。

八组记录训练进程时钟合计238.11小时，另有共享数据准备65.58进程小时、已记录废弃重算4100次更新。每组PD约74.60M答案token、493M含教师前向token，SFT约72.16M、145M。训练时钟含共享GPU与存储等待，不能当独占GPU小时、FLOPs或公平速度排名；原范围LoRA的r=1024仍有12.73亿可训练参数。

原六组接续batch为SFT-full／LoRA／down的16／16／16、PD-full／LoRA／down的8／4／8；两组down-LoRA均batch8。失败与重算记录保留。恢复见[四卡交接](../results/queue-recovery-20261023/HANDOFF.txt)、[调度修订](../results/queue-recovery-20261023/knowledge/amendment.json)与[原接续记录](../results/knowledge-method-batchsize-20261022/RUNNING.txt)，必须再核对实时容器与控制器。

主评价使用Qwen3-4B非量化权重并关闭thinking，沿用作者评分消息与true/false解析，报告true／有效判分数及无效覆盖。旧32案例EM/F1与新语义重评分别保留。[云端核验快照](../results/knowledge-factorial-analysis-20261025/cloud-snapshot.json)、[W&B](https://wandb.ai/zsqzz/llm-memory-editability)。

<a id="knowledge-use-coverage-v1"></a>
## 用途覆盖：开发六节点核验完成，正式比较尚未完成

**2026-10-25 18:01（芝加哥）只读整理：全参数PD A／B／C的200192与400384监督token节点均有独立重载通过、本地完整评价和云端核验收据。** 三个400384节点评测容器实际均退出0。**18:07补查：正式集基线已正常退出，PD A／B／C三组正式新增训练均运行，已有正更新步数与有限loss／梯度；SFT及改写尚无终点结果。** 本次逐题复算六节点的直接／分步评分并核对旧题转移，不新增训练、模型前向或云端API核验。[时间与来源哈希快照](../results/knowledge-paper-synthesis-20261025/evidence.json)、[六节点数据](../results/knowledge-paper-synthesis-20261025/development-metrics.csv)。

A为少量原子QA重复，B为多问法原子QA，C为原子加多用途QA。下表为直接回答；“C训练用途的新问法”是三臂共同测试池，A／B并未接受这些用途训练。

| 条件 | 主监督token | 原问法 /64 | 原子留出问法 /128 | C训练用途的新问法 /224 | 双新组合 /32 | 留出关系组合 /32 | 留出用途类别 /32 | 原正确旧题保留 /244 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 基座 | 0 | 0 | 0 | 88 | 0 | 2 | 0 | 244 |
| A | 200192 | 64 | 118 | 7 | 0 | 0 | 0 | 153 |
| B | 200192 | 64 | 127 | 0 | 1 | 0 | 0 | 80 |
| C | 200192 | 64 | 128 | 137 | 1 | 3 | 0 | 134 |
| A | 400384 | 64 | 118 | 7 | 0 | 0 | 0 | 154 |
| B | 400384 | 64 | 127 | 2 | 1 | 0 | 0 | 81 |
| C | 400384 | 64 | 128 | 136 | 1 | 2 | 0 | 130 |

**开发观察。** 扩大问法覆盖使原子留出问法从118/128到127/128；增加用途后，所练用途的新问法达到137/224，基座为88/224。双新直接组合仍≤1/32，留出用途类别仍0/32。不能将137项全当成新增事实带来的净收益，也不能把这列成绩称为留出用途泛化。预算加倍后这些列没有明显改善，这是两个开发节点的观察，不是能力上限。

模型自己生成第一跳答案，再由评测程序代入第二问时，六节点的双新组合均32/32；单次请求“分步回答”只有1–2/32。**外部拆问再输入、单次分步生成和直接回答是三种不同操作。** 前者提供了调用原子问答的可行路径，尚未证明内部推理按同样的过程实现。

**保持损失须进入主分析。** 本批相同自然旧题池的基座为244/317，不能借用八组批次的247分母。按本批冻结评分，首节点A／B／C分别丢失原正确题91／164／110道，同时新增答对20／10／13道；全量正确173／90／147，不能拿它们代替原正确保持。第二节点全量174／90／143。这里尚未另作生成格式或语义复核，不把所有失分直接定位成事实删除。

按原规则选择所有PD臂原子提取达到80%的最早共同节点，现已选择开发200192、正式800768、再次改写200192主监督token；选择只读开发原问法，不根据用途成绩挑节点。[已保存预算选择](../results/knowledge-use-coverage-20261025/runtime/budget-selection.json)。正式128事实对、PD／SFT A/B/C及六组改写仍须完成，开发32事实对的结果不替代正式比较。

自然问题初版9/96结构有效、语义复核未通过；有界提示修订已完成203次生成，结构有效提高到46/96，仍有50项未通过，语义审核状态为pending、未写批准记录。**生成进程完成不等于自然文档应用可执行。** 控制器顶层`waiting_quality_review`与正式PD训练运行并存，不代表全部任务停摆。原270次生成及592.58秒、513674输入token、64753输出token的成本与失败仍保留。[四卡修订](development-artifacts/knowledge-use-coverage-v1/four-card-amendment-20261025.md)、[原全参数修订](development-artifacts/knowledge-use-coverage-v1/full-start-amendment-20261025.md)、[执行入口](../results/knowledge-use-coverage-20261025/RUNNING.txt)；旧交接只记录启动时状态，恢复必须读最新controller与实际Docker。

<a id="knowledge-method-diagnosis-development-v1"></a>
## 已完成开发批：训练学到目标，知识复用待语义复核

32案例、312段、9360条问答；四组各11700步。10月21日训练、独立重载及云端终点核验均完成，13个正式容器退出0、W&B 9/9 finished。见[完成摘要](../results/knowledge-method-diagnosis-20261021/completion-summary.json)、[云端核验](../results/knowledge-method-diagnosis-20261021/cloud-completion-check.json)。

| 方法 | 闭卷EM | 带文档EM | 原正确旧题保留 |
| --- | --- | --- | --- |
| PD-LoRA | 0/32 | 1/32 | 208/247 |
| SFT-LoRA | 0/32 | 1/32 | 213/247 |
| 全参数SFT | 0/32 | 0/32 | 213/247 |
| 仅down SFT | 2/32 | 4/32 | 219/247 |

共同基座闭卷EM为3/32、带文档13/32。四组训练目标NLL均明显下降，但回答变长造成正确完整句被EM判零，也有实质错误；不能用这些EM直接给知识复用排名，更不能断言算法普遍无效。旧正确题保持与全部旧题准确率是不同指标。全部读数、格式核查与配方差异见[完整记录](reports/knowledge-history-20261022/results.md#knowledge-method-diagnosis-development-v1)。

<a id="knowledge-evidence"></a>
<a id="knowledge-paper-synthesis-20261025"></a>
## 对当前主线仍有效的证据

从“知识应可被多种用途重复调用并支持修改”的愿景出发，现有结果回答以下问题。这个分类是2026-10-25的事后综合，不是对原实验假设的追溯改写。各行保留自己的模型、数据、分母和审计状态；不合并成同一实验。

| 问题 | 关键观察 | 证据用途与完整记录 |
| --- | --- | --- |
| 表达怎样影响提取 | bioS规范QA为5.07／35.65／99.96%；多表达加换序后，换名84–91%跟名字；单篇92–94%跟前文 | 提取与地址形成：[bioS](reports/knowledge-history-20261022/results.md#bios)、[地址](reports/knowledge-history-20261022/results.md#bios-write-address-v1) |
| 用途经历能否被原子重复替代 | 三个确认知识图中，有组合经历100%，匹配原子重复25.01%，基础原子3.01% | 使用经历有额外价值；同时改变上下文和监督，不唯一定位内部过程：[组合经历](reports/knowledge-history-20261022/results.md#grok-usage) |
| 搭配覆盖扩大到哪里 | 固定事实角色次数等边际分布，广泛搭配使ID组合15.11→66.30%；独立实体OOD仅1.57→2.16% | 局部组合收益与跨事实迁移分开；三个知识图×两种子：[搭配分布](reports/knowledge-history-20261022/results.md#text-structure) |
| 事实能回答是否意味着任意角色可用 | Ye原方法复现中，加全原子监督后ID／OOD单跳2996/3000、1993/2000，受限第二跳组合仍2/3000；另一个无原子监督臂原子模板0%而Test-II 54.77% | 原子问答成功不保证角色迁移，原子模板失败也不等于内部完全没有事实：[原方法复现](reports/knowledge-history-20261022/results.md#second-hop-paper-reproduction-v1) |
| 原问法与新用途是否相同 | 原问法100%，留出问法37.61–48.62%；标准小模型两条新事实各自答对64/64、组合0/64 | 区分能力，避免重复证明相同失败：[问法](reports/knowledge-history-20261022/results.md#sequential-phrasing-20261011)、[双新事实](reports/knowledge-history-20261022/results.md#ce-cross-sample-double-new-v1) |
| 答案监督写在哪里 | 后段梯度集中于末位置，写入方向偏向输出答案；小模型只施加答案位置写入仍能回答新事实 | 连接行为与计算；梯度份额不等于训练后参数增量分解：[梯度](reports/knowledge-history-20261022/results.md#answer-gradient-address-v1)、[位置](reports/knowledge-history-20261022/results.md#write-position-20261020) |
| 自然模型的层与位置 | 实体／问题依赖随位置和层变化；两Qwen读取阻断效应明显不同 | 份额不直接定位知识／推理／答案：[地图](reports/knowledge-history-20261022/results.md#knowledge-reasoning-map-v1)、[逐层整理](reports/layer-evidence-chain-20261021.md) |
| 上下文与参数写入 | 组合用途在早中段从问句读取上下文；“先桥后读”预定判断四组均不成立 | 保留反证；旧M3等不能代替原方法：[读出比较](reports/knowledge-history-20261022/results.md#write-mode-readout-v1) |
| 自然模型用途练习 | Qwen3.5-2B某集未练事实直接使用.26→.85，另一集.60→.46；自然两跳退化 | 有正收益，也有保持问题；本批未存权重：[用途练习](reports/knowledge-history-20261022/results.md#use-adapter-development-v1) |
| 改写目标与传播 | 小型受控模型三种目标都编辑24/24成功，传播3.13→68.75→85.42%；旧组合96.47→89.84% | 传播和保持一起报；不是Qwen的24例：[写入目标](reports/knowledge-history-20261022/results.md#write-target-editing-v1) |
| 自然模型编辑是否同步改变后果 | GPT-J上ROME／MEMIT原子编辑88.10／95.53%，多跳直接回答7.37／7.63%，CoT21.20／13.13%；编辑成功且必要单跳正确子集仍有大量失败 | 每法3000案例、6015事实编辑；完整生成新进程重放通过。跳位优势与CoT交互不跨方法、数据普遍成立：[编辑及跳位分析](reports/knowledge-history-20261022/results.md#paper-edit-reproductions-v1) |
| 深度与交接 | 未共享＋词表交接OO 6.19%，共享＋词表交接95.66%；该配方单独修一个条件时OO≤6.2% | 受控架构证据；相对普通Loop约2.05倍估算矩阵训练FLOPs需计入：[架构对照](reports/knowledge-history-20261022/results.md#untied-discoloop-development-v1)、[成本](reports/knowledge-history-20261022/results.md#supervised-alignment-development-v1) |
| 新知识持续积累 | 邻居保护下DiscoLoop新事实128/128、双新63/64、旧原子／组合保持99.22／96.99%；普通Loop仅部分改善 | 保留已成功的条件，不概括为全无进展：[邻居保护](reports/knowledge-history-20261022/results.md#neighbor-protection-development-v1) |
| 普通微调怎样迁移及保持 | 先学组合再加新事实有部分迁移；回放旧组合使旧能力改善，新组合仍约21%；旧MoE和共享KV扩展未形成统一解 | 顺序迁移、回放与失败已回答的问题不重新包装；原始差异与停止决定保留：[顺序迁移](reports/knowledge-history-20261022/results.md#sequential-transfer-v1)、[回放](reports/knowledge-history-20261022/results.md#sequential-replay-v1)、[MoE](reports/knowledge-history-20261022/results.md#moe-sequential-v1) |
| 学习更新规则是否已有可用结论 | PERK重建中FT-ICR／PERK的作者Accuracy为98.00／13.51%；后续固定读取器校准失败，四条科学训练未启动 | 原重建未获优势，不推广成原方法无效；不能把未启动的监督比较写成负结果：[PERK](reports/knowledge-history-20261022/results.md#perk-reconstruction-20261011)、[校准停止](reports/knowledge-history-20261022/results.md#perk-writer-calibration-20261012) |

上表不提升原批次的审计状态；部分诊断仍是“云端未独立核验”。九项论文复现与五项编辑跳位分析已经完成，路径一致性仍是工作假说，见[冻结分组结果](reports/knowledge-history-20261022/results.md#path-hop-analysis-v1)。

**综合判断。** 已有证据覆盖了提取地址的形成、使用经历、架构的再次访问与交接、事实改写的传播以及持续写入的保持，且包含可复用的正方法结果。中心解释可以开始成文；当前尚缺的是同一自然模型中“用途覆盖→未练用途→原子改写后后果变化→旧能力保持”的正式综合证据。八组已证实实际闭卷收益，当前开发批已区分表达覆盖与所练用途的新问法；二者都不能替代尚未完成的改写与跨用途结果。论文组织以[理论愿景与关键比较](roadmap.md#evidence-chain)为准，不因已有结果有利或不利而删除原问题。

## Loop与历史材料

Loop先读[当前摘要](loop-roadmap.md)与[按问题查重](loop-history.md)。下列Loop条目保留结论和旧锚点，完整表格、失败及调度原文收在[证据档案](reports/loop-evidence-history-20261022.md)；旧启动记录不代表实时状态。

<a id="loop-abc-method-check-150-v1"></a>
<a id="loop-abc-continuation-1500k-v1"></a>
**ABC各150步已执行，B梯度核验判断修订。** 三组固定R8、首轮e，复用原多跳数据与完整初值，共450步；保存的独立重载和云端数值核验通过。raw/R8的B四跳答案NLL从10.707变为A7.246、B13.393、C7.283，准确率均0%。后续BF16检查发现B前4轮no_grad污染权重缓存，10个线性参数没有梯度，故撤回原“截断梯度路径符合设计”的判断，原始证据保留。三组已从真实150步状态接续，目标各150万步，第151步起统一禁用缓存；完整状态与原曲线接续，修订边界单列。见[原结果及更正](../results/loop-abc-method-check-150-v1/report.md)与[接续协议](../results/loop-abc-continuation-1500k-v1/protocol.md)。

<a id="loop-fourway-terminal-20261024"></a>
**2026-10-24四臂统一配方正式比较已启动，尚无终点结论。** 最新invariant、普通Loop、DiscoLoop、InfiLoop分别使用GPU6/4/7/8，各150万步，共600万步。共同使用一个共享Transformer block、batch512、逐样本Poisson(8)越界重抽到1–16、终点有效token平均损失与全程反传；四组初始主干哈希相同。InfiLoop按用户确认采用单状态统一主干，保留作者代码的固定随机初态、原始候选与N/D合并；不是数独H/L完整架构复现。DiscoLoop按原文符号任务公式实现。10项必要CPU检查通过，四组均已有正式更新、有限损失及梯度和W&B记录；启动时桥接网络失败发生在第0步，已保留失败容器与日志，使用host网络解决，未改变科学配方。七条旧初始化训练继续暂缓。见[协议](../results/loop-fourway-terminal-20261024/protocol.md)、[运行交接与四组链接](../results/loop-fourway-terminal-20261024/RUNNING.txt)。

<a id="loop-feedback-controls-20261024"></a>
**2026-10-24两组软反馈对照已共享启动，尚无终点结论。** InfiLoop+feedback与invariant-no-feedback分别共享现有四臂的GPU4/8，各150万步，共新增300万步授权；原四组PID与启动时间未变、步数持续增加。前者只将原始聚合候选改为H_t+q_t，后者只把RMSNorm(H_t+q_t)改为RMSNorm(H_t)，保留概率打分、γ和历史合并。配方、数据、主干初值与各自基线配对。两组各6次临时CUDA更新及EMA/R32检查通过，峰值预留显存4.875/5.330 GiB；临时权重不保存、不用于正式训练。两组正式更新的损失与梯度有限，W&B在线配置一致；七条旧初始化仍暂缓。见[协议](../results/loop-feedback-controls-20261024/protocol.md)、[启动核验](../results/loop-feedback-controls-20261024/runtime/startup-verification.json)与[运行链接](../results/loop-feedback-controls-20261024/RUNNING.txt)。

**10-24 22:13（America/Chicago）六组同步数快照：** 六组均运行，完整共同评测节点为16,000步，统一参数EMA/R8、每个B池5,000题。在该节点，普通Loop／DiscoLoop／InfiLoop／InfiLoop+feedback／invariant／invariant-no-feedback的B两／三／四跳分别为0.22/0.12/0.02、0.08/0.04/0.02、0.02/0.02/0、0/0/0、0.04/0.04/0.08、0.08/0.02/0.06（%）。两项InfiLoop的B原子准确率仅0.56/1.20%，其余四组100%；前提未掌握时不把组合低分直接归因于软反馈。后续同为128,000步的DiscoLoop／普通Loop／InfiLoop，B两跳为9.86/0.12/0%，DiscoLoop三／四跳仍为0.06/0.12%；invariant和新增两组尚无该节点。40份同节点或最新节点逐题预测复算与汇总一致，共同16,000步的曝光和深度采样完全一致；新增训练和模型前向均0。见[完整快照](../results/loop-feedback-controls-20261024/comparison-sixway-20261024-221306/snapshot.json)、[同步数表](../results/loop-feedback-controls-20261024/comparison-sixway-20261024-221306/matched-step.csv)与[所有已保存EMA/R8节点](../results/loop-feedback-controls-20261024/comparison-sixway-20261024-221306/all-ema-r8-nodes.csv)。

**10-24 22:23六组学习曲线：** 同步数训练loss与B原子EMA/R8已绘制，另保留全程图及RAW/EMA、梯度、β辅助图。16k两项InfiLoop的loss为2.956/2.874、B原子0.56/1.20%；8k答案NLL约ln(500)，EOS已100%，说明迟启动首先发生在事实学习。原InfiLoop32k／64k的B原子为66.56／100%；加反馈24k的loss为2.185，原组同节点1.128，尚未消除迟启动。两组初始固定h0 RMS为1.025，B提示embedding RMS为0.0283，相差36.21倍；这是统一GPT适配中值得核查的尺度条件，不能据此认定因果。35份B原子预测复算一致，本次新增训练与前向均0。见[曲线与机制分析](../results/loop-feedback-controls-20261024/learning-curves-20261024-222400/report.md)。

<a id="loop-terminal-cuda-20261024"></a>
**2026-10-24新终点监督配方GPU检查：40次临时更新完成，正式训练0。** batch 512、一个共享block、逐样本Poisson(8)重抽到1–16、全程反传，实测0.921秒／步、PyTorch峰值已分配显存0.666 GiB；全512条四跳题跑16轮为0.100秒／步、4.956 GiB。随机深度组依次计算带来明显执行开销。损失、参数、梯度与Adam状态均有限，终点梯度到达首轮h/N/D/p；裁剪前总范数峰值分别1093／5714，裁剪后约1，不能将短程通过等同于完整学习率下稳定。检查容器正常退出、GPU5已释放，未保存临时权重，七条旧训练仍暂缓。见[测量、边界与原始记录](../results/loop-terminal-cuda-20261024/report.md)。

<a id="loop-terminal-masked-cuda-20261024"></a>
**同日整批掩码改版与复测：新增40次临时更新、正式训练0。** 每条样本仍独立抽深度，到期冻结h/N/D/p并保留计算图，单次终点损失完整反传。相同GPU5、batch512与深度上限16下，Poisson耗时0.921→0.104秒／步（8.83倍），峰值已分配显存0.666→5.248 GiB；全16轮0.100→0.102秒／步，全部15步损失与梯度范数和旧版一致。随机深度改变dropout分配，梯度峰值1093→660不能解释为稳定性提升；全16轮仍为5714。127项相关CPU检查通过；扩展检查两项旧冻结哈希不匹配在前一版本已存在，守卫未改。GPU已释放，未保存临时权重。见[代码语义、对比和限制](../results/loop-terminal-masked-cuda-20261024/report.md)。

<a id="loop-results-20261024"></a>
**2026-10-24完成结果收集：七条未启动初始化比较由用户明确暂缓，共1050万步，恢复须新的授权。** 已收集16条训练终点与完整曲线，528份终点预测复算一致，新增训练与模型前向0。新初始化1 invariant已到150万步，B两／三／四跳92.58／65.28／50.06%，原子与A留出组合100%；与原初始化四跳79.42%相比低29.36pp，不能宣称稳定在原水平。其源机独立重载／深度／云端和本次Docker退出核验仍缺，新置换ema同样分列缺口。

state-MLP150万步四跳60.90%，已从60万步2.94%的低谷恢复，重载与云端核验通过。关闭dropout的50万步B组合均0%、A留出四跳0.16%、训练组合及原子100%，重载与云端通过；原state-MLP同50万四跳5.66%，不自动延长预算。新置换invariant／ema四跳26.96／7.10%，多对一89.74／0.22%；多对一基础DiscoLoop29.96%、普通Loop0%，两条训练与本地评测通过但W&B配置核验失败。完整曲线保留中期高点与后期回落，不以最好节点代替终点。见[统一分析与全部终点](../results/loop-results-20261024/report.md)、[四跳曲线PDF](../results/loop-results-20261024/four-hop-curves.pdf)、[状态与核验缺口](../results/loop-results-20261024/evidence.json)。

<a id="loop-accuracy-20261023"></a>
2026-10-23 00:15 UTC只读快照：state-MLP训练约78.1万步，最新70万步B两／三／四跳77.86／28.46／4.04%，原子与A留出四跳100%。B四跳25.6万71.24%→50万5.66%→60万2.94%→70万4.04%，仍在低位、最近略回升，尚未到150万终点。同步数比较：新置换世界130万invariant／常数73.52／9.34%；多对一50万53.20／72.52%，方向不同，均未到终点。关闭dropout仍等待资源，科学更新0。14组保存记录、68份来源哈希及1419绘图点核对通过；源机两组为最新保存记录，本次未作远程容器核验，模型评测与新增训练0。[结果表与解释](../results/loop-queue-target-20261021/accuracy-20261023-0015/report.md)、[四跳图](../results/loop-queue-target-20261021/accuracy-20261023-0015/four-hop-focus.png)、[六页完整PDF](../results/loop-queue-target-20261021/accuracy-20261023-0015/loop-accuracy-comparison.pdf)、[同步数数据](../results/loop-queue-target-20261021/accuracy-20261023-0015/paired-condition-common-step.csv)。

<a id="loop-accuracy-20261022"></a>
2026-10-22 15:30 CDT曲线快照：参数EMA、R8、各池5000题。state-mlp的B四跳从30万步56.38%降至40万步23.88%，同在40万步invariant为29.04%；原世界invariant的150万步终点79.42%。新置换／多对一invariant在110万步分别47.84／70.88%，均未训练完成。14组现有曲线保留回落和缺失组，不混合不同配方或以不同步数端点作配对结论。[原世界曲线](../results/loop-queue-target-20261021/accuracy-20261022-2030/original-world.png)、[换数据与初始化](../results/loop-queue-target-20261021/accuracy-20261022-2030/robustness.png)、[六页PDF](../results/loop-queue-target-20261021/accuracy-20261022-2030/loop-accuracy-comparison.pdf)、[最新读数](../results/loop-queue-target-20261021/accuracy-20261022-2030/latest-accuracy.csv)、[同条件同步数比较](../results/loop-queue-target-20261021/accuracy-20261022-2030/paired-condition-common-step.csv)。[前次记录](reports/loop-evidence-history-20261022.md#loop-accuracy-20261022)。

<a id="loop-state-regression-diagnosis-20261022"></a>
2026-10-22状态MLP门的已有检查点诊断：69组、34.5万题次、零训练。25.6万→50万步第二轮第二跳候选85.12→21.16%，四跳R8为71.24→5.66%；晚期前8轮曾答对与R8答对均283/5000，主要为形成失败。固定晚期权重，原子问答的attention后状态供体使候选99.98%、四跳27.96%；错误实体供体为0／1.90%。仅回退B输入范数未恢复（5.46%），早期交接整体移植损伤原子/A。同一供体也使早期四跳升至88.98%，干预后早晚仍差61.02pp；接口是实质瓶颈，尚未解释大部分训练退化、确定优化根因或得到通用LLM修复。供体含真实中间实体，仅作诊断。[三页报告](../results/loop-state-regression-diagnosis-20261022/checkpoint-regression-report.pdf)、[逐题配对](../results/loop-state-regression-diagnosis-20261022/paired-stage-analysis.json)、[核验](../results/loop-state-regression-diagnosis-20261022/validation.json)。

<a id="loop-state-handoff-development-v1"></a>
state-mlp唯一150万步训练、独立重载与云端核验均完成；B两／三／四跳99.56／90.64／60.90%。门直接读取旧状态和候选，输入与容量同时改变；完整曲线与对照见[10-24汇总](#loop-results-20261024)。 [完整记录](reports/loop-evidence-history-20261022.md#loop-state-handoff-development-v1)。

<a id="loop-queue-expansion-20261022"></a>
10-22调度记录：普通Loop的150万步训练、重载、深度与云端核验完成，多对一ema接续开训。原队列已迁移并扩为最多四卡；恢复读最新修订与控制器，禁止重启旧入口。 [完整记录](reports/loop-evidence-history-20261022.md#loop-queue-expansion-20261022)。

以下是已被10-24暂缓决定覆盖的历史：10-23用户明确授权共享GPU接续余下九条训练，每条仍150万步，初始化比较最后。目标机GPU4/7的多对一普通Loop／基础DiscoLoop通过原32步短训和独立重载后已正式开训；GPU5/8接续已完成模型的缺失评测。四卡CUDA与W&B访问检查、23项CPU检查和Ruff通过；原配方和预算保持。此前调度快照的并发写入失败仅发生在工程阶段，日志与已通过检查保留；不算科学训练失败。状态见[最新交接](../results/loop-queue-target-20261021/RUNNING.txt)与[共享接续修订](../results/loop-queue-target-20261021/runtime/shared-20261023-v3/amendment.json)。

<a id="invariant-attention-temperature-20261022"></a>
固定权重统一降低attention温度有有限收益：70万步B四跳R8由6.86%到10.46%，130万步由93.74%到97.62%；没有消除训练低谷。 [完整记录](reports/loop-evidence-history-20261022.md#invariant-attention-temperature-20261022)。

<a id="invariant-second-hop-20261022"></a>
第二跳局部读取修复使70万步候选正确率74.40%→98.94%，完整四跳仅6.86%→13.86%；读取与后续保持均需解释。 [完整记录](reports/loop-evidence-history-20261022.md#invariant-second-hop-20261022)。

<a id="invariant-checkpoint-dip-20261022"></a>
invariant在25.6万／70万／130万步的B四跳为66.10／6.86／93.74%；单换打分、衰减或阻止特定覆盖未救回低谷，不能归结为一个门参数。 [完整记录](reports/loop-evidence-history-20261022.md#invariant-checkpoint-dip-20261022)。

<a id="loop-invariant-robustness-v1"></a>
旧invariant全池B四跳R8／R32／R256为79.42／81.98／74.18%；固定256题R8→R16384为78.91→74.61%。短范围收益与长循环退化并存；已完成的初始化与数据终点见[10-24汇总](#loop-results-20261024)，剩余七条初始化比较暂缓。 [完整记录](reports/loop-evidence-history-20261022.md#loop-invariant-robustness-v1)。

<a id="loop-long-hop-review-20261025"></a>
**10-24（America/Chicago）旧终点5–8跳复核：** 原世界native四臂150万步、参数EMA、R16，每个世界／跳数5000题。invariant的A五／六／七／八跳为19.24／0.18／0.08／0.10%，B为0.66／0.18／0.16／0.18%；常数ema、content、同配方DiscoLoop的B五跳均0%。invariant的A四跳仅前补一个pad即由100%降至15.60%，位置／陌生前缀影响仍在，不能把长跳失败直接解释为循环算法上限。复用五模型（含异配方旧DiscoLoop）的240份预测，逐题分数和权重哈希一致，新增训练／模型前向0；原R4参考差异invariant最大0.12pp、ema最大0.18pp，超过原0.1pp门槛的状态保留。见[完整对照、逐轮结果与核验](../results/loop-long-hop-review-20261025/report.md)；新初始化、换世界、普通Loop与state-MLP不在这批长跳预测中。

<a id="loop-position-controlled-long-hops-20261025"></a>
**同日固定旧权重的位置干预完成：** 四臂×原始／去位置／截到P4／线性压缩到P0–P4，224池、112万题次、零训练。无padding，仅位置编号+1，invariant的A/B四跳从100/82.18%降至8.46/0.28%，确认显式位置输入本身足以破坏组合。去位置后A/B原子仍100/99.58%，A/B四跳仅0.20/0.10%；B四跳98.54%的题所需原子均能单独答对，其中组合也仅0.1015%。插值使content的A五跳0→44.60%，invariant为19.24→20.20%，但两者B五跳仅0.10/0.12%，各干预未恢复B五至八跳。支持组合执行依赖原布局，不能声称位置影响已被排除或四跳是算法上限。四组参数不变、基准逐轮预测与历史完全一致、全部224池复算通过；独立评测容器退出0，原六组训练进程未重启。见[完整结果与解释](../results/loop-position-controlled-long-hops-20261025/report.md)。

<a id="invariant-five-hop-diagnosis-20261025"></a>
**旧invariant五跳定位及DiscoLoop长跳补测完成：** B五跳前16轮始终未答对99.14%，曾正确后丢失仅0.20%；五跳题自身第四前缀R16可读82.76%，第五条事实单独调用100%。R5仅强制第五位置读原生第四位置V，A最终19.24→94.98%，B仅0.66→8.90%；同形式改读原头为0.14/0.08%。完整原子调用供体使B当轮99.80%、R16为61.64%，显示形成后仍有丢失；该供体可能已含第五答案，不能据此单独归因于attention或MLP。已有RoPE DiscoLoop/参数EMA的A/B四跳99.92/44.92%，五跳0.22/0.12%，B六至八跳0.16/0.16/0.06%；绝对位置的同配方及旧固定R4版本也未恢复长跳。支持多个旧模型的长度泛化失败，本批没有换训练世界，不能直接称“data sensitivity”。100池零训练完成，invariant原生预测及RoPE原形状重放均逐token复现；原六个训练进程保持。见[逐阶段证据、全时刻干预与DiscoLoop完整对照](../results/invariant-five-hop-diagnosis-20261025/report.md)。

<a id="disco-five-hop-diagnosis-20261025"></a>
**旧DiscoLoop同方法五跳诊断完成：** 四版本原生轨迹、RoPE/参数EMA及旧固定R4/raw的完整六条件干预，共104池零训练。RoPE的B第五事实独立首轮/R16均100%，五跳99.88%在前16轮从未正确、0%曾正确后丢失；第四桥R4 raw52.36%→carry46.92%，272题失去可读性，但仍可读的2346题下一轮第五答案正确数为0。R5强制读原生V4的A/B最终仅0.28/0.10%，不同于invariant的94.98/8.90%；R4给规范真桥则56.44/31.98%。完整原子结果在R5接入后，RoPE B从当轮100%保持到R16的99.98%，而invariant为99.80→61.64%；错误供体的错误后继也保持99.96%。支持共同的答案形成失败及不同的时机／保持反应，不能归成同一模块根因。旧固定R4绝对位置版第四桥R16已仅A0.46%、B0.06%，上游也失败。原生逐token复现、权重不变、原六组训练保持，见[全量轨迹、干预和机制边界](../results/disco-five-hop-diagnosis-20261025/report.md)。

<a id="loop-receiver-attention-20261025"></a>
**旧RoPE第五轮接收状态拆分完成：** 两阶段280池、零训练。固定正确原子attention输出，RoPE B当轮raw在原接收残差下从R4的31.98%降至R5的0.10%，缩放或清零残差后两轮均100%；但保留原生attention、只缩放残差时两轮仍0%。支持可逆的旧残差干扰及原生读取不足，不能称不可逆锁死或证明第五关系彻底丢失。原子attention自身含有效关系信息，错关系交叉对照确认不能把其救援归为旧状态保留关系。invariant给规范真桥、只缩放自身输入，B当轮恢复81.20%；固定前缀遍历10种末关系后仍78.66%，443/500前缀可答至少两个不同正确后继，证明进入块的h+e仍有可用关系信息，不证明h单独保持。校准原子计算后RoPE B当轮100%→R16 99.98%，invariant 99.80%→61.64%。第一阶段全头单源桥V在RoPE的fresh正控不成立，负结果不作关系丢失证据；原始失败、修订与全量结果保留。两次成功推理均退出0，280池逐题复算、旧权重及原六组训练PID保持。见[接收残差、错关系对照与校准报告](../results/loop-receiver-attention-20261025/report.md)。

<a id="loop-baseline-robustness-v1"></a>
基础DiscoLoop旧配方raw/R8的B三跳88.26%，当前配方EMA/R8仅0.02%，须同时报告。当前普通Loop已完成，B两／三／四跳为1.10／0／0%；新置换和多对一基线也已完成训练与本地评测，新增初始化基线暂缓，见[10-24汇总](#loop-results-20261024)。 [完整记录](reports/loop-evidence-history-20261022.md#loop-baseline-robustness-v1)。

<a id="loop-depth-baseline-comparison-20261023"></a>
2026-10-23复用原世界终点预测，绘制[深度对比图（两页PDF）](../results/loop-depth-baseline-comparison-20261023/loop-depth-comparison.pdf)：当前配方的invariant、普通Loop、基础DiscoLoop和content共同覆盖R1–32；训练两段累计最多10轮，R>10为深度外推。另以虚线保留旧固定R4/raw的较强DiscoLoop（仅到R16）。content采用InfiLoop内容打分与递推聚合，但保留DiscoLoop候选和本项目训练配方，图中标为“InfiLoop式交接适配版”，不能称原版InfiLoop整体复现。当前配方B两跳R8→R32：invariant 99.32→99.18%、DiscoLoop 40.36→40.22%、content 15.28→3.08%、普通Loop 1.10→1.44%；后三者三／四跳接近零，不能把低准确率持平解释为有效保持。R1–256附图中DiscoLoop/content在R32终止，缺失区间不延长；普通Loop B原子R8→R256为100→1.14%，invariant为100→91.22%，常数交接ema保持100%。[全量曲线与R8正确答案保持统计](../results/loop-depth-baseline-comparison-20261023/curves.csv)、[复算核验和来源哈希](../results/loop-depth-baseline-comparison-20261023/validation.json)。当前五臂11池逐题复算、题序／训练配置／初始骨干核对及三臂R1–32长短评测逐token一致，共115项检查通过；旧配方直接复用既有审计汇总。新增训练与模型前向均为0，既有队列及预算不变。

Loop的当前设计、运行批次和下一步以[Loop路线图](loop-roadmap.md)为准；10月22日已有结果保留在下面的原锚点入口。完整历史不因当前主线变化删除，也不从历史报告的“运行中”恢复旧入口。

<a id="history-index"></a>
### 旧批次与旧标题索引

此索引只负责找证据，不是待办清单。详细原文保留原先的数值、失败、停止原因与限定；后续新结果写在本页相应主题，不改写历史快照。

<details>
<summary>历史链接兼容索引（按需展开）</summary>

| 旧入口 | 记录 |
| --- | --- |
| <a id="layer-evidence-chain-20261021"></a>layer-evidence-chain-20261021 | [原文](reports/knowledge-history-20261022/results.md#layer-evidence-chain-20261021) |
| <a id="write-mode-readout-v1"></a>write-mode-readout-v1 | [原文](reports/knowledge-history-20261022/results.md#write-mode-readout-v1) |
| <a id="knowledge-reasoning-map-v1"></a>knowledge-reasoning-map-v1 | [原文](reports/knowledge-history-20261022/results.md#knowledge-reasoning-map-v1) |
| <a id="bios-write-address-v1"></a>bios-write-address-v1 | [原文](reports/knowledge-history-20261022/results.md#bios-write-address-v1) |
| <a id="loop-beyond-four-hops-20261020"></a>loop-beyond-four-hops-20261020 | [原文](reports/knowledge-history-20261022/results.md#loop-beyond-four-hops-20261020) |
| <a id="e3-learning-rate-control-v1"></a>e3-learning-rate-control-v1 | [原文](reports/knowledge-history-20261022/results.md#e3-learning-rate-control-v1) |
| <a id="write-position-20261020"></a>write-position-20261020 | [原文](reports/knowledge-history-20261022/results.md#write-position-20261020) |
| <a id="neighbor-protection-development-v1"></a>neighbor-protection-development-v1 | [原文](reports/knowledge-history-20261022/results.md#neighbor-protection-development-v1) |
| <a id="sequential-write-interference-20261020"></a>sequential-write-interference-20261020 | [原文](reports/knowledge-history-20261022/results.md#sequential-write-interference-20261020) |
| <a id="untied-discoloop-development-v1"></a>untied-discoloop-development-v1 | [原文](reports/knowledge-history-20261022/results.md#untied-discoloop-development-v1) |
| <a id="use-adapter-development-v1"></a>use-adapter-development-v1 | [原文](reports/knowledge-history-20261022/results.md#use-adapter-development-v1) |
| <a id="answer-gradient-address-v1"></a>answer-gradient-address-v1 | [原文](reports/knowledge-history-20261022/results.md#answer-gradient-address-v1) |
| <a id="additive-carry-freezing-20261020"></a>additive-carry-freezing-20261020 | [原文](reports/knowledge-history-20261022/results.md#additive-carry-freezing-20261020) |
| <a id="knowledge-read-interface-v2"></a>knowledge-read-interface-v2 | [原文](reports/knowledge-history-20261022/results.md#knowledge-read-interface-v2) |
| <a id="knowledge-read-interface-v2-reanalysis"></a>knowledge-read-interface-v2-reanalysis | [原文](reports/knowledge-history-20261022/results.md#knowledge-read-interface-v2-reanalysis) |
| <a id="knowledge-read-interface-followup-v1"></a>knowledge-read-interface-followup-v1 | [原文](reports/knowledge-history-20261022/results.md#knowledge-read-interface-followup-v1) |
| <a id="knowledge-read-interface-development-v1"></a>knowledge-read-interface-development-v1 | [原文](reports/knowledge-history-20261022/results.md#knowledge-read-interface-development-v1) |
| <a id="rule-gate-round-evaluation-v1"></a>rule-gate-round-evaluation-v1 | [原文](reports/knowledge-history-20261022/results.md#rule-gate-round-evaluation-v1) |
| <a id="rule-gate-training-v1"></a>rule-gate-training-v1 | [原文](reports/knowledge-history-20261022/results.md#rule-gate-training-v1) |
| <a id="loop-readiness-gate-test-20261019"></a>loop-readiness-gate-test-20261019 | [原文](reports/knowledge-history-20261022/results.md#loop-readiness-gate-test-20261019) |
| <a id="loop-decode-stability-gate-test-20261019"></a>loop-decode-stability-gate-test-20261019 | [原文](reports/knowledge-history-20261022/results.md#loop-decode-stability-gate-test-20261019) |
| <a id="loop-gate-timetable-intervention-20261019"></a>loop-gate-timetable-intervention-20261019 | [原文](reports/knowledge-history-20261022/results.md#loop-gate-timetable-intervention-20261019) |
| <a id="loop-round-extrapolation-diagnostic-20261019"></a>loop-round-extrapolation-diagnostic-20261019 | [原文](reports/knowledge-history-20261022/results.md#loop-round-extrapolation-diagnostic-20261019) |
| <a id="sequence-dependency-gate-v1"></a>sequence-dependency-gate-v1 | [原文](reports/knowledge-history-20261022/results.md#sequence-dependency-gate-v1) |
| <a id="residual-gate-injection-ablation-v1"></a>residual-gate-injection-ablation-v1 | [原文](reports/knowledge-history-20261022/results.md#residual-gate-injection-ablation-v1) |
| <a id="ungated-baseline-stability-20261018"></a>ungated-baseline-stability-20261018 | [原文](reports/knowledge-history-20261022/results.md#ungated-baseline-stability-20261018) |
| <a id="p-gate-root-cause-research-20261018"></a>p-gate-root-cause-research-20261018 | [原文](reports/knowledge-history-20261022/results.md#p-gate-root-cause-research-20261018) |
| <a id="p-gate-instability-diagnostic-20261018"></a>p-gate-instability-diagnostic-20261018 | [原文](reports/knowledge-history-20261022/results.md#p-gate-instability-diagnostic-20261018) |
| <a id="gate-complete-figures-20261018"></a>gate-complete-figures-20261018 | [原文](reports/knowledge-history-20261022/results.md#gate-complete-figures-20261018) |
| <a id="candidate-update-gate-gpu1-v3"></a>candidate-update-gate-gpu1-v3 | [原文](reports/knowledge-history-20261022/results.md#candidate-update-gate-gpu1-v3) |
| <a id="compact-wang-joint-gate-v1"></a>compact-wang-joint-gate-v1 | [原文](reports/knowledge-history-20261022/results.md#compact-wang-joint-gate-v1) |
| <a id="candidate-update-gate-multihop-v1"></a>candidate-update-gate-multihop-v1 | [原文](reports/knowledge-history-20261022/results.md#candidate-update-gate-multihop-v1) |
| <a id="joint-update-gate-wang-v1"></a>joint-update-gate-wang-v1 | [原文](reports/knowledge-history-20261022/results.md#joint-update-gate-wang-v1) |
| <a id="joint-update-gate-multihop-v1"></a>joint-update-gate-multihop-v1 | [原文](reports/knowledge-history-20261022/results.md#joint-update-gate-multihop-v1) |
| <a id="gate-generalization-extension-v1"></a>gate-generalization-extension-v1 | [原文](reports/knowledge-history-20261022/results.md#gate-generalization-extension-v1) |
| <a id="paper-update-gate-continuation-v1"></a>paper-update-gate-continuation-v1 | [原文](reports/knowledge-history-20261022/results.md#paper-update-gate-continuation-v1) |
| <a id="paper-update-gate-multihop-u95-v1"></a>paper-update-gate-multihop-u95-v1 | [原文](reports/knowledge-history-20261022/results.md#paper-update-gate-multihop-u95-v1) |
| <a id="paper-update-gate-multihop-v1"></a>paper-update-gate-multihop-v1 | [原文](reports/knowledge-history-20261022/results.md#paper-update-gate-multihop-v1) |
| <a id="gated-loop-depth-extrapolation-v1"></a>gated-loop-depth-extrapolation-v1 | [原文](reports/knowledge-history-20261022/results.md#gated-loop-depth-extrapolation-v1) |
| <a id="norm-handoff-multihop-v1"></a>norm-handoff-multihop-v1 | [原文](reports/knowledge-history-20261022/results.md#norm-handoff-multihop-v1) |
| <a id="dual-gate-multihop-v1"></a>dual-gate-multihop-v1 | [原文](reports/knowledge-history-20261022/results.md#dual-gate-multihop-v1) |
| <a id="dual-gate-variance-diagnostic-v1"></a>dual-gate-variance-diagnostic-v1 | [原文](reports/knowledge-history-20261022/results.md#dual-gate-variance-diagnostic-v1) |
| <a id="loop-discoloop-lr-control-v1"></a>loop-discoloop-lr-control-v1 | [原文](reports/knowledge-history-20261022/results.md#loop-discoloop-lr-control-v1) |
| <a id="ce-cross-sample-ablation-v1"></a>ce-cross-sample-ablation-v1 | [原文](reports/knowledge-history-20261022/results.md#ce-cross-sample-ablation-v1) |
| <a id="ce-cross-sample-double-new-v1"></a>ce-cross-sample-double-new-v1 | [原文](reports/knowledge-history-20261022/results.md#ce-cross-sample-double-new-v1) |
| <a id="loop-hop-depth-support-v1"></a>loop-hop-depth-support-v1 | [原文](reports/knowledge-history-20261022/results.md#loop-hop-depth-support-v1) |
| <a id="loop-hop-unified-v1"></a>loop-hop-unified-v1 | [原文](reports/knowledge-history-20261022/results.md#loop-hop-unified-v1) |
| <a id="ce-architecture-novel-v1"></a>ce-architecture-novel-v1 | [原文](reports/knowledge-history-20261022/results.md#ce-architecture-novel-v1) |
| <a id="ce-scope-discoloop-novel-v1"></a>ce-scope-discoloop-novel-v1 | [原文](reports/knowledge-history-20261022/results.md#ce-scope-discoloop-novel-v1) |
| <a id="loop-query-adapter-development-v1"></a>loop-query-adapter-development-v1 | [原文](reports/knowledge-history-20261022/results.md#loop-query-adapter-development-v1) |
| <a id="loop-query-control-development-v1"></a>loop-query-control-development-v1 | [原文](reports/knowledge-history-20261022/results.md#loop-query-control-development-v1) |
| <a id="alphaedit-discoloop-novel-v1"></a>alphaedit-discoloop-novel-v1 | [原文](reports/knowledge-history-20261022/results.md#alphaedit-discoloop-novel-v1) |
| <a id="ttcd-foundation-novel-fact-v1"></a>ttcd-foundation-novel-fact-v1 | [原文](reports/knowledge-history-20261022/results.md#ttcd-foundation-novel-fact-v1) |
| <a id="ttcd-novel-fact-transfer-v1"></a>ttcd-novel-fact-transfer-v1 | [原文](reports/knowledge-history-20261022/results.md#ttcd-novel-fact-transfer-v1) |
| <a id="discoloop-training-stage-diagnostic-v1"></a>discoloop-training-stage-diagnostic-v1 | [原文](reports/knowledge-history-20261022/results.md#discoloop-training-stage-diagnostic-v1) |
| <a id="discoloop-edit-round-diagnostic-v1"></a>discoloop-edit-round-diagnostic-v1 | [原文](reports/knowledge-history-20261022/results.md#discoloop-edit-round-diagnostic-v1) |
| <a id="ttcd-one-step-training-v1"></a>ttcd-one-step-training-v1 | [原文](reports/knowledge-history-20261022/results.md#ttcd-one-step-training-v1) |
| <a id="ttcd-merge-hops-v1"></a>ttcd-merge-hops-v1 | [原文](reports/knowledge-history-20261022/results.md#ttcd-merge-hops-v1) |
| <a id="ttcd-one-step-second-hop-v1"></a>ttcd-one-step-second-hop-v1 | [原文](reports/knowledge-history-20261022/results.md#ttcd-one-step-second-hop-v1) |
| <a id="ttcd-speedup-v1"></a>ttcd-speedup-v1 | [原文](reports/knowledge-history-20261022/results.md#ttcd-speedup-v1) |
| <a id="ttcd-architecture-comparison-v1"></a>ttcd-architecture-comparison-v1 | [原文](reports/knowledge-history-20261022/results.md#ttcd-architecture-comparison-v1) |
| <a id="ttcd-double-new-eval-v1"></a>ttcd-double-new-eval-v1 | [原文](reports/knowledge-history-20261022/results.md#ttcd-double-new-eval-v1) |
| <a id="discoloop-success-restore-v1"></a>discoloop-success-restore-v1 | [原文](reports/knowledge-history-20261022/results.md#discoloop-success-restore-v1) |
| <a id="ttcd-cross-sample-v1"></a>ttcd-cross-sample-v1 | [原文](reports/knowledge-history-20261022/results.md#ttcd-cross-sample-v1) |
| <a id="ttcd-second-hop-reader-control-v1"></a>ttcd-second-hop-reader-control-v1 | [原文](reports/knowledge-history-20261022/results.md#ttcd-second-hop-reader-control-v1) |
| <a id="ttcd-second-hop-scratch-v1"></a>ttcd-second-hop-scratch-v1 | [原文](reports/knowledge-history-20261022/results.md#ttcd-second-hop-scratch-v1) |
| <a id="supervised-alignment-development-v1"></a>supervised-alignment-development-v1 | [原文](reports/knowledge-history-20261022/results.md#supervised-alignment-development-v1) |
| <a id="self-decode-alignment-development-v1"></a>self-decode-alignment-development-v1 | [原文](reports/knowledge-history-20261022/results.md#self-decode-alignment-development-v1) |
| <a id="z1-path-overlap-20261011"></a>z1-path-overlap-20261011 | [原文](reports/knowledge-history-20261022/results.md#z1-path-overlap-20261011) |
| <a id="z1-position-followup-20261011"></a>z1-position-followup-20261011 | [原文](reports/knowledge-history-20261022/results.md#z1-position-followup-20261011) |
| <a id="edit-scope-development-20261011"></a>edit-scope-development-20261011 | [原文](reports/knowledge-history-20261022/results.md#edit-scope-development-20261011) |
| <a id="edit-restoration-20261011"></a>edit-restoration-20261011 | [原文](reports/knowledge-history-20261022/results.md#edit-restoration-20261011) |
| <a id="recovery-readouts-20261011"></a>recovery-readouts-20261011 | [原文](reports/knowledge-history-20261022/results.md#recovery-readouts-20261011) |
| <a id="recall"></a>recall | [原文](reports/knowledge-history-20261022/results.md#recall) |
| <a id="sequential-phrasing-20261011"></a>sequential-phrasing-20261011 | [原文](reports/knowledge-history-20261022/results.md#sequential-phrasing-20261011) |
| <a id="bios"></a>bios | [原文](reports/knowledge-history-20261022/results.md#bios) |
| <a id="bios-original"></a>bios-original | [原文](reports/knowledge-history-20261022/results.md#bios-original) |
| <a id="bios-same-weight"></a>bios-same-weight | [原文](reports/knowledge-history-20261022/results.md#bios-same-weight) |
| <a id="capacity-scaling-development-v1"></a>capacity-scaling-development-v1 | [原文](reports/knowledge-history-20261022/results.md#capacity-scaling-development-v1) |
| <a id="capacity-recall-controls-v1"></a>capacity-recall-controls-v1 | [原文](reports/knowledge-history-20261022/results.md#capacity-recall-controls-v1) |
| <a id="capacity-stability-check-v1"></a>capacity-stability-check-v1 | [原文](reports/knowledge-history-20261022/results.md#capacity-stability-check-v1) |
| <a id="standard-storage-composition"></a>standard-storage-composition | [原文](reports/knowledge-history-20261022/results.md#standard-storage-composition) |
| <a id="storage-stability"></a>storage-stability | [原文](reports/knowledge-history-20261022/results.md#storage-stability) |
| <a id="composition-data-curves-v1"></a>composition-data-curves-v1 | [原文](reports/knowledge-history-20261022/results.md#composition-data-curves-v1) |
| <a id="composition"></a>composition | [原文](reports/knowledge-history-20261022/results.md#composition) |
| <a id="perk-reconstruction-20261011"></a>perk-reconstruction-20261011 | [原文](reports/knowledge-history-20261022/results.md#perk-reconstruction-20261011) |
| <a id="perk-writer-calibration-20261012"></a>perk-writer-calibration-20261012 | [原文](reports/knowledge-history-20261022/results.md#perk-writer-calibration-20261012) |
| <a id="implicit-reasoning-paper-reproduction-v2"></a>implicit-reasoning-paper-reproduction-v2 | [原文](reports/knowledge-history-20261022/results.md#implicit-reasoning-paper-reproduction-v2) |
| <a id="second-hop-paper-reproduction-v1"></a>second-hop-paper-reproduction-v1 | [原文](reports/knowledge-history-20261022/results.md#second-hop-paper-reproduction-v1) |
| <a id="grok-usage"></a>grok-usage | [原文](reports/knowledge-history-20261022/results.md#grok-usage) |
| <a id="text-structure"></a>text-structure | [原文](reports/knowledge-history-20261022/results.md#text-structure) |
| <a id="text-pretraining"></a>text-pretraining | [原文](reports/knowledge-history-20261022/results.md#text-pretraining) |
| <a id="text-loss-source"></a>text-loss-source | [原文](reports/knowledge-history-20261022/results.md#text-loss-source) |
| <a id="representation-alignment"></a>representation-alignment | [原文](reports/knowledge-history-20261022/results.md#representation-alignment) |
| <a id="independent-alignment-v1"></a>independent-alignment-v1 | [原文](reports/knowledge-history-20261022/results.md#independent-alignment-v1) |
| <a id="bridge-reencoding"></a>bridge-reencoding | [原文](reports/knowledge-history-20261022/results.md#bridge-reencoding) |
| <a id="grok-depth"></a>grok-depth | [原文](reports/knowledge-history-20261022/results.md#grok-depth) |
| <a id="depth-hop-extension"></a>depth-hop-extension | [原文](reports/knowledge-history-20261022/results.md#depth-hop-extension) |
| <a id="grok-depth-bridge"></a>grok-depth-bridge | [原文](reports/knowledge-history-20261022/results.md#grok-depth-bridge) |
| <a id="grok-loop"></a>grok-loop | [原文](reports/knowledge-history-20261022/results.md#grok-loop) |
| <a id="grok-loop-continuous"></a>grok-loop-continuous | [原文](reports/knowledge-history-20261022/results.md#grok-loop-continuous) |
| <a id="grok-loop-same-bridge"></a>grok-loop-same-bridge | [原文](reports/knowledge-history-20261022/results.md#grok-loop-same-bridge) |
| <a id="grok-loop-supervision"></a>grok-loop-supervision | [原文](reports/knowledge-history-20261022/results.md#grok-loop-supervision) |
| <a id="latent-scaling"></a>latent-scaling | [原文](reports/knowledge-history-20261022/results.md#latent-scaling) |
| <a id="latent-confirmation"></a>latent-confirmation | [原文](reports/knowledge-history-20261022/results.md#latent-confirmation) |
| <a id="latent-support"></a>latent-support | [原文](reports/knowledge-history-20261022/results.md#latent-support) |
| <a id="memory-scaling-theory"></a>memory-scaling-theory | [原文](reports/knowledge-history-20261022/results.md#memory-scaling-theory) |
| <a id="multihop-scaling"></a>multihop-scaling | [原文](reports/knowledge-history-20261022/results.md#multihop-scaling) |
| <a id="strict-loop-hops-20261012"></a>strict-loop-hops-20261012 | [原文](reports/knowledge-history-20261022/results.md#strict-loop-hops-20261012) |
| <a id="depth-step"></a>depth-step | [原文](reports/knowledge-history-20261022/results.md#depth-step) |
| <a id="recurrence-use"></a>recurrence-use | [原文](reports/knowledge-history-20261022/results.md#recurrence-use) |
| <a id="loop-block-depth-v1"></a>loop-block-depth-v1 | [原文](reports/knowledge-history-20261022/results.md#loop-block-depth-v1) |
| <a id="grokking-dynamics"></a>grokking-dynamics | [原文](reports/knowledge-history-20261022/results.md#grokking-dynamics) |
| <a id="grokking-reproduction"></a>grokking-reproduction | [原文](reports/knowledge-history-20261022/results.md#grokking-reproduction) |
| <a id="bridge-workspace"></a>bridge-workspace | [原文](reports/knowledge-history-20261022/results.md#bridge-workspace) |
| <a id="bridge-component-20261010"></a>bridge-component-20261010 | [原文](reports/knowledge-history-20261022/results.md#bridge-component-20261010) |
| <a id="realworld-composition"></a>realworld-composition | [原文](reports/knowledge-history-20261022/results.md#realworld-composition) |
| <a id="realworld-loop-diagnosis"></a>realworld-loop-diagnosis | [原文](reports/knowledge-history-20261022/results.md#realworld-loop-diagnosis) |
| <a id="oo-budget-attribution-20261005"></a>oo-budget-attribution-20261005 | [原文](reports/knowledge-history-20261022/results.md#oo-budget-attribution-20261005) |
| <a id="parametric-architecture"></a>parametric-architecture | [原文](reports/knowledge-history-20261022/results.md#parametric-architecture) |
| <a id="shared-cache-branch-v1"></a>shared-cache-branch-v1 | [原文](reports/knowledge-history-20261022/results.md#shared-cache-branch-v1) |
| <a id="residual-cache-comparison-v1"></a>residual-cache-comparison-v1 | [原文](reports/knowledge-history-20261022/results.md#residual-cache-comparison-v1) |
| <a id="small-lm-composition"></a>small-lm-composition | [原文](reports/knowledge-history-20261022/results.md#small-lm-composition) |
| <a id="small-lm-composition-v2"></a>small-lm-composition-v2 | [原文](reports/knowledge-history-20261022/results.md#small-lm-composition-v2) |
| <a id="natural-composition"></a>natural-composition | [原文](reports/knowledge-history-20261022/results.md#natural-composition) |
| <a id="architecture-bridge"></a>architecture-bridge | [原文](reports/knowledge-history-20261022/results.md#architecture-bridge) |
| <a id="frozen-twohop"></a>frozen-twohop | [原文](reports/knowledge-history-20261022/results.md#frozen-twohop) |
| <a id="updating"></a>updating | [原文](reports/knowledge-history-20261022/results.md#updating) |
| <a id="paper-edit-reproductions-v1"></a>paper-edit-reproductions-v1 | [原文](reports/knowledge-history-20261022/results.md#paper-edit-reproductions-v1) |
| <a id="path-hop-analysis-v1"></a>path-hop-analysis-v1 | [原文](reports/knowledge-history-20261022/results.md#path-hop-analysis-v1) |
| <a id="hebbian-interface"></a>hebbian-interface | [原文](reports/knowledge-history-20261022/results.md#hebbian-interface) |
| <a id="memory-reuse"></a>memory-reuse | [原文](reports/knowledge-history-20261022/results.md#memory-reuse) |
| <a id="memory-interface-next"></a>memory-interface-next | [原文](reports/knowledge-history-20261022/results.md#memory-interface-next) |
| <a id="write-target-editing-v1"></a>write-target-editing-v1 | [原文](reports/knowledge-history-20261022/results.md#write-target-editing-v1) |
| <a id="write-target-per-edit-20261010"></a>write-target-per-edit-20261010 | [原文](reports/knowledge-history-20261022/results.md#write-target-per-edit-20261010) |
| <a id="sequential-transfer-v1"></a>sequential-transfer-v1 | [原文](reports/knowledge-history-20261022/results.md#sequential-transfer-v1) |
| <a id="sequential-replay-v1"></a>sequential-replay-v1 | [原文](reports/knowledge-history-20261022/results.md#sequential-replay-v1) |
| <a id="loop-learning-v1"></a>loop-learning-v1 | [原文](reports/knowledge-history-20261022/results.md#loop-learning-v1) |
| <a id="loop-kv-learning-v1"></a>loop-kv-learning-v1 | [原文](reports/knowledge-history-20261022/results.md#loop-kv-learning-v1) |
| <a id="moe-sequential-v1"></a>moe-sequential-v1 | [原文](reports/knowledge-history-20261022/results.md#moe-sequential-v1) |
| <a id="knowledge-change-v1"></a>knowledge-change-v1 | [原文](reports/knowledge-history-20261022/results.md#knowledge-change-v1) |
| <a id="qwen"></a>qwen | [原文](reports/knowledge-history-20261022/results.md#qwen) |
| <a id="organization"></a>organization | [原文](reports/knowledge-history-20261022/results.md#organization) |
| <a id="twohop-module"></a>twohop-module | [原文](reports/knowledge-history-20261022/results.md#twohop-module) |
| <a id="hebbian"></a>hebbian | [原文](reports/knowledge-history-20261022/results.md#hebbian) |
| <a id="reproduction"></a>reproduction | [原文](reports/knowledge-history-20261022/results.md#reproduction) |
| <a id="研究结果知识的回答组合与更新后使用"></a>研究结果：知识的回答、组合与更新后使用 | [原文](reports/knowledge-history-20261022/results.md#研究结果知识的回答组合与更新后使用) |
| <a id="读者导引"></a>读者导引 | [原文](reports/knowledge-history-20261022/results.md#读者导引) |
| <a id="2026-10-11z1路径重合的原冻结判读"></a>2026-10-11：Z1路径重合的原冻结判读 | [原文](reports/knowledge-history-20261022/results.md#2026-10-11z1路径重合的原冻结判读) |
| <a id="2026-10-11z1位置诊断的开发结果"></a>2026-10-11：Z1位置诊断的开发结果 | [原文](reports/knowledge-history-20261022/results.md#2026-10-11z1位置诊断的开发结果) |
| <a id="2026-10-11编辑写入范围的开发结果"></a>2026-10-11：编辑写入范围的开发结果 | [原文](reports/knowledge-history-20261022/results.md#2026-10-11编辑写入范围的开发结果) |
| <a id="2026-10-11同一次联合编辑的两轮权重恢复"></a>2026-10-11：同一次联合编辑的两轮权重恢复 | [原文](reports/knowledge-history-20261022/results.md#2026-10-11同一次联合编辑的两轮权重恢复) |
| <a id="2026-10-11读取侧修复与mquake正式读出"></a>2026-10-11：读取侧修复与MQuAKE正式读出 | [原文](reports/knowledge-history-20261022/results.md#2026-10-11读取侧修复与mquake正式读出) |
| <a id="一知识能被回答形成提取与保持"></a>一、知识能被回答：形成、提取与保持 | [原文](reports/knowledge-history-20261022/results.md#一知识能被回答形成提取与保持) |
| <a id="2026-10-11顺序知识注入终点的留出问法"></a>2026-10-11：顺序知识注入终点的留出问法 | [原文](reports/knowledge-history-20261022/results.md#2026-10-11顺序知识注入终点的留出问法) |
| <a id="表达组织与事实提取bios"></a>表达组织与事实提取：bioS | [原文](reports/knowledge-history-20261022/results.md#表达组织与事实提取bios) |
| <a id="同一权重的提取与规则使用"></a>同一权重的提取与规则使用 | [原文](reports/knowledge-history-20261022/results.md#同一权重的提取与规则使用) |
| <a id="早期知识量开发实验及学习控制"></a>早期知识量开发实验及学习控制 | [原文](reports/knowledge-history-20261022/results.md#早期知识量开发实验及学习控制) |
| <a id="标准transformer的知识量与优化稳定性"></a>标准Transformer的知识量与优化稳定性 | [原文](reports/knowledge-history-20261022/results.md#标准transformer的知识量与优化稳定性) |
| <a id="低知识量下已能组合之后两跳样本支持与知识量曲线"></a>低知识量下已能组合之后：两跳样本支持与知识量曲线 | [原文](reports/knowledge-history-20261022/results.md#低知识量下已能组合之后两跳样本支持与知识量曲线) |
| <a id="二知识能被组合使用经历表示与计算结构"></a>二、知识能被组合：使用经历、表示与计算结构 | [原文](reports/knowledge-history-20261022/results.md#二知识能被组合使用经历表示与计算结构) |
| <a id="perk原方法比较双臂训练全池重放与云端核验完成"></a>PERK原方法比较：双臂训练、全池重放与云端核验完成 | [原文](reports/knowledge-history-20261022/results.md#perk原方法比较双臂训练全池重放与云端核验完成) |
| <a id="固定读取模型上的写入监督比较读取校准未通过"></a>固定读取模型上的写入监督比较：读取校准未通过 | [原文](reports/knowledge-history-20261022/results.md#固定读取模型上的写入监督比较读取校准未通过) |
| <a id="ye-2025原实验原子监督与第二跳角色覆盖"></a>Ye 2025原实验：原子监督与第二跳角色覆盖 | [原文](reports/knowledge-history-20261022/results.md#ye-2025原实验原子监督与第二跳角色覆盖) |
| <a id="同一事实的组合经历"></a>同一事实的组合经历 | [原文](reports/knowledge-history-20261022/results.md#同一事实的组合经历) |
| <a id="组合搭配分布"></a>组合搭配分布 | [原文](reports/knowledge-history-20261022/results.md#组合搭配分布) |
| <a id="文本中的联合上下文与组合结果"></a>文本中的联合上下文与组合结果 | [原文](reports/knowledge-history-20261022/results.md#文本中的联合上下文与组合结果) |
| <a id="已学事实继续训练时哪些损失维持组合使用"></a>已学事实继续训练时，哪些损失维持组合使用 | [原文](reports/knowledge-history-20261022/results.md#已学事实继续训练时哪些损失维持组合使用) |
| <a id="小型参数共享模型第一跳可解码与能被继续使用"></a>小型参数共享模型：第一跳可解码，与能被继续使用 | [原文](reports/knowledge-history-20261022/results.md#小型参数共享模型第一跳可解码与能被继续使用) |
| <a id="小型标准两层模型复核分布外组合的强收益没有复现"></a>小型标准两层模型复核：分布外组合的强收益没有复现 | [原文](reports/knowledge-history-20261022/results.md#小型标准两层模型复核分布外组合的强收益没有复现) |
| <a id="用模型自己解码的桥实体做解码再编码以及几何监督的覆盖范围"></a>用模型自己解码的桥实体做解码—再编码，以及几何监督的覆盖范围 | [原文](reports/knowledge-history-20261022/results.md#用模型自己解码的桥实体做解码再编码以及几何监督的覆盖范围) |
| <a id="深度与组合形成"></a>深度与组合形成 | [原文](reports/knowledge-history-20261022/results.md#深度与组合形成) |
| <a id="参数共享的循环模型与执行次数"></a>参数共享的循环模型与执行次数 | [原文](reports/knowledge-history-20261022/results.md#参数共享的循环模型与执行次数) |
| <a id="数据支持宽度独立深度与循环"></a>数据支持、宽度、独立深度与循环 | [原文](reports/knowledge-history-20261022/results.md#数据支持宽度独立深度与循环) |
| <a id="多跳使用经历与执行计算"></a>多跳使用经历与执行计算 | [原文](reports/knowledge-history-20261022/results.md#多跳使用经历与执行计算) |
| <a id="固定每次迭代四层分别训练更多的执行次数"></a>固定每次迭代四层，分别训练更多的执行次数 | [原文](reports/knowledge-history-20261022/results.md#固定每次迭代四层分别训练更多的执行次数) |
| <a id="长期优化与原论文规模复现"></a>长期优化与原论文规模复现 | [原文](reports/knowledge-history-20261022/results.md#长期优化与原论文规模复现) |
| <a id="中间表示的因果复用"></a>中间表示的因果复用 | [原文](reports/knowledge-history-20261022/results.md#中间表示的因果复用) |
| <a id="真实事实的闭卷组合与架构适用范围"></a>真实事实的闭卷组合与架构适用范围 | [原文](reports/knowledge-history-20261022/results.md#真实事实的闭卷组合与架构适用范围) |
| <a id="跨迭代kv读取"></a>跨迭代KV读取 | [原文](reports/knowledge-history-20261022/results.md#跨迭代kv读取) |
| <a id="残差结构与跨迭代kv读取的配对扩展"></a>残差结构与跨迭代KV读取的配对扩展 | [原文](reports/knowledge-history-20261022/results.md#残差结构与跨迭代kv读取的配对扩展) |
| <a id="完整qwen的事实呈现与组合"></a>完整Qwen的事实呈现与组合 | [原文](reports/knowledge-history-20261022/results.md#完整qwen的事实呈现与组合) |
| <a id="其他架构与冻结自然模型诊断"></a>其他架构与冻结自然模型诊断 | [原文](reports/knowledge-history-20261022/results.md#其他架构与冻结自然模型诊断) |
| <a id="三新增或修改知识后已有用法能否继续工作"></a>三、新增或修改知识后，已有用法能否继续工作 | [原文](reports/knowledge-history-20261022/results.md#三新增或修改知识后已有用法能否继续工作) |
| <a id="自然模型编辑复现与按被编辑事实所在的跳的预注册分析"></a>自然模型编辑复现，与按“被编辑事实所在的跳”的预注册分析 | [原文](reports/knowledge-history-20261022/results.md#自然模型编辑复现与按被编辑事实所在的跳的预注册分析) |
| <a id="简化架构的事实模块替换"></a>简化架构的事实模块替换 | [原文](reports/knowledge-history-20261022/results.md#简化架构的事实模块替换) |
| <a id="已有强组合能力之后的原子编辑传播"></a>已有强组合能力之后的原子编辑传播 | [原文](reports/knowledge-history-20261022/results.md#已有强组合能力之后的原子编辑传播) |
| <a id="同一新事实写到哪里决定更新后能否继续使用"></a>同一新事实写到哪里，决定更新后能否继续使用 | [原文](reports/knowledge-history-20261022/results.md#同一新事实写到哪里决定更新后能否继续使用) |
| <a id="先学组合再学新事实有迁移但远未形成完整的新知识使用"></a>先学组合、再学新事实：有迁移，但远未形成完整的新知识使用 | [原文](reports/knowledge-history-20261022/results.md#先学组合再学新事实有迁移但远未形成完整的新知识使用) |
| <a id="继续回放旧组合保持明显改善新组合仍停留在约21"></a>继续回放旧组合：保持明显改善，新组合仍停留在约21% | [原文](reports/knowledge-history-20261022/results.md#继续回放旧组合保持明显改善新组合仍停留在约21) |
| <a id="循环模型的同起点比较容量重复计算与新增知识的使用"></a>循环模型的同起点比较：容量、重复计算与新增知识的使用 | [原文](reports/knowledge-history-20261022/results.md#循环模型的同起点比较容量重复计算与新增知识的使用) |
| <a id="标准四层模型与带跨迭代kv读取的循环模型本配方下没有稳定的新增知识收益"></a>标准四层模型，与带跨迭代KV读取的循环模型：本配方下没有稳定的新增知识收益 | [原文](reports/knowledge-history-20261022/results.md#标准四层模型与带跨迭代kv读取的循环模型本配方下没有稳定的新增知识收益) |
| <a id="moe标准模型与宽模型三个层次上的顺序学习成本比较"></a>MoE、标准模型与宽模型：三个层次上的顺序学习成本比较 | [原文](reports/knowledge-history-20261022/results.md#moe标准模型与宽模型三个层次上的顺序学习成本比较) |
| <a id="在知识变化中学习调用"></a>在知识变化中学习调用 | [原文](reports/knowledge-history-20261022/results.md#在知识变化中学习调用) |
| <a id="其他编辑证据与前提边界"></a>其他编辑证据与前提边界 | [原文](reports/knowledge-history-20261022/results.md#其他编辑证据与前提边界) |
| <a id="局部约束与参数更新保持"></a>局部约束与参数更新保持 | [原文](reports/knowledge-history-20261022/results.md#局部约束与参数更新保持) |
| <a id="数据组织与关系编辑历史系列"></a>数据组织与关系编辑：历史系列 | [原文](reports/knowledge-history-20261022/results.md#数据组织与关系编辑历史系列) |
| <a id="早期模块与深度证据"></a>早期模块与深度证据 | [原文](reports/knowledge-history-20261022/results.md#早期模块与深度证据) |
| <a id="hebbian启发学习及已结束支线"></a>Hebbian启发学习及已结束支线 | [原文](reports/knowledge-history-20261022/results.md#hebbian启发学习及已结束支线) |
| <a id="证据完成状态与复现"></a>证据、完成状态与复现 | [原文](reports/knowledge-history-20261022/results.md#证据完成状态与复现) |

</details>
