# Loop：当前问题、证据与实验

更新：2026-10-25。本文是Loop方向唯一当前规划；[知识主线](roadmap.md)分开维护。历史按[问题索引](loop-history.md)查阅，完整数字与执行记录按需展开。本文依据已保存记录整理，实时进度见批次状态文件。

<a id="thesis"></a>
<a id="goal"></a>
## 1. 现在要回答什么

**只学过原子形式的事实，怎样在循环Transformer中成为可重复调用的推理步骤？** 这是已有研究主线，不能再当作新idea。工作假说是每轮查找与交接需要兼容原子学习；尚未证明某一种门、位置处理或词表交接是必要、充分条件。

**独立四臂正式比较已启动；七条旧初始化训练继续暂缓。** 10-24（America/Chicago）用户授权最新版invariant、普通Loop、DiscoLoop和InfiLoop四卡并行，各150万步，共600万步，已分别在GPU6/4/7/8推进正式更新。统一主干、独立Poisson深度、终点监督与全程反传；InfiLoop的单状态统一主干适配已由用户确认，具体定义见[冻结协议](../results/loop-fourway-terminal-20261024/protocol.md)，运行信息见[交接](../results/loop-fourway-terminal-20261024/RUNNING.txt)。这不是恢复七条旧初始化队列。已完成结果继续按[10-24汇总](results.md#loop-results-20261024)分析；历史训练波动、换数据差异与长问题失败仍待解释。

<a id="principles"></a>
**研究原则。** 以已有InfiLoop方法评审为参照；监督只有答案与EOS，深监督也只在各段末监督答案。桥实体、跳数、轮次、位置角色、图身份标签不进入监督或架构控制信号。观察、机制假说和方法效果分开写；不以扩大网格代替关键比较。

<a id="feedback-controls"></a>
**新增两项直接对照已按10-24用户授权共享启动：** InfiLoop将聚合候选从H_t改为H_t+q_t，不新增整个候选的归一化，保留原始初态与聚合；invariant将候选从RMSNorm(H_t+q_t)改为RMSNorm(H_t)，仍以解码概率计算打分与γ，保留历史合并。q_t为DiscoLoop解码的soft-feedback embedding。分别复用正在运行的InfiLoop/invariant作为基线，各150万步，共额外300万步；新增两组分别共享GPU4/8，原四组进程未中断，七条旧初始化仍暂缓。原81项相关CPU检查及新增7项共享入场检查通过；两组最长输入/16轮CUDA检查、首轮梯度与EMA/R32检查通过，12次临时更新单列且不复用权重。两组已有正式更新及W&B记录，尚无终点结论。历史content已含软反馈，但初态、候选归一化和监督配方不同，不能替代这两项配对比较。见[独立对照协议与历史核对](../results/loop-feedback-controls-20261024/protocol.md)、[运行交接](../results/loop-feedback-controls-20261024/RUNNING.txt)。

<a id="recipe-ablation"></a>
**配方最小消融新增四组，各25.6万步，保留150万步学习率时间轴。** 10-25用户已授权拆分分段监督S、段间梯度截断G及浅深度影响，复用当前Invariant为基线；新增S、G、S+G和T→max(T,4)四组，共102.4万次科学更新。原始完整初值、数据、样本顺序、Poisson深度随机流及其余架构条件保持；独立K∈{4,5,6}随机流，T>K时S在K/T各给半权重，G只截断h/N/D/p并保留原始e的梯度。按用户指定共享CUDA6/7/4/8；原六组持续运行，七条旧初始化继续暂缓。主要比较25.6万步EMA/R8的A留出四跳与B四跳，不能把早期结果直接当作150万步终点结论。见[冻结协议](../results/loop-recipe-ablation-20261025/protocol.md)与[启动及核验记录](../results/loop-recipe-ablation-20261025/RUNNING.txt)。

<a id="abc-method-check-150"></a>
**通用LLM训练目标下，ABC已从150步接续，目标各150万步。** 三组统一固定R8和首轮e，A末轮CE完整反传、B末轮CE仅反传后4轮、C完整反传并加入0.1随机中间深度同目标CE。沿用原多跳数据、状态和学习率时间轴，新增4499550次更新；保存完整曲线，比较组合泛化与训练成本。接续前发现B原150步存在BF16权重缓存造成部分参数无梯度的实现缺陷：原始结果保留，第151步起修复，B历史边界单列；不把旧短程检查当作方法已验证。见[运行交接](../results/loop-abc-continuation-1500k-v1/RUNNING.txt)、[接续协议](../results/loop-abc-continuation-1500k-v1/protocol.md)及[原结果更正](../results/loop-abc-method-check-150-v1/report.md)。自然文本有效性仍待验证，其他在训和暂缓安排保持。

<a id="terminal-training"></a>
## 已授权的代码改动：逐样本随机深度、终点监督

2026-10-24用户确定并授权实现[新配方](../configs/loop-invariant-poisson-terminal-v1.json)：
一个共享Transformer block，每条样本每次出现独立抽Poisson(8)，越界重抽至1–16；
首轮输入e、后续输入h+e，保留身份无关打分、DiscoLoop反馈与归一化历史合并。
初始化仍为h0=N0=e、D0=1，p0由e解码；整条轨迹共用一次embedding dropout。
仅对最终合并状态监督，全部轮次反传。代码及GPU小规模验证完成后，用户已授权将此配方用于上述独立四臂正式比较；
七条旧初始化实验继续暂缓，旧配置与冻结源码保持历史行为。

损失按整个逻辑batch的有效监督token数归一化：CE总和／有效token总数，忽略-100。
当前每题恰为一个答案token和一个EOS token，所以原先“二者平均”的数值相同；
多token答案或不等长掩码下不能按类别、样本或深度组等权平均。
先独立抽深度，再让整个batch一起推进；样本到期后用可微掩码冻结h/N/D/p，不detach，
保留最初的e及embedding dropout。最后一次终点损失反传，统一裁剪、更新参数与参数EMA。
采样RNG随检查点保存，记录深度直方图、采样轮数、含到期样本无效计算的执行轮数与监督token数；
矩阵FLOPs按实际整批执行估算。旧按深度分组函数仅保留作数值参照。
推理以R8为主，检查R4/R16/R32；R32是训练深度之外的评测。

实现位于[训练入口](../scripts/run_loop_native_handoff.py)、
[终点训练函数](../src/llm_memory_editability/loop_terminal_training.py)与
[契约测试](../tests/test_loop_terminal_training.py)。[整批掩码GPU复测](../results/loop-terminal-masked-cuda-20261024/report.md)
已完成40次临时更新：batch 512的Poisson配方从[旧分组版](../results/loop-terminal-cuda-20261024/report.md)
0.921降到0.104秒／步，约快8.83倍；峰值已分配显存从0.666增到5.248 GiB。
全16轮为0.102秒／步，逐步损失及梯度范数与旧版相同。梯度有限且能传回首轮，
但仍有裁剪前尖峰；检查仅覆盖学习率预热初期，尚未建立长期稳定性。127项相关CPU检查通过，
旧冻结源码守卫的两项既有测试失败单列保留。该工程检查没有保存临时权重，检查GPU已释放；
后续正式比较使用独立批次、全新初值和明确的600万步授权。

<a id="evidence"></a>
## 2. 可以据此判断的证据

原世界、原初始化、当前配方，均为150万步、参数EMA、R8（循环8轮）；A的组合参加训练，B只学原子。`ema`是常数打分的交接方法名，与用于评测的参数EMA不同。

| 交接 | A留出四跳 | B两跳 | B三跳 | B四跳 |
| --- | ---: | ---: | ---: | ---: |
| 普通Loop | 0.22% | 1.10% | 0% | 0% |
| 基础DiscoLoop（recipe-additive） | 0.08% | 40.36% | 0.02% | 0.04% |
| ema（常数打分） | 100% | 97.40% | 54.10% | 10.78% |
| content | 100% | 15.28% | 0.28% | 0.14% |
| invariant | 100% | 99.32% | 96.12% | 79.42% |
| state-MLP | 100% | 99.56% | 90.64% | 60.90% |

[原四臂证据](reports/loop-evidence-history-20261022.md#evidence)与[基线比较](results.md#loop-baseline-robustness-v1)。基础DiscoLoop在**旧固定四轮配方、raw/R8**的B三跳为88.26%；不能只用当前弱配方代表它，也不能把跨配方差异全归因于架构。

- **终点优势没有解决训练波动。** invariant的B四跳70万步6.86%、130万步93.74%、终点79.42%；单换打分或衰减没有救回低谷。[诊断](results.md#invariant-checkpoint-dip-20261022)
- **状态MLP门的本次退化主要发生在正确答案形成前。** 25.6万→50万步第二跳候选85.12→21.16%；固定晚期权重，原子调用接口供体使四跳5.66→27.96%，输入范数回退未恢复。供体依赖真实中间实体，尚无通用修复或优化根因结论。[零训练诊断](results.md#loop-state-regression-diagnosis-20261022)
- **读取修复有效，但不足以修好整题。** 70万步第二跳候选74.40%→98.94%，完整四跳仅6.86%→13.86%；统一降低attention温度也只是有限改善。[读取](results.md#invariant-second-hop-20261022)／[温度](results.md#invariant-attention-temperature-20261022)
- **多算有短范围收益，也会破坏旧正确答案。** invariant全池B四跳R8／R32／R256为79.42／81.98／74.18%；固定256题R8→R16384为78.91→74.61%。两种样本范围分开报告。[深度结果](results.md#loop-invariant-robustness-v1)
- **更长问题与更多轮次是两个问题。** invariant在R16的A／B五跳为19.24／0.66%，六至八跳接近随机。新零训练位置干预中，无padding仅位置编号+1便使B四跳82.18→0.28%；去位置后B原子仍99.58%、四跳仅0.10%。位置插值可使content的A五跳0→44.60%，但未恢复各组B长跳，不能据此声称位置影响已排除。[位置干预完整对照](../results/loop-position-controlled-long-hops-20261025/report.md)／[位置历史](loop-history.md#position-depth)
- **第五答案主要未形成，形成后也可能丢失。** 旧invariant的B五跳99.14%在前16轮从未答对，原生曾正确后丢失仅0.20%；R5强制读第四位置使A五跳19.24→94.98%，B仅0.66→8.90%。完整原子调用供体使B当轮99.80%、R16为61.64%，不能把含答案的接口旁路当作纯读取修复。已有RoPE DiscoLoop的B四／五跳44.92／0.12%，也未恢复六至八跳；共同长度泛化失败不等于已经检验换数据敏感性。[完整诊断与对照](../results/invariant-five-hop-diagnosis-20261025/report.md)
- **RoPE DiscoLoop的保持与干预时机不同。** 同方法诊断中B五跳99.88%前16轮从未正确，原生曾正确后丢失为0；R5强制桥读取几乎无效，R4规范真桥有部分恢复。完整原子结果接入后B从100%保持到99.98%，错桥结果也稳定保持；与invariant供体后的明显丢失不同。第四桥虽有部分软反馈读出损失，已有桥的题仍无法形成第五答案。现有证据定位共同失败阶段和不同修复反应，尚未定位唯一模块原因。[DiscoLoop诊断](../results/disco-five-hop-diagnosis-20261025/report.md)
- **晚轮旧接收残差会压制新读取，但并非不可逆锁死。** 固定正确原子attention，RoPE B当轮raw在原残差下从R4的31.98%降至R5的0.10%；缩放或清零残差后两轮均100%，保留原生attention则仍0%。原子attention携带关系信息，不能据救援证明旧状态关系尚存。invariant给纯桥并缩放自身输入，在配对末关系池B仍78.66%，支持进入块的h+e含可用关系；形成后仍有明显丢失。全头单源桥V的RoPE fresh正控不成立，其负结果不证明关系彻底丢失。280池零训练已完成，不改变在训或暂缓安排。[校准拆分与模型差异](../results/loop-receiver-attention-20261025/report.md)

invariant门只读熵、最大概率、相邻解码一致度、状态相对变化、余弦五个统计量；名称所指的输入变换不变性不等于完整模型对实体或数据分布不敏感。[当前三张独立架构图：普通Loop／Highway-gated Loop（适配设计）／Invariant](../paper/figures/loop-residual-comparison-v3/README.md)。后两张统一共享Transformer、输入注入、软反馈与候选RMS，再比较聚合模块；Highway分支是待实现的对照设计，不是已训练方法。普通Loop仅作总体架构参照，严格直接交接对照应保留共同候选并令h_t=c_t。[原始Highway层间图](../paper/figures/loop-residual-comparison-v2/README.md)、[NDR图](../paper/figures/loop-residual-comparison-v1/README.md)、[Invariant单图](../paper/figures/invariant-loop-v12/README.md)与[文献核查](../paper/figures/loop-architecture-comparison-v1/README.md#cross-loop-residual-comparison)保留。图示更新没有修改冻结实验。

<a id="current"></a>
<a id="raw-state-handoff"></a>
## 3. 旧批次执行范围：收齐结果，暂缓七条初始化训练

**10-24用户决定覆盖此前“共享GPU接续九条／初始化最后”的执行指令。** 暂缓的是 `init-1-ema`、`init-1-loop`、`init-1-discoloop`、`init-2-invariant`、`init-2-ema`、`init-2-loop`、`init-2-discoloop`。每条原定150万步，共1050万步暂缓；不是完成、取消或可自动恢复的候补。没有新增预算。

| 比较 | 现在状态 | 原设计 |
| --- | --- | --- |
| invariant／常数打分：初始化与换数据 | 8条中5条已保存150万步终点，3条暂缓；优先收齐新初始化1 invariant与新置换ema的核验材料 | [协议](development-artifacts/loop-invariant-robustness-v1/protocol.md) |
| 普通Loop／基础DiscoLoop | 9条中5条完成训练与本地评测，4条初始化重复暂缓；多对一两条W&B配置核验失败保留 | [协议](development-artifacts/loop-baseline-robustness-v1/protocol.md) |
| 直接读状态的state-MLP | 唯一150万步完成，独立重载和云端核验通过；输入与容量同时改变，不能单独归因于不变性 | [协议](development-artifacts/loop-state-handoff-development-v1/protocol.md) |
| 关闭dropout的state-MLP | 唯一50万步完成，独立重载和云端核验通过；按50万同步数比较，不续训到150万 | [协议](development-artifacts/loop-dropout-development-v1/protocol.md) |

原18条×150万步及额外50万步合计2750万步；已保存终点对应1700万步，1050万步暂缓。这里是科学更新数，不含工程短训与迁移重算，也不代表全部审计已通过。

当前顺序：整理固定终点与完整曲线 → 核对新初始化1 invariant／新置换ema的源机退出、独立重载与云端缺口 → 联合分析初始化、数据与门的差异。现有权重、预测和失败原样保留；不重新训练，不因收齐材料补齐暂缓矩阵。

执行恢复先读[最新交接](../results/loop-queue-target-20261021/RUNNING.txt)、[用户暂缓记录](../results/loop-queue-target-20261021/runtime/training-hold.json)、[调度修订](../results/loop-queue-target-20261021/runtime/schedule-amendment.json)及[控制器](../results/loop-queue-target-20261021/runtime/controller-state.json)。旧共享／迁移／恢复入口均不可启动训练。源机两条终点可从共享持久盘读取；源机名称暂不能解析，远程Docker退出未在本次核验。只读收集不等于终点重载或云端审计通过。

<a id="position-invariance"></a>
native四臂、position三臂及10-12对齐两批已完成；不重复训练。原始失败与各自核验状态留在[档案](reports/loop-evidence-history-20261022.md)。

<a id="next"></a>
<a id="scope"></a>
<a id="idea-history"></a>
## 4. 怎样读后续结果与提出新问题

先比较固定终点、完整曲线和初始化最差值；换数据时同时检查原子掌握，多对一另报答案分布与猜测基准。同题深度曲线分别报告新增正确与原正确丢失；相同步数不等于相同计算成本。

提出新idea前按[历史问题索引](loop-history.md)核对相关正负结果与未完成项，再读对应原记录和协议。候选写清：**已有证据 → 尚未解释的具体现象 → 实质新增条件 → 能区分解释的比较**。重要问题可以重访，但换名称、架构组合或继续扩大门／KV／轮数网格不自动成为新贡献。

旧KV配方扩展已由用户结束；其他未进入当前授权的候选不自动排程。新想法、文献启发与旧诊断都不自动增加训练预算，也不修改正在运行的冻结配方。

<a id="interface-robustness-design"></a>
<a id="dropout-recipe-design"></a>
关闭dropout的单因素比较已经完成：三处dropout从0.1改0，沿用原state-MLP、初值、样本与150万步学习率时间轴，但本条预算仅50万步。B两／三／四跳终点均0%，原子事实100%，A留出四跳0.16%；未改善本次组合泛化。与原state-MLP的50万步及完整曲线配对，不能把50万终点与150万终点当作同步数比较，也不自动延长训练。该现象与组合过拟合相容，尚不能确定优化根因。原[设计与文献核对](development-artifacts/loop-dropout-recipe-design-v1/design.json)、[执行协议](development-artifacts/loop-dropout-development-v1/protocol.md)和[最终状态](../results/loop-dropout-development-v1/runtime/controller-state.json)保留。

先前[接口对抗方案](development-artifacts/loop-interface-robustness-design-v1/calling-interface-training-design.pdf)保留为未验证候选，当前不优先；[文献核对](development-artifacts/loop-interface-robustness-design-v1/literature-assessment.json)未建立其为常见Loop配方。去dropout也可能增加高重复数据上的组合过拟合，不能保证修复；不自动叠加归一化、学习率、监督或门/KV网格。

<a id="docs"></a>
需要查证时：先看[历史问题索引](loop-history.md)，再查[完整记录](reports/loop-evidence-history-20261022.md)或[阶段PDF（10-21快照）](reports/loop-transformer-summary-20261021.pdf)。理论仍在[理论依据](memory-scaling-theory.md)，不在本页重复展开。
