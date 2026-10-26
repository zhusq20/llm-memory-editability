# Loop：当前问题、证据与实验

更新：2026-10-23。本文是Loop方向唯一当前规划；[知识主线](roadmap.md)分开维护。历史按[问题索引](loop-history.md)查阅，完整数字与执行记录按需展开。本文依据已保存记录整理，实时进度见批次状态文件。

<a id="thesis"></a>
<a id="goal"></a>
## 1. 现在要回答什么

**只学过原子形式的事实，怎样在循环Transformer中成为可重复调用的推理步骤？** 这是已有研究主线，不能再当作新idea。工作假说是每轮查找与交接需要兼容原子学习；尚未证明某一种门、位置处理或词表交接是必要、充分条件。

当前等待三类答案：已有优势能否跨初始化与数据重复；同一道题增加计算能否继续改善且保住正确答案；直接读取旧状态与候选的小MLP能否替代手工统计量。训练波动、后续覆盖与长问题失败已有诊断，但没有统一解释。 最新state-MLP四跳在60万／70万步为2.94／4.04%，仍远低于早期峰值，尚未到终点；[10-23结果快照](results.md#loop-accuracy-20261023)。

<a id="principles"></a>
**研究原则。** 以已有InfiLoop方法评审为参照；监督只有答案与EOS，深监督也只在各段末监督答案。桥实体、跳数、轮次、位置角色、图身份标签不进入监督或架构控制信号。观察、机制假说和方法效果分开写；不以扩大网格代替关键比较。

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

[原四臂证据](reports/loop-evidence-history-20261022.md#evidence)与[基线比较](results.md#loop-baseline-robustness-v1)。基础DiscoLoop在**旧固定四轮配方、raw/R8**的B三跳为88.26%；不能只用当前弱配方代表它，也不能把跨配方差异全归因于架构。

- **终点优势没有解决训练波动。** invariant的B四跳70万步6.86%、130万步93.74%、终点79.42%；单换打分或衰减没有救回低谷。[诊断](results.md#invariant-checkpoint-dip-20261022)
- **状态MLP门的本次退化主要发生在正确答案形成前。** 25.6万→50万步第二跳候选85.12→21.16%；固定晚期权重，原子调用接口供体使四跳5.66→27.96%，输入范数回退未恢复。供体依赖真实中间实体，尚无通用修复或优化根因结论。[零训练诊断](results.md#loop-state-regression-diagnosis-20261022)
- **读取修复有效，但不足以修好整题。** 70万步第二跳候选74.40%→98.94%，完整四跳仅6.86%→13.86%；统一降低attention温度也只是有限改善。[读取](results.md#invariant-second-hop-20261022)／[温度](results.md#invariant-attention-temperature-20261022)
- **多算有短范围收益，也会破坏旧正确答案。** invariant全池B四跳R8／R32／R256为79.42／81.98／74.18%；固定256题R8→R16384为78.91→74.61%。两种样本范围分开报告。[深度结果](results.md#loop-invariant-robustness-v1)
- **更长问题与更多轮次是两个问题。** invariant在R16的A／B五跳为19.24／0.66%，六至八跳接近随机。位置影响明显，但RoPE三臂未统一改善。[位置与长跳](loop-history.md#position-depth)

invariant门只读熵、最大概率、相邻解码一致度、状态相对变化、余弦五个统计量；名称所指的输入变换不变性不等于完整模型对实体或数据分布不敏感。[架构示意图（中英文、矢量与源码）](../paper/figures/invariant-loop-v1/README.md)

<a id="current"></a>
<a id="raw-state-handoff"></a>
## 3. 已授权、尚待收齐的比较

| 问题 | 已定比较与新增预算 | 证据入口 |
| --- | --- | --- |
| 重复性与数据变化 | invariant／ema：两个新初始化、新置换、多对一世界，共8条×150万步；旧模型复用 | [协议](development-artifacts/loop-invariant-robustness-v1/protocol.md) |
| 普通Loop与基础DiscoLoop是否同样受益 | 原世界三个初始化、新置换、多对一；复用原世界基础DiscoLoop，只补9条×150万步 | [协议](development-artifacts/loop-baseline-robustness-v1/protocol.md) |
| 直接读状态是否足够 | state-mlp：`[旧状态;候选]→16维GELU→1`，仅1条×150万步；复用invariant／ema／content | [协议](development-artifacts/loop-state-handoff-development-v1/protocol.md) |

合计18条、2700万新增更新，是既有授权总量，不是剩余预算。测试时深度复用权重，新增训练0。state-mlp同时改变输入与参数量，不能将胜负单独归因于“不变性”。[10-22曲线快照](results.md#loop-accuracy-20261022)中的新门与换数据读数仍是中途结果；原世界普通Loop已完成训练、重载、深度与云端核验。

执行恢复只查[目标机交接](../results/loop-queue-target-20261021/RUNNING.txt)、[调度修订](../results/loop-queue-target-20261021/runtime/schedule-amendment.json)、[控制器](../results/loop-queue-target-20261021/runtime/controller-state.json)及对应批次收据，再核对实际LM Docker。10-23用户明确允许共享GPU执行余下九条训练，资源池4/7/5/8，本机该用户全部批次最多四卡，新初始化1、2最后；原科学配方及150万步终点保持。四卡CUDA与W&B检查通过，后续短训、正式训练及评测状态见实时交接。已完成训练不重跑，缺失评测接续。10-22迁移的完整恢复一致性与重算成本仍保存在[迁移记录](../results/loop-queue-target-20261021/runtime/gpu-migration-20261022-13/amendment.json)；旧入口不重启。源两条已完成训练的后续按[迁移交接](../results/loop-queue-transfer-20261021/HANDOFF.txt)核对，不重新训练。

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
用户要求对照Huginn/Ouro并优先简洁训练。当前拟议只做**三处dropout从0.1改为0的单因素配方比较**，复用原state-MLP对照，目标是检查B组合完整曲线与后期保持，不以更低训练loss判成功。已有随机段深、分段答案监督、detach、EMA保留。当前第二段会重采输入embedding mask，携带状态与新注入输入来自不同mask；这是假设来源，不能称已证bug或根因。Huginn提供的训练代码及所选attention后端确无这些dropout；Ouro只核到官方发布配置/代码，未独立核验全部历史预训练stage。标准配方适配不是新算法；旧零dropout任务同时改变其他条件，未发现此同世界干净配对。[具体方案与逐项对照](development-artifacts/loop-dropout-recipe-design-v1/design.json)。2026-10-22用户随后明确授权执行一条，新增预算50万步，学习率仍沿用原150万步时间轴；复用原state-MLP对照，不重启旧任务。执行代码和检查单独冻结，空卡及本机四卡配额通过后启动，重点比较25.6万至50万步的四跳保持。[执行协议](development-artifacts/loop-dropout-development-v1/protocol.md)与[实时队列](../results/loop-dropout-development-v1/runtime/controller-state.json)。此前设计冻结件中的“未授权”只代表当时状态。

先前[接口对抗方案](development-artifacts/loop-interface-robustness-design-v1/calling-interface-training-design.pdf)保留为未验证候选，当前不优先；[文献核对](development-artifacts/loop-interface-robustness-design-v1/literature-assessment.json)未建立其为常见Loop配方。去dropout也可能增加高重复数据上的组合过拟合，不能保证修复；不自动叠加归一化、学习率、监督或门/KV网格。

<a id="docs"></a>
需要查证时：先看[历史问题索引](loop-history.md)，再查[完整记录](reports/loop-evidence-history-20261022.md)或[阶段PDF（10-21快照）](reports/loop-transformer-summary-20261021.pdf)。理论仍在[理论依据](memory-scaling-theory.md)，不在本页重复展开。
