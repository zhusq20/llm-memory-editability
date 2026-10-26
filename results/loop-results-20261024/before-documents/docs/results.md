# 研究结果：当前摘要与证据索引

整理：2026-10-22（芝加哥）。本页先给LM Knowledge当前比较与核心证据；研究优先级见[Roadmap](roadmap.md)。详细结果移至[历史正文](reports/knowledge-history-20261022/results.md)，所有旧批次锚点在页末保留。

<a id="knowledge-method-factorial-1000-v1"></a>
## 当前八组比较：尚无新终点结论

Qwen3-1.7B，SFT／Prompt Distillation各配full、原范围LoRA、down-only、down-LoRA；1000案例、9928段、297840条训练问答。八组各5轮、1489200次样本曝光；原batch4对应每组372300步。10-22用户授权从已有进度增大batch，后续实际更新数另报，不再固定原步数。[原协议](development-artifacts/knowledge-method-factorial-1000-v1/protocol.md)、[接续修订](development-artifacts/knowledge-method-batchsize-20261022/amendment.md)。

**10月23日队列恢复：SFT-full／SFT-down已完成训练，等待独立重载和终点评测；其余六组已启动。** 用户授权本机最多四卡并释放GPU4旧预约后，资源池扩为GPU4/3/6/8。新控制器PID1725118接管原四个训练容器，容器ID、PID和启动时间保持；两组down-LoRA从未启动状态入场，均选batch8，试测更新丢弃、初始权重／Adam／RNG恢复后已产生正式训练损失和梯度。原六组接续batch分别为SFT-full／LoRA／down的16／16／16，PD-full／LoRA／down的8／4／8。每组终点仍为1489200次曝光，没有新增科学预算。

旧控制器的Docker资源检查失败记录保留；新入口遇到资源查询失败时阻止新入场并重读，排队评测也不再阻挡可安全共享GPU的新训练。17项CPU检查、Ruff和GPU4矩阵计算／反向／同步通过。两组新W&B run已读回running，云端指标上传与终点核验分开记录；不能由启动核验代替训练终点。恢复见[四卡交接与Loop手动入口](../results/queue-recovery-20261023/HANDOFF.txt)、[持久调度修订](../results/queue-recovery-20261023/knowledge/amendment.json)；原接续记录见[旧交接](../results/knowledge-method-batchsize-20261022/RUNNING.txt)。

主评价使用Qwen3-4B非量化权重并关闭thinking，沿用作者评分消息与true/false解析，报告true／有效判分数及无效覆盖。旧32案例EM/F1与新语义重评分别保留。训练完成、独立重载、云端核验逐项更新；目前不预填效果排名或整批加速倍数。[W&B](https://wandb.ai/zsqzz/llm-memory-editability)。

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
## 对当前主线仍有效的证据

以下是跨批次证据索引，各行按自己的模型、数据与评分解释，不合并成同一实验。

| 问题 | 关键观察 | 证据用途与完整记录 |
| --- | --- | --- |
| 表达怎样影响提取 | bioS规范QA为5.07／35.65／99.96%；多表达加换序后，换名84–91%跟名字；单篇92–94%跟前文 | 提取与地址形成：[bioS](reports/knowledge-history-20261022/results.md#bios)、[地址](reports/knowledge-history-20261022/results.md#bios-write-address-v1) |
| 原问法与新用途是否相同 | 原问法100%，留出问法37.61–48.62%；标准小模型两条新事实各自答对64/64、组合0/64 | 区分能力，避免重复证明相同失败：[问法](reports/knowledge-history-20261022/results.md#sequential-phrasing-20261011)、[双新事实](reports/knowledge-history-20261022/results.md#ce-cross-sample-double-new-v1) |
| 答案监督写在哪里 | 后段梯度集中于末位置，写入方向偏向输出答案；小模型只施加答案位置写入仍能回答新事实 | 连接行为与计算；梯度份额不等于训练后参数增量分解：[梯度](reports/knowledge-history-20261022/results.md#answer-gradient-address-v1)、[位置](reports/knowledge-history-20261022/results.md#write-position-20261020) |
| 自然模型的层与位置 | 实体／问题依赖随位置和层变化；两Qwen读取阻断效应明显不同 | 份额不直接定位知识／推理／答案：[地图](reports/knowledge-history-20261022/results.md#knowledge-reasoning-map-v1)、[逐层整理](reports/layer-evidence-chain-20261021.md) |
| 上下文与参数写入 | 组合用途在早中段从问句读取上下文；“先桥后读”预定判断四组均不成立 | 保留反证；旧M3等不能代替原方法：[读出比较](reports/knowledge-history-20261022/results.md#write-mode-readout-v1) |
| 自然模型用途练习 | Qwen3.5-2B某集未练事实直接使用.26→.85，另一集.60→.46；自然两跳退化 | 有正收益，也有保持问题；本批未存权重：[用途练习](reports/knowledge-history-20261022/results.md#use-adapter-development-v1) |
| 改写目标与传播 | 小型受控模型三种目标都编辑24/24成功，传播3.13→68.75→85.42%；旧组合96.47→89.84% | 传播和保持一起报；不是Qwen的24例：[写入目标](reports/knowledge-history-20261022/results.md#write-target-editing-v1) |
| 深度与交接 | 未共享＋词表交接OO 6.19%，共享＋词表交接95.66%；该配方单独修一个条件时OO≤6.2% | 受控架构证据；相对普通Loop约2.05倍估算矩阵训练FLOPs需计入：[架构对照](reports/knowledge-history-20261022/results.md#untied-discoloop-development-v1)、[成本](reports/knowledge-history-20261022/results.md#supervised-alignment-development-v1) |
| 新知识持续积累 | 邻居保护下DiscoLoop新事实128/128、双新63/64、旧原子／组合保持99.22／96.99%；普通Loop仅部分改善 | 保留已成功的条件，不概括为全无进展：[邻居保护](reports/knowledge-history-20261022/results.md#neighbor-protection-development-v1) |

上表不提升原批次的审计状态；部分诊断仍是“云端未独立核验”。九项论文复现与五项编辑跳位分析已经完成，路径一致性仍是工作假说，见[冻结分组结果](reports/knowledge-history-20261022/results.md#path-hop-analysis-v1)。

## Loop与历史材料

Loop先读[当前摘要](loop-roadmap.md)与[按问题查重](loop-history.md)。下列Loop条目保留结论和旧锚点，完整表格、失败及调度原文收在[证据档案](reports/loop-evidence-history-20261022.md)；旧启动记录不代表实时状态。

<a id="loop-accuracy-20261023"></a>
2026-10-23 00:15 UTC只读快照：state-MLP训练约78.1万步，最新70万步B两／三／四跳77.86／28.46／4.04%，原子与A留出四跳100%。B四跳25.6万71.24%→50万5.66%→60万2.94%→70万4.04%，仍在低位、最近略回升，尚未到150万终点。同步数比较：新置换世界130万invariant／常数73.52／9.34%；多对一50万53.20／72.52%，方向不同，均未到终点。关闭dropout仍等待资源，科学更新0。14组保存记录、68份来源哈希及1419绘图点核对通过；源机两组为最新保存记录，本次未作远程容器核验，模型评测与新增训练0。[结果表与解释](../results/loop-queue-target-20261021/accuracy-20261023-0015/report.md)、[四跳图](../results/loop-queue-target-20261021/accuracy-20261023-0015/four-hop-focus.png)、[六页完整PDF](../results/loop-queue-target-20261021/accuracy-20261023-0015/loop-accuracy-comparison.pdf)、[同步数数据](../results/loop-queue-target-20261021/accuracy-20261023-0015/paired-condition-common-step.csv)。

<a id="loop-accuracy-20261022"></a>
2026-10-22 15:30 CDT曲线快照：参数EMA、R8、各池5000题。state-mlp的B四跳从30万步56.38%降至40万步23.88%，同在40万步invariant为29.04%；原世界invariant的150万步终点79.42%。新置换／多对一invariant在110万步分别47.84／70.88%，均未训练完成。14组现有曲线保留回落和缺失组，不混合不同配方或以不同步数端点作配对结论。[原世界曲线](../results/loop-queue-target-20261021/accuracy-20261022-2030/original-world.png)、[换数据与初始化](../results/loop-queue-target-20261021/accuracy-20261022-2030/robustness.png)、[六页PDF](../results/loop-queue-target-20261021/accuracy-20261022-2030/loop-accuracy-comparison.pdf)、[最新读数](../results/loop-queue-target-20261021/accuracy-20261022-2030/latest-accuracy.csv)、[同条件同步数比较](../results/loop-queue-target-20261021/accuracy-20261022-2030/paired-condition-common-step.csv)。[前次记录](reports/loop-evidence-history-20261022.md#loop-accuracy-20261022)。

<a id="loop-state-regression-diagnosis-20261022"></a>
2026-10-22状态MLP门的已有检查点诊断：69组、34.5万题次、零训练。25.6万→50万步第二轮第二跳候选85.12→21.16%，四跳R8为71.24→5.66%；晚期前8轮曾答对与R8答对均283/5000，主要为形成失败。固定晚期权重，原子问答的attention后状态供体使候选99.98%、四跳27.96%；错误实体供体为0／1.90%。仅回退B输入范数未恢复（5.46%），早期交接整体移植损伤原子/A。同一供体也使早期四跳升至88.98%，干预后早晚仍差61.02pp；接口是实质瓶颈，尚未解释大部分训练退化、确定优化根因或得到通用LLM修复。供体含真实中间实体，仅作诊断。[三页报告](../results/loop-state-regression-diagnosis-20261022/checkpoint-regression-report.pdf)、[逐题配对](../results/loop-state-regression-diagnosis-20261022/paired-stage-analysis.json)、[核验](../results/loop-state-regression-diagnosis-20261022/validation.json)。

<a id="loop-state-handoff-development-v1"></a>
state-mlp已于10-22正式开训；唯一新增150万步，旧对照复用。门直接读取旧状态和候选，输入与容量同时改变；终点结论待完成。 [完整记录](reports/loop-evidence-history-20261022.md#loop-state-handoff-development-v1)。

<a id="loop-queue-expansion-20261022"></a>
10-22调度记录：普通Loop的150万步训练、重载、深度与云端核验完成，多对一ema接续开训。原队列已迁移并扩为最多四卡；恢复读最新修订与控制器，禁止重启旧入口。 [完整记录](reports/loop-evidence-history-20261022.md#loop-queue-expansion-20261022)。

10-23用户明确授权共享GPU接续余下九条训练，每条仍150万步，初始化比较最后。目标机GPU4/7的多对一普通Loop／基础DiscoLoop通过原32步短训和独立重载后已正式开训；GPU5/8接续已完成模型的缺失评测。四卡CUDA与W&B访问检查、23项CPU检查和Ruff通过；原配方和预算保持。此前调度快照的并发写入失败仅发生在工程阶段，日志与已通过检查保留；不算科学训练失败。状态见[最新交接](../results/loop-queue-target-20261021/RUNNING.txt)与[共享接续修订](../results/loop-queue-target-20261021/runtime/shared-20261023-v3/amendment.json)。

<a id="invariant-attention-temperature-20261022"></a>
固定权重统一降低attention温度有有限收益：70万步B四跳R8由6.86%到10.46%，130万步由93.74%到97.62%；没有消除训练低谷。 [完整记录](reports/loop-evidence-history-20261022.md#invariant-attention-temperature-20261022)。

<a id="invariant-second-hop-20261022"></a>
第二跳局部读取修复使70万步候选正确率74.40%→98.94%，完整四跳仅6.86%→13.86%；读取与后续保持均需解释。 [完整记录](reports/loop-evidence-history-20261022.md#invariant-second-hop-20261022)。

<a id="invariant-checkpoint-dip-20261022"></a>
invariant在25.6万／70万／130万步的B四跳为66.10／6.86／93.74%；单换打分、衰减或阻止特定覆盖未救回低谷，不能归结为一个门参数。 [完整记录](reports/loop-evidence-history-20261022.md#invariant-checkpoint-dip-20261022)。

<a id="loop-invariant-robustness-v1"></a>
旧invariant全池B四跳R8／R32／R256为79.42／81.98／74.18%；固定256题R8→R16384为78.91→74.61%。短范围收益与长循环退化并存；新增初始化与数据比较另待终点。 [完整记录](reports/loop-evidence-history-20261022.md#loop-invariant-robustness-v1)。

<a id="loop-baseline-robustness-v1"></a>
基础DiscoLoop旧配方raw/R8的B三跳88.26%，当前配方EMA/R8仅0.02%，须同时报告。当前普通Loop已完成，B两／三／四跳为1.10／0／0%；其他新增基线待终点。 [完整记录](reports/loop-evidence-history-20261022.md#loop-baseline-robustness-v1)。

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
