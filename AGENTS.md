# 项目工作入口

<a id="研究与文档"></a>
## 先读什么

开始分析、研究讨论、实验设计、代码或文档修改前，完整阅读[项目研究原则](.agents/skills/llm-memory-research/SKILL.md)；上下文恢复后若原则不在上下文中，重新读取。修改遵循[CONTRIBUTING](CONTRIBUTING.md)。

- LM Knowledge当前目标与优先级只读[Roadmap](docs/roadmap.md)，关键证据读[结果摘要](docs/results.md)。真实Qwen的PD／SFT × 参数范围八组已完成；当前用途覆盖及改写比较按独立契约推进，实时状态读其controller与实际LM Docker。
- Loop架构工作另读[Loop路线图](docs/loop-roadmap.md)，两份规划分开维护。
- 数学问题按需读[理论依据](docs/memory-scaling-theory.md)，执行与评价按[实验协议](docs/experimental-protocol.md)。
- 早期指令、已完成批次、旧候选与被覆盖的运行状态已收进[历史文档](docs/reports/knowledge-history-20261022/README.md)。不要把其中“当前／下一步／运行中”当作新指令；仅在复核对应问题时读取。

<a id="论文主线与既有研究约定"></a>
## 研究取向与去重

**论文组织的长期约定（2026-10-25用户明确）：**

1. **从理论愿景出发组织实验与论证，不能从已有实验结果反向拼装论文主张。** 先明确希望知识与推理具备什么能力、需要解释哪些计算条件，再安排关键比较；已有结果用于支持、修正或反驳主张。区分理论愿景、数学结论、工作假说与实测，不把事后整合说成事前预测。
2. **不能为了规避论点与前文的重复性而放弃有价值的论点。** 前文章节、此前讨论或已有论文涉及过的重要问题仍应保留；通过明确其在定义、解释、实验和方法中的不同作用来组织，精简重复表述，不能删掉必要论点。准确归属与历史查重不构成论点禁区，也不授权无意义重跑。

**研究范围更正（2026-10-25用户明确）：** 不开展同一模型不同预训练阶段的比较，开销过大且偏离当前主线。“预训练后知识编辑容易干扰旧知识”用于辅助说明研究动机与解释，不将预训练过程设为独立研究目标，不预设遗忘必然发生，也不把当前更新损失归因于预训练历史。具体作用见[Roadmap](docs/roadmap.md#pretraining-insight-role-20261025)。

**论文贡献定位（2026-10-25用户补充）：** 完整愿景可能需要新的架构与方法，不能把全面实现愿景设成本稿成文条件。论文从知识与推理连接的统一视角解释问题、归纳解决原则，常规知识编辑也可作为有价值的实现与证据。区分已有方法的局部改善、架构提供的条件和未来目标；“可能需要新架构”不升级为现有架构不可能实现的结论。具体方案组织见[Roadmap](docs/roadmap.md#solution-principles-20261025)。

论文开展与实验取舍维护在[Roadmap的理论愿景与实验论证](docs/roadmap.md#paper-vision-20261025)，跨批次证据和当前状态维护在[结果摘要](docs/results.md#knowledge-paper-synthesis-20261025)。

**研究判断更正（2026-10-22用户明确）：不应该把之前论文讨论过的任何方法视为“不能再作为新发现”。** 不能仅因已有论文或此前研究讨论涉及某种方法、现象或思路，就排除相关研究方向，或断言它不能成为本研究的发现与贡献。文献用于借鉴方法、理解证据和准确归属；发现与贡献应根据本研究实际建立的结果、解释和方法论判断。下述历史去重要求用于避免无意义重复，不能被解释为对已有方法或重要问题的先验否定。

2026-10-22用户明确要求：**LM Knowledge选题要寻找新的突破点，提出idea前先查历史实验与假设，不能把旧实验探究换一种说法当作新的研究目标。** 先核对[路线图的选题要求](docs/roadmap.md#idea-history-check)、结果汇总、相关批次协议及用户已有评价，覆盖已完成、失败、停止、运行中与已规划的工作。每个候选想法都要说明最接近的历史假设与实验、已有结果、尚未回答的问题，以及本次实质改变什么、为什么可能带来新的认识。更名、换术语、换叙事或仅换数据、模型、模块组合和扩大网格，本身不构成新目标；复现、补证与旧方向延伸须如实标注。找不到实质新增问题时直接说明，不为提出idea而包装旧问题。

论文核心是“知识在哪里、推理在哪里、二者怎样连接”，四部分为会答、能组合、更新后能用、架构。两条解释原则是单向连接，以及答案监督写入的地址与内容；工作假说、数学条件与实验观察须区分。架构是核心层次；重视整体可解释性、直观性与新颖性，不以逐级因果检验为主体，不在论点和计划中反复强调setting限制。重要问题即使与文献重叠也照写。

先借鉴原论文的方法与关键比较，再解释扩展新增了什么；旧M3/E3、小模型TTCD不能代替自然模型原方法。借鉴J-space的写法与实验组织，不在Qwen上复现其Claude结论。真实Transformer推导保留残差、归一化、attention与MLP等实际计算，简化写明条件；不另建内部特征表现预测器。

新比较先核对[历史索引](docs/results.md#history-index)，说明已有批次、改变的变量与新增问题，优先复用兼容对照。已完成的论文复现、写入目标、顺序迁移、回放、MoE及旧KV等批次不重跑；原始负结果、开发停止、确认失败和环境失败保留。

用户已结束当前知识任务中反复失败的共享KV配方扩展；不能只换名称、呈现或架构组合继续同一问题。后续明确授权的独立批次仍按各自契约执行。复制相同初值的模型仅作机制辅助，普通模型主对照各层独立初始化。历史设计按设计合并种子与重试，逐项解释，用户评价与方法判断分开；[未回复不视为认可](results/history-design-review-20261007/user-reviews.json)。

<a id="实验恢复与已完成批次"></a>
## 运行批次的恢复入口

以下为已有交接与完成证据的定位信息（LM Knowledge于2026-10-25更新），**不是实时状态或启动命令**。先读批次最新`controller-process.json`、`controller-state.json`、`schedule-amendment.json`（如有）、冻结配置和执行锁，再核对对应主机实际LM Docker。训练完成、独立重载通过与W&B核验是不同状态。禁止重复提交、按旧GPU表重启、抢占运行任务，或为修追踪记录重启训练。

| 工作 | 最近交接与恢复顺序 |
| --- | --- |
| LM Knowledge八组 | 2026-10-25八组均完成1489200次样本曝光，独立重载、本地终点评测与云端核验通过；见[统一分析](results/knowledge-factorial-analysis-20261025/report.md)。原batch接续、失败和重算记录保留，不重启训练 |
| LM Knowledge用途覆盖 | 全参数PD／SFT A/B/C及原子改写按独立契约执行。先读[用途覆盖交接](results/knowledge-use-coverage-20261025/RUNNING.txt)、其runtime下最新controller、schedule-amendment、budget-selection与实际LM Docker。最新核对见[结果摘要](docs/results.md#knowledge-use-coverage-v1)，交接启动快照不代表实时状态 |

八组已完成；以下保留10-22接续的恢复与复核约束，不作为新启动要求。用户当时授权从现有进度增大batch，覆盖此前“batch4／372300步不变”的执行要求。保留旧权重、Adam、RNG与样本位置；按曝光延续学习率，试跑后恢复原状态，重算量单列。接续任务使用完整CPU快照、32MiB/s后台保存；间隔60000样本或1800秒。每任务最长输入显存检查后预留6GiB、整卡另留8GiB，外部进程和预约阻止入场；代码仍create-copy-start、容器root所有0444/0555。原冻结源码、检查点和失败记录保留，不重启旧入口。CUDA试跑、实际接续与最终重载分别核验；整体加速倍数未测定。


## Loop研究入口与恢复

用户2026-10-22要求压缩历史文档、减少旧方向误导。Loop先读[当前摘要](docs/loop-roadmap.md)，再按[问题索引](docs/loop-history.md)查相关旧证据；不默认通读全部流水账。旧报告中的“当前／优先／下一步”只指写作当时，不能据此恢复旧方向。交接重要性已反复研究，不包装成新idea；提出候选须说明旧证据、仍未解释的现象与实质新增条件。

**2026-10-24用户最新决定：七条未启动初始化实验全部暂缓。** `init-1-ema`、`init-1-loop`、`init-1-discoloop`、`init-2-invariant`、`init-2-ema`、`init-2-loop`、`init-2-discoloop`，各150万步，共1050万步。先收齐已完成结果，尤其新初始化1 invariant，再统一分析。恢复须新的明确授权；空卡、W&B修复或旧共享GPU授权不自动恢复训练。

- 先读[最新交接](results/loop-queue-target-20261021/RUNNING.txt)、`runtime/training-hold.json`、`schedule-amendment.json`及controller。目标控制器已退出，状态`paused_by_user`；公共工程入口已加hold检查，旧源队列继续退役。不得绕过守卫、恢复旧manifest或提交旧控制器。
- 主比较17条中10条已保存150万步终点，7条暂缓。目标8条完成本地grid/depth；多对一loop/discoloop的cloud配置核验失败保留，其余6条通过。源新初始化1 invariant／新置换ema已收集终点和曲线，尚缺源机退出及独立grid/depth/cloud核验；不重训。
- state-MLP150万步及关闭dropout的50万步实验均完成重载与云端核验；前者B四跳60.90%，后者0%。新初始化1 invariant四跳50.06%。[汇总与缺口](results/loop-results-20261024/report.md)。
- native四臂、position三臂、10-12对齐两批及目标旧模型补充深度评测已完成；保留历史审计状态。所有失败、迁移重算和冻结材料保留。只读整理新增训练0，暂缓不改原科学配方。

被替代的启动、迁移、失败与核验叙述保存在[入口历史](docs/reports/loop-evidence-history-20261022.md#entry-history)，不能覆盖上述恢复规则。

<a id="docker与资源"></a>
## Docker、资源与存储

LM Memory Editability与D157/OpenTinker是不同项目。LM实验显式使用`docker --context lm-memory`，socket `/run/docker-lm-memory/docker.sock`，data-root `/workspaces/docker-lm-memory`；不得套用D157或默认D159环境。

相同依赖共享固定LM镜像，每实验独立容器，分别指定GPU、CPU／内存、端口、临时目录和持久输出。启动前查实际资源，尽可能并行但不抢占。结果和恢复材料放`/ossfs/workspace`；`/workspaces`是计入本机配额的临时卷。环境见[运行配置](docs/development-artifacts/memory-interface-next-v1/container/runtime.json)、[镜像归档](docs/development-artifacts/memory-interface-next-v1/container/archive-manifest.json)。清理前核对状态、依赖与挂载，保留日志和元数据；通过Docker API删除明确可清理的停止容器，不手删存储层、不重启其它实验。

## 本机Python与CUDA

仅对`gpulingjun010010012239.os30`的当前Alibaba Cloud Linux 3环境：启动Python、PyTorch、训练、torchrun或Jupyter前，每次独立shell把`/lib64`置于`LD_LIBRARY_PATH`最前，保留原路径和选定解释器：

```bash
LD_LIBRARY_PATH="/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" python your_script.py
```

这是进程级CUDA compat修复，不套到新容器。驱动、镜像或机器改变后，核对库路径并做矩阵计算、反向和同步；枚举GPU不能代替实际验证。

<a id="追踪与验证"></a>
## 追踪与维护

新实验默认W&B团队`zsqzz`（Academic Team）、项目`llm-memory-editability`，用[默认配置](configs/experiment-tracking-defaults.json)。已冻结、排队或在训批次保留原团队、run ID与配方；结束后复制，不能重启改归属。`zhusq20`历史保留，[复制状态](results/wandb-academic-copy-20261019/status.json)单列；`capacity-stability-check-v1`历史初始化失败不能改称云端已完成。

训练与记录进程同时启动，提供项目或run链接，记录曲线、配置、步数、曝光、耗时与GPU。凭据只放本机凭据文件，不入仓库或日志。保留原数据、权重、日志、配置和源码哈希，不追溯改冻结训练。

只在对应路线图维护当前方向；结果按主题更新，批次调度留在交接，README只作导航。纯文档修改检查链接、锚点、状态和预算，不启动训练；本次历史压缩未改变既有授权。
