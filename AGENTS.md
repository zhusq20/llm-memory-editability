# 项目工作入口

## 研究与文档

开始分析、研究讨论、实验设计、代码修改或文档写作前，完整阅读并遵循[项目研究原则](.agents/skills/llm-memory-research/SKILL.md)。上下文恢复后若原则不在上下文中，重新读取。

2026-10-12完成状态更新：`self-decode-alignment-development-v1`于北京时间06:15、`supervised-alignment-development-v1`于08:03全部完成，各六臂300k步、合计360万新增更新；七池独立重载与W&B云端终点状态/步数/审计摘要核验均通过。08:35实际LM Docker无运行容器，两批训练及审计退出码均为0。不重启旧入口。DiscoLoop OI/IO/OO为99.30/97.53/95.66%，估算矩阵训练FLOPs为基线约2.05倍；原监督对齐OI41.23%（基线33.50%），但IO下降；两批KV融合均未超过对齐单独。完整七池及成本见[结果](docs/results.md#supervised-alignment-development-v1)。只是一张已观察图/一个初始化，不证明通用效率、唯一机制或更新传播；没有追加新训练。以下启动记载保留历史，本段完成状态优先。

2026-10-12用户随后采纳六组增补并要求编写代码、并行启动。`supervised-alignment-development-v1`已提交：原监督标签概率加权的对齐/中间CE各有无同位置首轮KV两组，另加仅同位置KV与DiscoLoop，共六臂各300k步、180万新增更新；复用前批普通Loop基线。固定镜像138项契约、CUDA基线逐值/全部梯度/连续优化一致性、六臂全尺寸32步短训与新容器七池重载全部通过。北京时间05:36启动控制器，先用GPU6/7/0/3运行四臂，另两臂随旧训练及审计完成释放资源后自动并行，不抢占旧批次。恢复先读[新增协议](docs/development-artifacts/supervised-alignment-development-v1/protocol.md)、本批`controller-process.json`、`controller-state.json`、`science-schedule.json`及实际LM Docker状态，不重复提交。新损失为原监督位置的`1-cos`，不是旧全位置自预测平方距离；KV仅直接读取同位置缓存，当前轮仍读因果前缀。可选第七臂、长训、新图及旧网格均未加入。正式终点、独立审计与云端核验尚待完成，当前没有新科学结论。

2026-10-12用户授权选择并开展下一项实验，随后明确要求利用八张卡并行做**首轮KV×自解码表示对齐**二乘二，并确认另保留自预测CE对照。`self-decode-alignment-development-v1`六臂已提交GPU0–5各300k步（总180万新增更新），同Wang φ7.2/wd.1图、原第0步权重与共同数据流；主四格另加有/无KV两条CE。固定镜像58项契约、六条CUDA短训与新容器七池逐token重载通过；固定权重诊断已完成，尚无正式训练结论。首次容器启动因cgroup只读未进入模型执行，失败保留；LM在无运行容器时恢复独立挂载namespace，其他项目未动。恢复先读[本批协议](docs/development-artifacts/self-decode-alignment-development-v1/protocol.md)、`results/self-decode-alignment-development-v1/controller-process.json`与`controller-state.json`、`runtime-repair/amendment.json`和实际LM Docker状态，不重复提交。该授权只重新打开KV与新增无真值损失的交互问题，不恢复旧窗口/R/detach/mHC网格，不自动追加长期训练或新图确认。

2026-10-07用户明确：论文围绕知识学习的三个层次——**知识能被回答、能被组合、更新后能继续被使用**。每层分别做实验并解释；第一层不是后两层的掌握率检查。表达清楚直白，不因话题与文献重合就回避重要问题。

用户随后明确两条推进路线：**严格基于实际Transformer架构的数学解释**，以及**通用架构降低三层能力学习成本的验证**。Loop、MoE、DeltaNet、Gated DeltaNet和mHC均可研究；从观察推导机制或由数学启发实验均可。保留完整残差、归一化、attention/MLP等实际计算，简化时写明条件；不以抽象记忆类比代替模型证明，不预设架构必有优势。训练效率须区分数据曝光、token、计算和总/激活参数，第三层还需计入旧能力保持。这是此前路线授权；当前执行优先级以用户2026-10-08的“先复现论文”要求和roadmap为准。

2026-10-08 用户明确否定继续由agent自行设计实验的推进方式，要求先挑出相关论文中对本项目有贡献价值的原实验，实际复现后再寻找缺口。当前先复现 Ye et al. (NeurIPS 2025)《How do Transformers Learn Implicit Reasoning?》§2.3 / Figure 3 的两条监督曲线：只学Train-II与Train-II加ID原子事实。使用作者数据函数及其指定的训练仓库，不把缩小版、自制损失或架构比较称为原实验复现。现有2024年grokking长训练保留，不重复提交。来源、配方、必要代码修正及实际执行状态见[roadmap](docs/roadmap.md#paper-reproduction-first)和[本批协议](docs/development-artifacts/implicit-reasoning-paper-reproduction-v1/protocol.md)。原模型、原评分、单世界/初始化、环境差异与未完成状态须明确；论文缺口不能在结果出来前当作已建立贡献。 v1首次尝试在训练前因未映射补齐词表类别解码失败，完整保留；6项契约及全18,986条CUDA短训练/新进程重载通过后，已提交[修订v2](docs/development-artifacts/implicit-reasoning-paper-reproduction-v2/protocol.md)两条百万迭代任务，GPU0/1独立容器运行中，不重启v1。

2026-10-08用户采纳**路径一致性**作为论文主线的工作假说：事实只在它被学过或被改写的那条计算路径上可用，能否组合、更新能否传播取决于新用途是否经过同一路径。三层仍各自做实验，由这一假说串起；它可被复现结果否定，不能在证据出来前写成已建立结论，见[roadmap §1.1](docs/roadmap.md#path-consistency)。已在两项编辑复现产生结果前冻结[跳位分组契约](configs/path-hop-analysis-v1.json)（`path-hop-analysis-v1`，零训练），只在对应运行完成且独立重放审计通过后执行，不改分组。架构效率比较降级为Loop与普通独立层的机制对比；容量bits、MoE/KV/mHC/Delta扩展不再默认入队。复现之后的受控比较须另行授权。

2026-10-09用户要求完成剩余两条审计及冻结分组分析：九条论文复现均有通过的终点验证，五条编辑复现的预定跳位分析全部完成，见[完成清单](results/path-hop-analysis-v1/completion-manifest.json)与[结果](docs/results.md#path-hop-analysis-v1)。RECENT按原冻结源码在七个独立容器中全池分段重放，1948案例/45042查询、11034次恢复及全部参数哈希通过；完整覆盖和原记录哈希核对后才停止冗余串行审计。原控制器、进度及退出码保留于[干预记录](results/rippleedits-paper-reproduction-v1/audit-scheduling-intervention-20261009/plan.json)，当前controller按替代证据收尾；不重启训练或重复编辑。ROME的CoT缩小跳位差距8.34pp，MEMIT约0pp；RippleEdits未建立各池稳定首跳优势，路径一致性仍为工作假说。分组、评分、bootstrap均未修改。W&B本地终点已到，本次没有新做云端独立核验；复现之后的新训练仍须另行授权。以上完成状态更新前文运行中的历史记录。

唯一当前规划是[Roadmap](docs/roadmap.md)。[结果汇总](docs/results.md)维护科学证据，[理论依据](docs/memory-scaling-theory.md)维护推导，[实验协议](docs/experimental-protocol.md)维护共同规则及历史机器契约。README只作导航，不再追加逐时调度播报。旧计划文件只留脚本兼容入口，不作为另一份路线。

2026-10-08用户进一步明确授权编写代码并行运行优先的原论文实验。新增7任务已提交：`second-hop-paper-reproduction-v1`两条百万步训练在GPU2/3；`mquake-paper-reproduction-v1`两条GPT-J 6B ROME/MEMIT全3000案例在GPU4/5；`rippleedits-paper-reproduction-v1`的GPT-2 XL ROME POPULAR/RANDOM在GPU6/7，RECENT由控制器排队。GPU0/1原Figure3训练继续保留，不重启。16项最终契约检查、两条第二跳全14000查询短训练重载、五条编辑CUDA短测及新进程逐token重放通过。编辑v1短测使用错误padding，MEMIT另有keyword输入hook失败，全部保留；v2恢复作者右侧编辑padding，生成单独左侧补齐，GPT-J输入适配前后logits和梯度不变。开发原文本进一步校准cloze对象名前缀计分，不改旧产物。来源、固定镜像、完整矩阵、数据/环境/评分差异见[并行复现协议](docs/development-artifacts/paper-reproductions-parallel-v1/protocol.md)。Ripple发布池为885/1922/1948，不能冒称论文同一具体样本；空标签、前提筛选和编辑失败全部保留覆盖。本批仍运行中，没有最终科学结论，恢复时读取各批冻结配置、controller和实际Docker进程，不重复提交。

2026-10-07用户要求先核对完整历史，避免持续重做失败实验，并考虑将首轮共享KV接入当前Loop路线；随后质疑“解共享”用语及复制相同随机初值作为普通模型基线。表达直接说明具体做法。新实验先对照roadmap的[历史去重表](docs/roadmap.md#history-dedup)，说明已有批次、改变的变量和新增问题，优先复用兼容对照与检查点。当前[主比较](docs/roadmap.md#loop-kv-next)为普通四层Transformer、两层Loop两轮、同Loop加共享KV；普通模型各层独立随机初始化，已有复制初值模型仅作辅助机制证据。用户随后明确授权先launch，`loop-kv-learning-development-v1`四条新增训练、独立重载及W&B云端核验现已全部完成：48,000新增更新、复用两条旧Loop，GPU0–3容器已退出。固定镜像50项测试、四条CUDA短训练及旧样本流/关闭KV一致性核验通过。共享KV的BB净增配对一负一正、均值与原Loop同为3.91pp，终点21.09%低于原Loop22.66%；预定后续门槛未通过，无B续训未启动，不扩大网格。原四格计划未启动。旧R/宽度、mHC网格、回放比例及普通原子编辑失败不再默认入队；有明确新条件时保留重新研究的可能。

用户要求用人能听懂的语言逐项解释历史实验，并由用户评价其常见程度与合理性。按设计合并种子及重试，分别说想问什么、实际改变什么、结果、设计限制、是否值得再做；不要一次要求用户评价整个长表。方法判断与用户评价分别记录，不能把用户未回复当成认可。[历史设计清单](results/history-design-review-20261007/catalog.json)及[用户评价记录](results/history-design-review-20261007/user-reviews.json)维护审查游标。用户评价旧共享KV实验为“设计有问题或没有必要”，并明确原因是“已反复失败，没必要再扩展”。结束当前知识任务的共享KV配方扩展，撤下无B续训及R/窗口/detach/mHC/世界网格的后续候选；不能仅换数据呈现、架构组合或实验名称就延续同一失败问题。保留已有熟悉收益与负结果，不把用户评价改写成架构普遍无效的证明。复制初值对照不当作普通模型主证据；继续逐项审查其它设计，未授权新训练。

用户于2026-10-07授权设计并利用空闲GPU并行开展的顺序新知识迁移、写入目标比较均已完成：四个开发/正式批次共53个任务，独立重载和W&B云端核验全部通过。具体结果与后续问题见roadmap和结果汇总，不从旧执行入口重复提交。用户在会话中的授权优先，既有批次按自己的冻结契约解释。

用户随后授权先查论文、再按相近配方实测旧组合回放。`sequential-replay-development-v1` / `sequential-replay-paired-v1` 共16任务、100,000新增更新均已完成，独立重载及云端核验全部通过。复用旧阶段A权重开展配对扩展，未重启旧训练；来源与结果见roadmap和结果汇总。完整回放优于本轮原子回放，但BB20.83%与前轮20.57%接近，不能称突破先前水平。

用户随后明确授权进行的MoE顺序学习三臂比较已经完成。`moe-sequential-development-v1`普通独立四层D4、top-2八专家M4、近等总参数宽MLP W4，采用同一已观察2Wiki开发划分、两个初始化及A8000/B4000完整回放；复用两条D4，只新增四条48,000更新。76项契约测试、四条CUDA短训练、旧D4的A/B四端点7,900条预测复算通过；新增四条科学训练、独立重载及W&B云端核验全部完成，容器退出、GPU释放。普通/MoE/宽BB23.44/22.66/21.09%，净增7.81/5.47/3.91pp，原正确AA保持80.74/77.72/72.86%。MoE旧AA终点较高、估算矩阵FLOPs约少14%，但BB配对方向混合、原正确AA保持较低，本实现耗时更长；未建立三层统一效率优势。结果、解释及完成清单见roadmap和结果汇总；不重启、不自动追加专家/路由/lr网格。该授权及完成状态更新此前“未授权新训练”的历史状态，旧批次产物未改。

## 实验恢复与已完成批次

恢复时先读取该批 `results/<batch>/controller-process.json`、`controller-state.json`、`schedule-amendment.json`（如有），再核对对应冻结配置、执行锁和实际Docker进程。不要根据旧文档中的GPU分配重启旧入口、重复提交或抢占运行实验。使用原冻结源码；调度修订和失败记录保留在相应批次目录。

2026-10-07本次整理核对的近期完成批次包括 `independent-alignment-v1`、`shared-cache-branch-v1`、`knowledge-change-v1`、`residual-cache-comparison-v1`、`composition-data-curves-v1`、`loop-block-depth-v1`，以及 `write-target-{development,confirmation}-v1` 和 `sequential-transfer-{development,confirmation}-v1`。状态与结果入口见[结果汇总](docs/results.md)；已完成批次不得重启。科学训练完成、独立重载通过和W&B同步是不同状态，不相互代替。

保留容量开发早期停止、后续完整扫描授权、负结果与所有失败记录。`capacity-stability-check-v1`的四条科学训练已完成，其历史W&B初始化失败不能改写成云端已完成。`residual-cache-comparison-v1`的GPU冲突暂停/恢复见其 `scheduling-intervention/`，受干预任务耗时不当作独占GPU基准。涉及三个旧世界的配对扩展不称新世界确认。

## Docker与资源

LM Memory Editability 与 D157/OpenTinker 是不同项目。LM实验显式使用 `docker --context lm-memory`，socket为 `/run/docker-lm-memory/docker.sock`，data-root为 `/workspaces/docker-lm-memory`；不得套用D157或默认D159环境。

相同依赖共享固定LM镜像，每个实验独立容器，分别指定GPU、CPU/内存、端口、临时目录和持久输出。启动前查看实际资源占用，用户要求尽可能并行，但不得抢占其它运行任务。结果和恢复材料保存在 `/ossfs/workspace`；`/workspaces`是计入本机配额的临时卷。

已有环境记录：[容器运行配置](docs/development-artifacts/memory-interface-next-v1/container/runtime.json)、[镜像归档说明](docs/development-artifacts/memory-interface-next-v1/container/archive-manifest.json)。清理前核对状态、依赖和挂载，保留日志及元数据；通过Docker API删除明确可清理的停止容器，不手删存储层、不重启其它实验。

## 本机Python与CUDA

仅对主机 `gpulingjun010010012239.os30` 的当前Alibaba Cloud Linux 3环境：启动Python、PyTorch、训练、torchrun或Jupyter前，将 `/lib64` 放在 `LD_LIBRARY_PATH` 最前，保留选定解释器与原有库路径。每次独立shell都需设置：

```bash
LD_LIBRARY_PATH="/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" python your_script.py
```

这是已验证的进程级修复：默认CUDA compat库曾与驱动不匹配。不能假定系统动态链接配置已经永久修复，也不把主机方案套到新容器。驱动、镜像或机器改变后重新核对库路径并实际计算验证；GPU枚举不代替矩阵计算、反向和同步。

## 追踪与验证

新实验默认启用Weights & Biases：团队 `zhusq20`，项目 `llm-memory-editability`，使用[默认配置](configs/experiment-tracking-defaults.json)。训练启动时同步启动记录进程，提供项目或运行链接，记录曲线、配置、步数、曝光、耗时与GPU资源。凭据只保存在本机凭据文件，不写入仓库或日志。

代码与文档修改遵循[CONTRIBUTING](CONTRIBUTING.md)。不得修改已有冻结训练以追溯满足新偏好。原始数据、权重、日志、配置和源码哈希保留；纯文档整理不启动训练。
