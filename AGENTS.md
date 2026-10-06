# 项目工作入口

在本项目中开始分析、研究讨论、实验设计、代码修改或文档写作之前，先完整阅读并遵循 [项目研究原则](.agents/skills/llm-memory-research/SKILL.md)。新会话或上下文恢复后，如果这些原则不在当前上下文中，应先重新读取。

随后按任务需要阅读 `README.md`、`CONTRIBUTING.md` 及相关研究计划与实验协议。具体实验的冻结契约和历史结果仍按对应文档执行。

用户已要求后续新实验默认启用 Weights & Biases，账号/团队 `zhusq20`，项目 `llm-memory-editability`。采用[默认配置](configs/experiment-tracking-defaults.json)，启动训练时一并启动记录进程，向用户提供项目或运行链接；记录学习曲线、配置、步数、曝光、耗时与GPU资源。已有冻结训练保持原样，不因这项未来偏好追溯修改。凭据仅保存在本机凭据文件，不写入仓库、日志或冻结配置。

2026-10-04 用户明确：LM Memory Editability 与 D157/OpenTinker 是不同项目。LM 新实验使用专属 `docker --context lm-memory`（socket `/run/docker-lm-memory/docker.sock`，data-root `/workspaces/docker-lm-memory`），不要套用 D157 的 context、容器或依赖环境。相同依赖共享固定 LM 镜像，每个实验独立容器，指定资源与持久输出；本轮 GPU 为2–5。环境恢复记录在 [本轮容器目录](https://huggingface.co/datasets/zsqzz/llm-memory-editability/blob/main/docs/development-artifacts/memory-interface-next-v1/container/runtime.json) 与 [镜像归档说明](https://huggingface.co/datasets/zsqzz/llm-memory-editability/blob/main/docs/development-artifacts/memory-interface-next-v1/container/archive-manifest.json)。主机 Python 仍须启动前将 `/lib64` 放在 `LD_LIBRARY_PATH` 首位；容器使用已独立验证的库配置，不套用主机修复。

2026-10-05 用户终止D157/OpenTinker后，授权排队的LM实验优先使用释放的GPU 6、7。当前架构主实验继续使用GPU 2–5；`oo-optimizer-attribution-v1`按[调度修订](docs/development-artifacts/oo-optimizer-attribution-parallel-20261005/amendment.md)在GPU 6–7独立并行，不再等待架构批次结束。原等待控制器已由修订入口替换；恢复时读取该批`controller-process.json`与`schedule-amendment.json`，不要重新启动旧等待入口或重复占用这两张卡。

2026-10-06 用户随后明确授权容量实验提前使用空闲GPU0–1，已按[调度修订](docs/development-artifacts/capacity-scaling-gpu01-20261006/amendment.md)解除对上述两个旧批次的等待；GPU2–7的旧实验继续运行。原等待控制器已停止，不要重新启动。恢复时读取`results/capacity-scaling-development-v1/controller-process.json`、`controller-state.json`、`schedule-amendment.json`及[当前执行配置](docs/development-artifacts/capacity-scaling-gpu01-20261006/runtime-fix-v1/execution-config.json)。启动时发现阶段执行锁缺失，已单独封存工程修复，科学配方保持原样。[本批协议](docs/development-artifacts/capacity-scaling-development-v1/protocol.md)限定低负载校准、四档负载和固定目标背景对照，通常8条、至多9条训练；低负载前提不满足则停止扩展。新宽度、架构及独立世界确认尚未加入该队列。

同日05:05，本批两条1,000曝光校准及唯一3,000曝光长校准已全部完成并审计/追踪通过；长校准原子100%、开发II仅0.68%，旧控制器以`development_prerequisite_not_met`结束。该停止记录是历史事实，后续执行按下述用户追加授权进行。

同日用户明确要求“继续原定的这个所有那些探测实验，并且保留这个所有结果”。已按[完整扫描修订](docs/development-artifacts/capacity-scaling-full-scan-20261006/amendment.md)在GPU0–1恢复原定3条高负载扫描及3条固定目标背景对照，复用上述3条校准，不重复训练；共9条的原预算、lr=0.003、3,000曝光以及模型/数据/评分不变。低分门槛改为报告项，不再阻止开发扫描；结果全部保留，包含低分、负结果、失败尝试和原停止记录。不得将低负载组合失败直接归因为容量或串扰。恢复时读取根控制器状态及`schedule-amendment.json`，当前配置为[完整扫描冻结配置](docs/development-artifacts/capacity-scaling-full-scan-20261006/execution-config.json)，入口为其冻结源码中的`continue_capacity_scaling.py`；不要重启旧控制器或重复占用GPU。新增宽度、架构、独立世界确认仍不在本批队列。

2026-10-06 07:12，本轮用户授权调查研究、设计并自动开展下一步实验。旧`capacity-scaling-development-v1`九条全部完成，GPU0–1已释放；新`capacity-recall-controls-v1`已在同两卡启动（LM专属Docker）。该批9个旧权重只读连续调用评估、6条原子/等计算原子复习、2条同父状态长期续训，共8条新训练。协议`docs/development-artifacts/capacity-recall-controls-v1/protocol.md`，冻结入口`source/scripts/execute_capacity_controls.py`，恢复先读`results/capacity-recall-controls-v1/controller-process.json`及`controller-state.json`，不要重复占卡或重启已结束旧扫描。所有训练及评分源码从本批冻结目录加载；旧架构与OO归因保持原队列。单世界开发不能称为新世界确认，外部拆题参照不能称为原生组合或内部串扰机制。

同日07:19，首批原子复习出现拟合后退化（H256：100%→29.10%），另冻结并启动等待控制器`capacity-stability-check-v1`：H256/4096 × mixed/atomic_replay，共4条lr=.001配对。读取该批`controller-process.json`/`controller-state.json`；必须等`capacity-recall-controls-v1`全部17项完成并审计，再使用GPU0–1。旧17项原样继续，切勿绕过前置条件抢占GPU。该附批是开发校准，不自动追加其它学习率、架构或世界。

2026-10-06 09:04，用户批准最近论文方法复核后的新实验计划。旧连续调用/学习控制17项全部完成且W&B核验通过；稳定性4条训练全部完成/重载通过，其控制器`finished_with_failures`仅记录W&B网络初始化失败，不能改称云端同步完成。GPU0–1已启动`composition-data-curves-v1`，入口为该批冻结`source/scripts/execute_composition_curves.py`，先读其`controller-process.json`、`controller-state.json`和`docs/development-artifacts/composition-data-curves-v1/frozen-config.json`。每条训练独立LM容器，其他GPU上的旧矩阵照常运行，不重复启动旧等待控制器。

新批复用论文随机关系图与已成功训练模块，先一个开发世界校准2/4层宽64（固定4096实体词表、32关系、度16、lr1e-4/wd.1），128k预算不满足前提时仅允许预定512k；成功后冻结所选深度，在三个新世界运行7档嵌套独立组合支持，等128k计算与等4000曝光分别报告（长校准分支预定8000曝光）。三个正式低负载锚点全部通过才展开6档N=128–4096负载曲线，最低档复用；最多40条新训练/64 GPU小时。保留全部结果，组合答案不增加独立bits，不将变化的闭合图称为嵌套事实干预，不按正式测试重新选择配方。

同日09:07开发两条128k完成并重载/W&B远端核验通过：四层原子/训练组合100%、固定II86.62%（完整2926题86.64%）通过门槛，两层固定II52.34%保留。已选定四层、464,896参数、4000曝光，自动启动三个新世界的21条组合支持矩阵；不再启动512k开发延长。当前正式矩阵运行中，之后的15条高负载需三个正式锚点全部通过；实际预计新训练38条含已完成开发，没有新批最终曲线结论。

同日09:44，用户要求利用空闲GPU并行加速，已按[调度修订](docs/development-artifacts/composition-data-curves-parallel-20261006/amendment.md)将新曲线批次扩为GPU0–4，GPU6–7在OO归因六项全部完成、独立审计通过且控制器complete后自动加入；GPU5继续旧HC8。新增三卡FP32/BF16实际计算、19项相关测试及静态检查通过。原控制器396245已终止，仅接管现有容器，原W&B进程保留。恢复读取根controller-process.json/controller-state.json/schedule-amendment.json及本修订schedule.json，使用其冻结execute_composition_parallel.py，勿重启旧入口。训练与阶段科学配置不变；负载队列只按实体数降序调度，原三世界门槛和全部38条标准分支预算保持不变。
