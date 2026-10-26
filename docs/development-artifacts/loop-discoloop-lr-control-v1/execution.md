# 执行记录

2026-10-16 CDT最新接续：用户指定本机 GPU1／GPU2，并批准使用本机重建镜像。四组迁移、环境差异、零更新挂载失败与修复、双卡队列及实时状态入口见[本机执行记录](sn4622128200-execution.md)。原机两组及监控保持；下述暂停记录作为交接历史保留，不再代表四组当前执行状态。

2026-10-16 22:28 CDT用户暂停交接：四组未开始正式训练的Loop 5e-4／1e-3、DiscoLoop 2e-4／5e-4已暂停自动启动；原调度器1431925退出。GPU7 Loop 2e-4、GPU6 DiscoLoop 1e-3容器继续，独立监控1963751／1963753已成功附着原容器，接续审计与云端核验。暂停前状态与动作收据保留于持久`handoff/`；两组实时进度改读各自training/status及handoff监控状态。原冻结科学源码不变，无重复科学更新，目标GPU尚未验证。四组使用[专用脚本与交接说明](four-arm-handoff.md)，不恢复旧六组入口。下文为历史记录。

最终授权矩阵为普通Loop与DiscoLoop各`2e-4、5e-4、1e-3`，共六条300k；此前低学习率提案未启动。协议及源码冻结后不再随结果调整。

2026-10-16 CDT开发检查：固定LM镜像中117项CPU契约通过，Ruff检查与格式检查通过。一个原作者notebook比对测试因本机无该参考notebook跳过；原世界和面板字节及哈希已有独立来源记录，不重建数据。另两项原runner基线一致性测试需要CUDA，明确移至每臂cuda-contracts阶段，尚不能称已通过。

首次CPU完整选择执行为117通过、2失败、1跳过；两项失败均因原测试调用CUDA，而该容器未挂GPU。原测试与训练核心保留不改，失败容器`loop-discoloop-lr-control-v1-development-contracts`及日志保留。调整CPU/GPU测试分工后，`loop-discoloop-lr-control-v1-development-contracts-v2`为117通过、1跳过、2移至CUDA阶段。早期静态长行问题已修正；静态容器保留。最早宿主测试收集缺少PYTHONPATH，未执行模型或训练。

新入口复用原训练核心；新旧两条基线中的模型、数据流、优化器、CUDA图和评分五份核心模块已与当前源码逐字节核对一致。两条原生300k基线的检查点、七池独立审计、完整云端JSON审计均核验通过；原初始化文件SHA256在两条基线间一致。本批只复制原初始化和数据及小型基线记录，复用终点不重新训练。

首次资源检查十张GPU均有任务或显式预留。控制器必须等实际独占空闲后才进行各组CUDA检查、32步短训、独立重载及正式训练；队列提交不代表科学更新已经开始。实际最新状态以持久目录的`controller-state.json`与`science-schedule.json`及LM容器为准。

2026-10-16 20:45 CDT后台提交，控制器PID1315580；20:46 CDT完成215项文件冻结，冻结配置SHA256为`5d71382bf5b2b678e72dcf9706b93ce4f6fde0aebc2d80ed319736d16d3399f8`，source-manifest SHA256为`d057cc10f615ea8c87978d58bb56f610aae6127b94336411d073083365e28ba3`。冻结版本CPU契约容器退出0，117通过、1参考notebook跳过、2明确移至CUDA阶段。核对时控制器为`running_or_waiting`，六臂均在`cuda-contracts`前等待实际空闲资源，正式科学更新0，没有任何本批GPU容器。后台调度继续；原有任务未修改。

仓库`results/loop-discoloop-lr-control-v1/`只保存小型提交/冻结/状态快照，`snapshot-origin.json`注明来源及快照时间；后续恢复必须读持久目录的实时状态，不把快照当作最新运行状态。

用户随后明确指定GPU7先跑，其余空闲后自动并行；已按[GPU7调度修订](gpu7-scheduling-amendment.md)接管。六项资源/锁契约与Ruff通过，v2控制器PID1431925；第一臂普通Loop lr2e-4已在GPU7通过两项原CUDA契约，继续全尺寸验证。实际GPU7已有其它任务，按用户指定共卡并记录真实占用；其它五臂仍只等实际空闲。原科学源码、配置、输入和已完成阶段未改、不重复，旧等待控制器及第一次仍受全局锁阻挡的调度修订均保留。

2026-10-16 21:05 CDT：首组32步全尺寸短训完成，七池独立重载答案/EOS逐token一致、NLL差0，全部通过。`loop-discoloop-lr-control-v1-loop-lr2e-4-train`在GPU7启动，W&B运行[emvpeilq](https://wandb.ai/zhusq20/llm-memory-editability/runs/emvpeilq)在线，核对时第0步初始化。GPU6释放后`loop-lr5e-4`自动进入CUDA验证，其余四臂等待；共享GPU7上的其它任务保留。短训实际32工程更新，正式步数以原始状态/曲线为准，无新科学结果。

21:06 CDT实际首组已完成1000/300000步，loss=3.46897、梯度范数1.16360，均有限；实测科学训练41.38秒，属于共卡观测，不外推独占耗时。GPU6短暂空闲期间两条其余普通Loop各通过CUDA契约，资源再被外部占用后按原空闲规则等待preflight；三条DiscoLoop继续等卡。原source-manifest摘要再次核对不变，队列自动接续生效，没有新增或重复科学轨迹。

持久目录：`/ossfs/workspace/llm-memory-editability/loop-discoloop-lr-control-v1`。恢复按协议核对controller、source-manifest、阶段claim/exit、锁与实际容器；不重复提交入口。W&B运行链接由每条正式阶段开始时生成，未开始前使用[项目入口](https://wandb.ai/zhusq20/llm-memory-editability)。
