# 本机执行记录

2026-10-17 08:48:44 UTC 已从冻结源码提交后台控制器，PID558545。批次根见 [持久状态](../../../results/gated-loop-depth-extrapolation-v1/)。

开发阶段：固定派生镜像39项CPU测试全部通过（6.84秒），包含小模型训练与独立重载、事实表重排不变性、三seed合并统计与四个真实PDF输出；9文件Ruff检查与格式通过。开发复制过程曾因目标目录尚未齐全而出现两次“no tests ran”退出4；完整复制后实际39项执行通过，未占用GPU或启动科学训练。开发临时CPU容器已停止，日志/镜像保留。

GPU1/2原lr-control两路lane均finished、每job训练/audit/cloud-audit均退出0，且independent_audit_passed/cloud_verified均真。现两卡显存0、利用率0，仍以每次准入时实际进程和预约为准。旧norm-handoff控制器已退出needs_attention；本批不重启它，也不把失效控制器当成永久资源依赖，若其恢复则动态保护预约。

本批在每个GPU任务前校验冻结源码。正式CPU contracts、数据生成、CUDA短测、pilot、正式训练和云端核验分开记状态；本记录不把已提交视为已通过CUDA。读取controller-process.json、controller-state.json、science-schedule.json及execution/内退出码后恢复，不重复提交。

选择checkpoint只在每1000步及固定终点的A-short/R5验证节点进行；第0步验证仅诊断，不是保留的候选checkpoint。best与固定预算last均独立重载并完整评估。

W&B项目：[llm-memory-editability](https://wandb.ai/zhusq20/llm-memory-editability)。每条训练开始时创建独立online run，具体链接写入smoke|pilot|train/<arm>-seed<seed>/tracking.json。

## 调度修订与已观察执行

08:55:39 UTC提交调度guard，当前控制器PID571808，原PID558545的记录保留。持久盘源码核验与最初资源观察之间可能相隔较久，因此在创建和启动新GPU容器前分别即时复查；busy时保留队列等待。3项guard与15项原调度测试全部通过，Ruff通过，独立科学契约39项不变。`schedule-amendment.json`绑定guard SHA256及原freeze SHA256；原训练源码、配置、数据、镜像与预算未改。原两条smoke容器已经退出0，未停止、重启或重复其训练，控制器按已有claim接回并继续独立audit。

恢复执行入口为 `runtime-amendment/guard_gated_depth_launch.py run --base <批次根> --wandb-token-file <原私有凭据文件>`；先核对controller-process及实际进程，不重复启动。`controller-amendment-v2.log`与`admission-rechecks/`记录当前调度；原controller.log/submission/attempts及冻结材料保留。

正式CPU容器39项测试通过（7.51秒），退出0。数据审计通过，2048原子全覆盖，A组合train/val/test零交叉；A/B短测试各5120、长测试各6144题，全部答案分布平衡。

N-Full与N-Fixed均已完成32步CUDA smoke，退出0；相同主干hash与完整样本流hash，32轮有限、5轮前缀逐值一致。N-Full另通过真实BF16原生前向与全主干梯度逐值一致。这是工程验证，不是短链已学会或外推成功的证据。两臂独立重载随后也已通过；其余四臂由后台继续执行。

实际短测W&B：[N-Full](https://wandb.ai/zhusq20/llm-memory-editability/runs/tzyjpq4l)、[N-Fixed](https://wandb.ai/zhusq20/llm-memory-editability/runs/eum2ife1)。正式18条训练只有pilot门槛通过后才启动。
