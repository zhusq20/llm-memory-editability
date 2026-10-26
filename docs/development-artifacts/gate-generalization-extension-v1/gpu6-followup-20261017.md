## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run / code_runner_agent（当前会话执行）
- Origin Date: 2026-10-17T20:38:01-05:00
- Verification Status: UNVERIFIED
- Version Label: exp_result_v1

# CUDA6 接续剩余新图实验

用户本次授权：“现在有没有排队的实验?可以在cuda6跑”。本次只迁移已冻结、未开始的五项任务，全部来自 `gate-generalization-extension-v1`，科学总预算仍为1230万新增更新。迁移任务合计150万步，属于原预算；没有增加实验条件或延长训练。

第一组新图的原论文门P、固定.95、学习.95均完成300k及原定推理轮数评测，原接收控制器于2026-10-17 20:30 CDT退出0。本次启动前，CUDA6显存0 MiB、利用率0%、没有计算进程或容器预约。使用本机 `sn4622128200` 的CUDA6，UUID `GPU-d1972508-51b0-ae3d-188b-f1613ec33e07`，CPU12–13；原GPU1/2/5实验保持运行。

| 次序 | 冻结任务 | 步数 | 提交状态 |
| --- | --- | ---: | --- |
| 1 | confirmation-2--fixed-update | 300000 | 已启动 |
| 2 | confirmation-2--token-update | 300000 | CUDA6自动接续 |
| 3 | confirmation-3--paper | 300000 | CUDA6自动接续 |
| 4 | confirmation-3--fixed-update | 300000 | CUDA6自动接续 |
| 5 | confirmation-3--token-update | 300000 | CUDA6自动接续 |

新图2的P仍由原控制器执行。每项按原配方完成训练与终点grid后，下一项才准入；原Wang与跳数外推任务保持原队列。第一项在线记录为 [W&B lhb035e801](https://wandb.ai/zhusq20/llm-memory-editability/runs/lhb035e801)。

2026-10-17 20:39:12 CDT实际启动核验：首条已达4000/300000步，loss=1.72773、梯度范数=8.81028，均有限；第1000步完整评测与检查点已落盘。容器绑定上述CUDA6 UUID及CPU12–13，实际GPU利用率99%、显存2567 MiB，控制器存活，其余四项为 `queued_on_gpu6`。此处仅确认训练开始，不解释早期性能。

## 执行与核验

复用 `/tmp/gate-extension-gpu6-v1` 的冻结科学源码、原始配置和数据，冻结清单SHA256为 `ee4dde6eed29bad1b65962bd4f1bd51356589a476f9d24dab0d8750ab90a0efc`。控制器已在提交前重新逐文件核验，并沿用本机第一组新图已使用的固定镜像 `sha256:fa200da9dd78d3423ffc30572bda8b88a5b1c7e72a25509bf63e6ae4a5a3ef6f`。与原主机镜像不同这一环境限制保留，不声称跨环境逐值等价。

仅调度程序增加显式任务选择、保留历次迁移修订和本机GPU排他锁；本次使用真实利用率检查，不沿用上次显示异常的放宽。10项迁移契约在主机及固定镜像均通过，Ruff检查和格式检查通过。科学训练代码、数据、损失、初始化、步数及评分均未修改；原短训与独立重载/云端终点审计豁免保留。

共享目录的原子创建在Docker创建前阻止旧控制器重复提交，并通过相对链接保留原报告入口。旧控制器可能把已接收任务记为目录冲突，须以本次接收控制器及实际产物解释，不能由此重试训练。本次修订写入新文件 `schedule-amendment-gpu6-handoff-v2.json`，没有覆盖第一次迁移。持续监控容器存活、日志、资源和原配置阶段超时；失败保留并停止接续，不自动重试。

恢复时先读[本次提交记录](../../../results/gate-generalization-extension-v1/gpu6-handoff-v2/submission.json)、[控制器进程](../../../results/gate-generalization-extension-v1/gpu6-handoff-v2/controller-process.json)、[实时状态](../../../results/gate-generalization-extension-v1/gpu6-handoff-v2/controller-state.json)和[调度修订](../../../results/gate-generalization-extension-v1/gpu6-handoff-v2/schedule-amendment.json)，再核对实际LM Docker。首个PID为2224290，仅作历史记录；不得凭该PID或旧GPU分配重复提交。完整操作程序及哈希已保存在接收目录 `operation/` 与 `submission.json`。本记录是执行交接，不是性能结论或独立模型审计。
