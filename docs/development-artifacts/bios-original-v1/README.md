# §14.27 原版bioS开发执行

2026-09-29用户授权开展实验。六条10k开发预训练已在GPU 0–5启动；全部使用约123.65M模型、每人540次完整BIO曝光。数据与实施契约见[计划§14.27.7](../../hebbian-learning-plan-v1.md#14277-开发执行契约与适配记录2026-09-29)。

- [配置](../../../configs/bios-original-development-v1.json)：两个世界、同初始化、三种BIO，随后6条QA适配及12条任务分支。
- [数据审计](data-audit.json)：官方材料、tokenizer、人物真值、5篇底层传记、模板ID、问法及日期四格的哈希与划分。
- [端到端预飞](pipeline-preflight.json)：独立200人物/2层工程检查，含连续/恢复权重逐位一致；不是科学结果。
- [启动清单](launch.json)：PID、GPU、冻结代码入口及日志位置。

结果在`results/bios-original-development-v1/world-{seed}/init-1427/{condition}/`，各阶段保存状态、训练指标、配置、模型及优化器检查点。`execution-source/`封存实际执行源码、计划、配置和依赖。小规模失败检查分别保存在`results/bios-original-preflight-v1/`及`-v2/`；通过版在`-v3/`。

```bash
.venv/bin/python scripts/run_bios_original.py status
```

每10轮保存恢复状态，0/30/90/180/360/540轮额外保存权重。重复启动需先确认原进程已退出、设备空闲，并使用同一冻结配置；文件锁防止同一轨迹重复执行。失败信息在`worker-status.json`及`worker.log`。

尚未完成：开发训练终点及其汇总、oracle和独立调用诊断、额外超参数校准、日期更新，以及100k正式矩阵。当前队列会自动执行开发训练与直接/CoT评价，不自动启动参数尚未冻结的100k实验。
