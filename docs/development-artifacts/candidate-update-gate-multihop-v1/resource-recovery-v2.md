# GPU2排队恢复（原失败尝试完整保留）

2026-10-17 CDT，首个GPU0正式容器在完成第0步全池评测、保存第0步完整检查点后退出137。Docker OOMKilled=false，未记录Python异常；没有1000步心跳，失败前未保存的更新数未知，不能宣称科学更新为零。原控制器已退出1，冻结源码、全部初始评测、W&B lhdec13c00、Docker容器与失败日志保留。

本机公开服务显示按用户GPU配额执行器在运行，当时已有GPU1/2/5/6四张LM卡；源码默认上限4与故障现象吻合，但实际配置及执行日志需管理员权限，因此归因为待确认。没有关闭服务、修改配额、改变所有者或启动第五张卡重试。

在用户已授权新增这一组实验的范围内，准备独立调度修复尝试`attempts/resource-recovery-v2`：等待GPU2的`ungated-loop-continuation-v1/loop-l1-r4`完整150万步训练及R1–4评测成功，核对配置、模型/图及阶段身份后，再在同一条既有卡队列运行。每个新阶段启动前检查两套Docker预约、真实GPU占用及本用户现有GPU并集，加入新阶段后不得超过保守上限4。任何依赖失败或新尝试失败均停止，不循环重试。

模型、科学runner、数据、初始化、样本流、训练预算、损失、优化器、评测和固定镜像保持原冻结字节；仅新配置的调度元数据及GPU从0改2，W&B因配置哈希变化产生独立ID，保留第一次运行。新尝试从同一初始化开始，完整目标仍150万步、不是新科学臂；失败尝试未保存计算单独记为未知。CPU24–25/16GiB保持；重新做1次丢弃工程更新后训练。

当前只是排队等待，不能称为正式在训。恢复读取新尝试controller-process/state、freeze及阶段收据；原根控制器needs_attention是保留的失败状态，不重启该入口。

[新配置](../../../configs/candidate-update-gate-multihop-resource-recovery-v2.json)、[当前队列](../../../results/candidate-update-gate-multihop-v1/execution/attempts/resource-recovery-v2/controller-state.json)。
