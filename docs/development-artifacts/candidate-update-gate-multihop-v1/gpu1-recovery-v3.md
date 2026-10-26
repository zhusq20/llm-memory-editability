# 候选门实验迁至GPU1：第三次执行尝试

## Material Passport

- Workflow: ARS experiment-agent/run，当前会话执行。
- Status: RUNNING，启动状态以本尝试controller-state和训练status为准。
- 用户确认：GPU1结束后，将候选门实验安排到GPU1；“ok,请你安排这个吧”。

2026-10-17 23:25 CDT，GPU1的无学习门控DiscoLoop训练与R1–4评测均退出0，原控制器退出0；
实际GPU1显存/计算/容器预约检查为空，本用户原有GPU2/5/6三张卡，加入GPU1不超过四张。
只迁移已有候选门一臂：同模型、同图、初始化、样本流、优化、150万步目标、评测与固定镜像。
CPU改为已释放的4/5，显存比例.16、16GiB主存不变。725份原冻结源码/数据等文件逐字节保持；
唯一改写的已有冻结文件为调度配置，另增加调度器、参考配置和检查材料，当前共731文件冻结。
新配置与旧配置逐字段校验，14项迁移/容量/前序成功契约及Ruff通过；不重训任何已有对照。

## 启动故障与保留

第一尝试GPU0退出137，最后保存step0，未保存更新量未知；第二尝试在创建容器时失败。
两个原控制器均已退出1，容器、权重、配置、日志及W&B全部保留。
本次先进行无GPU目录检查，实际得到Docker bind mount permission denied：原批次父目录和attempts目录为0700，
Docker无法穿过该NFS目录链。该检查发生于模型执行前、科学更新0；日志保留在
[挂载失败记录](../../../results/candidate-update-gate-multihop-v1/execution/attempts/gpu1-recovery-v3/host-validation.log)。
这提供了与旧恢复路径一致的具体启动障碍，第二次未保留daemon错误，仍不追溯声称已读到其原报错。

修改原目录为0711的操作被自动审批拒绝，未执行。本次采用独立新输出目录
`/mnt/disk2_from_server2/siqizhu4/lm-memory-persistent/workspace/llm-memory-editability/candidate-update-gate-gpu1-v3`；
所有既有目录权限不变，新目录仅开放路径通行，初始归档仅当前用户可读，凭据仍从原私有凭据文件注入。
新目录全部731文件可读、输出可写及固定镜像14项契约均通过后才启动GPU控制器。
原路径下的第三次尝试仅保留失败的无GPU检查，**实际训练状态在下列独立目录**，不从旧状态重复提交。

## 执行与恢复

- [当前控制器](../../../results/candidate-update-gate-multihop-v1/gpu1-recovery-v3/controller-state.json)
- [实际训练步数](../../../results/candidate-update-gate-multihop-v1/gpu1-recovery-v3/runs/candidate-token/status.json)
- [调度修订](../../../results/candidate-update-gate-multihop-v1/gpu1-recovery-v3/schedule-amendment.json)
- [冻结配置](../../../configs/candidate-update-gate-gpu1-v3.json)

控制器PID2734490，GPU1 UUID GPU-364f0e00-7f5f-e6a7-9fea-03eb31c0bb3b。
GPU工程检查通过后，从同一确定性初始化开始150万步，再自动做R1–4终点评测；不使用失败尝试作效果结果。
保留原短训/独立审计豁免，在线W&B不是终点审计。每阶段检查GPU占用与用户GPU并集，错误记录包含daemon stderr，
168小时阶段上限，失败不循环重试。源码/数据及本次调度器归档保存在当前独立输出目录。

GPU1工程17项检查全部通过，唯一工程更新丢弃；正式训练容器于23:26 CDT启动，
[W&B新运行](https://wandb.ai/zhusq20/llm-memory-editability/runs/lh14e7ae00)已联机。
