# 一步训练启动修复与CUDA5/6调度

本记录只修订宿主启动与资源准入。[科学协议](protocol.md)、三架构冻结源码、共同模型/AdamW/RNG、每事实一步与各10000外层步不变。

首次控制器PID3161452于2026-10-16 00:23 UTC提交，三架构均在Docker create前失败：本批新输出挂载目录权限700，Docker无法遍历。容器、模型及科学更新均未开始。创建命令诊断保留原stderr；只将本批根、三架构根及各evaluation根共7个目录改为755，source子树与凭据权限保持。create-only探针随后成功，未启动模型。失败controller/process/state及三组prepare声明另存`launch-repair-20261016/failed-attempts`，恢复入口只接受这个零更新失败状态，不覆盖已有科学输出。

用户指定CUDA5、6；准备期间其他项目在两卡启动。只读采集GPU5的PID3168776以及GPU6的PID3164122、3165961，按GPU UUID、PID、启动时间、UID1047及NVML报告的解释器路径绑定身份。外部`/proc/PID/exe`不可读，记录明确使用`nvidia-smi process_name`，不声称核验了readlink。外部进程不被修改或停止；新未知PID、身份改变、无法核验的进程或不足显存阻止下一阶段。

宿主调度在任何资源线程开始前加载独立冻结修订。仍只允许GPU5/6，维持原本批容器的镜像、源码清单、运行用户、执行命令、设备和输出验证；CPU、全局GPU锁与本批独立lease保留。每个活动/正在启动阶段预留普通Transformer28GiB、Loop20GiB、DiscoLoop20GiB，另为外部保留8GiB余量；外部实际用量加本批全部预留及余量不得超过显卡容量，实际空余亦须容纳新预留及余量。第一批优先分布两卡，随后在余量允许时共卡。此预留用于准入，不是对外部进程未来分配量的保证；共享墙钟不作为独占GPU效率证据。

恢复入口为`scripts/resume_ttcd_one_step_training_launch.py`，资源模块为`scripts/ttcd_one_step_gpu56_resources.py`；持久副本、清单、外部快照、测试收据及当前process记录位于`/ossfs/workspace/llm-memory-editability/ttcd-one-step-training-v1/launch-repair-20261016/`。未来恢复先读本批最新controller-process/state、各架构阶段收据及实际LM Docker，不重跑已开始阶段。正式启动和步数以[结果汇总](../../results.md#ttcd-one-step-training-v1)与实际输出为准。

## 沿用CUDA核验，修复正式训练入口

三架构prepare与真实CUDA preflight全部退出0，原/加速的四损失、全部快状态和100/52/52份参数梯度逐值一致，原完整模型、AdamW与数据RNG继承通过。峰值显存约10.76/6.99/7.26GiB；优化器更新0、正式D查询0，完整收据保留于各架构`training-run/result-preflight.json`及批次`cuda-preflight-verification.json`。

PID3200759随后遇到两类宿主启动错误：TF/Disco的`--out /output`虽与科学配置目录绑定同一宿主路径，`Path.resolve`仍不认为两名称相同，原训练输出隔离断言在优化器/训练循环之前停止；两组`training`目录为空。Loop在下一阶段资源核对时，另一容器恰好退出，Docker top错误未被转换为重新采样资源；其train容器尚未创建。全部0新增θ更新，旧失败W&B记录保留。

新增`scripts/resume_ttcd_one_step_training_stages.py`仅将preflight/train相关输出参数映射到科学配置原有的`/frozen/results/ttcd-one-step-training-v1/{arm}/training-run`，模型代码与配置不变。训练容器名另加`-output-v2`以保留失败容器；只有Docker top明确返回容器已停止/进程不存在的竞态才重采样，其他异常仍停止。9项新增固定镜像契约与Ruff通过。恢复保留prepare/preflight结果哈希，归档失败train声明、空training目录和失败W&B，跳过已通过阶段直接接续train及后续原流水线，不重复CUDA核验。

本次恢复源码与收据位于`launch-stage-repair-20261016/`，清单SHA `3d6205d43898f50a83618653f7a798d478034b634b99e07750267538589d8ba9`；最新宿主PID3225656，入口登记于批次`controller-process.json`和`schedule-amendment.json`。原科学manifest SHA `74f34347e643ad767b279c7571eebea2e9f167909fef7850f930a30c4d51ead5`及全部科学源码/参数未改，原四步训练模型进程继续运行。

## 按真实峰值接管调度，三架构共卡并行

Loop先获GPU6、Disco先获GPU5，二者已经开始更新。初始普通Transformer28GiB的预留加外部用量及Loop20GiB预留，使其在GPU6略超准入总量；实测普通模型完整CUDA核验峰值10.76GiB，故仅把它的宿主预留降至24GiB，Loop/Disco20GiB与8GiB余量保持。科学模型、损失、微批、精度、写入步数与总更新预算不改。

接管先捕获容器/模型PID、源码、GPU、挂载、W&B、当前步数；只停止已核验的旧宿主与其只读Docker监控客户端进程组，两个容器模型各自的进程组不同，不调用Docker stop/kill。新宿主先取得现有两模型的CPU/GPU租约，再继续监控同一容器；截止仍按原容器开始时间加48小时，新监控日志独立保存。Transformer尚未启动的声明另存，随后才提交其正式训练。原始捕获时Loop164步、Disco142步，之后继续增长，没有从检查点重启。

初次接管PID3263565在任何调度变更前因Docker `Mounts`数组顺序变化而停止；只读诊断确认全部Source/Destination/RW成员及其他模型身份完全相同。原捕获与失败日志保留；v2只按Destination规范排序，不删改任何挂载成员，新9项固定镜像契约及Ruff通过。最终入口为`gpu56-sharing-handoff-20261016/mount-order-repair-v2/source/scripts/share_ttcd_one_step_training_gpu56.py --execute`，清单SHA `9181fd81d6df45e121dcf377f6c9fc7ffb98fdc183e37549256e72cba513c2f5`，最新宿主PID3275341。恢复以本批最新controller-process/schedule-amendment中的`host_handoff`为准，前文旧PID/入口仅作历史。

2026-10-16 00:48 UTC实际三架构均为train运行中：DiscoLoop GPU5，普通Transformer与Loop GPU6。Loop容器`2d04a4c8…`/模型PID3227427、Disco容器`f55353ff…`/模型PID3227476及W&B身份接管前后完全保持；普通模型为新提交。当前追踪：[Transformer](https://wandb.ai/zhusq20/llm-memory-editability/runs/76nf7rp1)、[Loop](https://wandb.ai/zhusq20/llm-memory-editability/runs/6laga3kw)、[DiscoLoop](https://wandb.ai/zhusq20/llm-memory-editability/runs/6r8hlxu6)。科学步数和完整资源以批次`submission-verification.json`和各训练progress为准；训练启动不等于终点独立/云端审计或正式组合成绩完成。
