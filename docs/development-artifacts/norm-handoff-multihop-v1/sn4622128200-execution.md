# R/N/A/C本机CUDA1、2执行记录

**2026-10-17 02:06 CDT最新调度：用户告知GPU4空闲，已将候选卡扩为4、1、2。** 初查GPU4显存0、无计算进程；正式切换前复查出现VLLM PID317671，显存87217 MiB，因此未占用GPU4。现控制器PID325899先在三张候选卡中等待实际可用卡做F重放，再按空闲卡动态分配四臂；GPU1/2仍须先通过旧队列训练、独立审计和云端核验。旧等待控制器PID290862已停止，当时只有CPU契约通过，无GPU阶段claim、无模型运行被中断；原记录存于`gpu4-scheduling-history/`。新增调度相关33项测试、Ruff及格式检查通过。35项CPU契约复用，原包/科学配方/预算不变。当前GPU预检和正式训练均未开始。

本次恢复优先读`gpu4-amendment.json`、`gpu4-submission.json`、`gpu4-controller-process.json`、`gpu4-controller.log`及`run/controller-state.json`、`run/waiting/`；新入口为哈希绑定的`launcher-source/scripts/resume_norm_handoff_gpu4.py`。原`launch-request.json`保留初次双卡申请，当前显卡范围由本次amendment更新；不得重复运行任一提交入口。下述固定两路分配为原调度历史。

2026-10-17 CDT，用户要求运行已交付的四臂实验，随后明确指定CUDA1、2。已启动后台控制器；旧学习率批次正在使用这两张卡，新任务等待对应旧队列训练、独立审计和云端核验完成，再核查实际显存、计算进程及两个Docker context的GPU预留。GPU1安排R→A，GPU2安排N→C；不抢占、不共卡，不扩展科学预算。

每臂300000步，共1200000新增正式更新；F仅复用。CPU模型/数据契约→F全量r1–4重放→全部四臂预检、32步短训及原轮数/r1–4独立重载→四臂正式训练及终点审计→固定激活诊断和云端报告。任何失败均保留并停止相应流程，不自动重试。四臂工程门槛全部通过前，不启动任何正式训练。

原搬机包SHA256及内部文件/基线/数据哈希核验通过。新增接续逻辑与原启动器共26项无模型测试、Ruff及格式检查通过。GPU检查和正式训练的实际状态以以下收据为准；提交队列不代表通过CUDA预检或获得科学结果。

**2026-10-17 01:54 CDT核对：本机无GPU容器的35项模型/启动器契约及完整数据契约通过，退出码0。** 控制器PID290862仍存活，正在等待GPU1的旧队列释放后执行F重放。当前GPU预检、短训及正式训练均未开始，新增正式更新0。测试中只读源码目录无法写pytest缓存的警告不影响通过结果，原日志保留。

原镜像`sha256:9643a7d6aeda1b4c9ac2ac50c2d84474b082d649a87e423bc6e0bdc0994097ee`本机缺失。本次复用本项目已用于学习率批次的重建镜像`sha256:27b3ab677fb92aaf4e4cd953e3a87e953c4b164a0d4d4c4785dc9779851c11d2`；实测PyTorch2.8.0+cu128、CUDA12.8、Transformers4.57.3、NumPy2.2.6、W&B0.30.0、huggingface-hub0.36.0。主要依赖版本相同不代表镜像字节或跨设备训练逐位一致。RTX PRO 6000 Blackwell上的F答案/EOS逐token及NLL原容差重放仍必须通过，否则停止，不补训F、不放宽评分。原冻结包中的image字段保留历史值，实际镜像绑定在runtime-amendment、launch-request、阶段command及W&B的WANDB_DOCKER中。

持久目录为`/mnt/disk2_from_server2/siqizhu4/lm-memory-persistent/workspace/llm-memory-editability/norm-handoff-multihop-v1/sn4622128200`；仓库入口为[本机材料与状态](../../../results/norm-handoff-multihop-v1/sn4622128200)。它在NFS持久盘上，原机`/ossfs/workspace`本机不存在。新复制包保持哈希一致，仅新目录/文件设置Docker可遍历权限；原包及私有凭据不改。

恢复先读该目录下`submission-process.json`、`runtime-amendment.json`、`controller.log`、`run/controller-process.json`、`run/controller-state.json`及`run/waiting/`，再核对`run/execution/*/*/command.json`、`exit.json`和`docker --context lm-memory`实际容器。启动器执行副本在`launcher-source/scripts/launch_norm_handoff.py`，由amendment绑定哈希；冻结科学源码在`bundle/source`。不要重跑提交入口；已有进程或阶段claim不能自动重试。旧队列收据缺失、失败或审计不通过时不会自动接管显卡。

W&B在每条正式训练启动时在线初始化，链接届时生成：[项目入口](https://wandb.ai/zhusq20/llm-memory-editability)。`completion-manifest.json`仅在四臂训练、全部重载、固定诊断及云端回读都通过后生成。
