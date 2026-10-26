# 双门多跳比较执行记录

2026-10-17 CDT，用户明确授权编写代码、检查通过后利用空闲GPU并行运行五组；可复用的固定DiscoLoop不重跑，先完成多跳再决定新增知识实验。

固定镜像CPU科学契约43项、Ruff检查和格式检查通过，收据见`results/dual-gate-multihop-v1/{contracts-final.xml,cpu-checks.json}`。模型、训练/评价适配器、控制器和报告为新文件，旧运行入口与冻结源码未改。基线复用核验passed，包括F完整300k、两类独立重载、关键文件哈希、七份科学核心源码、数据与全部训练配方、原镜像；F新增更新0。

控制器提交PID2546706，提交收据/日志在`results/dual-gate-multihop-v1/`。本批持久目录`/ossfs/workspace/llm-memory-editability/dual-gate-multihop-v1`；提交时先封存source/input，再固定镜像CPU检查、四组CUDA/工程短训/两类重放，最后正式训练。刚提交不代表已开始科学更新。恢复先读取持久目录`controller-process.json`、`controller-state.json`、`freeze.json`、`baseline-reuse.json`及实际Docker，不重复提交。

资源核对时物理GPU9完全空闲，其余九张有外部/其它实验占用。调度器检查全部十卡，优先各空闲卡独立运行；只在本批已准入且无外部任务的卡上有界共卡，最多四条、每进程16% allocator上限。外部任务不停止、不抢占；每阶段独立2CPU/16GiB容器，LM专用Docker和原固定镜像不变。共享墙钟不作独占性能证据。CPU检查运行时原训练容器保留。

新增P/G/D/V各300000步，合计1200000科学更新。32步短训共128，另四次真实预检更新恢复模型/Adam/RNG后丢弃。没有新事实写入、跨图确认或后续网格进入队列。原F父批云端summary调用失败记录保留，本批五条件总报告将只读绑定原本地结果并上传/下载完整JSON校验，不改写父批状态。

W&B项目：[zhusq20/llm-memory-editability](https://wandb.ai/zhusq20/llm-memory-editability)，组名`dual-gate-multihop-v1`。各科学运行在更新前初始化并写`runs/<arm>/tracking.json`；终点逐臂新容器原轮数与r1–4全网格重放，完整canonical JSON云端SHA核验。最终五条件报告另外验证曝光数组与F逐值相同，生成`comparison.json`、`report.md`与`report-cloud-audit.json`。

当前无新终点结论，GPU工程检查和正式启动须按实际文件补录。

首次四组预检图捕获反向失败，非零学习率正式更新0；日志在原execution、失败与原源码保持。释放对照检查仍存活的计算图引用并补齐capturable AdamW后，独立GPU9 scalar探测12项全尺寸检查通过、退出0，工程1更新全部恢复。修订只影响工程检查生命周期与原生已有的优化器图捕获选项，不改变科学配方。新修订入口使用独立`attempts/preflight-repair-v2`，原失败不覆盖；恢复必须先读`current-attempt.json`。CPU修订契约再次通过后接续，不启动新增知识。

v2首次创建容器因新建`attempts`目录0700不能被NFS Docker遍历而失败，未创建容器、模型执行0。只把本批新目录改为0755，封存失败command/state，冻结v2源码及199项文件哈希保持不变。单次部署恢复脚本`resume_dual_gate_transport.py`持有原批及v2执行锁、核对没有任何科学train claim/已有容器后接续；恢复PID2567696，详情在v2的`transport-amendment.json`及`controller-process-transport.json`。原失败控制器PID2562501记录保留，勿再执行提交入口。

2026-10-17 00:46 CDT核对：四臂CUDA预检12项全部passed，四条32步短训及原轮数独立重载全部通过，正在做推理r1–4工程网格/重放，尚未开始正式训练。四条实际GPU均9，仅本批并行。活跃执行目录为`/ossfs/workspace/llm-memory-editability/dual-gate-multihop-v1/attempts/preflight-repair-v2`，仓库`results/dual-gate-multihop-v1/current`链接到该目录。冻结清单SHA256=`ca3837176c4d40ea8b5432889cbde00e92bb4ed762576ff46cc58d7ae1af6b15`，199项；[CUDA收据](../../../results/dual-gate-multihop-v1/cuda-preflight-summary.json)保留逐项检查。

随后四组全部工程网格及独立重放通过，`engineering-complete.json`为passed，控制器进入`scientific_training`，四个train容器均实际运行在GPU9，W&B均已在科学更新前初始化。每组目标300000步，不是终点已完成；步数、损失和有限梯度读取各自`runs/<arm>/status.json`。以下运行链接已由实际tracking.json核对：

|条件|W&B|
|---|---|
|P 原文学习门|[lh608e0400](https://wandb.ai/zhusq20/llm-memory-editability/runs/lh608e0400)|
|G 标量注入门|[lh608e0401](https://wandb.ai/zhusq20/llm-memory-editability/runs/lh608e0401)|
|D 双门|[lh608e0402](https://wandb.ai/zhusq20/llm-memory-editability/runs/lh608e0402)|
|V 同参数注入对照|[lh608e0403](https://wandb.ai/zhusq20/llm-memory-editability/runs/lh608e0403)|

正式终点、独立重载和云端核验仍待完成，无新科学结论。工程/部署失败均已保留，本次没有科学训练重试。

2026-10-17 00:48 CDT启动核验：P/G/D/V均已实际达到至少1000/300000科学步，损失与梯度有限；LM Docker实际四个对应train容器仍运行，W&B链接与运行身份一致。四臂全部工程门槛及短训新容器重载已核对，收据见[启动验证](../../../results/dual-gate-multihop-v1/launch-verification.json)。当前只继续本批既定训练及终点审计/云端验证，不追加新任务。
