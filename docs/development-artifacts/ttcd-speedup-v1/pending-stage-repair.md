# TTCD后续阶段启动与空闲GPU调度修订

2026-10-15 23:47 UTC核对。一步评测三架构均等待各自原stream10000训练及独立重载；merge三架构prepare均成功，等待对应一步评测本地与云端复核。两批正式评测均未开始，没有组合或保持成绩。依赖按架构独立推进，不要求三个模型同时就绪。[实际状态快照](../../../results/ttcd-speedup-v1/runtime-status-20261015.json)记录控制器、模型容器、GPU进程和进度。

原DiscoLoop single已完成10000步，train-audit首次启动在模型执行前因挂载权限退出126。此前复制修订仅拦截helpers.docker，漏掉修复控制器的subprocess.run启动路径。本次以该控制器专属代理接入相同create→copy→start流程，不替换全局subprocess。恢复后又在加载模型前暴露缺失ttcd_double_new_eval模块：原父批source-v2早于补充评测源码，扩展无条件导入它。缺失依赖、原退出、日志、运行收据及失败容器全部保留；新入口仅在父包缺少模块时加载已冻结补充模块，已有模块不替换，数值实现不改。

最终固定镜像49项CPU检查及Ruff通过；实际老父批source-v2的完整入口帮助解析通过，模型查询和更新均为0。原扩展损失、梯度、快状态与重放回归保留；仅补足模块装载，不重复真实检查点训练。新runtime清单SHA为`d10365ad67b980a2ba69c46f1f8dd9c4bc4ac93cf6d92f5b9bc4b891653d74fa`。[工程清单](../../../results/ttcd-speedup-v1/pending-stage-repair-20261015/source-manifest.json)、[老父入口收据](../../../results/ttcd-speedup-v1/pending-stage-repair-20261015/import-entry-receipt.json)。

Loop共同读取10000步已完成，后续独立重载原先只等待被其他项目占用的GPU9。本次将普通Transformer/Loop后续未启动阶段改为原卡优先、其余实际独占空闲卡回退；保持原CPU集合、内存、GPU容量/空闲阈值、显式LM Docker、全局排他锁和原等待截止时间。已有训练不迁移。Loop当前仍没有可用整卡，等待截止时间沿原共同训练退出锚点，不重新计时。

核对期间，其他项目又在GPU4启动两个CUDA进程，原GPU4准入检查因此使宿主监看退出，模型容器未停。本次恢复只接管仍在运行的原模型；外部进程出现时继续监看已有模型，新GPU4阶段继续等待授权cohort准入，禁止把未知进程纳入共享名单。single复核现已修复并排队，当前没有重新执行。GPU4原连续训练5362/10000、GPU8普通Transformer连续训练816/10000，两者容器ID及模型PID均与交接前一致。其余GPU均被其他项目占用，实际独占空闲卡0。

最新架构宿主PID2976601，恢复入口为`ttcd-speedup-v1/pending-stage-repair-20261015/architecture/`；GPU4父组宿主PID3001499，恢复入口为`ttcd-speedup-v1/pending-stage-dependency-repair-v2-20261015/parent/`。未执行的依赖修订初稿及更早失败也保留。恢复先读相应批最新controller-process-speedup.json、schedule-amendment.json及本次snapshot/claim/日志，不重启旧控制器。一步和merge原控制器PID2787232/2854789继续等待，科学配置、事实、监督、每事实一步与新增训练0均不变。
