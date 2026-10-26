# 普通Q adapter300k对照：执行记录

## Material Passport

- ID：`loop-query-adapter-development-v1`；类型：执行状态，不是终点科学结果。
- 本批：[协议](protocol.md)、[配置](../../../configs/loop-query-adapter-development-v1.json)。
- 父比较：[独立32维状态方案](../loop-query-control-development-v1/execution.md)，旧训练继续，不重新提交。

已冻结并正式启动新增一条300000步对照，当前已核实通过1000步，W&B：[c1p3gwzy](https://wandb.ai/zhusq20/llm-memory-editability/runs/c1p3gwzy)。普通共享Q adapter为768→48→768 tanh，每个物理块从当前LN(hidden)重算，仅加Q；不保存独立状态、不读取额外原输入旁路，无DiscoLoop/KV/辅助损失。新增73,728参数，控制分支74,752，相差1,024（新增部分1.37%）；整模69,504,768与69,505,792。

固定镜像10项CPU契约与Ruff check/format通过；冻结容器CPU契约退出码0，全尺寸CUDA七项检查均通过：起点输出、全部backbone梯度逐值一致，adapter上投影非零有限梯度，以及capture后权重/Adam/CPU与CUDA RNG恢复。预检查0存活更新；没有额外短科学训练。

实际初始checkpoint、world、panels逐字节与父freeze相同；五个共同训练/评分实现文件也匹配父哈希，所有训练及评价配置逐项核对。第0步全七池19,004题的rows/answer/stop/NLL数组逐值一致，原始预测文件哈希和核验收据保留为持久根的`paired-start-audit.json`。正式固定300k后，各自七池独立重载、W&B终点完整JSON通过，再核对实际曝光并写控制−adapter配对差值。当前无终点结果。

资源：GPU4与登记父容器并行；adapter CPU84–87、父80–83，各96GiB，GPU至少保留16GB空闲，未知GPU PID/其它owner拒绝。不改变父科学源码/配方，外部共享时段记录于两批schedule/co-location文件。共享阶段耗时不作独占性能比较。近等参数比较仍包含输入来源、归一化、更新频率和计算差异，不唯一归因于独立状态。

持久根：`/ossfs/workspace/llm-memory-editability/loop-query-adapter-development-v1`。恢复先读controller-process/state、freeze、paired-control、resource-admission和实际LM容器；入口`scripts/launch_loop_query_adapter.py`。全部stdout、退出码、GPU资源心跳和W&B记录保留，失败不自动重试，禁止重复提交。
