# 统一八臂实际执行

2026-10-16，按用户指定物理cuda9。前两套同题准备入口已停止，正式更新均0；其源码、工程短训和中断记录保留。当前只运行`loop-hop-unified-v1`，不恢复旧入口。工作区旧入口增加被替代提示，统一入口另持有跨批次执行锁，防止重复提交。

## 已验证与启动

- 固定镜像18项CPU契约、Ruff与格式检查通过。
- 八臂全尺寸CUDA预检通过，包括关闭通道时原生前向／全部梯度逐值一致、图捕获后的参数／Adam状态／RNG恢复及真实计算更新。
- 八条32步工程短训完成，共256工程更新，全部丢弃；八份原轮数独立重载、八份测试r1–4全网格独立逐题重放通过。
- 2026-10-16 18:01 CDT，统一八条300k正式训练已全部启动；全部物理GPU为9。八个W&B运行在科学更新前初始化，已观察首条轨迹超过1000步。尚无终点科学结论。

控制器主机PID为317114。实际恢复入口在`/ossfs/workspace/llm-memory-editability/loop-hop-unified-v1`的controller-process.json、controller-state.json、freeze.json及各执行阶段exit.json；主机PID不能直接在沙箱PID命名空间判断。容器内cuda0对应物理cuda9。八任务只与本批共享同一GPU，各进程Torch allocator上限8%，同卡墙钟不能作为独占吞吐比较。

## 冻结身份与数据

封存清单SHA256：`71aa3f424bff94fc777a88c13629333e7780a63f3b948032a962a8a8b66edb95`，共189个源码／配置／输入文件。数据逐字节复用已封存40k版本；world=`e132eaf6bba5786bbf05f6f36c1b7383f698e93ee0dee313f568f6d50a77026f`，panels=`c991cebeeeec05639be87d981cd20031016f47965c43fed8bf5cc55579c3b468`。没有与25k版本合并。

全部六个L1模型初值参数哈希相同：`39a487823c05311a223db99c3d0e2df3e7a1752498e261337f4298abe0cfa7b0`；L2两架构共同初值为`95e4c6372d44514435affae90f13877406afd7846e7fb13153c9ee48722599a0`。L1唯一参数8,653,824，L2为15,741,696；DiscoLoop通道无额外可学习参数，但有额外计算。固定300k，正式总预算2400000步，不自动追加。

## 跟踪

| 条件 | W&B |
| --- | --- |
| 普通Loop 1×2 | [lh034bfc00](https://wandb.ai/zhusq20/llm-memory-editability/runs/lh034bfc00) |
| DiscoLoop 1×2 | [lh034bfc01](https://wandb.ai/zhusq20/llm-memory-editability/runs/lh034bfc01) |
| 普通Loop 1×3 | [lh034bfc02](https://wandb.ai/zhusq20/llm-memory-editability/runs/lh034bfc02) |
| DiscoLoop 1×3 | [lh034bfc03](https://wandb.ai/zhusq20/llm-memory-editability/runs/lh034bfc03) |
| 普通Loop 1×4 | [lh034bfc04](https://wandb.ai/zhusq20/llm-memory-editability/runs/lh034bfc04) |
| DiscoLoop 1×4 | [lh034bfc05](https://wandb.ai/zhusq20/llm-memory-editability/runs/lh034bfc05) |
| 普通Loop 2×2 | [lh034bfc06](https://wandb.ai/zhusq20/llm-memory-editability/runs/lh034bfc06) |
| DiscoLoop 2×2 | [lh034bfc07](https://wandb.ai/zhusq20/llm-memory-editability/runs/lh034bfc07) |

终点后自动执行原轮数审计、全r网格、新容器重放及云端完整JSON核验。结果以实际运行状态为准；启动与工程检查通过不代替终点完成。

[统一协议](protocol.md)、[历史去重及合并](history-review.md)、[本机统一收据](../../../results/loop-hop-unified-v1/unification-receipt.json)。

## 2026-10-17训练中核对

00:45 UTC宿主核对八个训练容器与PID317114控制器均正常，物理GPU9占用18194MiB、利用率100%。00:47 UTC读取八臂进度69k–152k／各300k，没有完成的正式终点。

统一64k节点、各自训练轮数下，图B原子99.96–100%，两／三／四跳0–0.20%；必要原子前提覆盖99.82–100%，条件组合率仍接近零。图A训练组合诊断99.60–100%，未见关系模板组合0.06–0.66%。当前未观察到DiscoLoop优势或增加轮／层的可靠组合收益，可迁移组合操作尚未建立，不能定位表示能力上限。事后错误分布与完整逐臂数值见[初步报告](../../../results/loop-hop-unified-v1/preliminary-20261017T0045Z.md)。

本次只读取已保存曲线／预测，CPU复算160项通过，没有新增GPU评价、科学更新、调度或冻结配置修改。正式终点测试r1–4、独立模型重载及云端核验仍待完成。初步读数不替代预定终点，也不按中间成绩提前停训或追加预算。

## 2026-10-17代码与数据排查

按用户要求与成功两跳实际配方逐项对照：189项原冻结文件哈希不变，新旧五份模型／训练核心源码相同；独立原子字典复核73000条训练／评价记录标签全对。八个64k模型在纯CPU容器严格重载，152项显式前向、因果、位置与padding检查通过，参数哈希与只读性通过。额外图A诊断用同一2048条“模板已见、问题未见”两跳题，八组仅0–6题答对；故原ID模板留出不是充分解释。

旧成功配方为4×2、273600两跳题、边级原子留出及52463词表，本批还改变了图结构、任务长度混合、支持量和词表。旧64k的OO也近零，128k后才升高；当前设计没有保留成功4×2新任务锚点，不能单变量归因层／轮。完整比较与学习曲线见[报告](../../../results/loop-hop-unified-v1/code-data-audit-20261017/report.md)。

仅使用固定镜像的4CPU／12GiB无网络容器，未暴露GPU；首次私有目录bind失败未执行模型，改用Docker API复制脚本／输出，未放宽原权限。CPU诊断退出0、结果保留、临时容器已删除。正式更新和追加GPU评价均0；八条原训练继续，正式终点全r网格与独立BF16审计仍待完成。
