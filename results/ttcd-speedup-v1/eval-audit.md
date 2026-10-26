# TTCD 评测与审计提速：只读检查

2026-10-15。本次只检查源码、冻结面板和小型运行记录；没有启动训练、评测、GPU、性能探测或改变已有产物。以下是静态工作量及实施建议，不是已测得的加速比。

优先减少重复冻结模型哈希，然后移除双新事实评测的跨分支等待。保持 FP32 正式读取、batch128、原写入步骤、完整原始状态与查询评分。独立重放仍须覆盖每个状态和每次写入转换。

## 已核对的工作量

当前正式源与工作区对应文件逐字节相同：

| 文件 | SHA256 |
|---|---|
| `ttcd_editing.py` | `391790498d6412081baa06d73d2105c1a8a2c78c0cca0256a11d201e2e72d28f` |
| `ttcd_continual_eval.py` | `b1202073012c3b09b3ca09ad0fa06eb4d96167728bb2282c2d9ead69241d260e` |
| `ttcd_double_new_eval.py` | `baa79163fe4261b14a320e34f57dafb93dddb4485a8689116d319b1d6f428a8b` |

父源为 `/ossfs/workspace/llm-memory-editability/ttcd-successful-base-v1/source-v2`，双新源为相邻 `ttcd-double-new-eval-v1/source`。双新面板文件 SHA256 必须保持 `ade07fa92e528f1b19bbc10965d984f747ed7f7daf66f5eafa254971ac7d08f6`，原面板必须保持 `ac6082dcb857fab85fc106d4d3d39a7a3adb67de33976edeb13c4a1bd201c081`。

| 每个 arm 的正式矩阵 | 状态数 | 有查询状态 | 查询数 / 一遍 |
|---|---:|---:|---:|
| 原跨样本与再次更正 | 1,313 | 1,121 | 441,002 |
| 双新事实 | 1,089 | 1,089 | 477,880 |

这是当前原图/原面板的计数。D 每事实实际有6、7或8条，U 为64原子加64组合；不能套用旧小图366,900查询或把D覆盖补成统一8条。两个面板正式评测与独立审计合计每 arm 重放1,837,764条查询。

四层 Loop/Disco 的一个 FP32 快状态为37,748,736字节（36MiB）；每 arm 两个面板保留全部 raw fast 至少84.445GiB。D8有八个独立矩阵，以上状态体积翻倍。两个旧 arm 加新 Loop 与 D8 的 raw fast 合计约422.227GiB，未计JSON、训练检查点及文件格式开销。

## 建议依次实施

1. **缓存独立进程已核验的冻结模型哈希。** `save_fast`、`load_fast` 每次都对整个 `reader.backbone.state_dict()` 做 `tensor_digest`，逐张GPU参数转CPU。原 evaluate 约1,315次、原 audit 2,561次、双新 evaluate 与 audit 各2,179次，合计8,234次/arm。按69,431,040个FP32参数计算，仅这一项至少扫描和传输2,129.73GiB/arm。开始时完整核验一次；每次使用缓存前验证参数/缓冲区对象、shape、dtype、device及版本未变；结束时再次完整哈希。每个 fast payload 仍核对原 `backbone_digest`，不能缓存 fast digest 或跳过其独立读取。新的优化适配源码及入口须另存哈希并写入执行修订证明，不能只在旧源旁静默替换函数后宣称执行字节未变。

2. **复用已经加载的前态，减少重复读盘。** 双新 evaluate 目前对每个非baseline状态都 `load_fast(previous)`，即使前态就是刚生成的状态。保持状态计划与读写顺序，连续main流可直接使用当前GPU fast；32个single供pair诊断可用约1.125GiB缓存或有界缓存。所有1,089个原始fast文件仍保存。audit必须独立加载每个当前状态一次，但其经完整验证的saved fast可以作为下一状态的前态缓存；不得用重建结果替代从文件读取的前态。原审计当前1,313次当前态加载加1,248次前态加载可明显缩减；每轮A8分叉保留一个经验证的锚点，防止错误地从A32开始再次更正。

3. **保存时只创建一次CPU快照。** `save_fast` 当前先为fast digest做一次D2H，再为payload做一次D2H。先构造同一个`fast_cpu`，让原digest函数和原payload都使用它，保留schema、dtype、逐元素字节、provenance及全部fast文件。两个面板共2,402次保存可少传约84.445GiB/arm，D8翻倍。不要改写、压缩或覆盖已保存文件。

4. **合并预测结果的小传输。** `evaluate_queries` 每个batch把四个int64字段和三个float字段分别`.cpu()`，共7次。可分别stack四个整数列、三个原float dtype列，各传CPU一次，再拆回原字段。模型前向、batch划分、answer反馈位置、EOS前向、排名与margin算术完全保持。禁止将所有字段混成同一dtype。原始`answer/stop/rows/old_answers/new_answers/old_rank/new_rank/new_minus_old/old_margin/new_margin`全部保留。

上述缓存有明确生命周期：只能绑定当前已审计checkpoint、device和单个独立阶段进程；模型重载、参数版本/对象变化或身份改变即失效。始末完整哈希及每个read_state已有的fast前后完整digest仍保留。`validate_fast`逐前向有限值扫描也会触发GPU同步，但缓存这一检查需要额外处理版本、别名及in-place写入；本轮不把它列为优先变更。

## 调度的关键路径

父控制器`repair_ttcd_verified_preflight.run_phase`已经对每个 arm 独立执行train→train-audit→calibrate→evaluate→audit→cloud-audit，两个arm各在自己的线程中前进；不能误判为原评测也在等另一arm。只读快照时single为step3127、stream为step1322，均仍训练中，尚无本轮正式评测结论。

双新控制器有两层额外屏障：

- `launch_ttcd_double_new_eval.wait_dependencies`遍历两个ARMS，只要另一个未完成就阻塞已完成arm；`run_arm`又强制生成并反复核对两份parent receipt。
- `share_ttcd_gpu4.execute_supplement`在启动两个supplement worker之前等待父控制器的整个GPU4 cohort锁；父控制器直到两个原arm全部结束才释放该锁。

将依赖改成**每arm自己的**common gate/checkpoint、最终训练checkpoint、train-audit、calibration、原evaluate/audit/cloud完整成功且SHA固定，即可让先完成arm的双新评测与另一arm剩余训练重叠。无需等待另一arm的checkpoint，也不改变查询、面板、标签或训练。仍保留两arm最终共同完成的总汇总门槛。若本arm任何数值审计失败，就保留失败并停止该arm，不能绕过。

GPU资源继续只使用已授权的GPU4共享cohort，CPU父single24–27/stream28–31、双新single32–35/stream36–39不重叠；每阶段继续验证容器源码、入口、UID、GPU UUID、未知PID拒绝及至少24GiB空闲显存。8/9属于新架构正式队列，不在其训练间隙插队。并发不保证更短总耗时，可能与剩余stream训练争用GPU；先解除依赖，再由现有资源检查决定是否进入，不宣称独占GPU耗时。

cloud-audit本身只读取W&B及本地结果JSON，可以用原CLI的`--device cpu`并不请求GPU，保留全部云端终点/结果SHA核对。父CLI使用cuda时会在cloud-audit前执行无关CUDA探测；CPU云核验可避开这一点，并提前释放GPU任务槽。

## 独立审计分片：后续可用，当前非首选

每个rotation独立，适合按完整rotation分片。原面板每轮main32+revision8；双新每轮main32；baseline/single及双新pair诊断另成固定分片。跨分片的parent fast仍从同一不可变raw文件读取，baseline及single依赖不得重建替代。当前没有额外空闲GPU，先做上述缓存和依赖修订更直接；无需启动多个CPU大模型重放任务来代替GPU。

若以后分片，每个worker须为独立新进程，使用同checkpoint、固定镜像、原FP32及batch128，原始目录只读，receipt写新的修订目录。reducer必须核对：分片源码/输入/raw-state清单SHA一致；状态key不重不漏且等于完整state_plan；1,312/1,088次写转换全覆盖；答案、EOS、标签、排名精确相等，分数沿用原rtol/atol=1e-5；历史写入记录与fast digest一致；原面板1,121个readout及双新1,089个readout、查询数完整；用全部重放readout重新计算原summary后才生成总审计证明。不能仅相加分片accuracy或把分片完成当作全局独立审计通过。

训练检查点审计另有一个小重复：`audit_training`同时重放终点`checkpoint-010000.pt`和`parent.pt`。可在同一独立进程按已完整strict-load并验证的model digest复用native readout，仍逐文件SHA、模型SHA及saved readout比较。它只省一次终点七池读取，优先级低于数千次整模型哈希。

所有建议都保留原快状态、记录、写转换、覆盖率、原始token/分数和云端核验；不增加训练、短训或改变科学配方。实际提速仍待实施后的正式阶段耗时记录，不能从静态传输量直接推算加速倍数。
