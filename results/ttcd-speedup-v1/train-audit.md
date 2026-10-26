# TTCD训练开销与精确恢复检查

2026-10-15 UTC。检查对象为`/ossfs/workspace/llm-memory-editability/ttcd-successful-base-v1/source-v2`，以冻结源码及已有曲线为证据；没有做GPU profiling、优化器更新、停机或续训。本报告之后所述CPU验证只计算固定输入的前向和反向，不训练模型。

结论：优先删去训练写入中已经不再被使用的后续计算。单纯减日志开销不足以解释或解决stream的主要耗时。当前入口没有精确续训或安全保存信号接口，不能直接把正在运行的模型切换到新实现。

## 1. 已有耗时证据

从两条已有`training/curves.jsonl`分别读取一个不跨验证节点的20更新窗口：

| arm / 更新范围 | 计入训练计时，秒/步 | 墙钟，秒/步 | 训练计时之外，秒/步 |
| --- | ---: | ---: | ---: |
| single 3101→3121 | 1.5380 | 1.6965 | 0.1585 |
| stream 1300→1320 | 4.0315 | 4.2009 | 0.1695 |

这里“训练计时之外”包括采样、trace、Python记录、持久写入及W&B调用，不能全部归因于I/O。即使全部消掉，这两个窗口也分别只省约9.3%和4.0%的墙钟。它们是在共享GPU4下观察到的窗口，不是独占GPU基准。

每1000步另有七池验证与约796–798MiB检查点。该成本在曲线的**下一步**体现，因为当前`wall_seconds`在验证前取值。single的1001/2001/3001步有约31.86/33.55/49.36秒训练计时之外的间隔，stream的1001步约33.37秒。没有证据说这些周期成本都来自存盘。

## 2. 首要加速点：只保留训练写入所需的首轮

冻结`ttcd_scratch.session_write`（254–295行）仅使用`capture['features']`生成快权重。`TTCDReader.forward`在首轮各物理层已填齐这些特征（215–224行），仍继续执行Disco词表通道、第二轮、最终LayerNorm和词表投影（225–231行）；返回的logits被训练写入调用直接丢弃。

因此，**训练且`capture.outer_grad=True`**时，在完整首轮后提前返回，不改变所需的teacher/student特征、快权重或其外层梯度。推理`reader.write`还保存每次写入的teacher_argmax，不能用同一理由删除它的完整读出。正式native/answer/EOS查询也仍须执行完整架构，Disco词表通道不能被候选词限制或删除。

以当前width768、MLP3072、词表52463、23-token packet估算，省去的每个写入样本计算为：第二轮四个fast MLP block、一次Disco双词表投影及最终一次词表投影。宏128事实×每事实4次写入约省`2.83×10^12`次前向稠密矩阵FLOP/更新；这是源码公式估算，**不是实测加速倍数**。原backward成本是2×forward代理，也包含这些没有参与损失的后续计算，不能把代理下降当作实测反向速度收益。

已按后续明确授权新增`ttcd_train_speedup.py`，没有修改原文件。它固定原`ttcd_editing.py` SHA并做AST变换，保留首轮的原运算；其它调用直接调用原函数。原compute四个键保留，只扣除没执行的前向估算及对应2×代理。

一个重要数值细节：直接early-return在CPU BF16下虽fast/loss逐位一致，tied embedding梯度末位会变化。原因对应被删除decoder原本建立的autocast缓存共享cast图。保留零行`F.linear(empty[0,width], wte.weight)`后，既没有词表矩阵乘法，又保留该cast图；D8/Loop/Disco、K1/K8、FP64/FP32/CPU BF16的全部fast/loss/梯度/RNG均逐位一致。21项CPU检查通过；没有调用`optimizer.step`。真实CUDA、真实宏512/128、原BF16和micro64/32的逐值核验仍须通过，才能用于正式执行。

## 3. 必需计算与其它可省开销

- **配方必需**：stream的8条事实串行依赖同一A，每条4次写入，前后A保持完整BPTT。single与stream每宏都处理128事实、512次样本级写入，但当前micro32使single有16次batched capture调用，stream有128次。不能通过少写几次、detach历史、并行打散同一流或缩短10k预算来加速。
- **没有activation checkpoint重算**：冻结reader/session_write/backward_objective没有`torch.utils.checkpoint`调用。磁盘checkpoint与激活重计算不是同一回事。当前stream较慢与更多小batch调用/同步一致，但没有profiling不能量化归因比例。
- **有限值检查同步**：`validate_fast`逐物理层执行GPU `isfinite(...).all()`并转Python布尔。按当前四层、每宏计算，single约160次、stream约1056次这样的逐层检查，另有capture位置检查。可把同一次检查的四个布尔一次性传回CPU，仍逐层判断并保留同一失败边界；不应直接删掉全部有限值检查。
- **损失标量同步**：每宏约8个native、4个reader、8个fast损失标量分别转`float`。可先保存detach标量，最后一次传回CPU并按原顺序做Python加权记录；backward顺序不变。单独移除结尾`cuda.synchronize`作用有限，且必须重新保证计时准确。
- **可缓存只读数据**：每步读取并解码`training-split.json`，重新构建`world.support(phi)`，生成相同尺寸teacher/student mask；student mask还在各层重复构建。可以缓存已核验的只读数组和mask，维持原NumPy采样顺序、数组顺序及RNG调用次数。不能删掉现在未进入损失的old_atoms/old_chains随机抽样，它们仍推进随机流并进入trace。
- **记录与I/O**：每步curves flush、progress临时文件+rename、一次W&B `run.log`确实存在；没有每步`fsync`或强制云端finish。模型/输入全文哈希发生在入口、周期checkpoint和终点，不是每步模型哈希。可异步保留全部记录，但需有序队列、失败传播、结束flush及恢复边界；收益受上述约0.16–0.17秒/步之外开销约束。

正式评测另有可观重复工作：`save_fast/load_fast`为每个A反复CPU复制并哈希约278MB骨干；一次1313状态保存就约365GB骨干哈希输入。冻结reader可考虑一次绑定骨干digest、每状态检查不可变版本、起终点实际重哈希，保留所有A与审计；这是另一个待严格核验的运行修订。`evaluate_queries`两次forward都投影两个位置而每次只用一个位置，可省最终未使用投影，但改变GEMM形状须验证全部token/EOS/rank/score。不能减少查询池、状态、顺序、答案/EOS评分或独立重放覆盖。

## 4. 当前检查点与精确恢复缺口

检查时最新周期节点：single3000（SHA`86674453ac2458140cf084ed3a9c7539a053c699924ea32b363625756b18dcae`）、stream1000（SHA`3e734114d22e71ee5ce9a6d3091b3e97047c3bc7f4641cf242403c2ca2a41f8f`）；它们落后于上述live进度。

冻结`verified_base.train`的checkpoint已保存：model与digest、完整AdamW状态、phase/step/identity/foundation、native sampler的size/Generator状态/remaining、correction RandomState的完整MT状态、CPU/CUDA torch RNG、曝光counts、trace digest、readout及training_seconds。BF16使用autocast，没有GradScaler状态。A只存在于当前宏更新的计算图中，宏更新结束后丢弃，因此恢复边界应是**一个完整optimizer更新结束之后、下一次采样之前**，而不是中间事实或microbatch。

但当前`train`没有resume参数、信号处理器或barrier；入口要求新建training目录，curves也以`x`打开。现有loader只接受完整终点，不能拿中途checkpoint冒充终点。仅更换host controller只接管监控，不会替换已经运行的Python函数。向当前进程发送“保存信号”并没有已实现的保存语义。

还有两个必须补齐的恢复条件：

1. SHA只存`trace.hexdigest()`，没有可续接的SHA内部状态。必须从该phase原起点的两个sampler状态重新生成1…s的batch字节，核对saved digest，再续接原hash对象；同时核对重生成sampler状态与saved状态一致。不能把digest字符串重新hash当作续接。
2. 创建reader/optimizer后再恢复saved CPU/CUDA RNG；完整加载Adam动量与其继承自common的step计数；恢复counts及累计时间，从`s+1`执行。新实现还需保留同一sampler部分尾batch和原micro归约顺序。

一个新安装的安全barrier可以在完整更新、现有记录完成、尚未采样下一步时保存上述状态，原子写receipt后才退出；原目录/失败/已完成记录不覆盖，新segment记录源码与运行修订，并核对第一批trace及全部终点审计。**当前活进程没有这个hook**，不能承诺即时无丢步切换。从落后checkpoint重算会重复已有更新与曝光，必须单独记录并获明确接受，不能称精确live接续。

## 5. 执行判断

先验证训练capture剪枝的真实CUDA数值一致性；正式评测尚未开始，可以独立验证评测加速，而不处理活训练恢复。当前训练若要采用新路径，先确定可审计的恢复/切换方式；不能为追求速度削减步数、内层4次、K8连接、宏batch、精度、TF32设置或完整评测覆盖。

冻结主要来源SHA：verified_base `6188e83844ffde27350dd954e7ca0a231b84614bb618f5ab9becfa23f149817f`；ttcd_editing `391790498d6412081baa06d73d2105c1a8a2c78c0cca0256a11d201e2e72d28f`；ttcd_scratch `e423b8d7df23b5bee31ee6bc147caef89f4dbe784e4da946cfeb7073353acbf0`；ttcd_continual `3f13cc364f44c41702d0b57e6bda8afd1d541efbb56e7bfae5e455ea52ab7d97`。
