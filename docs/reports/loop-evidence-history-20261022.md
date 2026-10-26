# Loop证据与入口历史（2026-10-22整理）

这是按原文保存的查证档案，不维护当前研究方向或实时队列。正常阅读从[Loop当前摘要](../loop-roadmap.md)和[按问题查重](../loop-history.md)开始。

正文保留旧结论、完整表格、负结果、失败、预算与核验状态；只调整搬迁后的相对链接。文中的“当前”“待启动”“下一步”均属于各段记录时点，不能据此重启任务或推断现在仍未完成。发生冲突时，科学判读先查后续证据，执行先查持久修订、最新控制器与实际容器。本次未重新执行模型或核验云端。

- [科学结果原文](#evidence-records)：从结果汇总移入的Loop架构条目；原结果页锚点仍可用。
- [整理前路线图](#roadmap-history)：保存当时的完整证据与规划，包含已被替代的调度叙述。
- [入口历史](#entry-history)：从AGENTS移入的Loop启动与恢复历史。

<a id="evidence-records"></a>
## 科学结果原文

<a id="loop-accuracy-20261022"></a>
**2026-10-22 18:35–18:36 UTC：14组Loop准确率曲线只读快照。** [原世界六组折线图](../../results/loop-queue-target-20261021/accuracy-20261022-1830/original-world.png)、[前40万步放大](../../results/loop-queue-target-20261021/accuracy-20261022-1830/original-world-early.png)、[换数据与新初始化](../../results/loop-queue-target-20261021/accuracy-20261022-1830/robustness.png)、[六页PDF（含原子/A图、位置对照及最新数表）](../../results/loop-queue-target-20261021/accuracy-20261022-1830/loop-accuracy-comparison.pdf)、[全部绘图指标CSV](../../results/loop-queue-target-20261021/accuracy-20261022-1830/accuracy-curves.csv)。当前配方11组统一参数EMA/R8，每池5000题；位置/旧配方另列，旧基础DiscoLoop的raw/R8三跳88.26%保留。状态MLP门最近30万步B两/三/四跳为99.86/93.60/56.38%，同一步数旧invariant为83.30/45.76/24.60%；这是中途结果，尚无新门150万步终点。新世界/多对一invariant最近100万步四跳45.38/66.26%，新初始化1最近90万步27.76%。曲线保留所有保存节点和退化，无平滑、外推或缺失组补零；训练进度与最近评测步数分列。68份来源快照、1976条CSV指标及1308个绘图点核对通过；相同步数的数据曝光核对通过，不代表相同计算量。本次不执行模型、不修改训练或队列，来源哈希与复绘脚本随图保存。

<a id="loop-state-handoff-development-v1"></a>
**2026-10-22：原始两路状态门于14:09 UTC在GPU0正式开训，尚无150万步终点。** 已冻结826文件，固定镜像139项CPU契约与Ruff通过；控制器PID3084195保持。用户授权直接用旧状态和本轮候选学习接收分数。本批固定一条150万步训练，复用 native 的原世界、原初始化和完整配方；比较旧 invariant、ema、content，不重训对照。新门为1536→16→1的GELU MLP，连同衰减参数24,610个参数，输出层零初始化；CUDA初始前向、共享参数梯度及随机流与ema一致，32步工程短训与33组/30,216查询R4/R8独立重放均通过。终点再做R1–32重载及云端核验。见[冻结协议](../development-artifacts/loop-state-handoff-development-v1/protocol.md)、[执行状态](../../results/loop-state-handoff-development-v1/runtime/controller-state.json)及[交接](../../results/loop-state-handoff-development-v1/RUNNING.txt)。一次冻结副本检查在文件复制完成前启动而失败，原日志保留；最终冻结副本的139项检查已通过。

<a id="loop-queue-expansion-20261022"></a>
**2026-10-22 18:06 UTC：按用户要求利用新空卡，GPU2已接续原世界普通Loop的终点评测。** 普通Loop于12:52 UTC完成150万步、训练容器退出0；其R1–32独立重载于18:06:52退出0，深度评测与云端核验于18:10全部退出0；18:11多对一ema通过32步短训与新容器重载，在GPU2进入原定150万步训练（[W&B](https://wandb.ai/zsqzz/llm-memory-editability/runs/lnh20bd6d01)）。控制器PID2638715接管原15条训练与3项旧深度评测，原PID2075489退出；两条迁移invariant、GPU0状态门和记录修复器的容器ID/PID/启动时间全部不变。科学源码、配置、数据、初始化及训练预算不变，前组全部成功后才运行七条新初始化任务。原始`schedule.json`保留，当前资源池5/6/8/2及恢复材料见[调度修订](../../results/loop-queue-target-20261021/runtime/schedule-amendment.json)、[交接收据](../../results/loop-queue-target-20261021/runtime/scheduling-expansion-20261022/controller-stop-receipt.json)和[实时状态](../../results/loop-queue-target-20261021/runtime/controller-state.json)。

固定镜像10项调度/防重复/容量检查与Ruff通过；GPU2真实CUDA计算、反向和同步通过，旧invariant raw/EMA各43,000题R4/R8逐token重放通过。GPU3同时预检退出137、OOMKilled=false，未启动新科学训练；现场有运行中的每用户GPU配额守护，其代码默认四卡，但根权限配置和信号审计不可读，不能确认具体终止来源。保留失败且不重试，调度将本机该用户所有批次的占用与待入场分配合计限制为四卡；GPU1/4/7仍有其他项目预约。工程首轮Ruff因只读仓库缓存路径失败，改为`/tmp/ruff`后通过；原日志和容器保留。已完成训练、终点重载、深度评测和云端核验分别记录。

<a id="invariant-attention-temperature-20261022"></a>
**2026-10-22 UTC：统一降低 attention 温度，缓解低谷但没有消除低谷；后期准确率同时提高。零新增训练。** 使用25.6万、70万、130万步的原参数 EMA，预先固定温度1.0/0.9/0.8/0.7/0.5；对所有 head、位置、层和轮次统一执行 `softmax(QKᵀ/(√d·τ))`。不读取位置角色、轮次或实体标签，动态打分与跨轮聚合保持原实现。相同五池各5000题，共75组37.5万题次，主指标为R8完整答案准确率。见[冻结契约](../../results/invariant-attention-temperature-20261022/protocol.json)、[三页结果 PDF](../../results/invariant-attention-temperature-20261022/attention-temperature-report.pdf)、[完整 CSV](../../results/invariant-attention-temperature-20261022/full-answer-results.csv)与[逐题复算及配对得失](../../results/invariant-attention-temperature-20261022/analysis.json)。

| 统一 attention 温度 | 25.6万步 B四跳 R8 | 70万步 B四跳 R8 | 130万步 B四跳 R8 |
| --- | ---: | ---: | ---: |
| 1.0（原模型） | 66.10% | 6.86% | 93.74% |
| 0.9 | 66.80% | 7.56% | 95.36% |
| 0.8 | 67.60% | 8.28% | 96.20% |
| 0.7 | 68.48% | 8.78% | 97.00% |
| 0.5 | 69.60% | 10.46% | 97.62% |

**完整答案的收益成立，不能只归结为第二跳读出改善。** 在本次范围内，0.5的三个checkpoint四跳均最高；70万步净增3.60pp，130万步净增3.88pp，25.6万步净增3.50pp。70万步B两跳66.32→79.52%、三跳25.48→34.26%；三个checkpoint、全部温度的A留出四跳与B原子R8均100%。所有75组、全部16轮的“答案+EOS”联合正确率与答案正确率完全相同。准确率保持不意味着逐题无损：τ=0.5时，三个checkpoint分别修正264/203/204题、损伤89/23/10题，净增175/180/194题。R16的B四跳在τ=1.0→0.5时分别为67.72→71.34%、6.86→10.56%、93.70→97.58%；增加推理轮数仍不能救回70万步的低谷。

**机制线索与剩余问题。** 70万步B四跳的第二轮、第二关系位置，attention给中间位置的平均权重在τ=1.0→0.5时由51.68%升到59.63%，给原起点仍有38.92%。MLP后正确率48.04→61.00%，候选74.40→87.20%，聚合状态62.88→79.12%；最终四跳仅到10.46%。这支持读取混合是可以改善的一个环节，但均匀锐化不足以恢复整个组合过程。25.6万步的第二跳候选反而90.22→89.96%，其最终四跳仍提高，进一步说明不能用局部候选指标替代完整答案。由于每一位置、每一轮都被同时改变，不能把全部收益唯一归因于第二跳。当前只建立了这三个已有checkpoint、该世界的推理收益，未证明消除训练退化或普适最优温度，也未扩展更低温度搜索。

**验证与执行。** 温度1.0在3×5池的全部16轮答案与历史预测逐token相同；30组工程检查覆盖原块逐值一致、温度公式、未来答案不影响当前答案，以及联合评分与两token贪心生成一致。全部保存预测已从世界置换表独立重建真值并复算；参数、checkpoint及冻结源哈希不变。[W&B记录](https://wandb.ai/zsqzz/llm-memory-editability/runs/vxw11sg9)已finished，75条云端指标与本地逐项一致。科学容器退出0，GPU5共享推理区间02:55:00–03:01:45 UTC不作独占吞吐基准，三条原训练ID/PID/启动时间未变。工程首轮因验证代码的int16→int64赋值类型不匹配退出，修正并通过全部检查后才启动科学推理，失败日志保留。完整材料持久归档见[完成核验](../../results/invariant-attention-temperature-20261022/completion-manifest.json)。

<a id="invariant-second-hop-20261022"></a>
**2026-10-22 UTC：三个已有 checkpoint 的第二跳诊断完成，零新增训练。** 复用参数 EMA、冻结科学实现和相同 B 四跳/B 两跳/A 留出四跳池，各5000题。先运行15项位置局部干预，再根据70万步结果固定“屏蔽原起点/屏蔽当前位置/只清零原起点 value”的跟进对照；这是事后机制诊断，未增加学习探针或更新任何权重。见[主契约](../../results/invariant-second-hop-20261022/protocol.json)、[跟进契约](../../results/invariant-second-hop-20261022/routing-followup/protocol.json)、[诊断 PDF](../../results/invariant-second-hop-20261022/second-hop-diagnosis.pdf)、[完整数值](../../results/invariant-second-hop-20261022/all-interventions.csv)与[逐题复算](../../results/invariant-second-hop-20261022/analysis.json)。下述 attention 均指第二轮、第二关系位置；MLP 指 Transformer 共享块内的 MLP，动态打分网络保持原样。

**观察：70万步的中间实体已形成，但传给 MLP 的混合输入退化。** 下表是同一组 B 四跳题。attention 为12个 head、全部查询的平均；中间实体/候选使用模型原有读出，不将其等同于完整内部语义。

| 指标 | 25.6万步 | 70万步 | 130万步 |
| --- | ---: | ---: | ---: |
| 第一轮中间位置读出正确实体 | 99.94% | 100.00% | 99.80% |
| attention 给原问题起点的权重 | 10.68% | 41.14% | 24.01% |
| attention 给中间位置的权重 | 67.75% | 51.68% | 74.68% |
| 第二跳 MLP 后正确率 | 88.22% | 48.04% | 98.62% |
| 第二跳解码反馈后的候选正确率 | 90.22% | 74.40% | 98.76% |
| 对应原子事实的单轮 MLP 后正确率 | 100.00% | 100.00% | 99.80% |

**无真值的读取干预支持：有效中间信号的混合比例是低谷的重要环节。** 所有干预只改变第二轮的第二关系位置，随后自然执行到 R8；参数、关系输入、打分与聚合未改。强制读取现有中间位置不读取实体标签，也不移植其他 checkpoint 的表示。它在70万步修复了2598个原始 MLP 后错误中的2187个，同时损伤2个原正确项。对原起点清零但不重分配权重反而更差，因此不能把机制简化为“移除原起点信息即可”；中间信号强度、混合方向和后续非线性共同参与。

| 70万步、B四跳的第二跳干预 | MLP 后正确率 | 候选正确率 | 四跳最终 R8 |
| --- | ---: | ---: | ---: |
| 原模型 | 48.04% | 74.40% | 6.86% |
| 强制各 head 读现有中间位置 | 91.74% | 98.94% | 13.86% |
| 只屏蔽原起点、重新分配权重 | 80.38% | 91.78% | 11.34% |
| 只屏蔽当前位置 | 54.82% | 81.70% | 7.76% |
| 只清零原起点 value、不重分配权重 | 39.74% | 57.64% | 6.64% |
| 换入对应原子问题的 attention 增量 | 99.60% | 99.66% | 21.80% |
| 仅换入对应原子问题的 MLP 输入 | 100.00% | 100.00% | 15.14% |

**MLP 仍能调用事实，组合接口的适用性有差别。** 原子供体使用同一 checkpoint，只输入“正确中间实体、第二关系”，其后全部 padding，不输入第二跳答案。把原子问题的归一化 MLP 输入交给原 MLP，即使保留接收侧原残差，也使第二跳达到100%；换成错误实体的原子 attention 后状态，则100%输出那个错误实体对应的事实。这支持事实调用仍在，组合形成的输入更难被正确使用。原子 attention 后的直接答案读出为0%，但不能因此排除其隐含答案信息；这项供体干预不被称为纯实体注入。直接把实体嵌入转换成 key/value 并不普遍有效，甚至损伤两端健康 checkpoint，完整负结果已保留，不用这种分布变化替代自然机制证据。

**attention 权重本身不足以解释泛化。** 70万步 A 留出四跳的原起点/中间位置平均权重为40.11/52.47%，接近 B 的41.14/51.68%，但第二跳为100%。A/B 的组合 MLP 输入与对应原子输入平均余弦分别为0.9550/0.8136；B 在25.6万/130万步为0.8853/0.9050。这些几何量只是描述；结合定点读取干预，当前支持“B 组合输入对混合更敏感，读取/MLP 接口失配”而非知识被忘记，也不能把整个训练过程唯一归因于 attention 参数。B 两跳最终 R8 在强制读中间位置后由66.32%升到94.46%，但 B 四跳只由6.86%升到13.86%；与上一轮已见的后续覆盖一起，说明修复第二跳仍不能解决后续各跳与保持问题。

**验证与执行。** 主矩阵135组67.5万题次、跟进36组18万题次的指标已从保存预测与世界置换表独立复算；工程短测与科学结果分开保存。原生共享块逐值一致、全部9组 checkpoint×题池的8轮答案及第二跳原始/候选/聚合读出逐token复现；跟进同样重放。各干预的其他位置逐值不变检查通过，未来答案不影响第二跳，权重与 checkpoint 哈希不变。工程首版在审阅中发现“重置关系位置”会影响后续位置的 key/value 读取，科学运行前已局部化并加范围检查；初版记录保留。[两项 W&B 记录](../../results/invariant-second-hop-20261022/cloud-verification.json)均 finished，135/36条云端记录的全部指标与本地逐项一致。两科学容器退出0；GPU5共享推理区间为02:22:32–02:27:24及02:28:43–02:30:50 UTC，工程检查另有短暂共享，不作独占吞吐基准；原三条训练ID/PID/启动时间未变。完整预测、隐藏接口状态与来源已持久归档，见[完成核验](../../results/invariant-second-hop-20261022/completion-manifest.json)。

<a id="invariant-checkpoint-dip-20261022"></a>
**2026-10-21：原世界 invariant 的25.6万、70万、130万步诊断完成，零新增训练。** 统一参数EMA，三个checkpoint的B四跳R8为66.10/6.86/93.74%，R16为67.72/6.86/93.70%。每个条件覆盖相同的B两/三/四跳、A留出四跳、B原子五池，各5000题。先固定16项打分/β交换与消融，再根据逐轮记录固定4项时刻干预；均为事后机制诊断，不能解释为预注册确认或新架构训练。见[初始契约](../../results/invariant-checkpoint-dip-20261022/protocol.json)、[时刻干预修订](../../results/invariant-checkpoint-dip-20261022/timing-amendment.json)、[完整图PDF](../../results/invariant-checkpoint-dip-20261022/checkpoint-diagnosis.pdf)和[全部条件CSV](../../results/invariant-checkpoint-dip-20261022/intervention-results.csv)。

**观察：候选形成和已有结果保持都发生变化。** 在同一组B四跳题的第二关系位置，25.6万步R2→R3的聚合状态读出正确率为86.02→90.22%，70万步为62.88→33.46%，130万步为98.48→98.50%。R2新候选的对应正确率已经是90.22/74.40/98.76%，所以低谷不仅是把一个原本同样好的候选丢掉。再取70万与130万步共同满足“R2正确、R3候选错误”的1630题：前者1244题被改错，后者仅6题；新候选接收比例γ均值为17.90%与2.84%，原始分数均值−2.85与−4.79。这是同题条件子集的描述；中间实体用模型自带读出诊断，不等同于证明模型执行显式符号算法。

**干预：简单的交接参数漂移或一次覆盖不足以解释整个低谷。** 固定70万步其余权重，结果如下。阻止候选进入聚合时同步更新分子、分母和状态，保留原候选概率历史；选择位置/轮次对所有题一致，不读取标签。时刻对照使答案形成延后，却未恢复最终四跳表现。

| 70万步模型的推理干预 | B四跳R8 |
| --- | ---: |
| 原模型 | 6.86% |
| 仅β换为25.6万／130万步 | 7.14%／6.66% |
| 仅打分网络换为25.6万／130万步 | 0.34%／0.00% |
| 打分网络和β一起换为25.6万／130万步 | 0.30%／0.00% |
| 分数恒为零，保留原β | 0.00% |
| 阻止第二关系位置R3接收候选 | 6.84% |
| 各关系位置均阻止其形成后的下一轮接收 | 6.60% |
| 阻止第二关系位置R2写入（时刻对照） | 7.36% |

β替换及上述时刻干预的A四跳/B原子仍为100%；打分网络跨checkpoint移植会同时损伤保持能力，例如仅换130万步打分时A四跳0%、B原子43.30%，不能把移植失败当作B泛化的专一因果证据。三个原模型推理时分数归零，B四跳分别为0/0/0.20%；这是冻结模型移除功能，不是重训`ema`对照的结果。当前支持主干候选与接收规则需要配合，排除了所测β变化单独解释落差；尚未证明哪项训练动力学导致先退化后恢复。

**聚合系数必须结合状态尺度解释。** β为0.5932/0.6472/0.6631；答案位置R4平均γ为31.66/19.87/7.13%，而实际相对状态变化为5.37/5.97/6.67。答案位置初始嵌入范数随checkpoint为1.258/0.568/0.239，候选经过RMS归一化。因此γ更小不能直接读作更新更小，R8初始分量权重较大也不等于初始向量贡献较大。详细统计见[逐轮与配对分析](../../results/invariant-checkpoint-dip-20261022/analysis.json)和[参数几何](../../results/invariant-checkpoint-dip-20261022/parameter-geometry.json)。

**验证与资源。** 100份新预测、50万题次独立复算通过；原模型R4/R8全部逐token重放、原生trace一致性和未来答案不影响因果前缀检查通过。原权重哈希和模型参数未变，两容器退出0；[两项W&B记录均finished，80/20条评测完整](../../results/invariant-checkpoint-dip-20261022/cloud-verification.json)。GPU5只在2026-10-22 UTC 01:16:23–01:21:30及01:22:56–01:23:58分批共享推理，三条训练未停止或改动；这两个并发区间不作独占吞吐基准。材料持久归档于本批同名目录；保留挂载失败与审批拒绝后改用容器内输出的处理记录，没有放宽主机目录权限。

<a id="loop-invariant-robustness-v1"></a>
**2026-10-21：invariant 的同题深度测试完成至 16,384 轮；短循环有收益，长循环退化。八条新训练中两条留在源机，其余六条已与九条新基线一起迁至目标机GPU5/6/8队列；尚无新训练终点。** 设计见[协议](../development-artifacts/loop-invariant-robustness-v1/protocol.md)，规划见独立的 [Loop 路线图](../loop-roadmap.md)。

**19:22 CDT目标机提交。** `sn4622129202`控制器PID2075489按用户要求每卡一个实验；首轮GPU5新世界invariant、GPU6原世界普通Loop、GPU8多对一invariant。新初始化1、2的七条全部排最后，等待其他训练与旧深度评测完成。原1,967文件校验、三卡实际CUDA和四组各43,000题R4/R8旧预测重放通过；本机重建镜像`c77af324439d`完整依赖清单与源环境相同，镜像字节不同，原配方不改。每条短训及重载后进入正式150万步，终点评测和云端核验自动接续；当前实际阶段见[控制器状态](../../results/loop-queue-target-20261021/runtime/controller-state.json)与[交接](../../results/loop-queue-target-20261021/RUNNING.txt)。源两条训练后续仍待源终点和退出0证据，不重训；下文18:06记录保留为历史。

**19:27 CDT启动核验通过。** 三条均已实际完成至少1,000步，损失与梯度有限，32步短训及独立重载退出0，骨干初值哈希一致；W&B在线运行。启动核对发现W&B省略空字典，随后用独立记录修复器仅补回原空字段，不写曲线、摘要或运行状态；原科学代码和训练不变，后续记录同样检查。修复前对比与修复收据保留，原终点cloud审计仍按完整配置严格比较。见[启动收据](../../results/loop-queue-target-20261021/runtime/launch-validation.json)及[元数据修订](../../results/loop-queue-target-20261021/runtime/tracking-repair-amendment.json)。

**19:46 CDT历史曲线对照（只读保存结果）。** 统一参数EMA/R8，原世界invariant在3.2万步B两/三/四跳为0/0/0.04%，6.4万步为1.10/0/0%，12.8万步为99.98/93.02/37.78%；明显跃升位于6.4万至12.8万之间，未保存中间评测，不能精确定位。此时新世界与多对一invariant各训练2万步、最新评测1.6万步：B两/三/四跳分别0.08/0.02/0%与0.34/0/0.02%；原世界相同1.6万步为0.02/0/0.02%。模型、骨干初值、训练配方及评测节点一致，当前尚无泛化延迟证据；多对一两跳仍低于该池最佳常数答案0.74%，不把0.34%当作明显泛化。原世界后续明显起伏，四跳70万步降至6.86%、150万步为79.42%，出现泛化不等于持续改善。见[合并曲线与早期放大PDF](../../results/loop-queue-target-20261021/invariant-comparison-20261021-194607/invariant-learning-comparison.pdf)、[同口径数值和来源快照](../../results/loop-queue-target-20261021/invariant-comparison-20261021-194607/summary.json)。本次未修改训练或调度。

**18:06 CDT 调度变更。** 两批控制器 PID3310766/3423943 已退出，本机共15条新训练及3项旧模型深度评测停止排队。`init-1-invariant` 与 `new-world-ema` 保持原容器继续训练，其后的独立重载、深度评测及云端核验待接续。旧 invariant/ema 参数EMA深度直接复用。见[交接说明](../../results/loop-queue-transfer-20261021/HANDOFF.txt)和[停止核验](../../results/loop-queue-transfer-20261021/stop-result.json)；目标机器尚未启动，原调度入口不得重提。

固定原 150 万步终点、参数 EMA，在同一道题上增加共享块执行次数。图 B 只训练原子事实，组合未训练。全量 11 池共 43,000 题逐轮测 R1–256；下表每池 5,000 题。

| 模型／图 B 题目 | R8 | R16 | R32 | R256 |
| --- | ---: | ---: | ---: | ---: |
| invariant 原子 | 100% | 100% | 100% | 91.22% |
| invariant 两跳 | 99.32% | 99.28% | 99.18% | 88.46% |
| invariant 三跳 | 96.12% | 96.14% | 95.98% | 85.70% |
| invariant 四跳 | 79.42% | 82.12% | 81.98% | 74.18% |
| ema 四跳 | 10.78% | 11.58% | 9.70% | 8.46% |

四跳 R8→R256：invariant 修正 135 道、丢失 397 道原正确答案；三跳丢失 546 道、修正 25 道。三／四跳准确率仍优于 ema，但原子与两跳在长循环时弱于 ema（后者 R256 分别为 100/95.02%）。A 留出两／三／四跳均保持 100%。因此结果不能概括成“invariant 在所有题目上都更稳定”。

预先固定、与模型结果无关的每池 256 题，逐轮运行到 R16,384：

| 图 B 题目 | invariant R8→R16,384 | ema R8→R16,384 |
| --- | ---: | ---: |
| 原子 | 100→94.53% | 100→100% |
| 两跳 | 98.44→88.28% | 96.09→92.58% |
| 三跳 | 96.09→86.72% | 58.20→50.78% |
| 四跳 | 78.91→74.61% | 9.77→6.25% |

以上子集与全量分别计分，不能把不同样本的端点连成同一条曲线。两模型全部固定子集在 R1024 后答案不再变化，直到 R16,384。当前支持“能执行很深、短范围有收益、随后退化并进入平台期”，不支持同题持续计算改善；这是对 [InfiLoop 深度目标](https://arxiv.org/html/2610.11570v2)的检验，不是其数独／迷宫原任务复现。旧训练曲线也有波动：invariant 四跳在 130 万步 93.74%，150 万步终点 79.42%，相邻记录节点最大下降 41.50pp。多初始化稳定性与新数据分布结果仍待新训练。[首批实测 PDF](../../results/loop-invariant-robustness-v1/invariant-robustness-initial-report.pdf)将两档曲线与运行状态分开呈现。

**核验与执行。** 两模型新进程全池 R4/R8 与原保存答案逐 token 一致；两条深度容器退出 0。固定子集 R1–1024 和 R1–16,384 均完成，答案预测保留；主指标为全词表答案准确率，不冒称自由生成答案+EOS。见[全量曲线 PDF](../../results/loop-invariant-robustness-v1/depth-r256.pdf)、[预测复算与哈希](../../results/loop-invariant-robustness-v1/depth-verified.json)。新批共八条各 150 万步（总 1200 万更新），新增两个初始化、一个新置换世界、一个多对一关系世界，各配 invariant/ema。44 项契约检查通过，已运行的两条 32 步 GPU 短训及独立重载通过，正式任务为 init-1-invariant 与 new-world-ema。启动前的目录权限、运行库重复初始化失败均保留，没有科学训练被覆盖或重启；原执行目录为持久区本批 `attempt-3`；恢复先读 `queue-transfer-20261021.json`，只接续两个在训任务的后续，不恢复已停止队列。W&B 位于 `zsqzz/llm-memory-editability`；训练完成、终点重载和云端审计另行记录。

<a id="loop-baseline-robustness-v1"></a>
**2026-10-21 用户要求加入普通 Loop 与基础 DiscoLoop。** 原世界 150 万步、R8 旧终点如下；旧固定四轮配方用 raw，当前配方用参数 EMA，不能把跨配方差异全归因于架构。

| 结构／训练配方 | B 两跳 | B 三跳 | B 四跳 |
| --- | ---: | ---: | ---: |
| 普通 Loop／旧固定四轮 | 1.92% | 0.02% | 0.06% |
| 基础 DiscoLoop／旧固定四轮 | 98.38% | 88.26% | 0% |
| 普通 Loop／当前配方（2026-10-22完成） | 1.10% | 0% | 0% |
| 基础 DiscoLoop／当前配方（recipe-additive） | 40.36% | 0.02% | 0.04% |
| invariant／当前配方 | 99.32% | 96.12% | 79.42% |

基础 DiscoLoop 在旧配方的 B 三跳明显强于当前配方，必须同时保留。2026-10-22当前配方普通Loop已完成150万步及独立重载、深度与云端核验；参数EMA/R8的B两/三/四跳为1.10/0/0%，两世界原子均100%，A留出两/三/四跳为14.32/0.32/0.22%。其他新增基线终点仍待训练。旧固定四轮的 DiscoLoop 三跳 R4→R16 为 88.22→88.22%，普通 Loop 的 B 原子为 99.98→79.76%。这里复用已有 R1–16 诊断，不冒称新增模型推理。来源、哈希与分组见[已有比较](../../results/loop-baseline-robustness-v1/existing-comparison.json)。

新增[配对协议](../development-artifacts/loop-baseline-robustness-v1/protocol.md)覆盖原世界三个初始化、新置换世界、多对一世界，两种基线固定同一当前配方。原世界基础 DiscoLoop 及所有 invariant/ema 复用，只新增九条各 150 万步；57 项契约检查通过，CUDA 全尺寸前向/全部梯度与各自既有实现精确一致。两条 32 步工程短训、新容器重载及 66 组/60,432 查询的答案、EOS、似然与轨迹核验全部通过，196 份工程产物归档。九条新训练曾进入独立队列；18:06 CDT 按用户要求终止控制器 PID3423943，全部停止本机排队并列入三卡迁移清单，尚无新基线科学结果。调度器系统解释器缺少 numpy 的导入失败保留，随后使用原验证环境提交；无科学任务被重启。见[状态快照](../../results/loop-baseline-robustness-v1/active-status.json)、[比较 PDF](../../results/loop-baseline-robustness-v1/baseline-comparison.pdf)。

<a id="loop-beyond-four-hops-20261020"></a>
**2026-10-20：两图Loop检查点上的5–8跳（零训练）。现有检查点无法回答"能否多算一跳"：凡是答案落在第5个及以后位置的问题，各模型都答不对，与跳数无关。** 用户同意的零训练测试，[协议与追加说明](../../results/loop-beyond-four-hops-20261020/protocol.json)在运行前写定。

**做法：**
- 用同一世界的置换关系生成5–8跳题，图A与图B各一组，每组10个关系模板×500个起点。训练中从未出现长于4跳的组合。
- 每题从头执行，读出R1–32每轮的答案。
- 位置控制：把已会做的题前面补1–4个pad，使答案落到第5–8个位置。

**模型：**
- A0：无门DiscoLoop，150万步；
- [Loop原生交接批](../development-artifacts/loop-native-handoff-development-v1/protocol.md)的recipe-additive与ema终点EMA权重；
- 探索性参照：content第30万步、invariant第12.8万步。

**结果（R16，%）：**

| 模型 | 5–8跳（图A/图B） | 4跳题右移1位（答案在第5位） | 3跳题右移1位（答案在第4位） |
| --- | --- | --- | --- |
| A0 | 0.00–0.04 | 0.00 | 0.22 |
| recipe-additive | 0.00–0.22 | 0.00 | 0.16 |
| ema | 0.00–0.08 | 0.00 | 71.44 |
| content（30万，探索） | 0.02–0.20 | 0.00 | 1.44 |
| invariant（12.8万，探索） | 0.00–0.20 | 0.00 | 48.62 |

- **随机水平：** 0.2%。5–8跳题在任何轮数都没有明显高于这一水平。
- **原因：** 答案位置≥5时，各终点模型约100%输出EOS。训练序列长6，第5个位置只学过"四跳答案之后预测EOS"；第6个及以后的位置向量从未训练，权重衰减后为0。
- **按预定规则：** 位置控制全部不通过，因此5–8跳结果属于位置混淆，不能解读为组合失败。
- **只在实体中取最大值的读出（追加，事后诊断）：** 结论不变。5–8跳仍≤0.3%，答案落在第5位及以后的控制题仍为0。所以失败不只是EOS压过了答案。
- **对照：** ema与invariant在答案仍落在第4位时，能部分承受一位右移，A0不能。

**限制与复核：**
- 一个开发世界、一个初始化；前导pad本身也从未见过。
- R4复算与保存记录在图B三/四跳上相差≤0.3个百分点，其余池逐值一致。按预定的0.1个百分点阈值，三个模型的复算记为不通过；差异来自bf16与批次形状。
- content与invariant的150万步终点会在训练结束后自动补测，届时追加到[汇总](../../results/loop-beyond-four-hops-20261020/summary.json)。

**含义：** 要检验多跑能否多算，需要让模型在训练时见过更长序列的位置，例如训练时随机平移位置编号，或不用位置编码；这需要新训练。见[Loop路线图](../loop-roadmap.md#next)。

<a id="additive-carry-freezing-20261020"></a>
**2026-10-20：加性跨轮交接的1/r冻结（零训练再分析）。超过训练的4轮后，四类Loop模型在答题位置的每轮相对更新都约按1/r衰减：多跑的轮次只能微调已有状态。** 本节只读取[R1–16回放](../../results/loop-round-extrapolation-diagnostic-20261019)保存的逐题数组，不跑模型。定义`rel_r = ‖H_r − X_r‖/‖X_r‖`，即块输出相对本轮输入状态的变化，取中位数。

| 模型·池 | rel R8 | rel R16 | r·rel R8/R12/R16 | ‖X‖ R4/R8/R16 |
| --- | ---: | ---: | --- | --- |
| 无门DiscoLoop·图B三跳 | .166 | .057 | 1.33/1.01/.91 | 37/198/506 |
| 无门DiscoLoop·图A留出四跳 | .203 | .078 | 1.63/1.36/1.24 | 123/330/928 |
| 普通Loop·图B三跳 | .141 | .061 | 1.13/1.01/.98 | 23/48/85 |
| 学习u=.90·图B三跳 | .181 | .085 | 1.44/1.49/1.37 | 22/204/552 |
| P门·图B三跳 | .195 | .072 | 1.56/1.26/1.15 | 38/502/1256 |

**能说明：** 块更新的幅度大致不变（无门DiscoLoop约30），而状态沿已收敛的方向线性增长，所以每轮能改变的方向按1/r缩小。普通Loop同样如此，这一性质来自加性残差累积本身，而不是解码反馈。它与"多跑不变坏"一致。

**不能说明：** 这些模型在R4时已经不会四跳，所以不能据此推出"衰减阻止了多算一跳"；那是待检验的假说。数值来自单一开发图、单一初始化。依据与后续设计见[Loop架构评审](loop-native-handoff-review-20261020.md#3-零训练测量加性交接的1r冻结)。

<a id="rule-gate-round-evaluation-v1"></a>
**2026-10-19 07:15 UTC：规则门相对无门DiscoLoop的终点R1–16评测已排队，尚无本批结果。** 固定三个150万步检查点、11池43,000题，新增训练0；两条规则门等待原GPU0/1训练与grid成功后执行，无门基线在GPU0随后重放。完整记录答案与生成答案后的EOS、逐轮同题增失、原正确保持、首次/持续正确轮次、基线从未正确题的解决数，以及含额外规则计算的成本；R8图B三跳为预定主比较。首次正确轮次是事后诊断，不当作可部署停止规则。代码8项CPU检查及Ruff通过；固定训练镜像缺Matplotlib，CPU汇总改用已有主机环境，合成预测汇总和PNG/PDF生成验证通过。没有新增CUDA预检、独立训练审计或训练预算。[协议](../development-artifacts/rule-gate-round-evaluation-v1/protocol.md)、[实际队列](../../results/rule-gate-round-evaluation-v1/controller-state.json)、[验证记录](../../results/rule-gate-round-evaluation-v1/validation.json)。

<a id="rule-gate-training-v1"></a>
**2026-10-19：规则门从头训练两臂已启动，尚无科学结果。** 用户授权。`stability-u`（解码一致度门）与`readiness-u`（依赖闭合门）各150万步，沿用`fixed-q-plus-u`的交接、数据、初始化与普通训练；门无参数、无梯度。按用户要求立即启动，未做源码冻结、哈希校验与CUDA预检；源码为仓库工作树只读挂载，固定镜像CPU契约81项通过。主指标为训练后R1–16回放的R8图B三跳，对照复用的`fixed-q-plus-u`与无门DiscoLoop。[协议](../development-artifacts/rule-gate-training-v1/protocol.md)、W&B：[stability-u](https://wandb.ai/zhusq20/llm-memory-editability/runs/lh881c4c00)、[readiness-u](https://wandb.ai/zhusq20/llm-memory-editability/runs/lh881c4c01)。

<a id="loop-readiness-gate-test-20261019"></a>
**2026-10-19：依赖闭合稳定性门的零训练预检——5个检查点中1个通过。** 用户授权。规则：一个位置只有自己的解码在相邻两轮一致，且它注意力读取的前序位置当时都已就绪，门才打开。全部与实体无关，在推理时替换学到的u。
- **通过：学习u=.90。** 图A降幅在1个百分点内（四跳99.0→98.3%），图B两/三跳92.9→94.3%、84.3→90.3%（R8）。
- **修好了旧猜测问题。** 相对单纯稳定性规则，学习u=.90/.95的图A四跳由0%回到98.3/99.4%。规则复现了学习u=.90的逐位置开门时刻；答题位置的注意力来源在92–100%的情况下含上一跳位置。τ与头聚合方式影响很小。
- **不通过的检查点学了不同时刻表：**
  - `fixed-q-plus-u`答题位置第1轮后即打开，图A四跳降至约0%；
  - 只训两跳组的门几乎常闭，图A两跳降至34.3%；
  - 学习u=.95在图B上的中间解码不稳定，规则常闭，图B三跳降至61.8%。

零训练预检只说明规则与已学时刻表是否兼容。[完整结果](loop-depth-extrapolation-gate-stability-survey-20261019.md#79-依赖闭合的稳定性门零训练预检)、[汇总](../../results/loop-readiness-gate-test-20261019/summary.json)、[协议](../../results/loop-readiness-gate-test-20261019/protocol.json)。

<a id="loop-decode-stability-gate-test-20261019"></a>
**2026-10-19：解码稳定性规则替代学到的门——零训练检验不合格。** 用户授权。在5个门控检查点推理时，用相邻两轮解码的一致度（一致概率、余弦或argmax是否相同）替换学到的u，其余不变。事先固定的标准是图A各池R8不降超过2个百分点、图B两/三跳不低于原生，所有模型、所有规则形式都未通过。
- **图A四跳：** 由约99%降至0–1.3%。
- **图B三跳：** 学习u=.90由84.3%降至59.8–68.6%，`fixed-q-plus-u`降至0%。
- **原因：** 规则复现了图A关系位置的“形成轮关闭、下一轮打开”，也堵住了图B形成轮的泄漏；但答题位置在最后一跳之前的解码可能停在旧猜测上，规则把它当作稳定而提前开门（学习u=.90图A四跳第2轮后：学到的门0.00，规则1.00）。“未轮到”需要题目进度信号，解码稳定性无法提供。

冻结模型带学到的门训练，零训练失败不否定从头用规则训练。[完整结果](loop-depth-extrapolation-gate-stability-survey-20261019.md#78-解码稳定性规则能否替代学到的门零训练)、[汇总](../../results/loop-decode-stability-gate-test-20261019/summary.json)、[协议](../../results/loop-decode-stability-gate-test-20261019/protocol.json)。

<a id="loop-gate-timetable-intervention-20261019"></a>
**2026-10-19：门值干预的零训练因果检验完成。** 用户授权。6个门控检查点在推理时只覆盖连续更新门u，其余计算与权重不变；原生条件与已保存预测逐题一致，交接重算逐位相同。结果：
- **门时刻表对图A/B组合差距有因果作用。** 把图A的门时刻表移植到图B，6个模型中5个的图B组合明显提高，本图对照不变（R8）。共同门图B四跳由10.7%升至42.4%，`fixed-q-plus-u`110万步图B三跳由42.8%升至79.8%，只训两跳组图B两跳由25.0%升至77.9%。反向移植使图A下降，`fixed-q-plus-u`图A三跳降至25.2%。
- **q在门外的模型中，关键是形成轮只经解码反馈传递中间实体。** 第j个关系位置在第j跳形成那一轮拒收连续更新，中间实体只经解码反馈传递。单独施加这一条：学习u=.90图B三跳由84.3%升至93.0%，只训两跳组图B两跳由25.0%升至63.9%，图A不受损；全部放行则图A受损（学习u=.95图A四跳降至71.5%）。
- **例外：** 学习u=.95对移植基本不响应。
- **开发图模型的第+1轮瞬态不能由任何门值替换消除。**

事后选择的探索，覆盖门值偏离模型自身轨迹。[完整结果](loop-depth-extrapolation-gate-stability-survey-20261019.md#77-门值干预零训练因果检验)、[移植汇总](../../results/loop-gate-timetable-intervention-20261019/summary.json)、[规则汇总](../../results/loop-gate-timetable-intervention-20261019/formation-rule-summary.json)、[协议](../../results/loop-gate-timetable-intervention-20261019/protocol.json)。

<a id="loop-round-extrapolation-diagnostic-20261019"></a>
**2026-10-19：13个已有检查点的R1–16零训练逐轮回放完成。** 用户授权；本诊断不训练、不写权重，回放的R1–4（只训两跳组另含R6/R8）与已保存逐题预测共2,329,000条全部相同。结果分三点：
- **多跑不变坏，但不会多算。** 无门DiscoLoop在1–4跳上从R4跑到R16，各池完全不变（图B两/三跳98.4/88.2%）；P门图B持续下降。所有模型的四跳，以及只训两跳组的三跳，都不因多跑而提高。
- **学习u门在答题位置学成依赖题型和图的开关时刻表。** 答案算出前u≈0，之后打开；图B时刻表错位。错位伴随答案形成后下一轮（第+1轮）的一次性掉分，只出现在训练未监督该轮的“题型×轮次”上，再下一轮恢复。学习u=.95的图B三跳R3/R4/R5为70.4/32.1/68.3%；从30万到150万步，R4降46.1个百分点，R8只降10.4个百分点。`fixed-q-plus-u`从110万到150万步的下降（R8 42.8→22.6%）是持久的。
- **其他观察。** q累积使状态范数线性增长，但不能解释失败。失败时的错误多为图A实体，不是“多走一跳”。Kohli停止规则会在四跳第2–3轮过早停止。

单开发图/初始化的事后探索，门值与掉分的关系是相关观察。[完整诊断与文献整理](loop-depth-extrapolation-gate-stability-survey-20261019.md#7-零训练诊断r116逐轮回放)、[汇总](../../results/loop-round-extrapolation-diagnostic-20261019/summary.json)、[逐轮CSV](../../results/loop-round-extrapolation-diagnostic-20261019/round-accuracy.csv)、[协议](../../results/loop-round-extrapolation-diagnostic-20261019/protocol.json)。

<a id="sequence-dependency-gate-v1"></a>
**2026-10-18：序列依赖任务八臂已提交，首对ListOps训练正在GPU2/5运行；暂无终点性能结论。** 复用NDR作者ListOps与括号算术，保持普通答案+EOS CE和AdamW；每任务普通Loop、仅u、固定q、固定q+u各100k步，总800k更新。宽256/4头/共享单层，分别训练20/15轮，u初值.95、末轮直接读H。参数分别为ListOps无u 811520/有u 811777、算术811264/811521；矩阵FLOPs估算每步分别为ListOps无q 2.649T/有q 2.679T、算术1.987T/2.008T，点操作不含在内。

完整数据已独立解释核验：ListOps训练1000000，算术100200，两任务各四个1000题留出池，训练/留出与留出相互交集均为0。深度测试中含至少两条非叶子分支汇合的题分别为719/1000、884/1000；这是语法结构，不保证每条分支都影响最终答案。两任务全部留出答案位置和表达式长度均被训练覆盖。ListOps没有未见符号×位置组合；算术深度验证31/1000、测试190/1000题含未见组合，不能称完全排除了位置分布变化。[数据核验](../../results/sequence-dependency-gate-v1/input-audit-summary.json)。

69项固定镜像CPU契约、96项完整尺寸CUDA检查及Ruff通过；6000个同RNG生成样本与作者原方法逐字段相同。两任务各四臂主干初值一致；固定/学习u初始前向与共同梯度一致，CUDA graph恢复权重/Adam/RNG，八次工程更新均丢弃。首次CPU调度夹具因两CPU容器与四CPU假设不符失败，修复夹具后通过；首次挂载因新目录0700在模型执行前失败，独立可挂载副本238文件字节不变。失败未删除，未扩大科学范围；沿用短训/独立重载/云端终点审计豁免。[验证](../../results/sequence-dependency-gate-v1/validation.json)、[修订](../../results/sequence-dependency-gate-v1/amendment.json)。

训练只观察训练诊断与两个验证池；最终100k测试与预定轮数网格固定后自动汇总同题增失、增加轮数后的原正确保持、分支/深度/长度分组及完整训练多数类基线。ListOps是主要任务，算术保留原作者右枝偏置与乘零捷径。原数据来自双向encoder任务，本批是因果LM适配；一个数据实现/初始化及固定100k不支持通用优越性或架构上限结论。[协议](../development-artifacts/sequence-dependency-gate-v1/protocol.md)、[实际队列](../../results/sequence-dependency-gate-v1/execution/science-state.json)、[普通Loop曲线](https://wandb.ai/zhusq20/llm-memory-editability/runs/dg2889f0264b38bf)、[仅u曲线](https://wandb.ai/zhusq20/llm-memory-editability/runs/dga798ad65db3065)。

<a id="residual-gate-injection-ablation-v1"></a>
**2026-10-18：两条注入消融已提交GPU0/1，训练及科学结论待完成。** 用户明确确认两条各150万步：`only-u`为`X+u(H−X)`，`fixed-q-plus-u`为`X+u(H−X)+q`，u初值.95；均无学习注入g，末轮直接读H。保持当前40k训练/43k评测、共同主干初始化与数据流、1层×4轮、普通末轮答案+EOS CE及原优化配方。两臂各8,654,593参数；估算矩阵训练FLOPs/步分别527.30G和613.34G，不把同曝光称同计算量。新增预算300万更新，旧对照不重跑。

启动后已确认两个实际训练容器运行、两臂均完成至少1000步，初始完整模型摘要相同，实际PyTorch为2.8.0+cu128；[启动快照](../../results/residual-gate-injection-ablation-v1/execution-at-launch.json)。在线曲线：[仅残差门](https://wandb.ai/zhusq20/llm-memory-editability/runs/lh78c48600)、[固定q＋残差门](https://wandb.ai/zhusq20/llm-memory-editability/runs/lh78c48601)。这是运行确认，不是终点结果或云端独立审计。

60项固定镜像CPU契约、26项CUDA工程检查及Ruff通过。两臂初始参数逐值相同，骨干摘要匹配旧初值；同通道固定/学习u初始前向及全部骨干梯度一致，门梯度有限非零，CUDA graph捕获恢复权重/Adam/RNG，单次工程更新已各自丢弃。它们不保证两种注入架构具有相同初始函数或训练轨迹。首次打包遗漏原数据验证文件，两条在模型执行前失败；修订尝试补齐同一文件，完整保留旧产物。当前734文件冻结，原科学源码/配置/数据不变，仅执行端并行读取哈希；[修订](../../results/residual-gate-injection-ablation-v1/amendment.json)、[工程验证](../../results/residual-gate-injection-ablation-v1/validation.json)。

固定150万步/R4的B三跳为主，同时保留全部11池、训练曲线与终点R1–4。解释时区分训练节点波动和增加推理轮数后的退化；原子、训练组合和留出组合分别报告。旧gq+u的g₀=.99，而本批固定为1，且历史执行环境不同，不能把比较称为完全隔离门可学习性。当前只有一个已观察图和初始化，不推及一般LLM。用户此前免除的32步短训、独立重载和云端终点审计未重新加入，不记通过。恢复以[当前控制器](../../results/residual-gate-injection-ablation-v1/execution/controller-state.json)和实际容器为准；[协议](../development-artifacts/residual-gate-injection-ablation-v1/protocol.md)、[文献与历史核查、依赖数据设计](discoloop-injection-and-dependency-data-20261018.md)。

<a id="ungated-baseline-stability-20261018"></a>
**2026-10-18：先核对无学习门DiscoLoop基线，新增训练/前向均为0。** 重算同图三模型全部21个保存节点、11池的693条曲线记录，以及150万步R1–4的132条记录；均与保存的原始JSON逐值一致。另行核对30–150万步429条记录和132条网格记录与实际运行目录原件一致。以下固定训练四轮、全词表答案评分，下降以保存节点计算，不推断节点之间的变化。

| 图B评测 | 无学习门DiscoLoop | P学习注入门 |
| --- | ---: | ---: |
| 两跳最大相邻节点下降 | 7.46pp（30→40万步） | 54.58pp（60→70万步） |
| 三跳最大相邻节点下降 | 2.86pp（100→110万步） | 41.90pp（50→60万步） |
| 两跳150万步终点 | 98.40% | 61.90% |
| 三跳150万步终点 | 88.22% | 1.78% |
| 终点两跳R2→R3→R4 | 97.56→98.42→98.40% | 97.84→75.10→61.90% |

无门两跳在60–150万步保存节点为96.88–98.90%，该窗口最大先前高点回撤0.58pp。其三跳总体上升、后期小幅变化，没有复现P的反复大崩塌。但无门四跳训练题终点99.9%，A留出四跳仅0.14%、B四跳0%；普通Loop的B组合也长期接近零，不能把地板表现当稳定成功。P的A留出四跳同样出现110万99.68%→120万13.04%→130万99.86%的大起伏，故其不稳定并不限于B答案域。原子和训练组合在30–150万步都接近满分。

实现上raw为全部token、各非末轮固定`H+q`；P为同位置`H+gq`，末轮都直接读H，完整块、词表读出及归一化一致。原数据和评测池哈希相同，300k/1500k的40000维曝光数组逐元素一致，共同骨干第0步哈希已由旧预检核对。P新增769参数，初始g=.99而raw=1，因此不是完全相同初始函数；续训主机及实际镜像不同，虽GPU型号/PyTorch/BF16/TF32相同，未验证跨环境逐步轨迹等价。单图/初始化的当前证据支持“严重崩塌不是所有基线共有”，不证明数据毫无问题，也不独立分离门可学习性与初始化/环境效应。

[两/三跳完整曲线](../../results/ungated-baseline-stability-20261018/baseline-two-three-hop.png)、[四跳完整曲线](../../results/ungated-baseline-stability-20261018/baseline-four-hop.png)、[全部11池CSV](../../results/ungated-baseline-stability-20261018/learning-curves.csv)、[逐池统计](../../results/ungated-baseline-stability-20261018/summary.json)、[实际来源复核](../../results/ungated-baseline-stability-20261018/evidence-audit.json)、[保存记录核对](../../results/ungated-baseline-stability-20261018/verification.json)。

<a id="p-gate-root-cause-research-20261018"></a>
**2026-10-18：学习门深度诊断定位150万步的主要损坏环节，固定跳轮被全池检查否定；新增训练0。** 在已有四个P检查点上，完成56个模块记录/消融条件、28万条条件×查询；150万步仅去掉第三轮答题位置的MLP更新，图B两跳从61.90%恢复到97.34%，恢复1773道原错题、损失1道原对题，图A两跳保持99.96%。恢复题全部来自原1798道第二轮对、第四轮错的集合，覆盖98.61%。固定最终LayerNorm分母的有符号分解与此一致，第三轮MLP在1797/1798题上降低正确B答案相对错误A候选的分数；投影贡献与消融总效应分别解释。

后续在原11池43000题比较原生与相同消融：原子保持，但A留出三跳查询99.98%→95.90%、A留出四跳查询99.92%→0%、B三跳1.78%→0.90%。A训练包含两/三/四跳，这里不涉及未训练深度。这证明该位置更新对不同任务的作用不同，不能把固定删除作为通用修复。70万步同样干预仅23.64%→38.80%，其第二轮已经低分，不能由150万步的后续损坏完全解释。原生4万条模块诊断答案、补充4.3万条答案及生成EOS均与旧记录逐条一致，模型/检查点摘要不变；成功容器均退出0，首次挂载失败保留。全部测量来自已观察图和初始化；本次不证明学习门造成训练历史的唯一原因，也未验证新训练方法。

[完整诊断、实际Transformer推导及七篇原文方法比较](learning-gate-root-cause-20261018.md)、[模块逐题复算](../../results/p-gate-root-cause-research-20261018/module-summary.json)、[全部11池](../../results/p-gate-root-cause-research-20261018/ablation-transfer/diagnostics.json)、[诊断协议](../../results/p-gate-root-cause-research-20261018/protocol.json)、[补充协议](../../results/p-gate-root-cause-research-20261018/transfer-protocol.json)。方法候选仍须区分第二轮组合形成和后续保持；本条是结果，不增加训练排程。

<a id="p-gate-instability-diagnostic-20261018"></a>
**2026-10-18：P门图B两跳波动的固定检查点诊断完成，新增训练0。** 按用户要求，选择已有50/70/90/150万步高分、低分、恢复与终点检查点，完成两阶段104个条件、52万条条件×查询评测；这是同一图两个5000题池上的重复干预，不是独立新样本或世界。八个原生池共4万条答案与旧记录逐条一致，手工分轮与原生首批logits逐值一致，自身门值重放与干预后模型摘要核对通过。两次执行均退出0。[完整诊断及文献核验](../../results/p-gate-instability-diagnostic-20261018/index.html)、[四页图](../../results/p-gate-instability-diagnostic-20261018/diagnosis-figures.pdf)、[来源与检查](../../results/p-gate-instability-diagnostic-20261018/validation.json)。

图B首轮桥实体读出始终99.88–99.96%，两跳第2轮为96.82/41.34/95.56/97.84%，第4轮为96.14/23.64/91.90/61.90%；图A两跳最终99.96–100%。70万步已在第二轮退化，限制B词表的最终分数也仅23.84%；150万步则第二轮已答对，后续轮次破坏，最终37.94%输出A实体，限制B词表评分98.26%（仅诊断，不替代全词表评分）。两者不能统一解释为跨图偏置或后续保持失败。

将50万步门参数装入70万步模型仅从23.64%到31.12%，反向仍96.14%；同题门标量值重放也不能转移高低分状态。150万步换早期门能改善到74.00%，故门有直接作用，但不足以单独解释全部退化。强制所有门为1或关闭后续注入会破坏已有模型，不能据此否定从头固定系数1的无门基线。当前定位为首跳之后的组合与答案保持问题；图B组合缺少直接训练约束、整体组合路径随参数变化是工作假说，尚未定位具体主干/嵌入/读出模块或证明训练因果链。不把sigmoid饱和、门关闭、梯度爆炸认定为根因。原论文§5.1符号任务固定系数1、§5.2自然语言才用学习门；ICML2020门饱和、ICML2018完整状态转移及NeurIPS2025门改善稳定性的论文提供相关分析，均未直接诊断本项目P波动。此次为已观察世界与检查点的事后探索。

<a id="gate-complete-figures-20261018"></a>
**无门基线补充：** 用户指出部分分组图仅含P，需要明确比较不带门的DiscoLoop。现已将同开发图的`disco`（从训练开始固定注入系数1、保留软再编码通道）与P单独绘成[八页完整对比](../../results/gate-ungated-comparison-20261018/complete-atlas.pdf)，包括全部11池训练曲线、30万/150万步及终点轮数评测，另有[主要对比](../../results/gate-ungated-comparison-20261018/comparison-summary.pdf)。这组已有实验不重跑。图B三跳无门/P在30万步为0.06/37.20%，150万步为88.22/1.78%；图A未见四跳的150万步则为0.14/99.92%，不能概括成某方全面更好。共同数据/评测哈希、训练配方字段与两个预算的数据曝光核对一致；不同执行主机/镜像历史仍保留，不宣称新跨环境等价验证。三个新图、长度外推、紧凑Wang及原Wang150万步仍缺匹配的无门对照；原Wang30万步已有旧无门结果，不能代替150万步端点。仅比较P/固定更新门/学习更新门，能回答新增更新门相对P的作用，不能回答整个门控方案是否优于无门DiscoLoop。本次补图新增训练0。

**2026-10-18 13:55 CDT结果快照：门控实验完整比较图册。** 按用户要求，仅读取已有输出制作[完整图册](../../results/gate-complete-figures-20261018/index.html)、[主要比较PDF](../../results/gate-complete-figures-20261018/comparison-summary.pdf)及[全部图像PDF](../../results/gate-complete-figures-20261018/complete-atlas.pdf)。覆盖10个开发条件、G/D/V历史对照、3个新图的9组确认、原Wang四方法与紧凑模型、长度外推三图的9个计划条目；36个条目中28个训练及终点轮数评测完成、2个运行中、6个待运行。各run实际输出优先于尚未更新的父控制器；下方“刚启动”的历史记录不代表本快照进度。本次新增训练及模型前向均为0。

图册保留全部已存评测节点、全部评测池、固定30万/150万步、终点推理轮数及估算矩阵训练FLOPs；运行中曲线止于最后完整评测，未完成终点留空。多跳使用全池答案准确率，Wang使用答案＋EOS联合准确率；紧凑配置同时改变词表、层数和轮数，长度外推使用NoPE且不训练三/四跳，分别解释。原论文门P、固定.90续训亦已完成：图B三跳终点为1.78/59.68%；学习.90、学习.95、共同门、候选门为83.48/32.06/69.46/2.16%，无学习门DiscoLoop为88.22%。三图学习.95减固定.95的三跳配对差为−9.24/−6.00/+34.04pp，不构成稳定跨图优势。第一组长度外推150万步三臂在R4的图B三跳为0.24/0.20/0.16%，增加到R8仍未显示成功外推。完整曲线展示训练后的波动，不选峰值替代终点。

原Wang共同门与紧凑模型OO终点分别97.70/98.20%，不是单变量比较。Wang固定.95与学习.95在快照中分别训练至141.8万/21.3万步；图像只使用各自最近的完整评测。所有原始JSON、SHA256、数值CSV、生成脚本与[记录一致性检查](../../results/gate-complete-figures-20261018/verification.json)随[下载包](../../results/gate-complete-figures-20261018/charts-and-data.zip)保留。完成run的终点、学习记录、训练轮数对应grid评分及模型哈希相符，续训与父批共同节点评分相符；这些是保存记录的核对，不是新独立模型审计，近期批次的独立重载/云端终点审计豁免保持。

<a id="candidate-update-gate-gpu1-v3"></a>
**2026-10-17 23:26 CDT：候选门实验已按用户确认迁至GPU1并启动，旧GPU2等待记录已失效。** GPU1前序无学习门控DiscoLoop的150万步训练、R1–4评测与控制器均退出0。新尝试`gpu1-recovery-v3`保留原模型、数据、初始化、样本流、优化、评测和150万步目标；725份原文件逐字节不变，仅调度配置改GPU1/CPU4–5，当前731文件冻结。14项固定镜像迁移契约、目录读写、17项CUDA工程及Ruff检查通过，一次工程更新已丢弃。旧两次失败完整保留；本次无GPU检查明确遇到原目录0700导致Docker bind mount permission denied，报错保存。修改旧权限被自动审批拒绝且未执行，改用新建独立输出目录成功启动，既有目录权限保持不变。用户GPU并集上限仍四张，无新科学臂、无终点结论；原短训/独立审计豁免保留。**实际当前状态须读[GPU1控制器](../../results/candidate-update-gate-multihop-v1/gpu1-recovery-v3/controller-state.json)，不要从旧根目录或旧GPU2入口重复提交。** [迁移与恢复说明](../development-artifacts/candidate-update-gate-multihop-v1/gpu1-recovery-v3.md)、[在线曲线](https://wandb.ai/zhusq20/llm-memory-editability/runs/lh14e7ae00)。

<a id="compact-wang-joint-gate-v1"></a>
**2026-10-17 23:06 CDT：当前一层×四轮共同门在原 Wang 数据上的紧凑词表实验已于本机 GPU2 启动。** 用户采纳当前小模型后明确指定GPU2。本批一臂150万步、从头训练；完整保留世界790101的原图/训练支持/七池字节与哈希，313600条训练样本和19004条原评测查询，另保留40000原子诊断。词表由52463改为2203，实体/关系分别2000/200、PAD/保留/EOS三位置；一层×四轮、宽768/12头共同门初值.95，参数9,569,282。固定1500k/R4的OO答案+EOS为主，保留300k及全部节点、原子/各组合池和终点R1–4。固定镜像25项CPU检查、45项全数据映射核对、16项CUDA工程检查及Ruff通过；一次工程更新已丢弃。1029文件冻结，在线W&B与持久控制器已启动，尚无终点结论。保持近期短训/独立重载/云端终点审计豁免；原Wang大模型及其它批次未改。词表删除改变输出竞争与软再编码，且模型深度/轮数同时改变，不是单变量归因或原配方复现；仅一个已观察图/初始化。[协议](../development-artifacts/compact-wang-joint-gate-v1/protocol.md)、[当前控制器](../../results/compact-wang-joint-gate-v1/execution/controller-state.json)、[在线曲线](https://wandb.ai/zhusq20/llm-memory-editability/runs/lhef69f000)。恢复时读取本批process/state、冻结配置与实际容器，不重复提交。

<a id="candidate-update-gate-multihop-v1"></a>
**2026-10-17 CDT：融合后候选门新增一臂150万步，当前排队等待GPU2。** `candidate-update-gate-multihop-v1`只把共同门u的输入从LN(H)改为LN(C)，C=H+gq，交接仍X+u(C)(C-X)，末轮直接读H；同原图/初始化/样本流/优化/评测与固定镜像，旧joint-token只读复用。49项CPU检查、17项CUDA/恢复检查及Ruff通过。首个GPU0正式容器完成第0步全池评测后退出137，OOMKilled=false，末次保存step0，未保存更新未知；按用户GPU配额服务在运行，疑似第五张卡触发限制，但实际执行日志不可读，原因未确认。原失败容器、控制器、权重、W&B及722份冻结文件完整保留。独立`resource-recovery-v2`只修调度：等待GPU2普通Loop续训及grid成功，再在原卡队列执行；每阶段加入后本用户GPU并集≤4，不关闭配额、不增加第五张卡，15项恢复检查通过。新尝试尚未开始科学更新，150万步目标与单臂范围不变；原短训/独立审计豁免保留，无终点结论。[协议](../development-artifacts/candidate-update-gate-multihop-v1/protocol.md)、[恢复说明](../development-artifacts/candidate-update-gate-multihop-v1/resource-recovery-v2.md)、[当前队列](../../results/candidate-update-gate-multihop-v1/execution/attempts/resource-recovery-v2/controller-state.json)。

**2026-10-17 20:38 CDT：CUDA6已接续五项原计划新图实验。** 第一组新图三条300k训练及grid全部完成，接收控制器退出0。用户指定CUDA6后，将新图2的固定.95/学习.95与新图3的P/固定.95/学习.95排他迁移；首条固定.95已启动，其余四条自动接续。五项各300k，合计150万步属于原1230万预算。10项迁移契约及Ruff通过，冻结科学源码/配置/数据核验通过；未增加短训或独立审计，无新终点结论。原Wang/外推仍在原队列。实际状态以[新接收控制器](../../results/gate-generalization-extension-v1/gpu6-handoff-v2/controller-state.json)为准；[调度交接](../development-artifacts/gate-generalization-extension-v1/gpu6-followup-20261017.md)保留旧调度防重复冲突与环境差异。下方图表继续对应原快照。

**图表清晰版（沿用2026-10-17 20:13 CDT快照，数值不变）。** 将两跳、三跳、四跳分别展示，每张学习曲线只保留一组对照，放大文字并把模型及数值移到图外。[已完成终点总览](../../results/gate-progress-figures-20261017-2013/clear-v2/completed-overview.png)、[两跳图册](../../results/gate-progress-figures-20261017-2013/clear-v2/2-hop-charts.pdf)、[三跳图册](../../results/gate-progress-figures-20261017-2013/clear-v2/3-hop-charts.pdf)、[四跳图册](../../results/gate-progress-figures-20261017-2013/clear-v2/4-hop-charts.pdf)、[全部13页PDF](../../results/gate-progress-figures-20261017-2013/clear-v2/all-charts.pdf)。缺失终点仍留空，旧图与数据快照保留。

**2026-10-17 20:13 CDT：门控与多跳现有数据图表快照。** 只读汇集开发图九个条件及三组新图当前产物，共1,958条“节点×评测池”记录；新增训练和模型评测均为0。[学习曲线](../../results/gate-progress-figures-20261017-2013/learning-curves.png)、[30万/150万步比较](../../results/gate-progress-figures-20261017-2013/fixed-budget-comparison.png)、[新图验证](../../results/gate-progress-figures-20261017-2013/new-graph-validation.png)、[完整11池](../../results/gate-progress-figures-20261017-2013/all-evaluation-pools.png)、[合并PDF](../../results/gate-progress-figures-20261017-2013/experiment-charts.pdf)、[完整CSV](../../results/gate-progress-figures-20261017-2013/all-learning-data.csv)、[实时步数快照](../../results/gate-progress-figures-20261017-2013/training-status.csv)及[来源哈希/核对记录](../../results/gate-progress-figures-20261017-2013/provenance.json)。固定150万步/R4，学习门初值.90原LR/半门LR/初值.95的图B三跳为83.48/81.18/32.06%，四跳20.12/2.70/.70%；其余开发条件尚未到终点。无门控DiscoLoop与普通Loop已接续训练，第一组新图P/固定.95已完成、学习.95仍运行；旧汇总器的迁移目录冲突与接收端实际状态区分。曲线不平滑、不外推，未完成终点留空，半LR在30万步前复用同一父轨迹。60份来源JSON快照及14项身份/计分核对保留；这是现有输出一致性检查，不是独立模型或云端终点审计。单开发图/初始化、跨环境差异、计算成本差异及新图配对未完成的限制保留；下方提交时状态作为历史。

**2026-10-17 18:23 CDT：无学习门控DiscoLoop与普通Loop的150万步对照已提交等待。** `ungated-loop-continuation-v1`复用父批`loop-hop-unified-v1`的`disco-l1-r4`与`loop-l1-r4`完整30万步状态，各新增120万步至150万步，共240万新增科学更新。无门控DiscoLoop固定注入系数1，普通Loop无再编码通道；同一层共享块×四轮、原图/初始化/流/Adam/RNG恢复，不重复warmup。31项CPU及实际父检查点恢复检查、Ruff、容器输出写入验证通过；223文件冻结。GPU1/2分别等待当前`token-update`/`token-update-half-lr`训练及grid成功退出，之后自动做丢弃的CUDA恢复检查、续训、R1–4终点评测；当前两臂均`waiting_for_predecessor`，新增科学更新0。保留每10万步11池评测，按固定150万步比较，不用峰值替代终点。旧入口不重启，原审计豁免/环境差异保留，不抢占其它实验。[协议](../development-artifacts/ungated-loop-continuation-v1/protocol.md)、[实际队列](../../results/ungated-loop-continuation-v1/execution/controller-state.json)、[提交记录](../development-artifacts/ungated-loop-continuation-v1/execution.md)。

<a id="joint-update-gate-wang-v1"></a>
**2026-10-17 CDT执行状态：原Wang数据的共同门控150万步已排入GPU5接续队列，尚未开始科学训练。** 用户要求沿用原数据并选择等当前GPU5训练/评测完成后接续。原world/panels与历史存档逐字节一致；四层共享块×两轮、原52463词表、初值/数据流/优化和1500k OO主指标均与现有Wang学习门对齐，仅非末轮新增再编码项也乘u。最终冻结包19项CPU契约、数据读取身份、Ruff通过；1022文件冻结，GPU检查将在前序结束后进行。前序失败不会当作完成启动新任务。现有Wang三基线不重复提交，在线W&B将在正式训练开始时开启；无终点或新性能结论。[协议](../development-artifacts/joint-update-gate-wang-v1/protocol.md)、[状态](../../results/joint-update-gate-wang-v1/execution/controller-state.json)、[检查记录](../../results/joint-update-gate-wang-v1/execution/validation.json)。

<a id="joint-update-gate-multihop-v1"></a>
**2026-10-17 22:37 UTC执行状态：新共同门控150万步实验已启动，无终点结论。** 本机GPU5运行`X+u(H+gq-X)`，门初值.95，四轮；与原连续门保留共同初始化、原图、数据流、损失及优化配方，唯一模型变化是非末轮再编码也乘u。实际已超过9000步，W&B[新门运行](https://wandb.ai/zhusq20/llm-memory-editability/runs/lh19b01a00)。300k为指定中间比较，1500k/R4图B三跳为主指标，保留11池及R1–4。38项门控测试、17项实际CUDA/恢复检查和Ruff通过，丢弃工程更新1；未做32步短训或终点独立/云端审计。本地固定依赖镜像已构建，跨机器环境差异保留，不称真实预训练大模型性能验证。[协议](../development-artifacts/joint-update-gate-multihop-v1/protocol.md)、[运行状态](../../results/joint-update-gate-multihop-v1/execution/controller-state.json)。

**旧队列迁移：** GPU6接收第一组新图的P/固定.95/学习.95三项，各300k，P已启动[在线记录](https://wandb.ai/zhusq20/llm-memory-editability/runs/lh1df31400)，另两项接续。1065份原冻结文件哈希与4项排他迁移契约通过；科学源码/配置和总预算不变，实际运行镜像另记。共享输出占位阻止旧调度器重复创建容器，原报告路径以相对链接读取新产物。原控制器未来可能为迁移项记录目录已存在，实际成功/失败以[接收控制器](../../results/gate-generalization-extension-v1/gpu6-handoff/controller-state.json)与[修订](../../results/gate-generalization-extension-v1/gpu6-handoff/schedule-amendment.json)为准。两条原续训保持，不把迁移当新增实验。

<a id="gate-generalization-extension-v1"></a>
**2026-10-17 21:09 UTC：全面扩展队列已提交，u=.95学习门/固定门分别在GPU5/7续训，两条均已超过301k，另21项自动排队。** 当前是执行状态，没有新的终点结论。GPU4实际利用率100%且存在预约，控制器等待其独立空闲。

| 问题 | 已冻结的比较 | 新任务／新增更新 |
| --- | --- | ---: |
| 优势是否持续，还是主要更早学会 | 固定.95／学习.95从300k完整状态续至1500k；P只复用另机已排队续训 | 2／240万 |
| 新图和初始化上能否重复 | 3组新图／初始化，每组P、固定.95、学习.95，各300k | 9／270万 |
| 在原Wang两跳任务上是否有效 | 同一原图、数据与初始化，三方法各1500k；300k为预定次终点 | 3／450万 |
| 能否外推到未训练的跳数 | 同3组新图，仅训练原子与两跳；三方法各300k，评测三／四跳，统一NoPE | 9／270万 |
| 合计 | 23项新任务；另机P只读复用，不计新增 | 23／1230万 |

三方法均保留完整Transformer块和P软再编码注入，只有新增连续更新门不同。续训保留Adam、样本流、RNG和学习率；新图内初始化及曝光配对。固定终点／推理轮数为主，保留全部曲线、原子前提覆盖和各评测池，另报曝光、token、矩阵FLOPs及墙钟。确认与外推共用三个图种子，不构成六个独立世界；Wang与续训仍各一个已观察图。旧.90负结果、跨机P环境差异和旧DiscoLoop注入方式差异均保留，不预设统计显著或机制归因。

82项CPU契约、11项CUDA工程检查和Ruff已通过；恢复检查包括完整模型、Adam、下一批样本与RNG逐值一致，工程更新15次并丢弃。源码／输入1065文件冻结，科学代码与镜像未因调度修复改变。首次尝试23项均在Docker创建时因本批NFS父目录0700权限失败，未创建训练容器、零科学更新；改为0755并通过独立无GPU容器持久写入验证后，原源码在`execution-v2`重新提交，旧`execution`保留。此前用户豁免的独立重载和云端终点审计未自动加入，不将输出一致性汇总冒充独立审计。

[完整协议](../development-artifacts/gate-generalization-extension-v1/protocol.md)、[配置](../../configs/gate-generalization-extension-v1.json)、[当前控制器](../../results/gate-generalization-extension-v1/execution/controller-state.json)、[自动汇总](../../results/gate-generalization-extension-v1/execution/summary.json)、[工程结果](../../results/gate-generalization-extension-v1/engineering-summary.json)、[运行修订](../../results/gate-generalization-extension-v1/runtime-amendment.json)、[W&B学习.95](https://wandb.ai/zhusq20/llm-memory-editability/runs/lhc9c71501)、[固定.95](https://wandb.ai/zhusq20/llm-memory-editability/runs/lhc9c71500)。

<a id="paper-update-gate-continuation-v1"></a>
**2026-10-17 20:55 UTC：四组300k→1500k续训已提交，两条学习门分支均已到301k。** GPU1运行原门LR，GPU2仅把新增更新门LR减半；两条均精确恢复同一个300k模型、全部Adam状态及采样/RNG状态。主干和原注入门保持1e-4，新门分别1e-4/5e-5。P和固定u=.90分别排在两卡后续队列，四臂各新增120万步，共480万次新增更新。当前没有续训终点结论；完整11池每新增100k评测，主指标固定总1500k、R4、图B三跳。

45项CPU测试、Ruff与真实父检查点CUDA恢复一致性检查通过：三个模型连续更新/保存恢复后的全部参数、Adam、样本、RNG与损失逐值一致；半LR当步只改变新门位移。224个源码/输入文件冻结，旧15份核心源码身份保留；10次废弃有效工程更新与28次撤销的图准备更新单记，不计入科学步数。两条已启动运行损失和梯度有限。按原偏好不加32步短训及独立终点/云端审计。初值.95完成结果保留在下节，本轮不按其结果扩大矩阵。W&B：[原门LR](https://wandb.ai/zhusq20/llm-memory-editability/runs/lhaf4e6b02)、[半门LR](https://wandb.ai/zhusq20/llm-memory-editability/runs/lhaf4e6b03)。[协议](../development-artifacts/paper-update-gate-continuation-v1/protocol.md)、[配置](../../configs/paper-update-gate-continuation-v1.json)、[实际控制器](../../results/paper-update-gate-continuation-v1/execution/controller-state.json)、[启动快照](../../results/paper-update-gate-continuation-v1/execution-at-launch.json)。

<a id="paper-update-gate-multihop-u95-v1"></a>
**2026-10-17 15:45 CDT完成：u=.95 两条300k主实验及终点R1–4评测完成。** 控制器与两条train、两条grid共四个阶段均退出0，GPU5/7释放；新增600,000次更新。以下均为300k、推理四轮的全词表答案准确率，P只读复用：

| 评测 | 论文门P | 固定更新门.95 | 学习更新门，初值.95 |
| --- | ---: | ---: | ---: |
| 图A原子 | 100.00% | 100.00% | 99.98% |
| 图B原子 | 99.98% | 99.92% | 99.88% |
| 图A未见两跳 | 100.00% | 100.00% | 99.94% |
| 图B两跳 | 94.22% | 95.40% | 98.30% |
| 图A未见三跳 | 99.68% | 99.68% | 99.80% |
| 图B三跳，主指标 | 37.20% | 39.64% | 78.18% |
| 图A未见四跳 | 0.68% | 1.10% | 97.02% |
| 图B四跳 | 0.14% | 0.00% | 14.30% |

三条训练题池均为99.8–100%，完整11池见[完成汇总](../../results/paper-update-gate-multihop-u95-v1/completion-summary.json)。学习门比固定门的主指标高38.54pp，比P高40.98pp；图B四跳仍低。学习门图B三跳在推理三轮为81.30%、四轮为78.18%，预定主指标仍报四轮，不选最好轮数。此为同一已观察开发图、一个初始化的比较，尚不归因于具体内部机制。

两条曝光数组与旧P逐元素一致，终点与R4记录一致。这些是现有输出文件核对，不是独立模型审计。用户明确要求本机建立镜像后直接启动；专用镜像从原P镜像构建，主要依赖版本断言通过，但与交接指定镜像ID不同，未新增跨环境数值等价验证。新短训0、独立重载/云端终点审计未执行，不记为通过。W&B：[固定.95](https://wandb.ai/zhusq20/llm-memory-editability/runs/lh8fd92600)、[学习.95](https://wandb.ai/zhusq20/llm-memory-editability/runs/lh8fd92601)。[控制器终态](../../results/paper-update-gate-multihop-u95-v1/execution/controller-state.json)、[执行与恢复说明](../development-artifacts/paper-update-gate-multihop-v1/sn4622129202-execution.md)、[环境修订](../../results/paper-update-gate-multihop-u95-v1/runtime-amendment.json)。

<a id="paper-update-gate-multihop-v1"></a>
**2026-10-17 20:20 UTC完成：u=.90两条300k训练及R1–4终点评测均完成，全部阶段退出0。** P/固定.90/学习.90的图B两跳为94.22/89.06/68.36%、三跳37.20/34.18/15.40%、四跳.14/.08/.18%；图A未见四跳为.68/8.04/**98.18%**。学习门在这个300k预算下的图B三跳较低，但图A四跳明显较高，不能概括为没有优势，也不据尚在变化的曲线判断最终能力。两条曝光数组与P逐元素相同，300k与R4记录一致；完整11池和文件摘要见[完成汇总](../../results/paper-update-gate-multihop-v1/completion-summary.json)。后续只在新的[续训批次](../results.md#paper-update-gate-continuation-v1)从完整状态分叉，不重启此入口。W&B：[固定.90](https://wandb.ai/zhusq20/llm-memory-editability/runs/lh9336ab00)、[学习.90](https://wandb.ai/zhusq20/llm-memory-editability/runs/lh9336ab01)。

151项CPU检查及Ruff通过，1项旧完整数据容器测试跳过；12份核心源码和旧P的11份产物只读核验通过。按用户要求本批新增短训0、独立重载/云端结果审计未执行，不记为通过。当前镜像与旧P不同，未新增跨环境等价验证；共享目录挂载失败发生于模型执行前，保留于[失败记录](../../results/paper-update-gate-multihop-v1/engineering-runtime/container-start-failure.json)，后用本地只读快照和持久镜像输出完成。单图/初始化限制不变。[协议](../development-artifacts/paper-update-gate-multihop-v1/protocol.md)、[实际控制器](../../results/paper-update-gate-multihop-v1/execution/controller-state.json)。

<a id="gated-loop-depth-extrapolation-v1"></a>
**2026-10-17 CDT完成状态：六臂20k pilot已完成，控制器以`pilot_no_go`停止，18条100k正式训练未启动。** 六条32步CUDA短训、六条pilot及独立重载通过；数值有限性、配对初始化/曝光与损失下降通过，但N-Full/Q-Full未满足冻结的原子及3–5跳学习阈值。该停止是预定调度结果，不能据20k判定长期学习或架构上限；历史曲线已有晚期泛化跃升，见[学习历史核对](../../results/discoloop-learning-history-review-20261017/evidence.json)。用户指出末轮多余反馈后，已撤回“为保持前缀一致必须在末轮解码再编码”的解释：旧Q方案末轮读出改变了原DiscoLoop，旧产物完整保留。当前优先比较原文再编码门与连续净更新门的建议见[语义更正与方案排序](../loop_semantics_audit.md#terminal-readout-correction-20261017)，尚未据此启动新任务。独立云端终点核验未在本次补做。[协议](../development-artifacts/gated-loop-depth-extrapolation-v1/protocol.md)、[配置](../../configs/gated-loop-depth-extrapolation-v1.json)、[控制器终态](../../results/gated-loop-depth-extrapolation-v1/controller-state.json)。

<a id="norm-handoff-multihop-v1"></a>
**2026-10-17 CDT回顾更新：没有300k正式终点。** 本机控制器在R工程短训/重载阶段后因GPU4繁忙或显存不足退出为`needs_attention`，四臂正式训练未开始；下述排队状态保留为历史。本次只核对记录，未恢复执行。

**2026-10-17 02:06 CDT调度更新：已按用户指示新增候选GPU4。** 初查GPU4为空，但启动前复查被新VLLM进程占用，未强行运行。新控制器已接管等待，在GPU4/1/2中按实际释放情况调度；GPU1/2仍需完成旧队列及审计。35项CPU契约复用，33项调度/启动器测试通过，原包和科学预算不变。当前GPU预检、短训和正式训练均未开始。[最新调度与恢复入口](../development-artifacts/norm-handoff-multihop-v1/sn4622128200-execution.md)。

**2026-10-17 01:54 CDT：R/N/A/C已提交本机CUDA1、2接续队列，尚未开始GPU预检或正式训练。** 用户明确授权本机执行并指定两卡；GPU1依次R/A、GPU2依次N/C，每臂300k，共120万新增科学更新，F仅复用。本机35项模型/启动器契约、完整数据契约及26项接续/启动器测试通过；原包SHA256和内部文件核验通过。两卡当前仍被旧学习率实验使用，新控制器等待旧任务训练、独立审计及云端核验完成，再做F全量r1–4重放及四臂全部工程检查；通过后才开始科学训练。原镜像本机缺失，复用本项目现有重建镜像，主要依赖版本相同但镜像字节不同；原包不改，实际环境单独绑定，F重放不符即停止。当前新增正式更新0，无新科学结论。[本机执行与恢复记录](../development-artifacts/norm-handoff-multihop-v1/sn4622128200-execution.md)、[本机状态目录](../../results/norm-handoff-multihop-v1/sn4622128200)、[W&B项目](https://wandb.ai/zhusq20/llm-memory-editability)。

<a id="dual-gate-multihop-v1"></a>
**2026-10-17 CDT完成：P/G/D/V四条300k训练、独立终点/r1–4网格重载与云端审计全部通过。** 新增120万科学更新，F复用一层四轮DiscoLoop的300k终点，新增更新0；五组共同数据曝光。固定300k、推理4轮，全词表答案准确率如下，图B三跳为预定主指标：

| 条件 | 图B两跳 | 图B三跳 | 图B四跳 |
| --- | ---: | ---: | ---: |
| F 固定注入 | 55.60% | 0.06% | 0.00% |
| P 原文学习门 | 94.22% | 37.20% | 0.14% |
| G 共同输入标量注入门 | 75.82% | 0.06% | 0.02% |
| D 逐维连续门＋标量注入 | 91.54% | 0.02% | 0.00% |
| V 逐维注入门 | 89.72% | 0.00% | 0.06% |

本开发图/初始化中P的三跳优势明显，四跳仍低；不能据此声称普遍胜出，也不能把P与其它参数化的差异仅归因于“有门”：门输入、范围、初值或粒度亦不同。P初值.99是项目选择，原文未指定；各臂末轮均直接读取H。更复杂的D/V未改善本次主指标。全部池、答案＋EOS及推理网格保存在[完整比较](../../results/dual-gate-multihop-v1/local-completed/comparison.json)，见[完成清单](../../results/dual-gate-multihop-v1/local-completed/completion-manifest.json)与[云端总报告](https://wandb.ai/zhusq20/llm-memory-editability/runs/dg-report-608e048478)。父批云端收尾失败记录不追溯改写；本批报告独立绑定复用F的证据。

固定镜像43项CPU契约、Ruff、四臂CUDA短训和重载已通过，工程失败及修复保留于`attempts/preflight-repair-v2`，不重启原入口。无新增知识或独立图确认；门操作成本不同，不能据同等数据曝光宣称同等计算效率。[协议](../development-artifacts/dual-gate-multihop-v1/protocol.md)、[执行记录](../development-artifacts/dual-gate-multihop-v1/execution.md)、[冻结配置](../../configs/dual-gate-multihop-v1.json)。

**同批学习曲线补充。** P在图B三跳128k/256k/300k为0.08/10.30/37.20%；64k时P的两跳1.10%，G/D/V则71.52/77.00/79.68%。因此早期两跳提升更快没有对应本批更高的终点三跳。P的300k检查点按推理2/3/4轮，图B三跳为0/30.78/37.20%，两跳为93.14/94.26/94.22%；G/D/V的两跳从3到4轮均下降。所有预存节点与网格保留，主指标仍为固定300k/4轮，不选最好节点替代。见[九个记录点的完整曲线](../../results/gate-design-review-20261017/curves.csv)、[双面板图](../../results/gate-design-review-20261017/learning-curves.png)、[来源哈希与解释边界](../../results/gate-design-review-20261017/evidence.json)。本次只读取现有记录，没有训练或模型评价；方差诊断仍仅为0/8000步，不代表300k门值。

<a id="dual-gate-variance-diagnostic-v1"></a>
**2026-10-17 CDT：循环方差的文献核对与早期只读诊断完成，新增训练0。** 用户要求先查“hidden-state方差随loop累积”的论文，再诊断P/G/D/V。直接来源为[Wang等残差缩放](https://arxiv.org/abs/2606.18524v2)的条件平方范数分析、[LOOM §3.1](https://arxiv.org/html/2610.01153#S3.SS1)的跨轮方差观察；DiscoLoop的直接动机是表示交接，不能改写成整个状态的方差修正。定义、完整块与门的区别及点态界见[理论补充](../memory-scaling-theory.md#loop-variance-gates-20261017)，[来源核验](../../results/dual-gate-variance-diagnostic-v1/source-verification.json)。

测量前固定[协议](../../results/dual-gate-variance-diagnostic-v1/protocol.json)：五组共同0/8000步不可变检查点，全部11池各128题、共同抽样seed2026101701，四轮推理。独立CPU容器2线程、断网、数据/源码/检查点只读，eval/FP32前向、FP64统计；逐token先求通道样本方差再平均，另报未中心化二阶矩、均值平方、共享块与边界增益、两分支交叉项及门值。答案token和padding不进入激活统计。此为CPU诊断重放，不冒称原BF16/CUDA终点评分审计；没有门clamp、下游反事实或额外轮数扫描。

图B三跳128题，第8000步、第四轮完整块后：

| 条件 | 第一关系位置的通道方差 | 最后关系／答题位置的通道方差 | 第一关系位置第三次注入门g均值 |
| --- | ---: | ---: | ---: |
| F 固定DiscoLoop | 7.7926 | 5.9708 | 1.0000 |
| P 原文学习门 | 7.6959 | 6.0338 | 0.9940 |
| G 共同输入标量门 | 11.5358 | 4.3796 | 1.8479 |
| D 逐维连续门 | 10.8630 | 4.0222 | 1.8288 |
| V 逐维注入门 | 11.2168 | 3.9102 | 1.8181 |

**观察与边界。** G/D/V在第一关系位置的方差比F高约39%–48%，在答题位置却低约27%–35%；图B三跳全部因果查询token均值为F/P/G/D/V=6.7662/6.7284/6.3305/6.0722/6.2773。不能凭最后位置或全token均值宣称所有位置得到抑制。三个模型在答题位置首个交接g仅.001343/.000434/.000886，第一关系位置随后则增强注入。D在第一关系位置三次carry二阶矩增益为.99858/1.00876/1.00293；新增r当前接近1，并未单独产生大幅carry放大。以该位置第三次交接为例，D的方差精确分解为carry2.9688＋injection3.3954＋交叉项4.3800=10.7442；F为3.3600＋.9979＋2.9697=7.3276。这说明实际注入及正相关交叉项不能省略，不是仅凭r>1推断放大。

V在上述位置三次注入与原q的平均余弦均>.9997，当前逐维方向改动很小。完整块增量第3/4轮的平均余弦在图B三跳查询token上为F/P/G/D/V=.9569/.9556/.8625/.8572/.8591；观察到正相关，但四轮单检查点不能建立渐近平方增长律。图B三跳答题位置的低方差也未伴随已建立的成功：此128题子集F/P/G为0正确、D为1、V为2，不作架构排序或终点预告。方差不是“错误信息”的测量，较大值本身不证明有害或导致失败。

固定镜像Ruff与格式检查通过；各模型每池首32题的手动trace/native logits逐值一致，边界公式逐值一致，最大二阶矩/方差加法恒等式残差约5.4e-14；权重digest不变、全部12份数据/检查点SHA及原数据元信息复核通过。容器退出0、无GPU、原四条训练仍运行，已有冻结运行未改。完整[逐池/逐轮测量](../../results/dual-gate-variance-diagnostic-v1/measurements.json)、[摘要](../../results/dual-gate-variance-diagnostic-v1/summary.json)、[哈希核验](../../results/dual-gate-variance-diagnostic-v1/hash-validation.json)、[执行](../../results/dual-gate-variance-diagnostic-v1/execution.json)及[诊断脚本](../../scripts/diagnose_dual_gate_variance.py)保留。单图/初始化、早期8000步、抽样诊断，不涉及300k主指标选择、跨图确认或持续学习结论。

<a id="loop-discoloop-lr-control-v1"></a>
最新核对（2026-10-16 22:48 CDT）：本机GPU1普通Loop 5e-4、GPU2普通Loop 1e-3均已完成1000／300000正式步，损失与梯度有限，W&B在线正常；两组本机32步工程及七池独立重载全部通过。DiscoLoop两组等待对应队列自动接续，尚无新的终点结论。

**2026-10-16 CDT本机接续：四组迁至 GPU1／GPU2，尚无新终点。** 用户明确批准本机重建镜像；GPU1依次普通Loop 5e-4／DiscoLoop 2e-4，GPU2依次普通Loop 1e-3／DiscoLoop 5e-4，各300k、合计120万正式更新。215个冻结文件哈希通过，科学配方不变；本机重新进行每组32步工程及重载验证，最多另增128工程更新。原机两组及监控保持。实际镜像与原镜像字节不同，不宣称跨环境逐位一致；首轮NFS挂载失败发生在进入模型前，零正式更新并保留记录。状态与恢复入口见[本机执行记录](../development-artifacts/loop-discoloop-lr-control-v1/sn4622128200-execution.md)。

**2026-10-16 22:28 CDT：四组暂停交接，两组训练继续。** 用户要求将尚未正式启动的普通Loop 5e-4／1e-3与DiscoLoop 2e-4／5e-4移到另一GPU，四组均已置`paused_for_handoff`，不存在正式训练claim。只停止原调度器，GPU7普通Loop 2e-4与GPU6 DiscoLoop 1e-3容器保留，并已接上各自独立监控与终点审计/云端核验流程；后续实时状态读持久目录`handoff/<job>-status.json`及`jobs/<job>/training/status.json`。交接复用已有成功检查，保留冻结文件及科学预算；四组无自动启动。三项交接防重复/失败不重试/只读计划测试、Ruff及shell语法检查通过；目标GPU环境尚未执行验证。[运行脚本与说明](../development-artifacts/loop-discoloop-lr-control-v1/four-arm-handoff.md)。

**2026-10-16 21:06 CDT调度更新：GPU7首组正式训练已到1000/300000步。** 用户明确指定先用GPU7，其余空闲后自动并行；`loop-lr2e-4`已通过两项CUDA一致性、全尺寸32步短训及新容器七池逐token/NLL重载，正式300k容器与[W&B](https://wandb.ai/zhusq20/llm-memory-editability/runs/emvpeilq)正常运行，1000步损失与梯度有限。GPU6短暂释放时，`loop-lr5e-4`与`loop-lr1e-3`已自动通过CUDA契约，随后占用恢复，两组继续等preflight，其余三组仍排队。GPU7有其它任务，按用户指定共卡，既有任务与锁保留，耗时不当作独占效率基准；第一臂24GiB空闲工程门槛及本批独立租约、其余臂原空闲准入见[调度修订](../development-artifacts/loop-discoloop-lr-control-v1/gpu7-scheduling-amendment.md)。原215项冻结文件、科学配置与预算未改，没有重做已完成阶段；两个旧等待控制器与调度历史保留。当前没有新的科学结论。

**2026-10-16 CDT提交：普通Loop／DiscoLoop高学习率对照，尚无新科学结果。** 最终用户指定两架构各`2e-4、5e-4、1e-3`，每条300k，新增科学预算1,800,000；更早的低学习率四条未运行。保持原第0步权重、初始化791101、流792101、dropout种子20261012、原Wang图/七池、batch512、2000步warmup、答案/EOS监督、裁剪1.0及完整架构不变。AdamW衰减分别`0.05、0.02、0.01`，使每步解析`eta_t*lambda`与原`1e-4×0.1`一致，全部300k的累计乘子约0.0502869372；bias/LayerNorm仍零衰减。控制的是直接乘法衰减，不保证相对梯度的正则化强度或最终权重一致。

复用当前固定镜像下已完成且通过七池独立重载/云端核验的两条1e-4原生基础：普通Loop来自`ttcd-architecture-comparison-v1`基础，OI/IO/OO=34.30/43.73/3.29%；DiscoLoop来自`discoloop-success-restore-v1`基础，99.20/97.67/95.66%。初始化文件SHA与五份模型/训练核心源码逐字节一致；较早环境的终点另留历史，不混成同次执行。原基线、共同读取与后续写入任务均不重训。

代码及固定镜像117项CPU契约、Ruff检查与格式检查通过；一个缺原notebook的历史参考测试跳过，两个原测试会调用CUDA，明确安排各臂cuda-contracts阶段，原CPU容器无驱动失败保留。每臂须通过CUDA契约、全尺寸32步短训及新容器七池重载才进入正式300k；终点再独立重载、与1e-4曝光逐值核对及W&B完整canonical JSON核验。短训工程192更新另计，48次零学习率图捕获调用均恢复权重/Adam/RNG，不计新增学习步。

首次检查十卡均被占用，控制器已提交并只等实际独占空闲后动态并行，最多六臂；当前不把排队写成已开始科学训练。主读数固定300k OI，保留完整七池、预定曲线、失败、更新范数与成本；不按最高节点选结果，不自动重试/调参/延长。单图/初始化与DiscoLoop约2.05倍矩阵计算量的边界保留。[冻结配置](../../configs/loop-discoloop-lr-control-v1.json)、[协议](../development-artifacts/loop-discoloop-lr-control-v1/protocol.md)、[执行](../development-artifacts/loop-discoloop-lr-control-v1/execution.md)、[W&B项目](https://wandb.ai/zhusq20/llm-memory-editability)。持久目录为`/ossfs/workspace/llm-memory-editability/loop-discoloop-lr-control-v1`，恢复先读controller、science-schedule、source-manifest及实际容器，不重复提交。

<a id="loop-hop-depth-support-v1"></a>
**2026-10-17增补提交：2／3层×2／3／4轮及最小数据支持对照，尚无新增科学结果。** 原低支持2×2两架构复用；新增普通Loop／DiscoLoop各2×3、2×4、3×2、3×3、3×4共10条，再加2×2／3×2各两架构的更多组合支持4条，合计14×300k／420万新增更新。每hop支持10000／30000，原图、原子、训练前缀及43000评价题不变；采样保持原类别与原子索引、dummy及监督token逐batch一致，避免支持增加同时改变原子权重。固定镜像24项CPU契约及Ruff通过，尚未执行CUDA工程或正式新增训练。控制器因父批云端收尾未生成完整完成清单而退出为`needs_attention`；14臂仍pending，新增科学训练0；不重训原2×2、不改变旧批资源。单开发图／初始化、深度同时改变参数及FLOPs，不能预设数据或表示上限归因。[协议](../development-artifacts/loop-hop-depth-support-v1/protocol.md)、[执行](../development-artifacts/loop-hop-depth-support-v1/execution.md)、[准备收据](../../results/loop-hop-depth-support-v1/readiness.json)。

<a id="loop-hop-unified-v1"></a>
**2026-10-17 CDT终点回顾：八组均完成300k，独立终点及推理r1–4重载均通过。** 父批云端收尾仍为`needs_attention`，不写成全部云端验证通过。以下固定300k、使用各自训练轮数推理，报告图B全词表答案准确率；图B训练只提供原子事实。

| 模型 | 物理层×轮数 | 图B两跳 | 图B三跳 | 图B四跳 |
| --- | --- | ---: | ---: | ---: |
| 普通Loop | 1×2 | 0.44% | 0.00% | 0.02% |
| DiscoLoop | 1×2 | 0.00% | 0.00% | 0.00% |
| 普通Loop | 1×3 | 0.36% | 0.08% | 0.06% |
| DiscoLoop | 1×3 | 25.42% | 0.02% | 0.02% |
| 普通Loop | 1×4 | 0.18% | 0.18% | 0.08% |
| DiscoLoop | 1×4 | 55.60% | 0.06% | 0.00% |
| 普通Loop | 2×2 | 0.06% | 0.04% | 0.02% |
| DiscoLoop | 2×2 | 0.04% | 0.04% | 0.00% |

所有模型原子答案准确率99.96–100%，已见训练组合诊断99.8–100%；原子和已见组合掌握不等于未见组合泛化。固定DiscoLoop一层四轮的图A两/三/四跳为95.00/0.48/0.18%，两跳出现晚期提升，三/四跳仍低。加入原文学习门后的结果另见[门控批次](../results.md#dual-gate-multihop-v1)，不得将这八组固定注入结果推广到全部门方案。训练包含1–4跳，没有5跳，也不是未见长度外推。

本次回顾按12个独立训练终点去重：本批8条加门控新增4条；门控F复用本批一层四轮，不重复计数。[完整A/B端点及两种评分](../../results/multihop-300k-review-20261017/endpoints.csv)、[全部推理轮数网格](../../results/multihop-300k-review-20261017/inference-grid.csv)、[来源哈希与审计状态](../../results/multihop-300k-review-20261017/evidence.json)。单开发图/初始化，等数据曝光不等参数或FLOPs。本次仅读取已有产物，新增训练及模型评价0。[协议](../development-artifacts/loop-hop-unified-v1/protocol.md)、[历史初步读数](../../results/loop-hop-unified-v1/preliminary-20261017T0045Z.md)。

**2026-10-17代码与数据排查：未发现标签、输入、评分或DiscoLoop前向错误，但新旧实验不是单变量扩展。** 189项冻结哈希一致；独立原子字典复算73000条训练／评价记录真值全对；八组64k检查点CPU FP32重载、152项显式前向／因果／位置／padding检查通过，参数哈希与只读性通过；新旧五份模型及训练核心源码逐字节相同。旧成功配方是4×2／8次层计算、273600两跳训练题、随机具体问题留出、同图随机5%原子边只受原子监督、52463词表；本批执行2–4次层计算、每长度10000组合、同长度整组关系模板留出、整图B实体不参与组合、1013词表，多个条件同时改变。

旧DiscoLoop在64k的II／OO为36.03／0.05%，128k为99.90／85.78%，300k为99.93／95.66%；不能用旧终点与新中间节点直接判架构差距。图A异常仍真实：本次固定64k另测“关系序列见过、具体问题未见”的2048题，八组仅0–6题正确，不能只归因模板留出。旧成功II中未见模板的4题也全对，但子集很小。当前支持已给题目掌握而组合操作未迁移，不定位单一原因或表示下界。没有保留成功4×2新任务锚点是设计缺口；后续定位须另定最小配对桥接，未自动新增训练。本次仅CPU诊断、零正式更新、零追加GPU评价，原八组继续。[逐项对照与边界](../../results/loop-hop-unified-v1/code-data-audit-20261017/report.md)、[CPU原始结果](../../results/loop-hop-unified-v1/code-data-audit-20261017/output/result.json)。

<a id="supervised-alignment-development-v1"></a>
**2026-10-12原监督位置对齐、同位置KV与DiscoLoop：六组增补全部完成。** 北京时间08:03完成训练、独立审计、比较汇总及云端终点核验。复用前批普通Loop，只新增六臂各300,000步、合计1,800,000更新。固定镜像138项契约、CUDA基线输出/全部梯度/连续优化一致性及六臂32步短训练、新容器重载已通过。

共同设定：Wang φ7.2/wd.1、2000实体/200关系、273600条训练组合，宽768、四个完整GPT块循环两轮；同一第0步权重、样本流及dropout种子。以下均为固定300k终点的答案与EOS联合准确率（%），OI为预定主指标，不选择最佳节点。

| 方法 | 原子ID | 原子OOD | 训练II | 测试II | IO | OI | OO |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 普通Loop（共同基线） | 100.00 | 99.75 | 100.00 | 99.33 | 43.07 | 33.50 | 3.44 |
| 原标签方向对齐 | 100.00 | 99.95 | 99.97 | 98.93 | 36.57 | 41.23 | 3.74 |
| 仅同位置KV | 100.00 | 99.85 | 99.97 | 99.03 | 43.20 | 33.07 | 3.34 |
| 同位置KV＋原标签对齐 | 100.00 | 99.95 | 100.00 | 97.80 | 29.90 | 32.67 | 2.59 |
| 原标签中间CE | 100.00 | 99.95 | 99.87 | 87.27 | 7.50 | 7.03 | 0.50 |
| 同位置KV＋原标签CE | 99.97 | 100.00 | 99.93 | 85.00 | 8.37 | 6.10 | 0.35 |
| DiscoLoop | 99.97 | 98.80 | 99.90 | 99.93 | 97.53 | 99.30 | 95.66 |

**观察与边界。** DiscoLoop的OI/IO/OO为99.30/97.53/95.66%，较基线分别增加65.80/54.47/92.22个百分点；原子OOD为98.80%，低于基线99.75%。原标签对齐的OI增加7.73个百分点，但IO下降6.50个百分点；不能称全面改善。同位置KV单独OI差为−0.43个百分点，融合相对对齐单独下降8.57个百分点、相对KV单独下降0.40个百分点，四格交互为−8.13个百分点。对齐相对匹配CE的OI差：无KV为+34.20个百分点、有KV为+26.57个百分点；匹配CE自身严重退化，不能仅靠胜过CE声称通用优势。

**成本。** 每臂普通样本曝光153,474,816，有效输入token 594,323,514、普通监督token 306,949,632；相同步数与曝光不代表相同计算。估算矩阵训练FLOPs及实测训练时间如下，耗时为本次并行调度观测，不当作独占GPU基准。

| 方法 | 估算矩阵训练FLOPs（×10¹⁷） | 训练秒数 |
| --- | ---: | ---: |
| 普通Loop（共同基线） | 2.832 | 4691.5 |
| 方向对齐 | 3.079 | 5112.0 |
| 仅同位置KV | 2.833 | 4293.8 |
| 同位置KV＋原标签对齐 | 3.080 | 4725.7 |
| 原标签中间CE | 3.574 | 5716.4 |
| 同位置KV＋原标签CE | 3.575 | 5328.1 |
| DiscoLoop | 5.802 | 8060.8 |

DiscoLoop估算矩阵训练计算量约为基线2.05倍；推理也保留软解码再编码通道。原标签对齐采用原监督位置的概率加权`1-cos`；旧批采用全位置自预测的单位方向平方距离。新旧配方还改变KV范围，不能将差值单独归因于位置、标签、损失尺度或缓存范围。DiscoLoop为Wang设定上的GPT适配，不是原论文完整复现。

**复核。** 六条正式训练及六个新容器终点审计均正常退出，全七池逐token答案/EOS和检查点哈希核对通过。云端核验覆盖运行状态、300k步数及审计摘要，不代表逐个历史点核对；本次读取已有云端核验记录。2026-10-12北京时间08:35通过LM Docker API核对，两批容器均已退出且LM无运行容器。

[冻结配置](../../configs/supervised-alignment-development-v1.json)、[协议（本地归档）](../development-artifacts/supervised-alignment-development-v1/protocol.md)、[完成清单（本地归档）](../../results/supervised-alignment-development-v1/completion-manifest.json)、[完整统计（本地归档）](../../results/supervised-alignment-development-v1/factorial-summary.json)。GitHub保存本节结果、代码和配置；原始权重、预测、协议与执行产物按仓库规则保留在`/ossfs/workspace/research-recovery/supervised-alignment-development-v1/`等本地持久目录，本次不声称已上传新的数据归档。

W&B：[方向对齐](https://wandb.ai/zhusq20/llm-memory-editability/runs/0o7t9wud)；[同位置KV＋原标签对齐](https://wandb.ai/zhusq20/llm-memory-editability/runs/hbru0rg3)；[原标签中间CE](https://wandb.ai/zhusq20/llm-memory-editability/runs/kglnutwj)；[同位置KV＋原标签CE](https://wandb.ai/zhusq20/llm-memory-editability/runs/w8c0zjlb)；[仅同位置KV](https://wandb.ai/zhusq20/llm-memory-editability/runs/lr3lx6gc)；[DiscoLoop](https://wandb.ai/zhusq20/llm-memory-editability/runs/ci3x3pk1)。

证据SHA-256：

- `factorial-summary.json`：`b8b0305d03129d49def026c04d4dade3f8428791adc5484e7cda125ad583f622`。
- `completion-manifest.json`：`fbceeb46f826c6eb14fde5f6e0cb2115a31735c2ef26e8353167a3394be38afd`。
- `cloud-verification.json`：`183501651847dbc059ab68289ceb71c3928d52bb85a0b837264d6e336c4af69e`。

只支持一个已观察知识图、一个初始化上的开发观察；查询或检查点不是独立世界。300k结果不能证明长期无效，组合成绩不能代替知识改写后的传播与保持检验。不自动追加长训、新图或旧KV网格。

<a id="self-decode-alignment-development-v1"></a>
**2026-10-12自解码对齐×全前缀KV：六臂全部完成。** 北京时间06:15完成训练、独立审计与云端终点核验，各300,000步、合计1,800,000更新。固定镜像58项契约、六臂CUDA短训练与新容器重载已通过。首次cgroup只读失败发生在模型执行前，失败及独立挂载修复记录保留，不计作科学负结果。

共同设定：Wang φ7.2/wd.1、2000实体/200关系、273600条训练组合，宽768、四个完整GPT块循环两轮；同一第0步权重、样本流及dropout种子。以下均为固定300k终点的答案与EOS联合准确率（%），OI为预定主指标，不选择最佳节点。

| 方法 | 原子ID | 原子OOD | 训练II | 测试II | IO | OI | OO |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 普通Loop（共同基线） | 100.00 | 99.75 | 100.00 | 99.33 | 43.07 | 33.50 | 3.44 |
| 自预测CE | 100.00 | 100.00 | 100.00 | 99.43 | 55.67 | 35.87 | 5.04 |
| 自解码方向对齐 | 100.00 | 99.85 | 99.90 | 98.50 | 5.77 | 35.77 | 1.00 |
| 全前缀KV | 100.00 | 99.85 | 99.93 | 97.77 | 30.20 | 25.90 | 1.75 |
| 全前缀KV＋自预测CE | 100.00 | 99.90 | 100.00 | 98.77 | 43.30 | 30.40 | 2.69 |
| 全前缀KV＋自解码对齐 | 100.00 | 99.95 | 100.00 | 95.20 | 3.23 | 17.73 | 0.75 |

**观察与边界。** 自解码对齐的OI较基线增加2.27个百分点，但较自预测CE低0.10个百分点；IO从43.07%降到5.77%。融合OI为17.73%，低于KV单独25.90%和对齐单独35.77%；四格交互为−10.43个百分点。无KV/有KV条件下，对齐相对CE的OI差分别为−0.10/−12.67个百分点。未建立方向约束的额外作用或融合收益，保留其他池退化结果。

**成本。** 每臂普通样本曝光153,474,816，有效输入token 594,323,514、普通监督token 306,949,632；相同步数与曝光不代表相同计算。估算矩阵训练FLOPs及实测训练时间如下，耗时为本次并行调度观测，不当作独占GPU基准。

| 方法 | 估算矩阵训练FLOPs（×10¹⁷） | 训练秒数 |
| --- | ---: | ---: |
| 普通Loop（共同基线） | 2.832 | 4691.5 |
| 自预测CE | 4.812 | 7174.7 |
| 方向对齐 | 3.327 | 5509.4 |
| 全前缀KV | 2.833 | 4290.2 |
| 全前缀KV＋自预测CE | 4.813 | 6804.8 |
| 全前缀KV＋自解码对齐 | 3.328 | 5120.9 |

**复核。** 六条正式训练及六个新容器终点审计均正常退出，全七池逐token答案/EOS和检查点哈希核对通过。云端核验覆盖运行状态、300k步数及审计摘要，不代表逐个历史点核对；本次读取已有云端核验记录。2026-10-12北京时间08:35通过LM Docker API核对，两批容器均已退出且LM无运行容器。

[冻结配置](../../configs/self-decode-alignment-development-v1.json)、[协议（本地归档）](../development-artifacts/self-decode-alignment-development-v1/protocol.md)、[完成清单（本地归档）](../../results/self-decode-alignment-development-v1/completion-manifest.json)、[完整统计（本地归档）](../../results/self-decode-alignment-development-v1/factorial-summary.json)。GitHub保存本节结果、代码和配置；原始权重、预测、协议与执行产物按仓库规则保留在`/ossfs/workspace/research-recovery/self-decode-alignment-development-v1/`等本地持久目录，本次不声称已上传新的数据归档。

W&B：[普通Loop（共同基线）](https://wandb.ai/zhusq20/llm-memory-editability/runs/958fikwd)；[自预测CE](https://wandb.ai/zhusq20/llm-memory-editability/runs/bl1sffdw)；[方向对齐](https://wandb.ai/zhusq20/llm-memory-editability/runs/d4ac6c6r)；[全前缀KV](https://wandb.ai/zhusq20/llm-memory-editability/runs/vuxpe3de)；[全前缀KV＋自预测CE](https://wandb.ai/zhusq20/llm-memory-editability/runs/3mq2bck9)；[全前缀KV＋自解码对齐](https://wandb.ai/zhusq20/llm-memory-editability/runs/xweqi1as)。

证据SHA-256：

- `factorial-summary.json`：`b8aca4fb72bfd227f1457352c705f72223cfe7c46f17dcdd11a62806e310bd36`。
- `completion-manifest.json`：`3b0a9bc034fa443b2f312482fcab82dfe3b057e7d04eec177d969e8e5295745f`。
- `cloud-verification.json`：`2cef0d2de84bb5d538a2b6051f15964d4cd3e24ca7f5b9795ab9a8209a05bb24`。

只支持一个已观察知识图、一个初始化上的开发观察；查询或检查点不是独立世界。300k结果不能证明长期无效，组合成绩不能代替知识改写后的传播与保持检验。不自动追加长训、新图或旧KV网格。


<a id="roadmap-history"></a>
## 整理前路线图

以下为整理前快照，当前规划见[Loop路线图](../loop-roadmap.md)。

# Loop路线：循环Transformer跨跳复用参数知识的条件

更新：2026-10-22（UTC）。本文是Loop方向的唯一规划，与在写另一篇论文的[主路线图](../roadmap.md)分开维护。完成结果写入[结果汇总](../results.md)，推导在[理论依据](../memory-scaling-theory.md)，执行细节在各批协议。

**选题提醒（用户2026-10-22）：** “什么样的交接，能让一次学会的知识成为可反复调用的推理步骤？”已经反复讨论，并已有大量实验积累。它可以继续作为论文主线，但不能再当作新 idea 或阅读文献后发现的核心突破。后续探索必须先看历史跑过什么实验、检验过什么假设，再寻找超出现有认识的新突破点；提出新 idea 前执行[历史核对要求](../loop-roadmap.md#idea-history)。

<a id="thesis"></a>
<a id="goal"></a>
## 1. 论点

**工作假说：** 循环 Transformer 能否复用只学过原子形式的知识，取决于每轮的查找和交接是否与原子学习兼容。共享参数、经词表的交接、位置处理是三个候选条件；归一化历史聚合与接收分数决定多轮计算能否继续改善答案。

当前 invariant 的打分只接收熵、最大概率、相邻解码一致度、状态相对变化和余弦五个量，再由小 MLP 决定接收权重。这里的“不变”指这些输入在对应的词表重标号/共同正交变换下不变，不能直接推成完整网络对任意身份或数据分布不敏感。方法仍使用实体 embedding 与实际 Transformer 块。

现有单图结果支持这个方案继续检验，但没有证明上述条件必要、充分，或已经学到普遍的“一轮一跳”算法。现在优先回答：结果能否重复、是否依赖数据分布、同一道题多算是否持续有用。

<a id="principles"></a>
## 2. 原则（用户2026-10-20）

- 架构与训练以[InfiLoop](https://arxiv.org/abs/2610.11570)为主要参照，并提出新的架构构思。
- **监督只有答案与EOS；** 深监督也只在每段末监督答案。
- **不使用的信息：** 桥实体、跳数、轮次、位置角色、图身份等标签；架构中的控制信号也不能读取这些信息。
- **以机制解释为目标：** 每批先写明区分竞争解释的关键比较，不以扩大网格代替判断（[研究原则](../../.agents/skills/llm-memory-research/SKILL.md)）。

<a id="evidence"></a>
## 3. 当前证据

以下 native 四臂均已完成 150 万步；参数 EMA，固定 R8。A 的组合参加训练，B 只学原子。来自同一已观察世界、同一初始化；完整方法与历史见[阶段总结 PDF](loop-transformer-summary-20261021.pdf)。

| 交接 | A 留出四跳 | B 两跳 | B 三跳 | B 四跳 |
| --- | ---: | ---: | ---: | ---: |
| recipe-additive | 0.08% | 40.36% | 0.02% | 0.04% |
| ema（常数打分） | 100% | 97.40% | 54.10% | 10.78% |
| content | 100% | 15.28% | 0.28% | 0.14% |
| invariant | 100% | 99.32% | 96.12% | 79.42% |

- **终点优势不等于训练稳定。** invariant 的 B 四跳在 130 万步为 93.74%，终点 79.42%；相邻记录节点最大下降 41.50pp。content 早期高分后崩塌。
- **三个中间checkpoint的机制诊断已完成，零训练。** 25.6万→70万→130万步既有候选生成变化，也有中间实体被后续候选覆盖的差异；在共同1630题的局部条件下，70万步改错1244题、130万步仅6题。但单换β、移植打分或阻止特定覆盖都未救回低谷四跳，不能把训练退化归结为某一交接参数漂移。γ还受状态尺度影响。见[完整对照与限定](../results.md#invariant-checkpoint-dip-20261022)；不据此追加训练网格。
- **第二跳读取诊断已完成，仍为零训练。** 70万步第一轮中间实体100%可读，但第二轮 attention 约41%分给原起点；只强制读取现有中间位置，第二跳候选由74.40%升到98.94%，同一 MLP 对正常原子事实输入仍100%正确。只清零原起点而不重分配权重反而下降，A世界相近的 attention 比例仍能答对，支持读取与 MLP 输入接口的失配，不能只凭权重比例解释泛化。第二跳局部修复后四跳 R8 仅6.86→13.86%，后续保持与各跳仍有问题。见[三checkpoint、全池对照与负结果](../results.md#invariant-second-hop-20261022)；不追加训练。
- **多算已有有限收益。** invariant B 四跳 R8→R32 为 79.42→81.98%；5000 道题中修正 153 道、丢失 25 道。B 两/三跳各净降 0.14pp。因此既不能再写“所有模型多跑完全不变”，也不能写“答对后永不退化”。这些是旧保存预测的再分析，见[逐题统计](../../results/loop-invariant-robustness-v1/existing-evidence.json)。
- **统一降低 attention 温度有推理收益，但没有消除低谷。** 三个已有checkpoint对所有位置、head和轮次使用同一固定温度，零训练；τ=1.0→0.5时B四跳R8为66.10→69.60%、6.86→10.46%、93.74→97.62%。全部五档温度的A留出四跳与B原子R8保持100%，答案+EOS与答案准确率相同。70万步第二跳候选74.40→87.20%，完整四跳仍低；R16也仅10.56%。因此锐化读取是一个有效但有限的推理调整，不能称训练低谷已解决；不据此扩大温度网格或改变正在训练的配方。见[完整结果与逐题得失](../results.md#invariant-attention-temperature-20261022)。
- **新增长循环测试出现退化。** 全量 R256 时，invariant B 原子/两/三/四跳为 91.22/88.46/85.70/74.18%；四跳 R8→R256 修正 135 道、丢失 397 道。每池固定 256 题到 R16,384，B 四跳从子集 R8 的 78.91% 降至 74.61%；A 留出组合保持 100%。三/四跳仍强于 ema，原子/两跳的长循环稳定性弱于 ema。见[结果与核验](../results.md#loop-invariant-robustness-v1)，不能称持续深度扩展成功。
- **更长问题仍弱。** invariant 终点 R16 的 A/B 五跳为 19.24/0.66%，六至八跳接近随机；既有位置平移显示明显影响。这个轴与同一问题增加轮数分开。
- **RoPE 没有统一解决问题。** position 三臂均完成。R8 EMA：ema-rope 的 A 四跳/B 三跳/B 四跳为 4.64/63.98/0%；disco-rope-old 为 99.96/88.86/45.14%；ema-absolute-old 为 100/25.84/25.96%。不能沿用旧“位置换掉即可深度扩展”的结论。

上述终点来自持久产物与保存预测核对；invariant/ema 的新环境全量 R4/R8 独立模型重放已逐 token 通过。训练结束、独立重放和 W&B 云端核验分别记录。

<a id="current"></a>
## 4. 当前批次

<a id="raw-state-handoff"></a>
**2026-10-22 用户授权原始两路状态门：`loop-state-handoff-development-v1`。** 唯一新增臂用 `[旧聚合状态; 本轮候选] → 16维GELU → 标量分数` 代替五个手工统计量；候选生成、历史归一化聚合、原世界/初始化、样本流与答案/EOS监督均沿用 native 配方。新增一条150万步，直接复用 invariant、ema、content，不扩大种子或数据网格。新版本不称 invariant；原始状态信息与非线性参数量同时改变，不能仅凭胜负归因于不变性。见[协议](../development-artifacts/loop-state-handoff-development-v1/protocol.md)与[持久执行状态](../../results/loop-state-handoff-development-v1/runtime/controller-state.json)。14:09 UTC已在GPU0正式开训；139项CPU契约、CUDA初值/梯度、32步短训和33文件独立重放通过。终点重载与云端核验待训练完成。

**2026-10-22 18:06 UTC调度扩展。** 用户要求找空卡运行已准备的实验；GPU2已通过CUDA及旧预测重放，普通Loop的grid/depth/cloud于18:10 UTC全部退出0；18:11 UTC多对一ema完成短训及新容器重载，进入150万步正式训练。新控制器PID2638715采用5/6/8/2资源池，仅调度实际空闲卡，本机该用户全部批次合计最多四卡；旧调度器PID2075489已退出，原训练容器不中断，新初始化1、2仍最后，科学预算不变。GPU3预检137且非Docker报告的OOM，保留失败不重试；GPU1/4/7仍有其他项目预约。恢复先读[调度修订](../../results/loop-queue-target-20261021/runtime/schedule-amendment.json)与[实时状态](../../results/loop-queue-target-20261021/runtime/controller-state.json)，再核对容器，不重启旧入口。核验与失败解释见[结果记录](../results.md#loop-queue-expansion-20261022)；下文三卡提交记载为历史。

**2026-10-21 19:22 CDT目标机接续。** 用户指定`sn4622129202`的GPU5/6/8，并将新初始化1、2排到最后。三卡各一个实验，首轮依次为新世界invariant、原世界普通Loop、多对一invariant；原清单15条各150万步及3项旧深度评测由独立控制器PID2075489接管。前八条非初始化训练及三项旧评测全部完成后，才启动七条新初始化训练。1,967个原文件、三卡CUDA计算、四组各43,000题旧R4/R8预测重放均通过；重建镜像完整依赖与源机记录相同，科学配方不改。每条仍先短训和独立重载，再正式训练、grid、depth、cloud。当前阶段见[目标机交接](../../results/loop-queue-target-20261021/RUNNING.txt)与[控制器状态](../../results/loop-queue-target-20261021/runtime/controller-state.json)。源两条在训保持，其后续评测待源终点和退出证据；旧队列不重启。下文停止记载为迁移前历史。

**`loop-invariant-robustness-v1`**（[冻结协议](../development-artifacts/loop-invariant-robustness-v1/protocol.md)）：用户 2026-10-21 明确授权测试 invariant 的稳定性、换数据后的表现和 InfiLoop 同目标的深度扩展。

| 比较 | 具体改变 | 数量 |
| --- | --- | ---: |
| 重复性 | 原世界增加两个骨干/打分网络初始化；其余随机流保持 | invariant/ema 各两条 |
| 新世界 | 重新生成同规模置换图及查询 | invariant/ema 各一条 |
| 新关系分布 | 保留新世界查询，将双射换成随机确定函数 | invariant/ema 各一条 |
| 测试时深度 | 旧四臂 EMA 与 invariant raw；新臂终点也测 | 全量到 R256、固定子集到 R16384；零新增训练 |

八条新增训练各 150 万步，共 1200 万更新。完整学习曲线、固定终点、原子掌握、配对差、答对后丢失与计算成本均保留。新数据结果未出之前不宣布稳定或数据无关。2026-10-21 22:29 UTC，init-1-invariant 与 new-world-ema 已通过短训和重载，进入正式训练；其余六条于18:06 CDT按用户要求停止本机排队，待迁到另一机器三张GPU。原控制器PID3310766已退出，两个训练容器继续运行，后续grid/depth/cloud待接续。恢复先读持久区 `queue-transfer-20261021.json` 和[迁移交接](../../results/loop-queue-transfer-20261021/HANDOFF.txt)；前两次工程失败和已完成的旧模型深度评测保留，不重提原队列。

<a id="position-invariance"></a>
前批 `loop-native-handoff-development-v1` 四臂与 `loop-position-invariance-development-v1` 三臂已完成；本批复用，不重启。旧 GPU 调度记录只作历史，不用于当前资源分配。

**用户随后要求补普通 Loop 与基础 DiscoLoop：`loop-baseline-robustness-v1`**（[协议](../development-artifacts/loop-baseline-robustness-v1/protocol.md)）。普通 Loop 只传递共享块输出；基础 DiscoLoop 为固定 α=1 的软解码再编码加法，无学习 gate。两者覆盖原世界三个初始化、新置换世界、多对一世界；固定与 invariant 相同的当前训练配方。原世界基础 DiscoLoop 已是 recipe-additive，直接复用；只新增缺失的九条 × 150 万步，共 1350 万更新。两批原定共17条、2550万更新，其中两条继续训练；余下15条、2250万更新已停止本机排队待迁移。基线终点沿用相同深度测试与稳定性判读，原始配方的较强 DiscoLoop 三跳也另列，避免只用当前配方的弱结果作对照。57 项契约与两种模型的 CUDA 前向/全部梯度一致性通过；两条短训独立重载的 66 组/60,432 查询核验通过。九条原队列于18:06 CDT停止（控制器PID3423943已退出），尚无新增科学训练结果；另有三项旧模型深度评测一起停止排队。恢复读取本批持久区 queue-transfer-20261021、controller-process 与 controller-state；按迁移清单接续，不重启原入口。已有分配方结果见[比较表](../../results/loop-baseline-robustness-v1/comparison.md)及[比较 PDF](../../results/loop-baseline-robustness-v1/baseline-comparison.pdf)。

<a id="next"></a>
## 5. 判读与下一步

1. 先看三个初始化的固定终点及最差值，不能用最好检查点代表稳定性。
2. 换数据先检查原子知识是否学会，再比较组合与 invariant−ema 差。多对一数据另外报告答案分布与仅关系序列的猜测基准。
3. 固定问题的深度曲线分清“保持旧正确答案”和“新增正确答案”；固定子集与全量池分别报告。
4. 依据这批结果再确定论文主张和后续方向。数独/迷宫等另一任务、匹配打分 MLP 的机制对照、第三层编辑、自然语言格式均未加入当前预算，不自动排队。

<a id="scope"></a>
## 6. 范围与避免重复

当前授权包括 invariant 的重复性、数据分布变化与深度检验，以及10-22单独授权的一条原始两路状态门；不展开学习门的配方网格。旧首轮 KV、mHC、MoE、中间标签监督、规则门、轮次标签、课程与学习式停止不入队。已完成训练和旧预测直接复用。

<a id="idea-history"></a>
**提出新 idea 前的历史核对要求（用户2026-10-22）：**

- **先读历史，再构思。** 先核对本路线图、[结果汇总](../results.md)、[阶段总结](loop-transformer-summary-20261021.pdf)及相关批次协议，逐项确认旧假设、实际改动、正负结果、失败原因和仍在等待的结果。不能只看最新成功模型或文献摘要。
- **按问题核对已有工作。** 交接与表示问题要回看[自解码对齐](../results.md#self-decode-alignment-development-v1)、[监督对齐与KV](../results.md#supervised-alignment-development-v1)及[架构评审](loop-native-handoff-review-20261020.md)；候选、门与保持问题要回看[门控系列](../results.md#candidate-update-gate-multihop-v1)、本页[当前证据](../loop-roadmap.md#evidence)和[训练低谷诊断](../results.md#invariant-checkpoint-dip-20261022)；读取问题要回看[第二跳诊断](../results.md#invariant-second-hop-20261022)；深度与位置问题要回看已有轮数、长跳和位置实验。涉及其他旧方向时继续查对应记录，不把这些入口当作完整历史。
- **每个候选明确写出增量。** 用“已有假设与实验 → 尚未解释的具体现象 → 新机制或实质改变的条件 → 能区分新旧解释的关键比较”说明价值，并给出历史批次或结果链接。换名称、换架构组合、重复强调交接重要、扩大门/KV/轮数网格，本身都不构成新突破。
- **文献启发必须落到历史之外的缺口。** 先说明论文的机制、训练条件和关键比较与我们已做工作的对应关系，再说明它新增了什么。已有问题尚未完全解决，不等于它是新问题；已有工作的总结、复核与当前在跑的确认实验，应如实标为这些用途。
- **没有实质新点时直接说明。** 不为给出 idea 而把旧问题重新包装；重访旧问题需指出足以改变原判断的新证据或关键条件，并解释为什么旧实验不能回答。提出想法不自动增加训练预算，也不重启用户已结束的配方扩展。

<a id="docs"></a>
## 7. 入口

| 入口 | 用途 |
| --- | --- |
| [当前批协议](../development-artifacts/loop-invariant-robustness-v1/protocol.md) | 矩阵、预算、主要终点与调度规则 |
| [阶段总结 PDF](loop-transformer-summary-20261021.pdf) | Loop 方法、历史实验及关键结论 |
| [架构图](../../results/invariant-architecture-diagram-20261021/invariant-architecture.pdf) | invariant 的实际计算 |
| [Loop 架构评审](loop-native-handoff-review-20261020.md) | 对照 InfiLoop 的设计依据 |
| [结果汇总](../results.md) | 各批科学证据 |
| [主路线图](../roadmap.md) | 与本方向分开维护的论文规划 |

<a id="entry-history"></a>
## 入口历史

以下旧入口原文仅供核对授权、失败与迁移经过；已由[当前入口](../../AGENTS.md#loop研究入口与恢复)的恢复规则取代。

2026-10-22用户进一步要求找空卡运行已准备的实验。`loop-queue-target-20261021`于18:06 UTC由控制器PID2638715接管调度，原PID2075489已退出；新增GPU2，资源池为5/6/8/2，按实际空闲接续阶段，本机该用户所有批次合计最多四张GPU。原三条训练及记录修复器ID/PID/启动时间全部不变；普通Loop已完成150万步，GPU2接续的grid/depth/cloud于18:10 UTC全部退出0；18:11 UTC多对一ema通过32步短训和新容器重载，进入原定150万步正式训练。新初始化1、2仍最后，不新增训练预算。固定镜像10项调度检查与Ruff通过，GPU2真实CUDA及旧invariant raw/EMA各43,000题R4/R8精确重放通过。GPU3预检退出137、OOMKilled=false，未进入新科学训练；与运行中的四卡配额守护默认规则一致，但根权限配置/信号日志不可读，未确认终止来源，不重试、不绕过。GPU1/4/7仍有其他项目预约。恢复先读持久区`schedule-amendment.json`、`scheduling-expansion-20261022/`、最新`controller-process.json`与`controller-state.json`，再核对实际容器；禁止重启旧控制器或已完成训练。state-mlp已于14:09 UTC在GPU0正式开训，CUDA/32步短训/33文件独立重放均通过，原控制器PID3084195保持。本段覆盖下方候卡与旧三卡控制器历史状态。

2026-10-22用户明确授权用直接读取旧状态与候选的小MLP替代invariant的五个手工特征，并要求修改代码、运行实验和解释设计。`loop-state-handoff-development-v1`已冻结826文件，固定镜像CPU139项契约及Ruff通过；独立控制器PID3084195已提交于sn4622129202，当前`waiting_resource`，正式训练0步，CUDA检查/32步短训与重放尚待空卡。只新增原世界、原初始化的`state-mlp`一条150万步，复用旧invariant/ema/content，不重训。新分数为`concat(h_old,c_new)→16维GELU→1`，原候选、历史聚合、答案/EOS配方不变；不称invariant，也不将容量与输入共同改变的结果单独归因于不变性。十卡均有活动任务；不得抢占，GPU5/6/8在旧迁移批全部完成前保留其优先级。恢复先读[本批交接](../../results/loop-state-handoff-development-v1/RUNNING.txt)、`runtime/freeze.json`、`controller-process.json`、`controller-state.json`、`execution/`与`artifacts/`，再核对LM容器，不重复提交。冻结检查曾在复制完成前启动而失败，原日志保留；最终针对冻结副本的139项检查已通过，未涉及GPU或科学训练。本条只授权这一新门，不展开旧门/KV/数据/种子网格。

2026-10-21 19:22 CDT用户指定目标机`sn4622129202`的GPU5/6/8，并要求新初始化1、2最后，授权本机重建镜像。`loop-queue-target-20261021`控制器PID2075489已接收原15条×150万步和3项旧深度评测；首轮GPU5新世界invariant、GPU6原世界普通Loop、GPU8多对一invariant。其余非初始化任务全部完成后才运行七条新初始化任务，每卡一个实验。1,967文件哈希、三卡真实CUDA、四组各43,000题R4/R8旧预测精确重放通过；实际镜像`c77af324439d`与源镜像字节不同，完整pip依赖相同。恢复先读[目标机交接](../../results/loop-queue-target-20261021/RUNNING.txt)和该持久批次`schedule.json`、`controller-process.json`、`controller-state.json`，再核对LM Docker；源两批队列继续停用，不重复提交。源机两条在训未操作，其grid/depth/cloud仍待源终点和退出0证据后接续。该段覆盖下方“目标尚未启动”历史状态；当前训练阶段以控制器和各run/status为准。

19:27 CDT首轮三条均已完成至少1,000正式更新，短训/重载、初值哈希通过。19:30 CDT发现W&B省略空字典，已由独立无GPU容器`lqt-1021-tracking-config-repair`只补回原空字段，不改科学训练或曲线摘要；未来新任务同样核验。四项调度/元数据保护测试通过，前后证据在该批`tracking-repair*`和`cloud-startup-after-repair.json`；恢复时连同核对该记录容器，不能为修记录重启训练。

2026-10-21 18:06 CDT 用户要求停止排队任务，迁到另一机器三张 GPU 并行执行。已核验并终止两个调度器 PID3310766/3423943，取消源机器的六条 invariant/ema 与九条普通 Loop/基础 DiscoLoop 队列，合计15条各150万步；另停止三项旧模型深度评测排队。源机器 `init-1-invariant` 与 `new-world-ema` 两个训练容器保持原ID/PID继续运行；其后续 grid/depth/cloud 尚待接续，不重训。旧 invariant/ema 参数EMA深度已完成，不能重复。恢复优先读[迁移交接](../../results/loop-queue-transfer-20261021/HANDOFF.txt)、[停止证据](../../results/loop-queue-transfer-20261021/stop-result.json)及两批持久区 `queue-transfer-20261021.json`，本条覆盖下方旧“等待资源/活动控制器”状态；禁止重启旧队列。目标机器尚未启动，数据、配置和冻结科学源码不变。

2026-10-21用户进一步要求加入“discoloop（不加gate的最基础版本）、普通loop”。`loop-baseline-robustness-v1`已提交独立队列（控制器PID3423943，sn4622128200），九条各150万步、共1350万新增更新，当前全部等待空闲GPU。覆盖原世界三个初始化、新置换与多对一世界；复用原世界基础DiscoLoop（native的recipe-additive）及全部invariant/ema，不重训这些对照。普通Loop只传递隐藏状态；基础DiscoLoop固定α=1、温度1，无学习gate。57项契约、全尺寸CUDA前向/全部梯度一致性、两条32步工程短训及独立重载66组/60,432查询全部通过；196份工程产物已归档。系统Python缺少numpy的调度导入失败保留在`controller-import-failure.log`，改用原conda解释器；未创建科学容器、未改变冻结源码。恢复读本批持久区`bootstrap-amendment.json`、`controller-process.json`、`controller-state.json`，核对LM Docker，不重复提交。前批两条训练继续执行。旧固定四轮/raw与当前配方/EMA分开报告：基础DiscoLoop旧配方B三跳R8为88.26%，当前为0.02%，不能只取弱配方当基线。协议见[基线比较](../development-artifacts/loop-baseline-robustness-v1/protocol.md)，结果见[汇总](../results.md#loop-baseline-robustness-v1)。尚无新增基线科学结果。

2026-10-21用户授权测试 invariant 的重复性、换数据敏感性与 InfiLoop 同目标的测试时深度扩展。`loop-invariant-robustness-v1`固定八条新增训练（两个新初始化、新置换世界、多对一关系分布，各配对 invariant/ema），各150万步；旧权重复用，全池到R256、固定子集到R16384。44项契约通过。当前活动控制器在持久区该批 `attempt-3/`，恢复先读根目录 `schedule-amendment.json`，再读 `attempt-3/controller-process.json`、`controller-state.json`并核对LM Docker；不要重提。前两次运行环境/入口失败在模型训练前发生，记录保留；`attempt-2`的旧invariant、ema三档深度评测均已完成、容器退出0，第三次控制器复用，不能重启。已有新全池结果：invariant B四跳R8/R32/R256为79.42/81.98/74.18%，B原子R256为91.22%；ema B四跳R256为8.46%。固定256题四跳R8→R16384为78.91→74.61%；两模型全部固定子集R1024后答案均不再变化。两模型R4/R8各43,000查询逐token重放通过，66份保存预测、137,211,904答案复算通过。22:29 UTC，init-1-invariant与new-world-ema通过32步短训及独立重载进入正式训练，首个1000步损失/梯度有限，W&B已连接；其余六条等待资源。新初始化与换数据的终点结果尚未产生。协议与判读见[当前协议](../development-artifacts/loop-invariant-robustness-v1/protocol.md)、[结果](../results.md#loop-invariant-robustness-v1)及[Loop路线图](../loop-roadmap.md)，不写入主路线图。

2026-10-20运行中：用户授权`loop-position-invariance-development-v1`，三臂各150万步：ema-rope、disco-rope-old、ema-absolute-old，检验位置不变性与"一轮一跳"的来源。20:49 UTC在sn4622129202提交（控制器pid 535151），GPU 6/7立即开始，GPU 8等content臂及其5–8跳评测结束后再开始。恢复时：
- 先读持久区`loop-position-invariance-development-v1/controller-process.json`与`controller-state.json`，再核对LM Docker进程；
- 不重复提交；
- 设计与判读见[协议](../development-artifacts/loop-position-invariance-development-v1/protocol.md)，规划见[Loop路线图](../loop-roadmap.md)。

2026-10-20运行中：用户授权`loop-native-handoff-development-v1`，四臂各150万步。依据是对照InfiLoop的Loop架构评审，只监督答案，不把算法结构写进监督。04:05 UTC在sn4622129202的GPU 6–9上提交，按smoke→train→grid执行。恢复时：
- 先读持久区`loop-native-handoff-development-v1/controller-process.json`与`controller-state.json`，再核对实际LM Docker进程；
- 不重复提交；
- 设计与判读见[协议](../development-artifacts/loop-native-handoff-development-v1/protocol.md)；该方向的规划在单独的[Loop路线图](../loop-roadmap.md)，不写进主路线图。

2026-10-12完成状态更新：`self-decode-alignment-development-v1`于北京时间06:15、`supervised-alignment-development-v1`于08:03全部完成，各六臂300k步、合计360万新增更新；七池独立重载与W&B云端终点状态/步数/审计摘要核验均通过。08:35实际LM Docker无运行容器，两批训练及审计退出码均为0。不重启旧入口。DiscoLoop OI/IO/OO为99.30/97.53/95.66%，估算矩阵训练FLOPs为基线约2.05倍；原监督对齐OI41.23%（基线33.50%），但IO下降；两批KV融合均未超过对齐单独。完整七池及成本见[结果](../results.md#supervised-alignment-development-v1)。只是一张已观察图/一个初始化，不证明通用效率、唯一机制或更新传播；没有追加新训练。以下启动记载保留历史，本段完成状态优先。

2026-10-12用户随后采纳六组增补并要求编写代码、并行启动。`supervised-alignment-development-v1`已提交：原监督标签概率加权的对齐/中间CE各有无同位置首轮KV两组，另加仅同位置KV与DiscoLoop，共六臂各300k步、180万新增更新；复用前批普通Loop基线。固定镜像138项契约、CUDA基线逐值/全部梯度/连续优化一致性、六臂全尺寸32步短训与新容器七池重载全部通过。北京时间05:36启动控制器，先用GPU6/7/0/3运行四臂，另两臂随旧训练及审计完成释放资源后自动并行，不抢占旧批次。恢复先读[新增协议](../development-artifacts/supervised-alignment-development-v1/protocol.md)、本批`controller-process.json`、`controller-state.json`、`science-schedule.json`及实际LM Docker状态，不重复提交。新损失为原监督位置的`1-cos`，不是旧全位置自预测平方距离；KV仅直接读取同位置缓存，当前轮仍读因果前缀。可选第七臂、长训、新图及旧网格均未加入。正式终点、独立审计与云端核验尚待完成，当前没有新科学结论。

2026-10-12用户授权选择并开展下一项实验，随后明确要求利用八张卡并行做**首轮KV×自解码表示对齐**二乘二，并确认另保留自预测CE对照。`self-decode-alignment-development-v1`六臂已提交GPU0–5各300k步（总180万新增更新），同Wang φ7.2/wd.1图、原第0步权重与共同数据流；主四格另加有/无KV两条CE。固定镜像58项契约、六条CUDA短训与新容器七池逐token重载通过；固定权重诊断已完成，尚无正式训练结论。首次容器启动因cgroup只读未进入模型执行，失败保留；LM在无运行容器时恢复独立挂载namespace，其他项目未动。恢复先读[本批协议](../development-artifacts/self-decode-alignment-development-v1/protocol.md)、`results/self-decode-alignment-development-v1/controller-process.json`与`controller-state.json`、`runtime-repair/amendment.json`和实际LM Docker状态，不重复提交。该授权只重新打开KV与新增无真值损失的交互问题，不恢复旧窗口/R/detach/mHC网格，不自动追加长期训练或新图确认。

