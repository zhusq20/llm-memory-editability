# 普通Loop的低维Query控制通道：300k开发实验

## Material Passport

- ID：`loop-query-control-development-v1`；类型：代码实验；阶段：实现与启动。
- 用户请求：2026-10-16，按附件的最小架构修改进行300000步预训练；随后明确讨论本方法不应混入DiscoLoop，以及是否增加更难任务。
- 材料：[附件原文快照](source-plan.md) §5.2；原始数据、第0步权重、普通Loop终点与独立审计；本批冻结源码与配置。
- 本文提供执行契约，不声明架构有效或控制分工成立；未记录任何用户已读或已审核声明。

## 问题、历史去重与授权边界

主要问题：在同一知识图、原始起点、最终答案监督及300k更新预算下，低维循环状态只调制attention Query，能否改善普通Loop的分布外组合？本批属于受控行为开发实验，不是控制机制的因果证明。

对照roadmap的历史去重表：`self-decode-alignment-development-v1`、`supervised-alignment-development-v1`检验表示对齐、KV和DiscoLoop；它们没有独立的token级循环控制状态。本次不延续已失败的KV配方扩展。`ttcd-architecture-comparison-v1`已完成的普通Loop **foundation** 300k训练与独立七池审计可复用；其后common/stream训练不属于本次基线。旧批次均不重启。

只新增一条`query_control`，300000次科学更新。`d_c=32`固定，不扫描16/64。没有DiscoLoop、首轮KV、辅助CE/对齐、桥实体监督、HALT/REFINE/ADVANCE或额外工作区。用户关于更难任务的提问作为讨论；本批不自动追加新任务训练。附件要求先诊断的章节是研究建议；本次按用户直接要求执行最小架构试验，不能将该试验冒称已经完成前置因果诊断。

## 完整架构

保留原GPT-2 pre-LN、768维内容、4个独立初始化的物理块、2次共享执行、12头、MLP、位置embedding、tied readout及全部0.1 dropout。附件中的RMSNorm(Q/K/V输入)在本项目适配为原有LayerNorm，避免同时改变内容归一化。

令X为本次前向原token+position embedding经过原embedding dropout的输出。N为逐token、无学习参数的RMS归一化（epsilon=1e-6，仅用于controller输入）。新增四个无bias矩阵：

- P_X、P_D：768→32。
- W_C：32→32。
- U_Q：32→768。

定义：

```
I = P_X N(X)
C0 = tanh(I)
Q(layer,r) = original_Q(LayerNorm(D(layer,r))) + U_Q C(r)
K(layer,r), V(layer,r) = original_K,V(LayerNorm(D(layer,r)))
C(r+1) = tanh(W_C C(r) + P_D N(D_round_end(r)) + I)
```

采用代码中的列向量线性层约定。C在一整轮的4个物理块内固定，只在非最终轮的边界更新。四个新增矩阵既跨轮共享，也跨物理层共享；每轮仅计算一次Query增量，四层复用。当前层K/V只来自原内容路径；控制可以通过attention权重间接影响内容，因此不称严格数学解耦。没有跨位置池化，D位置i只依赖j≤i，X_i与C_i均无未来信息；每次forward重新初始化全部控制状态。

新增矩阵使用独立CPU种子20261016，P_X/P_D/W_C按原initializer_range=.02正态初始化，U_Q全零；不消耗backbone初始化后的RNG。初始Query增量为0，原backbone逐值复用；第一个真实更新U_Q可获得非零梯度，其后其它控制矩阵参与学习。不会向模型传入操作标签、正确中间实体或oracle轨迹。

原69,431,040参数；新增74,752（约0.1077%）；总69,505,792。每序列额外保存T×32控制状态，相当于768维内容宽度的4.17%，不意味着训练总激活仅增加4.17%。新增参数、状态与计算均报告，不宣称等参数或等FLOPs。

## 训练与数据

完全复用已恢复的Wang图：2000实体、200关系、每实体20事实；40000原子事实，以及φ7.2对应273600训练组合。OOD事实以原子形式训练，但不参与训练组合。七池与原数据顺序、身份、评分均保留。

第0步权重来自`ttcd-architecture-comparison-v1/input-mount/initial/loop.pt`，模型backbone摘要`ba0af28431d36db482f2676c1e111a725cf51041c3dca9d006dd10d7a7c18518`。本批复制输入至自己的`input-mount`并冻结SHA256；不从历史300k或TTCD终点续训。

- 初始化791101、数据流792101、dropout seed20261012。
- batch512；AdamW原分组、betas(.9,.999)、eps1e-8、weight decay.1；clip1。
- 学习率前2000步warmup至1e-4，此后常数；300000步。
- bf16 autocast、float32参数/Adam、TF32开启；原0.1 dropout不关闭。
- 原子与组合合并shuffle，epoch尾不足batch时原mask padding；只监督最后一轮原答案与EOS位置。
- 节点0/1k/8k/16k/32k/64k/128k/256k/300k；保存每节点完整预测与恢复状态，另保留第0/300k模型权重。无按成绩早停、择优检查点或自动延长。

这是沿用项目合成知识任务的从零基础训练；不是自然语料全token语言模型预训练。

## 对照与指标

主对照：普通Loop foundation，`ttcd-architecture-comparison-v1/loop/foundation`已独立审计的300k终点；没有DiscoLoop，没有TTCD适配。该基线OI/IO/OO=34.30/43.73/3.2934%。历史`self-decode`基线OI33.50%属于另一次轨迹，单列历史参照，不能替代本次冻结复用对照。

主指标：固定300k的`test_oi`答案+模型自行生成EOS准确率。完整保留atomic_id/atomic_ood/train_ii/test_ii/test_io/test_oi/test_oo、答案准确率、NLL及学习曲线。记录真实原子/组合曝光、有效与执行token、监督token、参数、矩阵FLOPs、训练/评价/总墙钟和GPU资源。单图、单初始化不证明通用收益；终点高分也不证明控制功能分工。

新增矩阵训练FLOPs近似为：

```
6 B T [2 R d dc + (R-1) dc²]
```

计入I一次、U_Q每轮一次、P_D/W_C每个非最终轮一次；不按物理层重复计算共享的Query增量。与原模型矩阵估算分别记录，排除norm、tanh、softmax、加法及访存。不将同更新预算写成同计算预算。

用户后续讨论的任务建议：保留两跳作为锚点；增加混合两跳/三跳与严格未见组合，首先核对[DiscoLoop v2附录A.2](https://arxiv.org/html/2607.00341v2#A1.SS2)的训练条件。不同长度需要统一循环预算或分别报告计算，普通Loop、DiscoLoop、本方法分别比较，本方法保持不含DiscoLoop。尚未授权新增这项训练；不直接把原两跳模型零样本三跳低分当控制失败。

## 执行、检查与恢复

固定镜像`sha256:9643a7d6aeda1b4c9ac2ac50c2d84474b082d649a87e423bc6e0bdc0994097ee`（当前恢复环境，与历史2026-10-12不同，不宣称逐位训练轨迹重现）。显式`docker --context lm-memory`；优先GPU4独占空闲准入，CPU80–83，96GiB，uid1015:gid1016，独立临时目录和持久输出。每次启动前核对实际GPU进程、显存/利用率、容器GPU/CPU、可用内存与存储；不抢占其它任务。资源等候上限24小时，各阶段硬超时12小时。

先运行固定镜像CPU契约与全尺寸CUDA前向/反向、零初值backbone全部梯度逐值一致性、CUDA graph capture后权重/Adam/RNG恢复检查。没有短训练科学阶段；capture内部warmup被还原，不计存活训练更新。然后一次正式300k、新容器七池全部逐token重载、云端终点与完整结果JSON校验。控制器每30秒记录容器存活，训练每1000步写进度，W&B同步记录曲线与损失；凭据只只读挂载原私有文件，不进入源码/参数/日志。

持久根目录：`/ossfs/workspace/llm-memory-editability/loop-query-control-development-v1`。先读`controller-process.json`、`controller-state.json`、`freeze.json`与实际LM容器；冻结来源/输入不变。各stage、失败、退出码、完整日志保留；无自动重复提交/重试。完成状态分开报告科学训练、独立审计与云端校验；任一失败保留并停后续。

## 方法来源核对

本批采用已验证原Wang训练路径与普通Loop基线；新增controller来自用户草案，属于扩展，不称论文原实现复现。实施前阅读了[Hyperloop §3/4.1](https://arxiv.org/html/2604.21254v1#S3)与[HRM §2](https://arxiv.org/html/2506.21734v3#S2)：前者在loop边界更新多residual streams并加入轮次参数；后者采用双模块不同时间尺度并改变梯度/监督方式。本批只有低维token级状态加Query，完整BPTT和原监督不变；未复现二者，也不以“有控制通道”宣称新颖或有效。更难任务依据DiscoLoop附录A.2，其图、训练比例、三轮条件与本Wang图不同，不混写成绩。
