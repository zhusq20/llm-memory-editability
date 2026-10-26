# 普通Transformer、Loop与DiscoLoop的跨样本TTCD对照

2026-10-15。用户要求补齐三种架构，优先复用checkpoint，在GPU8/9并行，重点比较新事实组合、早先写入事实和旧知识保持；核查历史归档无可用标准/Loop 300k权重后，用户明确要求按同一原配方补建两个起点，再接相同TTCD训练。用户随后明确取消所有短训、要求直接正式训练：本批不执行原拟native32或适配16步短训，仅保留零模型更新的身份/资源核对与后续正式评测审计。[配置](../../../configs/ttcd-architecture-comparison-v1.json)。

## Material Passport

- 模式：experiment-agent/run；本轮是已获授权的实现、核验与提交，不是科学结果。
- 输入：原Wang图、原GPT2配置与第0步权重；现有DiscoLoop成功恢复、TTCD stream与双新事实评测冻结材料。
- 完成条件：两个新增架构的基础训练、独立重载、共同读取与连续写入训练、原连续面板及双新面板、独立状态重放和云端完整JSON核验；第三架构复用已有对应结果。

## 问题与主张

接入相同TTCD写入流程，循环架构是否使测试时写入的知识更容易被回答、组合，并在后续写入后继续保留，同时保持未写旧知识？受控端到端比较即可建立当前条件下的架构效果；本批不以解释DiscoLoop内部机制作为完成前提，不增加路径干预。

历史去重：`ttcd-second-hop-scratch-v1`已有Loop/Disco两臂，但30k基础组合能力不足，且仅每事实独立A；`ttcd-cross-sample-v1`及`ttcd-successful-base-v1`比较的是固定DiscoLoop的single/stream训练。新增变量是合格审计的原规模300k普通D8及Loop4×2，采用与现有Disco stream相同的连续写入训练与正式面板。旧失败、低基础结果及已有single/stream继续保留。

## 三臂与复用

| 架构 | 独立物理层×轮数 | 起点及新增训练 | 执行 |
| --- | --- | --- | --- |
| 普通Transformer | 8×1，各层独立随机初始化 | 原配方补建300k＋共同10k＋stream10k | GPU8 |
| Loop | 4×2，四层跨轮共享 | 原配方补建300k＋共同10k＋stream10k | GPU9 |
| DiscoLoop | 4×2，保留soft解码再编码通道 | 复用恢复300k、共同10k及正在运行的stream10k | 原GPU4批次继续；不重复训练 |

新增科学更新为640000，两条路径各320000；工程短训另计。10k适配预算是与当前Disco对齐的固定开发预算，不宣称已收敛或充分。只比较共同的stream训练方式；已有Disco single/stream作为训练方式辅助证据，不增加六臂网格。

检查点盘点见`results/ttcd-architecture-comparison-v1/checkpoint-inventory.json`。当前宿主及持久目录没有标准/Loop终点；固定HF归档revision `6c580455ce1667274ea592c1c88f5d0fb1e5a190`明确不含模型/优化器状态。原grokking节点本身不含300k，300k学习记录不能冒充该节点权重，亦不以256k/512k/1.5M替代。Loop直接复用已恢复的原0步文件；标准模型按原seed构造八层独立参数并严格核对历史0步digest。

## 原配方基础

固定图SHA `e31ec9959ce0a7b25465191a1d9ef8e83ec42163ed967fa56a60605f6ac1c3b4`、七池SHA `8655858c0de4646cfb5c7ea1779b5374ef28b492e6ac59bb6eda9e89c7936bf9`；2000实体、200关系、度20、40000原子与273600训练II、词表52463。宽768、12heads、位置1024，所有GPT残差、LN、因果attention、GELU和MLP保留。独立层数量与循环轮数是架构变量。

原init791101、stream792101、dropoutseed20261012；dropout.1、native合并流batch512及partial-epoch尾部、原答案＋EOS CE、warmup2000后lr1e-4恒定、wd.1且bias/LN不衰减、原fused/capturable AdamW、BF16、TF32开。复用`run_supervised_alignment.py`原生执行与独立七池审计，不添加辅助loss、KV或通道。固定300k终点，不按效果延长或择点。

标准0步model digest为`11b1c7254c2c277676239094ab835bb3dca302a9d7197cec4e98380308d7fa3f`、97782528参数；Loop为`ba0af28431d36db482f2676c1e111a725cf51041c3dca9d006dd10d7a7c18518`、69431040参数。逐层digest检查独立物理层不是相同随机初值的复制。

## 相同TTCD适配

复用[成功基础TTCD配方](../ttcd-successful-base-v1/protocol.md)的原数据字节、训练更正排除键、随机采样、native512/correction128、micro64/32、lr1e-5、wd.1、clip1、dropout0/eval确定性、BF16及TF32关闭。共同10k用native CE＋更正记录读取CE；stream10k再加写后原子CE和未写原子保持CE，四个mean损失等权。同A连写8事实、每事实4次局部写入，在终状态读取8事实，梯度穿过前面的写入，不detach。外层更新完整骨干；测试时只更新A。没有更新后两跳训练标签或组合成绩择点。

局部规则沿[TTCD §3.1式(6)](https://arxiv.org/html/2608.01672v1#S3.SS1)的现有GPT/GELU适配，教师停止梯度、长/短窗口32/16、gap17、局部η.001；未实现原作者实践v9的完整off-policy scan、conv/projection、dynamicβ等。Disco沿[原论文§4](https://arxiv.org/html/2607.00341v2#S4)的soft词表加权embedding、RMS及残差通道；固定α=τ=1，无中间真值实体输入。

普通D8有8个独立MLP down增量；Loop/Disco有4个增量，且两轮读同一份。局部特征每物理层只捕获首轮，整包读取结束后统一提交。D8快状态18874368个FP32元素/72MiB，Loop/Disco9437184/36MiB。三者有效执行深度8，相同步数和曝光不等于等参数、等可写容量或等FLOPs；原生骨干及快状态数量属于本批架构配方的实际差异，均明报，不声称隔离纯参数共享效应或普遍效率优势。

## 前提与工程门槛

基础与适配所有固定检查点都独立重载，校验输入/配置/源码、model hash、完整七池逐token及分数；真实CUDA前向/反向/同步、A0包装与原生同模式一致、保存/重载/转移重放必须通过。按用户最新要求不进行任何短训，零模型更新的初始身份/资源核对后直接启动正式300k；固定镜像CPU身份/配置/调度契约不调用训练器，正式七池及A0检查按流水线执行。

新两臂的原子/组合、共同读取ID/OOD及写入校准准确率全部报告为前提，不用Disco的OO90%门槛淘汰普通模型，也不在低分后追加预算。`passed`在新批身份收据与共同门槛里只表示数值/身份有效；准确率条件另记`quality_passed`/`report_only`。保留低能力架构，解释相应限制。旧Disco批次原有门槛保持原契约，当前已通过共同门槛。

## 正式评测

原连续面板完全复用：32个更正训练排除事实、32个固定旋转顺序、原D与64 U原子/64 U组合、逐前缀和再次更正，共1313快状态/臂。独立问题清空原文、attention/KV和问题隐藏状态，仅传A。

双新事实面板完全复用[原补充评测](../ttcd-double-new-eval-v1/protocol.md)：文件SHA `ade07fa92e528f1b19bbc10965d984f747ed7f7daf66f5eafa254971ac7d08f6`。同32个排除键配16对，II/IO/OI/OO各4，第一条新尾是第二条头、第二条尾是新答案；原查询合法未训、原答案/第一条单写答案/双写答案互异。图结构选样，三架构不另选样；1089快状态/臂。复用A0、仅写其一、两条两种顺序、同32旋转，不因基础是否正确删掉正式题。

主读数：t32双新组合的pair-equal准确率及四类分组；早先写入事实相对其立即写入成功的保持、按年龄完整曲线及分母；未写旧原子与旧组合的写前/写后全量正确率、原正确覆盖和保持。另报两事实均能回答时的组合率及覆盖，pair相对首次完成时的保持，顺序/间隔，以及原D与再次更正。循环顺序只是同一组事实的重复测量，不作为独立图/种子。共同原子与写入前组合正确子集按预先规则另报交集、覆盖和分母，空集保留为空，不再选题补齐。

## 判读与成本

主结论是相同训练流程和预算下的完整架构配方效果；基础组合差异须与测试时适应的变化分开。Loop优于普通模型才支持普通循环架构收益；只有Disco改善时限定为该架构/配方。新事实提升而U下降报告权衡。较高双新组合若完全随原子保持提高，仍是有价值的端到端效果，但不额外宣布独立组合机制。低基础或教师读取失败不能唯一归因于写入接口。

记录native/适配更新、事实与token曝光、局部写入次数、参数/激活参数、快状态元素、估算矩阵计算、实际时间和GPU占用。D8原连续及双新快状态约169GiB，Loop约84GiB，另有基础/适配优化器检查点；新批启动预留至少400GiB。第三臂引用原批已完成科学阶段/审计/云端收据及哈希，不伪装成本为零；只新增成本为零。

## 执行与恢复

固定LM镜像`sha256:9643a7d6aeda1b4c9ac2ac50c2d84474b082d649a87e423bc6e0bdc0994097ee`，显式`docker --context lm-memory`。每阶段独立容器，Transformer固定GPU8/CPU40–43、Loop GPU9/CPU44–47、各96GiB、uid1015:gid1016；8/9申请独占空闲卡，不抢占外来任务。原GPU4两组及补充控制器继续。持久输出`/ossfs/workspace/llm-memory-editability/ttcd-architecture-comparison-v1`；每个执行阶段超时48小时、资源等候24小时，宿主监控进程存活/退出码/进度，每30秒记心跳。失败保留并停止对应后续，不自动重试科学任务或放宽审计。

冻结原科学源码及新增适配器、命令、配置、原初始化、数据/面板与工程收据；只新增批次，不改父冻结源码或权重。正式训练同步W&B，团队`zhusq20`、项目`llm-memory-editability`，各阶段保存并云端下载核验canonical完整JSON、身份、终点步数。恢复先读本批`controller-process.json`、`controller-state.json`、`source-manifest.json`、各臂`controller-state.json`、claim/exit与实际LM Docker；不重复提交。单图、单初始化、有限32事实及episode间重置，不宣称终身学习、长期巩固、新世界确认或首次结合Loop与TTT。
