# 普通Q adapter近等参数对照：300k开发实验

## Material Passport

- ID：`loop-query-adapter-development-v1`；类型：代码实验；阶段：实现、冻结与启动。
- 用户授权：2026-10-16明确要求新增一个参数量接近的普通Q adapter，也训练300000步，数据及训练配方保持一致，比较独立32维控制状态是否超出普通adapter的收益。
- 主对照：正在执行的`loop-query-control-development-v1`；复用原始Loop foundation，不重启任何旧任务。
- 证据：本批源代码、配置、固定输入哈希、测试、GPU准入和后续逐token审计；目前没有本批科学结果。不作人类已读/批准声明。

## 主要比较与历史去重

原模型768维hidden state已经跨轮传递；32维C并非首次赋予Loop跨轮状态。原控制分支在每轮四层间固定、轮末更新，还读取原输入X。用户要求用普通Q adapter检验其是否具有超出小型Q调节分支的实际收益。

历史`self-decode-alignment-development-v1`和`supervised-alignment-development-v1`没有这条普通非线性Q adapter；当前`loop-query-control-development-v1`已有控制分支，无需重复训练。新批只新增 **1×300000步**，与父批合计600000步新训练；本批不追加新图/种子、更难任务、纯线性LoRA、额外控制状态消融或长期预算。

本轮回答：在相同数据/基础起点/监督/优化/更新数、近等新增参数下，普通无独立状态Q adapter是否能达到控制分支的表现。比较的是两个完整方案；控制状态持久性并非唯一变化，不能把差异直接写成独立状态的唯一因果效应。

## 普通adapter架构与参数匹配

保持原GPT-2 LayerNorm、768维内容、4物理块×2次共享执行、12头、MLP、残差、位置embedding、tied输出和0.1 dropout。各物理块仍独立随机初始化，不复制同一层作为标准模型。

每个物理块的attention输入`z=LayerNorm(hidden)`，新增：

```
Q = original_Q(z) + U tanh(A z)
K = original_K(z)
V = original_V(z)
A:768→48，U:48→768，无bias
```

A/U跨物理层和循环轮次共享，与控制分支的共享范围一致。每次执行每个物理块都从该块当前z重新计算；没有独立C，没有跨轮额外缓存，没有原始输入旁路、跨位置池化、oracle/桥标签、DiscoLoop或KV融合。保留原有causal attention。48维只是本次计算的中间激活，不作为下一轮额外状态保存。

采用tanh以匹配控制分支的非线性类型。纯线性UA在全量训练W_Q时可合并入W_Q；本批使用普通非线性瓶颈，避免把对照限制成仅重新参数化原Q的线性分支。它是Q位置的adapter适配，**不是冻结backbone的PEFT/LoRA复现**；backbone和A/U全部按原配方训练。

- adapter新增：2×768×48=73,728；总69,504,768。
- 控制分支新增：3×768×32+32²=74,752；总69,505,792。
- 差1,024，为控制分支新增参数的1.37%，占整模约0.00147%。48为使两矩阵瓶颈接近原分支参数预算的固定选择，不调参。

额外初始化种子20261016，A按std.02正态，U全零；单独保存/恢复CPU RNG，不改变backbone初始化与原dropout seed。两种模型初始输出都与普通Loop逐值一致；首步U获得梯度，随后A学习。对控制分支与adapter不同形状的额外参数，不宣称逐值相同。

## 数据、训练与评价冻结

直接复制父批`input-mount`的完整输入，逐项对父`freeze.json`核验，不重新生成知识图或数据。第0步 checkpoint SHA256为`81e95a16656904a511f901b30966a4b4c4f1de708b67b5712d3e8cd2933fe347`；基础backbone摘要`ba0af28431d36db482f2676c1e111a725cf51041c3dca9d006dd10d7a7c18518`。

复用2000实体/200关系/degree20的Wang图、40000原子及φ7.2的273600组合，全部七池、顺序和原评分。五个共享训练/评分实现文件SHA也必须与父冻结字节相同。冻结父配置/清单摘要作为本批来源，父训练继续运行。

- 4物理层×2轮、宽768；初始化791101、merged stream792101、dropout seed20261012。
- batch512；原分组fused/capturable AdamW、betas(.9,.999)、eps1e-8、wd.1、clip1。
- warmup2000至1e-4，之后常数；300000步；bf16 AMP/FP32参数与Adam/TF32开启。
- 原子/组合合并shuffle、epoch尾原mask；最终轮答案/EOS CE，辅助损失0。
- 评价0/1k/8k/16k/32k/64k/128k/256k/300k，batch256，全部预测/恢复状态保留，模型权重另保留0/300k。
- 无科学早停、择优检查点或自动延长；启动只做0存活更新CUDA验证，无短科学训练。

主指标：固定300k的test_oi答案+模型生成EOS准确率，报告**控制分支−普通adapter**百分点差。共同报告全七池、NLL、曲线、实际原子/组合曝光、有效/执行/监督tokens、参数、矩阵计算及墙钟。结束后逐项核对两端真实曝光相同。单开发图/初始化不证明一般规律。

adapter新增矩阵训练FLOPs约为`12 B T d a L R`：每物理块每轮执行两次可微矩阵乘；本配方每步7,247,757,312。控制分支约1,220,542,464/步；新增部分约5.94倍，不等于整模5.94倍。norm、tanh、softmax、加法/访存不含在矩阵估算内。不称同参数即同计算。

剩余差异：控制分支独立状态、原输入旁路、RMS输入、轮末更新且一轮内固定；adapter当前LN输入、每层重算且无输入旁路。即使控制分支胜出，也不能唯一归因state；两轮中C的更新可用X和H1前馈展开。若adapter达到同等表现，则该预算下未显示独立状态方案超出普通adapter的行为收益。

## 执行、资源、追踪和恢复

固定同父镜像`sha256:9643a7d6aeda1b4c9ac2ac50c2d84474b082d649a87e423bc6e0bdc0994097ee`，显式`docker --context lm-memory`，uid1015:gid1016，各96GiB、独立tmpfs和持久输出。

GPU4只允许与已登记同项目父容器`loop-query-control-development-v1-{train,audit}-capture-v3`并行；adapter CPU84–87，与父80–83隔离。实际GPU进程PID必须属于登记父容器；其它owner/未知进程拒绝。GPU保留至少16GB空闲，登记父总显存不得超过16GB；不抢占或重启任何任务。共享时段、资源和调度修订单独记录，**父/子共享时段墙钟不作为独占吞吐基准**，不修改父冻结数据、配方或已输出科学测量。准入失败等候，不放宽owner白名单；等候24小时、单阶段硬超时12小时。

CPU契约→全尺寸CUDA零初值输出/全部backbone梯度一致、adapter上投影非零梯度、capture后全部权重/Adam/CPU和CUDA RNG恢复→正式300k→新容器全七池逐token重载→W&B完整终点JSON核验。预检查已采用释放eager logits/loss引用的修正，避免父启动时的legacy stream捕获故障。

W&B团队zhusq20/项目llm-memory-editability，正式训练开始前在线初始化、同步记录进度与曲线，host每30秒监控容器和GPU资源。凭据沿用私有本机文件，通过Docker API复制为本容器用户0400，不进入源码、参数或日志。只在本批新目录准备标准0755/0644绑定访问，旧批目录/凭据权限保持。

持久根`/ossfs/workspace/llm-memory-editability/loop-query-adapter-development-v1`。恢复先读controller-process/state、freeze、paired-control、资源准入和实际LM容器；拒绝重复提交。科学失败记录保留并停后续，无自动重试。父子各自独立审计/云端核验完成后才写正式comparison与完成清单。父若需要处理，本批保留自身终点，不以未审计父模型填结论。

## 参考方法核对

已阅读[Houlsby et al. ICML2019 §2/Figure2](https://proceedings.mlr.press/v97/houlsby19a/houlsby19a.pdf)的降维—非线性—升维瓶颈与近identity初始化，以及[LoRA §4.1](https://arxiv.org/html/2106.09685v2#S4.SS1)的同输入低秩加法、A随机/B为零与backbone冻结前提。本批只借鉴瓶颈和零增量起点，接入Q、共享跨层/轮且全量从零训练；不称任一论文完整复现。
