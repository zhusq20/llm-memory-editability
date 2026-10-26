# 固定5轮训练的整轮更新门：执行协议

2026-10-17。用户提供[完整方案](user-proposal.md)，明确授权编写代码并在空闲 GPU 启动；没有空卡则接在当前 GPU 任务之后。此次执行使用 `llm-memory-research` 原则及 `academic-research-suite` 实验执行流程。科学问题是额外循环能否保持正确答案、能否算对更长组合；两者分开报告，不预设稳定即推理增强。

## 方法与历史去重

[配置](../../../configs/gated-loop-depth-extrapolation-v1.json)、[模型语义审计](../../loop_semantics_audit.md)。六臂 N/Q × Full/Fixed/Gate，其中 Q 为每轮都启用的深度一致反馈，Gate 是整个候选与上一轮状态的标量插值。主比较为每个反馈条件内 Gate−Fixed。共享2层、宽256、4头、MLP1024，绝对位置编码，dropout0；257门参数，完整 Transformer 不改。Fixed 与 Gate 初始前向精确匹配。反馈不 detach，末轮不特殊处理。

旧 `loop-hop-unified-v1` 比较物理层数与轮数，最大训练4跳；`dual-gate-multihop-v1` 改再编码注入与 carry，沿用旧末轮关闭反馈；`norm-handoff-multihop-v1` 研究范数交接。本批新增的是固定5轮训练后扩展至10/20/32轮的整轮保留、匹配初值的固定阻尼和可学习阻尼比较，且包含未见长链。改变了词表/长度/支持/宽度和终端策略，因此重建本批六个配对基线；旧权重及分数不混入本批直接比较。不是重开首轮 KV、mHC 或旧注入门网格。

## 数据与监督

复用项目 `loop_hop_data.traverse` 及双图置换设计。A/B 各256实体，4个随机置换关系，实体集合完全不交叉；共享关系 token，事实映射独立。词表为520个符号，保留项目实体偏移3、关系偏移515；加入 NOOP519，query marker1，pad0，保留但不监督EOS2。固定 `head + 20关系槽 + query` 共22 token，目标只有最终实体；输入没有答案或中间节点。所有中间实体另存供诊断。

所有2048原子事实监督覆盖。每批均匀抽取 k=1..5；k1从A/B全部原子均匀取，k2..5只取A。A训练模板分别8/32/64/128条，每模板遍历全部256起点；验证和测试各4模板/长度，k>=2时训练/验证/测试的语义query完全不交叉。k1全部事实已经训练，验证/测试的k1明确标作已见事实提取，不能声称未见query。训练NOOP位置每次曝光随机生成，配对臂使用相同随机流；评价位置固定，准确保存序列、种子、关系表与哈希。

A/B长链为k=6/8/10/12，16/20为预定压力检查，每长度4模板×256起点；完全没有长链监督。每个关系序列是置换，保证答案频率精确均衡。NOOP控制使用相同A测试3跳题，分别8/12/20槽、左/右/随机放置，总9组；短槽右侧pad且在实际query位置读出，主实验始终20槽。控制有意改变有效prompt长度与NOOP数，保留固定实体和三次事实操作。

数据生成与hash/划分/真值审计在CPU完成。所有方法共用同一冻结图对；三个训练种子不是三个独立图。

## 预算与训练门槛

AdamW lr=5e-4、wd=.1，bias/LayerNorm/门参数无decay；betas(.9,.999)、eps1e-8、clip1、BF16。前1000步线性warmup（第一步为峰值/1000），随后常数。batch128；固定R=5，仅最终query位置CE。每1000步验证，不读取长链测试选择模型。

1. 六臂各32步 smoke，seed101，共192更新。验证真实GPU下前向/主干梯度匹配、门梯度、5/32轮前缀一致、数值有限与新容器重载；W&B上线后才更新。
2. 六臂各20,000步pilot，seed101，共120,000更新，不纳入三个正式seed统计。所有臂同样步数、主干初始化和完整样本/NOOP流。
3. N-Full与Q-Full在仅A-short/R5选中的checkpoint上，A/B原子各>=95%，每个k=3/4/5的固定训练探针>=90%、未见A验证>=80%；六臂重载通过、验证loss下降、32轮有限，Gate初始未饱和且梯度非零。训练探针每个长度固定前4条训练模板×256起点，不是全训练池。
4. 只有以上预定pilot门槛全部通过，才自动执行六配置×seeds11/22/33，各100,000步，共18条、1,800,000新增更新。总最大1,920,192更新。不同seed改变主干初始化与数据流，图保持同一对。失败保留证据并停在pilot，不自动扩大预算、改变任务或追加网格。

工程与pilot预算是本次把方案映射到现有实现时冻结的起点，不宣称已验证可学。若未达到门槛，只说明当前设置尚不足以开展正式门控外推比较，不能当作门控机制被否定。GPU任务失败也保留，不自动重试。

## checkpoint、评估与诊断

选择规则：仅A的k1..5验证准确率宏平均最大；同分按较低CE，再同分取最早。同时保留固定预算末步，二者分别报告。每10k覆写恢复用last.pt，保留训练曲线、优化器、CPU/CUDA与数据RNG、累计曝光；best.pt按上述固定规则更新。正式结束独立重载best/last，逐query对齐预测和loss/margin，核对权重与文件哈希；再在独立评价容器跑完整矩阵。

best与last均测R=1,2,3,4,5,6,8,10,12,16,20,24,32；每条轨迹附独立R5前向逐值核对。保存全部逐题预测、CE、margin、entropy，六臂同序。报告短链各R、RobustScore（R6/8/10/12/16/20均值）、correct→wrong与wrong→correct、Retain/Recover；A-long逐hop的R5/10/20与20−5增益；B单列。最高达到80%的hop只报观测离散值，不插值、不挑测试最佳R。

所有六臂在R5共同答对的短题子集、六臂共同掌握全部沿途原子的子集另报覆盖。后者用同一R5原子查询诊断定义，受查询格式限制，不是知识存在性的充分证明。报告每seed和均值±样本SD、同seed Gate−Fixed差、固定图上关系模板为cluster的bootstrap区间（每长链长度仅4模板，区间精度有限，不能替代训练seed方差）。

固定诊断子集保存每轮alpha分位数/位置热图数据、实际更新比、候选变化范数、状态RMS/max/cosine、答案margin/entropy与正确标记。可学习门另做验证集无标签校准的global/每轮/每位置均值替换与batch内循环交换；校准及干预均固定每hop32题，属于辅助诊断，不进入checkpoint选择，交换可能制造分布外状态。中间轮读取保持轨迹不变。JVP、固定步长网格、全局门重训、随机R训练、10轮训练与额外seed不自动执行。

独立报告参数、input/supervised/loop tokens、矩阵前向及3倍训练FLOPs估计、反馈softmax元素数、真实显存、训练墙钟（包含验证）、R5/10/20/32独立推理计时。矩阵FLOPs不包括norm/softmax/optimizer，门不减少完整block计算，不宣称early exit或等计算。

## 实际环境与队列

当前主机是sn4622128200，9张GPU；不是历史AGENTS记录的Alibaba主机。`/ossfs/workspace` 在当前主机不存在。本批持久根显式映射到 `/mnt/disk2_from_server2/siqizhu4/lm-memory-persistent/workspace/llm-memory-editability/results/gated-loop-depth-extrapolation-v1`，该路径与当前LM持久存储一致；不伪称已写入不存在的OSS挂载。

只用 `docker --context lm-memory`，固定派生镜像 `sha256:bc325b1d554d9bb46cf729ee6532ce40ebc764438da8cac2f8e8f796d70f6f2a`。在现有镜像上仅添加锁定wheel/hash的绘图依赖，保留torch/numpy/transformers等科学依赖；构建证据见本目录container-build.json。W&B使用本机已验证的host网络（bridge网络DNS不可用）。每任务独立容器、GPU、4 CPU、32GiB内存和tmpfs；最多两个并行，每卡一任务。优先GPU1/2，只有它们原有lr-control队列包括训练、独立audit、cloud-audit完整退出成功且lane finished后才释放；同时查CUDA进程、显存与其他Docker设备预约。其它实际空闲GPU可被使用，不抢占运行进程。此前norm-handoff排队进程已因GPU4忙碌而退出并处于needs_attention；失败记录不改，不自动重启。新队列不会无限等待这个失效进程，但若其控制器实际恢复，将保护它对GPU4/1/2的预约；PID需核对真实cmdline，避免PID重用。只读证据见prior-inactive-queue.json。

控制器本批锁、防重复stage锁、冻结源码/配置hash、持久state/process/science-schedule；重启仅接回已创建容器，不重训已完成或失败任务。W&B团队zhusq20/project llm-memory-editability，每个smoke/pilot/train独立run，凭据仅本机私有文件和容器内私有凭据文件（读取后删除），源码日志不保存secret。正式完成后核验云端finished、训练步数及审计文件hash。项目：https://wandb.ai/zhusq20/llm-memory-editability 。

CPU合同通过仅代表实现检查；CUDA smoke、pilot、正式训练、独立重载和云端核验分开记录。启动状态以持久controller-state及真实Docker为准。无需重启任何旧训练或修改冻结旧源码。

## 运行入口

`scripts/launch_gated_depth.py freeze --base <持久根>` 冻结源码与配置，随后从冻结source目录执行 `run --base <持久根> --wandb-token-file <本机私有凭据文件>`。控制器依次运行CPU contracts/data、GPU smoke/audit、pilot/audit、CPU pilot-gate，合格后18条正式链及最终分析。`verify`校验冻结hash；`plan`只查看阶段。凭据路径不写入可公开复现命令。所有当前结论仍是待验证假说。
