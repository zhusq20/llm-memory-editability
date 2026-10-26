# Loop与DiscoLoop学习率对照 v1

用户最终指定普通Loop、DiscoLoop各补`2e-4、5e-4、1e-3`，每条300k步，复用已有`1e-4`结果，控制累计权重衰减并在实际空闲GPU上并行。这覆盖先前仅降低学习率的提案；低学习率四条不执行。新增科学更新上限为1,800,000。

## 问题与证据层级

问题是：在同一初始化、数据流和300k预算下，普通Loop提高学习率后能否缩小与DiscoLoop的组合差距？也同时检验DiscoLoop的优势对学习率是否稳定。目标是单个已观察图/初始化上的受控行为规律；不预设学习率偏大或偏小，不把曲线差异直接当作唯一机制证明。

来源为[DiscoLoop v2](https://arxiv.org/html/2607.00341v2) §3.1–3.2、§4、§5.1及项目已有原规模GPT适配。原文符号比较保持训练超参数相同，并以固定权重的中间表示替换支持表示解释；没有给出符号实验的完整学习率/权重衰减扫描。这里复用已经验证的适配与原始Wang图，改变学习率并配平AdamW乘法衰减；不是原论文完整复现。

历史去重：`self-decode-alignment-development-v1`普通Loop与`supervised-alignment-development-v1`DiscoLoop已完成；原权重跨机器恢复后，`ttcd-architecture-comparison-v1`的原生Loop基础和`discoloop-success-restore-v1`原生DiscoLoop基础也已完成。后两条使用当前固定镜像、同原初始化和原训练核心，终点、完整七池重载及云端审计均通过，本次直接复用，不重训。`ce-cross-sample-ablation-v1`降低学习率研究的是每事实20步局部持续写入及遗忘，与本批从原第0步训练完整模型的基础组合学习不同，不能互作对照。

## 冻结矩阵

| 架构 | 峰值学习率 | AdamW衰减系数 | 状态 |
| --- | ---: | ---: | --- |
| 普通Loop | 1e-4 | 0.1 | 复用当前镜像下已审计的300k基础 |
| 普通Loop | 2e-4 | 0.05 | 新增300k |
| 普通Loop | 5e-4 | 0.02 | 新增300k |
| 普通Loop | 1e-3 | 0.01 | 新增300k |
| DiscoLoop | 1e-4 | 0.1 | 复用当前镜像下已审计的300k基础 |
| DiscoLoop | 2e-4 | 0.05 | 新增300k |
| DiscoLoop | 5e-4 | 0.02 | 新增300k |
| DiscoLoop | 1e-3 | 0.01 | 新增300k |

## 保持不变的计算与训练

- 宽768、12头、四物理GPT块循环两轮；69,431,040个参数，完整残差、LayerNorm、因果attention、MLP与0.1 dropout不变。
- Wang原图2000实体、200关系、出度20、φ7.2，273600组合训练题、52463词表；原始世界、七池字节及评分不变。
- 原第0步检查点文件SHA256为`81e95a16656904a511f901b30966a4b4c4f1de708b67b5712d3e8cd2933fe347`；模型张量摘要为`ba0af28431d36db482f2676c1e111a725cf51041c3dca9d006dd10d7a7c18518`。初始化791101、样本流792101、加载后dropout种子20261012。
- 原merged partial-epoch样本流、batch512、2000步warmup后常数学习率；第一个优化更新学习率为0。AdamW betas=(0.9,0.999)、eps=1e-8、fused/capturable、梯度裁剪1.0、FP32参数和优化器状态、BF16 AMP、TF32不变。
- 只监督原答案与EOS，不加中间标签、辅助损失或KV；DiscoLoop仅在非末轮边界加原软解码再编码通道，alpha=1、tau=1、RMS eps=1e-6，全梯度保留。
- 复用原`run_supervised_alignment.py`，不编辑历史冻结源码。新包装只在加载原初始化时以原lr/decay核查来源；训练、保存、独立重载均使用各自新的lr/decay并严格绑定配置。

世界SHA256为`e31ec9959ce0a7b25465191a1d9ef8e83ec42163ed967fa56a60605f6ac1c3b4`，七池SHA256为`8655858c0de4646cfb5c7ea1779b5374ef28b492e6ac59bb6eda9e89c7936bf9`。完整来源、基线终点与检查点哈希写入`baseline-reuse.json`，所有输入及源码冻结到`source-manifest.json`。

## 累计权重衰减控制

AdamW被衰减参数的一步乘法因子为`1-eta_t*lambda`。新峰值η的衰减系数取`lambda=(1e-4*0.1)/η`。三组使用相同warmup形状，因此在全部300k步中，解析`eta_t*lambda`与原配方逐步一致；bias与LayerNorm组继续为零衰减。三组解析累计乘子均约`0.050286937199844596`，逐步乘积误差上限1e-18。

CUDA preflight另核查实际FP32学习率量化后乘积与乘法因子的误差；不把解析相同写成整个优化轨迹逐位相同。此控制保持参数的直接乘法衰减，不保证相对于数据梯度的正则化强度相同，也不保证参数范数或最终权重相同。

## 预定读出与解释

固定节点为0、1k、8k、16k、32k、64k、128k、256k、300k。主读数是固定300k的test_oi答案加生成EOS准确率，同时完整保留原子ID/OOD、训练II、测试II/IO/OI/OO、NLL、逐题预测、样本曝光、token、参数、估算矩阵FLOPs与耗时。不按最高分节点/学习率汇报预定主结果；失败臂不丢弃、不把失败编码为零准确率。

每1000步在CUDA图外对最后一次真实更新记录全参数更新L2/RMS及相对参数范数，包含AdamW衰减；短训每8步记录。观察不消耗RNG、不改参数和优化器；其额外时间计入实测耗时。该读数描述实际更新尺度，不单独证明因果机制。

若普通Loop在更大学习率下缩小差距，支持原固定配方低估了其预算内表现；若DiscoLoop在所测范围仍占优，收窄“仅因1e-4偏向DiscoLoop”的解释。两者都不确定全局最优学习率或学习能力上限。只有一个已观察知识图、一个初始化，查询与检查点不是独立世界。相同300k预算/曝光不等于计算匹配，DiscoLoop约2.05倍矩阵训练FLOPs另报。当前镜像重建基线普通Loop OI/OO=34.30/3.2934%，DiscoLoop=99.20/95.6587%；早期环境33.50/3.44与99.30/95.66只保留历史参照。

## 验证、资源与执行

固定LM镜像`sha256:9643a7d6aeda1b4c9ac2ac50c2d84474b082d649a87e423bc6e0bdc0994097ee`，只使用`docker --context lm-memory`。先在该镜像执行矩阵/累计衰减/原训练CPU契约。原runner中的两项基线一致性测试会调用CUDA，保留原文并移至每臂的cuda-contracts阶段，不因CPU容器无驱动而修改历史源码。每臂按cuda-contracts→preflight→原尺寸32步短训→独立新容器七池重载→300k正式训练→独立新容器七池重载→W&B canonical完整JSON云端核验顺序执行。正式审计另逐值核对全部样本曝光计数与组合选择索引，必须与复用1e-4对照一致。每臂短训通过即允许其正式运行，不等待其它臂的短训，但不跳过任何门槛。

科学更新上限1,800,000；工程短训6×32=192更新另计。12次短训/正式图捕获各4次零学习率优化器调用、权重/矩/计数/RNG恢复，共48次捕获工程调用另记，不算新增学习步。preflight/梯度诊断没有优化器更新。

每阶段重新查询全部实际GPU、CUDA PID和LM/default两daemon显式GPU预留。GPU须无CUDA进程、已用显存≤1024MiB、利用率≤5%、至少80,000MiB空闲；不同GPU使用UUID与独立资源锁，最多六条并行，每臂四CPU、96GiB内存。全GPU可见容器的访问权限不当作所有卡的独占预留，其实际CUDA PID/显存仍阻止占用。已有任务、停止容器、日志和其它项目均保留；不抢占、不共享本批GPU，若无空闲则控制器等待，最长72小时。

每臂独立容器/临时目录/输出，W&B在线项目为[zhusq20/llm-memory-editability](https://wandb.ai/zhusq20/llm-memory-editability)，正式训练开始时即启动追踪，记录配置、曲线、预算和成本；凭据仅只读挂载原私有文件，内容不进入命令或日志。每阶段硬超时12小时，仅可停止本批具名容器；任何失败保留并置needs_attention，不自动重试、改学习率/裁剪或延长预算，其它有效臂继续。

恢复先读`controller-process.json`、`controller-state.json`、`science-schedule.json`、`source-manifest.json`及`execution/<job>/<stage>/{claim,exit,heartbeat}.json`，核对实际LM容器与锁。不得重复提交入口。输出根为`/ossfs/workspace/llm-memory-editability/loop-discoloop-lr-control-v1`。

执行入口为`python scripts/launch_lr_control.py --execute`；不带`--execute`只读输出配置。代码和冻结配置在仓库维护，原始权重、预测、日志和审计材料保留于本批持久目录。
