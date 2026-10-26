# 从各自20k检查点分支的写后读取训练对照

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run；设计已由用户明确选定。
- Origin Date: 2026-10-15 UTC
- Verification Status: CPU及两臂CUDA分支工程验证通过；控制器等待各自20k节点，尚未有本批科学结果。
- Version Label: ttcd-second-hop-reader-control-v1

## 授权、问题与历史去重

用户明确要求：“从各自20k的检查点分支，再训练10k步，按你说的方案在GPU8、9启动这两条对照，现有训练继续跑。”本批仅新增两条10k分支，允许各自与指定父训练共享GPU，不停父训练，不自动重试，不增加初始化、知识图、写入配方或扫描网格。

父批为[ttcd-second-hop-scratch-v1](../ttcd-second-hop-scratch-v1/protocol.md)。其前20k训练旧原子、旧II组合和长窗口更正单跳读取；后10k另加快权重下的新原子读取与无关旧原子保持。本批复用对应模型完整20k状态，后10k只保留前三项。主问题是：这两项额外外层训练，对同样测试时写入规则下的E更新、D组合使用及U保持有何整体作用？

已有20k节点不是30k主对照终点；已有TTCD/原子CE/sham是同父模型上的测试时写入参照，未控制父模型是否经过写后读取训练。旧监督对齐、KV与普通原子编辑批次没有这组共同20k起点的比较，本批不重跑这些历史问题。原论文[§2.2–3.3](https://arxiv.org/html/2608.01672v1#S3.SS1)将在线快权重更新纳入端到端预训练；本批是本项目专用损失的消融，不能称更忠实的原论文复现。

## 固定矩阵与公平性

| GPU | 模型 | 复用起点 | 新增外层更新 | 后10k损失 |
| --- | --- | --- | ---: | --- |
| 8 | DiscoLoop | 自己的全状态20k节点 | 10000 | 旧原子＋旧II组合＋上下文更正单跳 |
| 9 | 普通Loop | 自己的全状态20k节点 | 10000 | 旧原子＋旧II组合＋上下文更正单跳 |

两条对照的全局终点均为30000步。与父批对应架构共用同一20k权重、AdamW矩与计数、采样RNG、Torch/CUDA RNG和原30k余弦学习率时间轴；不重新预热，不重置优化器，不重放已完成的20k。每步继续生成相同普通样本与更正包，保持数据流消费顺序；仅不执行用于外层优化的快权重写入及其两项读取损失。各保留损失系数仍为1，相加后使用原梯度裁剪规则。

每条新增128万旧原子、128万旧II组合及128万上下文更正样例，答案/EOS监督token新增768万；连同复用起点，每条路径累计2304万。父批每条路径累计2816万。本批只新增20000更新，共1536万监督token；共同起点的40000更新不重复计作新增成本。父批首次被终端终止的1407次已记录更新仍单独保留在父批成本中。

本比较匹配全局更新数与三项普通训练样本流，不匹配总监督次数、输入token或计算。它估计两项额外损失及其可微快权重过程的整体效果，不能分辨两项各自的作用，也不能证明等计算效率。单一开发图、单一初始化；病例和查询不是独立知识图。

## 恢复与隔离

只接受父批各模型原路径的`training/checkpoint-0020000.pt`。父训练以临时文件写完后原子重命名发布；控制器分别等自己的完整`.pt`，不按成绩选择节点。父训练在发布前失败则对应分支停止等待并保留失败，不能改用其它节点。

固定父配置SHA256为`2225cfff7a781268f86760c9521f8dafb5ee762c04a1961160a3002a2077b29c`，父源码manifest SHA256为`7a13ab65c2c41d296c681746958c7370d90864b5376dcd7ec115140e7f019d10`。首次加载逐项验证父冻结源码、输入、配置、模型架构及权重哈希，确认20k曝光、零写入训练、AdamW步数和学习率、完整RNG与保存的验证面板；发布后记录检查点SHA256，并在新输出的`branch-input/`保存独立原字节副本。

父源码、配置、权重和输出只读。新源码冻结在本批`source/`，用新的`training_identity`绑定分支实现。终点`training/parent.pt`与训练日志均属于本批，不覆盖父批。累计计数和`inherited_cost`、`additional_counts`、`branch_training_compute`、`branch_training_seconds`及`branch_wall_seconds`分开记录；等待20k的控制器时间不混入训练时间。

训练期间保留原旧世界面板与校准原子的无梯度读取；没有更新后的两跳样例、标签或评测。验证中的临时快权重不会进入优化器，不能将“去掉写入训练”解释为完全不执行任何验证写入。

## 资源与执行

配置：[ttcd-second-hop-reader-control-v1.json](../../../configs/ttcd-second-hop-reader-control-v1.json)。GPU8仅允许与`ttcd-second-hop-scratch-v1-discoloop-train-restart1`共享；GPU9仅允许与`ttcd-second-hop-scratch-v1-loop-train`共享。核对Docker内全部主机PID与实际CUDA进程，发现其它进程则拒绝提交。每卡至少保留8192 MiB可用显存，新增任务各4 CPU线程/48 GiB内存；CPU分别8–11、12–15，与原训练0–7及彼此分开。共享调度耗时不作为独占GPU效率基准。

镜像保持父批`sha256:9643a7d6aeda1b4c9ac2ac50c2d84474b082d649a87e423bc6e0bdc0994097ee`。显式`docker --context lm-memory`，UID/GID 1015:1016、各自tmpfs HOME、只读project/结果根/父源码/冻结输入，仅本模型新输出可写。冻结输入另按原文件名复制至本批`input-mount/data/`，字节清单绑定原冻结备份；整体只读覆盖容器`/project/data`，不穿过原私有data子目录逐文件bind，不改变原目录权限。持久根仍为`/ossfs/workspace/llm-memory-editability`，没有真实挂载就拒绝启动。使用本批锁与短共享准入锁，保持父控制器原GPU锁不动。

```bash
/mnt/disk2_from_server2/siqizhu4/llm-memory/.venv/bin/python scripts/launch_ttcd_branch.py pipeline --execute
```

默认不加`--execute`只显示计划。各臂独立等待/提交并记录进程、容器、GPU UUID、CPU与资源准入凭据、日志、退出码；从等待起算12小时硬超时，只停止本臂自己的新容器，不自动重试。W&B online沿用团队zhusq20、项目llm-memory-editability；凭据仅从既有私有文件只读挂载，正式训练开始才创建对应W&B运行。

## 终点与评价

每1000步保存新节点，保存新分支的20k初始状态及30k终点。完成后自动在新容器依次执行训练重载、读取校准、正式评价和编辑重放；校准未通过则保留结果，停止该臂后续评价，不长训或选其它节点。沿用父批病例、分组、门槛、TTCD/atomic_ce/sham写入预算、E/D/U及同旧答案评分、条件结果与覆盖率。主比较使用全病例等权D的新答案＋EOS联合准确率，并同时报告E、旧前提/组合能力和U；条件结果不能替代全量结果。

训练完成、独立重载、读取校准、编辑有效、编辑重放与W&B云端核验分别记录。恢复先读本批`controller-process.json`、`controller-state.json`、每模型`controller-state.json`、父节点凭据和实际LM Docker状态，不能重复提交本批或改动原训练。

## 工程验证与当前状态

固定镜像58项CPU检查通过，原两项CUDA专项按默认规则跳过。验证包括普通更新对照的独立参考实现逐值一致、AdamW状态、下一批样本、Torch RNG、全局学习率、监督与成本计数、原产物保持、错误来源拒绝，以及新进程训练重载。GPU8/9各完成全尺寸16步预热＋16步写入的父fixture，再从16步完整状态分支16步普通训练；初始权重/优化器/CPU与CUDA RNG逐值一致、反向有限、原文件保持及新进程训练审计通过。工程fixture关闭追踪，不作为科学结果。

两臂等待控制器于2026-10-15 04:12 UTC首次提交，分别等待自己的原20k检查点，父训练继续。本批持久目录为`/ossfs/workspace/llm-memory-editability/ttcd-second-hop-reader-control-v1`，保存源码manifest、授权/调度修订、工程日志与退出状态、控制器日志和每模型等待状态。工程XML误放在容器tmpfs，退出后未能导出；完整原始容器测试日志、命令、镜像及退出状态已归档，不伪造XML或重复训练。

首轮正式布局预检在Python/模型执行前退出126：逐文件bind目标需穿过原700权限的data目录。只停止新批尚未开始训练的等待控制器，将新输入副本整体覆盖`/project/data`；原父训练、私有目录和冻结科学文件不改。失败容器/日志、旧控制器及零科学更新记录保留于`controller-layout-intervention-20261015/`，原manifest保持不变；修订后的主机调度代码另存副本和哈希，见`runtime-layout-amendment.json`及`controller-layout-intervention-20261015/layout-v2.json`。工作区控制器经18项检查验证目录权限、覆盖层完整性和损坏拒绝；正式修订布局的零更新CUDA预检见每模型`layout-preflight-v2.json`。恢复时同时读取这些修订，不按旧冻结调度脚本重新提交或忽略布局修正。
