# 从零训练 DiscoLoop × TTCD，并测试第二跳更正后的独立组合

状态：2026-10-15 两臂各30000步训练、终点七池及更正原子独立重载、读取校准、正式编辑、保存快权重的新进程重放与W&B云端核验全部完成。每模型32事实×三方法共96状态，全部重放通过；十个云端阶段的finished状态、身份、结果与完整预测数组核对一致。TTCD的普通Loop E/D为100/83.15%，原正确旧组合保持19.05%；DiscoLoop为65.63/6.09%，且原组合前提较弱。完整解释见[结果](../../results.md#ttcd-second-hop-scratch-v1)、[汇总](../../../results/ttcd-second-hop-scratch-v1/comparison.json)与[完成清单](../../../results/ttcd-second-hop-scratch-v1/completion-manifest.json)。本次后处理零新增训练。

恢复先读持久根`/ossfs/workspace/llm-memory-editability/ttcd-second-hop-scratch-v1`的`completion-manifest.json`、`post-training-state.json`、`schedule-amendment.json`、`post-training-20261015/`与实际LM容器。后处理因GPU9有其他任务，在GPU8分离CPU并行执行两个模型，与既有跨样本单事实组共用；只改宿主调度与输入挂载，原冻结源码、配置、数据、终点权重和评分不变。云端API的扁平字段及长数组表示导致的验证器失败全部保留，最终按远端原始JSON全量核对；没有重复科学阶段。旧train控制器记录保留历史，已完成训练和编辑不重复执行。

启动历史保留：[GPU8 DiscoLoop重启训练](https://wandb.ai/zhusq20/llm-memory-editability/runs/951rb2is)、[GPU9普通Loop训练](https://wandb.ai/zhusq20/llm-memory-editability/runs/bi3vklow)。GPU9最初因占用被拒绝，用户报告空闲后才提交。GPU8首次运行记录到1407步后退出137；主机journal在03:28:06 UTC记录终端`sudo kill -9 1769805`，同秒对应容器退出，Docker OOMKilled=false，限定退出时间段内核无事件。用户授权后完整归档首次产物，用原冻结配方从第0步重启，没有实现或冒称断点恢复；中断的已记录1407次更新及成本另计。旧产物位于`discoloop/attempts/interrupted-20261015T033501Z/`，恢复记录与旧容器均保留。

## 来源与问题

TTCD [§4.1、附录A.1.1](https://arxiv.org/html/2608.01672v1#A1.SS1.SSS1)使用两层、宽度128的小模型训练MAD任务，支持小模型从头学习上下文记忆的可行性。它没有验证旧参数知识与更正冲突后，独立两跳查询能否使用更新。本批借鉴长短窗口和式(6)，加入Loop/DiscoLoop与独立会话快权重读取，属于受控扩展。

既有 `supervised-alignment-development-v1` 证明过旧Wang设定上的DiscoLoop组合能力，但当前主机没有该父权重；HF归档也不含权重。此前 `ttcd-second-hop-development-v1` 是复用父模型的准备方案，保留其配置和工程证据。本批改为从随机初始化学习基础能力及更正接口，不把历史准确率当成本批前提。没有作者v9的可学习卷积、投影、归一化或off-policy scan，不称原TTCD完整复现。

主要问题：先学会旧 `(a,r1)→b`、`(b,r2)→c` 与普通两跳；只通过原子记录把第二跳改成d，独立查询 `(a,r1,r2)` 是否得到d，同时保持无关旧知识。

## 数据、模型与隔离

复用项目 `build_graph`（Wang生成逻辑），新开发图为256实体、32关系、每实体8条原子边，共2048原子、16384两跳链。5%原子标为组合角色OOD。所有原子的旧答案均可学习，包括角色OOD；普通两跳训练只用II支持，φ=2.0，另留10%的候选II作原始生成器测试池。φ之外的组合也用于未训练组合评价。这不是原论文原规模复现，也不是旧已观察图的独立确认。

不用GPT-2自然语言词表或下载tokenizer：PAD=0、EOS=1、实体2–257、关系258–289，总词表290。专用整数token及offset保存在complete.json。模型保持真实GPT的绝对位置embedding、LayerNorm、因果attention、GELU MLP、完整残差及绑权读出。宽128、4头、两个独立随机物理层、MLP中间512、循环两轮、位置表64、dropout全0；外层训练启用梯度而保持确定的eval前向。DiscoLoop使用既有软解码、软重编码与RMS缩放通道。

GPU8为DiscoLoop，GPU9为普通Loop；初始参数张量和采样数据流相同，物理层之间独立随机初始化。通道增加计算，不能把相同步数称等计算预算。单一新开发图、单一初始化不建立通用优势。

训练前固定校准8 ID＋8 OOD、正式评价16 ID＋16 OOD目标事实。二者都不参与更正样例训练；其旧原子答案仍可训练。正式D在φ普通组合训练支持外。写入不含两跳查询、组合标签或桥实体监督。训练更正尾实体每次重新随机采样，排除原旧尾。每个样例的保持原子排除同一个目标键。

数据准备固定world/panels、模型配置、划分和病例库哈希；训练与评测验证这些文件，拒绝漂移。准备命令不启动训练。

## 两阶段外层训练

| 阶段 | 固定预算/模型 | 每步监督 |
| --- | --- | --- |
| 基础与读取预热 | 20000步 | 128旧原子、128旧II组合、128更正记录后的单跳读取 |
| 原子写入学习 | 10000步 | 保留前三项，另加128快权重独立单跳新答案、128快权重下无关旧原子保持 |

每项交叉熵同时监督答案和EOS，各项系数1；分别求均值后相加。AdamW峰值lr=1e-3、最低1e-5、200步线性预热、全程余弦、wd=.1、梯度裁剪1。浮点32、TF32关闭。固定终点，不按结果延长或选择检查点；这些是尚未验证的开发起始值，不保证任务成功。

每模型总30000次外层更新、384万旧原子/旧II/上下文更正样例各一份，其中128万更正样例参与快权重独立读取；总监督目标token为2816万。两模型合计60000次新增外层更新。记录实际执行输入token、监督token、样例曝光、计算估算、训练耗时及含验证的墙钟；验证计算另列，不混入训练预算。矩阵FLOPs只估计稠密matmul，反向2倍前向仅为proxy，不代表实测总算力。

每个更正包为 `[b,r2,d,EOS] + 17 PAD + [b,r2]`。long=32、short=16；直接短窗口看不到完整更正，但多层与共享残差仍可能间接传递信息。读取预热监督长窗口在包末预测d，再预测EOS；PAD不作预测标签。

原子写入阶段每样例有自己的 `[physical layer, intermediate, hidden]` 快矩阵A，初始化0，连续写4次，η=1e-3。包整体读完再提交全部物理层更新，仅第一轮原子查询位置供应写入特征；所有循环轮次读取同一矩阵。批次维不作平均，不把不同知识更新混入同一个会话。

GPT Conv1D布局下，每会话只有一个写入位置，`A' = A + η z_S[:,None] (y_T-y_S)[None,:]`。教师yT停止梯度；学生zS、yS、先前A及后续独立单跳读取保持外层梯度。局部式(6)仍是把输入视为固定时的下投影梯度；外层CE再对整个可微写入计算求导，两种求导不要混淆。长窗口教师通过单跳读取CE学习，不需要外部预训练教师。

外层优化只更新基础模型参数；A没有优化器状态，不写回基础参数，每个训练步结束丢弃。训练期间没有任何“更新后的两跳”样例、标签或评测。常规旧世界两跳面板可报告基础能力；更正校准仅单跳，不用于自动选择配置或终点。

## 保存、独立重载与正式评测

保存第0步、每1000步、阶段交界和终点；检查点含基础权重、优化器、采样与Torch RNG状态、步数、曝光、时间和源码/数据/配置身份。终点为各模型 `training/parent.pt`，不覆盖已有训练目录。保留失败，控制器不自动重试；现有代码不自动恢复训练，失败后的继续需要单独记录恢复决定，保存状态供恢复使用。

`train-audit`在新进程重载，核对终点摘要、模型及文件哈希，重放全部保存的旧世界验证面板和校准原子的上下文/独立写入读取。token与排名精确核对，fp32浮点margin用1e-5容差。通过后才允许正式读取校准和编辑。

校准沿用开发规则：每组8例、长窗口新单跳答案准确率≥.6。不通过保留结果并停止，不自动长训或扫参。单跳校准不能代替原子写入或组合能力证明。

正式评测沿用现有TTCD、原子CE、sham三种写入：基础参数冻结，病例之间清零A，只给第二跳更正，清空文本/KV后固定A读取E/D/U、同旧答案和第一跳前提。主结果为全病例等权D的新答案＋EOS联合准确率；原前提/组合正确且E更新成功的条件结果另报覆盖率。上下文单跳/两跳参照与独立读取分开。CE/sham使用经过TTCD外层训练的同一父模型，属于同父模型写入参照，不称独立调优后的完整方法竞争。

正式编辑终点继续保存会话快权重并由新进程全病例×方法重放。训练完成、训练重载、读取校准、编辑有效、编辑重放和W&B云端完成分别记录。

## 入口与资源

配置：[ttcd-second-hop-scratch-v1.json](../../../configs/ttcd-second-hop-scratch-v1.json)。项目根目录下：

```bash
TASK_PY=/mnt/disk2_from_server2/siqizhu4/llm-memory/.venv/bin/python

# 仅生成/核对新图，不需要HF或旧权重。
"$TASK_PY" scripts/prepare_ttcd_scratch.py

# 默认只输出GPU8/9计划。
"$TASK_PY" scripts/launch_ttcd_second_hop.py train

# 正式运行入口；两臂preflight/train已执行，重启与当前状态见篇首。
docker --context lm-memory build --network=host -t lm-memory-ttcd:torch2.8-cu128 containers/ttcd-second-hop
"$TASK_PY" scripts/launch_ttcd_second_hop.py preflight --execute
"$TASK_PY" scripts/launch_ttcd_second_hop.py train --execute --wandb-token-file /path/to/private-wandb-token
"$TASK_PY" scripts/launch_ttcd_second_hop.py train-audit --execute --wandb-token-file /path/to/private-wandb-token
"$TASK_PY" scripts/launch_ttcd_second_hop.py calibrate --execute --wandb-token-file /path/to/private-wandb-token
"$TASK_PY" scripts/launch_ttcd_second_hop.py evaluate --execute --wandb-token-file /path/to/private-wandb-token
"$TASK_PY" scripts/launch_ttcd_second_hop.py audit --execute --wandb-token-file /path/to/private-wandb-token
```

正式任务显式使用独立LM Docker context，冻结源码、数据备份与镜像digest，每模型独立容器、GPU UUID、CPU/内存/tmpfs和持久输出。控制器、冻结源码/输入备份、日志和权重实际位于`/ossfs/workspace/llm-memory-editability/ttcd-second-hop-scratch-v1`；持久结果根整体只读映射到`/project/results`，再仅将本任务目录映射为可写的`/project/results/<batch>/<model>`。冻结源码另外只读映射到`/frozen`。任务使用调用者UID/GID及各自临时HOME，适配NFS root_squash。没有真实持久挂载则拒绝启动，不回退本机磁盘。拒绝占用、不换卡、不抢占；现有停止容器/日志保留。W&B默认online，团队zhusq20，项目llm-memory-editability；记录训练损失、旧世界能力、更正单跳、曝光和计算。凭据仅私有文件挂载。新主机不应用旧主机的/lib64兼容方案，也不修改借用的其它项目Python环境。

2026-10-15用户随后要求准备环境并提供W&B凭据；独立lm-memory daemon/固定镜像、NFS持久挂载和私有W&B凭据现已准备，见[环境记录](../ttcd-runtime-preparation-20261015/protocol.md)。镜像CPU契约、独立容器持久重载、正式目录布局CPU preflight及W&B云端核验通过。GPU枚举正确，实际CUDA计算核验待GPU8/9空闲。此前主机短测仍保留，显式关闭tracking并标为fixture；正式长训练没有启动。

## 工程验证

66项CPU检查通过，覆盖事实/组合训练隔离、停止教师目标的外层梯度、独立会话与逐例推理写入一致性、原模型前向兼容、训练终点绑定、新进程重载及完整编辑矩阵。冻结输入备份与持久存储限制亦通过检查。

GPU8的DiscoLoop与GPU9的Loop各完成16步读取预热＋16步写入外层训练，使用配置相同的256实体/32关系图、290词表、128宽/2物理层/2循环、512中间维、batch128和4次写入。两臂CUDA反向有限、保存/重载成功，随后各由新进程完成train-audit、calibrate、evaluate、audit。显存已释放。fixture仅保留小型评价池并显式把读取gate设为0以验证后续执行路径，因此通过不代表读取、更正或组合成功。此前较小词表的核心尺寸短测也保留。

证据目录：`results/ttcd-scratch-code-checks-20261015`，含`cpu-final.xml`、`controller-final.xml`、`gpu{8,9}-full.xml`和对应fixture下的`training/audit.json`、`audit.json`。Ruff检查及格式检查通过。后续LM镜像和online验证见环境记录；没有正式长训练或新的科学结论。

本批短测已退出并释放自身CUDA状态；结束后的再次资源核对发现GPU8/9已有其它项目进程（解释器为`/workspace/envs/python/bin/python`）。正式启动必须重新核对实际占用，不能按此前空闲记录提交或抢占。
