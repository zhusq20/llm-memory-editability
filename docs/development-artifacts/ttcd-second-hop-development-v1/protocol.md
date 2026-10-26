# DiscoLoop × TTCD：第二跳更正后的独立组合读取

状态：2026-10-15 代码准备及工程验证完成，真实检查点上的校准与编辑未启动。用户指定预计使用 GPU8、9。复用已建立两跳能力的父模型，不重新训练基础模型。

## 问题、历史与来源

已知 `(a,r1)→b`、`(b,r2)→c`；只提供第二跳更正 `(b,r2)→d`，之后检查独立查询 `(a,r1,r2)→d`。d 是已有实体、不同于 c。写入不含依赖的两跳问题或新组合答案。主要看新答案的组合使用与无关旧知识保持。

已有 `edit-scope-development-v1` 做过普通 Loop 的第二跳原子编辑；`supervised-alignment-development-v1` 已建立 DiscoLoop 在同一 Wang 图上的组合能力。本次新增 TTCD 上下文差异写入与快权重独立保存，不重开旧 KV/迭代范围/长训练网格。两个父模型原有组合水平不同，跨模型差值不能单独归因于编辑时的通道；同一父模型内的写入比较与前提覆盖率同时报告。

来源：[TTCD v1 §3/附录 A.1.1](https://arxiv.org/html/2608.01672v1)、[作者源码 revision](https://github.com/dangxingyu/ttcd/tree/ad67038cb4136ce9cc641d4f2570d2931e1f9454)、[DiscoLoop v2 §4](https://arxiv.org/html/2607.00341v2#S4)。本实现是 TTCD 式(6)核心梯度在现有 GPT/GELU 上的适配，不称作者完整 IP-TTCD-v9 复现。

## 计算与实现边界

保留完整 GPT attention、残差、LayerNorm、GELU MLP、绝对位置 embedding 与 DiscoLoop 软解码再编码通道。GPT Conv1D 下投影存储为 `[intermediate,hidden]`，局部更新为 `ΔW = η z_student.T @ (y_teacher-y_student)/N`。教师目标停止梯度，上游激活在这次求导中固定，不能称贯穿所有层/循环的完整梯度。

局部长短分支共享输入残差与 Q/K/V；仅在第一轮原子查询位置计算写入，每个物理层一次，所有循环共享同一个更正矩阵。全部层读完包后同时提交，当前包不会读取自己产生的更新。默认重复包 4 次，η=1e-4 是开发起点；没有作者的可学习卷积/投影、归一化特征或 off-policy scan，不直接套用其学习率 0.3，也不声称当前配方有效或最优。

读取能力校准使用完整长/短窗口前向，它们与局部共享残差分支不同。短窗口也可能通过多层传播获得远处信息，局部 student 的输入残差也来自 teacher；“记录在直接窗口外”不等于整个计算绝无该信息。独立查询清除文本/KV，用于进一步分离信息载体。

## 输入、病例与校准

HF 固定版本 `6c580455ce1667274ea592c1c88f5d0fb1e5a190`，仓库 `zsqzz/llm-memory-editability`。已恢复并核验原 `data/{world.npz,panels.npz,complete.json}` 和 GPT-2 `config.json`。873,212,794 字节的原数组包 SHA256 为 `0dfe4eedc158032690546dd85a2e11663a9922e5470a084dc20b89414e16786b`；下载器核验整包及所选成员的哈希，安全恢复原路径。

该版本没有所需的原 DiscoLoop/Loop `.pt` 权重，也没有 10 月 12 日 DiscoLoop 批次文件。可用性记录：`data/ttcd-second-hop-assets/hf-availability.json`。原权重需从原实验主机恢复，不以随机模型替代、不重启旧训练。

当前图含 40,000 原子事实、800,000 两跳链。预测前固定随机病例：校准 8 ID＋8 OOD 事实，评价另 16 ID＋16 OOD，编辑目标互不重叠；每例最多 8 条 D 查询、各最多 16 条无关原子/组合及同旧答案原子对照。D 排除 φ7.2 的原训练组合，目标边不同时充当第一跳。实际候选库共 48 事实、380 条 D 查询，两种父模型共用；包含失败，不按事后结果选样本。同一个已观察图不称新世界确认。

更正包：`[b,r2,d,EOS] + 17 个原 pad/EOS filler + [b,r2]`，长 23，不在查询后给答案。teacher 直接窗口 32、student 16。使用原 token 和自然递增位置；不重置位置 embedding 来隐藏长输入的分布变化。

关闭快权重更新，在校准事实上测 teacher 的新答案、答案/EOS联合准确率，另报完整短窗口及闭卷基线。每类至少 8 例、teacher 新答案准确率 ≥0.6 才继续。这是开发规则，不是论文统一要求。失败保存 `reading_needs_attention`、退出码3，停止后续写入；没有自动训练读取接口或自动超参数扫描。失败不能归因于两跳能力或 TTCD 一般无效。

## 方法、读出与核验

两个父模型分别比较 `ttcd`、`atomic_ce`、`sham`。GPU8：DiscoLoop；GPU9：普通 Loop。原子 CE 在同样物理下投影的更正张量上做20次 Adam、lr=1e-4，只用原子答案与EOS，无回放；sham为零更正。基础参数全部冻结，病例与方法之间重新从零更正开始。

每次保存终点，在不继续写入的条件下读取四格：

| 查询 | 有更正记录 | 独立读取，只保留快权重 |
| --- | --- | --- |
| 单跳 | 更正读取参照 | E：写入成功 |
| 两跳 | 上下文组合参照 | D：第二跳更新传播，主结果 |

独立查询没有文本、隐藏状态、KV或其它查询结果。主评分沿用答案与实际生成EOS联合正确；同时保存答案准确率、新旧答案排名与margin。开放上下文参照固定每例一条D，不与全部D独立主指标混报。

主指标是固定病例等权、全量D新答案联合准确率；编辑前两条原子事实和组合都正确、且E编辑成功的条件结果另报分母与覆盖率，不据此丢弃主结果。U、同旧答案及第一跳前提分别报编辑前后准确率、原本正确子集保持率。查询不当作独立事实/世界；不同方法条件子集不同，不能仅比较条件百分比声称配对优势。

快权重绑定基础模型哈希。新进程从原父模型和快权重重载，检查完整病例×方法矩阵、E/D/U/同答案的token、EOS、排名/margin及开放上下文参照，拒绝源码/配置/输入漂移和缺失矩阵。会话快权重更新不是永久基础参数编辑。

## 环境和命令

当前主机 `sn4622129202`，RTX PRO 6000 Blackwell，工程验证用 PyTorch 2.8.0+cu128、Transformers 4.57.3。旧主机的 `/lib64` 修复不套用此主机。当前没有 `lm-memory` Docker context；正式运行需要准备该独立 context，不回退到 default/D157/D159。

以下命令在项目根目录运行。HF 凭据只从本机私有文件读取，W&B密钥只读挂载：

```bash
# HF 数据已恢复；重跑会复用固定版本缓存并核验。
python scripts/prepare_ttcd_assets.py --arrays

# 先恢复原权重到配置路径，或在配置中指定实际位置。
# LM 独立容器环境准备完成后：
docker --context lm-memory build -t lm-memory-ttcd:torch2.8-cu128 containers/ttcd-second-hop

# 默认只显示GPU8/9的计划。
python scripts/launch_ttcd_second_hop.py calibrate

# 各阶段明确启动，校准失败时不进入evaluate。
python scripts/launch_ttcd_second_hop.py preflight --execute
python scripts/launch_ttcd_second_hop.py calibrate --execute --wandb-token-file /path/to/private-wandb-token
python scripts/launch_ttcd_second_hop.py evaluate --execute --wandb-token-file /path/to/private-wandb-token
python scripts/launch_ttcd_second_hop.py audit --execute --wandb-token-file /path/to/private-wandb-token
```

首次执行冻结源码与配置，后续使用原快照，改配置另立批次。每任务独立容器、GPU UUID、4CPU且cpuset不同、48GiB内存、独立tmpfs和持久输出；输入及源码只读，仅该任务输出可写。启动前检查显存、利用率、计算进程与LM GPU锁，拒绝占用，不换卡抢占。容器/失败/退出记录保留，硬超时只停止本控制器命名的任务。

W&B默认 `zhusq20/llm-memory-editability` online，各模型/阶段单独run，记录配置、GPU资源、各方法/组曲线、曝光、tokens、耗时和计算估算。前向FLOPs按实际执行的稠密matmul形状估算，反向另列常规2×前向proxy；不计归一化/softmax/内存流量，不能称实测算力或等计算比较。SDK结束、权重独立重载、云端核验分开记录；准备阶段没有科学W&B run。

## 已完成工程验证

CPU相关测试58项通过。GPU8/9各一项完整小模型流程通过，包括真实CUDA写入、CE反向、保存加载和新Python进程重放；显存已释放。Ruff静态及格式检查通过。证据：`results/ttcd-second-hop-code-checks-20261015/{cpu,gpu8,gpu9}.xml` 与各GPU fixture的 `audit.json`。

工程流程使用显式标记的随机小模型/玩具图，读取gate放宽只为走通代码。这不证明真实父模型可读取更正、编辑有效或组合传播成立。原300k权重、LM Docker context/镜像仍待恢复/准备；真实校准和编辑未启动。配置：`configs/ttcd-second-hop-development-v1.json`。
