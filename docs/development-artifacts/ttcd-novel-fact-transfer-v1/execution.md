# 新增知识两跳评测执行记录

## Material Passport

- Origin Skill: llm-memory-research；沿用academic-research-suite实验设计产物
- Origin Mode: run
- Origin Date: 2026-10-15 America/Chicago / 2026-10-16 UTC
- Verification Status: VERIFIED（独立新容器逐字段重放及三阶段云端完整JSON一致）
- Version Label: ttcd-novel-fact-transfer-execution-v1

用户已明确授权实际执行[原设计](protocol.md)，不重新训练基础模型、不覆盖旧事实。原设计配置及其历史“尚未执行”状态原样保留；实时状态记于`results/ttcd-novel-fact-transfer-v1/execution-state.json`，原设计纯图核查不当作模型结果。

实际检查点为原配方300k**独立复建**→common10k→single10k的已审计终点，累计320k慢参数更新；是之前两项逐轮诊断的同一θ，不是直接加载裸300k模型，也不是尚在训练的stream/一步终点。用户询问后已在执行阶段再次说明。文件与SHA256固定为`ttcd-successful-base-v1/single/training/parent.pt`、`fc9c65b71a863e1ed48325651213ce3859647804262e360bf0f155f452912b45`。本批新增慢参数更新0，8条开发/32条正式各独立由原A0起步，每事实4次快状态更新；另有一项小型随机模型的工程局部写入测试，单列，不计入160次科学快更新。

原模型与数据只读，`PYTHONPATH=/frozen/src`绑定父批`source-v2`。新增入口仅构造缺失键原子包，复用原`TTCDReader.write`和已验证的逐轮观察。没有旧答案字段、没有两跳训练标签、不跨事实累积A。每次两跳、分步和保持查询都重新计算，只传快状态，不携带记录/KV。全部事实两种用途共用同一A4。

固定LM镜像、CPU80–83/4线程/16GiB/FP32、单容器2小时硬超时；CPU核与六个在训容器的显式cpuset不重叠，无GPU请求。容器入口通过独立临时目录`/tmp/lm-ttcd-novel-fact-transfer-v1`保存输出，不更改旧结果权限。启动前记录真实容器与资源，日志、容器ID/命令和退出状态保留于该目录。

六项契约通过：新增包与旧键拒绝、非实体预测不补真值桥实体、实际局部写入冻结骨干且可重载，以及旧观察的数值隔离/因果前缀/位置映射三项。新文件Ruff检查与格式检查通过；首次样式检查仅发现导入排序，未进入测试或模型执行，修复记录保留。执行源码/配置/面板与其哈希保存在`source-manifest.json`，工作区与执行快照一致。

三个阶段依次执行，后两阶段不重新写开发快状态：

| 阶段 | 样本/快更新 | 入口参数 | 当前状态 |
| --- | --- | --- | --- |
| 开发 | 8条，32次快更新 | `--phase development --out /output/development` | 正常退出0；原输入、骨干与A0不变；旧128题答案/EOS与原CUDA FP32基线逐token一致 |
| 正式 | 32条，128次快更新 | `--phase evaluation --out /output/evaluation` | 正常退出0，233.0秒；骨干/A0/输入保持；所有题完整保留 |
| 独立重载 | 全40条，0次快更新 | `--phase audit --input-root /output --out /output/audit` | 正常退出0，266.5秒；仅载入已保存A，全部记录逐字段精确复现 |

共同命令为`python /experiment-source/scripts/run_ttcd_novel_fact_transfer.py --design /experiment-source/design.json --panel /experiment-source/panel.json --parent-config /frozen/configs/experiment.json`，追加上表参数与`--tracking online`。精确Docker挂载/限制/完整参数以各`launch-*.json`为准，不能只用这条容器内命令推断宿主路径。

W&B：[开发](https://wandb.ai/zhusq20/llm-memory-editability/runs/wel804ff)、[正式](https://wandb.ai/zhusq20/llm-memory-editability/runs/vq1nrnlt)、[独立重载](https://wandb.ai/zhusq20/llm-memory-editability/runs/xvpoi1nt)。各阶段均保存进度、原始逐题答案/EOS/阶段argmax与概率/排名、必要旧前提、模型自行分步结果及旧保持。三阶段均finished；只读云端核验容器退出0，更新计数及完整summary JSON与本地一致，不运行模型。

正式结果：新增原子首轮0/32（均EOS），第二轮第4块末32/32；新→旧首轮桥0/256、最终1/256（A0也是1/256，但不是同一道题）；旧→新首轮桥251/253、最终0/253，在251道前提成立题中仍0。自行分步的最终答案为254/256和253/253，若还要求第一步桥正确则254/256和251/253，两个口径分别保留。旧U原子/组合保持2030/2048、2038/2048；每池是64旧题×32快状态，不能当作2048独立事实。完整证据和解释见[结果](../../results.md#ttcd-novel-fact-transfer-v1)。

实际新增基础/writer慢参数训练0，科学快状态更新160（开发32、正式128），独立审计0。原科学设计与执行快照保留，不因负结果重选新键、增加写入步数或改用另一终点。全部原始结果、40份快状态、源码快照、日志/退出状态与核查收据保存于专属持久目录；归档只复制正常文件，容器内W&B绝对路径符号链接另存链接文本，避免跟随失效路径。最终完成及归档哈希见`results/ttcd-novel-fact-transfer-v1/completion-manifest.json`与`archive-verification.json`。
