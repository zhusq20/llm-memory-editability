# 300k基础模型直接TTCD写入：执行与核验

## Material Passport

- Origin Skill: llm-memory-research
- Origin Mode: run
- Origin Date: 2026-10-15 America/Chicago
- Verification Status: VERIFIED（独立新容器逐字段重放；云端完整JSON一致）
- Version Label: ttcd-foundation-novel-fact-v1

用户明确授权跳过common10k和single10k、直接从已有300k模型做TTCD蒸馏，只更新MLP down。实际加载`discoloop-success-restore-v1/discoloop/checkpoint-0300000.pt`，SHA256 `56d17cedf29db93c713e7ec118b55955118dfd54085aa5240806aee14401f671`；这是原配方复建终点，不称找回历史权重。原`foundation_inputs`核对原基础审计门槛、step300000、完整spec、文件哈希及内部模型digest。新增入口拒绝320k/不同哈希/非零common或writer加载步数。每阶段`initialization.json`直接记录加载来源，A0与原骨干logits逐值一致。

基础θ从始至终冻结，原基础检查点只读。四个物理层的有效down权重为`W_down+A`，只有A按长/短窗MLP输出差更新，down bias、MLP up、QKV、归一化和embedding均不变，两轮共用同一个A。没有借用320k teacher或fast状态，也没有新答案CE、原子回放或两跳训练。仅替换初始化，查询、写入、观察、评分和统计直接复用上一批冻结runner与父批`source-v2`，不改它们的字节。

8开发/32正式新增键与两种跳位问题完整复用上一批；它们在原图不存在，原图真值不改。每事实由新的零A独立开始，4次写入、η=.001、窗32/16、间隔17保持。开发结果没有用于修改配方，正式低分亦全部保留。

| 阶段 | 科学快更新 | 用时 | 核验 |
| --- | ---: | ---: | --- |
| 开发8事实 | 32 | 69.8秒 | 退出0；θ/A0/输入不变 |
| 正式32事实 | 128 | 255.4秒 | 退出0；128次更新均非零；完整两跳/保持题保留 |
| 独立重载全40事实 | 0 | 282.6秒 | 新容器退出0，全部记录逐字段精确一致 |
| 云端读取三阶段 | 0 | 约3秒 | 退出0，W&B finished、计数与完整JSON一致；模型前向0 |

本批慢参数训练0；160次科学快更新之外，小型随机模型工程检查1次局部更新单列。七项核心契约、Ruff检查通过。初次静态检查发现导入/长行格式错误，在任何科学执行前修复；未重试科学阶段或修改冻结父源码。

使用固定LM镜像、`docker --context lm-memory`、CPU80–83/4线程/16GiB/FP32，不占GPU、不抢占或重启当前六条旧任务。每个科学容器2小时硬超时，监测存活、逐事实进度和退出；精确挂载、模型ID/参数见`launch-*.json`，日志及退出状态均保留。容器内共同命令为`python /experiment-source/scripts/run_ttcd_foundation_novel_facts.py --design /experiment-source/design.json --panel /experiment-source/panel.json --parent-config /frozen/configs/experiment.json`，追加`--phase development|evaluation|audit --out /output/<phase> --tracking online`；audit另带`--input-root /output`。

W&B：[开发](https://wandb.ai/zhusq20/llm-memory-editability/runs/zz9ppmu8)、[正式](https://wandb.ai/zhusq20/llm-memory-editability/runs/3vbjhkwf)、[独立重载](https://wandb.ai/zhusq20/llm-memory-editability/runs/qcd2wbf7)。结果是长窗teacher0/32、新单跳首轮/最终0/32，新→旧1/256（同一道题写前已正确）、旧→新0/253且旧桥253/253正确；因此写入成功前提的条件覆盖为0，按NA报告。原检查池旧答案无变化，不能据此称模型已学到新知识。

只说明当前记录/4步TTCD的直接300k路径未成功，不能推定两阶段各10k必需，不能将无原子写入成功时的两跳低分归因于架构组合上限。详细数据、对照和边界见[结果汇总](../../results.md#ttcd-foundation-novel-fact-v1)。全部40份快状态、原始逐题记录、加载/源码哈希、容器日志、云端收据归档于`/ossfs/workspace/llm-memory-editability/ttcd-foundation-novel-fact-v1/`；容器特有W&B符号链接只保留链接文本，不跟随失效路径。完成/归档收据在批次专属`completion-manifest.json`与`archive-verification.json`。
