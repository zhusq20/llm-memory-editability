# 独立世界行为确认：复核入口

这批实验已完成并审计，检验训练相关性对原例外学习、新冲突传播及一致更新代价的作用，使用8个新生成世界、约758万参数模型、96条学习和768次配对编辑。主例外编辑收益未获确认，原例外学习收益和一致更新代价得到确认。它不构成跨规模内部路径确认或自然语言LLM验证。正式结论见[确认实验报告](../../shortcut-confirmation-results-v1.md)，与开发世界0/1的报告分开。

## 不可变设计与来源

- `preregistration/preregistration.md`：任何确认世界、模型或效果产生前的设计快照。
- `preregistration/preregistration-lock.json`：2026-09-27 05:14:04 UTC封存，SHA256为 `7c3af938277308fdbf8f384a6bd3afcbed2eb848a58b5501f9a2249a6cdde5f6`。
- `preregistration/queue.json`：完整96父任务，每个父任务含8编辑；不因效果增删任务。
- `preregistration/frozen-files/`：科学算子、汇总、统计及检验的实际源码副本；`snapshot-audit.json`逐项绑定。
- `statistics-design/`：三终点家族、世界级统计、功效敏感性及冻结来源。
- `preparation-audit.json`：8世界的24数据文件与10项官方来源文件；先通过前瞻锁，后生成数据。

独立单位是世界；初始化、组织、关系及支持集均嵌套在世界内。主结果使用8个世界内high−low配对均值的两侧t检验，三终点Holm校正，并同时给边际及Bonferroni同时区间。所有三项终点及全体传播、E/U代价无论结果方向都保留。支持集不能充作额外独立样本。

## 完成后复核顺序

原始产物入口为项目根下 `results/bios-shortcut-confirmation-v1/runs/`。全部96父任务于05:56:06 UTC完成；完整原始归档的10187个文件、83,593,945,025内容字节于06:44:45 UTC通过独立逐成员回读校验，随后才进入正式汇总。分析所用索引为同级 `verified-analysis-archive.tar.json`，如实指向 `/dev/shm/llm-memory-siqizhu4-confirmation-20260927/raw-complete.tar`。这是共享盘回读缓慢后的存储恢复：原NVMe来源和共享盘完整副本均保留，共享盘自己的回读完成状态另记，不用内存盘凭据冒称其已验证。最终状态见 `completion-audit.json`；存储恢复记录见结果目录的 `archive-analysis-gate.json`。2208份学习/编辑权重均须与producer封印相符；该过程核验存档权重及预测文件，不声称重新前向生成全部预测。

共享盘归档已于06:54:56 UTC通过完整逐成员校验，索引 `raw-complete.tar.json` 与分析索引的10187成员、字节数、SHA256完全一致；最终凭据为 `durable-archive-completion.json`。下列命令保留实际分析所用RAM索引；复核时也可使用已验证的持久归档索引。历史执行记录中的pending状态没有回写修改。

在原冻结源码与环境下，顺序执行：

```bash
PYTHONPATH=src python scripts/summarize_bios_shortcut_confirmation.py \
  --config configs/bios-shortcut-confirmation-v1.json \
  --lock results/bios-shortcut-confirmation-v1/runs/preregistration-lock.json \
  --archive-index results/bios-shortcut-confirmation-v1/verified-analysis-archive.tar.json \
  --output results/bios-shortcut-confirmation-v1/summary \
  --require-complete

PYTHONPATH=src python scripts/analyze_bios_confirmation_stats.py \
  --summary results/bios-shortcut-confirmation-v1/summary \
  --locked-design docs/development-artifacts/shortcut-confirmation-v1/statistics-design \
  --output results/bios-shortcut-confirmation-v1/statistics
```

第一步从保存的逐查询预测与固定真值重算分数，并验证完整矩阵、数据划分、曝光、配对支持与归档。第二步只接受已完整通过的汇总及终点哈希；576学习节点和3072编辑节点用于描述，事先指定的15360/512终点才进入三项确认检验。代码、环境或来源变更会被守卫拒绝，应在隔离的冻结副本中复核，不覆盖当前源码或已有产物。

固定支持上的每个确认编辑终点分母为9人；学习终点为每条链固定64名原例外留出人员。旧知识损伤只以原答对且真值未改变的知识为分母，另报共同已知的同真值集合与覆盖。自主两步要求首跳为合法组织并正常终止、最终城市正确且正常终止；不额外要求首跳组织真值正确，首跳正确率另报。


## 次要保持控制与报告实现的盲态补充

原设计已要求“共同已知且真值相同”的保持对照，但启动后的盲态核查发现原汇总器漏了导出入口。补充审计于05:42:44 UTC单独封存，14项开发/合成测试与独立复核通过，凭据为 `common-u-supplement-lock.json`；报告层于05:46:12 UTC封存，11项合成测试与独立复核通过，凭据为 `reporting-source-receipt.json`。两者均在任何确认效果查看之前实现，但未在模型启动前封存，不能将它们冒称原前瞻源码锁的一部分。原三项确认终点与统计实现未改，没有新增次级显著性检验。

主汇总与统计完成后，继续：

```bash
PYTHONPATH=src python scripts/summarize_bios_confirmation_common_u.py \
  --config configs/bios-shortcut-confirmation-v1.json \
  --lock results/bios-shortcut-confirmation-v1/runs/preregistration-lock.json \
  --receipt docs/development-artifacts/shortcut-confirmation-v1/common-u-supplement-lock.json \
  --summary results/bios-shortcut-confirmation-v1/summary \
  --archive-index results/bios-shortcut-confirmation-v1/verified-analysis-archive.tar.json \
  --output results/bios-shortcut-confirmation-v1/common-u

PYTHONPATH=src python scripts/report_bios_shortcut_confirmation.py \
  --summary results/bios-shortcut-confirmation-v1/summary \
  --statistics results/bios-shortcut-confirmation-v1/statistics \
  --common-u results/bios-shortcut-confirmation-v1/common-u \
  --output results/bios-shortcut-confirmation-v1/report
```

报告入口拒绝覆盖已有正式报告。共同旧正确集合取决于两种训练后的结果，仅用作描述性保持控制；同时报告原U、案例平均、合并损伤整数/旧正确分母、世界平均与有效世界数。该子集不替代三项主要效应。
