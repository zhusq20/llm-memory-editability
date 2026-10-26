# LLM Memory: Learning and Editability

研究语言模型怎样把事实变成可回答、可组合、更新后仍能使用的知识，以及架构怎样决定这些能力。核心问题是：**知识在哪里、推理在哪里、二者怎样连接。**

本仓库维护两个项目：**LM Knowledge**研究知识学习、调用与更新，**Loop**研究循环Transformer的架构与跨轮交接。两者共用代码、测试与实验工具，分别维护路线图；共享结果摘要按主题组织。2026-10-26确认继续使用同一个GitHub仓库。

## 从这里开始

| 需要了解什么 | 入口 |
| --- | --- |
| LM Knowledge现在做什么、为什么 | [当前路线图](docs/roadmap.md)，先读§1 |
| 已有什么证据、方法比较进行到哪里 | [结果摘要](docs/results.md) |
| Loop架构的独立工作线 | [Loop路线图](docs/loop-roadmap.md)与[历史问题索引](docs/loop-history.md) |
| 数学解释 | [理论依据](docs/memory-scaling-theory.md)，按问题选读 |
| 实验评价、执行与恢复 | [实验协议](docs/experimental-protocol.md)、[工作入口](AGENTS.md)、[开发约定](CONTRIBUTING.md) |
| 查旧实验、旧提案与被覆盖的状态 | [历史索引](docs/results.md#history-index)、[文档归档](docs/reports/knowledge-history-20261022/README.md) |

日常阅读止于路线图与结果摘要。历史材料用于查证，不构成另一份待办；完整原始证据和失败记录保留。训练完成、独立重载、W&B核验分别报告，已完成批次不重新提交。

## 实验材料

GitHub仓库保存源码、测试、配置、开发说明，两套Roadmap、结果汇总、理论依据、实验协议、文字报告和论文图表源文件；另保存少量关键运行状态与验证日志。2026-10-26的[提交材料清单](docs/github-snapshot-20261026.json)记录纳入文件的版本与检查结果。批量结果JSON、逐题预测、曲线导出、重复冻结源码、生成的PNG/PDF不纳入此次提交。运行状态是文件级快照，不代表实时进度；同步代码不启动或恢复训练。

部分`runtime/`在本机指向仓库外的实验盘；少量关键状态另存为[运行记录快照](results/github-sync-20261026/runtime-snapshots/manifest.json)，清单保留原路径、复制时间和SHA256。本机链接与运行目录保持原样。

原始数据、数组、大体积导出与未选入的运行材料仍按[存储配置](configs/artifact-storage.json)保存在实验存储，并按需归档到 [Hugging Face 数据集仓库](https://huggingface.co/datasets/zsqzz/llm-memory-editability)。本次GitHub同步不代表更新了Hugging Face归档。模型权重、优化器状态和Docker镜像不上传。`results/`默认仍被忽略，后续纳入新的结果文件须经检查后显式添加，已跟踪文件可正常更新。

只读结论时，直接阅读上表中的结果汇总和理论文档。需要复核原始证据时，按下节下载归档材料。Hugging Face上的归档是快照，可能落后于本机的最新结果；当前研究文档随代码维护，不要用归档中的旧计划覆盖它们。

| 归档材料 | 路径 |
| --- | --- |
| 数据与tokenizer | [data/](https://huggingface.co/datasets/zsqzz/llm-memory-editability/tree/main/data) |
| 原始预测、日志与运行状态 | [results/](https://huggingface.co/datasets/zsqzz/llm-memory-editability/tree/main/results) |
| 冻结配置、源码、审计与图表 | [docs/development-artifacts/](https://huggingface.co/datasets/zsqzz/llm-memory-editability/tree/main/docs/development-artifacts) |
| 数组归档及原路径索引 | [array-archives/](https://huggingface.co/datasets/zsqzz/llm-memory-editability/tree/main/array-archives)、[ARRAY_ARCHIVES.json](https://huggingface.co/datasets/zsqzz/llm-memory-editability/blob/main/ARRAY_ARCHIVES.json) |
| 文件与哈希清单 | [ARCHIVE_MANIFEST.json](https://huggingface.co/datasets/zsqzz/llm-memory-editability/blob/main/ARCHIVE_MANIFEST.json) |

2026-10-04核验过的归档版本是 [742098e8a543](https://huggingface.co/datasets/zsqzz/llm-memory-editability/tree/742098e8a5437c2640ee0634a4617ae5a88fc46a)：共62,376份逻辑材料，其中43,358份数组收录在24个归档包中。这个版本不包含之后在本机产生的结果，也不代表本次文档已同步到归档；之后的归档以相应的核验记录为准。

## 安装与下载

需要 Python ≥3.10。按需要安装运行、分析、实验追踪或归档依赖（下面的命令安装全部四组可选依赖）：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[neural,dev,analysis,archive]'
```

在主机 `gpulingjun010010012239.os30` 上，启动Python之前须按[AGENTS](AGENTS.md)的说明把 `/lib64` 放在 `LD_LIBRARY_PATH` 的最前面；独立容器使用其已验证的环境。

下载指定批次的结果及其冻结材料（只下载指定路径，避免用归档中的历史快照覆盖当前文档）：

```bash
python scripts/download_experiment_artifacts.py \
  --include 'results/parametric-architecture-development-v1/**' \
  --include 'docs/development-artifacts/parametric-architecture-development-v1/**'
```

需要完整复现历史结果时，可用 `--revision 742098e8a5437c2640ee0634a4617ae5a88fc46a` 固定归档版本；请下载到独立目录，或先检查原路径再下载。下载器会核对归档包和数组的SHA256，并恢复原来的目录结构。私有访问凭据只保存在本机。未上传的权重从实验主机恢复。

## 开发

修改代码或文档之前，先读[研究原则](.agents/skills/llm-memory-research/SKILL.md)、[Roadmap](docs/roadmap.md)和[CONTRIBUTING](CONTRIBUTING.md)。

- `src/llm_memory_editability/`：模型、数据生成、训练与分析代码。
- `scripts/`：实验执行、分析、审计、报告、实验追踪和下载的入口脚本。
- `configs/`：实验配置、数据来源和环境配置。
- `tests/`：检查真值、数据划分、训练是否符合冻结的训练设定、中断恢复和评分。
- `docs/reports/`：需要保留的历史详细报告；这里不维护新的研究路线。

运行测试与静态检查：

```bash
python -m pytest
ruff check .
ruff format --check .
```

- 修改代码时，按改动范围运行相应检查。只改文档时，检查链接、锚点、状态描述，以及脚本按标题读取的历史契约文本（见实验协议）是否保持不变。
- 部分检查依赖归档材料和未上传的权重。
- 静态检查排除冻结源码，以免改动历史审计哈希。
- `memory-editability` 与 `memory-editability-neural` 是工程示例，其结果不作为正式研究证据。
