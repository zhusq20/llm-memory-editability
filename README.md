# LLM Memory: Learning and Editability

本项目研究语言模型怎样把事实知识学进参数，并在之后用上这些知识。“用上”分三个层次，每个层次都有独立的实验和机制解释：

- **知识能被回答**：模型学过一条原子事实（atomic fact），例如“A的母亲是B”，之后能否答出它，换一种提问模板是否仍能答出。
- **知识能被组合**：模型能否把两条已学事实接起来，回答没有训练过的两跳查询（two-hop query）。例如由“A的母亲是B”和“B出生于C”回答“A的母亲出生于哪里”；其中B称为桥实体（bridge entity）。
- **更新后能继续被使用**：新增或改写一条事实之后（通过知识编辑或继续训练），依赖这条事实的多跳答案是否随之更新（编辑传播，edit propagation），不应改变的旧知识是否保持。

三个层次共同回答：训练过程与模型结构怎样形成可提取、可复用、可更新的知识。

**当前研究入口：[Roadmap](docs/roadmap.md)。**

## 主线工作假说：路径一致性

论文主线采用[路径一致性假说](docs/roadmap.md#path-consistency)：一个事实只能被那些读取路径与该事实被训练或编辑时相同的查询使用。读取路径包括三个方面：

- **层范围**：查询在哪些层读取这个事实，是否就是存储该事实的层。
- **token位置**：在哪个token位置读取；多跳查询中，前一跳是否在后一跳读取的位置上完成。
- **表示格式**：该位置的隐藏状态是否处在下一跳查表所读取的形式，例如是否与实体的输入embedding对齐。

按这一假说，事实能否被组合、编辑能否传播，都取决于新的用法是否经过事实被训练或编辑时的同一条路径。这是待检验的工作假说，不是已建立的结论。目前的状态：

- 相关论文的复现已完成；按被编辑事实所在的跳（第一跳或后续跳）分组、在看到结果之前固定的预注册分析也已完成。[自然模型上的结果](docs/results.md#path-hop-analysis-v1)尚未建立统一的路径机制。
- 顺序知识注入（先学旧知识，再学新原子事实）、编辑的训练目标比较和回放复测已经完成，结果见[结果汇总](docs/results.md#updating)。
- 第一层的学习与提取、第二层的组合使用，仍各自构成论文的独立内容。

各批次的执行进度与完整结果统一见[结果汇总](docs/results.md)。其中两项架构比较目前都没有建立稳定优势：[循环Transformer复制初始化与规模比较](docs/results.md#loop-learning-v1)没有显示三个层次上统一的效率优势；[MoE顺序学习比较](docs/results.md#moe-sequential-v1)没有显示新增组合与旧知识保持上的稳定优势。

## 文档

| 文档 | 用途 |
| --- | --- |
| [Roadmap](docs/roadmap.md) | 唯一的当前研究路线：论文立意、三个层次的问题、各项证据的用途与实验优先级 |
| [结果汇总](docs/results.md) | 三个层次的已有结果、最新科学状态和证据入口 |
| [理论依据](docs/memory-scaling-theory.md) | 数学推导及其假设、适用条件和解释边界 |
| [实验协议](docs/experimental-protocol.md) | 所有实验共同遵守的评价与执行规则，以及旧脚本仍要读取的历史预注册分析计划 |
| [项目工作入口](AGENTS.md) / [开发约定](CONTRIBUTING.md) | 环境、中断恢复、实验追踪和修改规则 |

每批的科学结果、新进程重载复算和W&B记录分别核查，证据与解释边界见结果汇总；已完成的批次不重新提交。当前实验范围与冻结配置的入口由Roadmap统一维护。

## 实验材料

GitHub仓库保存源码、测试、配置、开发说明，以及上表中的Roadmap、结果汇总、理论依据和实验协议。其余实验材料按[存储配置](configs/artifact-storage.json)归档到 [Hugging Face 数据集仓库](https://huggingface.co/datasets/zsqzz/llm-memory-editability)，包括实验数据、原始预测、日志、学习曲线、历史详细研究报告和冻结材料。模型权重、优化器状态和Docker镜像不上传。

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
