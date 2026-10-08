# LLM Memory: Learning and Editability

研究知识学习的三个层次：**知识能被回答、能被组合、更新后能继续被使用**。每一层都有独立的实验和机制解释，共同回答训练与模型结构怎样形成可提取、可复用、可更新的知识。

**当前研究入口：[Roadmap](docs/roadmap.md)。** 以三层已有观察为基础，推进[实际Transformer的数学解释](docs/memory-scaling-theory.md#transformer-conditional-results)与[通用架构的学习效率比较](docs/roadmap.md#architecture-efficiency)。顺序新知识迁移、写入目标比较和回放复测的完成结果见[结果汇总](docs/results.md#updating)；第一层的学习与提取、第二层的组合使用继续各自构成论文内容。

最新[Loop同起点与规模开发比较](docs/results.md#loop-learning-v1)完成16条训练及独立重载：共享、解共享、仅MLP共享、浅层参照，以及宽度/循环扩展。代码同时记录知识学习、组合使用、新增后使用和实际Transformer跨调用梯度；当前结果未建立Loop的三层统一效率优势。

[MoE顺序学习开发比较](docs/results.md#moe-sequential-v1)也已完成：四条新增训练、两条普通基线复用，独立重载与W&B核验通过；MoE尚未建立新增组合及旧保持的稳定优势，三层学习曲线与成本均已保留。

| 文档 | 用途 |
| --- | --- |
| [Roadmap](docs/roadmap.md) | 唯一当前路线：论文立意、三层问题、证据用途与实验优先级 |
| [结果汇总](docs/results.md) | 三层已有结果、最新科学状态及证据入口 |
| [理论依据](docs/memory-scaling-theory.md) | 条件推导、数学假设及解释边界 |
| [实验协议](docs/experimental-protocol.md) | 共同评价、执行规范与历史机器契约 |
| [项目工作入口](AGENTS.md) / [开发约定](CONTRIBUTING.md) | 环境、恢复、追踪和修改规则 |

各批科学结果、独立重载和云端同步状态分别核查，具体证据与解释边界见结果汇总；已完成批次不重新提交。当前实验范围与冻结入口由roadmap统一维护。

## 实验材料

本GitHub仓库保存源码、测试、配置、开发说明和当前roadmap。实验数据、原始预测、日志、学习曲线、详细研究报告与冻结材料按[存储配置](configs/artifact-storage.json)归档到 [Hugging Face 数据集仓库](https://huggingface.co/datasets/zsqzz/llm-memory-editability)。GitHub单文件小于200 MB；权重、优化器状态及Docker镜像不上传。

本机结果和理论文档可从上表阅读；仅克隆代码仓库时，按下节下载研究材料。远端是归档快照，可能落后于本机；当前roadmap随代码维护，不以远端旧计划覆盖。

| 归档材料 | 路径 |
| --- | --- |
| 数据与tokenizer | [data/](https://huggingface.co/datasets/zsqzz/llm-memory-editability/tree/main/data) |
| 原始预测、日志与运行状态 | [results/](https://huggingface.co/datasets/zsqzz/llm-memory-editability/tree/main/results) |
| 冻结配置、源码、审计与图表 | [docs/development-artifacts/](https://huggingface.co/datasets/zsqzz/llm-memory-editability/tree/main/docs/development-artifacts) |
| 数组归档及原路径索引 | [array-archives/](https://huggingface.co/datasets/zsqzz/llm-memory-editability/tree/main/array-archives)、[ARRAY_ARCHIVES.json](https://huggingface.co/datasets/zsqzz/llm-memory-editability/blob/main/ARRAY_ARCHIVES.json) |
| 文件与哈希清单 | [ARCHIVE_MANIFEST.json](https://huggingface.co/datasets/zsqzz/llm-memory-editability/blob/main/ARCHIVE_MANIFEST.json) |

2026-10-04已核验的历史版本为 [742098e8a543](https://huggingface.co/datasets/zsqzz/llm-memory-editability/tree/742098e8a5437c2640ee0634a4617ae5a88fc46a)：62,376份逻辑材料，其中43,358份数组收录于24个归档包。此版本不代表其后的本机结果或本次文档已同步；后续归档以相应核验记录为准。

## 安装与下载

Python ≥3.10。按需要安装运行、分析、追踪或归档依赖：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[neural,dev,analysis,archive]'
```

本机 `gpulingjun010010012239.os30` 在启动Python前按[AGENTS](AGENTS.md)将 `/lib64` 放在 `LD_LIBRARY_PATH` 首位；独立容器采用其已验证环境。

下载指定批次及其冻结材料，避免用历史快照覆盖活动文档：

```bash
python scripts/download_experiment_artifacts.py \
  --include 'results/parametric-architecture-development-v1/**' \
  --include 'docs/development-artifacts/parametric-architecture-development-v1/**'
```

需要完整历史复现时，可用 `--revision 742098e8a5437c2640ee0634a4617ae5a88fc46a` 固定版本，在独立目录或检查原路径后下载。下载器核对归档及数组SHA256，并恢复原目录结构；私有访问凭据仅保存在本机。未上传权重从实验主机恢复。

## 开发

先读[研究原则](.agents/skills/llm-memory-research/SKILL.md)、[Roadmap](docs/roadmap.md)和[CONTRIBUTING](CONTRIBUTING.md)。

- `src/llm_memory_editability/`：模型、数据生成、训练与分析。
- `scripts/`：执行、分析、审计、报告、追踪及下载入口。
- `configs/`：实验配置、来源和环境。
- `tests/`：真值、划分、训练契约、恢复及评分检查。
- `docs/reports/`：需要保留的历史详细报告；不维护新的研究路线。

```bash
python -m pytest
ruff check .
ruff format --check .
```

代码修改按范围检查；纯文档修改检查链接、锚点、状态及机器契约。部分检查依赖归档材料和未上传权重。静态检查排除冻结源码，不改历史审计哈希。`memory-editability` 与 `memory-editability-neural` 是工程示例，其结果不作为正式研究证据。
