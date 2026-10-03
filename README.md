# LLM Memory: Learning and Editability

研究事实记忆如何转化为知识引用，以及这种转化在什么条件下成立。以受控 Transformer、Loop、MoE 和 mHC 实验，检验事实提取、未见组合与参数编辑后的更新传播。

本 GitHub 仓库保存源码、测试、配置与开发说明。实验数据、原始预测、日志、学习曲线、研究报告和冻结执行材料的目标存储为 [Hugging Face 数据集仓库](https://huggingface.co/datasets/zsqzz/llm-memory-editability)。单个 GitHub 文件必须小于 200 MB；此限制不适用于 Hugging Face。模型权重、优化器状态与 Docker 镜像归档不上传。

2026-10-04 归档状态：本地快照和校验清单已准备，Hugging Face 因写入 token 过期尚未上传。下面列出目标路径，上传完成后才可访问和下载。原始实验材料仍保存在实验主机。

## 实验材料

Hugging Face 中保留原项目相对路径，下载后可继续使用原脚本。

| 材料 | Hugging Face 链接 |
| --- | --- |
| 实验数据集与 tokenizer | [data/](https://huggingface.co/datasets/zsqzz/llm-memory-editability/tree/main/data) |
| 原始预测、数值数组、日志和运行状态 | [results/](https://huggingface.co/datasets/zsqzz/llm-memory-editability/tree/main/results) |
| 冻结配置、源码快照、审计和图表 | [docs/development-artifacts/](https://huggingface.co/datasets/zsqzz/llm-memory-editability/tree/main/docs/development-artifacts) |
| 研究结果汇总 | [docs/results.md](https://huggingface.co/datasets/zsqzz/llm-memory-editability/blob/main/docs/results.md) |
| 研究计划 | [docs/hebbian-learning-plan-v1.md](https://huggingface.co/datasets/zsqzz/llm-memory-editability/blob/main/docs/hebbian-learning-plan-v1.md) |
| 完整研究进度索引 | [project-docs/README.md](https://huggingface.co/datasets/zsqzz/llm-memory-editability/blob/main/project-docs/README.md) |
| 文件清单、SHA256 与排除记录 | [ARCHIVE_MANIFEST.json](https://huggingface.co/datasets/zsqzz/llm-memory-editability/blob/main/ARCHIVE_MANIFEST.json) |

运行中实验只归档上传时的文件快照；后续节点需要再次同步，归档不代表实验已经完成。存储规则及路径映射另见 [configs/artifact-storage.json](configs/artifact-storage.json)。Git 历史中的旧实验材料不作历史重写。

## 安装与下载

Python ≥3.10。按需要安装运行、分析、追踪或归档依赖：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[neural,dev,analysis,archive]'
```

下载全部已归档材料到原目录：

```bash
python scripts/download_experiment_artifacts.py
```

也可以只下载某一批，或用 Hugging Face 提交 SHA 固定版本：

```bash
python scripts/download_experiment_artifacts.py \
  --include 'results/parametric-architecture-development-v1/**' \
  --include 'docs/development-artifacts/parametric-architecture-development-v1/**'
python scripts/download_experiment_artifacts.py --revision DATASET_COMMIT_SHA
```

私有数据集需要先在本机配置 Hugging Face 访问权限；凭据不要写进仓库。下载器只恢复 `data/`、`results/`、`docs/`，不会覆盖本仓库 README 或当前源码。未上传的权重需从原实验存储单独恢复。

## 开发

先阅读 [AGENTS.md](AGENTS.md)、[研究原则](.agents/skills/llm-memory-research/SKILL.md) 和 [CONTRIBUTING.md](CONTRIBUTING.md)。

- `src/llm_memory_editability/`：模型、数据生成、训练与分析实现。
- `scripts/`：执行、分析、审计、报告、追踪及材料下载入口。
- `configs/`：实验配置、来源清单及环境配置。
- `tests/`：数据划分、训练契约、恢复和评分检查。
- `docs/experimental-protocol.md`：部分历史脚本直接读取的实验协议。

```bash
python -m pytest
ruff check .
ruff format --check .
```

部分检查依赖 Hugging Face 中的材料和未上传权重；正式实验使用对应批次冻结环境。静态检查排除冻结源码快照，避免改动历史审计哈希。`memory-editability` 与 `memory-editability-neural` 为工程示例，其结果不作为正式研究证据。

本机 Python/CUDA 启动与独立 Docker 环境约定见 [AGENTS.md](AGENTS.md)。W&B 追踪按批次冻结配置执行；本次归档不改动运行中的训练或追踪设置。
