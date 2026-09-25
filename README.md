# LLM Memory Editability

**研究 LLM 如何组织知识，以及这种组织如何决定知识更新的成本。**

核心命题：共享表示应有利于一致的规律更新，却可能增加独立例外的更新成本；改变知识的内部组织、补充可独立控制的表示，应当系统地改变这种关系。

本项目的唯一现行实验计划是 [实验协议 v1](docs/experimental-protocol.md)。它包含文献依据、完整数据定义、配对约束、组织干预、编辑方法、统计分析与执行顺序。

## 模型与数据

| 用途 | 确定采用的模型 | 确定采用的数据 |
| --- | --- | --- |
| 主机制实验 | 从头训练的 GPT-2 式自回归 Transformer：8 层、宽度 768、12 个注意力头、MLP 宽度 3072 | 自建 **SharedOrg-v1**：4096 人、128 个组织、8 个默认属性、4 个个人属性，共 54,272 条基础事实 |
| 预训练 LLM 验证 | **`meta-llama/Meta-Llama-3-8B`，Base 版** | **SharedOrg-NL-v1**：同一事实表的英文自然语言版本，先学习旧世界，再编辑模型参数 |
| 机制测量校准 | 同一个 8 层 Transformer | **GrokkedTransformer composition**：使用作者的 2000 实体、200 关系生成器 |
| 真实知识外部验证 | 原始 Llama-3-8B Base | **RippleEdits**：作者发布的 4755 个案例，按旧知识替换与知识新增分别评价 |

主比较是 **内部组织方式 × 更新结构**：同一旧知识、同一模型容量、同一修改位置下，比较一致更新与独立例外；再用等参数预算的表示干预检验原因。编辑效果同时包括目标成功、必要传播、未改知识保持和自然语言改写泛化。

## 当前状态

实验协议已确定，正式的数据生成器、8 层语言模型训练、内部组织干预和 Llama 编辑流程待实现。协议中的配置是待执行的研究设计，没有对应的正式实验结果。

仓库已有 NumPy 解析示例与小型 MLP / Transformer 分类器，可用于基础训练和编辑流程的工程检查。它们的命令不会运行上述正式实验。旧实验计划及旧试跑报告已删除，不另建归档。

## 开发

需要 Python 3.10 或更新版本。以下是现有工程检查的 CPU 安装方式：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[neural,dev]'
python -m pytest
ruff check .
ruff format --check .
```

现有示例入口为 `memory-editability` 和 `memory-editability-neural`。正式实验的 GPU 环境与依赖将在实现时锁定版本；目前的安装命令不包含 Llama 训练环境。

代码位于 `src/llm_memory_editability/`。开发与实验记录要求见 [CONTRIBUTING.md](CONTRIBUTING.md)。用户提供的 [原始研究备忘录](docs/research-memo.md) 保留原文，作为问题来源；实际执行以新协议为准。
