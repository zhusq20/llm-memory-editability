# LLM Memory Editability

**在 Transformer / MLP 中研究：知识更新与已有规律的关系，如何影响编辑成功、保留知识与训练成本？**

本项目基于 [研究备忘录](docs/research-memo.md)。当前实现为从头训练的 PyTorch MLP 与小型因果 Transformer：先学习合成事实，再从同一训练检查点接受规律更新或实体级例外更新。NumPy 低秩示例保留为解析基线。

当前任务使用离散实体、属性和答案，不使用预训练 LLM 或自然语言数据。数据生成过程具有共享规律，但模型是否学到了相应的内部共享表示，仍需独立验证；标签矩阵的秩不提供神经网络容量下界。

## 快速开始

需要 Python 3.10 或更高版本。以下安装适用于 CPU 实验：

```bash
git clone https://github.com/zhusq20/llm-memory-editability.git
cd llm-memory-editability
python -m venv .venv
source .venv/bin/activate
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[neural,dev]'

memory-editability-neural --model mlp --seeds 0 1 2 --output results/mlp.json
memory-editability-neural --model transformer --seeds 0 1 2 --output results/transformer.json
```

等价入口为 `python -m llm_memory_editability.neural`。使用 `--help` 查看预算与模型配置。当前实验在 CPU 上运行；报告保留逐种子结果，生成结果默认不进入版本控制。

## 实验内容

| 环节 | 实现 |
| --- | --- |
| 事实世界 | 默认 24 个实体、8 个属性、3 个实体组、4 类答案；旧答案由组与属性共同决定 |
| MLP | 实体与属性嵌入，经 GELU 非线性前馈网络预测答案 |
| Transformer | 输入 `[entity, attribute, QUERY]`，经因果自注意力与前馈层，在查询位置预测答案 |
| 配对更新 | 在同一事实集合上比较规律与例外；匹配编辑条数、实体和属性覆盖、目标答案边际分布 |
| 编辑 | 复制同一个已训练检查点，采用相同步数和保留事实回放权重；默认只训练前馈层 |
| 从头训练 | 相同模型容量，两种目标从相同初始权重出发，训练全部参数拟合完整更新世界 |
| 测量 | 编辑与保留准确率、交叉熵、答案 margin、编辑前目标惊讶度、实际参数量、步数与用时 |

默认更新涉及第 0 组的全部 64 条事实，另外 128 条保留。规律更新统一改变该组的答案规律；例外更新为各实体分配平衡的目标映射，打破组内共同映射。两种条件修改同一批事实，每条目标答案都与旧答案不同。

默认先训练旧世界 400 步，再编辑 100 步；从头训练对照为 400 步。完整配置与解释见 [非线性实验协议](docs/nonlinear-protocol.md)。可用 `--edit-scope all` 运行全参数编辑对照。

当前控制仍有边界：目标惊讶度仅测量、未匹配；未测量自然语言改写泛化，也未干预内部共享程度。实验结果不能单独确认研究假设。

[首轮实验记录](docs/pilot-results.md) 覆盖两个模型、3 个初始化种子和同一个数据种子。它观察到有限训练预算内的编辑速度差异；相同容量从头训练均成功拟合目标，因此不能解释为例外目标不可表示，也不能视为 H1 已获确认。

## 研究路线

| 实验 | 状态与下一步 |
| --- | --- |
| A：固定编辑数量、改变更新结构 | 已实现 MLP / Transformer 配对训练与编辑；还需匹配惊讶度、扩大更新集合并验证预先定义的预测量 |
| B：固定知识内容、改变内部组织 | 待实现；需要对共享表示做可验证的干预 |
| C：等预算补充独立表示 | 待实现；需要匹配实际新增参数与优化预算 |
| D：相同容量从头训练 | 已实现有限预算诊断；训练失败不构成不可表示的证明 |
| 预训练 LLM | 待开展；需要自然语言查询、答案评价和更广的保留知识检查 |

完整假设和停止条件见 [实验计划](docs/experiment-plan.md)。编辑失败可能来自优化路径、参数冻结或信息约束，不能直接解释成模型容量不足。

## 解析基线

```bash
memory-editability --rank 1 --output results/linear-baseline.json
```

该入口运行 NumPy 秩 / SVD 示例，比较 2×2 数值矩阵的规律更新和单条例外，计算固定秩预算下的最佳重建误差。两种更新的编辑条数不同；它只说明该线性模型的表示限制。

## 开发与验证

```bash
python -m pytest
ruff check .
ruff format --check .
```

贡献和实验记录规范见 [CONTRIBUTING.md](CONTRIBUTING.md)。代码位于 `src/llm_memory_editability/`，协议与原始材料位于 `docs/`。原始备忘录中的文献条目尚未独立核验，不构成已完成的文献综述或新颖性结论。
