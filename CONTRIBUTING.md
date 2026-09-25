# 参与开发

使用 Python 3.10 或更新版本，在独立环境安装神经实验与开发依赖：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[neural,dev]'
python -m pytest
ruff check .
ruff format --check .
```

提交前运行上述检查。若需统一格式，运行 `ruff format .` 后重新检查。测试应验证配对约束、参数冻结、训练与编辑行为、结果记录或输入边界，避免只复述实现。

研究主线是 MLP / Transformer 的知识学习与编辑。先参照 [实验计划](docs/experiment-plan.md) 和 [非线性实验协议](docs/nonlinear-protocol.md) 明确假设、控制变量、预先可计算的结构量、质量阈值及停止条件。低秩模型只保留为解析基线，其秩约束不构成神经网络容量结论。

实验报告应记录代码版本、配置、环境、随机种子、实际训练与参数预算、各次运行和失败案例。分别报告表示成本、编辑成本与副作用，区分已匹配因素和仅测量的混杂因素。数据具有共享规律不意味着内部共享已获验证，有限训练失败也不能被称为不可表示。

生成结果放入 `results/`，不要提交环境目录、缓存、模型权重、访问令牌或未经授权的数据。如需保存可审查的正式结果，请另附小型文本报告并说明来源与复现命令。

`docs/research-memo.md` 是原始材料存档。研究进展、修正和已核验文献请另写文档，并区分原始假设、已实现功能与实验所得证据。
