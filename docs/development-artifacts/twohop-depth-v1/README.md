# 两跳深度与形成实验

**已完成并审计：28条轨迹、196评价节点。** 结果解释见[实验报告](../../twohop-depth-results-v1.md)。

执行前设计集中于[计划§14.28](../../hebbian-learning-plan-v1.md#1428-两跳推理的深度与形成实验2026-09-29执行前登记)。本目录的`lock.json`封存数据、采样流、代码、环境和参数量；`launch.json`记录GPU和进程；`execution-status.json`记录已完成轨迹与节点。

- `learning.csv`：逐世界、初始化、架构及训练节点的完整答案准确率、NLL、自主两次调用和计算量。
- `interventions.csv`：逐运行、节点、层及干预条件的反事实答案变化、原答案保持、错误关系率、条件子集分母和logit lens读出。
- `summary.json`：主预算4096与加长预算8192的架构汇总，包括每条轨迹的留出组合分数。
- `depth-accuracy.png/pdf`：深度与单跳、两跳、两次调用准确率。
- `bridge-formation.png/pdf`：仅替换第一跳关系位置的中间状态后，反事实答案增益随训练的变化。
- `audit.json`：全部端点重新加载、逐例预测复核、来源/结果文件哈希及原值/末层干预控制。
- `supplemental-audit.json`：196节点相同批次的前缀替换等价性、64/512分批的数值差异记录，以及6个训练准确率回退终点的CPU FP32评价。此为事后复核，不改变训练或主评分。
- `tests.log`、`preflight.json`：8项契约测试以及独立工程世界的GPU速度、显存和精确续跑验证。
- `world-means.csv`、`paired-differences.csv`：两个固定预算下的世界均值，以及同世界/初始化的6层相对浅层差值。
- `donor-query-strata.csv`：按替换来源对应的新两跳问题是否在训练中出现，拆分干预计数。该拆分是看到部分结果后增加的描述分析，不用于选模型或训练参数。
- `heldout-depth-width.png/pdf`、`bridge-readout-and-use.png/pdf`：留出组合的深度/宽度对照，以及6层模型中桥接读出与干预作用的训练曲线。`descriptive-analysis.json`记录事后分析范围和脚本/审计哈希。

所有图和汇总在运行期间可更新；只有`audit.json`中`complete=true`且`runs=28`、`nodes=196`时才表示整个批次已完成。原始数据位于`data/twohop-depth-v1/`；逐例预测、干预数组、模型和优化器位于`results/twohop-depth-v1/`，不纳入版本控制。

复现命令（项目根目录）：

```bash
PYTHONPATH=src .venv/bin/python scripts/run_twohop_depth.py prepare
.venv/bin/python -m pytest tests/test_twohop_depth.py -q
PYTHONPATH=src .venv/bin/python scripts/run_twohop_depth.py launch --gpus 2 4 5
PYTHONPATH=src .venv/bin/python scripts/run_twohop_depth.py status
PYTHONPATH=src .venv/bin/python scripts/report_twohop_depth.py aggregate
CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH=src .venv/bin/python scripts/report_twohop_depth.py audit --device cuda:5
PYTHONPATH=src .venv/bin/python scripts/analyze_twohop_depth.py
CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH=src .venv/bin/python scripts/audit_twohop_depth_extra.py --device cuda:5
```

已准备的批次拒绝重新prepare。启动器使用封存代码；同轨迹通过文件锁防止重复运行，完成轨迹自动跳过。共享GPU环境的实际耗时不能单独作为架构效率结论。

工程预飞的恢复权重保存在`results/twohop-depth-v1/preflight/resume.pt`；其世界不进入科学汇总。事后分析脚本要求28条轨迹及196节点完成审计之后才能运行。
