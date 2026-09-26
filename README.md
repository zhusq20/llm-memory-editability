# LLM Memory: Learning and Editability

**研究训练数据的组织方式如何塑造知识表征，并影响学习效率与后续可编辑性。**

核心问题：让模型现在更容易学会的知识组织，是否也让它以后更容易更新？允许共同改善、选择性获益、取舍和零效应。

理论与实验统一维护在 [实验协议 v2.4](docs/experimental-protocol.md)。当前主实验为第 6.5 节的 A/B/C 文档组织；历史 SA/AS 顺序实验单列。理论只保留三部分：

1. **知识形成：** 研究数据组织如何改变共享特征与成员差异；这是主要待证问题。
2. **学习效率：** 分析激活重叠与目标方向何时促进迁移、何时造成干扰。
3. **知识编辑：** 从真实 MLP 输出权重的条件性关系出发，分析共同修改、个别修改与旧知识保持。

简化模型负责推导；当前在真实 GELU 层上验证，SwiGLU LLM 验证属于后续计划。不展开通用响应矩阵、容量界、样本复杂度或最优课程理论。本轮固定事实呈现多重集、监督和预算，比较关联公司事实、同人属性聚合与打散文档，不与历史学习顺序交叉。

## 模型与数据

| 用途与状态 | 模型 | 数据 |
| --- | --- | --- |
| 当前符号组织实验，已完成 | 从头训练的 GPT-2 式自回归 Transformer：8 层、宽度 768、12 个注意力头、MLP 宽度 3072 | **bioS-Work-Organization-v1**：复用 bioS-Work 真值，2048 人、64 家公司、12,352 条基础事实；同一呈现多重集生成 A/B/C 三种七事实文档 |
| 后续 LLM 知识适应主验证，未运行 | **`Qwen/Qwen3-1.7B`**，计划关闭 thinking | **bioS-Work-NL-v1**：同一事实表的自然语言适配；训练组织与语言审计另行锁定 |
| 后续跨家族复现，未运行 | **`HuggingFaceTB/SmolLM2-1.7B-Instruct`** | 相同自然语言世界、配对目标与评价标准 |
| 后续真实知识外部验证，未运行 | 未经世界适应的 Qwen3-1.7B 与 SmolLM2-1.7B-Instruct | **RippleEdits**：作者发布的 4755 个案例，按旧知识替换与知识新增分别评价 |

A/B/C 三个组织条件在同一世界、同一初始化的配对块内从相同权重出发。后续 Qwen／SmolLM2 的 checkpoint、交互接口、通用回放和能力保持方案保留在协议中，尚未执行。

本轮只运行从头训练 Transformer 的受控符号文档学习与编辑。已有 LLM 的自然语言知识适应属于后续计划；当前结果不能代表完整预训练、后训练或强化学习流程的效果。

结果分开回答：**相同预算学会多少、达到同一知识标准花费多少、随后完成同一更新花费多少。** 保留完整曲线与未达标运行；固定预算比较和知识水平相近的条件分析分别报告。

更新实验对 A/B/C 施加完全相同的目标，比较一致更新与独立例外，当前采用完整 MLP／全参数编辑和终点公司均值干预。单层输出权重校准与分支干预属于历史 SA/AS 开发记录及后续扩展，不在本轮重跑。主评价使用完整答案自由生成，分别记录直接目标 E、必要传播 D 和旧知识保持 U；层内公式不替代行为结果。

## 当前状态

v2.4 的 A/B/C 数据已生成：两个世界、每组 2048 篇七事实文档，每轮 14,336 次事实呈现；各条件的真值、逐事实曝光和符号 token 预算一致，公司默认事实各重复 32 次。已实现文档训练、精确续训和审计，已完成 2 世界 × 2 初始化 × 3 条件的 12 条学习轨迹、96 个配对编辑和 12 个终点组织诊断（3840 个完整生成探针）。固定每模型 14,336 步；旧世界组合查询采用共同的独立训练流，因此测量已学组合的编辑传播。配置见 `configs/bios-organization-development-v1.json`。英文文档已从同一呈现清单渲染，尚未训练自然语言模型。当前批次结果单独存于 `results/bios-organization-dev-v1/`，不覆盖历史数据。

本轮 A/B 的首次观测 99% 达标平均成本均为 6272 步，C 为 9856 步；A 早期学习更快，但编辑优势随 MLP／全参数范围反转。96 个案例直接修改全部成功，48 个例外更新均未联合通过。84 个学习与 1056 个编辑检查点复核通过；85 项工程测试通过。结果、配对表与解释边界见 [A/B/C 组织实验结果](docs/organization-results-2026-09-26.md)，完成记录见 [校验清单](configs/bios-organization-completion-20260926.json)。

以下为已完成的 v2.3 顺序开发记录：

条件性学习与编辑关系已写入协议，学习顺序如何形成表征仍待推导。已实现 bioS-Work 符号数据与审计、等曝光课程、完整 8 层自回归模型、学习轨迹、配对编辑与初步组织干预。符号开发已完成 8/8 条学习轨迹、256 个 512 步编辑执行，以及 24 个 GPU 阶段/终点与 8 个 CPU 终点表征诊断；另有 8 个终点的 E/R 特征与真实编辑损失梯度诊断。例外更新的主要失败是默认/实际城市的传播混淆，局部保持损伤进一步限制联合达标；不同方法的课程效应并不一致。详细结果、重复配置与知识匹配口径见 [失败分析与表征诊断](docs/development-analysis-2026-09-26.md)，历史中断账本见 [9 月 25 日记录](docs/development-results-2026-09-25.md)。自然语言适配与人工语义审计、LLM 编辑和确认性实验仍待完成；尚无确认性研究结论。

官方 bioS 的[生成材料已公开](https://github.com/zhuzeyuan/PhysicsLM4/tree/211b28f2e9d453114ca5a1cbbb2d5a632ac60fcf/data-synthetic-pretrain/Capo-bioS-bioR)。本项目复用字段与表述材料，补充虚构公司映射、个人例外、QA、课程及配对编辑；不声称已取得原论文完整样本或复现其全部流程。来源、同类论文的数据选择及适配审计统一记录在协议 §4。

每个编辑单元修改 93 条基础事实，并评价 96 条必要派生查询。历史 SA/AS 的两个课程阶段各 2640 步；当前文档组织实验使用统一混合训练及完整轮次曝光匹配。旧数据计划与独立 composition 校准任务已移除，其他数据只保留选型依据。

当前 A/B/C 符号批次及控制审计已经完成。下一阶段需据此锁定自然语言组织适配与 Qwen 验证；确认矩阵、SmolLM2 复现与外部验证尚未运行。理论推导并行推进。

原有 NumPy 示例与小型 MLP／Transformer 分类器仍只用于工程检查。`bios_*` 模块支持符号开发；当前配置为 `configs/bios-organization-development-v1.json`，历史 SA/AS 配置为 `configs/bios-symbolic-development-v1.json`。旧研究备忘录和临时理论综述已合并删除，不并存多份计划。

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

现有示例入口为 `memory-editability` 和 `memory-editability-neural`。当前符号批次环境记录于 `configs/organization-environment-dev-v1.txt`，历史环境记录于 `configs/gpu-environment-dev-v1.txt`；上面的 CPU 安装不用于完整 GPU 实验，也不包含后训练 LLM 的环境。

符号开发复现入口（在项目根目录执行，GPU 编号按实际空闲设备指定）：

当前 A/B/C 批次：

```bash
.venv/bin/python scripts/prepare_bios_organization.py
.venv/bin/python scripts/run_bios_organization.py --gpus 4
.venv/bin/python scripts/summarize_bios_organization.py
```

历史 SA/AS 批次：

```bash
python scripts/prepare_bios.py
python scripts/run_bios_pipeline.py --gpus 6 7 8 9
python scripts/summarize_bios.py
```

绘图依赖可用 `pip install -e '.[analysis]'` 安装。当前产物逐步写入 `results/bios-organization-dev-v1/`，历史完整产物保存在 `results/bios-dev-v1/`，包括固定配置、逐查询预测、曝光清单、预定检查点及执行记录。当前开发采用固定中间层窗口和未做新答案 NLL 筛选的目标；不声称已完成超参数搜索、自然语言审计、AlphaEdit 或完整确认矩阵。矩阵运算 FLOPs 为估算值，服务器共享负载下的墙钟时间另列。

代码位于 `src/llm_memory_editability/`。开发与实验记录要求见 [CONTRIBUTING.md](CONTRIBUTING.md)。研究问题、执行决策和文献依据统一维护在实验协议中，不并存旧版计划。
