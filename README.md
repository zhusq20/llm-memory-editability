# LLM Memory: Learning and Editability

研究参数中的知识存储与多跳计算原理。当前关注原子事实的存储表征、可组合存储的容量与稳定性；Loop用于操纵计算深度。本轮直接使用完整标准Transformer/Loop作受控训练；历史简化架构证据与完整预训练模型复核各保留其适用范围。

## 当前状态

- **标准Transformer/Loop事实负载实验已完成。** 完整GPT-2形式，标准1/2/3层及共享1层循环2/3次，宽256、4头、MLP1024，全部参数训练。24条开发、30条新世界确认及2条事后诊断，56端点重载通过。1024/6144事实、测试必要事实曝光匹配；标准1层新组合32.66%/22.10%，Loop×2为40.66%/32.27%，Loop×3为43.33%/37.00%。单跳接近100%，严格无组合角色池仍0%–1.76%；标准2层一个终点发生训练退化，完整保留，不作容量证据。见[完整结果与边界](docs/results.md#standard-storage-composition)、[逐世界图](docs/development-artifacts/storage-composition-v1/confirmation/per-world.png)和[完成清单](docs/development-artifacts/storage-composition-v1/completion-manifest.json)。

- **完整Qwen3-0.6B快速开发已完成。** 三臂共同独立复习后，分开/无关联拼接/关联拼接的两单跳均100%，未训练两跳为6.25%/4.69%/15.63%，自主两次调用均100%。六个更新/原事实复习分支及九端点重载完成：新单跳全部学会，直接两跳更新收益依条件变化，未回放事实保持明显下降。一个世界、一个种子，尚非新世界确认或唯一MLP机制。见[完整结果](docs/results.md#small-lm-composition-v2)、[图](docs/development-artifacts/small-lm-composition-v2/comparison.png)与[研究计划](docs/hebbian-learning-plan-v1.md#next-experiment)。

- **简化架构的机制验证：固定调用过程后的新知识组合已完成。** 开发校准后，三个新世界×两个初始化×两臂的12项正式比较及全部重载核验通过。普通交叉熵/加入统一输出表示约束，原知识两跳99.46%/99.97%；冻结注意力并换入只接受单跳训练的新事实MLP后，单跳均100%、两跳66.39%/94.66%，六配对均改善。它支持其架构条件下的知识复用收益，是继续检验真实LM的依据，不直接等同于真实LM结论。见[结果](docs/results.md#memory-reuse)、[主图](docs/development-artifacts/memory-reuse-v1/confirmation/report/comparison.png)与[完成清单](docs/development-artifacts/memory-reuse-v1/completion-manifest.json)。

- **组合练习结构与跨事实迁移实验已完成。** 3条开发、三个新世界×两个初始化×两臂的12条正式训练，15端点和99节点审计通过。逐事实组合使用次数、唯一训练链数及token边际匹配；广搭配相对受限搭配，熟悉事实新链15.11%→66.30%，严格独立目标1.57%→2.16%，单跳与自主两次调用均100%。主要收益是局部搭配泛化，跨事实调用仍很弱。见[结果](docs/results.md#text-structure)、[学习曲线](docs/development-artifacts/text-structure-v1/confirmation/comparison.png)与[完成核验](docs/development-artifacts/text-structure-v1/confirmation/verified-completion.json)。

- **损失来源与参数变化实验已完成。** 4条开发、24条冻结后续续训及72个参数互换状态均完成。原子＋背景组合训练与仅原子训练的目标组合为84.59%/43.10%，两者单跳100%；四臂显示强交互。后者换入中性对照分支的MLP后恢复至70.96%，单跳仍100%，注意力也有部分贡献。既有三个世界的结果支持部分训练归因，尚非唯一内部机制。见[完整结果](docs/results.md#text-loss-source)、[参数互换图](docs/development-artifacts/text-loss-source-v1/followup/report/weights.png)与[完成清单](docs/development-artifacts/text-loss-source-v1/completion-manifest.json)。

- **文本预训练归因比较已完成。** 3条开发、18条新世界确认训练及36个节点内部干预全部完成。独立事实/联合上下文/再加入组合结果的未见组合正确率为69.16%/82.88%/99.84%，全部单跳100%；P0/P1严格匹配文本及监督，但P1收益在第三个世界反向。局部状态可交换，但单点转向未随能力单调增强。见[结果与边界](docs/results.md#text-pretraining)、[图](docs/development-artifacts/text-pretrain-v1/report/comparison.png)与[完成清单](docs/development-artifacts/text-pretrain-v1/completion-manifest.json)。

- **bioS同权重诊断已完成。** 18端点、6,912条生成复核通过；MP任务权重常用问法下，直接月份奇偶76.56%，自主提取再判断100%。预训练主体张量在适配后完全保持，原生续写下降不能称为主体记忆被擦除。见[同权重结果](docs/results.md#bios-same-weight)。

- **循环监督与长循环检查已完成。** 2条开发与12条后续续训、18个模型的R1…64检查完成并通过审计。多终点相对单终点的OOD差异在R4/R8/R16为+16.98/+1.25/−12.26个百分点，未建立普遍稳健性改善。原始状态持续移动且幅度增长，归一化变化变小不能直接解释为不动点收敛。见[结果与限制](docs/results.md#grok-loop-supervision)、[曲线](docs/development-artifacts/grok-loop-supervision-v1/followup-report/comparison.png)和[完成清单](docs/development-artifacts/grok-loop-supervision-v1/completion-manifest.json)。

- **事实使用经历的四臂开发及三个新世界确认已完成。** 共16条128k训练、320个节点，直接回答与自主调用重载审计全部通过。三个确认世界中，同一事实组接受组合训练、对应原子重复、仅基础单跳训练后的未见组合正确率为100.00%/25.01%/3.01%；主比较涉及的原子事实全部答对。见[完整结果](docs/results.md#grok-usage)、[形成曲线](docs/development-artifacts/grok-usage-v1/report-confirmation/role-aligned-learning-confirmation.png)与[完成清单](docs/development-artifacts/grok-usage-v1/completion-manifest.json)。

- **Loop Transformer 对照实验已完成。** 15条开发、6条敏感性及90条正式训练，111个终点GPU审计、210个机制评价和42个循环扫描全部完成。三个正式世界中，循环模型两/三/四跳的ID留出组合均接近100%；两跳L2的纯OOD组合为6.29%，三/四跳仍约1%。见[完整结果](docs/results.md#grok-loop)、[主汇总](docs/development-artifacts/grok-loop-v1/main-report/summary.json)和[完成清单](docs/development-artifacts/grok-loop-v1/completion-manifest.json)。
上述历史两到四跳小Transformer只对最终答案与EOS计算交叉熵；新增文本批次从随机初始化做完整下一词元训练。

- **连续评分复盘与两跳同桥供体实验已完成。** 111条轨迹、2,997节点的连续评分未显示大幅模式切换，低OOD表现也不是EOS失分。在图定义的共同供体子集上，L1/L2使用ID首跳完整状态的正确率为40.58%/38.94%，使用OOD供体为8.06%/12.33%；三个世界差异均为正。该子集只覆盖3.11%–8.16%的OOD题目，本次只评价35个既有端点、零新增训练。见[复盘](docs/results.md#grok-loop-continuous)与[供体结果](docs/results.md#grok-loop-same-bridge)。
- **历史配方的深度比较已完成。** 两跳2/3/4层留出准确率为96.44%/98.40%/98.29%；三跳独立世界的3/4层均值为90.29%/97.04%。该配方的四跳6层尚未学会大部分留出组合，区别于新Loop批次的开发结果。见[完整结果](docs/results.md#depth-hop-extension)与[八页对比图册](docs/development-artifacts/hop-depth-comparison-v1/complete-comparison.pdf)。
- **历史首跳状态置换已完成。** 两层模型第一块的r1位置换入纯首跳供体后，89.31%的答案正确转向新路径，h位置对照为0.13%。该历史批次未拆分注意力与MLP；Loop批次已单独完成子层比较。见[历史机制结果](docs/results.md#grok-depth-bridge)与[Loop机制报告](docs/development-artifacts/grok-loop-v1/mechanism-report/summary.json)。
- **原版bioS复盘、18端点抽样重载与36个预训练权重评价已完成。** 570万预测复算一致；规范六属性QA的S/M/MP为5.07%/35.65%/99.96%，留出问法仍有缺口。固定人物中，S的日期和出生地原生续写均100%，之后QA仅14.06%/1.56%，因此弱QA不能简单解释为没有记住事实。日期比较的训练题能答对，新日期对即使给真值仍失败，规则泛化前提尚未建立。见[完整结果](docs/results.md#bios-original)与[形成曲线](docs/development-artifacts/bios-original-trajectory-v1/native-attribute-curves.png)。100k矩阵与日期更新未执行。

**当前受控训练证据支持：事实可被单跳提取，并不保证它能用于未见组合；组合使用经历带来原子重复无法补足的收益。** 既有状态置换进一步支持首跳信息的下游可用性差异。历史使用经历比较采用一个初始化；新增文本比较采用三个世界、两个初始化，仍未定位唯一内部机制，也不直接建立自然语言预训练定律。

当前主线是[参数存储与多跳计算原理](docs/hebbian-learning-plan-v1.md#frontier-question-20261001)。既有组合经历、记忆替换和循环计算结果提供工具，尚未建立可组合存储的容量规律；原子掌握、有效操作、目标曝光和参数预算分别核查。知识更新开发保留为历史支线，旧低组合分数或长循环单跳退化不直接解释为存储能力上限。

## 文档入口

| 文档 | 内容 |
| --- | --- |
| [当前计划与理论依据](docs/hebbian-learning-plan-v1.md) | 训练目标、下一轮实验、理论与待办 |
| [研究结果汇总](docs/results.md) | 各批次最终状态、核心数字、适用范围及产物 |
| [实验协议](docs/experimental-protocol.md) | 统一评价与记录规范、脚本依赖的历史契约 |
| [冻结模型两跳报告](docs/twohop-frozen-results-v1.md) | Qwen两跳实测、精度复核和复现命令 |
| [开发指南](CONTRIBUTING.md) | 安装、检查、数据与文档维护规则 |

## 开发与复现

Python ≥3.10。仓库中的超大结果JSON使用Git LFS保存；克隆后运行 `git lfs install` 和 `git lfs pull` 获取完整文件。

CPU工程检查环境：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[neural,dev]'
python -m pytest
ruff check .
ruff format --check .
```

绘图依赖：`python -m pip install -e '.[analysis]'`。原版bioS的专用依赖见 [pyproject.toml](pyproject.toml) 的 `bios-original`；正式GPU实验使用各批冻结环境和配置，CPU环境不代替实验环境。

完整测试还需要相关批次的本地 `data/`、`results/` 与固定模型tokenizer，按下述复现入口准备；这些大体积产物默认不随仓库提交。

只读查看原版bioS状态：

```bash
.venv/bin/python scripts/run_bios_original.py status
```

完整运行命令与前提见[当前计划](docs/hebbian-learning-plan-v1.md#reproduction)及[两跳报告](docs/twohop-frozen-results-v1.md)。旧批次的精确复现使用对应源码快照、配置和冻结契约。

## 目录

- `src/llm_memory_editability/`：模型、数据与实验实现；`scripts/`：运行、分析、审计入口；`tests/`：契约检查。
- `configs/`：批次配置、来源清单和环境记录。
- `docs/development-artifacts/`：冻结设计、源码快照、小型汇总、图表和审计证据，保持原始内容。
- `data/`、`results/`、`checkpoints/`：本地数据、模型和大体积运行产物，默认不纳入版本控制。

`memory-editability` 与 `memory-editability-neural` 是工程示例入口，其结果不作为正式研究证据。
