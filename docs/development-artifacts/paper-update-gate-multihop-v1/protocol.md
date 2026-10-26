# 论文注入门加整轮更新门：代码与比较协议

2026-10-17 用户先授权编写模型、训练和评测代码，并要求每项修改对应一个从原方案出发的设计问题；随后明确授权直接训练，跳过启动前的短训及重载检查。当前安排为本地两条 $u_0=0.9$、远端两条 $u_0=0.95$，各 300k 步。配置分别为 [本地 u0.9](../../../configs/paper-update-gate-multihop-v1.json) 和 [远端 u0.95](../../../configs/paper-update-gate-multihop-u95-v1.json)。原 P 的训练、权重、源码和审计记录保持原样；实际提交与运行状态以各批控制器记录为准，本段授权不代替启动证据。

## 问题与最小比较

已有论文门 P 在单个开发图的图 B 三跳达到 37.20%，四跳仍低。这里检验两个问题：减小连续状态的整轮更新是否有帮助；由当前状态决定更新量，是否比固定减小更新更有帮助。尚不把任何收益解释成噪声删除、收敛保证或计算节省。

令本轮输入为 $X$，完整 GPT 块组输出为 $H=F_\theta(X)$，保留实际 attention、MLP、残差和归一化。沿用 P 的原通道：

$$
v(H)=\operatorname{softmax}(\operatorname{Readout}(H))E,\qquad
q(H)=\operatorname{RMSNorm}_{\epsilon=10^{-6}}(v(H)),
$$

$$
g_P(H)=\sigma(w_P^\top v(H)+b_P),\qquad
X_{\mathrm{next}}=X+u(H-X)+g_P(H)q(H).
$$

`Readout` 沿用最终 LayerNorm 和绑定词表输出头，温度为 1。$g_P$ 的输入仍是未归一化的软词向量，参数仍从原 $w_P=0,b_P=\log(99)$ 开始学习。它没有被固定，也没有改成旧 G/D/V 的参数化。论文方法的来源及原实现差异见 [旧协议](../dual-gate-multihop-v1/protocol.md) 和 [DiscoLoop §4](https://arxiv.org/html/2607.00341v2#S4)。

| 条件 | 对原 P 的唯一设计改变 | 要区分的问题 |
|---|---|---|
| P，复用 | $u=1$，完整沿用已有 P | 原方法的参照结果 |
| `fixed-update`，新增 | $u$ 固定为该批的 $u_0$ | 仅减小连续更新是否有帮助 |
| `token-update`，新增 | $u(H)=\sigma(w_u^\top\operatorname{LN}_0(H)+b_u)$，从同样的 $u_0$ 开始 | 状态相关的更新是否优于固定缩放 |

新增门采用每个 token 一个标量，参数跨轮和位置共享；$\operatorname{LN}_0$ 无仿射参数，eps 沿用原 GPT 的 $10^{-5}$。初始化为 $w_u=0,b_u=\log(u_0/(1-u_0))$，不消耗初始化随机数。本地 $u_0=0.9$，远端 $u_0=0.95$；每批的固定/学习臂使用相同初值。固定臂保留相同张量和前向算术，但作为不可训练 buffer；学习臂将这两个张量设为参数，宽 768 时只增加 769 个可训练参数。固定/学习臂的初始前向和共同参数原始梯度可严格匹配；新增参数参与全局梯度裁剪，不能据此声称两臂的优化器更新或后续轨迹相同。

初稿中的 0.99 未启动科学训练，已由用户本次指定的 0.9/0.95 取代。这两个值尚不是实测最优值；有限 sigmoid 不能精确等于 1，所以新臂不声称与原 P 初始前向相同。`full` 模式直接调用原 P，用于实现一致性检查和旧权重读取，不在新增训练臂中。没有添加去掉 $q$、逐维门或额外外层残差。0.9/0.95 分配在不同主机上，跨批初值差异与运行环境差异共同变化；主要受控比较是每批内部的学习门减去固定门，不能把两批之间的差异只归因于门初值。

模型在数值实现中先调用一次原 P 通道，再加 $(1-u)(X-H)$，即计算 $[H+g_Pq]+(1-u)(X-H)$；它在实数代数上等于上式，保留原通道的运算顺序。不会为了计算更新门再次解码。**只在非末轮交接时执行这两个门；末轮直接从完整块输出 $H$ 读答案。** 一轮推理因此没有轮间通道，末轮的 Transformer 更新也不被新增门缩放。

## 复用与未改变的配方

P 复用 `dual-gate-multihop-v1/attempts/preflight-repair-v2/runs/paper`。配置明确记录原目录、原冻结配置、11 份产物哈希和 12 份未修改核心源码哈希。只读检查涵盖第 0 步及 300k 权重、`run/complete/audit/grid/grid-audit/exposure/learning/canonical-result/cloud-audit`，以及原配方、模型身份、全部历史节点和独立重载覆盖。P 的原 300k 训练、R1–4 评测和云端验证结果全部复用，不重新训练或覆盖它们；旧 `complete.json` 中未回填的 `independent_reload=pending` 不能代替独立 `audit.json` 与 `grid-audit.json` 的通过结果。

新代码在独立模块中实现，仅适配模型构造和参数/成本记录。原 `dual_gate.py`、`run_dual_gate.py` 和原生训练、数据、评测、模型、成本及追踪实现等 12 份源码保持哈希一致。沿用同一个采样器、最终答案/EOS 损失、AdamW 分组与更新、warmup、裁剪、AMP/CUDA graph、评分和 W&B 写入实现；新增可学习门归入已有门参数的零衰减组。没有新增辅助损失、真值中间标签、KV、轮数或数据支持量。

数据仍是两个各 500 实体、10 关系的已观察置换图：所有 10k 原子事实，加图 A 各 10k 条两/三/四跳组合，共 40k 训练查询。world SHA256 为 `e132eaf6bba5786bbf05f6f36c1b7383f698e93ee0dee313f568f6d50a77026f`，panels SHA256 为 `c991cebeeeec05639be87d981cd20031016f47965c43fed8bf5cc55579c3b468`。词表 1013，pad=0、EOS=2，序列长度 6；输入验证同时检查这些元信息，避免数据字节相同而模型解释不同。

模型仍为宽 768、12 头、一个独立 GPT 层循环四轮，dropout 0.1；初始化 seed 2026101602，样本流 seed 2026101603，dropout seed 2026101604。训练仍为 batch 512、AdamW lr $10^{-4}$、wd 0.1、warmup 2000、300k 更新。学习曲线节点为 0/1k/8k/16k/32k/64k/128k/256k/300k，不以早期多跳分数筛掉方案，不自动延长训练。

**每批两条新增臂各 300k，共 600k 更新；本地和远端两批计划合计 1200k 更新。** 原 P 新增更新为零。用户本次要求直接开始 300k，因此不运行预检更新、32 步工程短训及其重载，也不运行原 P 的新环境 CUDA 重放。历史已做过的测试与审计仍保留，不能改写成本次替代环境通过了相同检查。

## 评测与可解释范围

主指标保持固定 300k、推理四轮、图 B 三跳的全词表答案准确率。训练过程保留两图的原子、两/三/四跳留出组合、图 A 训练组合探测，共 11 个面板、43k 查询；保留答案加生成 EOS、NLL、原子前提覆盖、条件组合和无重复实体子集。终点后只计算同一检查点的完整 R1–4 推理网格，不挑最好轮数。网格评测为读取已训练权重所需的一次评价，不另外执行独立重载审计或重复重放。没有五跳训练或五跳评测。

报告比较固定门减去 P、学习门减去固定门、学习门减去 P；新臂曝光数组必须与旧 P 逐元素一致。图/初始化仍各一个，查询、关系模板、轮数和检查点不能当作独立世界。训练数据量、监督 token、参数、矩阵 FLOPs 与实际耗时分别报告；新增门的归一化和逐点运算单列，矩阵 FLOPs 不含这些操作。所有完整 Transformer 块照常执行，不宣称节省计算或自适应早停。

门在 0–1 内并不保证多轮稳定。若学习门获益，本比较支持状态相关更新对端到端行为有帮助；要归因于具体信息保留或特定回路，还需独立证据。固定门较好也不证明每个位置都需要同样更新量。

## 手动阶段与检查

训练/评测入口是 [run_paper_update_gate.py](../../../scripts/run_paper_update_gate.py)。每次明确指定新批输出目录，已有旧产物保持只读。本次启动按用户新授权直接进入正式训练；下表保留各可用阶段的作用，但工程阶段不作为此次启动前置条件。

```bash
PYTHONPATH=src python scripts/run_paper_update_gate.py verify-baseline \
  --config configs/paper-update-gate-multihop-v1.json --base /path/to/new-output
```

其余阶段使用同样的 `--config`、`--base`；两条臂分别为 `--arm fixed-update` 和 `--arm token-update`：

| 阶段 | 作用与产物 |
|---|---|
| `verify-baseline` | 只读核对旧 P 与数据，向新目录写复用收据，不需要 `--arm` |
| `preflight` | 实际 CUDA/BF16 工程预检；本次跳过 |
| `smoke` | 独立 32 步工程训练；本次跳过 |
| `smoke-audit` | 工程终点重载；本次跳过 |
| `smoke-grid` | 工程终点 R1–4 网格；本次跳过 |
| `smoke-grid-audit` | 工程网格重放；本次跳过 |
| `check-engineering` | 工程收据汇总；本次不要求此收据 |
| `train` | 按本次明确授权直接开始新臂 300k 科学训练 |
| `audit` | 可用的正式终点独立重载工具；本次不自动执行 |
| `grid` | 正式终点 R1–4 网格 |
| `grid-audit` | 可用的正式网格重放工具；本次不自动执行 |
| `cloud` | 可用的 W&B canonical 上传、下载与身份核对；本次不自动执行 |

W&B 默认团队 `zhusq20`、项目 `llm-memory-editability`，训练时按原实现在线记录，新组名来自各批配置。凭据继续使用运行环境的私有凭据文件；是否已创建运行以实际 `tracking.json` 和控制器记录为准。本次自动流程为 `train → grid`，新臂独立重载状态明确记为未运行；在线训练记录不能称为终点 canonical 云端核验通过。

汇总入口 [report_paper_update_gate.py](../../../scripts/report_paper_update_gate.py) 要求三个条件已通过本地审计，因而本次直接训练流程不自动调用它，也不伪造缺失审计。该工具留作未来明确执行独立审计后的汇总，写 `comparison.json` 和保留全部面板/轮数的 `comparison.csv`，不做模型评测或网络发布，不覆盖已有同名报告：

```bash
PYTHONPATH=src python scripts/report_paper_update_gate.py \
  --config configs/paper-update-gate-multihop-v1.json --base /path/to/new-output
```

原 P 的 LM 固定镜像为 `sha256:9643a7d6aeda1b4c9ac2ac50c2d84474b082d649a87e423bc6e0bdc0994097ee`，记录在 `baseline.source_image`，必须与旧配置及 freeze 一致。当前实际镜像记录在 `runtime.image`；替代镜像必须保留 `runtime.amendment` 的原因、原/实际 image 及 `requires_baseline_replay=false`。本次依用户明确要求跳过新环境重放，只核对历史文件身份和已有证据；主要依赖版本相同也不等于运行环境数值等价，不能声称已验证跨环境优化轨迹一致。

执行显式使用 `docker --context lm-memory`，核对实际空闲资源并隔离容器，不抢占其他任务。主机特定动态库修复按项目入口的适用机器条件执行，不移植到容器中。
