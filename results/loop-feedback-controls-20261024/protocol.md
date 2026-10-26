# 两组软反馈对照

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run（用户已授权共享GPU启动）
- Origin Date: 2026-10-24（America/Chicago）
- Verification Status: 81项相关CPU检查及ruff通过；未做CUDA试跑，正式训练未启动
- Version Label: feedback_controls_v1

用户要求新增两组：InfiLoop加DiscoLoop解码的soft-feedback embedding；invariant去掉该embedding回注，仍可计算γ。
随后明确授权：检查资源后选择两张现有四组使用的GPU共享启动，各150万步，不中断现有任务。
复用[正在运行的四臂](../loop-fourway-terminal-20261024/protocol.md)中的InfiLoop和invariant作为各自基线。
本文件是独立对照契约，不修改四臂冻结材料。当前规划登记在[Loop路线图](../../docs/loop-roadmap.md#feedback-controls)。

## 改变什么

为避免与旧代码把“聚合候选”也记为c_t混淆，下面把用户所指的DiscoLoop软反馈记为q_t，把聚合候选记为z_t：

\[
p_t=\mathrm{softmax}(\mathrm{LMHead}(H_t)),\qquad
q_t=\mathrm{RMSNorm}(E^\top p_t).
\]

LMHead保留原最终LayerNorm和绑定词嵌入，温度τ=1，反馈系数α=1；全词表软解码，不取argmax、不采样、不detach。
此定义沿用[DiscoLoop式4–6](https://arxiv.org/html/2607.00341v2#S4)。

| 新组 | 唯一改变 | 保留 |
| --- | --- | --- |
| `infiloop-feedback` | 聚合候选从z_t=H_t改成z_t=H_t+q_t | 固定随机h0、原始N0/D0、F(h+e)、内容分数、学习β、N/D递推；不增加整个候选的RMSNorm |
| `invariant-no-feedback` | 聚合候选从z_t=RMSNorm(H_t+q_t)改成z_t=RMSNorm(H_t) | h0=N0=e、D0=1、首轮F(e)、后续F(h+e)、候选RMSNorm、五个打分统计量、学习β、N/D递推 |

两组都保持各自的递推：

\[
w_t=\exp(s_t),\quad N_t=\beta N_{t-1}+w_tz_t,\quad
D_t=\beta D_{t-1}+w_t,\quad h_t=N_t/D_t,\quad \gamma_t=w_t/D_t.
\]

InfiLoop仍以可学习向量aᵀRMSNorm(z_t)打分，分数截断到[-8,8]后取exp。
invariant仍可从p_t计算熵、最大概率和相邻解码一致度，从z_t和h_(t−1)计算相对变化与余弦。
去掉的是p_t→Eᵀp_t→H_t的embedding回注支路，**不是解码概率参与打分的支路，也不是γ或历史合并**。
γ按新轨迹正常计算，不强制等于完整invariant的γ。实现中直接跳过Eᵀp_t投影，避免误加回状态。

两组的终点均读合并后的h_T。InfiLoop新增反馈参与每个有效轮次的候选，包括最终候选；
不能套用纯DiscoLoop“只在继续下一轮时加反馈”的终点规则，否则会同时改变合并定义。

## 历史核对与比较能回答什么

最接近的是[native四臂的content](../../docs/loop-history.md)及[旧InfiLoop评审](../../docs/reports/loop-native-handoff-review-20261020.md)：
content已有内容打分、软反馈和归一化历史合并，原世界150万步B两/三/四跳为15.28/0.28/0.14%。
但其embedding初态、整个候选RMSNorm、分段监督/反传与本次InfiLoop基线不同，因此不能替代该直接对照。
旧普通Loop与invariant也同时改变了输入注入、候选归一化和聚合，不能回答“保留invariant只删软反馈回注”。
本次是已有主线的两项直接组件对照，不把架构组合本身包装成新发现。

- InfiLoop新组减InfiLoop基线：回答在现有InfiLoop适配中，加入软反馈后的训练与组合表现如何变化。
- 完整invariant减无反馈组：回答在保留概率打分及历史合并时，embedding回注带来多少变化。

这是两项各自配对的比较，不把两种架构横向差异全部归因于打分器；初态与候选尺度仍不同。
原子掌握、A留出组合、B两/三/四跳同时报告；固定终点与完整训练曲线并列，正负结果均保留。

## 训练与评测

沿用[四臂冻结配方](../loop-fourway-terminal-20261024/protocol.md)：相同40,000条训练数据、43,000题评测池、
相同主干初值/样本顺序/深度种子，一个共享GPT2 block、宽768、12 heads、dropout0.1、batch512。
逐样本Poisson(8)越界重抽到1–16，整批掩码推进、终点有效token平均CE、全程反传。
不增加中间实体标签、轮次标签或辅助损失。

已授权配对预算为每组150万次更新，共额外300万次；lr1e-4，2000步warmup，150万步余弦时间轴，
AdamW/裁剪/参数EMA均沿用原值。与已有四组一起比较对应900万次计划更新，七条旧初始化的1050万步继续暂缓。
本协议冻结时正式更新尚未启动；后续实时状态以[运行交接](RUNNING.txt)与`runtime/launch.json`为准。
本次新授权允许与现有四组共享GPU；GPU4和8具有最大显存余量，分别安排InfiLoop+feedback和invariant-no-feedback。

主指标：参数EMA、R8的各池答案准确率；同步EMA R4/R16，终点raw/EMA R4/R8/R16/R32和逐轮曲线。
记录实际执行轮数、样本曝光、FLOPs估算、耗时及GPU；匹配更新数与曝光，不声称匹配计算量。
保存模型、Adam、EMA、数据/RNG状态、逐题预测与完整曲线，W&B使用zsqzz/llm-memory-editability。

## 实现与执行入口

[配置](../../configs/loop-feedback-controls-20261024.json)和[独立训练入口](../../scripts/run_loop_feedback_controls.py)
复用四臂的数据、训练、评测和追踪流程。入口检查配对字段，拒绝多余臂或改变数据/初值/训练/评价配方。
方法开关位于[聚合实现](../../src/llm_memory_editability/loop_native_handoff.py)与[InfiLoop实现](../../src/llm_memory_editability/loop_comparison.py)。
旧配置默认保留原行为；在训四臂使用既有容器内冻结源码。

正式执行需按项目规则核对实际资源及预约，独立LM Docker、固定镜像、create-copy-start、root所有0444/0555源码、
输出写入独立持久目录`/ossfs/workspace/llm-memory-editability/loop-feedback-controls-20261024`。
每组使用相同入口，在容器中传入`--config configs/loop-feedback-controls-20261024.json --base /output --arm <组名>`。
每任务2 CPU、16 GiB主存、CUDA allocator fraction 0.16；两张卡各保留至少40,000 MiB空余才入场，
覆盖既有/新增任务各自的allocator上限及整卡8 GiB余量。CUDA试跑要求测得峰值加6 GiB仍低于本任务上限。
先各做3次Poisson及3次最长输入/16轮临时更新，并检查EMA/R32推理；12次工程更新单列，不保存或复用临时权重。
入场前重复核对四个原容器的PID/启动时间/状态，禁止新增外部CUDA进程或预约变化；现有未变化的预约记入资源快照。
不恢复旧队列，不重训基线，不中断现有任务。

验证覆盖混合深度的输出/梯度、因果性、配对初值、原始InfiLoop Connection对应、
软反馈有无的实际计算路径、继续运行/逐轮评测/重载端点一致性。检查结果另存[validation.json](validation.json)。
