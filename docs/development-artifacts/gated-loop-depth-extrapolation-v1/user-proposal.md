# Research Proposal：Gated Whole-Loop Updates for Depth Extrapolation

> 版本：2026-10-17 · 状态：待执行实验方案（尚未接入服务器代码或运行训练）  
> 目标：**训练固定 5 loops，推理扩展到 10、20 loops**，判断整轮更新门是否改善循环稳定性及更深的参数化知识组合。  
> 原则：最小模型、严格对照、固定训练预算、可证伪的机制解释。

## 0. 核心结论与任务边界

我们要区分两种不同的 depth extrapolation：

1. **Loop-count robustness**：对训练深度以内的题目（例如 3–5-hop），5 loops 已经答对；增加到 10、20 loops 后，能否保持正确，避免 overthinking？
2. **Useful reasoning-depth extrapolation**：对训练时未见的更长组合（例如 8、10-hop），增加到 10、20 loops 能否产生新的正确答案？

**第一种成功不等于第二种成功。** 门可能只会冻结第 5 轮的错误答案。因此必须同时测量 correct→wrong 与 wrong→correct。

研究假说：

- **H1（稳定性）**：可学习更新门能抑制部分额外循环导致的状态退化。
- **H2（继续计算）**：在保持已完成结果的同时，门不会阻止更长组合继续推进。
- **H3（自适应必要性）**：收益超越固定更新步长，而非仅来自普遍减小残差。
- **H4（与 DiscoLoop 的关系）**：状态更新门和再编码分支可能解决不同瓶颈，存在互补性；也可能互相替代。

### 成功的最低标准

先要求在固定 5-loop 训练条件下，learned gate 相对**匹配初始状态的 fixed gate**，在额外循环后的稳定性上有跨 seed 的一致收益，且 5-loop ID 性能不显著下降。只有更长的未见组合也改善时，才能主张 *useful depth extrapolation*。

---

## 1. 方法：门只放在完整 loop 之后

设 \(X_{r-1}\) 是第 \(r\) 轮的输入，\(F_\theta\) 是由共享参数的完整 Transformer block（可能包含多层）组成的一轮：

\[
H_r=F_\theta(X_{r-1}),\qquad C_r=H_r+\beta Q(H_r).
\]

其中 \(\beta\in\{0,1\}\) 控制是否使用再编码：

\[
Q(H)=\operatorname{RMSNorm}\left(E^\top\operatorname{softmax}(E\operatorname{LN}_f(H))\right).
\]

每个 token 一个**标量**更新门，跨所有 loops 共享参数：

\[
\alpha_{r,i}=\sigma\left(w^\top\operatorname{LN}_0(H_{r,i})+b\right),
\]

\[
\boxed{X_{r,i}=X_{r-1,i}+\alpha_{r,i}(C_{r,i}-X_{r-1,i}).}
\]

- \(\alpha=1\)：完整接受本轮候选。
- \(\alpha\approx0\)：近似保留上一轮输入。
- \(0<\alpha<1\)：沿本轮候选方向走一部分。
- 门参数量 \(d+1\)；**不改变共享 Transformer 内部每一层**。
- 所有 token 仍计算完整 \(F_\theta\)，**不节省 FLOPs，不是实际 early exit**。

### 1.1 为什么这个门有理论动机，但不保证外推？

忽略 \(Q\)，并把 \(\alpha\) 设为常数：

\[
G(X)=(1-\alpha)X+\alpha F_\theta(X).
\]

在 \(F_\theta(X^\star)=X^\star\) 附近：

\[
J_G=(1-\alpha)I+\alpha J_F,\qquad
\lambda_G=1-\alpha+\alpha\lambda_F.
\]

例如 \(\lambda_F=-2,\alpha=0.5\Rightarrow\lambda_G=-0.5\)，可以抑制某些振荡；但 \(\lambda_F=2\Rightarrow\lambda_G=1+\alpha>1\)，仍不稳定。对输入相关门，还会出现与 \(\nabla\alpha\) 有关的额外 Jacobian 项。

所以**这是可检验的稳定性假说，不是稳定性定理**。它也不会凭空创造新事实或保证一轮完成一跳。

### 1.2 必须消除 terminal-loop mismatch

原 DiscoLoop 习惯在**最后一轮关闭再编码**。若训练 \(R=5\)，推理 \(R=20\)，第 5 轮在训练时是末轮、测试时却是中间轮，会出现另一种不一致。

为隔离纯粹的循环次数外推，**本项目主实验采用统一递归定义**：每一轮都执行同一候选和门控函数，并从该轮更新后的状态读出：

\[
X_r=\operatorname{Update}(X_{r-1}),\qquad
\ell_r=E\operatorname{LN}_f(X_r).
\]

训练只对 \(\ell_5\) 计算主损失；推理时读取 \(\ell_R\)。对任意 \(R\ge5\)，前 5 个状态必须与独立运行 \(R=5\) 完全一致。

**重要限定**：\(\beta=1\) 的这一版本是 *depth-consistent DiscoLoop-style feedback*，并非与原论文“末轮无反馈”的实现完全相同。原论文版单独作为文献参照，不与该版本直接做机制归因。

---

## 2. 主实验：六个配置，三种更新规则

在两个反馈条件下重复同样的三个门控对照：

| ID | 再编码 | 更新规则 | 主要用途 |
|---|---|---|---|
| N-Full | \(\beta=0\) | \(\alpha=1\) | 纯共享 Transformer |
| N-Fixed | \(\beta=0\) | \(\alpha=0.9\) | 固定阻尼 |
| N-Gate | \(\beta=0\) | token-wise learned \(\alpha\) | 纯整轮门收益 |
| Q-Full | \(\beta=1\) | \(\alpha=1\) | 统一递归的 DiscoLoop 风格基线 |
| Q-Fixed | \(\beta=1\) | \(\alpha=0.9\) | 再编码条件下的固定阻尼 |
| Q-Gate | \(\beta=1\) | token-wise learned \(\alpha\) | 门与再编码是否互补 |

**核心比较**：N-Gate vs N-Fixed；Q-Gate vs Q-Fixed。跨 \(\beta\) 比较时需报告再编码的额外计算成本。

learned gate 初始化：

\[
w=0,\qquad b=\log 9,
\]

所以初始 \(\alpha=0.9\)，与 Fixed **前向精确匹配**。它并不与 Full 的初始前向匹配；不能只比较 Gate 与 Full 后把全部收益归因于自适应。

**正式预算：6 配置 × 3 paired seeds = 18 次训练。** 所有方法复用相同主干初始化、数据顺序、训练 tokens、优化器和训练轮数。每个 seed 内先生成一个共享主干初始化，再复制给各配置。

### 2.1 后续有条件对照（不是第一轮全部运行）

只有主实验出现稳定信号后，按顺序补充：

| 对照 | 排除的解释 |
|---|---|
| Fixed \(\alpha=0.5,0.75,0.95\)（选小网格） | 只是 0.9 没有调好 |
| 输入相关的**全局**标量门（所有 token 同一 \(\alpha_r\)） | 不需要 token-wise 差异 |
| 按 loop 位置固定的系数（只在 1–5 有定义） | 仅学习训练轮次的步长表；**不作为 20-loop 可外推模型** |
| 训练时随机 \(R\in\{3,4,5\}\)，对各方法相同 | 训练深度多样性是否已解释收益 |
| 仅末轮 adaptive gate | 收益是否来自最后的 readout 混合 |
| 忠实的“末轮关 \(Q\)”原论文基线 | 量化 terminal-policy mismatch 的影响 |
| 原 P/G/D/V 中最强者 | 对照此前的跨轮表示方案，不能把它们称为 adaptive-depth baseline |
| 仅测试时做多轮 logits 平均（无训练改动） | 退化是否主要来自选择末轮读出，而不是状态更新 |
| 另训固定 10-loop 的相同主干（可选上界） | 若 5→20 外推失败，延长训练深度本身能否解决；**不计入固定 5-loop 主比较** |
| 与已有 ACT/early-exit 方法比较（后期） | 区分状态轨迹改进与动态分配计算量；报告实际 FLOPs |

随机训练深度实验中，采样的任务跳数 \(k\) 应与可执行循环预算相容；不能强迫 3 loops 完成一个需要更多步骤的任务而把失败算成门的问题。计算预算要同时按 optimizer steps 与累计 loop-token FLOPs 报告。


---

## 3. 数据集与数据划分

### 3.1 优先复用现有多跳任务生成器

优先使用当前项目已有的 DiscoLoop / Looped Transformer 数据和 tokenizer；不要为了这个实验先重构模型。要求能控制：
- 原子事实训练；
- 组合训练的最大 hop 数；
- 组合是否出现在训练中；
- 测试 hop 数；
- 随机 seed 和实体划分。

推荐两个实体不重叠的图 A / B，共享关系 token、使用不同的事实映射 \(f_r\)。

| 数据划分 | 原子事实是否训练 | 组合是否训练 | 测试 hop | 检验目标 |
|---|---|---|---|---|
| A-short | 是 | A 上训练 \(k\le5\) | 1–5 | ID 与 loop-count robustness |
| A-long | 是 | 只训练 \(k\le5\) | 6、8、10、12（可选 16、20） | **主推理深度外推** |
| B-short | 是 | B 上不训练组合 | 1–5 | systematic composition |
| B-long | 是 | B 上不训练组合 | 6、8、10、12 | 联合泛化压力测试 |

**关键**：A-long 是同一事实库上的未见长组合，能较清楚地测量推理深度外推；B-long 同时增加“未见组合”和“更长链”两种难度，不应独立作为主要机制证据。

### 3.2 没有现成数据时的后备 pilot

- 图 A、B 各 256 个实体，4 个关系；每个关系为独立的随机置换，避免 many-to-one 退化。
- 所有 A/B 原子映射 \((e,r)\to e'\) 进入训练；训练组合只来自 A。
- A 训练组合长度 \(k\in\{1,2,3,4,5\}\)，按长度均衡采样。
- 验证集从 A 的训练长度内未见 query 采样，不能与训练 query 重复。
- A-long / B-long 从未见的长关系序列采样，记录每条 query 的全部中间实体，用于诊断。
- 目标输出只有最终实体 token；不暴露 CoT 或中间结果。
- 保持输出实体频率尽可能均衡，避免用实体先验猜答案。

**输入长度混杂**：长 hop 通常意味着更长 prompt。建议用固定的 20 个关系槽位，使用 `NOOP` / identity 关系填充，训练时随机放置 NOOP；测试时同样保持 20 槽位。但长链仍改变 NOOP 比例，故必须加一个控制集：在固定 hop 数下改变 NOOP 数量/位置，确认模型不是只对填充模式失效。

**Hop 数不等于 loop 数**：先用 pilot 验证训练 \(R=5\) 时 \(k\le5\) 能被学会。若 5-hop 训练准确率始终较低，先调整主干容量、任务生成或训练配方；不要把基础任务不可学误判为 gate 无效。

### 3.3 数据文件结构

```text
data/
  facts_train.jsonl
  comp_A_train_k1to5.jsonl
  comp_A_val_k1to5.jsonl
  eval_A_short.jsonl
  eval_A_long.jsonl
  eval_B_short.jsonl
  eval_B_long.jsonl
  metadata.json
```

每条记录至少保存：
`example_id, graph_id, start_entity, relation_ids, hop_count, answer_entity, intermediate_entities, split, rng_seed`。

检查：
- [ ] 图 A/B 实体不交叉；原子事实完整覆盖。
- [ ] B 没有组合监督；A-long 没有 \(k>5\) 的训练监督。
- [ ] Train/Val/Test 组合 query 不重叠；各组答案分布平衡。
- [ ] `NOOP` 控制集存在；不同数据集的 tokenization 一致。
- [ ] 记录所有中间实体，以便进行 teacher-forced / oracle bridge 诊断。

---

## 4. 训练协议

### 4.1 统一训练设定

- **所有主实验训练时固定 \(R_{\mathrm{train}}=5\)**，不随机循环次数，不用 6–20 轮的监督。
- 训练时仅从第 5 轮状态读出答案并计算主 CE loss。
- 不添加中间轮 CE、显式停止标签、额外稳定性正则或 Jacobian 正则；这些会改变主假说。
- Backbone：优先沿用当前项目的共享 Transformer、初始化、预训练语料/合成任务及 optimizer。
- 若需新 pilot：共享 block 可先用 2–4 个 Transformer layers、hidden size 256–512；这些只是工程起点，不是经验证的最优配置。
- 每组训练至同样的固定 token/step 预算；学习率、batch、weight decay、warmup、clip、precision 全部一致。
- 固定 checkpoint 选择规则：仅使用 **A-short 的 \(R=5\) validation** 选择模型；不能用 A-long、B-long 或 \(R=20\) 测试曲线选 checkpoint。
- 每组至少 3 个配对 seeds；有稳定正结果后扩展到 5 seeds。

### 4.2 训练前短跑

先各配置做一次短跑（6 次短跑；不计入正式统计），确认：
1. loss 能下降、单跳事实可学习；
2. \(R=5\) 的短链准确率有增长；
3. gate 初始均值为 0.9、梯度非零、未饱和；
4. 20/32 loops 评估没有数值异常；
5. q-on 的内存与计算成本在预算内。

**如果模型训练 5 loops 根本学不会 3–5-hop，先解决基础任务，不扩展正式矩阵。**

### 4.3 参考配置（需要映射到实际代码）

```yaml
experiment: gated_loop_depth_extrapolation
train_loops: 5
eval_loops: [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32]
train_hops: [1, 2, 3, 4, 5]
eval_hops: [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20]
feedback_beta: [0, 1]
gate_mode: [full, fixed, token]
fixed_alpha: 0.9
gate_init_alpha: 0.9
gate_input: post_loop_hidden
gate_granularity: token_scalar
gate_shared_across_loops: true
terminal_special_case: false
loss: final_loop_ce_only
seeds: [11, 22, 33]
checkpoint_selection: A_short_val_at_R5
```

### 4.4 参考前向伪代码

```python
def looped_forward(x, loops, feedback_beta, gate_mode):
    logits_at_r, alpha_at_r = [], []
    for r in range(loops):
        h = shared_transformer(x)
        if feedback_beta:
            p = softmax(tied_head(final_norm(h)), dim=-1)
            q = rms_norm(p @ tied_embedding.weight)
        else:
            q = zeros_like(h)
        candidate = h + feedback_beta * q

        if gate_mode == "full":
            alpha = 1.0
        elif gate_mode == "fixed":
            alpha = 0.9
        else:
            alpha = sigmoid(gate_linear(layer_norm_no_affine(h)))
        x = x + alpha * (candidate - x)
        logits_at_r.append(tied_head(final_norm(x)))
        alpha_at_r.append(alpha)
    return logits_at_r, alpha_at_r
```

初始化：`gate_linear.weight = 0`，`gate_linear.bias = log(9)`。Gate 输出应为 `[batch, seq, 1]`，对 hidden 维广播。实现应适配当前模型是否使用 RoPE、RMSNorm affine、embedding tying 等细节。

---

## 5. 评估矩阵：两个轴都必须扫描

训练结束后，**不更新任何参数**，对每个 checkpoint 进行：

- 测试循环数：\(R\in\{1,2,3,4,5,6,8,10,12,16,20,24,32\}\)。
- 任务 hop 数：\(k\in\{1,2,3,4,5,6,8,10,12\}\)，探索性加入 16/20。
- 数据组：A-short、A-long、B-short、B-long。

在 `eval()` 模式下，一次执行到 \(R_{\max}\) 并在各轮读取 logits；再用独立的 \(R=5\) 前向核对结果完全一致。不能通过截断不同 checkpoint 的轨迹来替代同一个模型的循环数扫描。

### 5.1 主要指标：稳定性

对 \(k\le5\) 的固定测试集，定义：

\[
\mathrm{Retain}(R)
=P(\hat y_R=y\mid \hat y_5=y),\quad R>5,
\]

\[
\mathrm{Recover}(R)
=P(\hat y_R=y\mid \hat y_5\ne y).
\]

另记录 `correct→wrong`、`wrong→correct`、`wrong→wrong` 的转移比例。

预先确定的主稳健性汇总：

\[
\mathrm{RobustScore}
=\frac{1}{6}\sum_{R\in\{6,8,10,12,16,20\}}
\mathrm{Acc}(k\le5,R).
\]

注意不同模型第 5 轮正确样本集合可能不同。Retain/Recover 必须同时报告，并在**所有模型共同于 \(R=5\) 答对**的交集子集上做辅助比较，避免条件样本差异造成误导。

### 5.2 主要指标：真正的推理深度外推

对 A-long 的 \(k=6,8,10,12\)：

- 完整画 \(\mathrm{Acc}(k,R)\) 的二维热力图。
- 报告 \(R=5,10,20\) 的准确率，尤其是 \(k=8,10\)。
- 报告 \(\mathrm{Acc}(k,20)-\mathrm{Acc}(k,5)\)，看**增加循环是否产生新能力**。
- 计算达到预设准确率门槛（如 80%）的最大 \(k\)，但只在达到门槛的离散 \(k\) 上报告，不做插值。
- 在各模型共同掌握全部相关原子事实的题目子集上重复评估。

**禁止**只报告每个模型在测试集上挑出来的最佳 \(R\) 并将其称为实际可部署性能。可以把 `oracle best R` 单独标为诊断上界；真实方法需预先指定 \(R=10/20\)，或用仅 ID validation 调好的停止规则。

### 5.3 必须报告的控制指标

- \(R=5\) ID accuracy / loss / margin（不能用更高 R 掩盖基础性能下降）。
- 原子事实准确率、B-short systematic accuracy。
- 不同 NOOP 分布下的固定 \(k\) 准确率。
- 训练 FLOPs、推理 FLOPs、显存、wall-clock、额外 softmax 成本。
- 各 seed 的 mean±SD，以及**同 seed 的 paired difference**。
- 评估集 bootstrap 置信区间（不能替代训练 seed 方差）。


---

## 6. 机制诊断：要证明门在做什么，而不只是涨分

### 6.1 每轮记录

对固定诊断子集（ID short、A-long、B-short，各取相同题目）记录：

| 量 | 目的 |
|---|---|
| \(\operatorname{mean}_i\alpha_{r,i}\)、分位数、位置热力图 | 门是否随循环、位置、任务而变化 |
| \(\|X_r-X_{r-1}\|/\|X_{r-1}\|\) | 实际更新大小；不能只看门值 |
| \(\|C_r-X_{r-1}\|\) | 主干提出的候选变化有多大 |
| \(\operatorname{RMS}(X_r)\)、max abs | 排查状态范数爆炸或衰减 |
| \(\cos(X_r,X_{r-1})\) | 状态方向变化 |
| 目标答案 logit margin、entropy | 正确答案是变稳还是逐渐被挤掉 |
| 每轮正确/错误标志 | 计算何时完成、何时翻转 |
| 各 hop 的中间实体可解码性（可选） | 检查是否真的推进了更长组合 |

注意：\(\alpha\) 接近零不代表模型已经答对；它也可能**冻结错误状态**。需同时看 `Recover(R)` 与 `Retain(R)`。

### 6.2 三个最有信息量的干预

1. **固定门替换**：在已经训练好的 learned-gate 模型上，把门替换为该模型在验证集的全局平均值、每轮平均值或每位置平均值；测试是否需要输入相关性。
2. **门交换**：同一 batch 内打乱不同样本的门值，保留门值分布，检查门是否与当前输入状态相关。
3. **中间轮读取**：在不改变轨迹的情况下，读取 \(R=1\ldots20\) 的 logits，分开统计“首次正确轮次”和“正确后首次退化轮次”。

门交换可能制造分布外状态，因此它只是辅助因果证据；需要和固定门对照一起解释。

### 6.3 局部动力学（有正结果后再做）

在小批量状态上，用 JVP / VJP 估计更新映射 \(G\) 的局部扰动增益：

\[
g_{\mathrm{local}}(X,v)=\frac{\|J_G(X)v\|}{\|v\|}.
\]

可以使用随机方向及有限次幂迭代的代理量，比较 \(r=4,5,8,12,20\)。不要把有限次 JVP 的范数直接当成精确谱半径，也不要由局部小增益推断全局收敛。

**最重要的对应关系**：若门改善 overthinking，但 `Recover` 没有提高，应明确定位为稳定性收益；若长链 `Recover` 增加，才有继续计算的证据。

---

## 7. 最小实验执行顺序与交付物

### Phase 0：仓库审计（实施前）

**目标**：确认 loop 的定义、原末轮行为、共享权重、输出头、训练 loss。

待确认的实际文件/函数：
- 完整 `shared_transformer_loop` 对应哪些 Transformer layers？
- 原始 `dual_gate.py` 的 \(P/G/D/V\) 插入位置；
- \(Q\) 的 `LN_f`、embedding tying、RMSNorm 参数与是否 detach；
- 原代码是否只在最后一轮算 CE、是否关闭末轮 \(Q\)；
- 推理接口是否支持 `num_loops` 覆盖训练值；
- dropout / cache / positional encoding 是否随循环改变。

**产出**：`docs/loop_semantics_audit.md`，含旧版与新 recurrence 的精确公式及最小前向一致性测试。

### Phase 1：实现与单元测试

建议新增模块（文件名仅为建议，按实际仓库调整）：

```text
src/.../whole_loop_gate.py
configs/depth_extrapolation/{N-Full,N-Fixed,N-Gate,Q-Full,Q-Fixed,Q-Gate}.yaml
scripts/train_depth_extrapolation.py
scripts/eval_loop_sweep.py
scripts/analyze_loop_dynamics.py
tests/test_whole_loop_gate.py
```

必须通过：
- [ ] `N-Full` 与无门共享 Transformer 的前向一致。
- [ ] `N-Gate/Q-Gate` 初始化与相应 `Fixed` 前向一致。
- [ ] 人工设 \(\alpha=0\) 时复制旧状态；设为 1 时复制候选。
- [ ] 共享参数没有意外复制成 5 套；20 loops 仍复用同一模块。
- [ ] 相同输入的 20-loop 前 5 轮状态，与独立 5-loop 完全一致。
- [ ] 门参数梯度非零、所有张量维度正确。
- [ ] eval 不使用训练标签、不会意外更新参数或状态。
- [ ] `beta=0` 不执行无意义的 full-vocab softmax。
- [ ] 5/10/20/32 loops 数值有限且显存可接受。

### Phase 2：数据生成与 baseline pilot

- [ ] 生成原子事实、A-short、A-long、B-short、B-long、NOOP 控制集。
- [ ] 保存生成 seed、关系表、query 去重报告、答案频率报告。
- [ ] 用 N-Full、Q-Full 先确认固定 5-loop 训练能够学会 short 任务。
- [ ] 运行 6 配置各一个短跑；排查训练崩溃、门饱和和末轮差异。
- [ ] 固定正式预算、checkpoint 选择规则和测试集，之后不再按 OOD 结果调整。

**Go/No-Go**：若短链 ID 始终不可学，先解决任务可学性；若基础模型已学会，但 R>5 没有任何退化、A-long 也不受 R 影响，应考虑这套 benchmark 是否足够暴露深度问题。

### Phase 3：正式训练

- [ ] 3 个配对 seeds × 6 配置，共 18 次训练。
- [ ] 固定 `R_train=5`，仅第 5 轮 CE；不使用测试长链训练。
- [ ] 保存训练曲线、模型 checkpoint、optimizer、RNG、配置、代码 commit。
- [ ] 仅以 A-short \(R=5\) validation 选 checkpoint。
- [ ] 每个 checkpoint 完成统一 \(R\) × \(k\) × split 评估。

建议结果文件：

```text
results/
  runs.csv                   # config, seed, checkpoint, train budget
  accuracy_by_hop_loop.csv   # config, seed, split, k, R, correct, total
  flip_transitions.csv       # 5->R correct/wrong transition
  dynamics_by_loop.csv       # alpha, norms, margins, entropy
  compute_cost.csv           # train/inference FLOPs and latency
  figures/
    accuracy_heatmap.pdf
    accuracy_vs_loops.pdf
    retention_recovery.pdf
    gate_and_state_dynamics.pdf
```

### Phase 4：分析与针对性补充

先画 4 张图：

1. **同一 \(k\le5\) 任务：准确率 vs 推理 loops**（稳定性）。
2. **不同 \(k\)：准确率热力图 \(k\times R\)**（真正的深度外推）。
3. **Retain/Recover vs loops**（冻结错误 vs 继续计算）。
4. **门值与实际状态变化 vs loops**（机制）。

根据结果选择补充实验，不一次性全跑：

- Gate≈Fixed：扫描固定 \(\alpha\) 的小网格，判断是否只需要数值阻尼。
- Gate>Fixed 但 token 门差异不明显：比较全局输入相关标量门。
- 只在 Q 条件有效：检查再编码分支的幅度、末轮定义与错误反馈。
- ID 更稳定但 A-long 不改善：定位为 overthinking 缓解，不宣称 10-hop 泛化。
- A-long 改善但状态不稳定：研究是否存在“继续计算与保留”的权衡。
- 只有较大 R 才有收益：追加 \(R=24,32\) 压力测试，确认收益窗口而不是任意外推。

---

## 8. 预注册的主要比较与结果解释

| 观察结果 | 支持什么 | **不**支持什么 |
|---|---|---|
| Gate > Fixed，short 在 R=20 保持更好 | 输入相关门改善 loop robustness | 自动获得更深推理 |
| Gate ≈ Fixed，二者 > Full | 固定阻尼就能解释收益 | token-wise 自适应必要 |
| Gate 在 \(k=8,10\) 的 R=20 显著更好 | useful depth extrapolation | 门必然通过稳定固定点实现 |
| Gate 保留正确，但 Recover 降低 | 抑制 overthinking，可能冻结错误 | 更强计算能力 |
| Q-Full > N-Full，但 Q-Gate≈Q-Fixed | 再编码有用，门的额外价值不足 | 门和再编码互补 |
| Q-Gate > Q-Fixed 且 N-Gate > N-Fixed | 门可能跨不同接口起作用 | 两者完全独立 |
| 仅末轮 adaptive 就能复现收益 | 可能主要是最终读出混合 | 中间轮状态保留机制 |
| ID \(R=5\) 显著下降 | 门可能损害原始训练目标 | “长循环更稳”可单独证明方法更好 |

**统计规则**：主结果至少报告 3 seeds 的均值、标准差与 paired seed 差；对正结果尽可能补到 5 seeds。不能只挑最优 seed、最优测试 R 或最优 hop。负结果限定为“当前任务、模型与预算下未获支持”，不作普遍否定。

### 明确的继续/停止标准

**继续作为研究主线**，需同时满足：
1. learned gate 相对 fixed 0.9 的收益跨 seeds 一致；
2. \(R=5\) 的基本组合能力没有实质性下降；
3. short 的 overthinking 或 long 的有效外推至少有一项改善；
4. 追加固定步长/全局门对照后，仍存在可解释的额外收益。

**收窄为稳定性工作**：仅 H1 成立、H2 不成立。  
**回退到简单阻尼**：Fixed≈Gate。  
**暂停门控方向**：训练预算内没有稳定收益，或所有收益都来自更改末轮读出约定。

---

## 9. 论文叙事：避免与 Early Exit、NDR、STARS 重合

**不应主张**“首次发现不同 token 需要不同计算量”，也不应把 whole-loop gate 称为新的 Early Exit。NDR 已研究 copy gate，STARS 已研究循环稳定性，相关 looped LM 已研究推理时增加循环与 overthinking。

**更合理的论文问题**：

> *Can a recurrent language model trained at a fixed loop depth learn an update rule that both preserves completed computations and continues useful computation when recurrence is extended far beyond training?*

需要用三组证据支持：
1. 固定 5-loop 训练，10/20-loop 测试的严格协议；
2. 固定阻尼、token 自适应、再编码有无的对照；
3. Retain 与 Recover、以及长链组合准确率的分离。

**如果只有稳定性提升**，结论就限定为 *robustness to excess recurrence*；如果长链也提升，才能进一步研究可扩展计算规则。

---

## 10. 相关工作与直接可用的参考资源

1. **DiscoLoop: Looping Discrete Embeddings and Continuous Hidden States for Multi-hop Reasoning** (2026)：https://arxiv.org/abs/2607.00341  
   用于再编码接口的文献基线；注意原实现的末轮反馈约定。
2. **Loop, Think, & Generalize: Implicit Reasoning in Recurrent-Depth Transformers** (2026)：https://arxiv.org/abs/2604.07822  
   参数化知识的系统泛化、深度外推和 overthinking；提供可借鉴的数据与评估协议。代码：https://github.com/OSU-NLP-Group/Loop-Think-Generalize
3. **Stabilizing Recurrent Dynamics for Test-Time Scalable Latent Reasoning in Looped Language Models (STARS)** (2026)：https://arxiv.org/abs/2605.26733  
   稳定性分析、Jacobian 正则、随机循环训练；是重要的机制与训练方法参照。
4. **The Neural Data Router: Adaptive Control Flow in Transformers Improves Systematic Generalization** (2022)：https://arxiv.org/abs/2110.07732  
   Copy gate 与组合任务上的深度控制；用于明确本方法与已有门控结构的区别。

---

## 11. 最终交付检查表

- [ ] 代码审计记录与原始 recurrence 的精确定义。
- [ ] 6 个可复现配置、统一 loop 接口、单元测试通过。
- [ ] 固定训练 \(R=5\)，没有使用 \(R>5\) 的训练监督。
- [ ] 原子事实/组合数据划分与去重、NOOP 控制验证。
- [ ] 18 次正式训练与完整的 seed、checkpoint、配置记录。
- [ ] 5→10/20 的稳定性、8/10-hop 的有效外推**分别**报告。
- [ ] Fixed gate 与 learned gate 的配对比较。
- [ ] `Retain/Recover` 与状态动力学诊断。
- [ ] 按结果决定是否追加固定步长网格、全局门、随机训练深度等。
- [ ] 结论严格对应证据，不以“稳定”替代“推理能力外推”。

**一句话版本**：用一个最小 whole-loop gate 测试“固定 5-loop 训练是否可以学到可安全延长的更新规则”；用 *保持正确* 与 *继续算对更长链* 两套指标，把稳定性和真正的深度外推区分开。
