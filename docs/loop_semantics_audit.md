# 完整 loop 与整轮更新门审计

> 历史语义审计：保留末轮读出更正及原批实际公式；下文方案排序不作当前待办。当前方向见[Loop路线图](loop-roadmap.md)，相关已做比较见[历史索引](loop-history.md)。

2026-10-17；对应 [用户方案](development-artifacts/gated-loop-depth-extrapolation-v1/user-proposal.md) 与[执行协议](development-artifacts/gated-loop-depth-extrapolation-v1/protocol.md)。

<a id="terminal-readout-correction-20261017"></a>
**2026-10-17 CDT语义更正与用户决定。** 用户要求遵循DiscoLoop的末轮直接读答案约定，重点比较论文学习门与轮间残差交接。此前用“保持前面计算一致”解释末轮也加反馈，混淆了答题状态与交接状态，该必要性判断撤回。原文、已冻结源码及已完成pilot不追溯修改；下文WholeLoopGateGPT公式仍描述原批实际执行，不代表修正已进入代码或新训练。

对于无整轮插值的原DiscoLoop，完整块输出为 $H_r=F_\theta(X_{r-1})$，本轮答案始终从 $H_r$ 读出；仅在继续下一轮时构造 $X_r=H_r+g_r q(H_r)$。相同权重、相同输入、确定性推理且不引入依赖总轮数的其它操作时，运行20轮的前5个块输出 $H_1,\ldots,H_5$ 与运行5轮一致。第5轮读答案不要求先加反馈。旧Q-Full从 $H_r+q(H_r)$ 再读答案，改变了读出函数与训练梯度；这属于额外架构变更，不能作为循环前缀一致性的必要修复，也不能将该批低分视为原DiscoLoop的同配方结果。

已重新核对[DiscoLoop v2 §4式(4)–(6)](https://arxiv.org/html/2607.00341v2#S4)：令 $v_r=\Phi(H_r)$、$q_r=\mathrm{RMSNorm}(v_r)$，论文学习门为每token标量 $g_r=\sigma(w^\top v_r+b)$，非末轮交接为 $X_r=H_r+g_r q_r$；门读取未归一化的软embedding，参数跨位置与轮次共享，末轮门为0。论文符号实验§5.1使用固定门1，不能把它的成绩直接当作学习门成绩。本项目GPT适配另保留原最终LayerNorm。现有[DualGateGPT的paper实现](../src/llm_memory_editability/dual_gate.py)对应这项门公式，继承原末轮关闭反馈逻辑；P的初始门.99是本项目开发选择，不能称论文给定初值。用户随后要求先按合理性排序不同方案，尚未选定新的公式、预算或任务。

**设计建议，非已验证性能排名。** 后续结合完整曲线的优先级修订见[已有结果与设计优化](roadmap.md#gate-evidence-refinement-20261017)；以下保留五种结构的定义。 记本轮输入为 $X$、完整Transformer输出为 $H=F_\theta(X)$，以下均仅描述非末轮交接，末轮直接读取 $H$。优先级为：(1) 原文再编码注入门 $H+gq$，明确控制再编码通道；(2) 连续更新门 $X+g(H-X)$，控制本轮相对输入的净变化，$g=0/1$分别对应保留输入/接受完整输出；(3) 独立组合 $X+g(H-X)+aq$，同时控制两类更新，但增加变量，应先理解前两项；(4) 完整连续状态门 $r\odot H+gq$，同时缩放已有状态和新增变化，适合检验连续通道的取舍，却不能解释成只抑制新增噪声；(5) 在输入上再加完整输出 $X+gH$，因 $H=X+\Delta$ 而成为 $(1+g)X+g\Delta$，重复累加输入，不优先推荐。第2项的 $H-X$ 是完整块的精确差值，不替换或简化内部attention/MLP/残差；门在完整块计算后作用，不节省该块计算，也不保证多轮动力学稳定。第3项与旧 $X+g(H+q-X)$ 不同：旧式同时将再编码系数约束为同一个 $g$。上述建议不等于用户已采纳，不修改旧冻结训练。

`grokking_reproduction.ReproductionGPT` 的一轮是 `transformer.h` 中所有物理层依次执行。每层保留 pre-LN、完整因果 SDPA、attention 输出投影与残差，再执行 pre-LN、GELU MLP 与残差。`SelfDecodeGPT._round` 复用同一计算。输入 token embedding 与可学习绝对位置 embedding 仅在第零轮相加；每轮没有新位置编码。读取时使用同一个可学习 `ln_f` 与输入 embedding 转置，没有独立输出权重。

旧 `SupervisedAlignmentGPT` 在非末轮应用 `H + RMSNorm(softmax(E LN_f(H)) E)`，末轮直接从 H 读出。RMSNorm 无 affine、epsilon=1e-6、温度1，无 detach。`dual_gate.py` 的 P/G/V 调整这个再编码分支，D 同时逐维重加权 H；它们都发生在非末轮的分支交接，不是保留上一轮输入的整轮插值。因此不能把旧 P/G/D/V 的训练结果当成本次 Fixed/Gate 对照。

已冻结的 `WholeLoopGateGPT` 批次每轮执行两层完整共享 Transformer：

\[
H_r=F_\theta(X_{r-1}),\quad C_r=H_r+\beta Q(H_r),\quad
X_r=X_{r-1}+\alpha_r(C_r-X_{r-1}),\quad \ell_r=E\operatorname{LN}_f(X_r).
\]

`Q` 复用上述全词表、可微分的软解码与再编码；所有轮次，包括第5轮与实际末轮，都执行同一个函数。token gate 输入为 H 的无 affine LayerNorm，参数只有共享的 w、b，共 d+1=257 个。它不接收答案、hop 标签或轮号。Fixed 与 Gate 都从 `sigmoid(float32(log(9)))` 开始，避免0.9的不同浮点计算导致初值不一致。Full 与人工 alpha=0/1 使用精确端点分支，避免减加消去误差。

本批关闭 dropout，不使用 KV 缓存、不 detach、不增加中间监督。`num_loops` 可覆盖推理次数；训练调用明确固定为5。beta=0 完全跳过反馈 softmax。计算所有 token 的所有层，没有 early exit。

CPU 契约检查原生 Full 的前向与全部主干梯度、Gate/Fixed 初始前向与主干梯度、端点复制、共享参数、因果前缀、5与32轮的前5个状态逐值相同、门梯度和状态不变。每个 GPU smoke 还在实际 BF16 环境重复匹配与32轮有限性检查，并由新容器重载验证。CPU 通过不替代尚待排队的 CUDA 检查。

文献核对：[DiscoLoop §4](https://arxiv.org/html/2607.00341v2#S4) 明确末轮反馈系数为零；本批改变了这一点，历史名称为 depth-consistent DiscoLoop-style feedback，该名称不表示原DiscoLoop的块输出前缀不一致。其原文门控制再编码注入，而本批门插值整轮候选与旧状态。[Loop, Think, & Generalize §3–4](https://arxiv.org/html/2604.07822#S3) 区分 ID、systematic 与 depth 泛化，并采用共享 GPT 层堆栈；本批沿用这种区分，额外固定输入槽位。[NDR §2.1](https://arxiv.org/html/2110.07732#S2.SS1) 的 copy gate 与本批保留旧状态的动机有关，但其逐维门、encoder 和 geometric attention 不在本批实现内。[STARS](https://arxiv.org/html/2605.26733) 的动力学视角提供稳定性参照，本批不加入它的正则或随机循环训练，不能称这些方法的完整复现。
