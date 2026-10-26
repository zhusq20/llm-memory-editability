# 位置不变性开发批协议（loop-position-invariance-development-v1）

**状态（2026-10-20）：用户授权训练。** 用户原文："把这套校正写进docs/loop-roadmap.md，并为第1项准备协议和代码.授权训练"。依据见[Loop路线图§1、§3](../../loop-roadmap.md#thesis)。2026-10-20 20:49 UTC在sn4622129202提交（控制器pid 535151）：GPU 6/7立即开始，`disco-rope-old`在GPU 8排队。恢复信息在§6。

## 1. 问题

**工作假说：** 在循环模型里，第k跳发生在第k个token位置。学到的绝对位置编码把事实查找绑定在训练时的位置上，只学过原子形式（主语在0、关系在1）的图B事实因此不能在后面的跳位上使用；第5个及以后的位置还从未训练过。

本批检验两个问题：
1. **位置：** 把位置编码换成只依赖相对偏移的RoPE后，图B三、四跳的落差和"答案在第5位及以后全部失败"是否消失？
2. **交接与配方：** ema臂的"一轮一跳"来自凸组合交接，还是来自新训练配方？

## 2. 设计

| 臂 | 交接 | 位置 | 配方 | 对照（复用） |
| --- | --- | --- | --- | --- |
| `ema-rope` | 凸组合，分数恒为0 | RoPE | 新：随机轮数两段深监督、余弦学习率、β₂=.95 | `loop-native-handoff-development-v1`的ema（绝对位置、新配方） |
| `disco-rope-old` | 加性DiscoLoop | RoPE | 旧：固定4轮单段、warmup后恒定学习率、β₂=.999 | A0（绝对位置、旧配方） |
| `ema-absolute-old` | 凸组合，分数恒为0 | 绝对 | 旧 | ema（只差配方）与A0（只差交接） |

**共同设定（与复用对照完全相同）：**
- **数据：** 两图置换世界，world/panels哈希`e132eaf6…`/`c991cebe…`；
- **模型：** 单个共享GPT-2块，宽768，12头，dropout 0.1，绑定词嵌入，DiscoLoop解码—再编码；
- **训练：** 批512，lr 1e-4，wd 0.1，梯度裁剪1.0，150万步；
- **监督：** 只有答案与EOS的交叉熵；
- **配对：** 初始化、样本流、dropout与轮数序列的种子都相同，第0步主干与A0逐位相同（`39a48782…`），入口会检查。

**RoPE的实现（`src/llm_memory_editability/loop_position_invariance.py`）：**
- 输入只用词嵌入；学到的位置参数仍留在状态字典里，但不参与前向，所以第0步主干与复用对照一致。
- 每层的查询与键做NeoX式旋转，覆盖整个头维度，底数为10000，在float32中计算。
- 其余计算完全不变。

**EMA：** 旧配方臂也维护EMA影子权重，只用于评测，不影响训练。

## 3. 评价

- **训练节点与终点网格：** 与[Loop原生交接批](../loop-native-handoff-development-v1/protocol.md#4-评价)相同，即21个节点评测EMA的R4/R8与原始权重的R4，终点做R1–32逐轮读出。
- **5–8跳：** 训练后用冻结的[5–8跳评测](../../../results/loop-beyond-four-hops-20261020/protocol.json)，同一组题、同一种子2026102101，测本批三臂。RoPE臂需要在评测脚本中加入相应模型构造，这是读出上的工程扩展，不改题目与判读规则。
- **主指标：** 150万步EMA权重下的以下各项，不选最佳节点或轮数：
  - 图B两/三/四跳、图A留出四跳在R8的答案准确率；
  - 各跳题首次达到终点准确率的轮次；
  - 5–8跳在R16的准确率与逐轮曲线。

## 4. 预测与判读（看结果前固定）

- **位置：**
  - 若绝对位置把查找绑定在训练位置，`ema-rope`的图B三/四跳在R8各比ema（54.1/10.8%）高至少15个百分点，`disco-rope-old`的图B四跳高于A0的0%。
  - 若两者都不升，图B差距不是位置效应。
- **多算一跳：**
  - 若相对位置下每轮算子与跳位无关，`ema-rope`的k跳题（k=5..8）在R16高于5%，并从第k轮起出现。
  - 有RoPE仍失败，说明每轮算子不能延伸到训练跳数以外。
- **交接与配方：**
  - 若凸组合本身诱导一轮一跳，`ema-absolute-old`的图A留出四跳接近100%，且k跳在第k轮形成。
  - 若它像加性臂那样在第1轮背出训练长链，新配方是必要的。

单图、单初始化，只作开发证据；不设统一的掌握率门槛。

## 5. 预算

3臂×150万步＝450万科学更新；另有每臂200步GPU工程冒烟，不保留。新世界训练0。

## 6. 执行

- **启动：** `scripts/launch_loop_position_invariance.py`在`lm-memory`上为每臂建一个容器。
  - 每臂等自己的GPU没有活动CUDA进程后才启动，所以可以排在正在收尾的任务之后；
  - 仓库只读挂载，持久区只读，只有批次输出目录可写；
  - W&B凭据只在训练阶段复制进容器，记录到`zsqzz/llm-memory-editability`；
  - 阶段顺序为smoke→train→grid，失败的阶段不重试。
- **输出目录：** `/mnt/disk2_from_server2/siqizhu4/lm-memory-persistent/workspace/llm-memory-editability/loop-position-invariance-development-v1/`。
- **恢复：** 先读其中的`controller-process.json`与`controller-state.json`，再核对LM Docker进程；不重复提交。
- **GPU：** 沿用用户已授权给Loop方向的GPU 6–9。`ema-rope`与`ema-absolute-old`在GPU 6/7；`disco-rope-old`排在GPU 8，等content臂收尾与其5–8跳评测结束后启动。

## 7. 代码与测试

- **代码：** 模型`src/llm_memory_editability/loop_position_invariance.py`；运行`scripts/run_loop_position_invariance.py`（复用`run_loop_native_handoff.py`的训练与评测函数，只替换模型构造并按臂套用配方）；启动`scripts/launch_loop_position_invariance.py`；配置`configs/loop-position-invariance-development-v1.json`。
- **测试：** `tests/test_loop_position_invariance.py`，16项，与原有30项一起在固定镜像中以CPU通过。覆盖：
  - 绝对位置版本与原模型的前向和全部梯度逐位相同；
  - RoPE版本主干与原模型相同，且不消耗随机数；
  - RoPE输出与整体位置平移无关；
  - RoPE不读取位置参数（无梯度，改动后输出不变）；
  - 旋转保持范数，内积只依赖相对偏移；
  - 逐轮读出等于从头执行；
  - 配置与各臂配方，旧配方学习率与A0逐步一致。

## 8. 限制

- 一个已观察开发图、一个初始化。
- RoPE同时去掉了绝对位置并改变注意力计算，所得效应是"本实现的相对位置"的效应，不能拆到单一性质。
- 解码—再编码通道依赖本模板"下一词就是中间实体"。
- 旧配方臂在eager模式下运行，数学上与A0的CUDA graph相同，墙钟不可比。
