# 原始两路状态打分：loop-state-handoff-development-v1

2026-10-22，用户明确授权设计、修改代码并执行实验。新增一条原世界、原初始化的 150 万步训练；工程检查和训练状态分开记录。

## 问题与依据

现有 invariant 用熵、最大概率、相邻解码内积、状态相对变化与余弦五项人工统计量控制跨轮交接。本批问：让小网络直接读取旧状态与候选，能否在相同数据和训练配方下学到有效的组合计算与跨轮交接？不预设它更稳定，也不把门分数解释成校准的正确概率。

方法依据已核对的原论文：[GRU，EMNLP 2014 §2.2](https://aclanthology.org/D14-1179.pdf)、[Highway Networks，2015 §2](https://arxiv.org/pdf/1505.00387)、[GTrXL，ICML 2020 §3.3](https://proceedings.mlr.press/v119/parisotto20a/parisotto20a.pdf)、[InfiLoop，2026预印本 §3、附录D](https://arxiv.org/pdf/2610.11570v2)。GRU/Highway/GTrXL 从原始状态学习信息交接；InfiLoop 用候选内容分数与历史质量归一化。这里将前者的两路输入用于后者的打分接口，是受原方法启发的适配，不是完整 GTrXL 复现。

## 历史去重与唯一改变

复用 `loop-native-handoff-development-v1` 的 invariant、ema 和 content 终点；不重训。旧 content 只从候选读取线性分数，本批同时读两路状态并增加一层非线性，因此没有重复原线性打分实验。旧 u/P 门、规则门、KV 交接及其网格不重新打开。

设旧聚合状态为 h，沿用原共享块、原始输入重新注入与软解码再编码得到候选 c。只改分数：

```
x_t = concat(h_{t-1}, c_t)                     768 + 768 维
s_t = w_out^T GELU(W_in x_t + b_in) + b_out    1536 → 16 → 1
w_t = exp(clip(s_t, -8, 8))
N_t = beta N_{t-1} + w_t c_t
D_t = beta D_{t-1} + w_t
h_t = N_t / D_t
gamma_t = w_t / (beta D_{t-1} + w_t)
```

不对两路输入额外归一化，不计算五个统计量，也不直接向打分器传入解码概率、真值、实体编号标签、跳数或轮次。候选保留原有 RMS 归一化和解码反馈。输出层零初始化，开始时所有分数为零，与旧 ema 的前向和骨干/衰减参数梯度相同；第一层初始梯度为零是该初始化的结果，后续输出权重更新后可学习。额外参数用独立随机数生成器创建，骨干、dropout 与样本流不受打分初始化消费随机数的影响。

打分器含24,609参数，连同 beta 共24,610；旧 invariant 为114、content为769、ema为1。原始状态信息与非线性容量同时改变，结果不能单独归因于不变性或某项统计量。新版本不称 invariant，不声称隐藏坐标旋转不变，也不把这一对称性等同于实体重命名。

## 配对、预算与评价

原两图置换世界、同一40,000训练行、11池43,000查询；数据哈希、初始化骨干哈希、样本流、dropout 和轮数流与旧 native 配方相同。宽768、12头、一个共享层、batch512、AdamW、学习率1e-4、warmup2000、余弦至1e-5、weight decay .1、betas(.9,.95)、梯度裁剪1、参数EMA .999。首段随机4–6轮、第二段随机1–4轮，两段均只监督答案与EOS，段间截断梯度；每批一次更新。

新增科学训练仅 `state-mlp` 一臂 × 1,500,000 步；工程短训32步另计，不接续其权重正式训练。没有新世界、新初始化、学习率或隐藏宽度网格。

主要终点：固定150万步、参数EMA、R8的B图两/三/四跳答案准确率。保留全部原21个节点、A留出四跳、两图原子、11池、原始权重，以及R4/R8学习曲线。终点独立容器重载R1–32，报告R4/8/16/32答案与EOS、逐题新增正确和丢失旧正确；不选最佳检查点或轮数。额外打分矩阵FLOPs单独计入。训练更新数、样本曝光、轮数、计算与耗时分别记录。

## 执行与恢复

固定LM镜像 `sha256:c77af324439d8a2f838b57c3a386fc8477135015a361e4fbf6a8389facbf9f14`，显式 `docker --context lm-memory`。独立只读源码快照、同绝对路径只读输入、独立可写 artifacts、每阶段独立容器、2 CPU/16GiB、每卡一个本批任务。无网络监听端口；每容器独立/tmp。W&B为 `zsqzz/llm-memory-editability`，凭据仅从私有文件注入训练/云端核验容器。

CPU契约与静态检查通过后提交独立控制器；等待候选GPU真正空闲，CUDA活动进程与其他容器预约均阻止接入，不覆盖预约。GPU5/6/8在现有迁移控制器全部完成前保持原优先级；不更改旧调度器、训练或冻结来源。若没有可用卡，状态必须明确为 `waiting_resource`，不能称正式训练已开始。

阶段：`environment → smoke → smoke-grid → train → grid → cloud`。先验证真实CUDA矩阵/反向/同步和全宽EMA初始一致性，再进行32步短训与新进程逐token重放；工程检查通过才训练。任何失败保留容器、日志和退出码并停止，不自动重试。终点重载、保存预测重放与W&B配置/步数/结束状态核验分别记录。

持久目录为 `/ossfs/workspace/llm-memory-editability/loop-state-handoff-development-v1`（同时可由原持久绝对路径访问）。恢复先读 `freeze.json`、`controller-process.json`、`controller-state.json`、`execution/` 和 `artifacts/` 中对应状态，再核对实际LM容器；禁止重复提交。源码更改须另留修订记录，不能改写冻结来源。
