# 融合后状态作为共同门输入：candidate-update-gate-multihop-v1

## Material Passport

- Type: controlled architecture experiment; ARS experiment-agent/run, executed inline.
- Authorization: 用户在核对已有共同门设计后要求“加一组这个修改后版本的这个实验”。
- Scope: 一条新多跳训练，150万步；直接复用已有共同门控为对照。不是追加原Wang任务或新图网格。
- Verification: 工程检查与科学结果分开；正式状态见本批controller与阶段收据。

## 唯一设计改变与历史去重

现有 `joint-update-gate-multihop-v1` 使用完整Transformer输出H计算门，再控制整个候选C=H+g(H)q(H)：

`X_next = X + sigmoid(w^T LN0(H)+b) * (C-X)`。

本批仅把门的条件输入改为融合后的C：

`X_next = X + sigmoid(w^T LN0(C)+b) * (C-X)`。

X是实际上一轮交接状态；H保留完整attention、MLP、残差及归一化。C不detach，q只计算一次；g仍读取未归一化软词向量，q仍RMS归一化。只修改前三次交接，第四轮直接从原始H读答案。不是直接uC，也没有额外累加X。u仍每token一个标量，参数跨轮与位置共享。

u初始w=0、b=logit(.95)。与旧共同门的参数、初始化前向和主干梯度可匹配；门权重梯度从第一次反向就可不同，不能声称后续优化轨迹一致。新增参数数目0，新增监督0。主要问题是：让更新门看到再编码后的候选，是否改善固定预算的多跳使用？候选门可以关闭，不保证它学会识别正确性、循环稳定或自动节省计算。

文献依据沿用讨论中核查的Highway（NeurIPS 2015，§2）、GTrXL（ICML 2020，§3.3/4.3）和Flamingo（NeurIPS 2022，§2.2）。它们提供门控插值/注入先例，不证明本条件输入更优。原文入口：
- https://proceedings.neurips.cc/paper/2015/file/215a71a12769b056c3c32e7299f1c5ed-Paper.pdf
- https://proceedings.mlr.press/v119/parisotto20a/parisotto20a.pdf
- https://arxiv.org/html/2204.14198#S2.SS2

## 配方、预算与评测

沿用旧共同门的同一40k训练查询、43k评测查询、图/初始化/样本流/dropout种子、宽768、12头、一个共享GPT层×四轮、batch512、AdamW lr1e-4、wd.1、warmup2000及恒定后续LR。只新训练candidate-token一臂1500000步；不重复已有joint-token。初始权重从共同种子生成，不接续已训练对照检查点。

主指标固定1500k/R4图B三跳全词表答案准确率；保留300k中间比较、原全部节点、400k起每100k全部11池、答案+EOS与NLL、终点R1–4。保留原评分、原始预测与训练曝光。只是一张已观察图/一个初始化；查询/轮次/检查点不当独立世界，不用峰值代替终点。

模型和训练依赖从旧共同门的实际冻结source复制，在同一不可变LM镜像运行。新配置固定旧配置及对照freeze哈希；新增独立模型/runner不修改旧冻结源码。GPU0独立容器，CPU24–25、16GiB RAM、独立tmpfs、持久输出；真实GPU占用与两个Docker上下文的预约均必须空闲，不继承旧GPU5/6利用率豁免。GPU或机器并发负载差异不作独占耗时证据。

## 检查、执行与恢复

CPU检查包括初始前向/主干梯度配对、门权重梯度差异、门实际读取C、经C的梯度、因果性、末轮读出、仅一次再编码和配置漂移拒绝。CUDA检查沿用父runner，包含全尺寸前后向、计算图与一次丢弃并恢复的工程更新，检查通过才训练。

沿用现有同系列的32步短训、独立终点重载和canonical云端审计豁免；不把未做的审计记为通过。W&B在线同步配置、曲线、步数、曝光、耗时和资源，团队zhusq20/llm-memory-editability；凭据不进入源码/日志。阶段为preflight→train→grid，每阶段168小时上限、进程与资源监看；失败保留且不自动重试。

恢复先读本批controller-process.json、controller-state.json、freeze.json、execution阶段退出码与实际LM Docker。当前训练未完成，不产生科学结论，不重复提交。
