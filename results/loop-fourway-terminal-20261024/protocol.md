# 四种循环机制的统一训练比较

用户于2026-10-24（America/Chicago）明确授权：四卡并行，四组各150万步，共600万步。
随后确认采用统一现有Transformer主干，并保留各方法的原始初始化、候选与跨轮合并机制。
本批次独立于已暂缓的七条旧初始化实验，不恢复旧队列。

## 固定比较

| 臂 | 初始化与候选 | 跨轮传递与终点 |
| --- | --- | --- |
| invariant | 最新版h0=N0=e、D0=1；首轮F(e)，后续F(h+e) | 身份无关打分、DiscoLoop反馈、归一化历史合并；读出h_T |
| loop | h0=e；逐轮F(h) | 直接传递原始输出；读出H_T |
| discoloop | h0=e；逐轮F(上轮H+软反馈) | 原文符号任务alpha=1、tau=1；反馈为RMSNorm(Eᵀsoftmax(LMHead(H)))；终点alpha=0，读出原始H_T |
| infiloop | 固定、非训练的随机h0，原始方差校正截断正态；候选F(h+e)，不归一化候选、不加词表反馈 | 原始qᵀRMSNorm(candidate)打分，clamp[-8,8]再exp，学习beta；N/D递推；原版outer初始化时的N0/D0在no_grad下评分 |

InfiLoop是用户确认的**单状态、统一主干适配**，不是数独专用的双状态H/L、二维卷积与双向attention模型。
其Connection与初态规则来自[官方代码固定提交](https://github.com/pixeli99/InfiLoop/blob/63716a9c46723d3592c8ad4fc5819855f22aaa47/sudoku/model.py)，
原始源码和MIT许可保存在[references](references/infiloop-model.py)。InfiLoop状态随机种子2026102401，
与主干随机数流分离；四组主干第0步哈希必须相同。
DiscoLoop采用[原文v2符号任务和式4–6](https://arxiv.org/html/2607.00341v2)的固定门机制；
本次在原文及公开仓库检索中未找到作者DiscoLoop源码，因此按明确的原文公式实现，不宣称核验了未取得的作者实现。

## 统一配方

所有臂一个共享GPT2 Transformer block，宽度768、12 heads、绑定输入输出嵌入、因果attention，dropout0.1。
复用已登记的40,000条训练数据与43,000题评测池、相同主干初值、样本顺序和深度采样种子。
每条样本每次出现独立抽Poisson(8)，越界重抽到1–16；整批推进、到期冻结状态，全部有效轮次反传。
只在各自终点监督，CE总和除以全batch有效监督token数，当前为答案和EOS，忽略-100。
末尾不足512条的训练batch沿用已有stream的loss-masked padding；四组有效曝光完全一致。

batch512，AdamW，lr1e-4，2000步warmup与150万步余弦时间轴、末端lr比例0.1，
betas=(0.9,0.95)，weight_decay0.1，梯度裁剪1，参数EMA0.999。
每臂150万次优化器更新；相同更新数和样本曝光不等于相同FLOPs，记录包含到期行的实际计算量与有效采样轮数。

主指标为参数EMA、R8下各池答案准确率；同时记录EMA R4/R16，最终raw/EMA R4/R8/R16/R32及逐轮曲线。
沿用既有评测节点，保留中期回落；不以最佳节点代替固定终点。保存模型、Adam、EMA、数据/RNG状态、
逐题预测、完整曲线与W&B链接。W&B团队zsqzz、项目llm-memory-editability。

## 执行准备

[冻结前配置](../../configs/loop-fourway-terminal-20261024.json)、
[独立入口](../../scripts/run_loop_fourway.py)、[方法实现](../../src/llm_memory_editability/loop_comparison.py)。
仅新增10项必要CPU检查，全部通过：每臂混合深度输出/梯度、同主干、因果性，以及InfiLoop与固定作者源码Connection的数值/梯度对应。
复用前次batch512、深度16的GPU显存验证，不重复展开无关审计。

使用GPU6/4/7/8分别运行invariant/loop/discoloop/infiloop，每任务单独LM Docker容器，2 CPU、16 GiB主存、
0.16 CUDA allocator fraction，固定镜像和root所有的只读源码。启动时重新核对实际资源。
用户先授权4/5/7/8；启动前GPU5出现新运行容器/frosty_banach，因此尚未创建本批次容器即停止入场。
用户随后明确授权改为当前空闲的4/6/7/8立即启动，
允许记录在案且未变化的/slime全卡预约与旧created容器预约例外；新增占用或预约变化仍阻止启动。
其他容器、旧冻结配方、旧暂缓守卫和七条暂缓任务保持原状态。
