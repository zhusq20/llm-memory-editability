# 无学习门控基线续训：ungated-loop-continuation-v1

## Material Passport
- Workflow: academic-research-suite / experiment-agent / run, inline with independent read-only resume-code review.
- Status: PREPARED; scientific completion is determined only by controller outputs.
- Authorization: 2026-10-17 user requested scheduling ungated DiscoLoop and ordinary Loop to 1.5M steps as controls.
- Evidence: two completed native 300k full checkpoints from loop-hop-unified-v1; no new research claim.

## 问题与去重
现有门控模型已在续训至150万步，而无学习门控DiscoLoop与普通Loop只有30万步终点。补齐相同数据曝光下的长期对照，不增加架构、图或超参网格。
复用父批disco-l1-r4与loop-l1-r4各30万步。新批各续120万步至总150万步，共240万新增科学更新；原入口不重启，父结果不改。

## 冻结比较
两臂均一个768维、12头共享块，执行四轮，原图/初始化/样本流/dropout/AdamW配方不变。DiscoLoop只用固定系数1的解码—再编码H+q；没有学习注入门、没有更新门；末轮直接读H。普通Loop没有再编码通道。完整恢复权重、Adam各参数状态、流位置、曝光计数、CPU/CUDA RNG和累计步数；不重复warmup。学习率1e-4、衰减.1、batch512不变。

主指标为总150万步/R4图B三跳答案准确率；同时报告答案+生成EOS、两/四跳、全部11池、原子前提覆盖、原始预测、每10万步曲线及终点R1–4。按相同步数比较已有门控模型，不用最佳节点替代终点。累计及新增步数、曝光、token、矩阵FLOPs、耗时分别记录。

## 执行
本机GPU1等待原token-update训练及grid成功后运行DiscoLoop；GPU2等待token-update-half-lr两阶段成功后运行普通Loop。各阶段重查实际进程、两个Docker上下文占用与GPU UUID；不抢占、不共卡、不更换既有队列。等待最多168小时，每阶段24小时；失败保留且不自动重试。全部LM Docker，2CPU/16GiB每容器，独立临时目录，W&B在线记录。

复用当前本机固定镜像，与父批镜像不同；明确保留环境差异，不声称跨机器连续轨迹逐位等价。训练前自动检查两次独立恢复的CUDA单步、完整Adam/RNG与graph捕获状态，工程更新丢弃（每臂2次，共4次）；未通过不得进入科学训练。既有用户短训/独立终点审计豁免沿用，不将未执行的审计报告为通过。终点R1–4评测仍执行。

## 边界
单个已观察开发图和初始化，非新图确认，非原Wang任务。训练步数相同不意味着FLOPs相同。父批科学终点与独立重载已通过，其历史云端收尾失败仍保留。此批只补齐长期基线，不因阶段曲线自动改变学习率、提前停止或追加训练。
