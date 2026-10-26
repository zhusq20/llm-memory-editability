# 恢复成功配方：DiscoLoop基础组合能力

2026-10-15，用户明确要求恢复此前成功的DiscoLoop配方，先确认基础组合能力，再比较单事实与跨样本连续写入。此批只重建基础模型；后续写入另见[配对协议](../ttcd-successful-base-v1/protocol.md)。当前为代码与工程核验阶段，尚无新科学结果。

## 为什么需要独立恢复

此前[高分批次](../../results.md#supervised-alignment-development-v1)的DiscoLoop在同一Wang开发图上，IO/OI/OO为97.53/99.30/95.66%。后来的[小模型TTCD批次](../../results.md#ttcd-second-hop-scratch-v1)同时改变宽度、层数、数据、组合支持、训练步数和优化；其20k写入训练前就没有可靠的旧组合能力，不能代替本检验。恢复成功条件用于核验前提，不能凭旧高分预先断言更正会传播。

原W&B运行[ci3x3pk1](https://wandb.ai/zhusq20/llm-memory-editability/runs/ci3x3pk1)的终点记录已核对。固定HF revision `6c580455ce1667274ea592c1c88f5d0fb1e5a190`的原图、面板与GPT2配置已恢复并核对归档及逐文件哈希。HF没有模型权重，W&B无模型Artifact，旧主机无现成可信SSH连接；未找到原300k权重。因此本批是**从完全一致的原初始张量，按原配方做独立的新训练**，不是恢复旧终点，也不重启完成的旧入口。恢复清单见[recovery-manifest.json](recovery/recovery-manifest.json)。

## 固定配方

[配置](../../../configs/discoloop-success-restore-v1.json)固定以下条件，科学训练直接复用现有`run_supervised_alignment.py`，不改原损失、优化器、数据流或CUDA graph更新。

| 项目 | 设置 |
| --- | --- |
| 世界 | 原Wang图：2000实体、200关系、出度20，seed790101；38000 ID及2000 OOD原子事实 |
| 组合支持 | φ7.2，273600训练II，与40000原子混合的partial-epoch打乱流 |
| 模型 | GPT2宽768、12头、4物理层×2轮、完整词表52463、位置1024；69431040参数 |
| 通道 | 首轮结束后原全位置软解码加法通道；α=1、τ=1、RMSNorm eps1e-6；无共享KV及辅助损失 |
| 初始化 | seed791101，初始张量SHA `ba0af28431d36db482f2676c1e111a725cf51041c3dca9d006dd10d7a7c18518`，已实际逐张量核对及独立重载 |
| 优化 | 300000步、batch512、lr1e-4、warmup2000后恒定；AdamW wd.1，bias/LN不衰减；原fused/capturable实现 |
| 数值 | 原dropout.1、dropout seed20261012、BF16 AMP、TF32开启、原梯度/图捕获流程 |
| 样本流 | 原stream seed792101，保留padding尾批和曝光计数 |

原镜像不可得；当前固定镜像为`sha256:9643a7d6aeda1b4c9ac2ac50c2d84474b082d649a87e423bc6e0bdc0994097ee`，Torch2.8/cu128、Transformers4.57.3，与历史环境不同。实际已确认新环境原初始张量完全一致；**不宣称完整随机数消费、训练轨迹或终点逐位复现**。

## 顺序、门槛与停止规则

控制器依次运行preflight、32步全尺寸工程短训、独立短训重载、300k正式训练、独立七池全量重载、基础能力门槛、W&B完整JSON与终点核验。工程短训与正式训练相互独立，不用于选超参数。

只检查固定300k终点：原子ID/OOD、训练II及测试II准确率各≥98%；IO/OI/OO各≥90%，且分别较历史97.5333/99.3/95.6587%下降不超过2个百分点。原answer+EOS评分及七池样本不改；要求独立新进程全池逐token、NLL、曝光和权重哈希一致。基础门槛只检查旧世界，不看任何更正后两跳答案。任一项不通过则保留结果并停止，不自动延长训练或搜索参数。

通过后才交接至[共同读取及写入配对批次](../ttcd-successful-base-v1/protocol.md)。恢复基础最多新增300000更新；后续共同读取10000、两臂各10000，总授权本轮最多330000更新。两道实际能力检查不能相互替代。

## 资源、追踪与恢复

显式LM Docker context `lm-memory`；固定镜像、冻结源码/配置/输入，各阶段独立容器。输出保存`/ossfs/workspace/llm-memory-editability/discoloop-success-restore-v1`。只等待GPU8单事实旧实验或GPU9连续写入旧实验**对应本地科学重放审计通过，且实际GPU无运行容器、无CUDA进程、充分空闲**后使用该卡；不等待另一卡，也不抢占任何任务。GPU9未知进程存在时继续等待。CPU24–31、96GiB内存、资源等待24小时、获卡后各基础阶段合计硬上限12小时，只停止本批自己的容器。

正式训练、审计、门槛分别记录W&B，团队`zhusq20`、项目`llm-memory-editability`。保存所有节点七池曲线、曝光/token、计算估算、墙钟、容器资源及失败。源码与输入在启动前冻结，后续配置/入口哈希在交接前复核。能力门槛失败不会创建后续训练。

恢复先读本批`controller-process.json`、`controller-state.json`、各stage claim/exit、`source-manifest.json`及实际LM容器；不重启旧控制器，不覆盖任何失败，也不根据历史GPU编号重复提交。
