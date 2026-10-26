# 执行记录

2026-10-16 20:20 CDT（2026-10-17 01:20 UTC）启动。控制器PID1163349；持久目录`/ossfs/workspace/llm-memory-editability/ce-cross-sample-ablation-v1`。原批次完成收据、预测和权重不修改。

- 冻结source-v1：`79a5cf4cf4870527c2d6653e39d86a647ea991ae8bcf4204208f4eb36154ad83`；完整444项父源码逐字节保留，新增10项。
- 固定镜像73项pytest、8项新增Python文件Ruff检查/格式全部通过。此前主机测试缺少PYTHONPATH已修正调用；主机无Torch，科学契约以固定镜像为准。
- CPU实际0–95；六臂分别80–83/84–87/88–91/92–95/0–3/4–7，各4核。两个prepare借各自第一臂资源，结束后才进入六臂。
- 启动时仅GPU7独占空闲，其余外部任务不抢占。每GPU阶段重新申请空闲卡，记录资源历史，不依赖旧GPU分配。
- 工程检查每架构46步后恢复theta0；开发/科学/审计更新分别计数。

当前仍执行中。先读持久目录controller-process.json、controller-state.json、各臂状态及实际LM Docker，不能重启已提交入口。准备和科学W&B链接在各阶段wandb.json；最终完成后追加结果与核验收据。

## GPU7本批三路共享调度修订

2026-10-16 20:52 CDT（01:52 UTC）切换至本批最多三个GPU容器共享GPU7，其它项目与其它GPU不动。原单臂实际总显存约2.3GiB，新Loop峰值1.42GB，旧同规模DiscoLoop峰值1.24GB，GPU7总约97GiB；仅允许本批精确身份容器的计算PID、至少16GiB空余，外部PID存在则等待。CPU与48g内存配额不改。共享耗时不作独占GPU效率比较。科学source-v1及plan哈希不改，优化预算不改。

原宿主控制器1163349在部署动作之间停止；正在运行的Docker审计不停止、不重启，所有已完成阶段沿用原launch/exit/result收据。新宿主控制器1364261，先只读复核源码、预算、Docker身份与实际显存，再三路接续。恢复须先读`shared-controller-process.json`、`schedule-amendment.json`与实际Docker，不能重启原入口。

首次接管漏关Python字节码，生成一份新缓存，被原源码文件集合检查拦下；全部454个清单文件逐字节未变，额外缓存及故障日志隔离保存，目录恢复原集合。第二次遇到Docker缺失容器的实际小写错误响应，所有新任务在创建前被拦下；原Loop取消截断的科学与独立审计已完成，未重做。补23项调度回归及真实只读Docker预检后第三次提交成功。两次故障均为零新增模型启动/优化，全部证据及三个调度脚本版本在持久`scheduling-intervention-shared-gpu7/`；不改写为科学失败或隐藏更新。

最终调度脚本SHA为`af7db2cd6a7b207985a471088d8ea6b1e60d3e9a2d7c3354ed43fc9c89a7c111`，测试SHA为`473eaed45d8ab8842a8008eadbdddd1e2921380cd36b45485371188dbd1899af`。阶段容器继续使用原固定source-v1，而非调度脚本的源码目录。

## 等待用户安排另一台服务器四卡

外部任务随后新占用GPU7，严格入场检查识别外部PID并等待，未与其共享。六臂开发门槛全部通过；DiscoLoop低学习率2e-5仅27/32、4e-5为32/32，固定选4e-5。Loop carry开发独立重载已通过，其余三个DiscoLoop开发重载待做。

用户要求列明实验，由其安排另一台服务器四张空闲GPU。已确认本批无运行容器、四个剩余science均尚未创建，停止等待中的宿主调度器1364261防重复；这不是完成全批或取消剩余实验。两个Loop臂完整科学/独立重载/W&B全通过，四臂剩余正式10240更新。批次状态`awaiting_four_gpu_remote_dispatch`；远端尚未提交。

[四卡交接说明](remote-handoff.md)和本地`results/ce-cross-sample-ablation-v1/remote-handoff-manifest.json`已生成；持久`cross-server-handoff/`含README、输入/阶段/hash清单及只读21.6MB源码归档。归档SHA`52f1ca328d58496068b977f4ce9c4fc456f8f3f9d736a463ae0cb83e7bf6d185`，不含凭据。重启任何旧controller都会破坏防重约定；远端只执行清单剩余阶段，完成四臂独立重载/云端核验后生成一次总报告。
