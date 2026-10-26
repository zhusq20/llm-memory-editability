# Query控制300k：执行记录

## Material Passport

- ID：`loop-query-control-development-v1`；类型：执行状态，不是终点结果。
- 冻结协议：[protocol.md](protocol.md)；最小架构来源：[用户附件](source-plan.md) §5.2。
- 当前实际入口：`scripts/resume_loop_query_control_capture.py`；只在原控制器的零科学更新启动故障后恢复，同一批次不重复冻结或提交训练。

正式任务已进入GPU4，W&B：[pb4lkaw0](https://wandb.ai/zhusq20/llm-memory-editability/runs/pb4lkaw0)。仅一条普通Loop＋32维循环控制→768维Query调节，300000步；无DiscoLoop、KV或辅助损失。Q自身仍为768维，控制分支不是对全部Q的压缩或逆映射。模型69,505,792参数。

固定镜像10项CPU契约、Ruff check/format检查通过。全尺寸512×4 CUDA预检查七项通过：初始输出逐值一致、全部backbone梯度逐值一致、Query映射获得有限非零梯度、capture后全部权重/Adam/CPU RNG/CUDA RNG恢复。预检查0存活更新。普通Loop的已审计300k foundation复用，不重训。

## 保留的启动故障与修订

1. 原控制器在CPU契约容器创建前失败：新建目录继承0700，LM daemon无法遍历绑定的source。只将本批新建source/input/output遍历目录设为0755；旧批次和私有凭据权限未改。凭据通过Docker API安装为当前容器用户只读0400，内容未进入仓库、参数或日志。
2. 同一冻结全尺寸preflight的CUDA capture失败：eager逐值梯度比较留下`logits/loss`计算图引用，在legacy stream上保留梯度累积节点。原失败、退出码和科学源码保留。用原正式训练路径单独验证capture成功、权重完整恢复后，仅在preflight wrapper释放这两个引用；完整七项检查通过。
3. 修订只影响启动环境与预检查对象生命周期。正式训练仍执行原`source/scripts/run_loop_query_control.py train`及原冻结模型/数据/配置。没有重启已产生科学更新的任务，没有额外短训练。

持久根：`/ossfs/workspace/llm-memory-editability/loop-query-control-development-v1`。恢复先读`controller-process.json`、`controller-process-resume-v2.json`、当前`controller-process-resume-v3.json`、`controller-state.json`、`freeze.json`、`runtime-repair/{amendment.json,capture-v3/amendment.json}`，再核对实际LM容器。当前容器名`loop-query-control-development-v1-train-capture-v3`。完整旧故障日志保留于`execution/contracts`、`execution/transport-v2`，当前阶段位于`execution/capture-v3`。

正式训练终点、独立七池重载、云端完整JSON核验分别待完成；控制器会依次执行，不提前宣布通过。单图/初始化、额外参数/状态/计算未配平；结果不能单独证明“内容—控制分离”，也不能替代与不保留跨轮状态的等参数Q adapter比较。

## 后续任务讨论

保留当前两跳检验，不将接近满分的DiscoLoop与普通Loop混称。用户提出更难任务的必要性后，建议再增加混合两跳/三跳，借鉴DiscoLoop附录A.2并保持架构独立比较；用户随后明确授权新增普通Q adapter300k，已另立[近等参数对照](../loop-query-adapter-development-v1/execution.md)，父训练不重启；更难任务仍未入队。
