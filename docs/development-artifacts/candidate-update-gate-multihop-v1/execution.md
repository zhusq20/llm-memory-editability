# 执行记录

**最新状态：首个GPU0尝试已退出137；新尝试已排队等待GPU2前序完成，尚未正式在训。** 失败原因未确认，疑似用户GPU并发配额；未保存更新未知。完整失败记录保留，详见[调度恢复](resource-recovery-v2.md)。下方为首次启动的历史记录。

2026-10-17 22:09 CDT，`candidate-update-gate-multihop-v1` 已在本机GPU0正式启动。

- 唯一变化：共同门从读取LN(H)改为读取LN(C)，C=H+gq；交接仍X+u(C)(C-X)，末轮原始H读出。
- 新增一臂150万步，u初值.95；复用GPU5上的旧joint-token对照，不改其训练或队列。
- 49项CPU检查和Ruff通过；17项全尺寸CUDA/计算图恢复检查通过，唯一工程更新已恢复且未保留。
- 722份源码/数据/协议文件冻结，716份父批冻结文件逐一保持原哈希；同一不可变LM镜像。
- GPU0、CPU24–25、16GiB RAM、独立临时目录和持久输出；真实利用率与两套Docker预约检查通过，没有利用率豁免。
- 控制器PID2503961，依次preflight→train→grid。工程阶段退出0，正式训练已启动；每阶段168小时上限、自动记录进程/显存/利用率和退出码。
- 源码归档保存在持久输出的source.tar.gz。初次准备因只读格式检查缓存权限停止，未进入模型；缓存从源码归档排除，失败原因保存在preparation.json。
- 沿用同系列短训/独立重载/canonical云端终点审计豁免，不将在线记录视为独立验证。当前无科学终点结论。

[冻结配置](../../../configs/candidate-update-gate-multihop-v1.json)、[协议](protocol.md)、[实时控制器](../../../results/candidate-update-gate-multihop-v1/execution/controller-state.json)、[进程](../../../results/candidate-update-gate-multihop-v1/execution/controller-process.json)、[CUDA检查](../../../results/candidate-update-gate-multihop-v1/execution/preflight/candidate-token.json)。

W&B：[candidate-token](https://wandb.ai/zhusq20/llm-memory-editability/runs/lhdec13c00)。在线记录已初始化；终点云端独立核验未执行。
