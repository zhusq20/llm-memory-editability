# GPU2执行记录

## Material Passport

- Workflow: ARS experiment-agent / run。
- Status: RUNNING，已完成提交与正式启动，终点未到。
- 验证范围：固定镜像25项CPU、45项全数据映射、16项CUDA工程检查与Ruff；新增独立终点审计未执行。
- 执行：当前会话完成；另有一名只读实现审查者，无额外GPU运行。

2026-10-17 23:06 CDT，本机sn4622128200 GPU2正式训练容器已启动，W&B已联机。
控制器PID2673104，GPU UUID为GPU-9c6ca0d7-11bb-e940-c845-efea3cc49c23。
固定镜像为sha256:fa200da9dd78d3423ffc30572bda8b88a5b1c7e72a25509bf63e6ae4a5a3ef6f。
正式容器compact-wang-joint-gate-v1-e19fa61c-joint-token-train。

- [控制器状态](../../../results/compact-wang-joint-gate-v1/execution/controller-state.json)
- [实时步数](../../../results/compact-wang-joint-gate-v1/execution/runs/joint-token/status.json)
- [W&B](https://wandb.ai/zhusq20/llm-memory-editability/runs/lhef69f000)
- [检查汇总](validation.json)

源码/数据1029份冻结文件及原始映射审计保存在持久目录；source-data.tar.gz可恢复精确源码与数据。
正式训练从新初始化开始，前置一次工程更新已恢复并丢弃。七池完整终点评测与R1–4自动接续。
本机/远端已有训练与旧失败控制器均未重启。
