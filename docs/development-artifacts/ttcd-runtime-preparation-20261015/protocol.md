# TTCD 的 LM 容器、持久挂载和 W&B 环境准备

2026-10-15，用户明确要求准备运行环境并提供 W&B 凭据。本次只做基础设施和工程检查，不提交正式训练、编辑或排队任务；GPU8/9已有其它项目任务，不抢占。

## 已准备

- 主机：`sn4622129202`，Ubuntu/kernel 6.8，用户 UID/GID 1015:1016。旧主机的 `/lib64` 修复不套用本机。
- 独立 context：`lm-memory`，endpoint `unix:///run/docker-lm-memory/docker.sock`，daemon ID `5b25d785-927a-4a4c-b8a0-6c49b9127f92`。
- 独立镜像存储：`/workspaces/docker-lm-memory`，overlay2，独立exec-root、pidfile、containerd namespace和`lm-memory.slice`。关闭bridge/iptables/IP转发/masquerade管理，任务使用host网络；原Docker配置、服务与网络不改。隔离依据见[Docker多daemon文档](https://docs.docker.com/reference/cli/dockerd/#run-multiple-daemons)。
- 系统服务：`docker-lm-memory.service`、`ossfs-workspace.mount`均active并enabled；本次未重启机器，开机恢复只完成配置，未做重启实测。
- 持久入口：`/ossfs/workspace`真实bind至`/mnt/disk2_from_server2/siqizhu4/lm-memory-persistent/workspace`，底层为已有NFS4 `172.22.224.42:/mnt/disk2`。新实验结果、冻结源码和恢复材料放在`/ossfs/workspace/llm-memory-editability`。记录实际NFS来源，不冒称恢复了旧主机的对象存储数据。
- 镜像：`lm-memory-ttcd:torch2.8-cu128`，实际ID `sha256:9643a7d6aeda1b4c9ac2ac50c2d84474b082d649a87e423bc6e0bdc0994097ee`。Dockerfile基础镜像固定至digest `sha256:417bd75df6365104c283ea4c1651fb3530d9eb5a4c2fafa51943cff2a94e6385`；构建使用`docker --context lm-memory build --network=host`。
- 镜像版本：Python3.11.13、Torch2.8.0+cu128、Transformers4.57.3、NumPy2.2.6、pytest9.1.1、Ruff0.14.0、W&B0.30.0、HFHub0.36.0。与此前主机借用的Python3.12环境不同；新镜像CPU检查重新完成。
- W&B凭据仅在`/home/siqizhu4/.cache/llm-memory-ttcd/wandb-token`，目录700/文件600。账号`zhusq20`、项目`llm-memory-editability`访问已验证；不保存到仓库、Docker环境配置或日志。launcher默认读取这个私有文件，只读挂载到容器内部再读取；可通过`--wandb-token-file`换路径。

## NFS与容器布局

NFS root_squash阻止容器root读取用户私有文件或写入用户目录。任务以调用者UID/GID运行，`HOME`与XDG缓存使用各容器独立tmpfs。基础project只读，整个持久结果根只读映射到`/project/results`，再仅将本任务输出覆盖为可写；冻结源码另只读挂载`/frozen`。这样避免Docker初始化时穿过本机私有`results/`目录，不改变该目录的私人权限。

只将新建持久batch/source/out挂载源目录设为755以便daemon定位；凭据不进入这些目录，模型文件仍由用户拥有。正式任务仍按GPU UUID、CPU、内存、tmpfs分配，并在启动前重新检查占用。实验源码与数据备份保留在该批持久目录，不重开旧执行入口。

## 验证

1. 独立daemon上的镜像构建成功，固定digest后缓存重建的镜像ID相同。默认Docker服务PID始终为3069；没有调用任何其它项目的停止、重启、删除操作。默认daemon仅用于一次性管理辅助容器来执行host初始化，实际环境检查全部通过`lm-memory`执行。
2. 镜像内66项CPU契约通过，2项明确CUDA测试跳过；包括短程外层训练、保存和新进程完整编辑重放。只有只读pytest缓存写入告警，不影响结果。
3. 一个容器在持久目录写CPU矩阵/梯度检查点；另一独立容器重载后逐值/梯度一致，SHA256为`92788b9c4083e2f72a93a05226853ba375d96810ed4abb091dd40b762d972608`。这确认数据不依赖容器生命周期。
4. 使用正式同样的只读project、只读结果根、只读冻结源码和单任务可写目录布局，额外CPU preflight通过。单独环境批`ttcd-runtime-layout-check-20261015`不属于正式训练批。
5. GPU8/9各一个只读NVML容器准确看到指定GPU UUID，驱动580.105.08；未分配CUDA计算。新镜像的GPU矩阵/反向/短训核验待GPU空闲后，由正式preflight/短测执行，不把枚举当作计算通过。
6. [W&B环境检查](https://wandb.ai/zhusq20/llm-memory-editability/runs/p9jvmy17)已finished，`scientific_training=false`、`training_updates=0`；API重新读取云端终点、summary和history均通过。它只验证环境与上传，不是科学实验结果。

所有检查容器均已退出0，停止容器和日志保留，LM daemon保持运行。工程证据在`/ossfs/workspace/llm-memory-editability/runtime-preparation-20261015`，本机日志副本在`results/ttcd-runtime-preparation-20261015`。首次bootstrap因NFS权限未读取脚本，失败记录保留；改用stdin传递脚本，随后修正Redis辅助入口继承的私有umask，最终挂载访问通过。NFS不支持所尝试的POSIX ACL；该尝试未改变原`results/`的700权限，最终通过容器内父目录只读映射解决。

## 使用

默认context未切换，其它项目继续原使用方式。项目根目录下：

```bash
TASK_PY=/mnt/disk2_from_server2/siqizhu4/llm-memory/.venv/bin/python
docker --context lm-memory info
docker --context lm-memory image inspect lm-memory-ttcd:torch2.8-cu128

# 只输出计划；不启动训练。
"$TASK_PY" scripts/launch_ttcd_second_hop.py train

# GPU8/9空闲且再次检查资源后，才执行GPU preflight。
# 凭据已经配置，默认从本机私有文件挂载。
"$TASK_PY" scripts/launch_ttcd_second_hop.py preflight --execute
```

从零训练的科学问题、数据划分和开发预算仍见[实验协议](../ttcd-second-hop-scratch-v1/protocol.md)。本次环境准备没有额外授权或自动提交正式长训练。
