# 本机四臂接续记录

用户指定本机 `sn4622128200`，明确允许重新构建镜像，并批准新增专属64GiB存储文件、格式化挂载及独立 `docker-lm-memory` 服务。之后要求快速启动，不重复无关验证。

## 当前状态

**2026-10-16 22:07 CDT全批完成。** 恢复PID3928286已完成低学习率原2560更新、129状态独立重载和完整云端核验；原GPU1 carry-adam由原控制器完成并通过独立重载/云端核验。六组全部通过，合计15360科学更新、4480开发更新、92工程更新，774正式状态重放通过。总报告W&B [`tedcqs8j`](https://wandb.ai/zhusq20/llm-memory-editability/runs/tedcqs8j)完整JSON下载哈希一致；science/audit/cloud/report容器退出码均0，LM无运行容器，GPU1/2显存0。不得重新启动任何已完成入口。

低学习率DiscoLoop终点新原子7/128、双新0/64、旧原子510/510、旧组合1256/1264；取消截断臂双新61/64但旧组合1023/1264。完整结论与限制见[结果汇总](../../../results.md#ce-cross-sample-ablation-v1)和[报告](../../../../results/ce-cross-sample-ablation-v1/report.md)。恢复原源码及失败、探针身份、格式化前后AST一致性和完成收据保存在`results/ce-cross-sample-ablation-v1/scheduler-repair-20261016/`；实际启动SHA为`84b3fd863371f11cbb2177015c4023670450e849fb5490aa24cbabdb95104787`，可审阅入口随后仅格式化且未重新运行。Ruff及格式检查通过。

**2026-10-16 21:51 CDT调度故障修复：低学习率组正式写入已在GPU2接续。** 用户要求修复并完成评测。原接管控制器为DiscoLoop lower-lr再次分配GPU时，复用了已经存在的CUDA检查容器名称，导致退出125；其开发审计容器实际已成功退出，但调度器没有收集退出收据。新增独立恢复入口`repair_lower_lr.py`（PID3928286）核对容器ID、镜像、GPU UUID、退出码和64状态审计结果/原结果哈希，补收退出记录并复用已通过检查；原gate与audit-gate不重做，原冻结454文件及科学配方保持。GPU2接续原2560科学更新和完整逐事实读出，之后129状态独立重载、云端完整JSON核验。其他控制器及GPU1任务保留，等其释放全批锁后汇总六组。原名称冲突、恢复前置检查发现缺失收据的失败和两份日志保留。恢复前六项探针身份/退出/GPU检查与固定镜像Ruff通过。当前尚待正式终点及全批报告；下文此前状态保留历史。

**已开始正式科学更新，尚未完成全批。** 用户提供正确凭据后，零优化W&B写权限确认运行`2545qad1`通过。两次原凭据故障均在模型加载前退出，无MLP状态/读出、优化0，已整体归档至 `sn4622128200-continuation/credential-failure-attempt-1/`；原失败容器保留。随后以明确零更新故障修订、`v2`新容器名恢复控制器PID3834890。

Loop carry在GPU1实际完成首个事实20步并继续执行，正式W&B：[bz82hlj7](https://wandb.ai/zhusq20/llm-memory-editability/runs/bz82hlj7)。DiscoLoop低学习率已在GPU2接续开发独立重载，其余两个DiscoLoop臂等待独占空卡；控制器会在可用卡上接续，不抢占其他任务。每臂仍为2560更新，合计10240；每臂后续129状态独立重载、完整云端核验及总报告尚待完成。

GPU1通过零优化CUDA检查。GPU3/4在准备期间先后出现外部vLLM占用；用户随后明确GPU2没有异常；该卡矩阵、反向及同步均通过，原利用率报告不能据此判为故障。已允许其在无进程/显存低于128MiB时入场，DiscoLoop低学习率的audit-gate容器已启动。薄控制器接管为PID3849485，原Loop科学容器及进程PID3835702未改变。记录见`gpu2-admission-amendment.json`；不再将GPU2写为异常。当前是两卡工作、两臂排队。

## 环境和证据

- 持久批次本机路径：`/mnt/disk2_from_server2/siqizhu4/lm-memory-persistent/workspace/llm-memory-editability/ce-cross-sample-ablation-v1`。接续收据：该目录下 `sn4622128200-continuation/`。
- 独立Docker上下文 `lm-memory`，socket `/run/docker-lm-memory/docker.sock`，data-root `/workspaces/docker-lm-memory`。现有默认Docker PID3049保持不变。
- 专属存储文件：`/mnt/disk2_from_server2/siqizhu4/lm-memory-persistent/runtime-sn4622128200/docker-ext4.img`，64GiB稀疏文件，经 `/dev/loop4` 挂载。没有配置重启后自动挂载/启动；主机重启后须检查恢复。
- 新镜像：`sha256:27b3ab677fb92aaf4e4cd953e3a87e953c4b164a0d4d4c4785dc9779851c11d2`，使用原Dockerfile及相同基础镜像digest。PyTorch2.8.0+cu128、Transformers4.57.3、NumPy2.2.6、W&B0.30.0等直接依赖固定；实际完整依赖见 `runtime-checks/requirements.txt`。新镜像并非原镜像字节一致的复制，不更改原冻结计划中的历史image字段；实际镜像另记于启动收据和调度修订。
- 原源码454文件逐一哈希通过；原检查点/教师/数据、源码归档、四臂开发result共14项检查通过。结果在仓库 `results/ce-cross-sample-ablation-v1/sn4622128200-{input,source}-preflight.json`。
- 新薄控制器5项防重复逻辑检查通过；遵照用户要求，不重复原73项契约、开发写入和工程prepare。
- 运行检查只确认凭据可认证及公开项目可读，**并未证明可写**；实际 `wandb.init` 的权限错误是当前阻塞依据。
- 初始环境安装被自动审批拒绝，用户明确批准后才执行。NFS root-squash导致两次初始化前失败；最终以文件所有者身份打开/格式化新文件、管理员身份挂载完成，没有放宽旧数据权限。辅助容器 `lm-runtime-bootstrap-20261017`、`-v2`、`-v3` 保留。

## 后续执行约束

仍只运行原四臂的剩余阶段，合计10240更新，不重跑两个完成Loop臂。每臂独立新容器，最多四卡并行，实际卡忙则等待；每臂science后129状态独立重载、W&B完整JSON核验，最后执行一次原报告器。任何阶段已有科学更新时禁止自动重写。两个权限失败已按模型加载前零更新证据归档，并记录新尝试；不得因旧失败而重启当前科学写入。
