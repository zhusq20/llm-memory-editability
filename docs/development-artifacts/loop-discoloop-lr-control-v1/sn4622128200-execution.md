# 本机 GPU1／GPU2 接续四组

2026-10-16 CDT，用户指定本机 GPU1／GPU2，并明确选择“复用本机镜像，验证后启动并记录环境差异”。首次控制器PID4035894因NFS挂载权限失败，四组均未创建容器、正式更新0，完整保留在 `startup-failure-1/`。迁移新执行目录并修正新目录的遍历权限及umask后，GPU1实际矩阵、反向、同步通过，控制器PID4046066接续。每条先通过本机 CUDA、全尺寸32步短训和独立七池重载，再进入正式300000步，随后自动独立重载、曝光核对与 W&B 完整 JSON 云端核验。提交不代表正式训练已开始，实际阶段读取下述状态文件。

| GPU | CPU | 第一组 | 第一组训练及审计完成后 |
| --- | --- | --- | --- |
| 1 | 64–67 | 普通Loop，lr5e-4，wd0.02 | DiscoLoop，lr2e-4，wd0.05 |
| 2 | 68–71 | 普通Loop，lr1e-3，wd0.01 | DiscoLoop，lr5e-4，wd0.02 |

**2026-10-16 22:48 CDT：两条普通Loop均已实际完成1000／300000步，损失及梯度有限，W&B在线追踪正常。** GPU1为 [3ypbuawf](https://wandb.ai/zhusq20/llm-memory-editability/runs/3ypbuawf)，GPU2为 [nxpyxz1x](https://wandb.ai/zhusq20/llm-memory-editability/runs/nxpyxz1x)。尚无正式终点；后续读训练status。两条DiscoLoop仍按表排队，第一组训练和审计完成后自动接续。

四组仍各300000步，共1200000正式更新，从原第0步权重开始；架构、数据、样本流、初始化、训练精度、batch、优化器及评分均不变。原机两组及其监控保持，原六组入口不重启。

## 环境修订与验证

- 原镜像 `sha256:9643a7d6aeda1b4c9ac2ac50c2d84474b082d649a87e423bc6e0bdc0994097ee` 在本机不存在。经用户明确授权，使用本机已构建镜像 `sha256:27b3ab677fb92aaf4e4cd953e3a87e953c4b164a0d4d4c4785dc9779851c11d2`，独立 `lm-memory` Docker，不使用默认项目环境。
- 两镜像的既有检查记录均为 PyTorch2.8.0+cu128、CUDA12.8、Transformers4.57.3、NumPy2.2.6、W&B0.30.0、huggingface-hub0.36.0。镜像字节不同，不宣称完整依赖或跨镜像训练逐位一致。原冻结配置继续保留历史 image 字段；实际镜像由 runtime-amendment、阶段 command 收据与 `WANDB_DOCKER` 记录。
- 本机 RTX PRO 6000 Blackwell，驱动580.65.06；启动前 GPU1／2 显存0，无CUDA进程，CPU集合互不重叠，每组4 CPU、96GiB内存。每阶段仍执行原资源检查，不抢占其它任务。
- 交接包 SHA256 `919eae0bf81cc3b78ba002a2fded8d788280634310e9774467c1cd1be6048dba` 核对通过；215个冻结文件和基线/衰减收据全部通过。源码清单 SHA256 `d057cc10f615ea8c87978d58bb56f610aae6127b94336411d073083365e28ba3` 不变。
- 新环境重新完成四组工程验证，新增最多4×32=128工程更新，与1200000正式更新分列。交接包内旧工程产物完整保留在 `imported-engineering-evidence/`，不覆盖原机数据。
- 新调度器6项防重复、失败拒绝重试、只读计划、输入完整性与镜像身份测试，以及 Ruff 检查/格式检查通过。外层队列持有原共享目录四组文件锁，防止旧交接入口在原机重复提交；原目录四条 `execution/<job>/train` 相对链接在迁移后正式阶段建立时生效，结束后仍保留防重复依据。失败臂保留，不自动重跑，另一独立臂可继续。
- GPU1普通Loop 5e-4、GPU2普通Loop 1e-3均已通过各2项CUDA契约、全尺寸preflight、32步短训及新容器七池重载；短训退出0、审计passed。DiscoLoop两组按各自队列等待，并须通过同样门槛。

## 持久产物与恢复

本机执行目录为 `/mnt/disk2_from_server2/siqizhu4/lm-memory-persistent/workspace/llm-memory-editability/loop-discoloop-lr-control-v1/sn4622128200-run`，位于NFS持久盘，不写入本机临时系统卷。仓库保留 [results/loop-discoloop-lr-control-v1/sn4622128200-run](../../../results/loop-discoloop-lr-control-v1/sn4622128200-run) 符号链接。父目录是原共享批次，其 `handoff/target-sn4622128200.json` 记录去向。只调整新目录的遍历权限，不放宽原仓库私有祖先目录或凭据权限。

先读本机 `handoff/runtime-amendment.json`、`two-gpu-process.json`、`lane-{1,2}.json`、`<job>-status.json`、`<job>-monitor.log`，再核对实际容器及 `execution/<job>/<stage>/`。训练步数和曲线分别在 `jobs/<job>/training/status.json`、`jobs/<job>/wandb-train.json`。最终必须达到300000步，并通过 `result-audit.json` 和 `result-cloud-audit.json`。后台队列及每臂脚本已复制到本机执行目录 `handoff/` 并记录哈希；不要重复执行提交命令。

[W&B项目](https://wandb.ai/zhusq20/llm-memory-editability)。本次只是既有四组的环境迁移与调度，无新的科学结论。
