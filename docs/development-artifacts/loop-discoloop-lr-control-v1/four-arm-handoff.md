# 四组学习率实验交接

本次只转移原六组中尚未开始正式训练的四组。原机普通Loop 2e-4与DiscoLoop 1e-3继续训练，由独立监控器接续终点七池重载和W&B核验。原六组调度器停止，四组不再自动占用原机空闲卡。原配置、数据、初始化、源码及已完成的检查均保留。

## 要跑什么

| 脚本中的job名称 | 架构 | 学习率 | AdamW衰减 | 正式预算 |
| --- | --- | ---: | ---: | ---: |
| loop-lr5e-4 | 普通Loop | 0.0005 | 0.02 | 300000步 |
| loop-lr1e-3 | 普通Loop | 0.001 | 0.01 | 300000步 |
| discoloop-lr2e-4 | DiscoLoop | 0.0002 | 0.05 | 300000步 |
| discoloop-lr5e-4 | DiscoLoop | 0.0005 | 0.02 | 300000步 |

合计120万正式更新。这是从原第0步权重开始的基础训练，不是知识编辑或从当前两组中途检查点续训。对照是已完成的各架构1e-4结果，不重训。

共同设置：4个物理Transformer块循环2轮、宽768、12头、batch512、warmup2000后恒定学习率、同一Wang图和样本流，只监督答案与EOS。学习率和对应衰减系数是本批改变的变量；不增加KV或辅助损失。完整定义见[冻结协议](protocol.md)。

## 准备材料

原机持久目录：`/ossfs/workspace/llm-memory-editability/loop-discoloop-lr-control-v1`。

交接包：同目录`handoff/four-arm-inputs.tar`及`four-arm-inputs.tar.sha256`。包内有冻结源码、配置、数据、原初始化、基线小型记录、四组已完成阶段的输出/退出码，以及独立运行脚本；不含W&B凭据和Docker镜像，也不含原机正在训练两组的权重。包内路径不含外层batch目录，解压到一个全新持久目录。

如在同一机器另选GPU，可直接使用原持久目录，无需解压或复制。不同机器必须先搬运交接包和固定镜像；不要分别复制多份包后重复执行同一job。

固定镜像ID为`sha256:9643a7d6aeda1b4c9ac2ac50c2d84474b082d649a87e423bc6e0bdc0994097ee`。原机导出方法（只导出，不训练）：

```bash
docker --context lm-memory image save \
  -o /你的持久目录/lm-lr-image.tar \
  sha256:9643a7d6aeda1b4c9ac2ac50c2d84474b082d649a87e423bc6e0bdc0994097ee
```

目标机安装可用的NVIDIA驱动、Docker及NVIDIA Container Toolkit，并准备自己的Docker context。导入到运行时使用的同一个context；下例用`lm-memory`，若目标机用`default`，两处一致替换：

```bash
docker --context lm-memory image load -i /你的持久目录/lm-lr-image.tar
sha256sum -c four-arm-inputs.tar.sha256
mkdir -p /你的持久目录/lr-four
tar -xf four-arm-inputs.tar -C /你的持久目录/lr-four
```

凭据在目标机自行准备为权限600的私有文本文件，只包含W&B API key，不放进包或仓库。运行账号需要读取Docker并写入解压目录；脚本按当前账号UID/GID运行容器。默认每组4个CPU、96GiB内存，GPU至少90000MiB总显存且80000MiB空闲，并要求实际无其他CUDA进程。磁盘至少预留60GiB。脚本会等待空闲，不抢占任务。

## 一张GPU顺序跑四组

进入解压目录。下例GPU为0，CPU为4、5、6、7；请换成目标机实际空闲的资源。第一次仅输出计划，不启动容器：

```bash
cd /你的持久目录/lr-four
bash handoff/run_lr_control_four.sh \
  "$PWD" 0 4,5,6,7 /你的私有目录/wandb-token
```

确认路径和资源后正式运行；可放在tmux中保持会话：

```bash
EXECUTE=1 bash handoff/run_lr_control_four.sh \
  "$PWD" 0 4,5,6,7 /你的私有目录/wandb-token
```

单卡依次完成四组及各自审计。目标机context为default时，在命令前加`DOCKER_CONTEXT=default`。宿主只运行Python标准库调度脚本，PyTorch和训练依赖来自固定镜像。脚本里的`/lib64`前缀遵循原机规则；目标机若库布局不同，应先核对解释器环境，可设置`PYTHON_BIN`指定宿主Python。

显存较小的目标卡不会按默认门槛启动。若确认要改资源门槛，可显式设置`MIN_TOTAL_GPU_MIB`和`MIN_FREE_GPU_MIB`；这只改变资源检查，不改变模型或batch，也不保证较小显存能跑通。未经实际全尺寸验证，不把该环境称为已验证。

## 多张GPU各跑一组

每组在一个终端执行下面形式的命令，分别指定job、GPU及互不重叠的4个CPU。四组必须共用同一份解压目录，逐job文件锁防止同目录重复启动。

```bash
LD_LIBRARY_PATH="/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" python3 \
  handoff/continue_lr_control_handoff.py \
  --batch "$PWD" --job loop-lr5e-4 --gpu 0 --cpus 4,5,6,7 \
  --wandb-token-file /你的私有目录/wandb-token \
  --docker-context lm-memory --execute
```

其余job替换为表中的三个名称，分配其它GPU和CPU。不运行`launch_lr_control.py --execute`或旧`resume_lr_control_gpu7.py`，它们不是本次交接入口。

## 怎样判断跑完

脚本按阶段退出码复用已完成的CUDA检查、preflight、32步短训及独立重载；仅执行剩余阶段。交接时普通Loop 5e-4已到正式训练前，普通Loop 1e-3待短训，两组DiscoLoop待preflight。遇到已有正式训练claim或失败退出码会拒绝重跑，不清空输出，不自动调整配方。

每组正式训练后自动进行新容器七池逐token重载、曝光核对及W&B完整JSON云端核验。进度在`jobs/<job>/training/status.json`，曲线链接在`jobs/<job>/wandb-train.json`，独立监控状态在`handoff/<job>-status.json`。完成要求300000步、`result-audit.json`和`result-cloud-audit.json`均通过；仅训练结束不算全部完成。

完整结果保持在该批目录，之后将四组的`jobs/<job>`、`execution/<job>`和`handoff/<job>-*.json`交回汇总，原机两组和已有1e-4对照一并比较。不要覆盖原机其它组或冻结文件。一个图/初始化的结果不证明通用效率；DiscoLoop计算成本单独报告。
