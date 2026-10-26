# u=.95 两臂本机执行记录

**2026-10-17 15:45 CDT完成：两臂各300,000步训练及R1–4终点评测完成，控制器和四阶段均退出0，GPU5/7释放。** 图B三跳固定/学习门为39.64/78.18%，四跳为0/14.30%；原始数据、源码、日志及权重已镜像到持久目录。曝光与旧P逐元素一致，终点与R4记录一致；未执行独立重载或云端终点审计。见[完成汇总](../../../results/paper-update-gate-multihop-u95-v1/completion-summary.json)。不重启本批；以下启动过程保留为历史记录。

2026-10-17 CDT，用户要求在本机建立镜像后自动、直接启动主实验。本次执行批次为 `paper-update-gate-multihop-u95-v1`，固定更新门与学习更新门各 300,000 步，共 600,000 次新增更新。GPU5 运行 `fixed-update`，GPU7 运行 `token-update`；每臂两核 CPU、16 GiB 主存，CPU 分别为 88–89、90–91。

两条独立容器已启动，阶段均为 `train`；结束后原控制器自动执行终点 R1–4 的 `grid`。按用户要求，不执行额外短训或独立重载/云端结果审计。未执行的审计不得记为通过。当前没有 300k 终点结论，也没有追加第三条实验。

启动核对时两臂均已完成 4,000 步，损失与梯度范数有限，W&B 在线记录已建立，实际镜像、GPU 和 CPU 分配符合本次修订。两臂初始模型摘要相同；见[启动快照](../../../results/paper-update-gate-multihop-u95-v1/execution-at-launch.json)。

## 镜像与冻结材料

交接指定镜像 `sha256:27b3ab677fb92aaf4e4cd953e3a87e953c4b164a0d4d4c4785dc9779851c11d2` 在本机缺失。用户明确授权本机建立镜像后直接启动，因此从已安装的原 P 镜像 `sha256:9643a7d6aeda1b4c9ac2ac50c2d84474b082d649a87e423bc6e0bdc0994097ee` 构建专用镜像 `sha256:b7933a57d6a4702700b6b2578fe0cc697f91b2080ef5e18d8fb916443281eacd`。构建不下载或升级依赖，仅加入批次标签、固定依赖清单并执行版本断言。

PyTorch 2.8.0+cu128、Transformers 4.57.3、NumPy 2.2.6、W&B 0.30.0、huggingface-hub 0.36.0 的版本断言通过。新镜像与交接镜像 ID 不同；没有新增跨环境数值等价验证。首次镜像构建因 BuildKit 将原始镜像 ID 解释为仓库名失败，尚未运行实验；第二次使用核对过 ID 的本地标签构建成功，两份日志均保留。

原交接包 SHA256 与全部 728 个冻结文件核对通过。原包及解包副本保留；新执行副本仅修改 u95 配置中的授权说明、运行镜像及环境限制，另外 727 个冻结文件字节不变。科学配置、模型代码、40,000 条训练查询、43,000 条评测查询、配对初始化、样本流、预算与评分均不变。新清单 SHA256 为 `a97ac25e7ef55c8fd17db626d5322566de2692b2e000e033259652954bd449dc`。

镜像构建来源见 [Dockerfile](../../../containers/paper-update-u95/Dockerfile)，全部差异与哈希见[运行修订](../../../results/paper-update-gate-multihop-u95-v1/runtime-amendment.json)。

## 进度、输出与恢复

- 本机执行目录：`/tmp/paper-update-u95-sn4622129202`，控制器 PID `1305177`；日志为其中的 `controller.log`。
- 持久镜像目录：`/ossfs/workspace/llm-memory-editability/paper-update-gate-multihop-u95-v1`，原控制器每分钟同步一次输出，并保留冻结源码归档。
- 仓库入口：[控制器状态](../../../results/paper-update-gate-multihop-u95-v1/execution/controller-state.json)、[进程记录](../../../results/paper-update-gate-multihop-u95-v1/execution/controller-process.json)、[提交记录](../../../results/paper-update-gate-multihop-u95-v1/submission-process.json)。
- W&B：[固定 .95](https://wandb.ai/zhusq20/llm-memory-editability/runs/lh8fd92600)、[学习 .95](https://wandb.ai/zhusq20/llm-memory-editability/runs/lh8fd92601)。

恢复时先读上述记录及 `runs/<arm>/status.json`、`execution/<arm>/train/` 日志，核对 `docker --context lm-memory` 实际容器。不要再次运行交接命令，也不要删除控制器锁、进程记录或旧阶段目录后重复启动。原 P 和另一机器的 u=.90 两臂未重启。
