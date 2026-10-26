# GPU8普通Transformer与Loop并行修订

2026-10-15用户明确授权：显存足够时，把Loop复核及后续训练放到GPU8，与普通Transformer并行。23:54 UTC已启动Loop共同阶段独立重载，普通Transformer原连续训练继续运行。原模型容器ID及PID2836453保持不变；新宿主PID3036041。[实际核验](../../../results/ttcd-speedup-v1/gpu8-parallel-20261015/verification.json)、[工程清单](../../../results/ttcd-speedup-v1/gpu8-parallel-20261015/source-manifest.json)、[Loop复核追踪](https://wandb.ai/zhusq20/llm-memory-editability/runs/zbqx9qy9)。

启动前GPU8只有本项目普通Transformer，显存占用22253MiB、空余74998MiB。旧DiscoLoop单事实TTCD训练记录的峰值分配约11.16GiB，Loop共同训练约1.72GiB；这些仅作资源估计，不代替Loop连续训练的实测峰值。采用新阶段至少32768MiB空闲的准入条件。23:55 UTC实际共享后占23400MiB、余73851MiB，Loop复核已有CUDA进程和在线W&B记录，尚未完成复核。后续Loop共同门槛/云端复核、原定stream10000训练及原评测均自动接在GPU8，不重新训练已完成阶段。

共享范围仅为原`ttcd-architecture-comparison-v1`普通Transformer与Loop两个架构臂，每架构最多一个运行阶段；核对冻结源码只读挂载、原CLI、运行用户、GPU UUID、容器及全部CUDA进程。未知进程阻止新阶段准入。两臂CPU仍分别40–43和44–47、每臂4线程与96GiB主机内存。两个线程共享同一GPU排他锁的独立租约，最后一个阶段结束才释放锁，避免与独占调度器同时分配；已运行模型不迁移或重启。GPU4任务及两批新增一步/merge控制器不变。

固定镜像27项检查及Ruff通过，覆盖显存不足、未知/重复进程、源码/命令/用户/设备不符拒绝、双架构共享锁、外部锁保留、CLI不变和原阶段接管/容器复制。沿用已核验的宿主恢复引擎及相同数值扩展清单`d10365ad67b980a2ba69c46f1f8dd9c4bc4ac93cf6d92f5b9bc4b891653d74fa`；没有短训、额外模型更新或科学配方修改。共享耗时不作为独占GPU效率证据。

恢复入口为`/ossfs/workspace/llm-memory-editability/ttcd-speedup-v1/gpu8-parallel-20261015/`，先读`source-manifest.json`、`architecture/`中的`snapshot.json`、`launch-process.json`、`claim.json`和`controller.log`，再核对原批最新`controller-process-speedup.json`/`schedule-amendment.json`和实际容器。旧GPU9固定或独占空闲回退仅保留历史，不重启旧宿主。原配方、权重、训练数据、监督、预算、截止时间与各阶段独立审计保留；一步部署与merge仍等待对应stream终点。
