# GPU4同卡并行调度修订

2026-10-15，用户在双新事实评测已准备并等待父训练期间明确要求：“请你在4号卡并行跑”。本修订仅改变当前`ttcd-successful-base-v1`与`ttcd-double-new-eval-v1`的资源调度，授权范围是这两批自己的容器共享GPU4；其它项目/未知CUDA进程不在共享范围内。

共同训练保持当前容器与模型进程继续运行。原宿主控制器先暂停并核对确实处于common/train，再保存进程/状态/训练进度/claim/command快照，仅终止这个宿主控制器；新控制器接管同一容器监控，沿用原剩余共同阶段超时。已有preflight/train不重跑，训练权重、AdamW、RNG和样本流不重置。共同训练完成后的原独立重载/能力门槛/云端核验仍按原顺序；全部通过后，single与stream两个线程同时在GPU4运行原固定10k及后处理。

补充评测仍等待父single与stream全部本地及云端完成，随后两臂共享GPU4并行执行原新增preflight→evaluate→audit→cloud-audit。原补评宿主控制器仅在waiting_for_parent_completion且没有任何模型阶段claim时被替换，不重复source冻结、pure graph prepare或原评测。

父两臂CPU24–27/28–31、补评两臂32–35/36–39保持，各96GiB内存、独立容器/端口默认与输出。GPU4约97.9k MiB，已验证父两臂工程峰值各约11.2GiB；启动每个新阶段前须至少24,000MiB可用显存。顶层控制器持一个GPU4 cohort锁，自己两线程按授权共享；核对精确GPU UUID、只读冻结源码挂载、原科学CLI、容器属主以及Docker top与NVML PID对应，拒绝未知CUDA进程。运行中定期重核，不停止其它任务。共享条件下的墙钟和GPU利用率不作独占吞吐/效率证据。

原配置、冻结源码、训练步数、学习率、writer、正式面板、门槛和评分保持原字节；各批另存`gpu4-parallel-20261015/`调度amendment、新宿主控制器与资源模块副本/哈希、停止旧宿主记录、接管记录、失败/退出码。冻结配置中的原GPU8/9以及上一修订4/7作为历史值保留，实际资源以本修订、每阶段admission、Docker GPU UUID与W&B资源标签为准。

此修订不新增科学训练、不扩大方法网格，也不改变预检失败即停止、完整结果和独立重放要求。恢复优先读两批`gpu4-parallel-20261015/controller-process.json`、`controller-state.json`、根`schedule-amendment.json`及实际LM容器，原source/config/面板哈希继续核对，不重复启动已在运行的训练或prepare。
