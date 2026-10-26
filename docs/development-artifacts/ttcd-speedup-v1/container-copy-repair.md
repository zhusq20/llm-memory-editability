# TTCD启动修复：逐容器复制扩展

2026-10-15 UTC。GPU8普通Transformer和GPU9 Loop的原生300k已完成，七池独立重载均通过。普通模型基础云端核验通过；Loop云端核验第一次下载审计JSON时404，后续只读检查确认相同文件已可见。两条TTCD尚未更新。原训练终点、云端失败、宿主退出与普通模型适配启动claim全部保留。

普通模型的适配容器在创建前退出126；只创建不执行的检查定位为Docker守护进程不能穿越加速根目录700。原加速清单600，内部源码755/644。自动审批拒绝755/644及更窄711/644的持久权限放宽；两次命令均未执行。本修复保持这些宿主权限，采用审查指出的非持久替代：每个后续容器先create，宿主以tar字节流复制原已验证的10文件扩展和原清单，再start。容器内文件root:root 0444、目录0555，对运行用户1015只读；结束后随容器释放。原冻结源码挂载、输出隔离、全部科学参数和CLI不变，实际create/copy/start及归档SHA另存收据；扩展入口仍在进程内核验每个原源码哈希。宿主凭据路径和权限不动，凭据不进入归档。

5项零模型更新检查覆盖：原始文件/清单字节与只读容器元数据、宿主700/600保留、原GPU/科学参数保留、copy先于start、容器ID和非TTCD Docker调用保留、改字节/缺挂载拒绝。没有短训、吞吐探测或科学重训。

架构恢复只跳过已有成功凭据的阶段：普通模型从适配preflight继续，Loop重做基础cloud-audit后继续。两个已知失败stage目录及Loop failure JSON移入`failed-attempts/`，旧容器保留；新尝试使用`-repair-v1`，不覆盖旧日志。重复的仅为零更新云端核验/未启动的preflight，不重复基础训练或审计。旧失败阶段截止时间保留，无自动重试；其他未来阶段按原48h规则。

GPU4原单事实/连续训练不重启，仅在记录宿主STOP快照后替换宿主控制器，使未来阶段也使用容器复制。补充评测控制器同样仅替换等待宿主，prepare不重跑，原等待截止时间和父两组完成屏障保留。当前模型ID/PID、原配方及全部正式评测覆盖不变。

恢复入口：`/ossfs/workspace/llm-memory-editability/ttcd-speedup-v1/container-copy-repair-20261015/`内`source-manifest.json`、`handoffs/<target>/`、`transport/`、`failed-attempts/`，以及各原批最新`schedule-amendment.json`/controller-process指针。原加速源码和数值证明不改；诊断28%/12%仍不是正式训练实测。
