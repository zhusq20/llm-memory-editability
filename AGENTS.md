# 项目工作入口

在本项目中开始分析、研究讨论、实验设计、代码修改或文档写作之前，先完整阅读并遵循 [项目研究原则](.agents/skills/llm-memory-research/SKILL.md)。新会话或上下文恢复后，如果这些原则不在当前上下文中，应先重新读取。

随后按任务需要阅读 `README.md`、`CONTRIBUTING.md` 及相关研究计划与实验协议。具体实验的冻结契约和历史结果仍按对应文档执行。

用户已要求后续新实验默认启用 Weights & Biases，账号/团队 `zhusq20`，项目 `llm-memory-editability`。采用[默认配置](configs/experiment-tracking-defaults.json)，启动训练时一并启动记录进程，向用户提供项目或运行链接；记录学习曲线、配置、步数、曝光、耗时与GPU资源。已有冻结训练保持原样，不因这项未来偏好追溯修改。凭据仅保存在本机凭据文件，不写入仓库、日志或冻结配置。

2026-10-04 用户明确：LM Memory Editability 与 D157/OpenTinker 是不同项目。LM 新实验使用专属 `docker --context lm-memory`（socket `/run/docker-lm-memory/docker.sock`，data-root `/workspaces/docker-lm-memory`），不要套用 D157 的 context、容器或依赖环境。相同依赖共享固定 LM 镜像，每个实验独立容器，指定资源与持久输出；本轮 GPU 为2–5。环境恢复记录在 [本轮容器目录](https://huggingface.co/datasets/zsqzz/llm-memory-editability/blob/main/docs/development-artifacts/memory-interface-next-v1/container/runtime.json) 与 [镜像归档说明](https://huggingface.co/datasets/zsqzz/llm-memory-editability/blob/main/docs/development-artifacts/memory-interface-next-v1/container/archive-manifest.json)。主机 Python 仍须启动前将 `/lib64` 放在 `LD_LIBRARY_PATH` 首位；容器使用已独立验证的库配置，不套用主机修复。
