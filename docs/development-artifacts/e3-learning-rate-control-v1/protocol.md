# e3-learning-rate-control-v1：Qwen3-1.7B的E3改用1.5e-4

2026-10-20用户授权“接下来的实验”（[roadmap §4](../../roadmap.md#next-experiments)）。

## 问题

[知识调用接口v2](../../results.md#knowledge-read-interface-v2)的E3中，Qwen3-1.7B的写入落在后段、为答案形式，Qwen3.5-2B的写入落在前段、能直接作第一跳。校准给两个模型选的学习率相差10倍（1.7B：Structural 1.5e-3、Synaptic 5e-4；2B：两臂都是1.5e-4），写入位置的差别可能来自学习率而不是模型。

## 做法

- 原样运行v2的E3阶段（同一50条事实、写入顺序、教师回复、回放池、评测与按段消融），只把Qwen3-1.7B两臂的学习率都固定为1.5e-4，即Qwen3.5-2B校准出的值。
- 入口`scripts/run_e3_lr_control.py`只替换W&B记录器，使本批有自己的运行，不续写v2的原运行。
- 输入：开发批`knowledge-read-interface-development-v1/runs/qwen3-1.7b`（只读）；输出：持久区`e3-learning-rate-control-v1/runs/dev/qwen3-1.7b/`。
- 判读：按段消融看去掉哪一段后U0降到0（写入所在），并比较U2直接回答的答案类型（正确/答v/答主语的国家）与U3。若改用1.5e-4后写入移到前段，写入深度随学习率变化；若仍在后段，随模型变化。

## 执行

- 容器`e3lr-qwen3-17b-dev`（GPU5），镜像`lm-memory-read-interface:20261019`，按用户偏好未做源码冻结与CUDA预检。
- 汇总：`scripts/summarize_e3_lr_control.py` → `results/e3-learning-rate-control-v1/summary.json`。

结果见[结果汇总](../../results.md#e3-learning-rate-control-v1)。
