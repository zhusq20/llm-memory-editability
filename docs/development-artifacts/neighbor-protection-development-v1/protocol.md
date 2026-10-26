# neighbor-protection-development-v1：按地址邻居保护的连续写入

2026-10-20用户授权“接下来的实验”（[roadmap §4](../../roadmap.md#next-experiments)）。配置：[`configs/neighbor-protection-development-v1.json`](../../../configs/neighbor-protection-development-v1.json)。

## 1. 问题与预测

[连续写入的损伤分组](../../results.md#sequential-write-interference-20261020)显示：取消截断后，写入的损伤落在与新事实同主语、同关系的旧事实上。由此在启动前写入roadmap的预测是：旧知识保护应针对写入的地址邻居，而不是随机抽样的旧事实。

预测：DiscoLoop在不加截断的同时，用地址邻居做KL保护，可以同时保留新事实、双新组合与旧知识。

## 2. 设计

与`ce-cross-sample-ablation-v1`的取消截断臂完全相同（同一native300k检查点、32对面板、正逆两条64步写入流、每条事实20步新Adam CE、KL系数10、梯度裁剪1、不加逐元素截断、每次写入后完整读出），只换每步KL保护的旧样本：

| 臂 | 每步旧原子 | 每步旧两跳 |
|---|---|---|
| discoloop-neighbor / loop-neighbor | 16条地址邻居 | 16条地址邻居 |
| discoloop-mixed / loop-mixed | 8条地址邻居＋原随机批的前8条 | 同左 |
| 对照（复用`*-no-bound`） | 16条原随机 | 16条原随机 |

- **邻居原子：** 一半是与新事实同主语的旧事实，一半是同关系的旧事实。
- **邻居两跳：** φ7.2已训练两跳中，一半首实体或桥实体是新事实的主语，一半第一或第二关系是新事实的关系。
- **教师：** native300k θ0在答案与EOS位置的输出分布，与原教师相同。
- **排除：** 七个旧评测池的每一行和混合题的每条旧伴随事实都不进入保护集。评测到的同主语旧事实因此都是未受保护的“兄弟”。
- **抽样：** 每条事实按种子加事实序号确定；池内不放回，用完后重洗。正逆两条流对同一事实使用相同保护序列。

## 3. 执行

- 2026-10-20，容器`np1-discoloop-neighbor`（GPU5）、`np1-discoloop-mixed`（GPU6）、`np1-loop-neighbor`（GPU7）、`np1-loop-mixed`（GPU2，与零训练的位置分解共用该卡），镜像`lm-memory-ce-ablation:20261017`，每臂约7分钟，退出码均为0。
- 每臂2560步科学更新，共10240步；保存每步完整读出与流终点状态。
- 按用户偏好未做源码冻结、哈希清单与单独CUDA预检；CPU契约测试（`tests/test_neighbor_protection.py`）与ruff通过。
- W&B：[discoloop-neighbor](https://wandb.ai/zsqzz/llm-memory-editability/runs/9g5leb1l)、[discoloop-mixed](https://wandb.ai/zsqzz/llm-memory-editability/runs/4vln8lvw)、[loop-neighbor](https://wandb.ai/zsqzz/llm-memory-editability/runs/rihaeme9)、[loop-mixed](https://wandb.ai/zsqzz/llm-memory-editability/runs/5ht3yof4)；本地显示同步完成，云端未独立核验。
- 输出在持久区`neighbor-protection-development-v1/<arm>/`，汇总由`scripts/summarize_neighbor_protection.py`生成到`results/neighbor-protection-development-v1/summary.json`。
- 本协议在DiscoLoop邻居臂终点出来之后整理成文；设计以启动前提交的配置文件为准。

结果见[结果汇总](../../results.md#neighbor-protection-development-v1)。
