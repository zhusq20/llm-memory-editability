# 交给另一台机器：只运行初值 0.95 的两个更新门

2026-10-17 用户授权另外两条各 300,000 步的正式训练，并要求提供可手动交接的代码包。本机负责初值 **0.90** 的固定门和学习门；另一台机器只运行本文件的 **0.95** 两臂。不要在另一台机器再次提交本机的 0.90 两臂，也不要重训原论文门 P。

交接包路径：`results/paper-update-gate-multihop-v1/handoff/paper-update-u95.tar.gz`。包中不含 W&B 凭据。完整配置为 [paper-update-gate-multihop-u95-v1.json](../../../configs/paper-update-gate-multihop-u95-v1.json)，批次名为 `paper-update-gate-multihop-u95-v1`。

## 比较什么

本轮输入为 $X$，完整 Transformer 输出为 $H=F_\theta(X)$。沿用原 P 的软解码、绑定词表再编码、RMS 归一化，以及可学习的论文注入门 $g_P$；新增门只控制连续更新 $H-X$：

$$
X_{\mathrm{next}}=[H+g_P(H)q(H)]+(1-u)(X-H)
=X+u(H-X)+g_P(H)q(H).
$$

原 P 的注入通道只计算一次。**仅在非末轮执行交接，末轮直接从原始 $H$ 读答案。** Transformer 内部的 attention、MLP、归一化和残差均沿用原实现。

| 条件 | 在哪里执行 | 连续更新门 |
|---|---|---|
| 原 P | 只读复用已经完成的结果 | $u=1$ |
| 固定门 0.90／学习门从 0.90 开始 | 本机 | 已分配给本机，不在交接机器重复启动 |
| `fixed-update` | 交接机器 | 固定 $u\approx0.95$ |
| `token-update` | 交接机器 | $u(H)=\sigma(w_u^\top\operatorname{LN}_0(H)+b_u)$，从约 0.95 开始 |

这里 $\operatorname{LN}_0$ 不含仿射参数；新门每个 token 输出一个标量，参数在轮次和位置之间共享。固定门和学习门的初始张量与前向运算完全配对，仅学习门的张量参与训练。初始化为 $w_u=0$、$b_u=\log(0.95/0.05)$；原论文门仍独立学习，没有被固定或重新参数化。

若 $H=X+\Delta$，0.90 和 0.95 分别接受 **90% 和 95% 的本轮更新**。等价地，在连续状态插值中保留旧输入的比例分别为 10% 和 5%。比较两个初值可以检查观察到的效果是否只出现在单一更新幅度；每个初值内部再用学习门对固定门，检查状态相关调节是否有帮助。这两个值都不声称最优，也不与原 P 初始计算完全相同。新增门参与全局梯度裁剪，初始共同梯度一致不代表后续优化轨迹一致。

## 固定配方和预算

除上表明确的更新门设置外，沿用现有多跳配方：

| 项目 | 设置 |
|---|---|
| 数据 | 两个各 500 实体、10 关系的已观察图；10,000 条原子事实，加图 A 各 10,000 条两／三／四跳组合，共 **40,000** 条训练查询 |
| 评测 | 原 11 个面板、**43,000** 条查询；图 B 三跳、300k 终点、推理四轮的全词表答案准确率为主指标 |
| 模型 | **1 层 × 4 轮**，宽 768，12 头，dropout 0.1，绑定输入／输出词表权重 |
| 训练 | batch 512，AdamW，lr $10^{-4}$，wd 0.1，warmup 2,000，各 300,000 次更新 |
| 种子 | 世界 2026101601；初始化 2026101602；样本流 2026101603；dropout 2026101604 |
| 曲线节点 | 0／1k／8k／16k／32k／64k／128k／256k／300k |
| 终点评测 | 同一终点的 R1–4 完整推理网格；保留答案、答案加 EOS、NLL 和原有条件指标 |
| 本次远程预算 | **2 × 300,000 = 600,000** 次新增更新；原 P 新增更新为零 |

world SHA256 为 `e132eaf6bba5786bbf05f6f36c1b7383f698e93ee0dee313f568f6d50a77026f`；panels SHA256 为 `c991cebeeeec05639be87d981cd20031016f47965c43fed8bf5cc55579c3b468`。不得换成旧 25,000 条训练查询版本。没有额外 KV、辅助损失、中间真值监督、五跳数据或额外轮数／学习率扫描。

## 文件布局与运行环境

解包后，使 `--base` 指向包含以下项目的目录，例如 `/local/disk/paper-update-u95`：

```text
paper-update-u95/
  source/       冻结代码、配置和交接材料；容器中只读挂载到 /project
  data/         原输入及复用材料；容器中只读挂载
  output/       本次训练、评测、日志、权重和控制器记录；可写挂载到 /output
  freeze.json   镜像身份及源码、数据的文件哈希
```

`--mirror` 指向另一处持久目录，控制器定期同步输出并保留冻结源码和清单。不要把它指向旧实验结果目录。`--base` 应使用本机可写磁盘，以免将训练中的频繁写入直接放到慢速远端存储。

使用显式 `docker --context lm-memory`。交接机器须有下面的**同一镜像 ID**；交接包本身不假定已经带上 Docker 镜像：

```text
sha256:27b3ab677fb92aaf4e4cd953e3a87e953c4b164a0d4d4c4785dc9779851c11d2
```

主机运行控制器需 Python 3.10 以上、NumPy、Docker CLI、NVIDIA 工具与 `rsync`；模型依赖由固定容器提供。主机特定的 `/lib64` 修复只适用于项目入口指定的原机器，不直接照搬到另一台机器或容器。

选择两张**实际空闲**的 GPU。下方命令中的 `0 1` 仅为示例；控制器会核对 GPU 进程、容器占用、剩余显存和利用率，不抢占其它任务。每臂一个独立容器和一张 GPU，使用两核 CPU、16 GiB 主存；还需四个可用 CPU affinity 槽位。

该实际镜像与旧 P 的 `sha256:9643a7d6aeda1b4c9ac2ac50c2d84474b082d649a87e423bc6e0bdc0994097ee` 不同。配置记录此变更；不能将已有主要依赖相符记录称为跨运行环境数值等价证明。

## 手动启动

在交接机器用私有文件提供 W&B token，权限设为 `0600` 或 `0400`。命令只传文件路径；不要将 token 写入交接包、配置、命令行或日志。训练使用在线 W&B：团队 `zhusq20`，项目 `llm-memory-editability`，组名 `paper-update-gate-multihop-u95-v1`。

先在解包后的源码目录查看两臂和执行阶段。**不带 `--execute` 只打印计划，不启动训练**：

```bash
cd /local/disk/paper-update-u95/source
python scripts/launch_paper_update_gate.py \
  --base /local/disk/paper-update-u95 \
  --mirror /persistent/llm-memory-editability/paper-update-gate-multihop-u95-v1 \
  --config configs/paper-update-gate-multihop-u95-v1.json \
  --gpus 0 1 \
  --wandb-token-file /private/path/wandb-token
```

准备就绪后，在同一命令末尾加入 `--execute`。长任务可在持久终端中运行，或使用：

```bash
nohup python scripts/launch_paper_update_gate.py \
  --base /local/disk/paper-update-u95 \
  --mirror /persistent/llm-memory-editability/paper-update-gate-multihop-u95-v1 \
  --config configs/paper-update-gate-multihop-u95-v1.json \
  --gpus 0 1 \
  --wandb-token-file /private/path/wandb-token \
  --execute \
  > /local/disk/paper-update-u95/controller.log 2>&1 &
```

路径、GPU 编号和私有凭据路径应替换为交接机器实际值；保留明确的 **u95 配置名**，不要依赖启动器默认配置。执行时会核对冻结文件和镜像身份，并检查两张 GPU；计划输出本身不代替这些执行时检查。

按用户最新要求，阶段为每臂 **`train` → `grid`**：直接运行正式训练，然后做终点 R1–4 评测；不执行 `preflight`、`smoke`、`smoke-audit`、`audit`、`grid-audit` 或独立云端结果审计。不得把“不执行独立审计”写成“审计通过”，也不因早期多跳成绩低而提前筛掉方案或自动追加步数。旧协议中的工程门槛由本次明确执行要求更新。

## 查看进度与保留结果

`output/controller-state.json` 记录两臂的当前阶段；`output/runs/<arm>/status.json` 和 `learning.json` 记录步数与曲线，`tracking.json` 给出 W&B 运行链接；`output/execution/<arm>/<stage>/` 保留命令、日志和退出码。训练终点写 `complete.json`，终点评测写 `grid.json`，最终控制器状态明确保留 `independent_audit_not_requested`。

输出和已结束容器记录均保留。启动器检测到已有 `controller-process.json` 会拒绝自动重启；发生失败时先读取已有状态、日志及实际容器，不删除旧记录后重复启动。结果解释仍限于一个已观察开发图、一个初始化；新的两条远程训练尚不能被写成已完成结果。
