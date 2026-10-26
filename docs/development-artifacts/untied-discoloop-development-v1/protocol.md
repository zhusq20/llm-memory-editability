# 标准Transformer＋经过词表的交接（untied-discoloop-development-v1）

## 状态

用户2026-10-20授权，作为主论文第4部分自己的“深度错、形式对”一格（[roadmap §3.4](../../roadmap.md#architecture)）。同日启动，单臂300k步。

## 问题

只学过原子的事实要作第二跳，需要两个条件：
- **深度：** 第二跳的事实存放在桥实体出现的深度；
- **形式：** 桥实体以入口认得的实体形式交接。

DiscoLoop 4×2两者都有：共享参数提供深度，经过词表的交接提供形式。本批保留经过词表的交接、去掉共享，问只有形式对时会怎样。

## 设计

- **模型：** 8个参数独立的GPT块，在第4块之后加一次与DiscoLoop完全相同的软解码再编码通道：`h ← h + RMSNorm(softmax(ln_f(h)Eᵀ)E)`，系数1、温度1；最后一块之后没有通道。若把第5–8块的权重复制成第1–4块，前向与DiscoLoop 4×2逐值相同（契约测试）。
- **其余全部沿用普通Transformer 8×1基础（`ttcd-architecture-comparison-v1`）：** 同一初始权重（`transformer.pt`，各层独立初始化，模型哈希`11b1c725…`）、同一Wang图与七个评测池、φ7.2、wd 0.1、lr 1e-4、warmup 2000、batch 512、样本流种子792101、dropout种子20261012、300k步、只监督答案与EOS。
- **代码：** [模型](../../../src/llm_memory_editability/untied_discoloop.py)、[运行器](../../../scripts/run_untied_discoloop.py)（复用原生`run_supervised_alignment.py`的训练与独立重载，只在本进程内替换模型类、架构标记与通道FLOP估算）、[启动器](../../../scripts/launch_untied_discoloop.py)、[配置](../../../configs/untied-discoloop-development-v1.json)、[测试](../../../tests/test_untied_discoloop.py)。

## 对照（复用，不重训）

| 模型 | 深度 | 交接形式 | OI/IO/OO |
| --- | --- | --- | --- |
| 普通Transformer 8×1 | 只在训练用途的深度 | 无 | 4.67/3.50/0.20% |
| 普通Loop 4×2 | 共享 | 隐藏状态 | 34.30/43.73/3.29% |
| DiscoLoop 4×2 | 共享 | 经过词表 | 99.20/97.67/95.66% |

## 预测（看结果前写定）

OI的第二跳是参加过组合训练的ID事实，后半网络在组合训练中学会了存放它们；OI只要求OOD的第一跳在前4块完成并以实体形式交接。IO与OO的第二跳是只学过原子的OOD事实，原子训练只把它们写在主语作为输入词的前段。

- **P1：** OI明显高于普通Transformer（4.67%），接近DiscoLoop——形式修好了。
- **P2：** IO、OO保持接近普通Transformer的低水平，远低于DiscoLoop——深度没有修好。
- 若IO/OO也接近DiscoLoop，说明在这一设置中经过词表的交接本身就让后段学会了存放只学过原子的事实，A1需要修改。
- 若OI也不提高，说明交接形式在不共享参数时也无法生效，A3需要修改。

主指标为固定300k终点的七池答案＋EOS准确率，同时报告学习曲线与独立重载。

## 执行与验证

- 按用户偏好未做源码冻结、哈希清单与单独的CUDA预检；CPU契约测试4项（复制权重后与DiscoLoop逐值相同、通道位置、非法形状、运行器替换）与ruff通过。
- 镜像`lm-memory-ce-ablation:20261017`（普通Transformer基础使用的镜像不在本机），GPU 2，容器`untied-discoloop-v1`；仓库只读挂载，输入取自持久区`ttcd-successful-base-v1/input-mount/data`与`ttcd-architecture-comparison-v1/input-mount/initial`，输出在持久区`untied-discoloop-development-v1/untied-discoloop-v1/`。
- W&B：`zsqzz/llm-memory-editability`，组`untied-discoloop-development-v1`，训练运行[zfm59tcn](https://wandb.ai/zsqzz/llm-memory-editability/runs/zfm59tcn)。
- **启动记录（两次失败都在0科学更新时发生，记录在持久区`execution/`）：**
  - `untied-discoloop-v1`：容器在默认bridge网络上，W&B反复解析域名失败；停止并删除，启动器改用`--network host`（本机此前的LM容器均如此）。
  - `untied-discoloop-v1-r1`：原生`initialize()`即使给了`initial_checkpoint`也会读取`source_run`，报KeyError；配置补上与普通Transformer基础相同的`source_run`。
  - `untied-discoloop-v1-r2`：2026-10-20 08:26 UTC起正常训练，约37秒/千步。

## 完成状态（2026-10-20 11:30 UTC）

300k训练与新进程独立重载（七池答案/EOS逐值）通过，容器退出码0，GPU2已释放。终点OI/IO/OO为36.17/19.47/6.19%，测试II 99.87%。P1（OI接近DiscoLoop）未成立，P2（IO/OO保持低）成立。结果与解释见[结果汇总](../../results.md#untied-discoloop-development-v1)，汇总在`results/untied-discoloop-development-v1/summary.json`。W&B云端未独立核验。
