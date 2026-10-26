# 一层四轮 DiscoLoop：在另一台机器手动启动

本批为代码交付，尚未运行模型检查、短训或科学训练。本机只做静态检查、无模型的启动器测试，以及文件复制和哈希核验。此前P/G/D/V批次及冻结源码不改动。

## 固定设计

原一层GPT完整执行四轮。所有原LN、attention、MLP、dropout、embedding和绑定读出保留。每轮内部为

\[
z=s+\gamma A_\theta(\mathrm{LN}_1(s)),\qquad
h=z+\gamma M_\theta(\mathrm{LN}_2(z)).
\]

q沿用原LN_f/绑定读出→温度1全词表softmax→绑定输入embedding→无仿射RMSNorm，eps=1e-6，不detach。只在前三轮交接：

| 条件 | γ | 交接 | 新参数（d=768） |
| --- | ---: | --- | ---: |
| F，复用 | 1 | h+q | 0 |
| R | .25 | h+q | 0 |
| N，主候选 | 1 | RMSNorm₀(h+q) | 0 |
| A | 1 | .5h+αq | 1537 |
| C | 1 | (1−α)h+αq | 1537 |

α=σ(w_hᵀLN₀(h)+w_qᵀq+b)，两个向量及偏置均零初始化、零weight decay，跨位置/轮次共享；LN₀无仿射，eps=1e-5。A/C初始化α=.5，初始前向和全部骨干梯度必须相同；门梯度允许不同。门分配不消耗初始化随机流。N先按原精度顺序形成h+q，再以FP32做额外RMS归一化；A/C在FP32混合后一次转回state精度。数值测试保留FP64。

R−F检验内部增量缩放，N−F检验相加后的归一化，C−A检验互补耦合。N−C只是完整方法比较。R的1/4是本项目L_ref=1、λ=1的循环轴参照，不是论文参考深度12的原数值配方。R不补偿学习率、不缩放q。

沿用同一40k唯一训练查询/43k评价题、宽768/12头、batch512、AdamW lr1e-4、warmup2000、wd.1及原答案/EOS监督。四条各300k，新增科学更新共120万；F新增0。工程短训4×32，预检4次各1次实际更新后丢弃，原CUDA graph构建热身亦恢复权重/Adam/RNG，均单列。没有1/2网格、R×N交互、额外辅助损失、KV、独立图或新知识阶段。

主指标固定300k、推理4轮、图B三跳全量答案准确率。所有原池、答案+生成EOS、条件覆盖、节点0/1k/8k/16k/32k/64k/128k/256k/300k及终点r1–4网格保留。不得选最好节点/轮数。单开发图/初始化不能建立通用机制或跨图结论。

## 搬运材料

打包入口只复制文件，不运行模型、Docker或W&B。在当前仓库中可手动生成新目录：

```bash
python scripts/launch_norm_handoff.py package \
  --destination /tmp/norm-handoff-bundle --tar
```

默认只读取现有原数据和原F，输出到指定的新目录。也可用`--input-dir`和`--baseline-root`选择已经搬运的相同材料。不可覆盖已有包；缺失/哈希不符时停止，不重训F。生成目录、`.tar`和`.tar.sha256`，将tar与校验文件复制到新机器。当前仓库的未跟踪源码会被显式打包，不能只搬运Git已提交文件。

包包括源码、配置、测试、原始数据、F的0/8k/300k检查点、独立审计和全量r1–4预测。`bundle.json`绑定所有文件SHA256，`baseline-reuse.json`绑定F复用检查。包不包含Docker镜像或W&B凭据。历史F的云端summary失败记录保留；本批完成后另外上传复用的本地审计证据，不改写父批次。

须另行搬运已使用的固定LM镜像。镜像ID：

```text
sha256:9643a7d6aeda1b4c9ac2ac50c2d84474b082d649a87e423bc6e0bdc0994097ee
```

可在原机器用`docker --context lm-memory image save --output lm-fixed.tar <上述ID>`导出，再在新机器用其本地Docker context的`image load --input lm-fixed.tar`导入。这两个命令只搬运镜像。不要调用旧实验启动器。镜像不一致时本入口拒绝执行，不自动构建/替换环境。

## 新机器操作

以下命令均由用户在**新机器**主动运行。需要Linux、Python≥3.10、可用的本地Unix Docker context、NVIDIA容器支持，以及所选的空闲GPU。每个运行独占一张GPU，要求空闲显存至少18,000 MiB；CPU和内存每条默认2核/16 GiB。可以只选1–3张卡，四臂依次排队；选四张卡则并行。不会抢占已有任务。

```bash
sha256sum -c norm-handoff-bundle.tar.sha256
tar -xf norm-handoff-bundle.tar
TASK_BUNDLE="$(pwd)/norm-handoff-bundle"
python "$TASK_BUNDLE/source/scripts/launch_norm_handoff.py" verify --bundle "$TASK_BUNDLE"
```

准备一个只包含W&B key的外部凭据文件，权限限定为本人可读。不要将key写进命令、源码、包或配置。然后替换下面四项：context、输出目录、GPU编号、凭据路径。context以新机器实际配置为准，不假定存在`lm-memory`；输出目录必须是新目录且位于包目录之外。若还有其他本地Docker daemon，重复添加`--reservation-context NAME`核查它们的GPU预留。

先预览，**不加`--execute`不会创建目录、容器或模型**：

```bash
python "$TASK_BUNDLE/source/scripts/launch_norm_handoff.py" run \
  --bundle "$TASK_BUNDLE" --output /persistent/norm-handoff-run \
  --docker-context YOUR_LOCAL_CONTEXT --gpus 0 1 2 3 \
  --wandb-key-file /secure/wandb-key
```

确认在新机器后，使用同一命令末尾加`--execute`主动启动，可通过终端会话保持工具让控制器持续运行。启动顺序为：

1. 校验全部文件/数据/F/原生源码/镜像，以及各Docker context均指向本地Unix socket。
2. 在无GPU容器运行模型契约测试和完整数据契约；随后在新GPU只读重放F的全量r1–4答案/EOS/NLL。设备差异导致重放不符即停止，不自动补训F。
3. 四臂预检、32步全尺寸短训和新进程全量重载/网格审计。全部通过才允许任何300k科学训练。
4. 四臂科学训练、全量独立重载、r1–4网格及W&B文件回读核验。
5. 在新机器CPU读取0/8k/300k检查点，使用旧诊断相同seed=2026101701、每池128题，记录输入/块后/交接后的二阶矩、通道方差、均值平方、原始carry/injection交叉项；区分查询前缀、答题位置和第一关系位置。N额外归一化之前的交叉项单独命名。它是FP32诊断，不替代BF16全量评分，也不用于选择节点。
6. 生成`comparison.json`、`report.md`、`activation-diagnostic.json`及云端回读审计。仅全部通过才生成`completion-manifest.json`。

正式训练通过W&B在线记录曲线/配置/曝光/步数/FLOPs估算/时间/设备。真实耗时与归一化/门O(d)额外操作单列，不宣称逐操作相同计算。容器保留便于审计；数据和F为只读挂载，输出独立持久化。控制器必须持续运行；不提交后台系统队列。

控制器用输出目录锁防止重复提交。`--resume`只跳过退出码0的完整阶段，并保留原包、设备和资源请求；已失败或缺少退出记录的阶段一律停止，不自动重试或从latest.pt续训。中断后先核对`controller-state.json`、`execution/*/*/command.json`/`exit.json`及本机真实容器状态。不要删除失败目录来绕过保护；需要恢复时先审查原因并另立记录。控制器中断时已有子容器可能继续运行，不能仅凭终端退出再次启动。

## 本次交付的验证边界

本机只运行启动器无模型测试、语法/风格检查与包哈希核验。模型契约已编写，包括原生前向/骨干梯度、A/C初始匹配、非恒定门公式、两个内部残差增量、因果性、只交接三次、可微性、门零衰减和重载；CUDA全尺寸预检和训练验证留待新机器主动执行。因此目前没有新科学结果，也不能声称CUDA已验证通过。
