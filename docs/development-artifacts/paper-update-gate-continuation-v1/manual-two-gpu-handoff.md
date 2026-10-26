# 两张卡手动接续：DiscoLoop P 基线与固定0.90更新门

## Material Passport

- 工作流：academic-research-suite / experiment-agent / run；本会话内执行调度交接。
- 日期：2026-10-17（美国中部时间）。
- 状态：本机两条学习门训练保留；以下两臂已从本机自动队列移出，等待用户在另一台服务器手动启动。
- 新增科学预算：0；移交原批次已授权的240万次更新，不从头重训。
- 验证：调度接管与容器身份检查已执行；另一台服务器尚未执行训练或环境验证。
- 用户随后明确只需要实验说明，不再追加测试、打包或远端操作。下文打包与启动内容仅供参考，交接包尚未制作；当前可直接使用原冻结目录的source、data和freeze.json。

## 两张卡各跑什么

| 目标卡 | 实验 | 精确 arm 名 | 起点 | 终点 | 本次新增更新 |
| --- | --- | --- | ---: | ---: | ---: |
| 第一张空闲卡 | 原论文再编码门 P，作为 DiscoLoop 基线 | `paper` | 300,000 | 1,500,000 | 1,200,000 |
| 第二张空闲卡 | 保留 P 再编码支路，另加固定更新门 u=0.90 | `fixed-update` | 300,000 | 1,500,000 | 1,200,000 |

两条同时运行，训练后分别自动评测推理1／2／3／4轮。P的非末轮传递为 `H + gq`；固定门为 `X + 0.90(H-X) + gq`。两者末轮均直接读取H。P是当前多跳实验中的论文门实现；不是把旧Wang任务的DiscoLoop检查点搬到新数据上。

本机继续跑 `token-update` 和 `token-update-half-lr`，不用在另一台服务器重复这两条。原0.95扩展队列也不由本说明启动。

这两个对照用于回答：当前学习门继续训练后的表现，相比同预算P基线和固定缩放究竟如何。另机只有这两条，不能改成新图、新初值或新学习率实验。

## 必须沿用的科学配方

- 批次：`paper-update-gate-continuation-v1`，原冻结配置 `configs/paper-update-gate-continuation-v1.json`。
- 原图种子2026101601、初始化2026101602、样本流2026101603、dropout种子2026101604；A/B各500实体、10关系。原40,000训练题和43,000评测题原字节复用。
- 宽768、12头、一个完整Transformer块循环四轮；原P再编码支路、答案/EOS监督保持不变。
- batch512，AdamW学习率1e-4、weight decay 0.1、原梯度裁剪及BF16/TF32设置；训练不重新warmup。
- 从各自300k的 `latest.pt` 恢复模型、全部Adam状态、计步、采样排列/游标、曝光和CPU/CUDA RNG。不要只加载模型权重，不要改用最佳评测节点，不要从第0步启动。
- 完整11池在总400k、500k、……、1500k评测；主指标固定总1500k、推理4轮、图B三跳。保留全部原子、两／三／四跳及训练池结果。
- 固定镜像：`sha256:27b3ab677fb92aaf4e4cd953e3a87e953c4b164a0d4d4c4785dc9779851c11d2`。同规格GPU不等于同软件环境；镜像不存在时先从原机导出并导入，不自动升级依赖或替换镜像。
- 每条独立LM Docker容器、独占一张实际空闲卡、2 CPU、16 GiB主存；W&B在线记录到 `zhusq20/llm-memory-editability`。保留此前免短训及免独立终点/云端审计的选择。

完整冻结清单SHA256：`da3f884bcb5ea084e29eecd875f08f432566285e30882392f2e785711656ae89`。
配置SHA256：`af4e6b763c9ad411170c3d58691c5c64acb89ef3a92455133c7383a9f43d1cf5`。
P父检查点SHA256：`8c0392b3d5ed460306b338be8e36d7b80a1796c13a0763633f4b9f8f363a30b0`。
固定门父检查点SHA256：`ccb7f1ad134c9587cf26ee66e39174efd263a77227f964feb42fa13d8e0cc43c`。

## 文件与队列所有权

原冻结输入位于本机 `/tmp/paper-update-gate-continuation-v1-execution/` 的source、data和freeze.json；持久执行目录保存source.tar.gz及freeze.json，父目录的inputs保存数据。尚未制作单独交接包。若之后需要打包，应包括原冻结source、原数据及全部父检查点、冻结清单、专用接收启动器，不包含W&B凭据和历史训练输出；保留全部父输入以满足原224文件清单，但只执行上表两臂。

本机旧调度器已退出，由只接管现有两条训练的监控进程接续其终点评测。P与固定门不会再在本机自动启动。交接凭证为原共享执行目录下的 `scheduling-handoff-v1/handoff-ready.json`。旧进程、旧状态、两条训练原PID和启动时间均保留在该目录；科学冻结文件没有改动。

原共享执行目录在当前机器的路径：

```text
/mnt/disk2_from_server2/siqizhu4/lm-memory-persistent/workspace/llm-memory-editability/paper-update-gate-continuation-v1/execution
```

如果另一台通过 `/ossfs/workspace` 挂载同一存储，改用该机对应路径。`--handoff-root` 必须指向这份真实共享凭证所在目录，不可复制到另一个独立目录后并行提交。接收端会在共享目录创建永久、排他的 `remote-launch-claim.json`，并拒绝已有输出，防止重复启动。

接收启动器设计为将新输出写到接收机本地交接目录的 `output/`，每分钟同步到共享执行目录的 `remote-handoff-v1/`；该启动器尚未在目标服务器运行。共享原 `runs/paper` 和 `runs/fixed-update` 入口尚未链接到新输出，启动交接时需记录实际输出路径以供扩展批次复用P结果。原冻结数据、配置和运行目录不覆盖。

## 手动启动

先确认目标服务器两个指定GPU没有计算进程或其它项目预约、`docker --context lm-memory` 可用、固定镜像已经导入。主机Python只用于调度，训练在固定镜像内执行。`/lib64` 的主机兼容修复只在原项目已验证的Alibaba主机适用，不盲目套到新服务器或容器。

将包解压到目标服务器的本地可写目录，例如 `/tmp/paper-continuation-handoff`。下面的GPU编号0／1和CPU编号0–3仅为示例，替换为该机实际空闲资源；凭据文件使用该机已有的私有文件，权限0600或0400。

```bash
lm_transfer_base=/tmp/paper-continuation-handoff
lm_shared_execution=/mnt/disk2_from_server2/siqizhu4/lm-memory-persistent/workspace/llm-memory-editability/paper-update-gate-continuation-v1/execution
lm_token_file=/path/to/private/wandb-token

mkdir -p "$lm_transfer_base"
tar -xf /path/to/continuation-two-arm-inputs.tar -C "$lm_transfer_base"

python "$lm_transfer_base/handoff/launch_continuation_handoff.py" \
  --base "$lm_transfer_base" \
  --handoff-root "$lm_shared_execution/scheduling-handoff-v1" \
  --mirror "$lm_shared_execution/remote-handoff-v1" \
  --gpus 0 1 --cpus 0,1,2,3 \
  --wandb-token-file "$lm_token_file"
```

上面只显示计划。核对两臂确为 `paper`／`fixed-update`、各从300k到1500k后，以同一命令加 `--execute` 在持久终端或后台运行。启动器逐文件核对原冻结包和镜像，保护两个Docker上下文的显卡预约，并且只运行train及终点grid。不要运行包内原四臂总启动器。

运行状态见 `remote-handoff-v1/controller-state.json`、`runs/<arm>/status.json`、`runs/<arm>/tracking.json`。训练进程开始且步数超过300k才记为续训启动；成功交接不等于已完成训练。若某阶段失败，保留其输出、共享claim和退出记录，先查原因，不删除claim后重跑。

若目标机器无法访问上述同一共享目录，先完成该目录映射；不要把它当作两个独立的新实验启动。目标卡号、目标镜像验证和实际远端启动均由用户手动完成，本机尚未代为执行。
