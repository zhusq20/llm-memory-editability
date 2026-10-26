# 四张空闲GPU的续跑交接

当前批次`ce-cross-sample-ablation-v1`。用户将安排另一台服务器四张空闲GPU；本机等待调度器已停止，当前没有本批运行容器，防止重复提交。远端尚未启动。这是原批次续跑，不是新实验。

## 四个任务：每张卡一个

|arm ID|固定架构|学习率|Adam|元素截断|未完成阶段|
|---|---|---:|---|---|---|
|`loop-carry-adam`|Loop 4层×2轮|3.2e-4|跨事实保留m、v和计数；换流重建|theta0±0.0005|science → audit → cloud|
|`discoloop-lower-lr`|DiscoLoop 4层×2轮|4e-5|逐事实重建|theta0±0.0005|audit-gate → science → audit → cloud|
|`discoloop-carry-adam`|DiscoLoop 4层×2轮|8e-5|跨事实保留m、v和计数；换流重建|theta0±0.0005|audit-gate → science → audit → cloud|
|`discoloop-no-bound`|DiscoLoop 4层×2轮|8e-5|逐事实重建|取消元素截断|audit-gate → science → audit → cloud|

四个开发写入检查均已完成，最终候选均32/32当前事实正确。DiscoLoop低学习率的2e-5仅27/32，按预定规则使用4e-5，禁止重新选择或重新运行gate。Loop carry的开发独立审计也已完成，另外三个分别还需重载64/32/32个开发状态，无优化。

每臂正式复用同一32对/64事实正逆流，每事实20步，总2560更新。四臂合计还需**10240正式更新**；基础/外层/新组合监督均0，不追加开发或工程优化。所有臂只改唯一物理MLP，原教师、KL10、clip1、Adam(0.9,0.999)/eps1e-8/wd0、FP32/eval/dropout0/TF32关闭不变。每个流起点恢复原300k检查点；同一流内参数持续。每臂保存128个新MLP状态及129读出，随后新容器独立重载全部129状态并重算完整统计。carry臂还核对两个流终点完整Adam状态。

## 已完成内容直接复用

`loop-lower-lr`与`loop-no-bound`的gate、science、独立审计、W&B核验全完成，不能重跑。两个架构的prepare各46工程更新并精确恢复已完成，不能重跑。普通Transformer及Loop/DiscoLoop原配方控制来自完成批次`ce-cross-sample-double-new-v1`，不能重启其训练。

Loop低学习率终点新事实105/128、双新0/64、旧原子512/512、旧组合689/704；Loop取消截断为101/128、11/64、44/512、7/704。后者两原子及组合联合正确10/64；不能只看新知识保持而忽略旧知识损失。

## 持久材料与环境

批次根：`/ossfs/workspace/llm-memory-editability/ce-cross-sample-ablation-v1`。

- `cross-server-handoff/manifest.json`列四个arm的学习率、剩余阶段、开发结果SHA、全部外部检查点/教师/图数据SHA与输出根。
- `cross-server-handoff/frozen-source.tar`为可直接传入Docker根目录的`frozen/`归档，包含全部454个只读源码文件及清单，没有凭据。SHA见manifest。
- 原源码：`source-v1`，清单SHA `79a5cf4cf4870527c2d6653e39d86a647ea991ae8bcf4204208f4eb36154ad83`。
- 固定LM镜像：`sha256:9643a7d6aeda1b4c9ac2ac50c2d84474b082d649a87e423bc6e0bdc0994097ee`，PyTorch2.8.0+cu128。
- 原配置`configs/ce-cross-sample-ablation-v1.json`及其`base_plan`、panel均在归档中，不改源码或科学配置。

若远端看不到同一OSS目录，须先按manifest取得相同字节的输入，以及本批prepare/gate/已完成audit-gate和全部开发状态。不能只复制基础模型便跳过开发审计。最终报告还要原控制的完整记录及其完成/独立审计/云端收据，和本批两个已完成臂。无需复制原控制的全部MLP状态；本批三个待审gate状态必须可访问。

远端使用经核实的LM Docker环境，不套用其它项目环境，不复用本机GPU UUID、CPU80–95等亲和设置。四卡各独立容器、建议各4核/48g内存，至少为每臂新增状态预留约10GiB持久空间，整体留足已有开发状态和下载/日志余量。先做真实CUDA矩阵计算/反向/同步零优化检查，不能只枚举GPU；本机`/lib64`修复不自动套到新主机。

输入父目录按原绝对路径只读挂载，并将项目持久父目录挂到`/persist`写输出；不要直接绑定0700子目录而触发root-squash。运行用户需能读取原私人材料（原容器UID1015:GID1016），先核对远端所有权。源码通过归档部署到`/frozen`，根所有/文件0444/目录0555。宿主导入与容器均设置`PYTHONDONTWRITEBYTECODE=1`；设置`PYTHONPATH=/frozen/src:/frozen/scripts`及`LM_SOURCE_MANIFEST_SHA256`为上述哈希。使用远端已有W&B凭据，只读挂载`/run/secrets/wandb`，不把密钥加入归档或日志。

## 每个阶段的调用与防重复

不要直接重启本机`launch_ce_cross_sample_ablation.py`或GPU7共享调度入口，它们有已提交锁和本机资源假设。远端只需一个薄调度层，使用原冻结runner，按上表阶段逐项执行；每阶段独立新容器，输出沿用原arm目录。启动前核对`cross-server-handoff.json`及实际Docker，并针对arm记录跨主机执行锁/调度修订；已有launch/exit/result须核对后复用，失败记录保留，禁止自动重复科学写入。

容器内通用命令（替换`STAGE`、`ARM`，不要改其它配方）：

```text
python scripts/run_ce_cross_sample_ablation.py STAGE \
  --arm ARM --plan configs/ce-cross-sample-ablation-v1.json \
  --root /persist/ce-cross-sample-ablation-v1/ARM/output \
  --device cuda:0
```

`cloud`阶段用`--device cpu`。它核验本臂各W&B运行与该架构原prepare：finished、优化计数、完整result.json下载SHA一致。团队`zhusq20`、项目`llm-memory-editability`。正式科学2560更新、审计0更新，开发候选数及计数沿用已有收据。

四臂完成后，连同两个已完成臂核对全部完成收据，再只执行一次原冻结报告器：

```text
python scripts/report_ce_cross_sample_ablation.py \
  --plan configs/ce-cross-sample-ablation-v1.json \
  --batch /persist/ce-cross-sample-ablation-v1 \
  --out /persist/ce-cross-sample-ablation-v1/report/output
```

报告器重算同架构变体与原配方的配对差值/32事实对bootstrap、共同前提覆盖、刚写/保持/双新/旧池，并核验总报告W&B完整JSON。将四臂completion、report/result.json、report/report.md及report/cloud-verification.json交回，才可宣布全批完成。只一张已观察图、每架构一个已有初始化；无通用架构优劣结论。
