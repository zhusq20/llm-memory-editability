# 序列依赖门评测：执行与恢复

2026-10-18已提交八臂，各100k步，合计800k科学更新。GPU2依次执行ListOps普通Loop、算术普通Loop、ListOps固定q、算术固定q；GPU5执行相应含u配对。每条完成训练后立即做预定轮数终点评测，全部八条完成才生成`comparison.json`和中文`report.md`。首对实际训练容器已运行，W&B在线，两臂均已记录至少1000个正式步且损失/梯度有限；终点尚未完成。

当前持久根为 `/mnt/disk2_from_server2/siqizhu4/lm-memory-persistent/workspace/llm-memory-editability/sequence-dependency-gate-v1-mountable-v2`，输出位于其`execution/`。仓库[执行链接](../../../results/sequence-dependency-gate-v1/execution)指向这一尝试。读取`science-process.json`、`science-state.json`、各任务`runs/<arm>/status.json`及真实LM Docker状态，不重复提交。控制器提交PID为4167624，以process文件与进程实际状态为准。

冻结238文件，当前freeze SHA256为`f5e720e5d035697a9c3c28a66c22bcc662e5fcca076b7071c3b38fe9a785b9d0`。原始首次目录继承umask077，Docker在创建容器时无法访问NFS上的source目录；两臂均在模型执行前失败，科学更新0。另建可挂载0755目录、0644文件的独立副本，没有修改旧目录权限。所有238份数据、配置、源码和来源文件字节一致，只在新freeze中记录修订来源；[修订记录](../../../results/sequence-dependency-gate-v1/amendment.json)和[原失败](../../../results/sequence-dependency-gate-v1/initial-attempt)保留。原数据生成不重复。

固定镜像CPU测试69项通过：数据28、作者函数一致性7、训练适配13、调度10、汇总11。首次调度测试夹具在只分配两CPU的测试容器中要求四CPU，修复模拟环境后通过，生产资源限制保持。CUDA完整尺寸每臂12项，共96项通过；两任务四臂主干、世界与评测池身份一致。各一次工程更新被恢复，另有CUDA graph捕获准备的零学习率优化调用并恢复；未进行32步短训，没有新增独立重载或云端终点审计。

当前曲线：[ListOps普通Loop](https://wandb.ai/zhusq20/llm-memory-editability/runs/dg2889f0264b38bf)、[ListOps仅u](https://wandb.ai/zhusq20/llm-memory-editability/runs/dga798ad65db3065)。其余六臂在进入训练后产生各自`tracking.json`。本批与旧40k多跳两臂独立，合计占用本用户GPU0/1/2/5四张，不关闭或抢占其他任务。
