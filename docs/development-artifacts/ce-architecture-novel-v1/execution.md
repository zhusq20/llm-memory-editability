# 新增事实整个MLP CE架构对照：执行记录

本批按用户2026-10-16授权补齐普通Loop和普通Transformer，复用[协议](protocol.md)中的两条原生300k基础与已完成DiscoLoop编辑。新增科学预算2560次编辑，基础/外层训练0；固定同32新键、256/253两跳及旧保持集合，不做累计写入。

## 当前恢复入口

持久根`/ossfs/workspace/llm-memory-editability/ce-architecture-novel-v1/`；最新宿主`controller-process-resume-v4.json`、源码`source-v4`、`schedule-amendment-v4.json`、各臂controller-state及实际LM Docker共同确定状态。已通过15项执行契约与Ruff；两架构完整CUDA预检均通过实际MLP改动、参数冻结、保存重载、基础精确恢复。独立保护输入与DiscoLoop逐字段相同，KL教师来自各自模型。两架构开发、正式编辑、独立64状态审计与完整云端核验均已完成，汇总云端JSON也通过；科学/审计容器均退出0，无本批运行容器。新增科学编辑2560、基础/外层训练0。

GPU7仅与配置中三个完整ID绑定的本项目merge审计有界共享，启动前拒绝外部CUDA PID、空余至少48GiB；两新臂各20%显存分配器上限。CPU分别80–83和84–87，各32GiB内存、独立私有输出卷；原任务不停止或搬移。共享耗时不作独占效率证据。

预检W&B：[普通Loop](https://wandb.ai/zhusq20/llm-memory-editability/runs/smz9j7fb)、[普通Transformer](https://wandb.ai/zhusq20/llm-memory-editability/runs/8hotczax)。正式各阶段链接见持久输出的`wandb.json`。

## 保留的零科学更新故障

1. 首次复制冻结源码时发现普通Transformer可能在旧开发组合池零正确，提前停止复制，保留`pre-submission-source-20261016`。保持率零分母明确记NA，排除最小值计算并保留覆盖；未增加样本、未改写为100%。此处理有独立契约。
2. 首次契约容器在启动前因根文件系统只读拒绝Docker cp，保留`transport-failure-v1`容器与收据。安装源码时允许容器根写入，实际冻结文件仍为root属主0444/目录0555，输入与凭据只读；未放宽主机私有目录。
3. 宿主加载冻结调度模块生成的pyc改变缓存哈希，在恢复复制时被抓到；保留原source及失败source-v2，禁用宿主字节码写入，并在source-v3之后仅排除生成缓存。所有继承科学.py逐字节验证，不改原缓存或源码。
4. source-v3的15项执行契约通过，Ruff导入分类因脱离项目上下文整理而失败。保留静态失败容器与源码，在完整项目上下文仅整理新增文件导入/格式；source-v4全部执行和静态检查通过。

上述阶段没有科学编辑；全尺寸预检每架构2次工程更新后精确恢复。原DiscoLoop批次、两条基础训练、当前其他审计均未重启。正式独立审计只加载保存MLP参数，不重新执行优化。


## 已完成的正式结果

普通Transformer/Loop/DiscoLoop的新原子均32/32，新→旧12/256、183/256、256/256，旧→新1/253、216/253、253/253。新原子与必要旧原子在三架构共同前提子集覆盖全部509道两跳，结果不被原子失败筛选解释。原正确旧原子保持100/99.99/99.90%，旧组合99.99/98.09/99.52%；完整分母及写前能力见[三架构报告](../../../results/ce-architecture-novel-v1/report.md)。

两个新架构开发候选均未达原最差池99.5%保持门槛，沿冻结fallback选择：Loop3.2e-4、Transformer5e-6，已有DiscoLoop8e-5；未追加更新。Transformer正式旧OO原正确0，保持NA；不能据旧保持高推定全部旧组合能力已建立。对应历史旧组合写前Transformer/Loop/DiscoLoop为103/253、183/253、252/253；基础能力与计算不同，当前整体优势不唯一定位更新传播机制。

最终科学source-v4 manifest：`61e2bb1b2be391104c8cb8b8bd2da6db6f350d18cc689b72724dd2c3331b7e09`。持久completion-manifest为complete，三架构汇总SHA256 `ecb980e0de74c01d7a13779abdf068b509170138e8994d4ce881c0db521c7966`。
正式W&B：[Loop](https://wandb.ai/zhusq20/llm-memory-editability/runs/886pq3eq)、[Transformer](https://wandb.ai/zhusq20/llm-memory-editability/runs/x8u0utnn)、[汇总](https://wandb.ai/zhusq20/llm-memory-editability/runs/qov0c6uz)。独立重载运行为vr4skmep/7xu6wdbx，八个阶段均finished，完整JSON下载哈希逐项相同。原DiscoLoop审计及云端收据绑定复用。
