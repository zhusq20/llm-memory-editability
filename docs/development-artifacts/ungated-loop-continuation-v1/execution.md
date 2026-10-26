# 实际提交记录

2026-10-17 18:23 CDT，控制器PID1831124已提交并核实存活，两臂均等待前序。GPU1接DiscoLoop固定注入，GPU2接普通Loop；只在对应原续训训练和grid成功完成后准入。当前新增科学更新0，CUDA恢复检查未执行，不记为通过。

31项检查通过（含两份真实父检查点CPU恢复及连续Adam更新一致性），Ruff检查/格式通过，实际容器输出目录写入通过。最初CPU测试容器用户身份不匹配导致读取拒绝，模型执行前失败；修正用户身份并改用本地只读快照后通过，失败容器保留。

源码/数据223文件冻结。固定镜像与父镜像不同，native训练源码hash与父批完全相同。源码、输入、freeze保存在持久归档`frozen-inputs-and-source.tar.gz`，控制器从本机/tmp冻结副本读取；所有运行输出直接写入持久存储。恢复先读控制器、锁、freeze和实际LM容器，不重复提交。

- [控制器](../../../results/ungated-loop-continuation-v1/execution/controller-state.json)
- [进程](../../../results/ungated-loop-continuation-v1/execution/controller-process.json)
- [检查](../../../results/ungated-loop-continuation-v1/execution/validation.json)
- [冻结配置](../../../configs/ungated-loop-continuation-v1.json)
- [W&B项目](https://wandb.ai/zhusq20/llm-memory-editability)；训练实际启动时生成独立运行链接。

本批沿用此前短训/独立终点审计豁免，终点评测不等于独立审计。ARS experiment-agent/run工作流：已提交，科学运行尚未开始。
