# 层数与组合支持增补：实际提交

2026-10-17 01:34 UTC提交`loop-hop-depth-support-v1`，主机控制器PID1247588。固定镜像24项CPU契约及Ruff已通过，01:35 UTC进入`waiting_parent_completion`，14条新臂均pending，正式新增更新0。CUDA预检、32步工程短训及原R／全网格重载尚待父批结束后执行，不能把CPU通过写成GPU或科学训练已经通过。

用户要求的2层×2／3／4轮、3层×2／3／4轮均保留普通Loop与DiscoLoop对照。现有两条低支持2×2直接复用原批300k，不重复训练；新结构10条。另在2×2与3×2各加入两架构的更多组合支持，共4条。合计14条×300k＝4200000新增正式更新；原八臂预算不变。

两档支持每长度10000／30000，训练池40000／100000唯一问题。保留原原子、图、原训练前缀及全部43000评价题原字节。高支持采样器保持原40000槽的类别、原子和partial-batch次序，只轮换组合支持块；各类曝光与监督token匹配，不通过降低原子权重来增加组合比例。完整真实数据的嵌套、隔离、曝光、恢复和L2初值一致性已由CPU契约核对。

父批`loop-hop-unified-v1`PID317114仍按原契约运行。新批只有CPU检查先执行，未获取GPU槽位。等待父批全训练／原R与全网格审计／云端核验完成、释放全局执行锁后，才在物理cuda9自动接续；最多8条并行，先14臂CUDA工程全部通过，再开始正式训练。没有抢占、暂停或重启父批，不使用其它GPU，也不改变父批资源契约。

恢复入口为`/ossfs/workspace/llm-memory-editability/loop-hop-depth-support-v1`内controller-process.json、controller-state.json、freeze.json、cpu-contracts-complete.json及各阶段exit.json。局部初始化不等于训练恢复；任何失败保留，不自动重训。全局loop-hop-current.json在实际GPU接续前仍指向父批；新批状态以本批控制器为准。

[协议](protocol.md)、[历史与复用](history-review.md)、[本机提交收据](../../../results/loop-hop-depth-support-v1/submission.json)、[准备状态](../../../results/loop-hop-depth-support-v1/readiness.json)、[CPU契约](../../../results/loop-hop-depth-support-v1/contracts.xml)、[W&B项目](https://wandb.ai/zhusq20/llm-memory-editability)。每条新训练科学更新前自动初始化专用W&B；当前无新科学曲线或结论。

封存200项文件，freeze清单SHA256为`80312889cbfc10ebc3d3f768625aeaed34686195b3dc69fc644718bddc5b5c5b`。更多支持world为`855261ce5ada8875b8610cba536cd20b49b788f7300dee11205efa761af3c64b`，共同panels仍为`c991cebeeeec05639be87d981cd20031016f47965c43fed8bf5cc55579c3b468`；两档均与各自封存清单绑定。
