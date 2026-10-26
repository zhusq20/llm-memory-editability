# 五跳失败的第二阶段：一次交接干预

这是看到第一阶段轨迹后的事后机制诊断，不是新训练或预注册确认实验。
第一阶段B五跳R16正确0.66%，前16轮从未正确99.14%，曾正确后丢失0.20%；
第四前缀在R16可读82.76%。本阶段区分已有桥的路由、规范实体的value接口、整个原子调用接口。
复用150万步旧invariant参数EMA、原世界A/B五跳各5000题、原绝对位置、旧首轮F(2e)。
主终点R16，保存全部逐轮预测；不选择最优时刻代表模型自然能力。

## 固定矩阵

每个条件仅在R4、R5、R6、R8之一的第五接收位置施加一次，其余计算沿用原实现。
六条件×四时刻×两个池，加两个无干预重放，共50池、250000次题目评测、400万题目循环轮。

1. force_bridge：attention各头的加权value直接改为当前原生第4位置的value；保留接收方query、残差与MLP。
2. force_original_head：同形式改读第0位置，作为路由的内容对照。
3. canonical_bridge_force：改读真正桥实体e4在独立原子输入第0位置的value，输入为2(E[e4]+P0)。不提供e5。
4. wrong_bridge_force：供体改为同图(e4+1) mod 500；其他相同。记录对应错误后继r5(e4_wrong)。
5. atomic_postattention：构造[e4,r5,pad,pad,pad,pad]，提取首轮接收位1的完整attention后残差z，替换五跳接收位5的z。
   因MLP逐位置，使用同块算出的z+MLP(LN2(z))作为第五位置raw H，数值上保留原子运行形状。
6. wrong_atomic_postattention：相同旁路，但供体头为错误桥，记录输出是否跟随其错误答案。

5/6同时改变桥、关系表示、位置格式和残差尺度，供体可能已经带答案信息。
因此raw H恢复属于供体构造的直接结果；真正需要观察的是候选、聚合接收及后续保持。
这只是oracle接口旁路，不能单独证明attention权重是根因，也不是可部署修复。
真值e5只用于事后计分，不输入模型。e4由真路径计算，属于明确的oracle中间实体干预。

## 计分与验证

报告干预当轮raw/candidate/h正确率、R16正确率、原生桥在前一轮可读子集及覆盖率；
全量收益和损害、错误供体答案率、首次正确与后续丢失均保留。原子供体另存raw和完整R16读出。
所有第五位之外的前缀与无干预逐轮逐题完全相同；干预前答案完全相同；
无干预与第一阶段历史预测完全相同；权重前后哈希不变。
事实可单独调用、桥可解码、整题调用成功是不同结论；不可解码不等于内部没有信息。

## 运行范围

独立LM Docker、root所有0444/0555冻结源码、只读原检查点、独立/ossfs结果。
GPU6共享原已授权LM训练，重新核对原预约例外、进程与40000MiB空余；2CPU、16GiB主存、显存上限0.16。
不重训、不恢复暂缓任务。持续监测，预计数分钟；30分钟上限。原训练PID与启动时间事后核对。

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-10-24（America/Chicago）
- Verification Status: posthoc protocol fixed before second-stage inference
- Version Label: five_hop_single_intervention_v1
