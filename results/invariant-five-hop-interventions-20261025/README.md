# 旧invariant五跳局部干预

50池完整矩阵已完成、零训练，详见[联合报告](../invariant-five-hop-diagnosis-20261025/report.md#2-一次局部干预进一步分开原因)及[全部干预指标](../invariant-five-hop-diagnosis-20261025/interventions.csv)。
冻结[协议](protocol.md)、runtime/source.tar、源码哈希、原始预测与历史权重保留。

实现审阅确认强制V4/V0的head维序、单块原子供体残差与错桥构造符合协议。
冻结输出中的`all_other_positions_unchanged`实际核对的是其它位置的**解码token**，不是完整hidden张量；联合汇总使用明确字段`all_other_position_readouts_unchanged`。
完整原子供体复制已由同块MLP算出的H1，检验的是接收与保持，不能独立归因于注意力读取。
主矩阵400万题目循环轮，另有272万原子供体循环轮；不更新参数。
