# 开发数值校准修订 v3

v2开发C1的OOD供体池跨batch高度最大logit差为0.0002164841，超过预设1e−4，执行再次停止；答案与EOS仍完全相同。v2锁、源码、供体和失败路径 `results/grok-loop-same-bridge-v1/execution-v2/development/dev-h2-c1/`保留。

本修订不再用跨batch高度容差接受差异，而是统一所有接收/供体条件的物理batch高度为原全量接收池的batch高度。小供体池用其最后一条记录重复填满；填充副本在计分、保存和分母统计前全部移除。两类供体、同批基线、self、反事实均采用相同形状；原全量基线保持历史形状，原子前提按原全量原子形状核验。

恢复跨供体池基线logits逐位相同检查；self、原前缀、both/full及C1无效应规则继续逐位检查。该修改只控制GPU运算形状，没有改变有效查询、供体、评分、权重或正式比较。新开发锁为 `development-lock-v3.json`，供体在 `donors/v3/`，新结果在 `results/grok-loop-same-bridge-v1/execution-v3/`。所有失败保留，无正式后续端点已评分。
