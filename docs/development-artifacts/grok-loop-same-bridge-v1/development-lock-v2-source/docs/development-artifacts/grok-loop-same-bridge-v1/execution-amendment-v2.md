# 开发数值校准修订 v2

初次开发执行在C1的OOD供体池同批基线核对中停止：完整接收池与扁平供体池的batch高度不同，GPU矩阵运算出现logit舍入差，答案和EOS一致，但原检查错误地要求跨batch高度logit逐位相同。本次未生成完整科学报告，失败记录保留于原 `results/grok-loop-same-bridge-v1/development/dev-h2-c1/failed.json`；初始开发锁、源码及供体表保留原位置。

修订只涉及跨batch形状的数值审计：要求答案及EOS完全一致，记录最大logit差，限值1e−4；各供体干预仍以相同batch形状基线比较。所有同形状self/原前缀控制、both/full恒等、native/traced和历史评分审计仍按原规则执行。未修改供体、查询、组件、架构、评分定义或统计单位。

新执行使用 `development-lock-v2.json`、`donors/v2/` 与 `results/grok-loop-same-bridge-v1/execution-v2/`，不覆盖初次失败或旧冻结来源。开发五端点全部通过后再冻结30个原正式端点后续比较。图表汇总脚本单独记录源码与测试哈希，并独立重算保存的逐查询指标，不参与供体选择或干预操作。
