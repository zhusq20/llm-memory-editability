# 原始生成的定性审阅

本文件是 Codex 对确定性样本的逐例审阅，不是全量人工标注，也不据此更改冻结评分。样本由 `error-review-samples.jsonl` 提供，源文件 SHA256 在 `audit.json` 中。

| 世界 / 条件 / 源预测行 | 目标 | 原生成 | 审阅判断 |
| --- | --- | --- | --- |
| 142701 / MP-CoT task / 5 | October 17, 2024 | October 17, 2024; Answer: October 17, 2024 | 直接作答提示下输出了重复答案与标签；事实日期本身正确，冻结直接评分因为格式而失分。 |
| 142701 / MP task / 103 | 1941 | 2092 | 输出是合法四位年份，但值不正确；不能解释为单纯格式错误。此例属于 dev，未当 test 示例统计。 |
| 142701 / MP task / 61 | Gabriella Wyatt Coulson | Aaliyah Madeline Blackwell | 原问题只询问 Mikayla Sadie Lloyd 与 Gabriella Wyatt Coulson；返回第三个人名。这是问题对象绑定失败的行为证据，不足以定位其训练形成机制。 |
| 142701 / MP-CoT task / 78 | 2024 | October 17, 2024; Answer: Answer: Answer:; Answer:; Answer: 2030 | 生成曾出现正确源年份，但最终答案为 2030。说明“文字中出现目标”不能自动救回为语义正确。 |

上述 MP-CoT 的第 5 行是 `view=2,cot=False`；第 78 行是 `view=2,cot=True`。两个输出说明历史 `cot` 分支的答案抽取规则必须保留，不能事后统一选最有利的答案位置。

源目录：`results/bios-original-development-v1/world-142701/init-1427/{condition}/task/evaluation/predictions.jsonl`。定量结果见 `report.md`、`task-view-groups.csv`，不能从这四条样本外推错误比例。
