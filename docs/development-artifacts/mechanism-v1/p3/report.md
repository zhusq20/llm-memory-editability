# P3：个人实际属性捷径可靠性干预

完成 12/12 对训练、72/72 对学习检查点、24/24 对链级两步诊断。完整审计：True。

低/高比例分别为每组 2/32 和 16/32 例外；两条链同时改变。主要比较使用完全相同的 derived QA 问题、真值和人员 ID。每链固定三层：原有例外 128 人、新增例外 896 人、仍为普通 1024 人；各层一半留出。

| 组织 | 查询链 | 同人层 | 完整配对块 | 低比例直接答题 | 高比例直接答题 | 差值 |
|---|---|---|---:|---:|---:|---:|
| company | company | all | 4 | 96.83% | 89.75% | -7.08% |
| company | company | newly_exception | 4 | 98.83% | 81.36% | -17.47% |
| company | company | original_exception | 4 | 64.84% | 83.98% | 19.14% |
| company | company | remaining_ordinary | 4 | 99.07% | 97.80% | -1.27% |
| company | project | all | 4 | 94.41% | 87.55% | -6.86% |
| company | project | newly_exception | 4 | 98.33% | 81.03% | -17.30% |
| company | project | original_exception | 4 | 35.55% | 80.08% | 44.53% |
| company | project | remaining_ordinary | 4 | 98.34% | 94.19% | -4.15% |
| neither | company | all | 4 | 94.34% | 82.71% | -11.62% |
| neither | company | newly_exception | 4 | 98.60% | 72.32% | -26.28% |
| neither | company | original_exception | 4 | 33.20% | 71.09% | 37.89% |
| neither | company | remaining_ordinary | 4 | 98.24% | 93.26% | -4.98% |
| neither | project | all | 4 | 93.53% | 81.47% | -12.06% |
| neither | project | newly_exception | 4 | 97.99% | 72.15% | -25.84% |
| neither | project | original_exception | 4 | 23.83% | 71.88% | 48.05% |
| neither | project | remaining_ordinary | 4 | 98.34% | 90.82% | -7.52% |
| project | company | all | 4 | 95.46% | 88.11% | -7.35% |
| project | company | newly_exception | 4 | 99.44% | 79.80% | -19.64% |
| project | company | original_exception | 4 | 33.98% | 81.64% | 47.66% |
| project | company | remaining_ordinary | 4 | 99.66% | 96.19% | -3.47% |
| project | project | all | 4 | 96.34% | 90.38% | -5.96% |
| project | project | newly_exception | 4 | 99.39% | 83.76% | -15.62% |
| project | project | original_exception | 4 | 55.47% | 83.98% | 28.52% |
| project | project | remaining_ordinary | 4 | 98.78% | 96.97% | -1.81% |

paired-learning.csv 和 paired-two-step.csv 保留每个世界/初始化的配对计数；queries/ 保存逐查询低/高预测、EOS、真值、人员层和桥接结果。organization-effects.csv 与 organization-interactions.csv 分别给组织交叉效应及其高减低变化。

phase-native-macro.csv 的普通/例外宏平均使用各比例自己的分组，人员构成不同；它是单独的描述指标，不能替代固定同人层对比。actual_on_conflict 仅统计实际属性与derived QA 真值不同的查询；普通人答案重合不作为路径证据。

自主两步使用模型预测的桥接实体。它衡量可访问的知识及外部组合效果，不能单独证明一次前向内部执行同一算法。两世界均为开发数据，四个配对块不等于四个独立世界。

本阶段没有编辑案例；高比例世界不套用原 E93 更新。审计不读取模型权重，权重来源由冻结训练身份与两步记录关联，已明确记录这一审计边界。
