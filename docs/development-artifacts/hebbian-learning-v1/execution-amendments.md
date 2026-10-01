# 2026-09-28 执行细化与覆盖修订

依据独立计划§4.4、§11，以下数据覆盖决定在模型开发训练开始前确定。原计划文件保持原文；实际清单由data-lock固定。

- 官方CounterFact共21,919条；指定关系6,649条。程序审核后5,106条进入基线评分。语义谓词规则进一步留下3,699条候选，实际各池合计1,384条。
- F_form/V_form/B_dev/B_eval/C_dev/C_eval分别为512/128/32/128/64/256，全部学习episode规模保持原计划。R_keep/V_keep/U_keep分别为96/64/104；合格的原模型已掌握事实不足时，以8条为单位缩减，保留2个百分点参考线及实际分母。
- 每个subject先进入唯一候选池，再评价基线。候选池的保持事实预留权重为计划数量的3倍，其他池为1倍；这项预留不使用基线结果。随后各池内部按关系平衡取样，未来学习优先采用待习得事实。
- 使用来源原事实标签，逐条应用明确的语义谓词规则。出生地/泛指来源地、母语/一般使用语言、制造者/开发者混淆进入review。P19因此未进入本次主分析。完整真实世界事实与所有主体别名未独立人工核验，规则审核不能记为人工逐例核验；该边界保留在结果报告中。
- 程序检查和语义规则审核分别写入programmatic-audit.json、semantic-audit.json；两个清单均保存逐条决定与原因。全部views一起保留或排除，review池不参与训练或主评价。
- C2使用固定种子、完整derangement枚举的匹配实现：优先不同答案，再增加同关系配对，最优解内固定种子破除并列。它把计划中的固定补配具体化为最小代价匹配，不是逐条贪心补配。全部置换在batch-manifest逐步保存；CE标签维持原值。此选择在任何C形成运行前记录。
- FP32权重、前向、优化器，eager注意力，TF32关闭，确定性算法开启。CPU上阶段A用FP64。GPU0当次硬件检查仍报告96次不可纠正ECC，使用3/4/5/7/8/9；不改动其他任务。
- 知识学习有效batch为8，保持batch为6事实加2文本，直接一次完成各有效batch，损失按每序列token均值再平均；几何24视图另行前向。实际训练、参考分布和几何计算分别记账。
- 学习节点生成在EOS或首个换行处停止，最多24 tokens；保存原生成和EOS行为，完整答案评分不接受额外解释或只匹配答案前缀。
- 通用文本按文档边界取128-token片段，训练/测试各128条；文件revision及原始字节哈希独立锁定。
- B开发三个学习率的平均V_keep损伤均高于2个百分点。按计划后备规则选取损伤最小的1e-4；平均损伤5.078125个百分点、Q-AUC 0.383728。该选择并不表示保持阈值已满足。

各阶段固定矩阵由execute_hebbian_matrix.py调度，阶段间等待完整回执，B预测器先于B_eval曲线封存，C-lock先于9个正式形成父模型保存。

## 复现入口

使用environment-lock.json记录的Python及environment-freeze.txt依赖。项目根目录设置PYTHONPATH=src，依次运行：

```bash
python scripts/run_hebbian_learning.py prepare
python scripts/run_hebbian_learning.py calibrate
python scripts/run_hebbian_learning.py audit-data
python scripts/run_hebbian_learning.py baseline --device cuda:3
python scripts/run_hebbian_learning.py lock-data
python scripts/run_hebbian_learning.py preflight --device cuda:3
python scripts/run_hebbian_learning.py diagnose --split dev --device cuda:3
python scripts/execute_hebbian_matrix.py --gpus 3,4,5,7,8,9
python scripts/report_hebbian_learning.py
```

已有运行目录通过完成回执跳过，未完成轨迹从最近节点恢复参数、优化器、随机状态和数据游标。已冻结数据不得重新清洗或重写；要变更清单应另建版本。大型权重与逐步结果留在results，原始来源和数据留在data，小型配置、图、清单和审计位于本目录。

## 正式C启动前的显存细化

共享服务器其他作业重新占用部分GPU约69GB显存。正式C的几何评价改为每24视图前向、FP64 CPU汇总，训练有效batch与目标、节点、选择规则保持原值。整批/分批几何等价检查通过后保存execution-source-v2；原始开发源码仍在execution-source。两套源码哈希都保留供独立审计。
