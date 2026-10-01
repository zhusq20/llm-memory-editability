# 独立世界确认：行为效应及必要代价

以下三个确认终点完整呈现；高比例减低比例，以世界为独立推断单位。区间、Holm 校正与符号翻转敏感性均直接使用冻结统计实现。

| 确认终点 | 均值变化 pp | 98.333% 同时区间 pp | Holm p | 状态 |
|---|---:|---|---:|---|
| editing_exception_fixed9 | +2.488 | [-2.171, +7.148] | 0.138778 | estimable |
| learning_original_exception | +34.456 | [+27.586, +41.327] | 3.10753e-06 | estimable |
| editing_coherent_fixed9 | -23.148 | [-32.032, -14.264] | 0.000161947 | estimable |

## 同时呈现的描述性检查

表中准确率与损伤单位为 %，NLL 保持原单位。每个世界先平均内部配对单元，再对有定义的世界等权平均；每项同时列出低/高比例的可用世界数。未额外检验或选择有利的次级结果。

| 检查 | 低比例 | 高比例 | 高 − 低 | 可用世界数 低/高 |
|---|---:|---:|---:|---:|
| learning_direct_all | 94.8059 | 83.9498 | -10.8561 | 8/8 |
| learning_direct_original_exception | 39.3066 | 73.7630 | 34.4564 | 8/8 |
| learning_direct_newly_exception | 98.4980 | 74.6024 | -23.8956 | 8/8 |
| learning_direct_remaining_ordinary | 98.5128 | 93.4021 | -5.1107 | 8/8 |
| learning_original_actual_error | 50.0977 | 8.4635 | -41.6341 | 8/8 |
| learning_original_other_error | 10.5957 | 17.7734 | 7.1777 | 8/8 |
| learning_original_two_step | 97.4121 | 98.4049 | 0.9928 | 8/8 |
| learning_original_bridge_correct | 97.3307 | 98.3073 | 0.9766 | 8/8 |
| learning_original_bridge_valid | 99.2513 | 99.8210 | 0.5697 | 8/8 |
| learning_original_actual_fact | 97.8027 | 96.2565 | -1.5462 | 8/8 |
| learning_original_membership_fact | 97.3307 | 98.3073 | 0.9766 | 8/8 |
| learning_original_root_fact | 99.9674 | 99.9186 | -0.0488 | 8/8 |
| base_accuracy | 98.8339 | 98.0991 | -0.7348 | 8/8 |
| base_nll | 0.0547 | 0.0849 | 0.0302 | 8/8 |
| mean_heldout | 94.8059 | 83.9498 | -10.8561 | 8/8 |
| coherent_E | 100.0000 | 100.0000 | 0.0000 | 8/8 |
| coherent_E_roots | 100.0000 | 100.0000 | 0.0000 | 8/8 |
| coherent_E_actual | 100.0000 | 100.0000 | 0.0000 | 8/8 |
| coherent_D | 74.7938 | 53.9768 | -20.8171 | 8/8 |
| coherent_D_heldout | 75.3364 | 52.1484 | -23.1879 | 8/8 |
| coherent_paired_reference_D_heldout | 87.7315 | 64.5833 | -23.1481 | 8/8 |
| coherent_U_full_damage | 0.6579 | 0.8436 | 0.1856 | 8/8 |
| coherent_U_full_coverage | 98.5429 | 96.8902 | -1.6527 | 8/8 |
| coherent_U_heldout_damage | 0.7298 | 1.2250 | 0.4951 | 8/8 |
| coherent_U_heldout_coverage | 98.0229 | 94.7153 | -3.3077 | 8/8 |
| coherent_local0_damage | 8.3333 | 4.3229 | -4.0104 | 8/8 |
| coherent_local0_coverage | 97.2222 | 96.0938 | -1.1285 | 8/8 |
| coherent_local1_damage | 94.3116 | 7.2104 | -87.1011 | 8/8 |
| coherent_local1_coverage | 98.8219 | 95.9201 | -2.9018 | 8/8 |
| coherent_local2_damage | 1.6544 | 1.8396 | 0.1852 | 8/8 |
| coherent_local2_coverage | 98.6765 | 97.5174 | -1.1592 | 8/8 |
| coherent_local3_damage | 0.5590 | 1.4027 | 0.8437 | 8/8 |
| coherent_local3_coverage | 98.3122 | 96.0430 | -2.2692 | 8/8 |
| coherent_local4_damage | 0.3540 | 0.5306 | 0.1766 | 8/8 |
| coherent_local4_coverage | 98.6344 | 97.2230 | -1.4114 | 8/8 |
| exception_E | 100.0000 | 100.0000 | 0.0000 | 8/8 |
| exception_E_roots | 100.0000 | 100.0000 | 0.0000 | 8/8 |
| exception_E_actual | 100.0000 | 100.0000 | 0.0000 | 8/8 |
| exception_D | 40.0011 | 28.9171 | -11.0840 | 8/8 |
| exception_D_heldout | 39.1710 | 27.5608 | -11.6102 | 8/8 |
| exception_paired_reference_D_heldout | 12.6157 | 15.1042 | 2.4884 | 8/8 |
| exception_U_full_damage | 0.6334 | 0.8627 | 0.2293 | 8/8 |
| exception_U_full_coverage | 98.5429 | 96.8902 | -1.6527 | 8/8 |
| exception_U_heldout_damage | 0.6869 | 1.2746 | 0.5877 | 8/8 |
| exception_U_heldout_coverage | 98.0229 | 94.7153 | -3.3077 | 8/8 |
| exception_local0_damage | 6.3194 | 4.4444 | -1.8750 | 8/8 |
| exception_local0_coverage | 97.2222 | 96.0938 | -1.1285 | 8/8 |
| exception_local1_damage | 87.5911 | 5.9403 | -81.6509 | 8/8 |
| exception_local1_coverage | 98.8219 | 95.9201 | -2.9018 | 8/8 |
| exception_local2_damage | 1.5937 | 1.7604 | 0.1667 | 8/8 |
| exception_local2_coverage | 98.6765 | 97.5174 | -1.1592 | 8/8 |
| exception_local3_damage | 0.5641 | 1.4149 | 0.8508 | 8/8 |
| exception_local3_coverage | 98.3122 | 96.0430 | -2.2692 | 8/8 |
| exception_local4_damage | 0.3404 | 0.5618 | 0.2214 | 8/8 |
| exception_local4_coverage | 98.6344 | 97.2230 | -1.4114 | 8/8 |

条件指标出现零已知分母的世界/条件/指标单元数：0。这些单元及各单元 available_cases、known_total、denominator_total 完整保留；未将无法定义的损伤率填为零。不同训练比例的已知覆盖可能不同。

## Case-macro 与 pooled 保持率分母

上表与图中 U 指标先平均每个编辑 case 的条件损伤 broken/known；known=0 的 case 保留为未定义。下表另列每世界内 sum(broken)/sum(known) 的 pooled 损伤，然后对八世界等权平均。二者不是同一个估计量。broken、known 和 n 总数是重复编辑/支持下的查询实例数，不是独立人员或独立世界样本量；不据此计算置信区间。每个 world 的两种估计及其分子分母完整保存在 descriptive-world-checks.csv。

| 损伤检查 | 低 case-macro % | 低 pooled % | 高 case-macro % | 高 pooled % | 低 broken/known | 高 broken/known |
|---|---:|---:|---:|---:|---|---|
| coherent_U_full_damage | 0.6579 | 0.6574 | 0.8436 | 0.8396 | 25467/3873539 | 31949/3808575 |
| coherent_U_heldout_damage | 0.7298 | 0.7296 | 1.2250 | 1.2171 | 10021/1373325 | 16127/1326984 |
| coherent_local0_damage | 8.3333 | 8.3093 | 4.3229 | 4.0781 | 93/1120 | 45/1107 |
| coherent_local1_damage | 94.3116 | 94.3266 | 7.2104 | 7.0253 | 7517/7969 | 543/7735 |
| coherent_local2_damage | 1.6544 | 1.6531 | 1.8396 | 1.8353 | 2443/147778 | 2678/146042 |
| coherent_local3_damage | 0.5590 | 0.5577 | 1.4027 | 1.3911 | 6229/1116889 | 15165/1091110 |
| coherent_local4_damage | 0.3540 | 0.3533 | 0.5306 | 0.5280 | 9185/2599783 | 13518/2562581 |
| exception_U_full_damage | 0.6334 | 0.6329 | 0.8627 | 0.8581 | 24515/3873539 | 32652/3808575 |
| exception_U_heldout_damage | 0.6869 | 0.6866 | 1.2746 | 1.2664 | 9430/1373325 | 16781/1326984 |
| exception_local0_damage | 6.3194 | 6.3402 | 4.4444 | 4.3406 | 71/1120 | 48/1107 |
| exception_local1_damage | 87.5911 | 87.6117 | 5.9403 | 5.8070 | 6982/7969 | 449/7735 |
| exception_local2_damage | 1.5937 | 1.5923 | 1.7604 | 1.7550 | 2353/147778 | 2561/146042 |
| exception_local3_damage | 0.5641 | 0.5624 | 1.4149 | 1.4027 | 6280/1116889 | 15292/1091110 |
| exception_local4_damage | 0.3404 | 0.3396 | 0.5618 | 0.5587 | 8829/2599783 | 14302/2562581 |

## 组织与测试关系的交叉结果

[完整组织报告](organization-report.md)无条件呈现所有固定学习 cohort、两种编辑的固定九人、全 D、heldout D、E 全体与 root/actual 分项。每项包含三组织×两测试链六个原始单元、匹配交互、company/project 各自匹配和不匹配查询相对 neither 的差异及双链平均收益；八个世界分别列低比例、高比例和高−低变化，不做次级显著性检验。

## 共同已知且同真值的旧知识保持

[配对保持控制报告](common-u-report.md)同时报告 unchanged U 交集、双侧原先均答对的共同已知子集，以及进一步要求同真值的预注册控制。两侧使用完全相同的已知分母，保留 full、unseen、heldout 和所有局部分层、覆盖与零分母。这一控制不能替代前述各 phase 自身旧知识上的破坏率。共同旧正确是例外比例干预后的条件子集，不代表例外比例对保持的主因果效应；均不增加显著性检验。

## 解释范围

该确认针对 758 万参数模型、固定合成数据生成器与训练/编辑预算下，训练相关性干预对原例外学习和编辑传播的行为效应。主编辑终点是 exception 更新的固定九名heldout 人员；coherent 更新的同九人是配对参照。学习没有按支持集重复；两初始化、三组织、两查询链与两编辑支持均为世界内因素。

直接准确率要求目标值与 EOS 正确。自主两步最终成功要求第一步输出合法组织并以 EOS 终止，第二步输出目标城市并以 EOS 终止；第一步不必等于真值组织，因为错误组织也可能共享城市。第一步真值准确率 bridge_correct 与合法率 bridge_valid 在表中分别报告。两步使用模型预测组织，不输入真值桥接；它增加推理调用，不能直接等同于单次前向内部算法。对应根事实按人员加权，同一 root 可被多名人员重复引用。基础知识、全体查询、全 D、E 拟合、局部与全局 U 损伤及覆盖必须共同解释，不能只报告固定九人的收益。

组织交互图只作描述，未追加显著性或等效性结论。世界级 t 推断有分布假设；符号翻转是依赖零假设符号对称性的敏感性分析。无显著差异不证明等效，行为确认也不唯一识别组织内部中介、电路或自然语言任务上的效果。
