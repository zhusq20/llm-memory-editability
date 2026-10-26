# 跳位分组预定分析：路径一致性在自然模型编辑上的检验

冻结：2026-10-08。机器契约为[configs/path-hop-analysis-v1.json](../../../configs/path-hop-analysis-v1.json)，执行脚本为[scripts/analyze_path_hop_position.py](../../../scripts/analyze_path_hop_position.py)，契约测试为[tests/test_path_hop_analysis.py](../../../tests/test_path_hop_analysis.py)。本批零训练、零新增模型前向，只读取两项复现的已审计逐例记录。

## 问题

[路径一致性工作假说](../../roadmap.md#path-consistency)预测：更新只传到使用被写入计算路径的查询。ROME/MEMIT在显式主体的末词元上写入早中层MLP。被编辑事实作首跳时，组合问题直接包含该主体；作后续跳时，主体只作为模型内部算出的桥实体出现，不经过写入时的词元位置。因此预测首跳编辑比后续跳编辑更能传到多跳答案，且让桥实体显式写出的CoT会缩小这一差距。

这是假说在自然模型编辑上的推论检验，不检验受控模型中的内部机制。相近观察已有文献：RippleEdits原文分别报告Compositionality I/II；IFMET等工作讨论多跳编辑需要修改更深层；Hopping Too Late讨论桥实体形成过晚。本分析的增量是同一冻结复现上的预定跳位分组，以及MQuAKE中直接回答与CoT的同案例对照。引用前须按原文核对。

## 冻结时状态

编写者未读取`mquake-paper-reproduction-v1`或`rippleedits-paper-reproduction-v1`的任何逐例记录或汇总。两批进度文件只含按总跳数的汇总，不含跳位分组。下表计数只来自公开数据，数据SHA256与两批冻结配置一致。脚本拒绝读取未完成或独立重放审计未通过的运行，避免查看中途结果。

## 分组与样本量

MQuAKE-CF-3k单编辑案例1093个。主比较只用2跳与3跳单编辑案例，两层等权：

| 总跳数 | 编辑在首跳 | 编辑在后续跳 |
| --- | ---: | ---: |
| 2跳 | 213 | 300（第2跳） |
| 3跳 | 198 | 158（第2跳107、第3跳51） |
| 4跳（仅描述） | 15 | 209 |

多编辑案例按是否包含首跳只作描述。

RippleEdits中，CI测试的组合查询主体就是被编辑主体，编辑事实作首跳；CII测试的首跳答案是被编辑主体，编辑事实作第二跳。数据核查结果：

| 子集 | CI案例/测试 | CII案例/测试（结构合格） | 两轴都有的案例 |
| --- | --- | --- | ---: |
| POPULAR | 274 / 1781 | 120 / 445 (445) | 26 |
| RANDOM | 627 / 2985 | 207 / 746 (746) | 20 |
| RECENT | 829 / 9789 | 285 / 1139 (1137) | 101 |

RECENT有2条CII测试的查询主体等于被编辑主体，不符合第二跳结构，排除并计数。

## 指标与比较

全部沿用两项复现的冻结评分，不重新计分。

- MQuAKE直接回答：三个问法任一与新答案或别名完全一致（原文口径）；CoT同理。前提子集：编辑前原链直接组合正确、编辑后全部新单跳正确、全部编辑成功。
- M1（主）：全池直接回答，首跳组减后续跳组，2跳/3跳两层等权。
- M2（主）：同一批案例中，直接回答的组差减去CoT的组差。分组对应不同案例，实体与关系类型会不同；CoT是同案例内的对照，能扣除这部分差异。
- S1：M1/M2在前提子集上的结果及覆盖。S2：编辑后多跳仍答旧答案的比例，只计新旧别名不相交的案例。S3：4跳与多编辑案例描述。
- R1（主）：RippleEdits每个子集中，编辑成功案例的已执行、结构合格测试，CI通过率减CII通过率。R2：两轴都有的案例内差值，仅描述。

区间：以案例为单位的百分位自助法，10000次，种子20261008；MQuAKE在跳数层内联合抽样。不以显著性作为唯一判读门槛。粗略模拟显示，在基础率约20%/10%时，M1的95%区间半宽约5个百分点，M2约7个百分点；更小的真实差距本批可能测不出。

## 预测与判读

| 编号 | 预测 |
| --- | --- |
| P1 | ROME与MEMIT都有M1 > 0 |
| P2 | M2 > 0：CoT显式写出桥实体后跳位差距缩小 |
| P3 | 三个RippleEdits子集R1 > 0 |
| P4 | 后续跳编辑后答旧答案的比例更高 |

P1、P2在两种编辑器上都成立，且P3方向一致，才算支持该推论。两种编辑器的M1区间都不高于0，则不支持。P1成立而M2约为0时，跳位差距更可能来自案例或实体类型差异，而非路径不一致。每个基准只有一个模型；RippleEdits没有CoT对照，R1只能作方向参考。

## 不允许的操作

不在未完成或未审计的运行上执行；读取结果后不改分组、分层、评分或抽样设置；不以前提子集替代全池主结果。确需修订时另立版本并保留本契约。

## 执行

两项复现各自完成且`audit.json`通过后，在实验主机上对每个运行执行：

```bash
python scripts/analyze_path_hop_position.py analyze --benchmark mquake \
  --data-file <冻结MQuAKE-CF-3k.json> --run-dir <该运行目录> \
  --output results/path-hop-analysis-v1/<运行名>.json
python scripts/analyze_path_hop_position.py analyze --benchmark rippleedits \
  --data-file <冻结popular/random/recent.json> --run-dir <该运行目录> \
  --output results/path-hop-analysis-v1/<运行名>.json
```

运行目录须含`run.json`、`cases/`、`complete.json`和`audit.json`；路径以各批控制器记录为准。`counts`子命令只读数据，可复算上表。
