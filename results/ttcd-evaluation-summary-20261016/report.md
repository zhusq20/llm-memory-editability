# 一步写入与merge完整结果（2026-10-16恢复）

恢复新增基础/外层训练0；原训练、检查点和原一步训练主报告均保留。四步训练终点的一步部署、两跳位merge、独立重放与完整云端JSON核验全部完成。

本批32事实为原图已知键的replacement纠正，复用冻结面板。下表单事实为每条从A0独立编辑；连续与merge为写完32条后的结果。训练内写次数为1或4，部署全部每事实一步；merge只加从A0独立计算的增量。原子、两跳答案均按原答案＋EOS评分。

|训练内写次数|架构|部署写法|已写事实可答|第一跳编辑后的组合|第二跳编辑后的组合|原正确第二跳组合更新成功|两原子当前均正确的第二跳子集|旧原子保持|旧组合保持|
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
|1|transformer|single|32/32 (100.00%)|6/54 (11.11%)|0/253 (0.00%)|0/96 (0.00%)|0/96 (0.00%)|2048/2048 (100.00%)|1566/1568 (99.87%)|
|1|transformer|continuous|924/1024 (90.23%)|90/1728 (5.21%)|17/8096 (0.21%)|17/3072 (0.55%)|17/2774 (0.61%)|2042/2048 (99.71%)|1432/1568 (91.33%)|
|1|transformer|independent_merge|416/1024 (40.62%)|0/1728 (0.00%)|32/8096 (0.40%)|32/3072 (1.04%)|0/1312 (0.00%)|1600/2048 (78.12%)|832/1568 (53.06%)|
|1|loop|single|32/32 (100.00%)|0/54 (0.00%)|1/253 (0.40%)|1/174 (0.57%)|0/173 (0.00%)|2048/2048 (100.00%)|1941/1952 (99.44%)|
|1|loop|continuous|436/1024 (42.58%)|0/1728 (0.00%)|3/8096 (0.04%)|3/5568 (0.05%)|0/1989 (0.00%)|1919/2048 (93.70%)|840/1952 (43.03%)|
|1|loop|independent_merge|0/1024 (0.00%)|0/1728 (0.00%)|0/8096 (0.00%)|0/5568 (0.00%)|0/0 (NA)|32/2048 (1.56%)|32/1952 (1.64%)|
|1|discoloop|single|0/32 (0.00%)|0/54 (0.00%)|0/253 (0.00%)|0/252 (0.00%)|0/0 (NA)|2048/2048 (100.00%)|2048/2048 (100.00%)|
|1|discoloop|continuous|0/1024 (0.00%)|0/1728 (0.00%)|0/8096 (0.00%)|0/8064 (0.00%)|0/0 (NA)|1978/2048 (96.58%)|1946/2048 (95.02%)|
|1|discoloop|independent_merge|0/1024 (0.00%)|0/1728 (0.00%)|0/8096 (0.00%)|0/8064 (0.00%)|0/0 (NA)|896/2048 (43.75%)|192/2048 (9.38%)|
|4|transformer|single|0/32 (0.00%)|0/54 (0.00%)|0/253 (0.00%)|0/92 (0.00%)|0/0 (NA)|2048/2048 (100.00%)|1536/1536 (100.00%)|
|4|transformer|continuous|10/1024 (0.98%)|0/1728 (0.00%)|12/8096 (0.15%)|12/2944 (0.41%)|1/1 (100.00%)|2048/2048 (100.00%)|1504/1536 (97.92%)|
|4|transformer|independent_merge|64/1024 (6.25%)|0/1728 (0.00%)|0/8096 (0.00%)|0/2944 (0.00%)|0/0 (NA)|2048/2048 (100.00%)|1504/1536 (97.92%)|
|4|loop|single|0/32 (0.00%)|0/54 (0.00%)|0/253 (0.00%)|0/180 (0.00%)|0/0 (NA)|2048/2048 (100.00%)|1888/1888 (100.00%)|
|4|loop|continuous|6/1024 (0.59%)|1/1728 (0.06%)|32/8096 (0.40%)|32/5760 (0.56%)|4/28 (14.29%)|2048/2048 (100.00%)|1888/1888 (100.00%)|
|4|loop|independent_merge|0/1024 (0.00%)|0/1728 (0.00%)|0/8096 (0.00%)|0/5760 (0.00%)|0/0 (NA)|1024/2048 (50.00%)|320/1888 (16.95%)|
|4|discoloop|single|26/32 (81.25%)|42/54 (77.78%)|0/253 (0.00%)|0/253 (0.00%)|0/205 (0.00%)|2048/2048 (100.00%)|2047/2048 (99.95%)|
|4|discoloop|continuous|732/1024 (71.48%)|768/1728 (44.44%)|16/8096 (0.20%)|16/8096 (0.20%)|16/5632 (0.28%)|2032/2048 (99.22%)|1982/2048 (96.78%)|
|4|discoloop|independent_merge|0/1024 (0.00%)|0/1728 (0.00%)|0/8096 (0.00%)|0/8096 (0.00%)|0/0 (NA)|0/2048 (0.00%)|0/2048 (0.00%)|

第一跳54查询覆盖27/32事实，5事实保留零覆盖；第二跳253查询覆盖32事实。所有旋转/前缀是同32事实上的重复测量，不把1024或8096个状态/题对当作独立事实。条件子集的覆盖、第一跳完整成绩、同题同顺序配对差及三架构共同A0正确集合见原完整JSON。

两原子当前均正确子集还要求组合在A0答对，分母随方法改变；0/0记为NA，不能解释为组合成功率0%。旧保持检查的是同64道旧原子与64道旧组合在32个状态下的重复读出，只在A0正确题上计算保持率。

独立审计要求答案与EOS、写入状态精确一致，浮点读出沿用冻结代码的1e-5容差；云端完整JSON以哈希核对。

模型基础能力、参数及快矩阵数量不同；单图单初始化，不以旧保持高但写入失败证明有效学习，也不以原子成功替代组合传播。GPU7仅本项目评测共享，耗时不作独占效率比较。

## 证据

- `/ossfs/workspace/llm-memory-editability/ttcd-one-step-training-v1/control-report/result-aggregate.json`：`596f720480dd6ad11358954f651f90929bb40409f111d36f69b191add091b88b`
- `/ossfs/workspace/llm-memory-editability/ttcd-one-step-second-hop-v1/transformer/evaluation/result-audit.json`：`bf59a17c34912b57e0c81e321ab6ff5982fcec5bc7421fd7e7714952c38d9657`
- `/ossfs/workspace/llm-memory-editability/ttcd-one-step-second-hop-v1/transformer/evaluation/result-cloud-audit.json`：`e26f1661d70a102594297b0a40980d226448cbedc83d1d143dadd151e9d8b43d`
- `/ossfs/workspace/llm-memory-editability/ttcd-one-step-second-hop-v1/loop/evaluation/result-audit.json`：`36f3307eef05779eb9ac4ad6540967dc723fb06ca1fc7065b804e6f1295675d3`
- `/ossfs/workspace/llm-memory-editability/ttcd-one-step-second-hop-v1/loop/evaluation/result-cloud-audit.json`：`2357681c79057d6188b940ca9e71ffe18b0cbeabd9c56d484b04125613cdfd3e`
- `/ossfs/workspace/llm-memory-editability/ttcd-one-step-second-hop-v1/discoloop/evaluation/result-audit.json`：`05166f7e72be85b4b2e3765368b8aa75cfe9ac7f5081d5d830b90653af5d8b74`
- `/ossfs/workspace/llm-memory-editability/ttcd-one-step-second-hop-v1/discoloop/evaluation/result-cloud-audit.json`：`84f96f4c3a522c8f8bf6f5f81037a7fb9691b73e78fd173328b039ddac61130e`
- `/ossfs/workspace/llm-memory-editability/ttcd-merge-hops-v1/transformer/evaluation/result-audit.json`：`8d153d27d77eb44886daae8e5183dd8b3db90e8b6406acc1cdee5b9deed34b16`
- `/ossfs/workspace/llm-memory-editability/ttcd-merge-hops-v1/transformer/evaluation/result-cloud-audit.json`：`4efd9c87c77d59778d1772f5c8dc65dfbb90e33e2e9ea7b622c199454c96d44f`
- `/ossfs/workspace/llm-memory-editability/ttcd-merge-hops-v1/loop/evaluation/result-audit.json`：`8d153d27d77eb44886daae8e5183dd8b3db90e8b6406acc1cdee5b9deed34b16`
- `/ossfs/workspace/llm-memory-editability/ttcd-merge-hops-v1/loop/evaluation/result-cloud-audit.json`：`8cd80f083b933a788d6bea32337b1b0c437b362a7e8e97350ccbbc54320800bc`
- `/ossfs/workspace/llm-memory-editability/ttcd-merge-hops-v1/discoloop/evaluation/result-audit.json`：`8d153d27d77eb44886daae8e5183dd8b3db90e8b6406acc1cdee5b9deed34b16`
- `/ossfs/workspace/llm-memory-editability/ttcd-merge-hops-v1/discoloop/evaluation/result-cloud-audit.json`：`a92d876bde31e3da4d9349692f9dfeae3dc9c116d05ae3de1d8818534bd8cfbb`
