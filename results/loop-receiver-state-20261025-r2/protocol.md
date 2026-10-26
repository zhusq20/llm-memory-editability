# 第四轮到第五轮：关系再提供与接收状态竞争

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-10-24（America/Chicago）
- Verification Status: protocol fixed before inference
- Version Label: receiver_state_v1

用户授权以旧RoPE DiscoLoop和旧invariant检验“第五关系无法再被利用”与“接收状态难以改写”。
这延续已完成的两批五跳诊断，不是新训练，也不恢复七条暂缓任务。
旧150万步参数EMA、同原世界A/B五跳各5000题、主终点R16；固定R4或R5单次干预。

## 计算与主要识别问题

固定第五接收位attention输出为canonical桥V时，单块计算严格化为：
S=Z+c_proj(Vbridge)，H=S+MLP(LN2(S))。
因此同一规范桥在R4/R5效果不同，来源于进入块的接收状态Z不同（以及invariant后续历史合并），并非轮次参数。
桥V仅由真实e4的规范原子源输入经LN1、c_attn产生，没有第五答案、原子postattention或rawH供体。
本次主要检验：保留Z但重新提供关系，或仅改变Z范数，是否能恢复当轮事实计算与最终答案。

RoPE的Z为上一轮H+q；旧invariant的Z为h+e，初始原子输入为2e。
invariant只改送入F的第五位置，不改既有N/D或上游状态；raw、candidate、聚合输出和R16分列。
原子位置供体：RoPE Wte(r)，invariant 2(Wte(r)+P1)。后者另测2(Wte(r)+P5)保持因子2、仅改变位置。
RoPE两种位置完全相同，只做一致性断言，不重复跑。

## 冻结条件

以下11项固定规范真桥V，最后一项按标注使用错桥；每项均R4/R5、两图、两模型：
1. bridge_old：原生Z，复现上批canonical-bridge基准。
2. bridge_scale：Z整体等比缩放到模型内固定范数，不加关系向量。
3/4. bridge_add_relation / bridge_add_wrong_relation：Z加真/错末关系token向量。
5/6. bridge_scale_add_relation / bridge_scale_add_wrong_relation：先同样缩放Z，再加真/错关系。
7. bridge_zero：Z置零，仅留下规范桥attention输出。
8. bridge_constant：Z置为固定的关系输入平均向量，归一到同一个固定范数。
9/10. bridge_fresh_atomic / bridge_fresh_atomic_wrong_relation：用规范原子关系输入替换整个Z。
11. wrong_bridge_fresh_atomic：规范真关系输入，但桥实体在同图循环加1。

另有native_add_relation / native_add_wrong_relation：完全相同的关系加回，但保留原生attention路由。
invariant额外bridge_fresh_site / bridge_fresh_site_wrong_relation两项；RoPE以向量精确相等断言替代。

固定范数取模型全部10种原子关系输入L2范数的中位数，所有题共用，不能按每题r5选择范数。
加入的真错token向量不带位置，分别归一到全部10个关系token embedding范数的中位数，确保两种注入等范数。
固定接收向量由全部10种原子关系输入平均后缩放到上述共同范数，不依赖当前关系、桥或答案。
错关系固定为(r5+1) mod 10；不同关系可能输出相同实体，保留全量并单列不碰撞子集。
错桥对应同一关系的后继必不等于真目标，逐题核对。

## 预先固定的解释规则

- 只缩放恢复：支持旧状态与新桥信息的相对尺度参与失败；不直接证明特定错误答案形成了动力学锁定。
- 旧Z加真关系有效、错关系跟随错后继：支持补充关系内容的因果作用，不能据失败条件宣称旧关系信息为零。
- 缩放无效而缩放加关系有效：支持减弱旧状态与补关系的共同作用，二者未必是互斥原因。
- 只有fresh有效：当前状态妨碍调用，但重置改变多个属性；不能唯一归因于关系丢失。
- fresh固定桥也失败：先判断单源V接口未充分校准，不把其负结果解释为状态内没有关系。

原数据末关系不均匀：A最常见占30%，B占40%。zero和constant是必需的默认运算对照。
按末关系、关系模板与同桥不同关系分别统计；只报告总体改善不能证明关系仍在。
主分析为干预当轮raw H；invariant后续会再次注入原r5，所以错关系对当轮和R16的影响分开解释。
同时保存R1–16轨迹、原生R32、输入变换前后范数和原LM头读出，报告是否延续R4的原错误输出。
不训练关系解码器，不拟合内部表现预测器。

## 配对末关系验证

另从每个原池10模板各取50个固定头实体（模板内每隔10行取一行），得到500个前四跳完全固定的前缀。
每前缀只将r5变为全部10种关系，重算真值，共5000题／图。桥、前四跳及输入长度保持。
仅测native、bridge_old/scale/zero/constant在R4/R5；不添加关系向量，检验旧Z保留的可用关系信息。
以当轮raw为主，逐前缀统计是否至少两种不同关系获得不同的正确答案；关系输出碰撞不能制造多答案成功。
zero/constant在同一前缀的当轮raw应不随r5变化，作为实现断言；invariant后续再注入原关系，不能用其R16代替此检验。
前缀raw/carry也须在10种末关系间一致。该配对验证是原有五跳问题的机制辅助，不与原池混合计算自然准确率。

## 执行与核验

原池RoPE每图27池，invariant每图31池，共116池；配对末关系验证另36池。
总152池、76万查询题次；8池原生R32、144池干预R16，共1280万题目块调用。
桥供体只有embedding/LN1/线性投影，不另运行事实MLP或原子完整循环。
独立LM Docker，GPU6原共享许可须重新核对，2CPU、16GiB、显存上限0.16；任务6GiB与整卡8GiB余量。
create-copy-start，root所有0444/0555源码、只读旧权重／数据，独立/ossfs输出。外层监控30分钟上限。
原生答案与历史逐token相同，bridge_old与上一批同条件逐token相同，干预前及其它位置读出不变，权重哈希保持。
CPU核验涵盖原生重放、固定范数不含当前关系、真错关系等幅、F(e)/F(2e)供体、单块等式、干预范围与原计算恢复。
原运行材料及失败保留，不改变六组正在训练的模型。

## 执行更正

首个attempt在首次保存context时因整数错关系ID与错关系向量均名wrong_relation而退出1；completed_pools=0，尚无正式评测。
整数ID改名wrong_relation_id；模型、条件、数据、统计规则和预算均不改。原容器、冻结源码及日志留在原batch。
此r2为独立输出目录与独立容器，未覆盖任何原产物。
