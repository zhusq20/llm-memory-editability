# 接收残差与实际attention读取的校准拆分

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-10-24（America/Chicago）
- Verification Status: posthoc calibration amendment fixed before inference
- Version Label: receiver_attention_v1

这是用户授权的接收状态诊断的必要校准延伸。第一阶段全头强制单源V，在RoPE即使使用fresh原子关系输入也未复现正常原子查询；
该接口的负结果不能支持“关系丢失”。第一阶段152池完整保留，本阶段不修改其设计或统计。
继续使用相同旧150万步RoPE参数EMA和旧invariant参数EMA、同A/B五跳各5000题、R4/R5单次干预、R16终点。

## 拆分原计算

原生块第五位置：A=c_proj(attention(...))；U=Z+A；H=U+MLP(LN2(U))。
先用原生Z运行原attention、捕获A，再在不重算Q/K的条件下改变残差Z。这样“native A + scale(Z)”不提供oracle桥或新关系。
重组U后用原MLP计算H，维持[B,6,D]原张量形状；只替换第五位置raw，其他位置原样保留。
invariant的历史N/D、打分、后续e注入均保留，raw/candidate/聚合终点分列。

另外构造真实e4+r5、真实e4+错r5、错e4+r5三种规范原子调用，捕获各自实际attention输出A_atomic。
只使用其attention输出作供体；原子H只计分用于正控，绝不拷贝进入干预状态。
A_atomic可能携带关系相关甚至答案相关信息，不能把使用它的条件称为“没有重新提供关系”。
真、错关系和错桥原子H各自正确率都保存。记录A与U原LM头读出及通过MLP后的结果；不可解码不等于信息不存在。

## 事前固定矩阵

每组均在R4/R5、A/B、两模型；全量保留，不选最佳时刻。
- native attention：Z缩放、Z=0、fresh真关系、fresh错关系；native基准复用完整原生轨迹。
- atomic true attention：原Z、缩放Z、Z=0、fresh真关系、fresh错关系。
- atomic wrong-relation attention：缩放Z、Z=0、fresh真关系。
- atomic wrong-bridge attention：fresh真关系。
共13个干预条件。统一固定范数与fresh定义沿用第一阶段：全部10种原子关系输入范数中位数，不依赖当前题目。
RoPE fresh=Wte(r)，旧invariant fresh=2(Wte(r)+P1)，两者不混用。

正控：native A+原Z须复现原生计算；fresh真关系+atomic true A须逐题复现原子H。
完整fresh正控是构造校准，不是独立自然修复证据。错关系/桥供体若不合格，其负结果不能作内容结论。

配对关系验证继续用500个固定前四跳前缀×10末关系的5000题／图，但只测native、native_scale、native_zero。
这两项不注入新关系或oracle桥；CF若成功，说明原生(Z,A)组合仍有可用关系信息，不能仅定位到Z。
以当轮raw计分，按每前缀正确关系数、不同正确后继数及关系分布完整报告；invariant R16有重新注入r5，不替代当轮检验。

## 预算与执行

原池每模型每图27池，共108池；CF每模型每图5池，共20池，总128池、64万主查询题次。
8原生池R32、120干预池R16，共1088万块循环调用。另8个模型/图/数据类型×3供体×5000=12万原子块调用，总1100万。
每个干预题次另重算一次完整形状MLP，共60万次[B对应样本,6,D]的额外MLP计算，单列，不计作训练。
2CPU、16GiB、GPU6共享资源重新核对、0.16显存上限，任务6GiB/整卡8GiB余量。独立LM Docker、只读旧权重。
create-copy-start冻结root所有0444/0555源码；专属/ossfs输出；外层30分钟墙钟上限。
原生历史预测、fresh原子正控、干预前及其它位置读出核验，权重哈希与原训练PID保持。失败和全部原产物保留。
