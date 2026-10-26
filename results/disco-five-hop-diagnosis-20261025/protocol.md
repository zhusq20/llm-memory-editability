# 旧DiscoLoop五跳：事实调用、桥交接与答案丢失

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-10-24（America/Chicago）
- Verification Status: protocol frozen before inference
- Version Label: disco_five_hop_diagnosis_v1

用户明确要求以同样的方法检验旧DiscoLoop，重点RoPE，与已完成invariant诊断比较。
这是已有五跳失败的机制延伸，不包装为新选题。既有证据见invariant-five-hop-diagnosis-20261025。
只读已完成150万步终点、零训练、不恢复暂缓队列，不改变正在训练的六组模型。

## 模型、数据与终点

原生观察：RoPE参数EMA（主）、同次训练RoPE raw、旧固定R4/raw、同native配方additive/参数EMA。
原世界同A/B五跳各5000题，R1–32完整记录，R16为主终点。
局部干预：事先选定RoPE参数EMA与旧固定R4/raw两组；不是按本次诊断结果挑选。
这两组此前短组合能力较强；其余两组保留原生轨迹和原子供体前提，不扩展干预网格。

DiscoLoop原式：x0=e；Hr=F(x_(r−1))；xr=Hr+q(Hr)。当轮正式答案读Hr，下一轮读取xr。
没有invariant的每轮e注入、候选外层RMS或N/D历史合并。桥前提按前一轮carry第四位置对齐。
保存input、attention后状态、raw、carry各位置原LM头的实体、概率、rank和范数；
末位置attention权重须按实际RoPE旋转Q/K后计算。原生真路径仅用于计分，不用于模型输入。
分类为R1–16始终未正确、曾正确后R16丢失、R17–32首次正确；raw与carry分别报告。
原生第一次正确后直接保留输出的oracle上界为ever-correct，不把它解释为冻结carry的结果。

## 单轮干预

与invariant同六类、R4/R5/R6/R8各施加一次，仅第五接收位；其余计算保持原式到R16。
1. force_bridge：c_proj前所有头的加权value改为本轮原生第4位置V。
2. force_original_head：同形式改读原生第0位置V，作内容对照。
3. canonical_bridge_force：改读真实桥e4在规范原子源位置的V，来自F(e_atomic)，不使用invariant的2e。
4. wrong_bridge_force：桥替换为同图(e4+1) mod 500，保留r5；置换关系保证错误后继不同。
5. atomic_postattention：完整原子调用[e4,r5,pad,pad,pad,pad]首轮attention后状态供体。
   单块、eval、逐位置MLP下，直接复制该原子raw H1等价于替换完整attention后残差再算同MLP。
6. wrong_atomic_postattention：错误桥的相同供体，记录其错误后继是否保留。

所有时刻和正负对照都报告，主终点固定R16，不选最好点代替原生性能。
原子供体仅输入桥e4与可观察r5，不输入最终e5；e4是真路径oracle。
完整原子供体已经经过MLP，可能含答案。当轮Disco正式答案就是供体H读出，这是构造结果；
主要检验后续保持，不能把其当轮收益与invariant门后接收直接相等比较，不能作为纯attention定位。
供体将多个接口属性同时改变；单源强制路由失败不排除其它路由修复。

## 比较与前提

逐题核对必要第五事实的独立原子R1/R16及真／错供体；全量结果与原生桥前一轮可读子集分列。
同时报告干预当轮raw/carry、R16、新增正确、原正确丢失、当轮正确后丢失与错误目标率。
同一世界固定题型不是独立世界；EMA/raw不是独立训练，跨配方不作架构单因素因果比较。
这次没有换训练世界，不能直接检验data sensitivity。

## 预算、执行与验证

原生8池×5000×32=128万题目循环轮；干预96池×5000×16=768万，主共104池、52万题次、896万循环轮。
每模型／图的真错原子供体只计算并缓存一次：8×2×5000×17=136万轮，共1032万循环轮。
预计几分钟、硬上限30分钟。逐池监测输出及进程状态；独立LM Docker、2CPU、16GiB主存、显存上限0.16。
仅在原GPU6共享例外仍满足时启动，重新检查预约、训练PID、40000MiB空余；预留任务6GiB、整卡8GiB。
代码create-copy-start，容器内root所有0444/0555；只读原权重／数据、独立/ossfs输出。
原生答案逐轮逐token复现历史；干预前答案及其它位置raw/carry读出保持；权重前后哈希一致。
CPU检查覆盖absolute/RoPE原生重放、桥V维序、干预范围、F(e)供体与残差、真值不影响模型输出。
所有失败及工程更正留存，不自动覆盖冻结证据。
