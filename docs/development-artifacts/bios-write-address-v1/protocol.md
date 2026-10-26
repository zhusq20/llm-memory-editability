# bios-write-address-v1：bioS属性以名字还是以上下文为地址（零训练）

## 状态

- **已完成（2026-10-20 21:11 UTC）：** B1–B5在两个世界全部成立；见[结果](../../results.md#bios-write-address-v1)。世界142702由辅助容器（GPU 6）并行计算，主运行复用其逐检查点文件。
- 2026-10-20用户授权（“请你做……2. bioS的写入地址”）。零训练：只读取[bioS原版开发批](../../results.md#bios)已保存的预训练检查点，不更新参数。
- 立即启动，不做源码冻结、哈希与CUDA预检（用户偏好）；CPU契约测试`tests/test_bios_address.py`通过（含与训练token逐位一致）。
- **预测写定之前已看过一次冒烟：** 世界142701、20人、S与MP的pass-540，在CPU上跑通读写两部分。冒烟中S的冲突题95%跟上下文，MP 88%跟名字；MP名字组写入份额.56，S为.15。以下预测因此不是盲预测；正式运行用1000人、两个世界、全部检查点。

## 1. 问题

[roadmap §3.1](../../roadmap.md#recall)：表达多样化改善提取，但是否把地址从“问题”（这里是属性句之前的传记上下文）移到实体（人名），需要位置证据。原则二给出两种地址：
- **上下文地址：** 属性由前文（前几句属性值、固定句序）触发，换名字不影响；
- **实体地址：** 属性由人名触发，换上下文不影响。

## 2. 读取地址：换名不换上下文

预训练原生格式。对人物A、规范句序中第j句的目标属性k（j=1…5），提示为A的variant-0前j句加第j句模板到属性值之前。四种提示：

| 提示 | 名字 | 上下文 | 实体地址预测 | 上下文地址预测 |
|---|---|---|---|---|
| original | A | A | A | A |
| conflict | B（同代词、k值不同） | A | B | A |
| context_only | 未出现过的名字（三人名字部分重组） | A | 都不是 | A |
| name_only | B，属性k的首句形式 | 无 | B | 都不是 |

贪心续写12个token，判断以A还是B的属性值开头；另报随机同代词第三人C的值作机会基线。S、M只见过规范句序（name_only的首句形式对它们是新格式），MP见过随机句序。每世界1000人×5属性×4提示；pass 30/90/180/360/540。

## 3. 写入地址：一次梯度落在哪些位置

对传记中某属性句的属性值token求和交叉熵，拆每层MLP输出投影的梯度`Σ_p δ_p k_pᵀ`。位置组：名字（首句人名token）、前文（其余前句token与起始EOT）、本句（代词与模板）、预测值（值前一位及预测后续值token的位置）。每组报精确Frobenius份额`‖ΔW_G‖/‖ΔW‖`（组间不互斥于方向，份额之和不为1）。三组文本：
- single：variant 0，规范句序（S的训练文本）；
- canonical：variant 0–4，规范句序（M的训练文本）；
- permuted：variant 0–4，每条随机句序（MP式）。

每个模型以自己的训练文本为“本分布”。另在canonical上算同一事实不同variant之间写入的余弦（名字组、预测值组分别）。每世界128人×6属性；pass 0/30/90/180/360/540。

## 4. 预测与判读（pass 540，两个世界分别判读）

| 预测 | 判读规则 |
|---|---|
| **B1** | S的conflict跟A（上下文）多于跟B；MP跟B（名字）多于跟A |
| **B2** | context_only下A值比例：S高于MP |
| **B3** | name_only下B值比例：MP高于S |
| **B4** | 本分布文本的名字组写入份额：MP高于S（pass 540与pass 90各判一次） |
| **B5** | canonical上名字组的跨variant余弦高于预测值组的层数过半（三种模型各判） |

M不设预测，作为中间条件报告。

## 5. 执行

- 输入：bioS原版目录为700权限，Docker无法挂载；把需要的36个检查点与数据按原相对路径复制到持久区`bios-write-address-v1/inputs/`，只读挂载为`/old`。
- 代码：`src/llm_memory_editability/bios_address.py`、`scripts/run_bios_address.py`；启动`scripts/launch_zero_training_maps.py bwa 5`（GPU 5）。
- 输出：持久区`bios-write-address-v1/`（`runs/<world>/<cond>/pass-NNN.json`及逐题`.rows.pt`、`figures/`、`summary.json`）；W&B run id `bwa-v1-main`。
