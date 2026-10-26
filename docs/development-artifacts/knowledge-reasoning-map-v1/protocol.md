# knowledge-reasoning-map-v1：知识与推理的位置×层地图（零训练）

## 状态

- **已完成（2026-10-20 20:23 UTC）：** K1在两个模型上不成立（实体token中段兼读问题），K2–K4成立；见[结果](../../results.md#knowledge-reasoning-map-v1)。
- 2026-10-20用户授权（“请你做1. 知识/推理功能地图”）。零训练：只在原权重上做前向和一次梯度，不更新参数。
- 立即启动，不做源码冻结、哈希与CUDA预检（用户偏好）；CPU契约测试`tests/test_function_map.py`通过。本协议在GPU运行前写定，预测与判读不随结果修改。

## 1. 问题

[roadmap §1](../../roadmap.md#mission-20261020)用功能定义区分知识与推理：
- **知识：** 以实体为地址的计算——换实体就变，换问题不变；
- **推理：** 以问题为地址的计算——换问题就变，对所有实体相同。

本批在Qwen3-1.7B与Qwen3.5-2B上直接测量每个位置×层属于哪一种，得到开篇“位置×层方格”的实测版，并叠加答案监督写入落在哪里（原则二）。

## 2. 设定

- **模型：** 与[知识调用接口v2](../knowledge-read-interface-v2/protocol.md)同一revision，float32，chat模板，关闭thinking。
- **实体：** `function_map.GOLD`中在该模型分词下为单token（前导空格）的国家。
- **问题：** 4种关系（capital、continent、language、currency）×2种问法（“What is the {r} of {e}?”、“Tell me the {r} of {e}.”），后接“\nAnswer with only the name.”。所有提示等长，逐位置对齐；关系词在实体之前，因此实体位置能看到问题。
- **量：** 每层的残差流（层输出）、MLP地址（down_proj输入，即原则二的写入地址）、MLP输出、混合器输出（attention或DeltaNet）。分析范围从问题首token到最后一个提示token。

## 3. 读出

1. **三分份额。** 每个层×位置，把实体×问题两因素的向量分解为`x = μ + a_e + b_q + c_eq`（平衡设计，平方和正交）：实体份额、问题份额、联合份额（三者和为1）。问题份额再拆成关系部分和问法部分。全部实体与“8个问题全部答对的实体”分别报告。
2. **答案监督写入。** 对每个提示，以同关系下另一国家的答案作为改写目标（反事实，模型无关、确定性），回复为“目标+<|im_end|>”，求和交叉熵，一次反向；每位置写入的近似份额为`‖δ_p‖‖k_p‖`占该行全部位置之和（与[answer-gradient-address-v1](../answer-gradient-address-v1/protocol.md)相同）。回复中预测后续回复token的位置合并为一列。
3. **单例（预定）：** France、Japan、Brazil、Egypt × 4种关系（问法一），16个MLP地址的中心化余弦矩阵：实体位置取第⌊L/3⌋层，最后提示token取第⌊2L/3⌋层。
4. **行为：** 贪心生成12个token，按别名子串判对错，只作已知事实的筛选与报告。

## 4. 预测与判读（MLP地址，全部实体；“前段”为前2/3层，“后段”为后1/3层）

| 预测 | 判读规则 |
|---|---|
| **K1 实体位置是知识** | 实体token的问题份额<0.1的前段层数过半 |
| **K2 最后位置是推理** | 最后提示token的问题份额大于实体份额的前段层数过半 |
| **K3 答案在后段形成** | 最后提示token的联合份额，后段均值大于前段均值 |
| **K4 答案监督写在推理区** | 后段中过半的层，最后提示token与回复位置的写入份额之和大于实体token份额的2倍 |

每条在两个模型上分别判读；结果出来后如实报告，不改规则。残差流与“已知实体”子集作为补充，不另设判读。

## 5. 执行

- 代码：`src/llm_memory_editability/function_map.py`、`scripts/run_function_map.py`；启动`scripts/launch_zero_training_maps.py krmap 2`（GPU 2，read-interface镜像；作图用plotting镜像，CPU）。
- 输出：持久区`knowledge-reasoning-map-v1/`（`runs/<model>/summary.json`、`generations.json`、`figures/`、`summary.json`）；W&B `zsqzz/llm-memory-editability`，run id `krmap-v1-main`。
