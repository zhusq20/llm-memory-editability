# 用途覆盖 A/B/C：数据与评价契约

这份实现服务于[路线图 §1.2](../../roadmap.md#natural-knowledge-methods-20261021)已规划的问题：同一监督 token 预算下，原子事实的表达覆盖与用途覆盖分别怎样改变提取、未练用途和保持。它没有启动训练；开发校准、固定回答采样、预算、模型更新和正式比较由批次执行协议另外记录。

## 与历史实验的关系

最接近的历史是[用途适配开发批](../use-adapter-development-v1/protocol.md)。该批在逐条 E3 实体式写入上比较共享用途练习与原子改写，Qwen3.5-2B 有跨事实用途收益，也有自然旧题退化，Qwen3-1.7B 主要改善分步回答。旧批两臂按步数匹配，监督 token 不等，写入每条事实的独立参数先冻结。本批不重跑旧实验：所有新事实累计写入同一个普通 Qwen，比较既有 off-policy PD／SFT 的 A/B/C 数据组织，并按监督 token 控制预算。共享城市背景及独立前提检查沿用旧批思路；不复用其事实划分作为正式确认集。

## 世界与真值

- 默认开发 32 个事实对，正式 128 个事实对；生成器支持 2–128 的偶数规模。每对含“虚构机构 → 指定分部”和“该分部 → 所在城市”两条新事实。开发、正式的虚构名称池严格不交叉，正式事实进入写入训练，正式测试用途题留出。
- 城市及国家来自本地 [read_interface_data.py](../../../src/llm_memory_editability/read_interface_data.py) 的冻结表；不用网络查询或新增当前地理知识。候选目标城市仅保留该表里存在另一座同国城市者，避免 yes 比较退化为与本市比较；这项过滤不读取模型表现。候选由 27 座过滤为 21 座，默认种子的开发／正式目标城市覆盖分别为 17／21 座，完整名单保存在每份 manifest 的 `city_selection`。开发与正式共享这些已有自然知识背景，不能称背景实体独立。每对的城市、同国参照城市、异国参照城市在世界生成时确定，完全不依赖测试题或答案。
- 每机构恰有一个分部，每分部恰属一个机构、恰有一个城市。训练前在共同原文中说明此规则。逆向题仅问唯一的“分部 → 机构”；不问真值可能不唯一的“国家 → 城市”。
- `gold` 由离线事实表和明确逻辑产生，只是 QA 质检及评价真值。它不等于教师生成回答，行中不预填 `assistant`。PD 和 SFT 固定回答分别由批次定义的采样过程生成和保存。
- 每对原文含相同的两条新增事实以及所用城市／参照城市的国家事实。A/B/C 对该对的 `context` 和 `context_fact_ids` 完全一致。C 的组合答案只用这份共同材料，`fact_ids` 列出答案实际依赖；不能靠增加背景材料得到收益。

## 初次写入材料

| 臂 | 每对唯一 QA 数 | 内容 |
| --- | ---: | --- |
| A | 2 | 每条原子事实一个固定问法；在外层训练流中重复 |
| B | 16 | 每条事实八个原子问法 |
| C | 16 | 每条事实四个原子问法，共八个；另有两个城市→国家组合、四个真假判断、两个同国比较 |

C 的真假判断与同国比较各自平衡 yes/no。组合练习使用“分部→城市”这条新事实和已有城市国家事实；机构分部事实练原子及候选判断。不同关系的用途机会并不完全对称，分析应按事实关系分层。所有 C 训练问题至多依赖一条新事实，双新事实链完整留出。

上述是唯一 QA 池大小，不是曝光数、更新数或监督 token 预算。回答采样、tokenizer 计量、重复次数、每事实曝光、教师计算和训练计算由执行层保存。A→B 改变原子表达覆盖；B→C 在同规模唯一 QA 池中改变用途分配。等 token 不自动等事实曝光或等生成成本，必须单列这些值。

训练函数 `training_rows` 不调用评价函数，不读测试模板；测试真值只在独立 `evaluation_rows` 内生成。存在一个明确标记的原问法复测 `atomic_seen`，其直接问法与训练相同。其余测试题干不出现在任一训练臂中。训练生成只读取 `World` 和训练模板，不读取任何评价 JSONL。

## 初次写入与改写评价

每对共 16 个问题，每题直接回答与显式分步各一行，合计 32 行：开发每阶段 1024 行，正式每阶段 4096 行。

| `category` | 每对问题数 | 含义 |
| --- | ---: | --- |
| `atomic_seen` | 2 | 两条事实的原训练问法；分步版本单独记 |
| `atomic_unseen` | 4 | 每条事实两个留出问法 |
| `seen_use_new_wording` | 7 | 城市国家组合、真假判断和同国比较的新问法；只有 C 练过这些用途 |
| `double_new_composition` | 1 | 机构→分部→城市；各臂均未训练该链 |
| `heldout_relation_composition` | 1 | 机构→分部→城市→国家 |
| `heldout_use_category` | 1 | 分部→机构的唯一逆向检索 |

每个状态分别评价闭卷与原文在场，两种条件不能混合汇总。`context` 是评测行携带的材料，闭卷执行时必须丢弃。背景前提 `background_rows` 以真实城市国家问题测量；保留全量结果，再按冻结基座的背景前提正确性、原文在场的用途能力及该状态必要原子事实掌握情况，报告条件子集和覆盖率。不能筛去失败世界后只报子集。

分步题要求最后一行 `Final answer: ...`，不在问题中提示金标桥实体。`chain` 为额外的可执行解码—再编码检查：第一问取得模型自己的回答，第二问中的 `{answer0}` 用这个回答替换，第三问的 `{answer1}` 同理。`answer_slot` 定义保存名。链中没有桥真值字段；无法解析或回答错误都保存原始输出，不能退回金标桥实体。显式分步、链式自生成桥替入与一次直接回答是三种不同读出，分别报告。

主评分 `score_answer(row, text)` 对最后答案做大小写、标点、重音和显式别名规范化，拒绝 `yes or no`、多个冲突城市及只在推理中提到真值的回答。返回 `correct`、`parsed_answer`、`parse_status`、`conflicting_candidates` 和 `format_valid`。冗长自然语言未被严格别名接受时保留为格式／语义审核对象，不能据此直接断言没有学会；语义审核与预定严格评分分列。

## 再次修改

将相邻两对的机构分部映射做两两互换，所有“分部→城市”事实保持不变。世界生成保证交换两对的城市和国家不同，因此依赖城市／国家答案必然改变，不增加新背景事实。

`rewrite_training_rows` 对各 A/B/C 臂提供同一份原子修改材料，每机构仅一条新“指定分部” QA。这里的教师原文也只含新指定分部和共同规则，不含城市、国家或依赖答案。训练从对应初次写入模型继续，不能重置为基座，也不使用 C 的用途监督补写依赖答案。

改写后的 `evaluation_rows(stage='rewrite')` 保留 `previous_gold`、`answer_changed`；被修改事实、变化的依赖答案、未变化分部城市分别统计。初次写入的统计单位为事实对；改写后同一交换块共享依赖，`cluster_id` 合并为两个事实对的交换块，正式规模只有 64 个改写块，不能按问法或模式虚增样本量。原子修改训练成本及改写后的评测成本与第一阶段分列。

冻结前修正了逆向“分部→机构”题的跨阶段配对：两阶段始终询问同一个原分部，改写后gold改为交换伙伴机构，previous_gold保留原机构；题目所属pair_id不变。原文及context_fact_ids切换为新拥有机构对应的改写材料，fact_ids指向该机构的新分配事实，保证原文明确支持答案。此修正只影响逆向评价，不改变训练C或其他评价模板；已生成的旧计划产物保留，正式冻结前重新材料化到新目录。

## 保持训练 R、保持评价 U 与背景前提

- 原有 317 条自然旧题继续作为 U，不混入保护训练 R。它们须按原冻结题库接入，基座原正确题保持与全量准确率分别报。同一国家的城市、货币与自然两跳问法合为一个统计块：原冻结317题对应91个国家簇，不按317个独立事实计算区间。材料化保留 `original_source_id`，国家从 `ret/capital_rev`／`ret/city_country` 金标元数据映射，`ret/currency` 直接用国家实体，`nat2`／`nat2short` 通过城市归组；不用模型预测分簇。无法识别的来源标 `cluster_id=null/cluster_status=unknown`，点估计保留，区间未知。
- `retention_rows` 提供真实城市国家相关保持；明确标注其背景已在三臂共同教师原文中出现，不称完全未见知识。在改写阶段另提供未变的分部城市事实保持。真实依赖已改写事实的答案不施加保持约束。
- 优先复用旧 use-adapter／read-interface 中与上述 U 不重叠的有效自然 QA 回放池。若暂时缺少可验证的独立池，`replay_rows` 给出 32 条加减法保护提示，用冻结基座自己生成的回答前缀做无原文 KL；其中 `gold` 不用于硬目标回放。
- `general_retention_rows` 提供 32 条不同算式的乘法 U，作为小型补充，不能把算术保持称为一般自然语言能力保持。`validate_retention_split` 检查 R/U 问题和事实 ID 不重叠，也拒绝把明确修改的目标事实放入保持约束。外接自然题库仍需核对来源、语义重复和题目哈希。
- 各臂的 R、R 的固定回答前缀、保护 token 预算和损失权重必须相同。保护成本与新事实训练成本分开，同时报告总成本。

## 真实文档验证接口

真实文档应用验证固定复用作者 CSV 索引 32–95 的 64 案，共 640 段；这些案例已在八组评测中出现，不是全新确认集。前 32 案只用于生成质量开发。训练入口只投影全文与段落，原评测 `question/answer` 独立加载；问题生成和修正均不读取原测试题或答案。所有 A/B/C 教师训练都得到同一案例全文。

A 固定每段原始 `q000`，B 固定 `q000:q004`，C 保留 `q000:q002` 并为每案预定至多两倍段数的组合／比较／判断请求。原始问题与 PD/SFT 回答按固定索引原样复用，保留来源行及文件哈希。这些自由生成旧题没有经过逐条原子性认证，统一标为 `auto-generated-original-questions`；A/B 还可能改变段内事实覆盖，不能把这里的差值解释成纯问法因素。旧回答轨迹来自段落背景，新 C 回答来自案例全文，差异也单列。

新组合／比较题必须实质使用两条独立事实，并提供至少两个不同的原文证据片段；证据可以来自同一段。`paragraph_id` 只是来源定位，不能当作原子事实 ID。重复同一句、将单个断言拆成重叠片段或附加无关引用，不能建立双事实用途；逐字引用存在只是结构核验，语义正确性、问题是否真正需要两事实及用途归类仍须另行审核。判断题至少提供一段原文引用。无法从共同材料构造时保留 `unavailable`，不补外部事实。放宽“必须两不同段”为“必须两不同事实证据”是生成协议修正，不是科学结果。

[自然执行脚本](../../../scripts/run_knowledge_use_coverage_natural.py)的 `--action calibrate` 在前 32 开发案例各取组合／比较／判断的首个固定请求，共 96 个；首轮质量和修正后质量分别报告。预定策略只有一次生成或最多三次尝试。后者仅对格式、引用或不可用输出反馈原文与结构错误，保持相同案例、用途和背景；完整保存每次输出、提示、错误与耗时，不根据正式测试选择、换案例或改训练目标。开发阶段可执行这两个候选供审核，正式阶段只采用已批准策略。三次后仍失败则记录缺失并停止准备，不悄悄丢样本继续训练，也不重训已有科学模型。

校准写入 `data/natural-development/{question-requests.jsonl,raw-questions.jsonl,parsed-questions.jsonl,quality-summary.json}`。即使存在无效或不可用题，`quality-summary.complete=true` 也只表示校准过程结束，状态保持 `awaiting_manual_semantic_review`。审核后才由负责人写 `calibration-approved.json`，必需字段是 `status="approved"`、`semantic_review_completed=true`、`question_attempts=1或3`、`generation_protocol_sha256` 与 `quality_summary_sha256`；后两项必须匹配当前代码／模型／采样协议和实际质量摘要。正式 `prepare` 还复核校准原文、解析结果和请求文件哈希，缺批准不得生成 1280 条正式请求。校准脚本不会自行批准。

正式材料生成后，独立检查与原测试问句的精确交叠；发现交叠只保留并停止，不据测试换题。主评分沿用 Qwen3-4B 无 thinking 的作者语义裁判、有效分母和无效覆盖记录；317 条旧题保留原别名评分。真实文档只检验应用效果，不能替代可控新事实的用途归因。

## 调用与核验

[实现](../../../src/llm_memory_editability/knowledge_use_coverage_data.py)与[测试](../../../tests/test_knowledge_use_coverage_data.py)：

```python
world = make_world(cohort="development", pair_count=32, seed=20261025)
world.manifest()                       # 来源、事实和改写置换
training_rows(world, "A")             # "A" / "B" / "C"
evaluation_rows(world, "addition")    # 直接与分步；context 由执行层决定是否传入
rewrite_training_rows(world)           # 全臂相同原子改写
evaluation_rows(world, "rewrite")
background_rows(world)
retention_rows(world, "rewrite")
audit_world(world)
```

核心行键统一为 `id/fact_ids/question/context/gold/aliases/category/split/variant`，另有 `mode/stage/cohort/pair_id/cluster_id/context_fact_ids/background_fact_ids/chain` 等审计字段。`split` 指行用途 train/evaluation/retention/replay，`cohort` 指 development/formal，不混用。

CPU 契约覆盖开发正式新增实体隔离、训练生成不读取评价、A/B/C 背景一致、C 没有双新监督、测试题隔离、两种读出齐全、改写真值及不变事实、R/U 隔离和最终答案评分。首次测试因当前解释器未安装项目包而收集失败；指定 `PYTHONPATH=src` 后 20 项通过。静态检查使用现有 `/tmp/lm-recovery-tools/bin/ruff`，因为当前 shell 没有全局 `ruff` 命令。

## CPU 材料化结果

[材料化脚本](../../../scripts/prepare_knowledge_use_coverage.py)读取 `configs/knowledge-use-coverage-v1.json`，支持 `--output` 和 `--factorial-batch`，拒绝覆盖任何已有输出目录。旧八组的 `runs/sft_full/paper-eval/auxiliary.json` 只提供 317 条保持题的 user 问题、gold、type 与 exclude，不复制原回答、token ID 或评分。新增[脚本测试](../../../tests/test_prepare_knowledge_use_coverage.py)覆盖这一隔离、完整材料化、跨集合实体检查、ID 唯一、改写保持基线和拒绝覆盖。

实际 CPU 产物在 `results/knowledge-use-coverage-v1-plan/`，总审计 `audit.json` 记录配置、代码、输入与输出 SHA256。没有启动训练或教师采样，没有写入旧持久批次。每个 cohort 保存 manifest、A/B/C 与改写训练、两个阶段评价与保持、回放及额外的独立 `background.jsonl`。

该计划目录保留早期原样，不覆写。冻结前修正了旧题的国家聚类与来源ID；正式批次必须由修正后的材料化脚本写入新目录，其总审计记录国家簇数和未知来源数。直接读取原冻结辅助文件核对为317题、91国家簇、0条未知。

| 每个 cohort 的行数 | 开发 32 对 | 正式 128 对 |
| --- | ---: | ---: |
| A／B／C 训练 | 64／512／512 | 256／2048／2048 |
| 原子改写训练 | 32 | 128 |
| 每阶段用途评价 | 1024 | 4096 |
| 每阶段保持及其基线 | 557 | 967 |
| 独立背景前提 | 80 | 106 |
| 算术保护回放 R | 32 | 32 |

保持文件由相关城市背景、分部城市、32 条独立算术和 317 条自然旧题组成。初次写入的分部城市行标记为 `learned_location_baseline/newly_learned/is_old_retention=false`；改写后的对应行为 `retention_unchanged_new_location`，使用初次写入终点作为基线，两阶段 ID 一一对应。初次写入成功不能被算成旧能力保持。

## 分析与开发选择

[统一分析脚本](../../../scripts/analyze_knowledge_use_coverage.py)将实际类别 `atomic_seen/direct` 固定为开发80%校准指标；`atomic_unseen/direct` 另报泛化。两个精确开发PD节点的三臂结果都齐后，选择最早共同达标节点；未达标也保留实验，使用较大原规划预算。正式与自然应用预算按同一开发选择映射，不能依据正式或自然题表现另选。当前更新范围尚是暂定配置。

全量与背景／原文在场／原子前提子集同时报告，开发与正式、新增与改写、受控事实与自然应用分开。直接、显式分步、自生成桥链分列；配对区间按事实对或改写交换块聚类，自然应用按案例聚类。`summary.json` 保存逐类别分母、覆盖率、配对得失、原正确题保持及精确节点成本；缺失与无效评分不填零。

分析器核对实际 `run.json.spec` 的完整配置、cohort、方法、A/B/C臂、阶段、节点及改写父检查点。自然应用直接复用执行脚本的 `natural_config(config, /factorial)`，不拿受控任务2048长度配置检查实际8192长度的自然任务。训练秒与进程总经历秒分别读取；工程成本读取 `exercise.json`、`reload.json`、`preflight.json`，只有exercise的独立秒数可相加，preflight更新数是两个exercise的汇总，不重复计费。所有 `failure*.json` 保留。
