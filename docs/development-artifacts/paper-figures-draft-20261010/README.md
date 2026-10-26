# 论文主图草稿 F2–F6（2026-10-10，零训练）

由 `scripts/plot_paper_figures_draft.py` 从 HF 数据集 `zsqzz/llm-memory-editability` 固定版本 `6c580455ce1667274ea592c1c88f5d0fb1e5a190` 读取归档文件生成，只读、不加载模型。复现：

```bash
python scripts/plot_paper_figures_draft.py --cache <hf-cache-dir>
```

`sources.json` 记录每个源文件的 sha256、每个面板的数据来源，以及 58 项与 results.md 数字的一致性核对（全部一致）。转录的数字在 `sources.json["transcribed"]` 中逐项给出锚点和转录原因。F1 是示意图，未画。配色用 dataviz 参考调色板（前四个类别色已过相邻 CVD 校验；aqua/yellow 对比度不足 3:1，靠直接标注和本表补足）。

## 每张图画了什么

| 文件 | 面板 | 数据 |
| --- | --- | --- |
| `F2-answering` | a bioS 三种组织：规范 QA vs 五个留出问法（2 个世界） | 原始：`bios-original-review-v1/overall.csv`，算得 5.07/35.65/99.96 与 0.57/14.06/32.90 |
| | b 同一权重：预训练端原生续写、QA 适配后原生续写、同权重 QA（出生日期，每格 64 人） | 原始：`bios-same-weight-v1/summary.json` |
| | c MQuAKE：cloze 写入成功后用问句检验，ROME 87.8%、MEMIT 54.3% | **转录**自 roadmap §1.1/§1.2 的部分记录预览，图中标注 PREVIEW；仓库里没有可复现它的脚本 |
| | d 负载曲线：原子准确率对 N，等 4000 曝光与等 128k 更新 | 原始：`composition-data-curves-v1/report/budget-points.csv`（三世界均值） |
| | e 曝光曲线：各 N 的原子准确率对每事实曝光，带为世界极差 | 原始：同批 `all-learning-nodes.csv` 全部节点 |
| `F3-composition` | a 使用经历：组合练习／匹配原子重复／另组重复／仅基础原子 100/25.01/6.99/3.01% | 原始：`grok-usage-v1/report-confirmation/summary.json` 逐世界端点 |
| | b Ye §2.4：Test-II 95.90/99.57%，受限第二跳均 2/3000 | **转录**自 results.md#second-hop-paper-reproduction-v1（HF 上只有到约 59 万步的训练损失） |
| | c Wang 原图四分面板（II/IO/OI/OO）随训练，普通 8 层 vs Loop 4×2，四个 φ/wd 条件 | 原始：8 条 `grokking-reproduction-v1/.../learning.json` 全部评测节点 |
| `F3d-address` | 地址实验四分面板：共享头地址 vs 独立头地址，两架构，128k | 原始：4 个 `realworld-loop-addressability-v1/.../endpoint.json`（复用 reanalyze 脚本的读取函数） |
| `F4-readable-vs-usable` | a 共享 GPT 三臂 × {原生、自解码重编码 α=1、注入真桥、注入错桥} 严格组合，及槽位读出 | 原始：`bridge-reencoding-confirmation-v1/.../summary.json` |
| | b 普通独立两层：首跳 CE vs 加对齐，熟悉/严格组合与严格状态余弦 | 原始：`independent-alignment-v1/report/summary.json` |
| | c Wang 原图第 4 层输出 r1 位置：原生 OI/II、删去桥输入 embedding 分量后的 OI/II、OOD 首跳 ×2 交换转向、随机旋转；每架构 12 端点（4 条件 × 700k/1M/1.5M），φ3.6 空心 | 原始：bridge-workspace `endpoints.csv`（复用 reanalyze 脚本的 `bridge`） |
| `F5-updates` | a 强父模型局部编辑：新答案 E、无关原子保持、派生 D（4 个 D 格均值，对齐父 4.08%） | 原始：`interface-editing-confirmation-v1/report/summary.json` |
| | b 写入目标三臂：第一轮读出翻转（/24）、派生新答案（/96）、旧派生答案残留（/96） | 原始：write-target `summary.json`（`by_objective_arm` 与逐编辑记录） |
| | c 逐编辑：翻转的 48 次编辑 D 148/192，未翻转 24 次 3/96，各自同案例 sham | 原始：同上，复用 reanalyze 脚本的 `write_target` |
| | d 干扰：父正确 U 严格组合保持（世界均值）编辑 vs sham | 原始：同上 |
| | e MQuAKE 首跳−后续跳差（直接/CoT，95% 案例 bootstrap）及 CoT 缩小量 | **转录**自 results.md#path-hop-analysis-v1（该批结果不在 HF 上） |
| `F6-continual` | a 新+新组合 BB：四种训练历史学 B 前/终点；b 旧组合 AA 同上；c 原子回放 vs 完整回放的 BB/AA | 原始：`sequential-transfer-confirmation-v1` 与 `sequential-replay-paired-v1` 的 report `summary.json` |

## 数字核对

- 计算值与 results.md 的 58 处数字全部一致（容差 0.006pp）。
- 只有舍入层面的差别，不是数据不一致：原子对照 BB 为 12/384=3.125%，results.md 表格写 3.13%，roadmap §5 和 summary 均值写 3.12%；bioS 同权重出生地 50/64=78.125%，results.md 写 78.13%。
- F2c 的预览数字来自 roadmap 而不是 results.md；results.md 目前没有这两个数，也没有生成它们的脚本。正式数应由实验主机上 S1/S2 读出替代。
- F5a 的 4.08% 是两跳角色 × 熟悉/严格四个 D 格的等权均值，与 results.md 一致；它不是合并计数 12/298。

## 还缺什么才能定稿

| 图 | 状态 | 缺口 |
| --- | --- | --- |
| F2 | 行为面板齐全 | c 需换成 S1/S2 正式读出（按编辑器、跳位的问句迁移）；理论推导 2 已补（理论§14.7，2026-10-10）；bioS 没有按训练格式定位写入层位的结果（可选） |
| F3 | 描述性面板齐全 | 缺 Z1 路径重合干预这一唯一的直接因果面板；b 的数字待实验主机终点文件归档后改为原始读取；c 只有一个世界、一个初始化；地址实验只有两档出度，且头实体新颖性与 OOD 首跳重合 |
| F4 | 可作草图 | c 只覆盖 700k/1M/1.5M 三个节点（Z2 补全已于 2026-10-10 撤下，不再补）；缺 R2 的“非循环＋通道/对齐”格 |
| F5 | 可作草图 | e 为转录值，S1/S2 正式读出后补前提子集与旧答案残留；缺 U1（写入层段 × 跳位 × 架构，每格 ≥500）；c 为事后逐编辑描述，翻转与写入目标完全混在一起 |
| F6 | 接近定稿 | 无硬缺口；持续学习阶段的问法与角色版本为可选 |

排版仍是草稿：静态 PDF/PNG，没有交互层；字体为 DejaVu Sans；标题措辞与面板取舍待论文定稿时调整。
